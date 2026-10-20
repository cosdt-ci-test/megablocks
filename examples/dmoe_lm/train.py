# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Train a small dMoE language model on Ascend NPU (or CUDA).

Single device::

    python train.py --steps 1000

Data is the wikitext-2 token stream produced by ``python data.py prepare``
(the stream is prepared automatically on first use). Pass ``--data-path``
with any uint16 token file to train on different data.

The trainer exercises the NPU-supported megablocks pathway end to end:
``dMoE`` blocks with grouped GEMM experts, the load-balancing aux loss, and
half-precision parameters (bf16 by default). It runs on one device only.
"""

from __future__ import annotations

import os

# Triton-ascend reads these at import time, before any megablocks import.
# Library code deliberately does not mutate os.environ; tests_ascend sets the
# same pair, and the cache directory keeps kernel compilation out of $HOME.
os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".triton_cache"),
)

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn

try:
    import torch_npu  # pyrefly: ignore[missing-import]  # registers the npu backend
except ImportError:
    torch_npu = None

from megablocks.layers.arguments import Arguments
from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss

from data import TokenStream, prepare
from model import (
    ModelConfig,
    build_model,
    next_token_loss,
    parse_config,
    resolve_device,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    model = parser.add_argument_group("model")
    model.add_argument("--hidden-size", type=int, default=512)
    model.add_argument(
        "--ffn-hidden-size", type=int, default=0, help="0 means 4x hidden_size",
    )
    model.add_argument("--num-attention-heads", type=int, default=8)
    model.add_argument("--num-layers", type=int, default=6)
    model.add_argument("--vocab-size", type=int, default=50257)
    model.add_argument("--num-experts", type=int, default=8)
    model.add_argument("--top-k", type=int, default=2)
    model.add_argument("--mlp-type", choices=("mlp", "glu"), default="glu")
    model.add_argument("--moe-loss-weight", type=float, default=0.1)

    data = parser.add_argument_group("data and batching")
    data.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    data.add_argument(
        "--data-path",
        default="",
        help="uint16 token file; defaults to <data-dir>/tokens_train.bin",
    )
    data.add_argument(
        "--eval-data-path",
        default="",
        help="uint16 token file; defaults to <data-dir>/tokens_val.bin, eval disabled when missing",
    )
    data.add_argument(
        "--dataset-source",
        choices=("hf", "modelscope"),
        default="hf",
        help="source used to auto-prepare the data when the token files are missing",
    )
    data.add_argument("--seq-length", type=_positive_int, default=512)
    data.add_argument("--micro-batch-size", type=_positive_int, default=8)

    train = parser.add_argument_group("training")
    train.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    train.add_argument("--steps", type=_positive_int, default=1000)
    train.add_argument("--lr", type=float, default=6e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--warmup-steps", type=_positive_int, default=100)
    train.add_argument("--seed", type=int, default=0)

    io = parser.add_argument_group("logging and checkpoints")
    io.add_argument("--log-interval", type=_positive_int, default=10)
    io.add_argument(
        "--eval-interval",
        type=_positive_int,
        default=100,
        help="interval in steps; > steps disables",
    )
    io.add_argument("--eval-batches", type=_positive_int, default=8)
    io.add_argument("--save", default="", help="checkpoint directory")
    io.add_argument(
        "--save-interval",
        type=_positive_int,
        default=200,
        help="interval in steps; > steps disables",
    )
    io.add_argument(
        "--load", default="", help="checkpoint file (or directory) to resume from",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size or 4 * args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        mlp_type=args.mlp_type,
        moe_loss_weight=args.moe_loss_weight,
        max_seq_len=args.seq_length,
        dtype=args.dtype,
    )


def _check_single_device() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise SystemExit(
            "This example runs on a single device; the multi-rank expert-parallel "
            "pathway is not supported on NPU yet. Do not launch it with torchrun.",
        )


def lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: nn.Module, val_stream: TokenStream, batches: int, device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    clear_load_balancing_loss()
    total = 0.0
    for _ in range(batches):
        inputs, targets = val_stream.next_batch()
        logits = model(inputs.to(device))
        total += next_token_loss(logits, targets.to(device)).item()
    model.train(was_training)
    clear_load_balancing_loss()
    return total / batches


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: ModelConfig,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": dict(config),
        },
        path,
    )
    return path


def _checkpoint_file(path: Path) -> Path:
    if path.is_dir():
        return path / "checkpoint.pt"
    return path


def _read_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    loaded = torch.load(_checkpoint_file(path), map_location=device, weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint {path} did not contain a dict")
    return loaded


def _apply_checkpoint(
    ckpt: dict[str, object], model: nn.Module, optimizer: torch.optim.Optimizer,
) -> int:
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def _resolve_streams(
    args: argparse.Namespace, seq_length: int,
) -> tuple[TokenStream, TokenStream | None]:
    if args.data_path:
        train_path = Path(args.data_path)
        if not train_path.is_file():
            raise SystemExit(f"token file not found: {train_path}")
    else:
        train_path = Path(args.data_dir) / "tokens_train.bin"
        if not train_path.is_file():
            print(
                f"token stream {train_path} not found; preparing the {args.dataset_source} dataset",
            )
            prepare(Path(args.data_dir), args.dataset_source)
    val_path = (
        Path(args.eval_data_path)
        if args.eval_data_path
        else Path(args.data_dir) / "tokens_val.bin"
    )
    val_stream = None
    if val_path.is_file():
        val_stream = TokenStream(
            val_path, args.micro_batch_size, seq_length, seed=args.seed,
        )
    elif args.eval_interval <= args.steps:
        print(f"no validation tokens at {val_path}; validation is disabled")
    train_stream = TokenStream(
        train_path, args.micro_batch_size, seq_length, seed=args.seed,
    )
    return train_stream, val_stream


def _train_step(
    model: nn.Module,
    mb_args: Arguments,
    optimizer: torch.optim.Optimizer,
    train_stream: TokenStream,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
) -> tuple[float, float]:
    for group in optimizer.param_groups:
        group["lr"] = args.lr * lr_multiplier(step, args.warmup_steps, args.steps)
    inputs, targets = train_stream.next_batch()
    logits = model(inputs.to(device))
    lm_loss = next_token_loss(logits, targets.to(device))
    aux_loss = batched_load_balancing_loss(mb_args)
    optimizer.zero_grad(set_to_none=True)
    (lm_loss + aux_loss).backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    clear_load_balancing_loss()
    return float(lm_loss.item()), float(aux_loss)


def train_loop(
    model: nn.Module,
    mb_args: Arguments,
    optimizer: torch.optim.Optimizer,
    train_stream: TokenStream,
    val_stream: TokenStream | None,
    args: argparse.Namespace,
    config: ModelConfig,
    device: torch.device,
    start_step: int = 0,
) -> tuple[float | None, float | None]:
    last_lm: float | None = None
    last_aux: float | None = None
    window_start = time.perf_counter()
    window_tokens = 0
    for step in range(start_step, args.steps):
        last_lm, last_aux = _train_step(
            model, mb_args, optimizer, train_stream, args, device, step,
        )
        done = step + 1
        window_tokens += train_stream.batch_size * train_stream.seq_length
        if done % args.log_interval == 0:
            elapsed = time.perf_counter() - window_start
            tokens_per_sec = window_tokens / elapsed if elapsed > 0 else 0.0
            lr = args.lr * lr_multiplier(step, args.warmup_steps, args.steps)
            print(
                f"step={done} lm_loss={last_lm:.4f} aux_loss={last_aux:.4f} "
                f"lr={lr:.2e} tokens_per_sec={tokens_per_sec:.0f}",
            )
            window_start = time.perf_counter()
            window_tokens = 0
        if val_stream is not None and done % args.eval_interval == 0:
            val_loss = evaluate(model, val_stream, args.eval_batches, device)
            print(f"step={done} val_loss={val_loss:.4f}")
        if args.save and done % args.save_interval == 0:
            path = save_checkpoint(Path(args.save), model, optimizer, done, config)
            print(f"checkpoint saved to {path}")
    return last_lm, last_aux


def _config_for_run(
    args: argparse.Namespace, device: torch.device,
) -> tuple[ModelConfig, dict[str, object] | None]:
    """Build a fresh config, or the config stored in ``--load``, before creating the model."""
    if not args.load:
        return build_config(args), None
    ckpt = _read_checkpoint(Path(args.load), device)
    raw_config = ckpt["config"]
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint config must be a dict")
    return parse_config(raw_config), ckpt


def main(argv: list[str] | None = None) -> dict[str, float | None]:
    args = parse_args(argv)
    _check_single_device()
    if args.data_path and not Path(args.data_path).is_file():
        raise SystemExit(f"token file not found: {args.data_path}")
    device = resolve_device()
    torch.manual_seed(args.seed)

    config, ckpt = _config_for_run(args, device)
    model, mb_args = build_model(config, device)
    num_params = sum(param.numel() for param in model.parameters())
    torch_npu_version = getattr(torch_npu, "__version__", "n/a")
    print(f"device: {device} (torch_npu {torch_npu_version})")
    print(
        f"model: {config['num_layers']} layers, hidden {config['hidden_size']}, "
        f"{config['num_experts']} experts x top-{config['top_k']}, "
        f"{num_params / 1e6:.1f}M params, dtype {config['dtype']}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    train_stream, val_stream = _resolve_streams(args, config["max_seq_len"])
    start_step = 0
    if ckpt is not None:
        start_step = _apply_checkpoint(ckpt, model, optimizer)
        print(f"resumed from {args.load} at step {start_step}")
    last_lm, last_aux = train_loop(
        model,
        mb_args,
        optimizer,
        train_stream,
        val_stream,
        args,
        config,
        device,
        start_step,
    )
    if last_lm is None or last_aux is None:
        print(f"nothing to train: resumed at step {start_step} >= --steps {args.steps}")
        return {"lm_loss": last_lm, "aux_loss": last_aux}
    if args.save:
        path = save_checkpoint(Path(args.save), model, optimizer, args.steps, config)
        print(f"checkpoint saved to {path}")

    summary: dict[str, float | None] = {"lm_loss": last_lm, "aux_loss": last_aux}
    print(f"summary: lm_loss={last_lm:.4f} aux_loss={last_aux:.4f}")
    return summary


if __name__ == "__main__":
    main()
