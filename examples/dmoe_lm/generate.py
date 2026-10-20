# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Generate text from a dMoE checkpoint trained by ``train.py``.

Single device::

    python generate.py --checkpoint checkpoints/checkpoint.pt --prompt 'Once upon a time'

The tokenizer saved by ``python data.py prepare`` (under ``<data-dir>/tokenizer``)
is used to encode the prompt and decode the output; pass ``--tokenizer-path``
when the checkpoint was trained on a different data directory.
"""

from __future__ import annotations

import os

# Triton-ascend reads these at import time, before any megablocks import;
# see train.py for the rationale.
os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".triton_cache"),
)

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    import torch_npu  # pyrefly: ignore[missing-import]  # registers the npu backend
except ImportError:
    torch_npu = None

from model import GptDmoEModel, build_model, parse_config, resolve_device

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PROMPT = "The meaning of life is"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint file (or directory) from train.py",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="defaults to <data-dir>/tokenizer, then the hub gpt2",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="fallback location of the tokenizer",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="prompt text; defaults to the first validation tokens",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--top-k", type=int, default=0, help="0 disables top-k filtering",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="1.0 disables nucleus filtering",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="argmax decoding, ignores the sampling flags",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _auto_tokenizer(name: str) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def load_tokenizer(path: str, data_dir: Path) -> PreTrainedTokenizerBase:
    candidates = ([path] if path else []) + [str(data_dir / "tokenizer")]
    for candidate in candidates:
        if Path(candidate).is_dir():
            return _auto_tokenizer(candidate)
    print(
        "no local tokenizer found; loading gpt2 from the hub (set HF_ENDPOINT if needed)",
    )
    return _auto_tokenizer("gpt2")


def prompt_token_ids(
    tokenizer: PreTrainedTokenizerBase, prompt: str, val_path: Path, limit: int = 32,
) -> list[int]:
    if prompt:
        return list(tokenizer.encode(prompt))
    if val_path.exists():
        ids = np.memmap(val_path, dtype=np.uint16, mode="r")[:limit]
        print(
            f"no --prompt given; using the first {ids.size} validation tokens as the context",
        )
        return [int(token) for token in ids]
    print("no --prompt given and no validation tokens; using the default prompt")
    return list(tokenizer.encode(DEFAULT_PROMPT))


def sample_token(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, greedy: bool,
) -> int:
    logits = logits.float()
    if greedy or temperature <= 0:
        return int(logits.argmax(dim=-1))
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        removable = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(removable, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits,
        )
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


@torch.no_grad()
def generate(
    model: GptDmoEModel,
    token_ids: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> list[int]:
    model.eval()
    context = torch.tensor([token_ids], dtype=torch.int64, device=device)
    generated: list[int] = []
    for _ in range(args.max_new_tokens):
        logits = model(context[:, -model.max_seq_len :])
        next_id = sample_token(
            logits[0, -1], args.temperature, args.top_k, args.top_p, args.greedy,
        )
        generated.append(next_id)
        context = torch.cat(
            [context, torch.tensor([[next_id]], dtype=torch.int64, device=device)],
            dim=1,
        )
    return generated


def _load_model(
    checkpoint: str, device: torch.device,
) -> tuple[GptDmoEModel, Path, int]:
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = parse_config(ckpt["config"])
    model, _ = build_model(config, device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt_path, int(ckpt["step"])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device()
    torch.manual_seed(args.seed)
    torch_npu_version = getattr(torch_npu, "__version__", "n/a")
    print(f"device: {device} (torch_npu {torch_npu_version})")

    model, ckpt_path, step = _load_model(args.checkpoint, device)
    config = model.config
    print(
        f"loaded {ckpt_path} (step {step}, "
        f"{config['num_experts']} experts x top-{config['top_k']}, dtype {config['dtype']})",
    )

    tokenizer = load_tokenizer(args.tokenizer_path, Path(args.data_dir))
    val_path = Path(args.data_dir) / "tokens_val.bin"
    token_ids = prompt_token_ids(tokenizer, args.prompt, val_path)
    print(f"prompt: {tokenizer.decode(token_ids)}")

    generated = generate(model, token_ids, args, device)
    print(f"generated: {tokenizer.decode(generated)}")


if __name__ == "__main__":
    main()
