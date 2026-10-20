# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""A minimal dMoE transformer for the Ascend NPU example.

The model is intentionally small (learned embeddings, manual causal attention,
``dMoE`` blocks) so the entry points run the megablocks dispatch chain without
pulling in a full training framework. Activations use the ``[sl, bs, hs]``
layout expected by megablocks layers. Half precision is required: the NPU
grouped-GEMM pathway is exercised with bf16 or fp16 parameters.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Mapping, TypedDict

import torch
import torch.nn.functional as F
from torch import nn

from megablocks.layers.arguments import Arguments
from megablocks.layers.dmoe import dMoE


class ModelConfig(TypedDict):
    """Model hyperparameters; stored as a plain dict inside checkpoints."""

    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_layers: int
    vocab_size: int
    num_experts: int
    top_k: int
    mlp_type: str
    moe_loss_weight: float
    max_seq_len: int
    dtype: str


def resolve_device() -> torch.device:
    """Pick the accelerator this example should run on."""
    if getattr(torch, "npu", None) is not None and torch.npu.is_available():
        torch.npu.set_device(0)
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    raise RuntimeError(
        "No accelerator found. On Ascend import torch_npu before running "
        "(pip install torch_npu), or run on a CUDA GPU.",
    )


def resolve_dtype(name: str) -> torch.dtype:
    """Return the parameter dtype; the entry points require half precision."""
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(
        f"The entry points require half precision; got dtype {name!r}, expected bf16 or fp16.",
    )


def default_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=512,
        ffn_hidden_size=2048,
        num_attention_heads=8,
        num_layers=6,
        vocab_size=50257,
        num_experts=8,
        top_k=2,
        mlp_type="glu",
        moe_loss_weight=0.1,
        max_seq_len=512,
        dtype="bf16",
    )


def build_arguments(config: ModelConfig, device: torch.device) -> Arguments:
    """Assemble megablocks ``Arguments`` for the NPU-supported pathway.

    ``mlp_impl='grouped'`` is the only dropless implementation available on
    NPU; bias and returned biases are disabled to match the verified
    configuration in ``tests_ascend/test_dmoe.py``.
    """
    dtype = resolve_dtype(config["dtype"])
    return Arguments(
        hidden_size=config["hidden_size"],
        ffn_hidden_size=config["ffn_hidden_size"],
        num_layers=config["num_layers"],
        moe_num_experts=config["num_experts"],
        moe_top_k=config["top_k"],
        mlp_type=config["mlp_type"],
        moe_loss_weight=config["moe_loss_weight"],
        mlp_impl="grouped",
        memory_optimized_mlp=False,
        bias=False,
        return_bias=False,
        fp16=dtype is torch.float16,
        bf16=dtype is torch.bfloat16,
        init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.02),
        output_layer_init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.02),
        device=device,
    )


def _require_int(raw: Mapping[str, object], name: str) -> int:
    value = raw.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config field {name!r} must be an int, got {value!r}")
    return value


def _require_float(raw: Mapping[str, object], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config field {name!r} must be a float, got {value!r}")
    return float(value)


def _require_choice(
    raw: Mapping[str, object], name: str, choices: tuple[str, ...],
) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"config field {name!r} must be one of {choices}, got {value!r}",
        )
    return value


def parse_config(raw: Mapping[str, object]) -> ModelConfig:
    """Validate a config mapping loaded from a checkpoint before rebuilding a model."""
    return ModelConfig(
        hidden_size=_require_int(raw, "hidden_size"),
        ffn_hidden_size=_require_int(raw, "ffn_hidden_size"),
        num_attention_heads=_require_int(raw, "num_attention_heads"),
        num_layers=_require_int(raw, "num_layers"),
        vocab_size=_require_int(raw, "vocab_size"),
        num_experts=_require_int(raw, "num_experts"),
        top_k=_require_int(raw, "top_k"),
        max_seq_len=_require_int(raw, "max_seq_len"),
        mlp_type=_require_choice(raw, "mlp_type", ("mlp", "glu")),
        moe_loss_weight=_require_float(raw, "moe_loss_weight"),
        dtype=_require_choice(raw, "dtype", ("bf16", "fp16")),
    )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention over ``[sl, bs, hs]`` activations."""

    num_heads: int
    head_dim: int
    qkv: nn.Linear
    proj: nn.Linear

    def __init__(
        self, hidden_size: int, num_heads: int, device: torch.device, dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}",
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(
            hidden_size, 3 * hidden_size, bias=False, device=device, dtype=dtype,
        )
        self.proj = nn.Linear(
            hidden_size, hidden_size, bias=False, device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sl, bs, hs = x.shape
        qkv = self.qkv(x).view(sl, bs, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 3, 1, 0, 4).unbind(0)  # each [nh, bs, sl, hd]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(
            self.head_dim,
        )  # [nh, bs, sl, sl]
        causal = torch.triu(
            torch.ones(sl, sl, dtype=torch.bool, device=x.device), diagonal=1,
        )
        probs = F.softmax(scores.float().masked_fill(causal, float("-inf")), dim=-1)
        mixed = probs.to(v.dtype) @ v  # [nh, bs, sl, hd]
        out = mixed.permute(2, 1, 0, 3).reshape(sl, bs, hs)  # [sl, bs, hs]
        return self.proj(out)


class DmoETransformerBlock(nn.Module):
    """Pre-norm transformer block whose FFN is a megablocks ``dMoE`` layer."""

    attn_norm: nn.LayerNorm
    attention: CausalSelfAttention
    moe_norm: nn.LayerNorm
    moe: dMoE

    def __init__(
        self, args: Arguments, num_heads: int, device: torch.device, dtype: torch.dtype,
    ) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        self.attn_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.attention = CausalSelfAttention(
            hidden_size, num_heads, device=device, dtype=dtype,
        )
        self.moe_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.moe = dMoE(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x


class GptDmoEModel(nn.Module):
    """Decoder-only dMoE language model built on the megablocks entry pathway."""

    config: ModelConfig
    max_seq_len: int
    wte: nn.Embedding
    wpe: nn.Embedding
    blocks: nn.ModuleList
    ln_f: nn.LayerNorm
    lm_head: nn.Linear

    def __init__(
        self, config: ModelConfig, args: Arguments, device: torch.device,
    ) -> None:
        super().__init__()
        dtype = resolve_dtype(config["dtype"])
        self.config = config
        self.max_seq_len = config["max_seq_len"]
        hidden_size = config["hidden_size"]
        self.wte = nn.Embedding(
            config["vocab_size"], hidden_size, device=device, dtype=dtype,
        )
        self.wpe = nn.Embedding(
            config["max_seq_len"], hidden_size, device=device, dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            DmoETransformerBlock(
                args, config["num_attention_heads"], device=device, dtype=dtype,
            )
            for _ in range(config["num_layers"])
        )
        self.ln_f = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            hidden_size, config["vocab_size"], bias=False, device=device, dtype=dtype,
        )
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map ``[bs, sl]`` token ids to ``[bs, sl, vocab]`` logits."""
        _, sl = tokens.shape
        if sl > self.max_seq_len:
            raise ValueError(
                f"sequence length {sl} exceeds max_seq_len {self.max_seq_len}",
            )
        positions = torch.arange(sl, device=tokens.device)
        x = self.wte(tokens) + self.wpe(positions).unsqueeze(0)
        x = x.transpose(0, 1)  # [sl, bs, hs]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x.transpose(0, 1))


def build_model(
    config: ModelConfig, device: torch.device,
) -> tuple[GptDmoEModel, Arguments]:
    """Create the model and its megablocks ``Arguments`` on ``device``."""
    args = build_arguments(config, device)
    return GptDmoEModel(config, args, device), args


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token cross entropy for ``[bs, sl, vocab]`` logits and ``[bs, sl]`` targets."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
    )
