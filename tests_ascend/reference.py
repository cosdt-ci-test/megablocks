# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""CPU / numpy goldens for permute, grouped GEMM, and dMoE."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    cpu = tensor.detach().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.to(torch.float32)
    return cpu.numpy()


def padded_gather_np(
    x: torch.Tensor,
    indices: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    x_np = _to_numpy(x).astype(np.float32)
    indices_np = _to_numpy(indices)
    bins_np = _to_numpy(bins)
    padded_np = _to_numpy(padded_bins)
    out = np.zeros((int(padded_np[-1]), x_np.shape[1]), dtype=np.float32)
    in_idx = 0
    for expert, end in enumerate(bins_np):
        out_idx = 0 if expert == 0 else int(padded_np[expert - 1])
        while in_idx < end:
            out[out_idx, :] = x_np[int(indices_np[in_idx]) // top_k, :]
            in_idx += 1
            out_idx += 1
    return torch.from_numpy(out)


def padded_scatter_np(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    x_np = _to_numpy(x).astype(np.float32)
    indices_np = _to_numpy(indices)
    bins_np = _to_numpy(bins)
    padded_np = _to_numpy(padded_bins)
    weight_np = None if weights is None else _to_numpy(weights).astype(np.float32)
    out = np.zeros((indices_np.shape[0] // top_k, x_np.shape[-1]), dtype=np.float32)
    out_idx = 0
    for expert, end in enumerate(bins_np):
        in_idx = 0 if expert == 0 else int(padded_np[expert - 1])
        while out_idx < end:
            store_idx = int(indices_np[out_idx])
            scale = 1.0 if weight_np is None else float(weight_np[store_idx])
            out[store_idx // top_k, :] += scale * x_np[in_idx, :]
            out_idx += 1
            in_idx += 1
    return torch.from_numpy(out)


def scatter_wgrad_np(
    x: torch.Tensor,
    grad: torch.Tensor,
    indices: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    x_np = _to_numpy(x).astype(np.float32)
    grad_np = _to_numpy(grad).astype(np.float32)
    indices_np = _to_numpy(indices)
    bins_np = _to_numpy(bins)
    padded_np = _to_numpy(padded_bins)
    out = np.zeros(indices_np.shape, dtype=np.float32)
    in_idx = 0
    for expert, end in enumerate(bins_np):
        x_idx = 0 if expert == 0 else int(padded_np[expert - 1])
        while in_idx < end:
            token = int(indices_np[in_idx]) // top_k
            out[int(indices_np[in_idx])] = np.sum(x_np[x_idx, :] * grad_np[token])
            in_idx += 1
            x_idx += 1
    return torch.from_numpy(out)


def grouped_matmul_cpu(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    *,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    sizes = batch_sizes.detach().cpu().tolist()
    if trans_a:
        return _wgrad_cpu(a, b, sizes)
    offset = 0
    pieces: list[torch.Tensor] = []
    for expert, size in enumerate(sizes):
        left = a[offset : offset + size].float()
        weight = b[expert].float()
        if trans_b:
            weight = weight.transpose(0, 1)
        if size == 0:
            pieces.append(left.new_zeros((0, weight.shape[1])))
        else:
            pieces.append(left @ weight)
        offset += size
    return torch.cat(pieces, dim=0)


def _wgrad_cpu(a: torch.Tensor, b: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    offset = 0
    pieces: list[torch.Tensor] = []
    cols_a = a.shape[1]
    cols_b = b.shape[1]
    for size in sizes:
        left = a[offset : offset + size].float()
        right = b[offset : offset + size].float()
        if size == 0:
            pieces.append(a.new_zeros((cols_a, cols_b), dtype=torch.float32))
        else:
            pieces.append(left.t() @ right)
        offset += size
    return torch.stack(pieces, dim=0)


def dmoe_tokens(
    tokens: torch.Tensor,
    router_weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    activation: Callable[[torch.Tensor], torch.Tensor],
    v1: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = tokens @ router_weight.t()
    scores = logits.softmax(dim=-1)
    if top_k == 1:
        expert_weights, expert_ids = scores.max(dim=-1, keepdim=True)
    else:
        expert_weights, expert_ids = scores.topk(top_k, dim=-1)
    hidden = tokens.shape[1]
    out = tokens.new_zeros(tokens.shape[0], hidden)
    for expert in range(w1.shape[0]):
        slot = expert_ids == expert
        if not bool(slot.any()):
            continue
        token_index, k_index = slot.nonzero(as_tuple=True)
        hidden_out = activation(tokens[token_index] @ w1[expert].t())
        if v1 is not None:
            hidden_out = hidden_out * (tokens[token_index] @ v1[expert].t())
        scaled = (hidden_out @ w2[expert]) * expert_weights[token_index, k_index].unsqueeze(-1)
        out.index_add_(0, token_index, scaled)
    return out


def default_activation(hidden: torch.Tensor) -> torch.Tensor:
    return F.gelu(hidden, approximate='tanh')
