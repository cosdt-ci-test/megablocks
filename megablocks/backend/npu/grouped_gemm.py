# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""grouped_gemm-compatible gmm implemented with torch_npu.npu_grouped_matmul."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import torch
from torch.autograd.function import Function, FunctionCtx

try:
    import torch_npu as _torch_npu_mod
except ImportError:
    _torch_npu_mod = None


class GroupedMatmulFn(Protocol):
    def __call__(
        self,
        x: Sequence[torch.Tensor],
        weight: Sequence[torch.Tensor],
        *,
        group_list: torch.Tensor | None = None,
        split_item: int = 0,
        group_type: int | None = None,
        group_list_type: int = 0,
    ) -> list[torch.Tensor]:
        """Subset of torch_npu.npu_grouped_matmul used by this shim."""


def npu_grouped_matmul_is_available() -> bool:
    return _torch_npu_mod is not None and hasattr(_torch_npu_mod, 'npu_grouped_matmul')


def _grouped_matmul() -> GroupedMatmulFn:
    if _torch_npu_mod is None or not hasattr(_torch_npu_mod, 'npu_grouped_matmul'):
        raise ModuleNotFoundError('torch_npu.npu_grouped_matmul is required for grouped GEMM on NPU')
    return cast(GroupedMatmulFn, _torch_npu_mod.npu_grouped_matmul)


def _group_list(batch_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    sizes = batch_sizes.to(dtype=torch.int64, device='cpu').flatten()
    return torch.cumsum(sizes, 0).to(device=device, dtype=torch.int64)


def _matmul_m_grouped(
    a: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    return _grouped_matmul()(
        [a],
        [weight],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=0,
    )[0]


def _matmul_k_grouped(
    a_t: torch.Tensor,
    b: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    return _grouped_matmul()(
        [a_t],
        [b],
        group_list=group_list,
        split_item=3,
        group_type=2,
        group_list_type=0,
    )[0]


def _to_compute(tensor: torch.Tensor) -> torch.Tensor:
    """Promote fp16/bf16 GEMM to fp32. CPU same-dtype matmul accumulates in fp32."""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def _run_gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    if trans_a and trans_b:
        raise ValueError('trans_a and trans_b cannot both be True')
    group_list = _group_list(batch_sizes, a.device)
    a_c = _to_compute(a)
    b_c = _to_compute(b)
    if trans_a:
        # group_type=2 rejects a contiguous copy: x must stay a transpose view.
        return _matmul_k_grouped(a_c.t(), b_c, group_list)
    weight = b_c.transpose(1, 2).contiguous() if trans_b else b_c
    return _matmul_m_grouped(a_c, weight, group_list)


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    c: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match grouped_gemm.backend.gmm, including optional in-place ``c``."""
    result = _run_gmm(a, b, batch_sizes, trans_a, trans_b).to(dtype=a.dtype)
    if c is None:
        return result
    c.copy_(result)
    return c


class GroupedGemm(Function):
    """Autograd wrapper matching grouped_gemm 0.3.0 ops.gmm (trans_b only)."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b  # type: ignore[attr-defined]
        # Keep fp32 so GELU and the next grouped GEMM do not round-trip through fp16.
        return _run_gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = bool(ctx.trans_b)  # type: ignore[attr-defined]
        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = _run_gmm(grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)
        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = _run_gmm(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def ops_gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    return cast(torch.Tensor, GroupedGemm.apply(a, b, batch_sizes, trans_b))
