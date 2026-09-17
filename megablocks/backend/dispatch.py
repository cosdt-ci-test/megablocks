# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Route indexing, permute, and grouped GEMM calls to CUDA or NPU backends."""

from __future__ import annotations

from typing import Protocol, cast

import torch

from megablocks.backend.device import Backend, backend_of
from megablocks.backend.npu import grouped_gemm as npu_gmm, indexing as npu_indexing

_GROUPED_GEMM_MISSING = (
    'Grouped GEMM not available. Please run `pip install git+https://github.com/tgale96/grouped_gemm@main`.'
)

_cuda_ops_error: ImportError | None = None
_cuda_kernels_error: ImportError | None = None
_npu_kernels_error: ImportError | None = None
_cuda_grouped_gemm_error: ImportError | None = None

try:
    import megablocks_ops as _cuda_ops_mod
except ImportError as error:
    _cuda_ops_mod = None
    _cuda_ops_error = error

try:
    from megablocks.backend import kernels as _cuda_kernels_mod
except ImportError as error:
    _cuda_kernels_mod = None
    _cuda_kernels_error = error

try:
    from megablocks.backend.npu import kernels as _npu_kernels_mod
except ImportError as error:
    _npu_kernels_mod = None
    _npu_kernels_error = error

try:
    import grouped_gemm as _cuda_grouped_gemm_mod
except ImportError as error:
    _cuda_grouped_gemm_mod = None
    _cuda_grouped_gemm_error = error


class _CudaIndexingOps(Protocol):
    def sort(
        self,
        x: torch.Tensor,
        end_bit: int,
        x_out: torch.Tensor,
        iota_out: torch.Tensor,
    ) -> None:
        """CUB DeviceRadixSort::SortPairs wrapper."""

    def histogram(self, x: torch.Tensor, max_val: int) -> torch.Tensor:
        """CUB DeviceHistogram::HistogramEven wrapper."""

    def inclusive_cumsum(self, x: torch.Tensor, dim: int, out: torch.Tensor) -> None:
        """CUB DeviceScan::InclusiveSum wrapper."""


class _PermuteKernels(Protocol):
    def gather(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor | None,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Gather tokens into expert-major order."""

    def scatter(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor | None,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Scatter expert outputs back to token order."""

    def scatter_wgrad(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Router-weight gradient for scatter."""

    def padded_gather(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor | None,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Gather with padded expert bins."""

    def padded_scatter(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor | None,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Scatter with padded expert bins."""

    def padded_scatter_wgrad(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Router-weight gradient for padded scatter."""

    def binned_gather(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor | None,
        bins: torch.Tensor,
        expert_capacity: int,
        top_k: int,
    ) -> torch.Tensor:
        """Capacity-binned gather used by MoE, not dMoE."""

    def binned_scatter(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor | None,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Capacity-binned scatter used by MoE, not dMoE."""

    def binned_scatter_wgrad(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Capacity-binned router-weight gradient."""


class _GroupedGemmBackend(Protocol):
    def gmm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_a: bool = False,
        trans_b: bool = False,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw grouped GEMM kernel."""


class _GroupedGemmOps(Protocol):
    def gmm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool = False,
    ) -> torch.Tensor:
        """Autograd grouped GEMM."""


def _require_cuda_ops() -> _CudaIndexingOps:
    if _cuda_ops_mod is None:
        raise ImportError('megablocks_ops is unavailable; build the c++ operations.') from _cuda_ops_error
    return cast(_CudaIndexingOps, _cuda_ops_mod)


def _kernels_for(kind: Backend) -> _PermuteKernels:
    module = _npu_kernels_mod if kind == 'npu' else _cuda_kernels_mod
    if module is not None:
        return cast(_PermuteKernels, module)
    # The module is in-tree, so this is a triton import failure, not a missing file.
    name = 'megablocks.backend.npu.kernels' if kind == 'npu' else 'megablocks.backend.kernels'
    error = _npu_kernels_error if kind == 'npu' else _cuda_kernels_error
    raise ImportError(f'{name} failed to import; its triton backend is unavailable.') from error


def _cuda_grouped_backend() -> _GroupedGemmBackend:
    if _cuda_grouped_gemm_mod is None:
        raise ImportError(_GROUPED_GEMM_MISSING) from _cuda_grouped_gemm_error
    return cast(_GroupedGemmBackend, _cuda_grouped_gemm_mod.backend)


def _cuda_grouped_ops() -> _GroupedGemmOps:
    if _cuda_grouped_gemm_mod is None:
        raise ImportError(_GROUPED_GEMM_MISSING) from _cuda_grouped_gemm_error
    return cast(_GroupedGemmOps, _cuda_grouped_gemm_mod.ops)


def grouped_gemm_is_available() -> bool:
    return _cuda_grouped_gemm_mod is not None or npu_gmm.npu_grouped_matmul_is_available()


def assert_grouped_gemm_is_available(device: int | torch.device | None = None) -> None:
    # An int device is always a CUDA ordinal, so only a torch.device can select NPU.
    if isinstance(device, torch.device) and device.type == 'npu':
        assert npu_gmm.npu_grouped_matmul_is_available(), (
            'torch_npu.npu_grouped_matmul is required for grouped MLP on NPU'
        )
        return
    assert _cuda_grouped_gemm_mod is not None, _GROUPED_GEMM_MISSING


def sort(x: torch.Tensor, end_bit: int) -> tuple[torch.Tensor, torch.Tensor]:
    if backend_of(x) == 'npu':
        return npu_indexing.sort(x, end_bit)
    x_out = torch.empty_like(x)
    iota_out = torch.empty_like(x)
    _require_cuda_ops().sort(x, end_bit, x_out, iota_out)
    return x_out, iota_out


def histogram(x: torch.Tensor, max_val: int) -> torch.Tensor:
    if backend_of(x) == 'npu':
        return npu_indexing.histogram(x, max_val)
    return _require_cuda_ops().histogram(x, max_val)


def inclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    if backend_of(x) == 'npu':
        return npu_indexing.inclusive_cumsum(x, dim)
    if x.ndim == 1:
        x = x.view([1, -1])
        out = torch.empty_like(x)
        _require_cuda_ops().inclusive_cumsum(x, 1, out)
        return out.squeeze()
    out = torch.empty_like(x)
    _require_cuda_ops().inclusive_cumsum(x, dim, out)
    return out


def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor | None,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return _kernels_for(backend_of(x)).gather(x, indices, bin_ids, weights, bins, top_k)


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor | None,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return _kernels_for(backend_of(x)).scatter(x, indices, bin_ids, weights, bins, top_k)


def scatter_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return _kernels_for(backend_of(x)).scatter_wgrad(x, grad, indices, bin_ids, bins, top_k)


def padded_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor | None,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return _kernels_for(backend_of(x)).padded_gather(
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        top_k,
    )


def padded_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor | None,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return _kernels_for(backend_of(x)).padded_scatter(
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        top_k,
    )


def padded_scatter_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return _kernels_for(backend_of(x)).padded_scatter_wgrad(
        x,
        grad,
        indices,
        bin_ids,
        bins,
        padded_bins,
        top_k,
    )


def _reject_npu_binned(x: torch.Tensor) -> None:
    if backend_of(x) == 'npu':
        raise NotImplementedError('binned permute kernels are not supported on NPU')


def binned_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None,
    bins: torch.Tensor,
    expert_capacity: int,
    top_k: int,
) -> torch.Tensor:
    _reject_npu_binned(x)
    return _kernels_for('cuda').binned_gather(x, indices, weights, bins, expert_capacity, top_k)


def binned_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    _reject_npu_binned(x)
    return _kernels_for('cuda').binned_scatter(x, indices, weights, bins, top_k)


def binned_scatter_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    indices: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    _reject_npu_binned(x)
    return _kernels_for('cuda').binned_scatter_wgrad(x, grad, indices, bins, top_k)


class GmmOps:
    @staticmethod
    def gmm(
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool = False,
    ) -> torch.Tensor:
        if backend_of(a) == 'npu':
            return npu_gmm.ops_gmm(a, b, batch_sizes, trans_b=trans_b)
        return _cuda_grouped_ops().gmm(a, b, batch_sizes, trans_b=trans_b)


class GmmBackend:
    @staticmethod
    def gmm(
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_a: bool = False,
        trans_b: bool = False,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if backend_of(a) == 'npu':
            return npu_gmm.gmm(a, b, batch_sizes, trans_a=trans_a, trans_b=trans_b, c=c)
        return _cuda_grouped_backend().gmm(a, b, batch_sizes, trans_a=trans_a, trans_b=trans_b, c=c)


ops: GmmOps = GmmOps()
backend: GmmBackend = GmmBackend()
