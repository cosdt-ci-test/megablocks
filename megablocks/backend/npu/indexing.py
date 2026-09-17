# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Torch-native replacements for CUB sort / histogram / inclusive_cumsum."""

from __future__ import annotations

import torch


def sort(x: torch.Tensor, end_bit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable sort with iota indices cast back to the input integer dtype.

    `end_bit` is accepted for signature parity with the CUB radix sort wrapper
    and is unused: `torch.sort` is not a bit-width-limited radix sort.
    """
    del end_bit
    values, indices = torch.sort(x, stable=True)
    return values, indices.to(x.dtype)


def histogram(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Count occurrences of each integer in ``[0, num_bins)``.

    Only the 1-d dMoE path is supported. 2-d batched histograms are used by
    the sparse topology path, which is out of scope.
    """
    if x.ndim != 1:
        raise NotImplementedError('NPU histogram only supports 1-d input')
    counts = torch.bincount(x.to(torch.int64), minlength=num_bins)
    return counts[:num_bins].to(dtype=torch.int32)


def inclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Inclusive prefix sum matching the CUDA Python wrapper's 1-d quirk.

    The CUDA wrapper reshapes 1-d inputs to ``[1, N]`` and scans the last
    dimension, ignoring ``dim``. 1-d inputs therefore scan along dim 0.
    """
    if x.ndim == 1:
        return torch.cumsum(x, 0)
    return torch.cumsum(x, dim)
