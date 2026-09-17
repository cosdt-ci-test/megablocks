# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from megablocks.backend.npu import indexing
from megablocks.ops import histogram, inclusive_cumsum, sort

pytestmark = pytest.mark.npu

_SORT_KEYS = [3, 1, 3, 2, 3, 1, 1, 2]


def _npu_int(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device='npu')


def test_sort_matches_cpu_and_is_stable() -> None:
    keys = _npu_int(_SORT_KEYS)
    values, indices = sort(keys, 32)
    cpu_values, cpu_indices = torch.sort(keys.cpu(), stable=True)
    assert torch.equal(values.cpu(), cpu_values)
    assert torch.equal(indices.cpu(), cpu_indices.to(keys.dtype))
    assert indices.dtype == keys.dtype
    for key in sorted(set(_SORT_KEYS)):
        original = (keys.cpu() == key).nonzero(as_tuple=False).flatten()
        got = indices.cpu()[values.cpu() == key]
        assert torch.equal(got, original)


def test_sort_via_indexing_module() -> None:
    keys = _npu_int(_SORT_KEYS)
    values, indices = indexing.sort(keys, end_bit=8)
    assert values.shape == keys.shape
    assert indices.dtype == torch.int32


def test_histogram_matches_bincount() -> None:
    keys = _npu_int(_SORT_KEYS)
    num_bins = 4
    got = histogram(keys, num_bins)
    expected = torch.bincount(keys.cpu().to(torch.int64), minlength=num_bins)[:num_bins].to(torch.int32)
    assert torch.equal(got.cpu(), expected)


def test_histogram_rejects_2d() -> None:
    keys = torch.randint(0, 4, (2, 8), dtype=torch.int32, device='npu')
    with pytest.raises(NotImplementedError, match='1-d'):
        histogram(keys, 4)


def test_inclusive_cumsum_1d_ignores_dim() -> None:
    values = _npu_int([1, 2, 3, 4])
    along_zero = inclusive_cumsum(values, 0)
    along_one = inclusive_cumsum(values, 1)
    expected = torch.tensor([1, 3, 6, 10], dtype=torch.int32)
    assert torch.equal(along_zero.cpu(), expected)
    assert torch.equal(along_one.cpu(), expected)


def test_inclusive_cumsum_2d_uses_dim() -> None:
    values = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device='npu')
    got = inclusive_cumsum(values, 1)
    expected = torch.tensor([[1, 3, 6], [4, 9, 15]], dtype=torch.int32)
    assert torch.equal(got.cpu(), expected)
