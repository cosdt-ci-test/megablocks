# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from megablocks.backend import dispatch
from megablocks.ops import gather, histogram, inclusive_cumsum, scatter, sort
from tests_ascend.precision import assert_close_to_sim
from tests_ascend.reference import padded_gather_np, padded_scatter_np, scatter_wgrad_np

pytestmark = pytest.mark.npu

_PERMUTE_SHAPES = (
    (32, 64, 4, 1),
    (32, 64, 4, 2),
    (16, 128, 8, 1),
)
_DTYPES = (torch.bfloat16, torch.float16)


def _route(
    tokens: int,
    hidden: int,
    experts: int,
    top_k: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, dtype=dtype, device='npu')
    top_expert = torch.randint(0, experts, (tokens * top_k,), device='npu', dtype=torch.int32)
    bin_ids, indices = sort(top_expert, 32)
    bins = inclusive_cumsum(histogram(top_expert, experts), 0)
    weights = torch.rand(tokens * top_k, dtype=dtype, device='npu')
    return x, indices, bin_ids, bins, weights


@pytest.mark.parametrize(('tokens', 'hidden', 'experts', 'top_k'), _PERMUTE_SHAPES)
@pytest.mark.parametrize('dtype', _DTYPES)
def test_gather(tokens: int, hidden: int, experts: int, top_k: int, dtype: torch.dtype) -> None:
    x, indices, bin_ids, bins, _weights = _route(tokens, hidden, experts, top_k, dtype)
    actual = gather(x, indices, bin_ids, bins, top_k)
    golden = padded_gather_np(x.float(), indices, bins, bins, top_k)
    sim = padded_gather_np(x, indices, bins, bins, top_k).to(dtype)
    assert_close_to_sim(golden, sim, actual.cpu())


@pytest.mark.parametrize(('tokens', 'hidden', 'experts', 'top_k'), _PERMUTE_SHAPES)
@pytest.mark.parametrize('dtype', _DTYPES)
def test_scatter(tokens: int, hidden: int, experts: int, top_k: int, dtype: torch.dtype) -> None:
    x, indices, bin_ids, bins, weights = _route(tokens, hidden, experts, top_k, dtype)
    gathered = gather(x, indices, bin_ids, bins, top_k)
    actual = scatter(gathered, indices, bin_ids, weights, bins, top_k)
    golden = padded_scatter_np(gathered.float(), indices, weights.float(), bins, bins, top_k)
    sim = padded_scatter_np(gathered, indices, weights, bins, bins, top_k).to(dtype)
    assert actual is not None  # megablocks.ops.scatter is annotated Optional[Tensor].
    assert_close_to_sim(golden, sim, actual.cpu())


@pytest.mark.parametrize(('tokens', 'hidden', 'experts', 'top_k'), _PERMUTE_SHAPES)
@pytest.mark.parametrize('dtype', _DTYPES)
def test_scatter_wgrad(tokens: int, hidden: int, experts: int, top_k: int, dtype: torch.dtype) -> None:
    x, indices, bin_ids, bins, _weights = _route(tokens, hidden, experts, top_k, dtype)
    gathered = gather(x, indices, bin_ids, bins, top_k)
    grad = torch.randn(tokens, hidden, dtype=dtype, device='npu')
    actual = dispatch.scatter_wgrad(gathered, grad, indices, bin_ids, bins, top_k)
    golden = scatter_wgrad_np(gathered.float(), grad.float(), indices, bins, bins, top_k)
    sim = scatter_wgrad_np(gathered, grad, indices, bins, bins, top_k).to(dtype)
    assert_close_to_sim(golden, sim, actual.cpu())
