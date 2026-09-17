# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from megablocks.backend.npu.grouped_gemm import gmm
from tests_ascend.precision import assert_close_to_sim
from tests_ascend.reference import grouped_matmul_cpu

pytestmark = pytest.mark.npu

_DTYPES = (torch.bfloat16, torch.float16)
_M, _K, _N = 32, 16, 8


def _run_case(
    batch_sizes: list[int],
    dtype: torch.dtype,
    *,
    trans_a: bool,
    trans_b: bool,
) -> None:
    torch.manual_seed(0)
    groups = len(batch_sizes)
    rows = sum(batch_sizes)
    sizes = torch.tensor(batch_sizes, dtype=torch.int64)
    if trans_a:
        a = torch.randn(rows, _N, dtype=dtype, device='npu')
        b = torch.randn(rows, _K, dtype=dtype, device='npu')
    else:
        a = torch.randn(rows, _K, dtype=dtype, device='npu')
        weight_n, weight_k = (_K, _N) if not trans_b else (_N, _K)
        b = torch.randn(groups, weight_n, weight_k, dtype=dtype, device='npu')
    actual = gmm(a, b, sizes, trans_a=trans_a, trans_b=trans_b)
    golden = grouped_matmul_cpu(a.cpu().float(), b.cpu().float(), sizes, trans_a=trans_a, trans_b=trans_b)
    sim = grouped_matmul_cpu(a.cpu(), b.cpu(), sizes, trans_a=trans_a, trans_b=trans_b).to(dtype)
    assert_close_to_sim(golden, sim, actual.cpu())


@pytest.mark.parametrize('dtype', _DTYPES)
def test_gmm_no_transpose(dtype: torch.dtype) -> None:
    _run_case([8, 8, 8, 8], dtype, trans_a=False, trans_b=False)


@pytest.mark.parametrize('dtype', _DTYPES)
def test_gmm_trans_b(dtype: torch.dtype) -> None:
    _run_case([8, 8, 8, 8], dtype, trans_a=False, trans_b=True)


@pytest.mark.parametrize('dtype', _DTYPES)
def test_gmm_trans_a(dtype: torch.dtype) -> None:
    _run_case([8, 8, 8, 8], dtype, trans_a=True, trans_b=False)


@pytest.mark.parametrize('dtype', _DTYPES)
def test_gmm_empty_group(dtype: torch.dtype) -> None:
    _run_case([8, 0, 16, 8], dtype, trans_a=False, trans_b=True)
    _run_case([8, 0, 16, 8], dtype, trans_a=True, trans_b=False)


@pytest.mark.parametrize('dtype', _DTYPES)
def test_gmm_single_expert(dtype: torch.dtype) -> None:
    _run_case([32], dtype, trans_a=False, trans_b=False)
    _run_case([32], dtype, trans_a=False, trans_b=True)
    _run_case([32], dtype, trans_a=True, trans_b=False)


@pytest.mark.parametrize('dtype', _DTYPES)
def test_gmm_writes_c(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    a = torch.randn(_M, _K, dtype=dtype, device='npu')
    b = torch.randn(1, _K, _N, dtype=dtype, device='npu')
    sizes = torch.tensor([_M], dtype=torch.int64)
    buf = torch.zeros(_M, _N, dtype=dtype, device='npu')
    out = gmm(a, b, sizes, c=buf)
    assert out.data_ptr() == buf.data_ptr()
    golden = grouped_matmul_cpu(a.cpu().float(), b.cpu().float(), sizes, trans_a=False, trans_b=False)
    sim = grouped_matmul_cpu(a.cpu(), b.cpu(), sizes, trans_a=False, trans_b=False).to(dtype)
    assert_close_to_sim(golden, sim, out.cpu())
