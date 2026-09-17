# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Three-way MARE / MERE / RMSE precision check against a CPU float32 golden."""

from __future__ import annotations

import torch

_MIN_ERR = 1e-7
_MARE_RATE = 10.0
_MERE_RATE = 2.0
_RMSE_RATE = 2.0


def err_threshold(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 2**-8
    if dtype == torch.float16:
        return 2**-11
    if dtype == torch.float32:
        return 2**-14
    raise ValueError(f'Unsupported dtype for precision check: {dtype}')


def _relative_error(golden: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    golden_f = golden.to(torch.float32)
    return torch.abs(actual.to(torch.float32) - golden_f) / (torch.abs(golden_f) + _MIN_ERR)


def mare(golden: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return torch.max(_relative_error(golden, actual).flatten())


def mere(golden: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return torch.mean(_relative_error(golden, actual))


def rmse(golden: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    golden_f = golden.to(torch.float32)
    return torch.sqrt(torch.mean(torch.pow(actual.to(torch.float32) - golden_f, 2)))


def _rate(npu_err: torch.Tensor, sim_err: torch.Tensor, floor: float) -> float:
    return float(npu_err / max(float(sim_err), floor))


def assert_close_to_sim(
    golden: torch.Tensor,
    sim: torch.Tensor,
    actual: torch.Tensor,
    *,
    what: str = 'tensor',
) -> None:
    """Pass iff NPU error / max(sim error, ulp floor) stays under the FLA defaults."""
    floor = err_threshold(actual.dtype)
    npu_mare, sim_mare = mare(golden, actual), mare(golden, sim)
    npu_mere, sim_mere = mere(golden, actual), mere(golden, sim)
    npu_rmse, sim_rmse = rmse(golden, actual), rmse(golden, sim)
    mare_rate = _rate(npu_mare, sim_mare, floor)
    mere_rate = _rate(npu_mere, sim_mere, floor)
    rmse_rate = _rate(npu_rmse, sim_rmse, floor)
    if mare_rate < _MARE_RATE and mere_rate < _MERE_RATE and rmse_rate < _RMSE_RATE:
        return
    raise AssertionError(
        f'{what}: NPU precision exceeded sim-relative thresholds: '
        f'MARE={mare_rate:.4g} (npu={float(npu_mare):.4g}, sim={float(sim_mare):.4g}, max {_MARE_RATE}), '
        f'MERE={mere_rate:.4g} (npu={float(npu_mere):.4g}, sim={float(sim_mere):.4g}, max {_MERE_RATE}), '
        f'RMSE={rmse_rate:.4g} (npu={float(npu_rmse):.4g}, sim={float(sim_rmse):.4g}, max {_RMSE_RATE}), '
        f'floor={floor:.4g}',
    )
