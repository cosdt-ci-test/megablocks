# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

from megablocks.backend import dispatch

try:
    import megablocks_ops as ops  # type: ignore
    _ops_import_error = None
except ImportError as error:
    ops = None
    _ops_import_error = error


# Autograd wrappers for cumsum kernels.
# NOTE: Does not support gradients.
class ExclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int):
        if ops is None:
            raise ImportError('megablocks_ops is unavailable; build the c++ operations.') from _ops_import_error
        if len(x.size()) == 1:
            x = x.view([1, -1])
            out = torch.empty_like(x)
            ops.exclusive_cumsum(x, 1, out)
            return out.squeeze()
        out = torch.empty_like(x)
        ops.exclusive_cumsum(x, dim, out)
        return out


exclusive_cumsum = ExclusiveCumsumOp.apply


class InclusiveCumsumOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int) -> torch.Tensor:
        return dispatch.inclusive_cumsum(x, dim)


inclusive_cumsum = InclusiveCumsumOp.apply
