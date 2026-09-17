# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

try:
    import stk
except ImportError:
    stk = None
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from stk import Matrix


@torch.jit.script
def _gelu_backward_inplace(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = (0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out))
    return g.mul_(ff)


def gelu_backward_(grad: 'Matrix', x: 'Matrix'):
    # NOTE: The two sparse matrices must have the same topology.
    if stk is not None and isinstance(grad, stk.Matrix) and isinstance(x, stk.Matrix):
        return stk.Matrix(
            x.size(),
            _gelu_backward_inplace(grad.data, x.data),
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t,
        )
    return _gelu_backward_inplace(grad, x)


def gelu(x: 'Matrix'):
    if stk is None:
        raise ImportError('stanford-stk is required for mlp_impl="sparse"')
    assert isinstance(x, stk.Matrix)
    return stk.Matrix(
        x.size(),
        F.gelu(x.data, approximate='tanh'),
        x.row_indices,
        x.column_indices,
        x.offsets,
        x.column_indices_t,
        x.offsets_t,
        x.block_offsets_t,
    )
