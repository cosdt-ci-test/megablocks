# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from megablocks.backend import dispatch
from megablocks.backend.device import custom_bwd, custom_fwd


# Autograd wrapper for binned_gather kernel.
class BinnedGatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        bin_size: int,
        top_k: int,
    ):
        ctx.save_for_backward(indices, bins)
        ctx.top_k = top_k
        return dispatch.binned_gather(x, indices, None, bins, bin_size, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        indices, bins = ctx.saved_tensors
        out = dispatch.binned_scatter(grad, indices, None, bins, ctx.top_k)
        return out, None, None, None, None


binned_gather = BinnedGatherOp.apply
