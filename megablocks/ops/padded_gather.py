# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from megablocks.backend import dispatch
from megablocks.backend.device import custom_bwd, custom_fwd


# Autograd wrapper for padded_gather kernel.
class PaddedGatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        ctx.save_for_backward(indices, bin_ids, bins, padded_bins)
        ctx.top_k = top_k
        return dispatch.padded_gather(
            x,
            indices,
            bin_ids,
            None,
            bins,
            padded_bins,
            top_k,
        )

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()

        indices, bin_ids, bins, padded_bins = ctx.saved_tensors
        out = dispatch.padded_scatter(
            grad,
            indices,
            bin_ids,
            None,
            bins,
            padded_bins,
            ctx.top_k,
        )
        return out, None, None, None, None, None


padded_gather = PaddedGatherOp.apply
