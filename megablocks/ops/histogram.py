# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

from megablocks.backend import dispatch


# Autograd wrapper for histogram kernel.
# NOTE: Does not support gradients.
class HistogramOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, max_val: int):
        return dispatch.histogram(x, max_val)


histogram = HistogramOp.apply
