# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch

from megablocks.layers.arguments import Arguments


def dtype(args: Arguments):
    if args.fp16:
        return torch.float16
    elif args.bf16:
        return torch.bfloat16
    return None


def cast_if_autocast_enabled(tensor):
    device_type = tensor.device.type
    if torch.is_autocast_enabled(device_type):
        return tensor.to(dtype=torch.get_autocast_dtype(device_type))
    return tensor
