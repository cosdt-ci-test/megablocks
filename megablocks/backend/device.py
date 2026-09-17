# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Device detection and autocast helpers for CUDA and NPU."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal, TypeVar, cast

import torch

Backend = Literal['cuda', 'npu']

_FLOATING_DEVICES = ('cuda', 'npu')

F = TypeVar('F', bound=Callable[..., object])


def npu_is_available() -> bool:
    npu = getattr(torch, 'npu', None)
    return npu is not None and bool(npu.is_available())


def default_device() -> int | torch.device:
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    if npu_is_available():
        return torch.device(f'npu:{torch.npu.current_device()}')
    return torch.device('cpu')


def autocast_device_type() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if npu_is_available():
        return 'npu'
    return 'cpu'


def backend_of(tensor: torch.Tensor) -> Backend:
    device_type = tensor.device.type
    if device_type == 'cuda':
        return 'cuda'
    if device_type == 'npu':
        return 'npu'
    raise ValueError(f'Unsupported device type for megablocks dispatch: {device_type}')


def _is_eligible(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() and tensor.device.type in _FLOATING_DEVICES and tensor.dtype is not torch.float64


def _cast(value: object, dtype: torch.dtype) -> object:
    if isinstance(value, torch.Tensor) and _is_eligible(value):
        return value.to(dtype=dtype)
    if isinstance(value, dict):
        return {_cast(key, dtype): _cast(item, dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cast(item, dtype) for item in value)
    return value


def custom_fwd(fwd: F) -> F:
    """Cast floating inputs to autocast dtype, then run the body without autocast."""

    @functools.wraps(fwd)
    def decorate_fwd(*args: object, **kwargs: object) -> object:
        device_type = autocast_device_type()
        if not torch.is_autocast_enabled(device_type):
            return fwd(*args, **kwargs)
        dtype = torch.get_autocast_dtype(device_type)
        with torch.autocast(device_type=device_type, enabled=False):
            return fwd(
                *cast(tuple[object, ...], _cast(args, dtype)),
                **cast(dict[str, object], _cast(kwargs, dtype)),
            )

    return cast(F, decorate_fwd)


def custom_bwd(bwd: F) -> F:
    """Disable autocast for custom backward, matching stanford-stk 0.7.1."""

    @functools.wraps(bwd)
    def decorate_bwd(*args: object, **kwargs: object) -> object:
        device_type = autocast_device_type()
        with torch.autocast(device_type=device_type, enabled=False):
            return bwd(*args, **kwargs)

    return cast(F, decorate_bwd)
