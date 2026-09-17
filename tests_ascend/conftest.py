# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""NPU session fixtures. Skip the suite when torch.npu is unavailable."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest
import torch

try:
    import torch_npu as _torch_npu
except ImportError:
    _torch_npu = None

_CACHE_DIR = Path(__file__).resolve().parent / '.triton_cache'
os.environ.setdefault('TRITON_ALL_BLOCKS_PARALLEL', '1')
os.environ.setdefault('TRITON_CACHE_DIR', str(_CACHE_DIR))


def _npu_is_available() -> bool:
    return _torch_npu is not None and bool(torch.npu.is_available())


def _install_npu_cxx_wrapper() -> None:
    """Forward-declare aclOpExecutor so triton-ascend can compile npu_utils.

    CANN 9.1.0 headers used by this workspace do not declare that type before
    torch_npu 2.7.1.post1 includes it. triton-ascend honors ``CC``.
    """
    compiler = os.environ.get('CC') or shutil.which('g++')
    if compiler is None:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stub = _CACHE_DIR / 'acl_op_executor_stub.h'
    stub.write_text('typedef struct aclOpExecutor aclOpExecutor;\n', encoding='utf-8')
    wrapper = _CACHE_DIR / 'gxx_wrapper.sh'
    wrapper.write_text(
        f'#!/bin/sh\nexec {compiler} -include "{stub}" "$@"\n',
        encoding='utf-8',
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.environ['CC'] = str(wrapper)


if _npu_is_available():
    _install_npu_cxx_wrapper()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if _npu_is_available():
        return
    skip = pytest.mark.skip(reason='NPU is not available')
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope='session', autouse=True)
def _npu_device() -> None:
    if _npu_is_available():
        torch.npu.set_device(0)
