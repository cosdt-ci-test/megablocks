# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Grouped GEMM facade. CUDA uses the grouped_gemm package; NPU uses torch_npu."""

from megablocks.backend import dispatch

ops = dispatch.ops
backend = dispatch.backend


def grouped_gemm_is_available():
    return dispatch.grouped_gemm_is_available()


def assert_grouped_gemm_is_available(device=None):
    dispatch.assert_grouped_gemm_is_available(device)
