# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial

import pytest
import torch

from megablocks.layers.arguments import Arguments
from megablocks.layers.dmoe import dMoE
from megablocks.layers.moe import clear_load_balancing_loss
from tests_ascend.precision import assert_close_to_sim
from tests_ascend.reference import default_activation, dmoe_tokens

pytestmark = pytest.mark.npu

# (bs, sl, hs, num_experts, top_k) from tests/layers/dmoe_test.py.
_SHAPES = (
    (16, 1024, 512, 1, 1),
    (16, 1024, 512, 2, 1),
    (16, 1024, 512, 4, 1),
    (16, 1024, 512, 8, 1),
    (8, 2048, 512, 1, 1),
    (8, 2048, 512, 2, 1),
    (8, 2048, 512, 4, 1),
    (16, 1024, 512, 2, 2),
    (16, 1024, 512, 4, 2),
    (16, 1024, 512, 4, 4),
    (16, 1024, 512, 8, 2),
    (16, 1024, 512, 8, 4),
    (16, 1024, 512, 8, 8),
    (16, 1024, 128, 1, 1),
)
_DTYPES = (torch.bfloat16, torch.float16)
_MLP_TYPES = ('mlp', 'glu')


def _args(hidden: int, experts: int, top_k: int, mlp_type: str, dtype: torch.dtype) -> Arguments:
    return Arguments(
        hidden_size=hidden,
        ffn_hidden_size=hidden * 2,
        moe_num_experts=experts,
        moe_top_k=top_k,
        init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.1),
        memory_optimized_mlp=False,
        mlp_type=mlp_type,
        mlp_impl='grouped',
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        device=torch.device('npu'),
        bias=False,
        return_bias=False,
        moe_loss_weight=0.0,
    )


def _param_views(
    layer: dMoE,
    experts: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    w1 = layer.experts.mlp.w1.detach().view(experts, -1, hidden)
    w2 = layer.experts.mlp.w2.detach().view(experts, -1, hidden)
    router = layer.router.layer.weight.detach()
    v1 = getattr(layer.experts.mlp, 'v1', None)
    v1_view = None if v1 is None else v1.detach().view(experts, -1, hidden)
    return router, w1, w2, v1_view


def _reference(
    tokens: torch.Tensor,
    router: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    v1: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    tokens_p = tokens.detach().clone().requires_grad_(True)
    router_p = router.detach().clone().requires_grad_(True)
    w1_p = w1.detach().clone().requires_grad_(True)
    w2_p = w2.detach().clone().requires_grad_(True)
    v1_p = None if v1 is None else v1.detach().clone().requires_grad_(True)
    out = dmoe_tokens(tokens_p, router_p, w1_p, w2_p, top_k, default_activation, v1_p)
    out.sum().backward()
    assert tokens_p.grad is not None
    assert router_p.grad is not None
    assert w1_p.grad is not None
    assert w2_p.grad is not None
    v1_grad = None if v1_p is None else v1_p.grad
    return out.detach(), tokens_p.grad, w1_p.grad, w2_p.grad, router_p.grad, v1_grad


def _to_cpu(
    router: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    v1: torch.Tensor | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    v1_c = None if v1 is None else v1.cpu().to(dtype)
    return router.cpu().to(dtype), w1.cpu().to(dtype), w2.cpu().to(dtype), v1_c


def _assert_grad(name: str, golden: torch.Tensor, sim: torch.Tensor, actual: torch.Tensor | None) -> None:
    assert actual is not None
    assert_close_to_sim(golden.float(), sim.float(), actual.cpu(), what=name)


def _compare_dmoe(
    layer: dMoE,
    x: torch.Tensor,
    out: torch.Tensor,
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
) -> None:
    bs, sl, hs, num_experts, top_k = shape
    router, w1, w2, v1 = _param_views(layer, num_experts, hs)
    tokens = x.detach().reshape(-1, hs)
    g_r, g_w1, g_w2, g_v1 = _to_cpu(router, w1, w2, v1, torch.float32)
    s_r, s_w1, s_w2, s_v1 = _to_cpu(router, w1, w2, v1, dtype)
    g_out, g_x, g_dw1, g_dw2, g_dr, g_dv1 = _reference(tokens.cpu().float(), g_r, g_w1, g_w2, top_k, g_v1)
    s_out, s_x, s_dw1, s_dw2, s_dr, s_dv1 = _reference(tokens.cpu().to(dtype), s_r, s_w1, s_w2, top_k, s_v1)
    assert_close_to_sim(
        g_out.view(sl, bs, hs),
        s_out.view(sl, bs, hs).to(dtype),
        out.detach().cpu(),
        what='forward',
    )
    x_grad = None if x.grad is None else x.grad.reshape(-1, hs)
    w1_grad = None if layer.experts.mlp.w1.grad is None else layer.experts.mlp.w1.grad.view_as(w1)
    w2_grad = None if layer.experts.mlp.w2.grad is None else layer.experts.mlp.w2.grad.view_as(w2)
    _assert_grad('x.grad', g_x, s_x, x_grad)
    _assert_grad('w1.grad', g_dw1, s_dw1, w1_grad)
    _assert_grad('w2.grad', g_dw2, s_dw2, w2_grad)
    _assert_grad('router.grad', g_dr, s_dr, layer.router.layer.weight.grad)
    if v1 is not None:
        assert g_dv1 is not None and s_dv1 is not None
        v1_grad = None if layer.experts.mlp.v1.grad is None else layer.experts.mlp.v1.grad.view_as(v1)
        _assert_grad('v1.grad', g_dv1, s_dv1, v1_grad)


@pytest.mark.parametrize(('bs', 'sl', 'hs', 'num_experts', 'top_k'), _SHAPES)
@pytest.mark.parametrize('dtype', _DTYPES)
@pytest.mark.parametrize('mlp_type', _MLP_TYPES)
def test_dmoe_forward_backward(
    bs: int,
    sl: int,
    hs: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    mlp_type: str,
) -> None:
    torch.manual_seed(0)
    layer = dMoE(_args(hs, num_experts, top_k, mlp_type, dtype)).to(dtype)
    x = torch.randn(sl, bs, hs, dtype=dtype, device='npu', requires_grad=True)
    out = layer(x)
    assert isinstance(out, torch.Tensor) and out.shape == x.shape
    out.sum().backward()
    _compare_dmoe(layer, x, out, (bs, sl, hs, num_experts, top_k), dtype)
    layer.zero_grad(set_to_none=True)
    clear_load_balancing_loss()
