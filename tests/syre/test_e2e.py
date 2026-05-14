"""End-to-end SYRE through ``Aurora.step`` on an Aurora group.

Zero-grad e2e (SYRE pulls toward theta_0), state_dict round-trip for
SYRE seeds, lazy seed2 initialization with ``advanced_removal=True``,
and a hand-rolled cautious-SYRE e2e check that captures U via a
parallel ``weight_decay=0`` optimizer.

The matching tests for the AdamW group live in ``test_adamw.py``.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from dion.aurora import Aurora


CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
HAS_TRITON_GPU = CUDA_AVAILABLE and TRITON_AVAILABLE

gpu_only = pytest.mark.skipif(
    not HAS_TRITON_GPU, reason="SYRE requires CUDA + triton"
)


@gpu_only
def test_syre_end_to_end_decays_toward_theta_0():
    """With zero gradient the orth update is zero. After Aurora.step
    with ``syre_wd=True`` the param should equal
    ``p_init - gamma * (p_init - theta_0)``.
    """
    torch.manual_seed(31)
    p = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.1)
    p_init = p.detach().clone()

    # gamma = lr * wd = 0.05 * 0.2 = 0.01
    opt = Aurora(
        [p], lr=0.05, weight_decay=0.2,
        syre_wd=True, syre_std=0.05,
    )
    # Pin seeds so we can hand-compute theta_0 below. Also pre-initialize
    # the momentum buffer because ``_get_or_initialize_state`` lazy-inits
    # via an empty-dict check, and writing the seed keys makes the dict
    # non-empty.
    opt.state[p]["momentum"] = torch.zeros_like(p)
    opt.state[p]["syre_seed1"] = 42
    opt.state[p]["syre_seed2"] = 0

    p.grad = torch.zeros_like(p)
    opt.step()

    from dion.syre import syre_wd_inplace
    theta = p_init.clone().contiguous()
    syre_wd_inplace(
        theta, gamma=1.0, seed1=42, std=0.05,
        seed2=0, d_bound=0.01, advanced_removal=False, offset_base=0,
    )
    theta_0 = theta

    expected = p_init - 0.01 * (p_init - theta_0)
    torch.testing.assert_close(p.detach(), expected, atol=1e-5, rtol=1e-5)


@gpu_only
def test_syre_state_dict_round_trip():
    """SYRE seeds in ``state[p]`` must round-trip through state_dict /
    load_state_dict. (``syre_std`` lives on the param group, so it round-
    trips via the standard param_group serialization.)
    """
    p = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.1)
    opt = Aurora(
        [p], lr=0.05, weight_decay=0.05,
        syre_wd=True, syre_std=0.04,
    )
    p.grad = torch.randn(8, 4, device="cuda") * 0.1
    opt.step()

    seed1 = opt.state[p]["syre_seed1"]
    seed2 = opt.state[p]["syre_seed2"]
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = Aurora(
        [p2], lr=0.05, weight_decay=0.05,
        syre_wd=True, syre_std=0.04,
    )
    opt2.load_state_dict(sd)
    assert opt2.state[p2]["syre_seed1"] == seed1
    assert opt2.state[p2]["syre_seed2"] == seed2
    assert opt2.param_groups[0]["syre_std"] == 0.04


@gpu_only
def test_syre_advanced_removal_initializes_nonzero_seed2():
    """``advanced_removal=True`` must lazy-draw a nonzero ``seed2``.
    With ``advanced_removal=False`` (default), ``seed2`` stays at 0
    (it's ignored by the kernel anyway).
    """
    p_basic = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.1)
    opt_basic = Aurora(
        [p_basic], lr=0.05, weight_decay=0.05,
        syre_wd=True, syre_std=0.04,
    )
    p_basic.grad = torch.randn_like(p_basic) * 0.1
    opt_basic.step()
    assert opt_basic.state[p_basic]["syre_seed2"] == 0

    p_ar = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.1)
    opt_ar = Aurora(
        [p_ar], lr=0.05, weight_decay=0.05,
        syre_wd=True, syre_std=0.04,
        advanced_removal=True,
    )
    p_ar.grad = torch.randn_like(p_ar) * 0.1
    opt_ar.step()
    # Drawn from ``torch.randint(0, 2**31, (1,))``; collision with 0 is
    # astronomically unlikely.
    assert opt_ar.state[p_ar]["syre_seed2"] != 0


@gpu_only
def test_syre_cautious_end_to_end_matches_hand_rolled():
    """End-to-end Aurora step with ``cautious_wd=True`` and
    ``syre_wd=True``. Capture the post-orth update ``U`` by running a
    parallel optimizer with ``weight_decay=0`` (so no decay fires), then
    hand-compute the cautious-SYRE step against ``U``.
    """
    torch.manual_seed(55)
    p_init = torch.randn(8, 4, device="cuda") * 0.1
    grad = torch.randn(8, 4, device="cuda") * 0.1

    p_u = torch.nn.Parameter(p_init.clone())
    p_u.grad = grad.clone()
    opt_u = Aurora([p_u], lr=0.05, weight_decay=0.0)
    opt_u.step()
    lr = 0.05
    # The standard post-orth does ``X -= adjusted_lr * U``. ``adjust_lr=
    # "spectral_norm"`` for an 8x4 (tall) matrix scales by sqrt(8/4) =
    # sqrt(2), so ``adjusted_lr = lr * sqrt(2)``. Recover U accordingly.
    adjusted_lr = lr * (8.0 / 4.0) ** 0.5
    U = (p_init - p_u.detach()) / adjusted_lr

    p_c = torch.nn.Parameter(p_init.clone())
    p_c.grad = grad.clone()
    opt_c = Aurora(
        [p_c], lr=lr, weight_decay=0.3,
        cautious_wd=True, syre_wd=True, syre_std=0.05,
    )
    opt_c.state[p_c]["momentum"] = torch.zeros_like(p_c)
    opt_c.state[p_c]["syre_seed1"] = 1234
    opt_c.state[p_c]["syre_seed2"] = 0
    opt_c.step()

    from dion.syre import syre_wd_inplace
    theta_for_t0 = p_init.clone().contiguous()
    syre_wd_inplace(
        theta_for_t0, gamma=1.0, seed1=1234, std=0.05,
        seed2=0, d_bound=0.01, advanced_removal=False, offset_base=0,
    )
    theta_0 = theta_for_t0

    gamma = lr * 0.3
    mask = (U * p_init >= 0).to(p_init.dtype)
    p_after_syre = p_init - gamma * (p_init - theta_0) * mask
    expected = p_after_syre - adjusted_lr * U

    torch.testing.assert_close(p_c.detach(), expected, atol=2e-5, rtol=2e-5)
