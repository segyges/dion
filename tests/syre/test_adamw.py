"""SYRE on ``algorithm="adamw"`` groups.

Lion + SYRE refusal, ``add_param_group`` triton gate for late-added
adamw groups, fused-AdamW back-compat when SYRE is off, zero-grad e2e
decay, state_dict round-trip, lazy seed2 init under AR, and a hand-
rolled cautious-SYRE e2e check that pins the mask source to the bias-
corrected update direction ``M_new / (sqrt(V_new/bc2) + eps)``.
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


def test_lion_with_syre_wd_raises():
    """SYRE is wired for aurora and adamw only; Lion + SYRE must refuse
    loudly rather than silently no-op.
    """
    p_aur = torch.nn.Parameter(torch.randn(8, 4))
    p_lion = torch.nn.Parameter(torch.randn(8))
    opt = Aurora([p_aur])
    with pytest.raises(NotImplementedError, match="lion"):
        opt.add_param_group({
            "params": [p_lion], "algorithm": "lion",
            "syre_wd": True, "syre_std": 0.05,
        })


def test_add_param_group_triton_gate(monkeypatch):
    """``add_param_group(..., syre_wd=True)`` must run the same triton
    import gate as ``__init__``; otherwise the construction-time check
    is trivially bypassed.
    """
    import sys

    p_aur = torch.nn.Parameter(torch.randn(8, 4))
    p_adamw = torch.nn.Parameter(torch.randn(8))
    opt = Aurora([p_aur])  # no SYRE -> no gate triggered

    monkeypatch.setitem(sys.modules, "dion.syre", None)
    with pytest.raises(ImportError, match="syre_wd=True requires"):
        opt.add_param_group({
            "params": [p_adamw], "algorithm": "adamw",
            "syre_wd": True, "syre_std": 0.05,
        })


def test_syre_adamw_back_compat():
    """An adamw group with ``syre_wd=False`` (plus every other SYRE kwarg
    at default) must produce a trajectory identical to one with no
    SYRE kwargs at all -- the standard fused AdamW path is preserved.
    """
    if not HAS_TRITON_GPU:
        # AdamW itself works on CPU via the foreach path; ``_fused_adamw_``
        # in the standard path needs CUDA. Skip on no-GPU.
        pytest.skip("AdamW fused path needs CUDA")

    torch.manual_seed(0)
    p_aur_a = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.1)
    p_aur_b = torch.nn.Parameter(p_aur_a.detach().clone())
    p_adamw_a = torch.nn.Parameter(torch.randn(8, device="cuda") * 0.1)
    p_adamw_b = torch.nn.Parameter(p_adamw_a.detach().clone())

    opt_a = Aurora([
        {"params": [p_aur_a]},
        {"params": [p_adamw_a], "algorithm": "adamw"},
    ], lr=0.05, weight_decay=0.01)
    opt_b = Aurora([
        {"params": [p_aur_b]},
        {"params": [p_adamw_b], "algorithm": "adamw",
         "syre_wd": False, "syre_std": None,
         "advanced_removal": False, "d_bound": None},
    ], lr=0.05, weight_decay=0.01,
       syre_wd=False, syre_std=None,
       advanced_removal=False, d_bound=None)

    for _ in range(3):
        g_aur = torch.randn(8, 4, device="cuda") * 0.1
        g_adamw = torch.randn(8, device="cuda") * 0.1
        p_aur_a.grad = g_aur.clone()
        p_aur_b.grad = g_aur.clone()
        p_adamw_a.grad = g_adamw.clone()
        p_adamw_b.grad = g_adamw.clone()
        opt_a.step()
        opt_b.step()

    assert torch.equal(p_aur_a.detach(), p_aur_b.detach())
    assert torch.equal(p_adamw_a.detach(), p_adamw_b.detach())


@gpu_only
def test_syre_adamw_end_to_end_decays_toward_theta_0():
    """With zero gradient on an adamw group + SYRE: the update direction
    ``M_new/denom`` is zero (since M starts at 0 and grad is 0), so the
    gradient step is a no-op and SYRE reduces to
    ``p <- p - lr*wd*(p - theta_0)`` (mask is all-true when ``U=0``
    because ``0 * p >= 0`` everywhere).

    This test runs with ``cautious_wd=True`` to also exercise the
    cautious-SYRE kernel path; both should land at the same answer for
    zero-grad.
    """
    torch.manual_seed(67)
    p = torch.nn.Parameter(torch.randn(8, device="cuda") * 0.1)
    p_init = p.detach().clone()

    opt = Aurora([
        {"params": [p], "algorithm": "adamw"},
    ], lr=0.05, weight_decay=0.2,
       cautious_wd=True,
       syre_wd=True, syre_std=0.05)

    # Pre-init the AdamW state buffers (momentum + variance) so the
    # ``_get_or_initialize_state`` empty-dict check still triggers --
    # writing the seed keys first would otherwise skip the buffer init.
    opt.state[p]["momentum"] = torch.zeros_like(p)
    opt.state[p]["variance"] = torch.zeros_like(p)
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

    gamma = 0.05 * 0.2  # lr * wd
    expected = p_init - gamma * (p_init - theta_0)
    torch.testing.assert_close(p.detach(), expected, atol=1e-5, rtol=1e-5)


@gpu_only
def test_syre_adamw_state_dict_round_trip():
    """SYRE seeds drawn on an adamw group must round-trip through
    state_dict, same contract as the aurora-group case.
    """
    p = torch.nn.Parameter(torch.randn(8, device="cuda") * 0.1)
    opt = Aurora([
        {"params": [p], "algorithm": "adamw"},
    ], lr=0.05, weight_decay=0.05, syre_wd=True, syre_std=0.04)
    p.grad = torch.randn(8, device="cuda") * 0.1
    opt.step()

    seed1 = opt.state[p]["syre_seed1"]
    seed2 = opt.state[p]["syre_seed2"]
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = Aurora([
        {"params": [p2], "algorithm": "adamw"},
    ], lr=0.05, weight_decay=0.05, syre_wd=True, syre_std=0.04)
    opt2.load_state_dict(sd)
    assert opt2.state[p2]["syre_seed1"] == seed1
    assert opt2.state[p2]["syre_seed2"] == seed2
    assert opt2.param_groups[0]["syre_std"] == 0.04


@gpu_only
def test_syre_adamw_advanced_removal_initializes_nonzero_seed2():
    """``advanced_removal=True`` on an adamw group must also lazy-draw a
    nonzero ``seed2``. Mirrors the aurora-group test.
    """
    p_ar = torch.nn.Parameter(torch.randn(8, device="cuda") * 0.1)
    opt_ar = Aurora([
        {"params": [p_ar], "algorithm": "adamw"},
    ], lr=0.05, weight_decay=0.05,
       syre_wd=True, syre_std=0.04, advanced_removal=True)
    p_ar.grad = torch.randn_like(p_ar) * 0.1
    opt_ar.step()
    assert opt_ar.state[p_ar]["syre_seed2"] != 0


@gpu_only
def test_syre_adamw_cautious_matches_hand_rolled():
    """End-to-end Aurora step on an adamw group with ``cautious_wd=True``
    + ``syre_wd=True``. Hand-roll the full AdamW + cautious-SYRE math
    and compare. Tests the per-element cautious mask is built from the
    bias-corrected update direction ``M_new / (sqrt(V_new/bc2) + eps)``.
    """
    torch.manual_seed(91)
    n = 16
    p_init = torch.randn(n, device="cuda") * 0.1
    grad = torch.randn(n, device="cuda") * 0.3

    lr = 0.05
    wd = 0.2
    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-8
    syre_std = 0.05
    seed1 = 4242

    p = torch.nn.Parameter(p_init.clone())
    opt = Aurora([
        {"params": [p], "algorithm": "adamw"},
    ], lr=lr, weight_decay=wd, betas=(beta1, beta2), epsilon=eps,
       cautious_wd=True, syre_wd=True, syre_std=syre_std)
    opt.state[p]["momentum"] = torch.zeros_like(p)
    opt.state[p]["variance"] = torch.zeros_like(p)
    opt.state[p]["syre_seed1"] = seed1
    opt.state[p]["syre_seed2"] = 0

    p.grad = grad.clone()
    opt.step()

    # Hand-compute expected. Step 1 with M_init = V_init = 0:
    #   M_new = (1-beta1) * grad
    #   V_new = (1-beta2) * grad**2
    #   denom = sqrt(V_new) / sqrt(bc2) + eps
    #   update_dir = M_new / denom
    #   adj_lr = lr / bc1
    # SYRE WD replaces standard decoupled WD:
    #   mask = (update_dir * p_init >= 0)
    #   p <- p_init - lr*wd * (p_init - theta_0) * mask
    #   p <- p - adj_lr * update_dir
    M_new = (1 - beta1) * grad
    V_new = (1 - beta2) * grad.pow(2)
    bc1 = 1 - beta1
    bc2 = 1 - beta2
    denom = V_new.sqrt() / (bc2**0.5) + eps
    update_dir = M_new / denom
    adj_lr = lr / bc1

    from dion.syre import syre_wd_inplace
    theta_for_t0 = p_init.clone().contiguous()
    syre_wd_inplace(
        theta_for_t0, gamma=1.0, seed1=seed1, std=syre_std,
        seed2=0, d_bound=0.01, advanced_removal=False, offset_base=0,
    )
    theta_0 = theta_for_t0

    gamma = lr * wd
    mask = (update_dir * p_init >= 0).to(p_init.dtype)
    p_after_syre = p_init - gamma * (p_init - theta_0) * mask
    expected = p_after_syre - adj_lr * update_dir

    torch.testing.assert_close(p.detach(), expected, atol=2e-5, rtol=2e-5)
