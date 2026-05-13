"""SYRE weight decay tests for Aurora.

Coverage:

* Validation: bad types / values raise at construction and at
  ``add_param_group``. ``syre_wd=True`` requires an explicit
  ``syre_std`` (no empirical default).
* Back-compat: ``syre_wd=False`` with all the new kwargs at defaults
  is bit-equal to construction without those kwargs at all.
* Triton gating: when triton can't be imported, ``syre_wd=True``
  raises ``ImportError`` at construction.
* num_heads + syre_wd raises NotImplementedError.
* Kernel-level (CUDA + triton): theta_0 extraction with gamma=1,
  determinism, offset_base shift, AR collapse at d_bound=0, zero-gamma
  noop, cautious math vs. hand-rolled reference, cautious all-true /
  all-false reductions.
* End-to-end through Aurora.step (CUDA + triton): zero-grad SYRE pulls
  toward theta_0, state_dict round-trips seeds, advanced_removal
  initializes a nonzero seed2.
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


# ---------------------------------------------------------------------------
# 1. Validation (CPU; no triton needed for the rejection paths).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 1, "true", None, 1.0, [], object()])
def test_invalid_syre_wd_raises(bad):
    """``syre_wd`` must be a real bool. ``int`` is a strict reject."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(TypeError, match="syre_wd"):
        Aurora([p], syre_wd=bad)


@pytest.mark.parametrize("bad", [0, 1, "yes", None, 1.0])
def test_invalid_advanced_removal_raises(bad):
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(TypeError, match="advanced_removal"):
        Aurora([p], advanced_removal=bad)


def test_invalid_d_bound_above_one():
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="d_bound"):
        Aurora([p], d_bound=1.0)
    with pytest.raises(ValueError, match="d_bound"):
        Aurora([p], d_bound=2.0)


def test_invalid_d_bound_negative():
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="d_bound"):
        Aurora([p], d_bound=-0.1)


@pytest.mark.parametrize("bad", ["nope", [], object()])
def test_invalid_d_bound_type(bad):
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(TypeError, match="d_bound"):
        Aurora([p], d_bound=bad)


def test_syre_wd_true_requires_syre_std():
    """No empirical default: ``syre_wd=True`` without ``syre_std`` is a
    hard error. If the user wants the param's own init magnitude they
    must compute it explicitly.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="syre_std"):
        Aurora([p], syre_wd=True)


@pytest.mark.parametrize("bad", ["nope", [], object(), True])
def test_invalid_syre_std_type(bad):
    """``syre_std=True`` would silently coerce to 1.0 -- explicitly reject."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(TypeError, match="syre_std"):
        Aurora([p], syre_wd=True, syre_std=bad)


def test_invalid_syre_std_nonpositive():
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="syre_std"):
        Aurora([p], syre_wd=True, syre_std=0.0)
    with pytest.raises(ValueError, match="syre_std"):
        Aurora([p], syre_wd=True, syre_std=-0.5)


def test_syre_defaults():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Aurora([p])
    g = opt.param_groups[0]
    assert g["syre_wd"] is False
    assert g["syre_std"] is None
    assert g["advanced_removal"] is False
    assert g["d_bound"] == 0.01


def test_add_param_group_validates_syre_kwargs():
    """SYRE kwargs added via ``add_param_group`` must be validated the
    same way as those passed to ``__init__``.
    """
    p1 = torch.nn.Parameter(torch.randn(8, 4))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    opt = Aurora([p1])
    with pytest.raises(TypeError, match="syre_wd"):
        opt.add_param_group({"params": [p2], "syre_wd": "yes"})
    with pytest.raises(ValueError, match="syre_std"):
        opt.add_param_group({"params": [p2], "syre_wd": True})


# ---------------------------------------------------------------------------
# 2. Back-compat: syre_wd=False is bit-equal to no-SYRE-kwargs.
# ---------------------------------------------------------------------------


def test_syre_wd_false_bit_equal_to_baseline():
    """Explicit ``syre_wd=False`` plus every other SYRE kwarg at its
    default must produce a trajectory identical to a construction that
    omits the SYRE kwargs entirely.
    """
    torch.manual_seed(0)
    p_a = torch.nn.Parameter(torch.randn(16, 8))
    p_b = torch.nn.Parameter(p_a.detach().clone())
    opt_a = Aurora([p_a], lr=0.05, weight_decay=0.01)
    opt_b = Aurora(
        [p_b], lr=0.05, weight_decay=0.01,
        syre_wd=False, syre_std=None,
        advanced_removal=False, d_bound=0.01,
    )
    for _ in range(3):
        g = torch.randn(16, 8) * 0.1
        p_a.grad = g.clone()
        p_b.grad = g.clone()
        opt_a.step()
        opt_b.step()
    assert torch.equal(p_a.detach(), p_b.detach())


# ---------------------------------------------------------------------------
# 3. Triton gating.
# ---------------------------------------------------------------------------


def test_syre_wd_requires_triton_at_construction(monkeypatch):
    """When ``dion.syre`` can't be imported, ``syre_wd=True`` must raise
    ``ImportError`` at construction (not silently fall back).

    Trick: assigning ``sys.modules["dion.syre"] = None`` makes Python
    treat the module as deliberately missing and raise ``ImportError``
    on the next ``import dion.syre``. The ``monkeypatch`` fixture
    restores the original entry on test teardown.
    """
    import sys

    monkeypatch.setitem(sys.modules, "dion.syre", None)

    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ImportError, match="syre_wd=True requires"):
        Aurora([p], syre_wd=True, syre_std=0.02)


# ---------------------------------------------------------------------------
# 4. num_heads + syre_wd is not supported.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="needs triton to construct")
def test_num_heads_with_syre_wd_raises():
    """num_heads + syre_wd is intentionally rejected on v1: the per-
    param SYRE state would need to follow the head-split mapping.
    """
    if not HAS_TRITON_GPU:
        pytest.skip("step() needs CUDA")
    # 8 rows = 4 heads * 2 head_dim.
    p = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.1)
    opt = Aurora(
        [{"params": [p], "num_heads": 4}],
        lr=0.05, weight_decay=0.1,
        syre_wd=True, syre_std=0.02,
    )
    p.grad = torch.randn_like(p) * 0.1
    with pytest.raises(NotImplementedError, match="num_heads"):
        opt.step()


# ---------------------------------------------------------------------------
# 5. Kernel-level math correctness (CUDA + triton).
# ---------------------------------------------------------------------------


@gpu_only
def test_syre_extracts_theta_0_with_gamma_1():
    """With ``gamma = 1`` and the trivial choice ``theta = 0``, the
    SYRE step ``theta - (theta - theta_0)`` collapses to ``theta_0``.
    Then check linearity: a second SYRE call with ``gamma = 0.3`` on a
    fresh ``theta`` matches ``theta - 0.3 * (theta - theta_0)``.
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(123)
    n = 1024
    theta = torch.zeros(n, device="cuda")
    syre_wd_inplace(
        theta, gamma=1.0, seed1=42, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    theta_0 = theta.clone()

    theta_in = torch.randn(n, device="cuda") * 0.5
    expected = theta_in - 0.3 * (theta_in - theta_0)
    theta_out = theta_in.clone()
    syre_wd_inplace(
        theta_out, gamma=0.3, seed1=42, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    torch.testing.assert_close(theta_out, expected, atol=1e-5, rtol=1e-5)


@gpu_only
def test_syre_determinism_same_seed():
    """Same ``seed1`` + ``offset_base`` -> same ``theta_0`` sequence."""
    from dion.syre import syre_wd_inplace

    theta_a = torch.randn(512, device="cuda")
    theta_b = theta_a.clone()
    syre_wd_inplace(
        theta_a, gamma=1.0, seed1=99, std=0.05,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    syre_wd_inplace(
        theta_b, gamma=1.0, seed1=99, std=0.05,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    torch.testing.assert_close(theta_a, theta_b, atol=0, rtol=0)


@gpu_only
def test_syre_different_seeds_differ():
    from dion.syre import syre_wd_inplace

    theta_a = torch.randn(512, device="cuda")
    theta_b = theta_a.clone()
    syre_wd_inplace(
        theta_a, gamma=1.0, seed1=1, std=0.05,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    syre_wd_inplace(
        theta_b, gamma=1.0, seed1=2, std=0.05,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    assert not torch.allclose(theta_a, theta_b, atol=1e-4)


@gpu_only
def test_syre_offset_base_shifts_theta_0():
    """Different ``offset_base`` -> different ``theta_0`` slice (so each
    shard sees unique randoms in the sharded case).
    """
    from dion.syre import syre_wd_inplace

    theta_a = torch.randn(256, device="cuda")
    theta_b = theta_a.clone()
    syre_wd_inplace(
        theta_a, gamma=1.0, seed1=5, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    syre_wd_inplace(
        theta_b, gamma=1.0, seed1=5, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=1024,
    )
    assert not torch.allclose(theta_a, theta_b, atol=1e-4)


@gpu_only
def test_syre_zero_gamma_is_noop():
    from dion.syre import syre_wd_inplace

    theta = torch.randn(128, device="cuda")
    theta0 = theta.clone()
    syre_wd_inplace(
        theta, gamma=0.0, seed1=1, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    assert torch.equal(theta, theta0)


@gpu_only
def test_syre_advanced_removal_collapses_at_d_bound_zero():
    """SYRE-AR with ``d_bound=0`` -> ``d ~ Uniform(1, 1) == 1``, so the
    AR multiplier is 1 and the result equals basic SYRE.
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(13)
    theta_basic = torch.randn(512, device="cuda")
    theta_ar = theta_basic.clone()
    syre_wd_inplace(
        theta_basic, gamma=0.4, seed1=17, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    syre_wd_inplace(
        theta_ar, gamma=0.4, seed1=17, std=0.1,
        seed2=99, d_bound=0.0, advanced_removal=True, offset_base=0,
    )
    torch.testing.assert_close(theta_basic, theta_ar, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 6. Cautious-SYRE math.
# ---------------------------------------------------------------------------


@gpu_only
def test_cautious_syre_matches_hand_rolled_reference():
    """Compute ``theta_0`` by running plain SYRE with ``gamma=1`` from
    ``theta=0``, then hand-roll the cautious-SYRE update in PyTorch and
    compare to the kernel's output.
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(2026)
    n = 1024
    theta = torch.randn(n, device="cuda") * 0.5
    theta_orig = theta.clone()
    U = torch.randn(n, device="cuda")

    seed1 = 17
    std = 0.1
    gamma = 0.3

    theta_zero = torch.zeros(n, device="cuda")
    syre_wd_inplace(
        theta_zero, gamma=1.0, seed1=seed1, std=std,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    theta_0 = theta_zero

    mask = (U * theta_orig >= 0).to(theta_orig.dtype)
    diff = theta_orig - theta_0
    expected = theta_orig - gamma * diff * mask

    theta_kernel = theta_orig.clone()
    syre_wd_inplace(
        theta_kernel, gamma=gamma, seed1=seed1, std=std,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
        U=U,
    )
    torch.testing.assert_close(theta_kernel, expected, atol=1e-5, rtol=1e-5)


@gpu_only
def test_cautious_syre_all_true_mask_matches_basic():
    """``U = theta`` -> ``U * theta = theta**2 >= 0`` everywhere, so
    cautious-SYRE reduces to basic SYRE (to fp32 ULP).
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(11)
    n = 512
    theta = torch.randn(n, device="cuda") * 0.5
    theta_basic = theta.clone()
    theta_cautious = theta.clone()
    U = theta.clone()
    syre_wd_inplace(
        theta_basic, gamma=0.4, seed1=99, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
    )
    syre_wd_inplace(
        theta_cautious, gamma=0.4, seed1=99, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
        U=U,
    )
    torch.testing.assert_close(theta_cautious, theta_basic, atol=1e-7, rtol=1e-6)


@gpu_only
def test_cautious_syre_all_false_mask_is_noop():
    """``U = -theta`` (and theta strictly nonzero) -> mask is 0
    everywhere, so the cautious kernel leaves theta untouched.
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(13)
    n = 512
    theta = (torch.randn(n, device="cuda") * 0.5).abs() + 0.01
    theta = torch.where(
        torch.rand(n, device="cuda") < 0.5, -theta, theta
    )
    theta_orig = theta.clone()
    U = -theta.clone()
    syre_wd_inplace(
        theta, gamma=0.4, seed1=99, std=0.1,
        seed2=0, d_bound=0.0, advanced_removal=False, offset_base=0,
        U=U,
    )
    torch.testing.assert_close(theta, theta_orig, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 7. End-to-end through Aurora.step.
# ---------------------------------------------------------------------------


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
