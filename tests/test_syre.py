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

from dion.aurora import Aurora, SyreTheorem3Warning


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
    # ``d_bound`` defaults to ``None`` (auto-resolves to ``0.1*syre_std``
    # when AR+SYRE are both on). AR is off here, so it stays ``None``.
    assert g["d_bound"] is None


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


def test_d_bound_autoresolves_to_one_tenth_syre_std():
    """When ``advanced_removal=True`` and ``d_bound`` is left at its
    default (``None``), Aurora should resolve it to ``0.1 * syre_std``
    so ``sigma_D / sigma_0 ~ 0.058`` -- comfortably in the
    ``sigma_D = o(sigma_0)`` regime that Theorem 3 of the SYRE paper
    (arXiv:2408.15495) requires.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Aurora(
        [p], syre_wd=True, syre_std=0.02, advanced_removal=True,
    )
    g = opt.param_groups[0]
    assert g["d_bound"] == pytest.approx(0.002)
    # The defaults dict should also reflect the resolved value so groups
    # added later inherit it.
    assert opt.defaults["d_bound"] == pytest.approx(0.002)


def test_d_bound_explicit_overrides_autoresolve():
    """Passing an explicit ``d_bound`` must be honored verbatim even
    when AR+SYRE are on (no auto-resolution).
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Aurora(
        [p], syre_wd=True, syre_std=0.02, advanced_removal=True,
        d_bound=0.005,
    )
    assert opt.param_groups[0]["d_bound"] == 0.005


def test_d_bound_zero_with_advanced_removal_raises():
    """``d_bound=0`` + ``advanced_removal=True`` collapses ``D`` to the
    identity and defeats AR's purpose entirely -- reject as operator
    error rather than silently no-op'ing.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="d_bound=0"):
        Aurora(
            [p], syre_wd=True, syre_std=0.02,
            advanced_removal=True, d_bound=0.0,
        )


def test_d_bound_zero_with_advanced_removal_off_is_allowed():
    """``d_bound=0`` is only an error when *combined* with
    ``advanced_removal=True``. With AR off, ``d_bound`` is unused, so
    any in-range value (including 0) is fine.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Aurora([p], advanced_removal=False, d_bound=0.0)
    assert opt.param_groups[0]["d_bound"] == 0.0


def test_add_param_group_d_bound_autoresolves():
    """Same auto-resolution must happen for groups added via
    ``add_param_group``. The user passes ``d_bound=None`` (or omits it)
    and gets ``0.1 * syre_std`` back.
    """
    p1 = torch.nn.Parameter(torch.randn(8, 4))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    opt = Aurora([p1])  # construction with SYRE off
    opt.add_param_group({
        "params": [p2],
        "syre_wd": True, "syre_std": 0.05,
        "advanced_removal": True,
        # d_bound omitted -- inherits None from defaults, resolves here
    })
    assert opt.param_groups[1]["d_bound"] == pytest.approx(0.005)


def test_add_param_group_d_bound_zero_with_ar_raises():
    """The ``d_bound=0`` + AR rejection must fire at ``add_param_group``
    time too, not just at construction.
    """
    p1 = torch.nn.Parameter(torch.randn(8, 4))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    opt = Aurora([p1])
    with pytest.raises(ValueError, match="d_bound=0"):
        opt.add_param_group({
            "params": [p2],
            "syre_wd": True, "syre_std": 0.02,
            "advanced_removal": True, "d_bound": 0.0,
        })


def test_d_bound_below_fp32_floor_with_ar_raises():
    """An explicit positive but tiny ``d_bound`` (below ~1e-6) combined
    with ``advanced_removal=True`` is rejected: fp32 precision around
    1.0 (ulp ~1.2e-7) makes the Uniform(1-d_bound, 1+d_bound) multiplier
    collapse to ~1 and AR silently no-ops. The error message must point
    at the fp32 floor so the user knows why.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="AR-meaningfulness floor"):
        Aurora(
            [p], syre_wd=True, syre_std=0.01,
            advanced_removal=True, d_bound=1e-7,
        )


def test_d_bound_at_fp32_floor_with_ar_allowed():
    """The boundary case ``d_bound == 1e-6`` (the floor) is allowed --
    the check is strict ``<``. Sanity-guards against off-by-one error
    in the threshold.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Aurora(
        [p], syre_wd=True, syre_std=0.01,
        advanced_removal=True, d_bound=1e-6,
    )
    assert opt.param_groups[0]["d_bound"] == pytest.approx(1e-6)


def test_d_bound_below_fp32_floor_with_ar_off_is_allowed():
    """Tiny ``d_bound`` is only rejected when AR is on. With AR off
    the kernel ignores ``d_bound`` entirely so any nonnegative value
    (including tiny ones) is fine.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Aurora(
        [p], syre_wd=True, syre_std=0.01,
        advanced_removal=False, d_bound=1e-9,
    )
    assert opt.param_groups[0]["d_bound"] == pytest.approx(1e-9)


def test_syre_std_too_small_autoresolves_below_floor_and_raises():
    """The classic footgun: user picks a very small ``syre_std`` and
    lets ``d_bound`` auto-resolve. With ``syre_std=1e-6``, the resolved
    ``d_bound = 0.1 * syre_std = 1e-7`` is below the fp32 floor, so
    construction must raise (pointing at ``syre_std`` as the cause).
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="syre_std"):
        Aurora(
            [p], syre_wd=True, syre_std=1e-6,
            advanced_removal=True,
            # d_bound omitted -- auto-resolves to 1e-7, below floor
        )


def test_add_param_group_d_bound_below_fp32_floor_raises():
    """The fp32-floor rejection must fire at ``add_param_group`` time
    too, not just at construction.
    """
    p1 = torch.nn.Parameter(torch.randn(8, 4))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    opt = Aurora([p1])
    with pytest.raises(ValueError, match="AR-meaningfulness floor"):
        opt.add_param_group({
            "params": [p2],
            "syre_wd": True, "syre_std": 0.02,
            "advanced_removal": True, "d_bound": 1e-8,
        })


def test_d_bound_above_theorem3_warns():
    """``sigma_D >= sigma_0`` puts AR outside the regime where Theorem 3
    proves symmetry-removal strength. We warn (not raise) because users
    may want to explore this empirically.
    """
    p = torch.nn.Parameter(torch.randn(8, 4))
    # syre_std = 0.01 -> sigma_0 = 0.01.
    # d_bound = 0.05 -> sigma_D = 0.05 / sqrt(3) ~ 0.0289 > sigma_0.
    with pytest.warns(SyreTheorem3Warning, match="sigma_D"):
        Aurora(
            [p], syre_wd=True, syre_std=0.01,
            advanced_removal=True, d_bound=0.05,
        )


def test_d_bound_below_theorem3_does_not_warn():
    """The auto-resolved ``d_bound = 0.1 * syre_std`` and reasonable
    user-passed values must not trip the Theorem-3 warning.
    """
    import warnings as _warnings

    p1 = torch.nn.Parameter(torch.randn(8, 4))
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", SyreTheorem3Warning)
        # Auto-resolved: d_bound = 1e-3 -> sigma_D ~ 5.77e-4 << 1e-2 = sigma_0.
        Aurora([p1], syre_wd=True, syre_std=0.01, advanced_removal=True)

    p2 = torch.nn.Parameter(torch.randn(8, 4))
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", SyreTheorem3Warning)
        # Explicit but comfortably perturbative.
        Aurora(
            [p2], syre_wd=True, syre_std=0.01,
            advanced_removal=True, d_bound=5e-3,
        )


def test_d_bound_theorem3_warning_off_when_ar_disabled():
    """The Theorem-3 condition only matters when AR is on. With
    ``advanced_removal=False`` ``d_bound`` is unused by the kernel, so
    even a comically large value must not warn.
    """
    import warnings as _warnings

    p = torch.nn.Parameter(torch.randn(8, 4))
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", SyreTheorem3Warning)
        Aurora(
            [p], syre_wd=True, syre_std=0.01,
            advanced_removal=False, d_bound=0.5,
        )


def test_add_param_group_d_bound_theorem3_warns():
    """The Theorem-3 warning must fire at ``add_param_group`` time too,
    not just at construction.
    """
    p1 = torch.nn.Parameter(torch.randn(8, 4))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    opt = Aurora([p1])
    with pytest.warns(SyreTheorem3Warning, match="sigma_D"):
        opt.add_param_group({
            "params": [p2],
            "syre_wd": True, "syre_std": 0.01,
            "advanced_removal": True, "d_bound": 0.05,
        })


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
        advanced_removal=False, d_bound=None,
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


@gpu_only
def test_syre_advanced_removal_additive_form_preserves_distinctness():
    """The AR kernel uses the additive form ``diff + diff*xi`` rather
    than forming ``(1 + xi)`` in fp32. Under the additive form, every
    element's per-element xi retains its full fp32 precision near 0,
    so the per-element AR contributions are distinct across nearly all
    elements even at small ``d_bound``.

    Under the naive multiplicative form, ``round_fp32(1 + xi)`` only
    has ``~2*d_bound / ulp(1.0)`` ~ 167 distinct values at
    ``d_bound=1e-5`` -- and a layer with thousands of elements would
    have massive collisions, violating the paper's "all D_ii distinct"
    hypothesis at the implementation level.

    Setup: ``std=0`` makes ``theta_0 = 0`` and ``diff = theta``
    exactly, so the AR-only contribution simplifies analytically to
    ``-gamma * theta * xi``. We recover xi per element and count
    distinct values.
    """
    from dion.syre import syre_wd_inplace

    torch.manual_seed(0)
    n = 65536
    # Well-separated thetas so per-element xi resolves at fp32.
    theta_orig = torch.rand(n, device="cuda") + 0.5  # in [0.5, 1.5)

    theta_ar = theta_orig.clone()
    syre_wd_inplace(
        theta_ar, gamma=1.0, seed1=11, std=0.0,
        seed2=22, d_bound=1e-5, advanced_removal=True, offset_base=0,
    )
    # With gamma=1 and std=0:
    #   result_AR = theta - 1 * (theta + theta * xi) = -theta * xi
    # so xi_recovered = -result_AR / theta_orig.
    xi_recovered = -theta_ar / theta_orig

    # Sanity: recovered xi in [-d_bound, +d_bound] modulo a tiny
    # rounding margin.
    assert xi_recovered.abs().max().item() < 1.5e-5

    # Distinctness: under the additive form, ~all 65536 elements
    # should have distinct xi. The naive multiplicative form would
    # cap this at ~167 (= 2*d_bound / ulp(1.0)).
    n_distinct = int(xi_recovered.unique().numel())
    assert n_distinct > 10_000, (
        f"Expected nearly all {n} elements to have distinct xi under "
        f"additive form; got {n_distinct} distinct values. The naive "
        "multiplicative form would have produced ~167."
    )


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


# ---------------------------------------------------------------------------
# 8. SYRE on ``algorithm="adamw"`` groups (mirrors segyges/aurora wiring).
# ---------------------------------------------------------------------------


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
