"""SYRE construction- and ``add_param_group``-time validation.

These tests exercise the triton-free config layer: type/value rejection,
``d_bound`` auto-resolution, the AR-meaningfulness fp32 floor, the
Theorem-3 soft warning, and the back-compat ``syre_wd=False`` invariant.
None of them need a GPU or triton to run.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from dion.aurora import Aurora, SyreTheorem3Warning


CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
HAS_TRITON_GPU = CUDA_AVAILABLE and TRITON_AVAILABLE


# ---------------------------------------------------------------------------
# Per-kwarg type / value rejection.
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


# ---------------------------------------------------------------------------
# d_bound auto-resolution + AR-meaningfulness floor.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Theorem-3 soft warning.
# ---------------------------------------------------------------------------


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
# Back-compat: syre_wd=False is bit-equal to no-SYRE-kwargs.
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
# Triton gating + num_heads.
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
