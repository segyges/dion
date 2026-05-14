"""Kernel-level math for ``dion.syre`` (CUDA + triton).

Single-tensor :func:`dion.syre.syre_wd_inplace` math: theta_0 extraction
with gamma=1, determinism, offset_base shift, AR-collapses-at-d_bound=0,
zero-gamma noop, and the additive-form distinctness property that
defines AR's pigeonhole avoidance.

Multi-tensor :func:`dion.syre.syre_wd_multi_inplace` parity vs. the
single-tensor loop, plus empty-list / zero-gamma / mixed-empty noop
behavior and dtype validation.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch


CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
HAS_TRITON_GPU = CUDA_AVAILABLE and TRITON_AVAILABLE

gpu_only = pytest.mark.skipif(
    not HAS_TRITON_GPU, reason="SYRE requires CUDA + triton"
)


# ---------------------------------------------------------------------------
# Single-tensor kernel math.
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
# Multi-tensor kernel parity.
# ---------------------------------------------------------------------------


@gpu_only
@pytest.mark.parametrize("advanced_removal", [False, True])
@pytest.mark.parametrize("cautious", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_syre_multi_matches_single_loop(advanced_removal, cautious, dtype):
    """``syre_wd_multi_inplace`` must produce bit-identical results to a
    loop over ``syre_wd_inplace`` for every variant. Covers the full
    cross-product of (basic/AR) x (non-cautious/cautious) x (fp32/bf16).
    """
    from dion.syre import syre_wd_inplace, syre_wd_multi_inplace

    torch.manual_seed(0)
    shapes = [(127,), (256, 8), (1023,), (33, 99), (8, 4, 7)]
    n = len(shapes)
    seeds1 = list(range(1, n + 1))
    seeds2 = list(range(101, 101 + n))
    offset_bases = [i * 1024 for i in range(n)]
    gamma = 3e-5
    std = 1e-3
    d_bound = 1e-4

    Xs_ref = [torch.randn(*s, device="cuda", dtype=dtype) for s in shapes]
    Xs_mul = [x.clone() for x in Xs_ref]
    if cautious:
        Us = [torch.randn(*s, device="cuda", dtype=dtype) for s in shapes]
        Us_ref = [u.clone() for u in Us]
        Us_mul = [u.clone() for u in Us]
    else:
        Us_ref = [None] * n
        Us_mul = None

    for i in range(n):
        syre_wd_inplace(
            Xs_ref[i], gamma=gamma, seed1=seeds1[i], std=std,
            seed2=seeds2[i], d_bound=d_bound,
            advanced_removal=advanced_removal,
            offset_base=offset_bases[i],
            U=Us_ref[i],
        )

    syre_wd_multi_inplace(
        Xs_mul, gamma=gamma, seeds1=seeds1, std=std,
        seeds2=seeds2, d_bound=d_bound,
        advanced_removal=advanced_removal,
        offset_bases=offset_bases,
        Us=Us_mul,
    )

    for i in range(n):
        assert torch.equal(Xs_ref[i], Xs_mul[i]), (
            f"param {i} (shape={shapes[i]}, dtype={dtype}): max abs diff "
            f"= {(Xs_ref[i] - Xs_mul[i]).abs().max().item():.3e}"
        )


@gpu_only
def test_syre_multi_empty_list_is_noop():
    from dion.syre import syre_wd_multi_inplace
    # No-op; must not raise.
    syre_wd_multi_inplace(
        [], gamma=1e-3, seeds1=[], std=0.01,
        seeds2=[], d_bound=1e-4,
        advanced_removal=False, offset_bases=[],
    )


@gpu_only
def test_syre_multi_zero_gamma_is_noop():
    from dion.syre import syre_wd_multi_inplace

    torch.manual_seed(0)
    Xs = [torch.randn(128, device="cuda") for _ in range(3)]
    snap = [x.clone() for x in Xs]
    syre_wd_multi_inplace(
        Xs, gamma=0.0, seeds1=[1, 2, 3], std=0.1,
        seeds2=[0, 0, 0], d_bound=0.0,
        advanced_removal=False, offset_bases=[0, 0, 0],
    )
    for x, s in zip(Xs, snap):
        assert torch.equal(x, s)


@gpu_only
def test_syre_multi_skips_empty_tensors():
    """An empty param mixed in with non-empty ones must be silently
    skipped (no kernel work for it), and the non-empty ones must produce
    the same result as if the empty had not been there at all.
    """
    from dion.syre import syre_wd_inplace, syre_wd_multi_inplace

    torch.manual_seed(0)
    x1 = torch.randn(64, device="cuda")
    x_empty = torch.empty(0, device="cuda")
    x2 = torch.randn(33, device="cuda")
    ref1 = x1.clone()
    ref2 = x2.clone()

    syre_wd_inplace(ref1, gamma=1e-3, seed1=1, std=0.01,
                    seed2=0, d_bound=0.0,
                    advanced_removal=False, offset_base=0)
    syre_wd_inplace(ref2, gamma=1e-3, seed1=3, std=0.01,
                    seed2=0, d_bound=0.0,
                    advanced_removal=False, offset_base=0)

    syre_wd_multi_inplace(
        [x1, x_empty, x2], gamma=1e-3, seeds1=[1, 2, 3], std=0.01,
        seeds2=[0, 0, 0], d_bound=0.0,
        advanced_removal=False, offset_bases=[0, 0, 0],
    )
    assert torch.equal(x1, ref1)
    assert x_empty.numel() == 0
    assert torch.equal(x2, ref2)


@gpu_only
def test_syre_multi_rejects_mismatched_dtype():
    from dion.syre import syre_wd_multi_inplace

    x1 = torch.randn(32, device="cuda", dtype=torch.float32)
    x2 = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        syre_wd_multi_inplace(
            [x1, x2], gamma=1e-3, seeds1=[1, 2], std=0.01,
            seeds2=[0, 0], d_bound=0.0,
            advanced_removal=False, offset_bases=[0, 0],
        )


@gpu_only
def test_syre_multi_rejects_unsupported_dtype():
    from dion.syre import syre_wd_multi_inplace

    x = torch.randn(32, device="cuda", dtype=torch.float64)
    with pytest.raises(TypeError, match="unsupported dtype"):
        syre_wd_multi_inplace(
            [x], gamma=1e-3, seeds1=[1], std=0.01,
            seeds2=[0], d_bound=0.0,
            advanced_removal=False, offset_bases=[0],
        )
