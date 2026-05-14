"""Per-tensor sigma_0 via ``syre_std_mode`` (preset name or callable).

Mutual exclusion with ``syre_std``, the back-compat invariant when
``syre_std_mode=None``, preset formulas (lecun / kaiming / xavier) on
2D, conv-shape, ndim<2 and 0D inputs, callable validation, kernel-level
per-param ``std`` parity, end-to-end resolve+cache, state_dict round-
trip, the AdamW-on-1D path, and the Theorem-3 warning's min-sigma_0
worst-case logic.
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
# Validation: mutual exclusion, neither-set, preset and callable typing.
# ---------------------------------------------------------------------------


def test_syre_std_and_mode_both_set_raises():
    """``syre_std`` and ``syre_std_mode`` are mutually exclusive."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="mutually exclusive"):
        Aurora([p], syre_wd=True, syre_std=0.01,
               syre_std_mode="lecun_fan_in")


def test_syre_neither_std_nor_mode_set_raises():
    """``syre_wd=True`` requires one of ``syre_std`` / ``syre_std_mode``."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="syre_std \\(positive float\\) or syre_std_mode"):
        Aurora([p], syre_wd=True)


def test_syre_std_mode_unknown_preset_raises():
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(ValueError, match="Unknown syre_std_mode"):
        Aurora([p], syre_wd=True, syre_std_mode="not_a_real_preset")


@pytest.mark.parametrize("bad", [0, 1, 1.5, [], object()])
def test_syre_std_mode_wrong_type_raises(bad):
    p = torch.nn.Parameter(torch.randn(8, 4))
    with pytest.raises(TypeError, match="syre_std_mode"):
        Aurora([p], syre_wd=True, syre_std_mode=bad)


def test_syre_std_mode_back_compat_unaffected():
    """When ``syre_std_mode=None`` (default), the existing scalar-only
    code path must be bit-identical to before this kwarg existed.
    """
    p1 = torch.nn.Parameter(torch.randn(8, 4))
    p2 = torch.nn.Parameter(p1.detach().clone())
    # Both constructions must produce the same defaults / group dict
    # entries except for the addition of ``syre_std_mode=None``.
    opt_old_like = Aurora([p1], syre_wd=False)
    opt_with_mode_kwarg = Aurora([p2], syre_wd=False, syre_std_mode=None)
    g1 = opt_old_like.param_groups[0]
    g2 = opt_with_mode_kwarg.param_groups[0]
    assert g1["syre_wd"] == g2["syre_wd"] == False
    assert g1["syre_std"] is None and g2["syre_std"] is None
    assert g1.get("syre_std_mode") is None
    assert g2.get("syre_std_mode") is None


# ---------------------------------------------------------------------------
# Preset / callable resolver formulas.
# ---------------------------------------------------------------------------


def test_resolve_syre_std_preset_formulas():
    """Each preset reproduces the documented per-tensor formula exactly."""
    import math
    from dion.syre import resolve_syre_std

    # 2D weight: fan_in = shape[-1] = in, fan_out = shape[0] = out.
    p = torch.nn.Parameter(torch.empty(64, 128))  # out=64, in=128
    fan_in, fan_out = 128, 64

    lecun = resolve_syre_std(p, "lecun_fan_in")
    assert math.isclose(lecun, 0.01 / math.sqrt(fan_in))

    kaiming = resolve_syre_std(p, "kaiming_fan_in")
    assert math.isclose(kaiming, 0.01 * math.sqrt(2.0 / fan_in))

    xavier = resolve_syre_std(p, "xavier_normal")
    assert math.isclose(xavier, 0.01 * math.sqrt(2.0 / (fan_in + fan_out)))


def test_resolve_syre_std_conv_shape_fan_in_includes_kernel():
    """For ndim>2 (e.g. conv), fan_in = prod(shape[1:]) per
    ``torch.nn.init`` convention.
    """
    import math
    from dion.syre import resolve_syre_std

    p = torch.nn.Parameter(torch.empty(32, 16, 3, 3))  # out=32, in=16, k=3
    expected_fan_in = 16 * 3 * 3
    lecun = resolve_syre_std(p, "lecun_fan_in")
    assert math.isclose(lecun, 0.01 / math.sqrt(expected_fan_in))


def test_resolve_syre_std_preset_ndim_lt_2_uses_numel():
    """For ndim<2 (bias / LayerNorm / scalar params), presets degrade
    to ``fan_in = numel, fan_out = 1`` -- the direct extension of the
    matrix formula. Lets users include those params in a SYRE-decayed
    AdamW group without writing a custom callable.
    """
    import math
    from dion.syre import resolve_syre_std

    p = torch.nn.Parameter(torch.zeros(16))  # 1D, numel=16
    lecun = resolve_syre_std(p, "lecun_fan_in")
    assert math.isclose(lecun, 0.01 / math.sqrt(16))
    kaiming = resolve_syre_std(p, "kaiming_fan_in")
    assert math.isclose(kaiming, 0.01 * math.sqrt(2.0 / 16))
    xavier = resolve_syre_std(p, "xavier_normal")
    assert math.isclose(xavier, 0.01 * math.sqrt(2.0 / (16 + 1)))


def test_resolve_syre_std_preset_scalar_param():
    """0D / numel=1 params use ``fan_in = 1, fan_out = 1``."""
    import math
    from dion.syre import resolve_syre_std

    p = torch.nn.Parameter(torch.zeros(()))  # 0D scalar
    lecun = resolve_syre_std(p, "lecun_fan_in")
    assert math.isclose(lecun, 0.01 / math.sqrt(1))


def test_resolve_syre_std_callable_accepts_any_ndim():
    """The callable escape hatch must work even on ndim<2 params -- the
    preset restriction is to fan-in-based formulas, not to the API.
    """
    from dion.syre import resolve_syre_std

    p = torch.nn.Parameter(torch.zeros(16))
    value = resolve_syre_std(p, lambda x: 0.02)
    assert value == 0.02


def test_resolve_syre_std_callable_non_positive_raises():
    from dion.syre import resolve_syre_std

    p = torch.nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        resolve_syre_std(p, lambda x: 0.0)


def test_resolve_syre_std_callable_non_numeric_raises():
    from dion.syre import resolve_syre_std

    p = torch.nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(TypeError, match="non-numeric"):
        resolve_syre_std(p, lambda x: "nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Kernel-level per-param std parity.
# ---------------------------------------------------------------------------


@gpu_only
def test_syre_multi_per_param_std_matches_per_param_scalar_calls():
    """``syre_wd_multi_inplace`` with a per-param ``std`` sequence must
    match running ``syre_wd_inplace`` once per param with that param's
    own scalar ``std``. This is the kernel-level contract for
    ``syre_std_mode``.
    """
    from dion.syre import syre_wd_inplace, syre_wd_multi_inplace

    torch.manual_seed(0)
    shapes = [(64,), (33, 5), (128,)]
    stds = [1e-3, 5e-4, 2e-3]
    seeds1 = [1, 2, 3]
    seeds2 = [10, 20, 30]
    offset_bases = [0, 1024, 2048]
    gamma = 1e-3

    Xs_ref = [torch.randn(*s, device="cuda", dtype=torch.float32) for s in shapes]
    Xs_mul = [x.clone() for x in Xs_ref]

    for i, _ in enumerate(shapes):
        syre_wd_inplace(
            Xs_ref[i], gamma=gamma, seed1=seeds1[i], std=stds[i],
            seed2=seeds2[i], d_bound=0.0,
            advanced_removal=False, offset_base=offset_bases[i],
        )
    syre_wd_multi_inplace(
        Xs_mul, gamma=gamma, seeds1=seeds1, std=stds,
        seeds2=seeds2, d_bound=0.0,
        advanced_removal=False, offset_bases=offset_bases,
    )
    for i, _ in enumerate(shapes):
        assert torch.equal(Xs_ref[i], Xs_mul[i]), (
            f"param {i}: max diff = "
            f"{(Xs_ref[i] - Xs_mul[i]).abs().max().item():.3e}"
        )


@gpu_only
def test_syre_multi_per_param_std_uniform_matches_scalar_form():
    """When all entries of the per-param ``std`` sequence are equal, the
    multi-tensor result must equal the same call with a scalar ``std``.
    Confirms the scalar fast-path is exactly the uniform-list path.
    """
    from dion.syre import syre_wd_multi_inplace

    torch.manual_seed(0)
    shapes = [(64,), (32, 5), (101,)]
    n = len(shapes)
    Xs_a = [torch.randn(*s, device="cuda") for s in shapes]
    Xs_b = [x.clone() for x in Xs_a]
    common = dict(
        gamma=2e-3, seeds1=[1, 2, 3], seeds2=[10, 20, 30],
        d_bound=0.0, advanced_removal=False,
        offset_bases=[0, 1024, 2048],
    )
    syre_wd_multi_inplace(Xs_a, std=7e-4, **common)
    syre_wd_multi_inplace(Xs_b, std=[7e-4] * n, **common)
    for a, b in zip(Xs_a, Xs_b):
        assert torch.equal(a, b)


@gpu_only
def test_syre_multi_per_param_std_length_mismatch_raises():
    from dion.syre import syre_wd_multi_inplace

    Xs = [torch.randn(8, device="cuda") for _ in range(3)]
    with pytest.raises(ValueError, match="length"):
        syre_wd_multi_inplace(
            Xs, gamma=1e-3, seeds1=[1, 2, 3],
            std=[1e-3, 2e-3],  # 2 != 3
            seeds2=[0, 0, 0], d_bound=0.0,
            advanced_removal=False, offset_bases=[0, 0, 0],
        )


# ---------------------------------------------------------------------------
# End-to-end resolve + cache + state_dict round-trip.
# ---------------------------------------------------------------------------


@gpu_only
def test_syre_std_mode_resolves_per_param_and_caches():
    """End-to-end: opt with ``syre_std_mode="lecun_fan_in"`` resolves and
    caches a per-param sigma_0 in optimizer state, and the cached value
    matches the preset formula for each param's shape.
    """
    import math

    torch.manual_seed(0)
    # Two params with different fan-in, so the preset resolves to
    # different sigma_0 per param.
    p1 = torch.nn.Parameter(torch.randn(64, 128, device="cuda") * 0.02)
    p2 = torch.nn.Parameter(torch.randn(32, 16, device="cuda") * 0.02)
    opt = Aurora([p1, p2], lr=1e-3, weight_decay=0.1,
                 syre_wd=True, syre_std_mode="lecun_fan_in")
    p1.grad = torch.zeros_like(p1)
    p2.grad = torch.zeros_like(p2)
    opt.step()

    assert math.isclose(
        opt.state[p1]["syre_std"], 0.01 / math.sqrt(128),
    )
    assert math.isclose(
        opt.state[p2]["syre_std"], 0.01 / math.sqrt(16),
    )


@gpu_only
def test_syre_std_mode_callable_resolves_per_param_and_caches():
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.02)
    p2 = torch.nn.Parameter(torch.randn(8, 4, device="cuda") * 0.02)

    # Per-param sigma_0 keyed off id(p) just to give the two distinct
    # values without using shape.
    explicit = {id(p1): 1e-3, id(p2): 3e-3}
    opt = Aurora([p1, p2], lr=1e-3, weight_decay=0.1,
                 syre_wd=True, syre_std_mode=lambda p: explicit[id(p)])
    p1.grad = torch.zeros_like(p1)
    p2.grad = torch.zeros_like(p2)
    opt.step()

    assert opt.state[p1]["syre_std"] == 1e-3
    assert opt.state[p2]["syre_std"] == 3e-3


@gpu_only
def test_syre_std_mode_state_dict_round_trip():
    """Per-param sigma_0 (resolved from the preset) must round-trip
    through ``state_dict`` -- a resumed optimizer sees the same cached
    values and doesn't re-resolve.
    """
    p = torch.nn.Parameter(torch.randn(64, 128, device="cuda") * 0.02)
    opt = Aurora([p], lr=1e-3, weight_decay=0.1,
                 syre_wd=True, syre_std_mode="lecun_fan_in")
    p.grad = torch.zeros_like(p)
    opt.step()
    cached = opt.state[p]["syre_std"]
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(p.detach().clone())
    # Resume with a DIFFERENT preset -- the cached value must win.
    opt2 = Aurora([p2], lr=1e-3, weight_decay=0.1,
                  syre_wd=True, syre_std_mode="kaiming_fan_in")
    opt2.load_state_dict(sd)
    assert opt2.state[p2]["syre_std"] == cached


@gpu_only
def test_syre_std_mode_e2e_pulls_toward_per_param_theta0():
    """With zero grad and the preset, each param decays toward a
    theta_0 drawn with that param's resolved sigma_0. Hand-roll the
    expected update against the kernel.
    """
    from dion.syre import syre_wd_inplace, resolve_syre_std

    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(32, 64, device="cuda") * 0.02)
    p2 = torch.nn.Parameter(torch.randn(16, 32, device="cuda") * 0.02)
    lr, wd = 0.05, 0.2
    opt = Aurora([p1, p2], lr=lr, weight_decay=wd,
                 syre_wd=True, syre_std_mode="lecun_fan_in",
                 adjust_lr=None)
    # Force deterministic seeds so the hand-roll can reproduce theta_0.
    # Pre-init momentum because writing seed keys makes state non-empty,
    # which short-circuits ``_get_or_initialize_state``'s lazy init.
    opt.state[p1]["momentum"] = torch.zeros_like(p1)
    opt.state[p2]["momentum"] = torch.zeros_like(p2)
    opt.state[p1]["syre_seed1"] = 4242
    opt.state[p1]["syre_seed2"] = 0
    opt.state[p2]["syre_seed1"] = 9999
    opt.state[p2]["syre_seed2"] = 0

    p1_init = p1.detach().clone()
    p2_init = p2.detach().clone()
    p1.grad = torch.zeros_like(p1)
    p2.grad = torch.zeros_like(p2)
    opt.step()

    std1 = resolve_syre_std(p1_init, "lecun_fan_in")
    std2 = resolve_syre_std(p2_init, "lecun_fan_in")

    # Reproduce theta_0 via gamma=1 from theta=0 of matching shape.
    t0_1 = torch.zeros_like(p1_init).contiguous()
    syre_wd_inplace(t0_1, gamma=1.0, seed1=4242, std=std1,
                    seed2=0, d_bound=0.0, advanced_removal=False,
                    offset_base=0)
    t0_2 = torch.zeros_like(p2_init).contiguous()
    syre_wd_inplace(t0_2, gamma=1.0, seed1=9999, std=std2,
                    seed2=0, d_bound=0.0, advanced_removal=False,
                    offset_base=0)

    gamma = lr * wd
    expected1 = p1_init - gamma * (p1_init - t0_1)
    expected2 = p2_init - gamma * (p2_init - t0_2)
    torch.testing.assert_close(p1.detach(), expected1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p2.detach(), expected2, atol=1e-5, rtol=1e-5)


@gpu_only
def test_syre_std_mode_preset_works_on_1d_adamw_param():
    """A preset on an AdamW group with a 1D param (bias / LN) must
    succeed -- the ndim<2 numel fallback gives a small but valid
    per-element sigma_0. The cached value matches the preset formula
    using ``fan_in = numel, fan_out = 1``.
    """
    import math

    n = 16
    p = torch.nn.Parameter(torch.randn(n, device="cuda") * 0.1)
    opt = Aurora([
        {"params": [p], "algorithm": "adamw"},
    ], lr=0.05, weight_decay=0.05,
       syre_wd=True, syre_std_mode="lecun_fan_in")
    p.grad = torch.randn_like(p)
    opt.step()  # must not raise
    assert math.isclose(opt.state[p]["syre_std"], 0.01 / math.sqrt(n))


def test_syre_std_mode_theorem3_warning_uses_min_per_param_sigma0():
    """The Theorem-3 warning for ``syre_std_mode`` groups uses the min
    resolved sigma_0 across the group (worst case). With min sigma_0 =
    1e-4 and d_bound=1e-3, sigma_D = 5.8e-4 > 1e-4 should fire.

    We exercise the warning helper directly rather than via
    ``opt.step()`` so this test stays cheap and doesn't depend on the
    Newton-Schulz compile cache for the chosen shapes.
    """
    from dion.aurora import _check_syre_per_param_sigma0

    # First call -- should fire.
    with pytest.warns(SyreTheorem3Warning, match="min per-param sigma_0"):
        _check_syre_per_param_sigma0(
            sigma0_list=[1e-4, 1e-3],  # min = 1e-4
            advanced_removal=True,
            d_bound=1e-3,             # sigma_D = 1e-3/sqrt(3) ~ 5.8e-4
        )

    # Below the threshold (sigma_D < min sigma_0) -- silent.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", SyreTheorem3Warning)
        _check_syre_per_param_sigma0(
            sigma0_list=[1e-2, 1e-3],  # min = 1e-3
            advanced_removal=True,
            d_bound=1e-4,              # sigma_D = 5.8e-5 < 1e-3
        )

    # AR off -- silent regardless of d_bound.
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", SyreTheorem3Warning)
        _check_syre_per_param_sigma0(
            sigma0_list=[1e-6],
            advanced_removal=False,
            d_bound=1e-3,
        )
