"""SYRE configuration / validation layer (triton-free).

This module owns the SYRE option-validation and ``d_bound`` resolution
logic. Crucially, it does **not** import triton or :mod:`dion.syre`
at module load time, so it can be imported and exercised in environments
that don't have triton installed. The triton-availability gate
(:func:`_check_syre_triton_available`) does a deferred ``importlib`` call
only when the caller has actually requested ``syre_wd=True``.

What lives here:

* :class:`SyreTheorem3Warning` -- soft warning when
  ``sigma_D = d_bound / sqrt(3)`` is not ``o(sigma_0)``.
* :data:`_SYRE_AR_D_BOUND_FP32_FLOOR` -- the smallest ``d_bound`` we let
  through when SYRE-AR is on.
* :func:`_validate_syre_kwargs` -- strict per-kwarg type/value rejection.
* :func:`_resolve_syre_d_bound` -- ``None -> 0.1 * syre_std`` auto-
  resolution and the AR-meaningfulness / Theorem-3 checks.
* :func:`_check_syre_per_param_sigma0` -- per-tensor sigma_0 Theorem-3
  warning for ``syre_std_mode`` groups (deferred to step time).
* :func:`_check_syre_triton_available` -- the triton import gate.

The Aurora class re-exports :class:`SyreTheorem3Warning` (and the
private helpers used by tests) from :mod:`dion.aurora` for back-compat.
"""

from __future__ import annotations

import math
import warnings


# Smallest ``d_bound`` we let through when SYRE-AR is on. The SYRE
# kernel uses the additive form (computes ``diff*(1+xi)`` as
# ``diff + diff*xi`` rather than forming ``1+xi`` in fp32), so the
# representation-side pigeonhole on ``D`` is avoided -- ``xi`` is
# precise near 0 down to denormals. The remaining concern is purely
# *meaningfulness*: at very small ``d_bound``, the per-element AR
# contribution ``gamma * diff * xi`` falls so far below fp32 ulp at
# typical weight magnitudes that it does not accumulate to a
# meaningful value even over a full WD half-life. 1e-6 is the
# rough threshold below which AR has no measurable effect on training
# for any realistic schedule (lr ~ 1e-3, wd ~ 0.1, ~30k step horizons).
_SYRE_AR_D_BOUND_FP32_FLOOR = 1e-6


class SyreTheorem3Warning(UserWarning):
    """``sigma_D = d_bound / sqrt(3)`` is not ``o(sigma_0)``.

    Theorem 3 of Ziyin et al. (2024) only proves SYRE-AR's symmetry-
    removal strength when ``sigma_D = o(sigma_0)``. The default
    ``d_bound = 0.1 * syre_std`` (auto-resolved) gives
    ``sigma_D/sigma_0 ~ 0.058``, comfortably perturbative. A user-passed
    ``d_bound`` large enough that ``sigma_D >= sigma_0`` puts AR outside
    that regime: basic SYRE still works, but AR's theoretical guarantee
    no longer applies and AR may dominate the symmetric pull from
    ``theta_0``. We warn rather than error because some users may
    deliberately want this regime for exploration.
    """


def _validate_syre_kwargs(
    syre_wd, syre_std, advanced_removal, d_bound, syre_std_mode=None,
):
    """Strict validation for SYRE-related per-group kwargs.

    Called from ``Aurora.__init__`` and ``Aurora.add_param_group`` so
    typos and bad types are caught at construction / mutation time
    rather than at the first SYRE step.

    ``d_bound`` may be ``None`` (auto-resolve at the call site to
    ``0.1 * syre_std`` when SYRE-AR is on) or a numeric in ``[0, 1)``.
    Passing an explicit ``d_bound == 0`` together with
    ``advanced_removal=True`` is rejected as operator error: it
    collapses ``D`` to the identity and defeats the entire purpose of
    advanced removal. Use ``advanced_removal=False`` if you don't want
    AR, or pick a positive ``d_bound``.

    ``syre_std`` (scalar) and ``syre_std_mode`` (preset name or callable)
    are mutually exclusive. When ``syre_wd=True``, exactly one must be
    set. ``syre_std_mode`` resolves to a per-tensor sigma_0 at param-
    registration time -- see :mod:`dion.syre` for the available presets.
    """
    if not isinstance(syre_wd, bool):
        raise TypeError(
            f"syre_wd must be a bool, got {type(syre_wd).__name__}: {syre_wd!r}"
        )
    if not isinstance(advanced_removal, bool):
        raise TypeError(
            f"advanced_removal must be a bool, got "
            f"{type(advanced_removal).__name__}: {advanced_removal!r}"
        )
    if d_bound is not None:
        if not isinstance(d_bound, (int, float)) or isinstance(d_bound, bool):
            raise TypeError(
                f"d_bound must be a float in [0, 1) or None, got "
                f"{type(d_bound).__name__}: {d_bound!r}"
            )
        if not (0.0 <= float(d_bound) < 1.0):
            raise ValueError(
                f"d_bound must be in [0, 1) or None, got {d_bound}"
            )
    if syre_wd and advanced_removal and d_bound is not None and float(d_bound) == 0.0:
        raise ValueError(
            "d_bound=0 with advanced_removal=True is rejected: the "
            "Uniform(1-0, 1+0) multiplier collapses D to the identity, "
            "which defeats AR's symmetry-breaking purpose. Either set "
            "advanced_removal=False, omit d_bound (auto-resolves to "
            "0.1 * syre_std per Theorem 3's sigma_D = o(sigma_0) "
            "condition), or pass a positive d_bound."
        )

    # syre_std vs syre_std_mode: mutually exclusive, exactly one required
    # when SYRE is on. Validate types/values for whichever side is set.
    if syre_wd:
        if syre_std is not None and syre_std_mode is not None:
            raise ValueError(
                "syre_std and syre_std_mode are mutually exclusive. Pass "
                "syre_std=<float> for a single uniform sigma_0, OR "
                "syre_std_mode=<preset_name_or_callable> for per-tensor "
                "scaling -- not both."
            )
        if syre_std is None and syre_std_mode is None:
            from .syre import SYRE_STD_PRESETS
            raise ValueError(
                "syre_wd=True requires either syre_std (positive float) or "
                "syre_std_mode (preset name or callable). There is "
                "intentionally no empirical default. Presets available: "
                f"{sorted(SYRE_STD_PRESETS)}. Or pass a callable like "
                "lambda p: 0.01 * <your_init_std_for_p>."
            )
        if syre_std is not None:
            if isinstance(syre_std, bool) or not isinstance(syre_std, (int, float)):
                raise TypeError(
                    f"syre_std must be a positive float, got "
                    f"{type(syre_std).__name__}: {syre_std!r}"
                )
            if float(syre_std) <= 0.0:
                raise ValueError(
                    f"syre_std must be a positive float, got {syre_std}"
                )
        if syre_std_mode is not None:
            if isinstance(syre_std_mode, str):
                from .syre import SYRE_STD_PRESETS
                if syre_std_mode not in SYRE_STD_PRESETS:
                    raise ValueError(
                        f"Unknown syre_std_mode preset: {syre_std_mode!r}. "
                        f"Known presets: {sorted(SYRE_STD_PRESETS)}."
                    )
            elif not callable(syre_std_mode):
                raise TypeError(
                    f"syre_std_mode must be a preset name (str) or a "
                    f"callable taking a Tensor and returning a positive "
                    f"float, got {type(syre_std_mode).__name__}: "
                    f"{syre_std_mode!r}"
                )


def _resolve_syre_d_bound(syre_wd, syre_std, advanced_removal, d_bound):
    """Auto-resolve ``d_bound=None`` to ``0.1 * syre_std`` when both
    SYRE and AR are on; otherwise pass through unchanged. Also rejects
    a resolved ``d_bound`` so small that AR has no measurable effect.

    Note: when the group uses ``syre_std_mode`` (per-tensor sigma_0)
    instead of a scalar ``syre_std``, this function is called with
    ``syre_std=None``; the auto-resolution and per-param Theorem-3 /
    AR-floor checks are deferred to :func:`_check_syre_per_param_sigma0`,
    invoked at step time once per-param sigma_0 is known. The d_bound
    range check (``[0, 1)``, plus the explicit-zero-with-AR refusal)
    still runs through :func:`_validate_syre_kwargs`.

    The paper (Ziyin et al., 2024) only proves AR's symmetry-removal
    strength under ``sigma_D = o(sigma_0)`` (Theorem 3). For
    ``D_ii ~ Uniform(1 - d_bound, 1 + d_bound)``, ``sigma_D = d_bound /
    sqrt(3)``. Setting ``d_bound = 0.1 * syre_std`` gives
    ``sigma_D / sigma_0 ~ 0.058`` -- comfortably in the perturbative
    regime. Users who want a different ratio can pass ``d_bound``
    explicitly.

    Floor (1e-6): a positive but very small ``d_bound`` is rejected
    with ``ValueError`` when AR is enabled. The kernel uses the
    additive form ``diff + diff*xi``, so the per-element AR multiplier
    has full fp32 precision (no pigeonhole on D values). But at
    ``d_bound < 1e-6``, the per-element AR contribution
    ``gamma * diff * xi`` is so small relative to fp32 ulp at typical
    weight magnitudes that it does not accumulate to a measurable
    value over realistic training horizons -- AR becomes silently
    irrelevant. This commonly happens if the user picks an unusually
    small ``syre_std`` (<= ~1e-5) and lets ``d_bound`` auto-resolve,
    or passes a tiny explicit ``d_bound``. The fix is to pick a larger
    ``syre_std`` (paper recommends ``0.01 / sqrt(d)``) or an explicit
    positive ``d_bound >= 1e-6``.

    Must be called *after* :func:`_validate_syre_kwargs` so we can rely
    on ``syre_std`` being a positive float when SYRE is on.
    """
    if (
        syre_wd
        and advanced_removal
        and d_bound is None
        and syre_std is not None
    ):
        d_bound = 0.1 * float(syre_std)
    if (
        syre_wd
        and advanced_removal
        and d_bound is not None
        and 0.0 < float(d_bound) < _SYRE_AR_D_BOUND_FP32_FLOOR
    ):
        raise ValueError(
            f"d_bound={d_bound!r} is below the AR-meaningfulness floor "
            f"({_SYRE_AR_D_BOUND_FP32_FLOOR:.0e}) for SYRE-AR: at this "
            "scale the per-element AR contribution gamma*diff*xi does "
            "not accumulate to a measurable value over realistic "
            "training horizons (fp32 storage ulp at typical weight "
            "magnitudes dominates). If you let d_bound auto-resolve "
            f"(d_bound=None), this means your syre_std ({syre_std!r}) "
            "is too small -- the paper recommends "
            "syre_std = 0.01 / sqrt(d). Otherwise pass an explicit "
            "d_bound >= 1e-6."
        )
    # Theorem-3-precondition soft check. ``sigma_D = d_bound / sqrt(3)``;
    # auto-resolved ``d_bound = 0.1 * syre_std`` gives ``sigma_D/sigma_0
    # ~ 0.058`` and never trips this. A user-passed combo that does is
    # warned but not refused -- AR may still help empirically, the paper
    # just doesn't prove it in that regime.
    if (
        syre_wd
        and advanced_removal
        and d_bound is not None
        and syre_std is not None
        and float(d_bound) / math.sqrt(3.0) >= float(syre_std)
    ):
        sigma_d = float(d_bound) / math.sqrt(3.0)
        warnings.warn(
            f"SYRE-AR: sigma_D ({sigma_d:.3g}) >= sigma_0 ({float(syre_std):.3g}); "
            "Theorem 3 of Ziyin et al. (2024) only guarantees AR's "
            "symmetry-removal strength when sigma_D = o(sigma_0). Default "
            "auto-resolution (d_bound=None) gives sigma_D/sigma_0 ~ 0.058. "
            "Pass d_bound smaller (e.g. <= 0.5 * syre_std) or omit it to "
            "stay in the proved regime.",
            SyreTheorem3Warning,
            stacklevel=3,
        )
    return d_bound


def _check_syre_per_param_sigma0(
    sigma0_list, advanced_removal, d_bound,
):
    """Per-param Theorem-3 + AR-floor check for ``syre_std_mode`` groups.

    Scalar-``syre_std`` groups handle both checks at construction time
    in :func:`_resolve_syre_d_bound`. When the group uses
    ``syre_std_mode``, sigma_0 is only known per-param at step time, so
    we defer both checks here. Fires at most once per ``(group, kind)``
    pair (callers cache after first invocation via the ``_checked`` flag
    on the optimizer state) so warnings/errors don't repeat every step.

    Uses ``min(sigma0_list)`` as the worst case for the Theorem-3 ratio
    -- if any param's sigma_0 is small enough to violate sigma_D <
    sigma_0, AR is outside the proved regime for at least that param.
    The AR-floor check is over min sigma_0 only when d_bound auto-
    resolves; with the user-set d_bound path, only d_bound matters
    (handled in :func:`_resolve_syre_d_bound`).
    """
    if not sigma0_list or not advanced_removal or d_bound is None:
        return
    sigma_d = float(d_bound) / math.sqrt(3.0)
    sigma0_min = min(sigma0_list)
    if sigma_d >= sigma0_min:
        warnings.warn(
            f"SYRE-AR: sigma_D ({sigma_d:.3g}) >= min per-param sigma_0 "
            f"({sigma0_min:.3g}); Theorem 3 of Ziyin et al. (2024) only "
            "guarantees AR's symmetry-removal strength when sigma_D = "
            "o(sigma_0). For per-tensor syre_std_mode this is checked "
            "against the smallest resolved sigma_0 across the group. "
            "Pass a smaller d_bound (e.g. <= 0.5 * smallest_sigma_0) or "
            "switch to a preset that gives larger sigma_0 to stay in "
            "the proved regime.",
            SyreTheorem3Warning,
            stacklevel=3,
        )


def _check_syre_triton_available():
    """Raise ``ImportError`` if the SYRE Triton module isn't importable.

    Uses ``importlib.import_module`` rather than ``from . import syre``
    so the lookup honors ``sys.modules`` entries set to ``None`` (the
    standard "module deliberately unavailable" sentinel used by the
    tests). With ``from . import syre`` the bytecode falls back to
    ``getattr(package, 'syre')``, which returns a cached module object
    set as a package attribute by an earlier successful import even
    when ``sys.modules`` has been patched.

    The error message is tailored for the ``syre_wd=True`` user path;
    callers only invoke this when SYRE is actually being requested.
    """
    import importlib
    try:
        importlib.import_module(".syre", "dion")
    except ImportError as e:
        raise ImportError(
            "syre_wd=True requires triton. Install dion's optional "
            "triton extra (or install triton directly) and retry."
        ) from e
