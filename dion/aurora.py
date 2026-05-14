import math
import warnings

import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import (
    DistributedOrthoBase,
    megabatch_orthogonalize_async,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
    adjust_lr_aurora_aspect,
)
from .muon import muon_update_pre_orthogonalize, muon_update_post_orthogonalize
from .opt_utils import AsyncTask, to_local
from .scalar_opts import adamw_update_foreach_syre_async


def _validate_syre_kwargs(
    syre_wd, syre_std, advanced_removal, d_bound,
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
    if syre_wd:
        if syre_std is None:
            raise ValueError(
                "syre_wd=True requires an explicit syre_std (positive float). "
                "There is intentionally no empirical default; if you want to "
                "scale by the param's init magnitude, compute it explicitly "
                "and pass syre_std=<your_value>."
            )
        if isinstance(syre_std, bool) or not isinstance(syre_std, (int, float)):
            raise TypeError(
                f"syre_std must be a positive float, got "
                f"{type(syre_std).__name__}: {syre_std!r}"
            )
        if float(syre_std) <= 0.0:
            raise ValueError(
                f"syre_std must be a positive float, got {syre_std}"
            )


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


def _resolve_syre_d_bound(syre_wd, syre_std, advanced_removal, d_bound):
    """Auto-resolve ``d_bound=None`` to ``0.1 * syre_std`` when both
    SYRE and AR are on; otherwise pass through unchanged. Also rejects
    a resolved ``d_bound`` so small that AR has no measurable effect.

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

    Must be called *after* ``_validate_syre_kwargs`` so we can rely on
    ``syre_std`` being a positive float when SYRE is on.
    """
    if syre_wd and advanced_removal and d_bound is None:
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
        importlib.import_module(".syre", __package__)
    except ImportError as e:
        raise ImportError(
            "syre_wd=True requires triton. Install dion's optional "
            "triton extra (or install triton directly) and retry."
        ) from e


class Aurora(DistributedOrthoBase):
    """
    Distributed Aurora optimizer for PyTorch FSDP2. Also compatible with DDP.

    Aurora is an optimizer for non-square weight matrices that achieves more
    balanced neuron utilization than standard Muon. Instead of applying the
    polar (Newton-Schulz) factor directly, which inherits non-uniform
    left-singular row norms, Aurora iteratively approximates a projection onto
    the intersection of the row oblique and Stiefel manifolds via diagonal
    preconditioning. The result is a leverage-uniform update.

    For square matrices Aurora reduces to standard Muon.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. Scaled by ``adjust_lr`` to convert from spectral
            norm 1 to a comparable RMS operator norm, same as Muon/NorMuon.
        mu: Momentum factor.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and parameter signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate ("spectral_norm", "rms_norm",
            "aurora_aspect", or None).
            "spectral_norm" (default, same as Muon/NorMuon): scales by sqrt(m/n)
            regardless of orientation. NOTE: this is NOT the Aurora-paper
            convention; wide-matrix updates will be smaller than the reference.
            See ``adjust_lr="aurora_aspect"`` for paper-faithful scaling.
            "aurora_aspect": scales by ``max(1, m/n)^0.5``, matching the Aurora
            reference exactly (wide matrices unscaled). Use this for
            reference-faithful Aurora.
            "rms_norm": Adam-comparable RMS scaling.
            None: no LR adjustment.
        flatten: Whether to flatten 3D+ tensors to 2D for the orthogonalization step.
        pp_iterations: Number of preconditioned-polar iterations. Each iteration
            calls the base polar (Newton-Schulz) once. ``pp_iterations=2`` is the
            Aurora paper default; ``pp_iterations=1`` is single-shot row-norm
            preconditioning.
        pp_beta: Exponent for the diagonal update between iterations.
        syre_wd: Whether to apply SYRE weight decay
            (``theta <- theta - lr*wd*(theta - theta_0)``) in place of the
            standard decoupled WD step. ``theta_0`` is regenerated from a
            stored per-parameter PRNG seed each step. Requires CUDA + triton.
            Applies to both ``algorithm="aurora"`` and ``algorithm="adamw"``
            groups; on the AdamW path the SYRE step runs after the bias-
            correction math so the cautious mask (when ``cautious_wd=True``)
            uses the update direction ``M_new / (sqrt(V_new/bc2) + eps)``,
            matching segyges/aurora. ``algorithm="lion"`` + ``syre_wd=True``
            is refused (not wired).
        syre_std: Required when ``syre_wd=True``. Scale of ``theta_0``. No
            empirical default -- if you want to couple this to your init
            magnitude, compute and pass it explicitly.
        advanced_removal: SYRE-AR variant. Multiplies the SYRE diff by a per-
            element ``Uniform(1 - d_bound, 1 + d_bound)`` factor before the
            decay step, breaking continuous symmetries the basic form leaves
            untouched. No-op when ``syre_wd=False``.
        d_bound: Half-width of the SYRE-AR uniform interval. Ignored unless
            ``advanced_removal=True``. ``None`` (default) auto-resolves to
            ``0.1 * syre_std`` so ``sigma_D / sigma_0 ~ 0.058``, satisfying
            the ``sigma_D = o(sigma_0)`` precondition of Theorem 3 in the
            SYRE paper (arXiv:2408.15495). Pass a positive float to override.
            An explicit ``d_bound=0`` with ``advanced_removal=True`` is
            rejected as operator error -- it collapses D to the identity.
        use_triton: Whether to use the Triton Newton-Schulz kernel.
        use_polar_express: Whether to use Polar Express for the base polar.
        newton_schulz_func: Optional custom base polar function. Aurora wraps
            this with its diagonal preconditioning loop.

    Aurora: https://blog.tilderesearch.com/blog/aurora
    Reference implementation: https://github.com/tilde-research/aurora-release
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = True,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        pp_iterations: int = 2,
        pp_beta: float = 0.5,
        syre_wd: bool = False,
        syre_std: Optional[float] = None,
        advanced_removal: bool = False,
        d_bound: Optional[float] = None,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", "aurora_aspect", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. "
                "Must be 'spectral_norm', 'rms_norm', 'aurora_aspect', or None."
            )
        if not isinstance(pp_iterations, int) or pp_iterations < 1:
            raise ValueError(
                f"Invalid pp_iterations: {pp_iterations}. Must be a positive integer."
            )
        if pp_beta < 0.0:
            raise ValueError(f"Invalid pp_beta: {pp_beta}")
        _validate_syre_kwargs(syre_wd, syre_std, advanced_removal, d_bound)
        d_bound = _resolve_syre_d_bound(
            syre_wd, syre_std, advanced_removal, d_bound,
        )

        # Gate the triton import at construction time when SYRE is on, so
        # users without triton don't have to wait until the first step
        # to find out it isn't available. ``add_param_group`` re-runs the
        # same gate for groups added after construction.
        if syre_wd:
            _check_syre_triton_available()

        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="aurora",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            syre_wd=syre_wd,
            syre_std=syre_std,
            advanced_removal=advanced_removal,
            d_bound=d_bound,
        )
        # Let the parent class resolve the standard polar function from the
        # usual option set; we then wrap it with Aurora's diagonal-
        # preconditioning loop. ``_create_ortho_tasks`` rebuilds the wrapper
        # each step so a scheduler can mutate ``pp_iterations`` / ``pp_beta``
        # the same way it can mutate ``lr`` / ``mu``.
        super().__init__(
            params, distributed_mesh, "aurora", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )
        self._aurora_base_polar = self._newton_schulz_func
        # Overwrite the parent's resolved func with the Aurora-wrapped version
        # so nothing accidentally reads the unwrapped polar at step time.
        self._newton_schulz_func = make_aurora_polar(
            base_polar=self._aurora_base_polar,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            eps=epsilon,
        )

    def add_param_group(self, param_group: dict) -> None:
        """Validate SYRE-related kwargs on each new group, mirroring the
        construction-time checks. Defaults are filled in by torch's base
        ``add_param_group`` from ``self.defaults`` if not present.

        Also gates the triton import for groups added later with
        ``syre_wd=True`` (the ``__init__`` gate only covers groups
        present at construction), and refuses
        ``algorithm="lion" + syre_wd=True`` (Lion + SYRE is not wired;
        see :meth:`_create_lion_tasks` for the same guard at step time).
        """
        syre_wd = param_group.get("syre_wd", self.defaults["syre_wd"])
        syre_std = param_group.get("syre_std", self.defaults["syre_std"])
        advanced_removal = param_group.get(
            "advanced_removal", self.defaults["advanced_removal"]
        )
        # ``None`` is meaningful here ("auto-resolve"); use ``in`` rather
        # than ``.get(...)`` so an explicitly-passed ``d_bound=None`` is
        # honored instead of being clobbered by the default.
        d_bound = (
            param_group["d_bound"] if "d_bound" in param_group
            else self.defaults["d_bound"]
        )
        _validate_syre_kwargs(syre_wd, syre_std, advanced_removal, d_bound)
        d_bound = _resolve_syre_d_bound(
            syre_wd, syre_std, advanced_removal, d_bound,
        )
        # Write the resolved value back so the group dict the base class
        # ends up storing has the auto-resolved float, not ``None``.
        param_group["d_bound"] = d_bound

        algorithm = param_group.get("algorithm", self.defaults["algorithm"])
        if syre_wd and algorithm == "lion":
            raise NotImplementedError(
                "syre_wd=True with algorithm='lion' is not supported. "
                "SYRE is wired for algorithm='aurora' and 'adamw' only."
            )
        if syre_wd:
            _check_syre_triton_available()
        super().add_param_group(param_group)

    def _get_or_init_syre_seeds(
        self, p: Tensor, advanced_removal: bool,
    ) -> Tuple[int, int]:
        """Lazily draw SYRE seeds for ``p`` and store them in ``self.state[p]``.

        Seeds are drawn from ``torch.randint``, so a global
        ``torch.manual_seed`` makes a run reproducible. ``seed2`` is only
        drawn (non-zero) when ``advanced_removal=True``; otherwise it stays
        0 and the kernel ignores it.

        State is keyed by the original parameter object, so the seeds
        round-trip through ``state_dict`` automatically.
        """
        s = self.state[p]
        if "syre_seed1" not in s:
            s["syre_seed1"] = int(torch.randint(0, 2**31, (1,)).item())
        if "syre_seed2" not in s:
            s["syre_seed2"] = (
                int(torch.randint(0, 2**31, (1,)).item())
                if advanced_removal else 0
            )
        return s["syre_seed1"], s["syre_seed2"]

    def _compute_syre_offset_base(self, p: Tensor) -> int:
        """Compute the per-rank Philox offset base for SYRE on ``p``.

        Sharded params: ``device_rank * padded_local_size`` where
        ``padded_local_size = ceil(global_numel / world_size)``. This is a
        sharding-plan-agnostic upper bound -- different ranks get disjoint
        offset ranges of at least ``local_numel`` each, with gaps that are
        never read. Uniqueness across ranks is what matters for SYRE
        correctness; tight packing isn't.

        Replicated params (regular Tensor, DDP, or single-GPU):
        ``offset_base = 0`` for every rank. All ranks pull toward the same
        ``theta_0`` so replicas stay synchronized after the step.

        Not stored -- recomputed each step so the value tracks the current
        sharding plan on resume (in case the user changes ``world_size``).
        """
        if isinstance(p, DTensor):
            global_numel = math.prod(p.shape)
            padded_local_size = (
                (global_numel + self._world_size - 1) // self._world_size
            )
            return self._device_rank * padded_local_size
        return 0

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched Aurora task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == "aurora"
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Aurora optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Re-read pp_iterations / pp_beta from the group every step so an
            # LR scheduler or warmup can mutate them (matching how lr/mu/etc.
            # are re-read here). Validate to fail fast on bad runtime values.
            pp_iterations = group["pp_iterations"]
            pp_beta = group["pp_beta"]
            if not isinstance(pp_iterations, int) or pp_iterations < 1:
                raise ValueError(
                    f"Invalid pp_iterations: {pp_iterations}. Must be a positive integer."
                )
            if pp_beta < 0.0:
                raise ValueError(f"Invalid pp_beta: {pp_beta}")

            syre_wd = group["syre_wd"]
            advanced_removal = group["advanced_removal"]
            # ``d_bound`` is ``None`` when AR is off (auto-resolution only
            # fires when AR+SYRE are both on); fall back to 0.0 since the
            # kernel won't read it in that branch anyway.
            d_bound_raw = group["d_bound"]
            d_bound = float(d_bound_raw) if d_bound_raw is not None else 0.0
            syre_std = float(group["syre_std"]) if syre_wd else 0.0

            update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=make_aurora_polar(
                    base_polar=self._aurora_base_polar,
                    pp_iterations=pp_iterations,
                    pp_beta=pp_beta,
                    eps=group["epsilon"],
                ),
                cautious_wd=group["cautious_wd"],
                syre_wd=syre_wd,
                syre_std=syre_std,
                advanced_removal=advanced_removal,
                d_bound=d_bound,
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            num_heads = self._resolve_num_heads(group)

            for (_shape, _sharding, _dtype), original_params in shape_groups.items():
                gradients = [p.grad for p in original_params]
                states = [
                    self._get_or_initialize_state(p, "aurora")
                    for p in original_params
                ]
                momentums = [s["momentum"] for s in states]
                params = original_params

                if num_heads is not None:
                    if syre_wd:
                        raise NotImplementedError(
                            "syre_wd=True with num_heads is not supported on "
                            "this branch. The post-orth per-param SYRE state "
                            "would need to follow the head-split mapping; "
                            "left as a v1 limitation."
                        )
                    params, gradients, momentums = self._prepare_head_split(
                        num_heads, params, gradients, momentums
                    )
                    megabatch_args = {**update_args, "process_group": None}
                    shard_dim = None
                else:
                    is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                        self._get_shard_info(params[0], group)
                    )
                    megabatch_args = update_args
                    if is_batch_sharded and not is_matrix_sharded:
                        megabatch_args = {**update_args, "process_group": None}
                    shard_dim = sharded_tensor_dim

                # Per-param SYRE state. Drawn lazily on first use; keyed by
                # the original (pre-head-split) param object so it round-
                # trips through ``state_dict``.
                if syre_wd:
                    syre_seeds1: List[int] = []
                    syre_seeds2: List[int] = []
                    syre_offset_bases: List[int] = []
                    for p in original_params:
                        s1, s2 = self._get_or_init_syre_seeds(
                            p, advanced_removal
                        )
                        syre_seeds1.append(s1)
                        syre_seeds2.append(s2)
                        syre_offset_bases.append(
                            self._compute_syre_offset_base(p)
                        )
                else:
                    syre_seeds1 = None
                    syre_seeds2 = None
                    syre_offset_bases = None

                yield AsyncTask(
                    aurora_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_dim,
                        syre_seeds1=syre_seeds1,
                        syre_seeds2=syre_seeds2,
                        syre_offset_bases=syre_offset_bases,
                        **megabatch_args,
                    )
                )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """AdamW task creation with SYRE support.

        Groups without ``syre_wd=True`` are delegated to the parent's
        standard AdamW path (``torch._fused_adamw_``-backed in
        ``scalar_opts.adamw_update_foreach``). Groups with
        ``syre_wd=True`` are routed through
        :func:`dion.scalar_opts.adamw_update_foreach_syre_async`, which
        runs the AdamW math eagerly so the post-bias-correction update
        direction is available as the cautious-SYRE mask source.
        """
        standard_groups = [g for g in param_groups if not g.get("syre_wd", False)]
        syre_groups = [g for g in param_groups if g.get("syre_wd", False)]

        # Delegate the no-SYRE path so we don't fork the fast fused path.
        yield from super()._create_adamw_tasks(standard_groups)

        for group in syre_groups:
            assert group["algorithm"] == "adamw"
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            advanced_removal = group["advanced_removal"]
            # See note in ``_create_ortho_tasks``: ``d_bound`` may be ``None``
            # when AR is off; the eager AdamW-SYRE path won't touch the AR
            # branch in that case.
            d_bound_raw = group["d_bound"]
            d_bound = float(d_bound_raw) if d_bound_raw is not None else 0.0
            syre_std = float(group["syre_std"])

            syre_seeds1: List[int] = []
            syre_seeds2: List[int] = []
            syre_offset_bases: List[int] = []
            for p in params:
                s1, s2 = self._get_or_init_syre_seeds(p, advanced_removal)
                syre_seeds1.append(s1)
                syre_seeds2.append(s2)
                syre_offset_bases.append(self._compute_syre_offset_base(p))

            yield AsyncTask(
                adamw_update_foreach_syre_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    step=torch.tensor(group["step"]),
                    epsilon=torch.tensor(group["epsilon"]),
                    cautious_wd=group.get("cautious_wd", False),
                    syre_seeds1=syre_seeds1,
                    syre_seeds2=syre_seeds2,
                    syre_std=syre_std,
                    syre_offset_bases=syre_offset_bases,
                    advanced_removal=advanced_removal,
                    d_bound=d_bound,
                )
            )

    def _create_lion_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Refuse ``syre_wd=True`` on Lion groups (not wired).

        ``add_param_group`` already raises if a Lion group is added with
        ``syre_wd=True``, but a user could mutate the group's
        ``syre_wd`` after the fact. Catching it again here keeps the
        "silently no-ops" failure mode closed.
        """
        for group in param_groups:
            if group.get("syre_wd", False):
                raise NotImplementedError(
                    "syre_wd=True with algorithm='lion' is not supported. "
                    "SYRE is wired for algorithm='aurora' and 'adamw' only."
                )
        yield from super()._create_lion_tasks(param_groups)


def aurora_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    cautious_wd: bool = False,
    syre_wd: bool = False,
    syre_std: float = 0.0,
    advanced_removal: bool = False,
    d_bound: float = 0.01,
    syre_seeds1: Optional[List[int]] = None,
    syre_seeds2: Optional[List[int]] = None,
    syre_offset_bases: Optional[List[int]] = None,
) -> Generator[None, None, None]:
    """
    Mega-batched Aurora update. Reuses Muon's pre/post-orthogonalize stages
    and the shared megabatch communication; ``newton_schulz_func`` is the
    Aurora-wrapped polar (see ``make_aurora_polar``).

    When ``syre_wd=False`` the post-orth path is the standard (compiled)
    ``muon_update_post_orthogonalize``. When ``syre_wd=True`` it routes
    through :func:`aurora_update_post_orthogonalize`, which is *not*
    ``torch.compile``'d because the SYRE step launches a Triton kernel
    per parameter.
    """
    N = len(X)
    assert N == len(G) == len(M)

    U = muon_update_pre_orthogonalize(
        G=to_local(G), M=to_local(M), momentum=momentum, nesterov=nesterov,
    )

    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

    if comm_dim is not None:
        if not isinstance(X[0], DTensor):
            raise TypeError(
                "Sharded path requires X[0] to be a DTensor so .shape gives "
                f"the global size; got {type(X[0]).__name__}."
            )
        global_comm_dim_size = X[0].shape[comm_dim]
    else:
        global_comm_dim_size = None

    U = yield from megabatch_orthogonalize_async(
        U,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
        global_comm_dim_size=global_comm_dim_size,
    )

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "aurora_aspect":
        adjusted_lr = adjust_lr_aurora_aspect(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    X_local = to_local(X)

    if syre_wd:
        aurora_update_post_orthogonalize(
            X=X_local,
            U=U,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            syre_seeds1=syre_seeds1,
            syre_seeds2=syre_seeds2,
            syre_std=syre_std,
            syre_offset_bases=syre_offset_bases,
            advanced_removal=advanced_removal,
            d_bound=d_bound,
        )
    else:
        muon_update_post_orthogonalize(
            X=X_local,
            U=U,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
        )


def aurora_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool,
    syre_seeds1: List[int],
    syre_seeds2: List[int],
    syre_std: float,
    syre_offset_bases: List[int],
    advanced_removal: bool,
    d_bound: float,
) -> None:
    """SYRE-aware post-orthogonalize step. Per-param, in order:

      1. SYRE WD step on ``X[i]`` -- cautious-masked by ``U[i]`` when
         ``cautious_wd=True``. Decay coefficient ``gamma = base_lr *
         weight_decay``.
      2. ``X[i].sub_(U[i], alpha=adjusted_lr)`` -- the standard
         post-orth param add, identical to
         :func:`dion.muon.muon_update_post_orthogonalize`'s final step.

    Intentionally not ``@torch.compile``'d: the SYRE step calls a Triton
    kernel per param, which fullgraph compile cannot trace. The non-SYRE
    path stays on the compiled ``muon_update_post_orthogonalize``.

    If SYRE later gains other clients (Muon, NorMuon, ...), the cleaner
    home for this helper is :mod:`dion.muon` alongside the standard
    post-orth, with the foreach fast path and the SYRE branch coexisting
    in one function.
    """
    from .syre import syre_wd_inplace

    gamma = float(base_lr) * float(weight_decay)
    adj_lr_f = float(adjusted_lr)

    # Skip the per-param SYRE launch entirely when gamma == 0 (e.g.
    # weight_decay=0). ``syre_wd_inplace`` early-returns internally,
    # but the per-param Python overhead is still measurable for groups
    # with many small params. Mirrors the guard in
    # ``scalar_opts.adamw_update_foreach_syre``.
    if gamma > 0.0:
        for i, (x, u) in enumerate(zip(X, U)):
            syre_wd_inplace(
                x,
                gamma=gamma,
                seed1=syre_seeds1[i],
                std=syre_std,
                seed2=syre_seeds2[i],
                d_bound=d_bound,
                advanced_removal=advanced_removal,
                offset_base=syre_offset_bases[i],
                U=u if cautious_wd else None,
            )

    for x, u in zip(X, U):
        x.sub_(u, alpha=adj_lr_f)


def make_aurora_polar(
    base_polar: Callable,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: Optional[float] = None,
) -> Callable:
    """
    Build an Aurora-flavored polar function that has the same signature as a
    standard Newton-Schulz / polar function (``func(X, epsilon) -> Tensor``)
    and can be plugged into ``megabatch_orthogonalize_async``.

    For square matrices this is just ``base_polar(X, epsilon)``. For
    non-square matrices it transposes to tall, then runs ``pp_iterations``
    rounds of diagonal row-preconditioning, calling ``base_polar`` once per
    round. Aspect-ratio scaling is left to the optimizer's ``adjust_lr``
    pathway (the same one Muon/NorMuon use), so the output here has
    spectral norm at most 1 and unit row-norm structure.

    Args:
        base_polar: standard polar / Newton-Schulz function.
        pp_iterations: number of preconditioned-polar rounds.
        pp_beta: row-norm diagonal-update exponent.
        eps: Python float used for row-norm / row_sq clamps. If ``None``, the
            wrapper falls back to coercing whatever ``epsilon`` is passed in at
            call time. The optimizer's ``_create_ortho_tasks`` bakes this in
            once per step so the closure does not need to ``float()`` a tensor
            on every call.

    Reference: https://github.com/tilde-research/aurora-release/blob/main/aurora.py
    """
    baked_eps = float(eps) if eps is not None else None

    def aurora_polar(X: Tensor, epsilon=1e-7) -> Tensor:
        m, n = X.size(-2), X.size(-1)

        if m == n:
            U = base_polar(X, epsilon=epsilon)
        else:
            transposed = m < n
            X_t = X.mT if transposed else X
            mm = max(m, n)
            nn = min(m, n)
            # Use a Python float for clamp(min=...) to avoid device-mismatch
            # when ``epsilon`` is a CPU Tensor (the megabatch path). Prefer the
            # value baked in at wrapper-construction time.
            if baked_eps is not None:
                eps_f = baked_eps
            elif isinstance(epsilon, Tensor):
                eps_f = epsilon.item()
            else:
                eps_f = float(epsilon)
            X32 = X_t.to(torch.float32)
            target_row_sq = nn / mm
            row_norm = X32.norm(dim=-1, keepdim=True).clamp(min=eps_f)
            D = 1.0 / row_norm
            eps_sq = eps_f * eps_f
            U = base_polar(D * X32, epsilon=epsilon)
            for k in range(1, pp_iterations):
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp(min=eps_sq)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
                U = base_polar(D * X32, epsilon=epsilon)
            if transposed:
                U = U.mT

        return U

    return aurora_polar
