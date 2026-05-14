import math

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
# The SYRE config/validation layer is triton-free; the kernel module
# (``dion.syre``) is only loaded lazily inside ``_check_syre_triton_available``
# and inside ``_validate_syre_kwargs`` (for the preset-name error paths).
# ``SyreTheorem3Warning`` and the private helpers are re-exported here so
# existing imports (``from dion.aurora import SyreTheorem3Warning, ...``) keep
# working without depending on the kernel module's triton import.
from .syre_config import (  # noqa: F401  (re-exported)
    SyreTheorem3Warning,
    _check_syre_per_param_sigma0,
    _check_syre_triton_available,
    _resolve_syre_d_bound,
    _validate_syre_kwargs,
)


# Cap on per-group SYRE-metadata cache entries. The working set in normal
# use is bounded by the number of distinct shape sub-groups within one
# parameter group (small). The cap is a safety net for the
# gradient-accumulation / partial-grad case where the
# ``p.grad is not None`` filter toggles subsets across steps. Wholesale-
# clear on overflow rather than LRU-evict to keep this dead simple; next
# call rebuilds.
_SYRE_META_CACHE_MAXSIZE = 16



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
            is refused (not wired). Note: SYRE-AR effectively no-ops under
            bf16 parameter storage (per-step contribution stays sub-ulp);
            see :func:`dion.syre.syre_wd_inplace`'s "Storage-dtype
            precision" section for the full accounting.
        syre_std: Scale of ``theta_0`` (a single positive float, shared
            across all params in the group). Mutually exclusive with
            ``syre_std_mode``. Exactly one of the two must be set when
            ``syre_wd=True``. No empirical default -- if you want to
            couple this to your init magnitude, compute and pass it
            explicitly, or use ``syre_std_mode``.
        syre_std_mode: Per-tensor sigma_0 specification. Either a preset
            name (one of ``"lecun_fan_in"``, ``"kaiming_fan_in"``,
            ``"xavier_normal"``) or a callable ``(p: Tensor) -> float``.
            Resolved per-tensor at param-registration time and cached in
            the optimizer state. Presets follow Ziyin et al. (2024)
            §5.4: ``sigma_0 = 0.01 * (init_std_for_p)``, with each
            preset using a different init scheme's std. For
            ``ndim < 2`` (1D bias / LayerNorm / scalar params) presets
            degrade to ``fan_in = numel, fan_out = 1`` -- a direct
            extension of the matrix formula; this gives a small per-
            element sigma_0 that won't conflict with typical 1D init
            scales. Custom inits (PyTorch Linear bias, LLaMA's 0.02,
            LN gamma=1.0) are out of scope -- use a callable for those.
            Mutually exclusive with ``syre_std``.
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
        syre_std_mode: Optional[Union[str, Callable[[Tensor], float]]] = None,
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
        _validate_syre_kwargs(
            syre_wd, syre_std, advanced_removal, d_bound,
            syre_std_mode=syre_std_mode,
        )
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
            syre_std_mode=syre_std_mode,
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
        syre_std_mode = param_group.get(
            "syre_std_mode", self.defaults["syre_std_mode"]
        )
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
        _validate_syre_kwargs(
            syre_wd, syre_std, advanced_removal, d_bound,
            syre_std_mode=syre_std_mode,
        )
        d_bound = _resolve_syre_d_bound(
            syre_wd, syre_std, advanced_removal, d_bound,
        )
        # Write the resolved value back so the group dict the base class
        # ends up storing has the auto-resolved float, not ``None``.
        param_group["d_bound"] = d_bound
        # Ensure the new group dict carries ``syre_std_mode`` even if the
        # caller didn't set it -- this keeps ``group.get("syre_std_mode")``
        # at step time consistent with construction-time groups.
        param_group["syre_std_mode"] = syre_std_mode

        algorithm = param_group.get("algorithm", self.defaults["algorithm"])
        if syre_wd and algorithm == "lion":
            raise NotImplementedError(
                "syre_wd=True with algorithm='lion' is not supported. "
                "SYRE is wired for algorithm='aurora' and 'adamw' only."
            )
        if syre_wd:
            _check_syre_triton_available()
        super().add_param_group(param_group)

    def _get_or_init_syre_std(
        self,
        p: Tensor,
        syre_std: Optional[float],
        syre_std_mode: Optional[Union[str, Callable[[Tensor], float]]],
    ) -> float:
        """Lazily resolve and cache the per-param sigma_0 for SYRE.

        When the group uses scalar ``syre_std``, the result is just that
        float (still cached so the step-time hot path doesn't recompute
        ``float(group["syre_std"])`` on every iteration). When the group
        uses ``syre_std_mode``, the preset / callable is invoked once
        per param and the result cached -- so callables that read
        e.g. ``torch.nn.init`` metadata aren't paying that cost every
        step.

        Cache lives at ``self.state[p]["syre_std"]`` and round-trips
        through ``state_dict``, so resume reproduces the exact same
        per-param sigma_0 sequence even if the user changes the preset
        between runs (the cached value wins). To force re-resolution
        after a config change, delete ``state[p]["syre_std"]``.
        """
        s = self.state[p]
        if "syre_std" not in s:
            if syre_std_mode is not None:
                from .syre import resolve_syre_std
                value = resolve_syre_std(p, syre_std_mode)
            elif syre_std is not None:
                value = float(syre_std)
            else:
                # Validation has already rejected this combo; defensive.
                raise RuntimeError(
                    "SYRE state init: neither syre_std nor syre_std_mode "
                    "is set. This is a bug -- _validate_syre_kwargs "
                    "should have refused this group earlier."
                )
            s["syre_std"] = float(value)
        return float(s["syre_std"])

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

    def _get_or_build_syre_meta(
        self,
        group: dict,
        params: List[Tensor],
        advanced_removal: bool,
        syre_std_scalar: Optional[float],
        syre_std_mode: Optional[Union[str, Callable[[Tensor], float]]],
    ) -> Tuple[List[int], List[int], List[int], List[float]]:
        """Build (or fetch from cache) the per-param SYRE metadata for a
        sub-group of params.

        Returns ``(seeds1, seeds2, offset_bases, stds)``. The underlying
        values are already cached on ``self.state[p]`` (seeds, std) or
        derived from stable topology (offset_base), so for a given param
        identity tuple the result is invariant across steps. The cache
        here just avoids the per-step N dict-lookups + method-call chain
        in the hot path.

        Cache lives on the group dict under ``_syre_meta_cache`` as
        ``{id_tuple: (seeds1, seeds2, offset_bases, stds)}``. Keyed by
        ``tuple(id(p) for p in params)`` so different shape sub-groups
        within one parameter group each get their own entry; growth is
        bounded in normal use by the number of distinct shape sub-groups.

        A wholesale-clear cap (``_SYRE_META_CACHE_MAXSIZE``) guards
        against degenerate growth when the ``p.grad is not None`` filter
        toggles across steps (gradient accumulation, conditional
        freezing): each distinct filtered subset would otherwise accrue
        a new entry. On overflow we clear and rebuild from scratch.

        Safe under `id` reuse: optimizer parameters are not freed during
        a live optimizer, and ``add_param_group`` adds (never removes)
        entries -- so an id collision would require a parameter to die
        and a new tensor be allocated at the same address while the
        optimizer is still tracking the old one. We don't try to defend
        against that; ``optimizer.state_dict() / load_state_dict()`` is
        the supported way to reset.
        """
        key = tuple(id(p) for p in params)
        cache = group.setdefault("_syre_meta_cache", {})
        hit = cache.get(key)
        if hit is not None:
            return hit
        if len(cache) >= _SYRE_META_CACHE_MAXSIZE:
            cache.clear()
        seeds1: List[int] = []
        seeds2: List[int] = []
        offset_bases: List[int] = []
        stds: List[float] = []
        for p in params:
            s1, s2 = self._get_or_init_syre_seeds(p, advanced_removal)
            seeds1.append(s1)
            seeds2.append(s2)
            offset_bases.append(self._compute_syre_offset_base(p))
            stds.append(
                self._get_or_init_syre_std(p, syre_std_scalar, syre_std_mode)
            )
        meta = (seeds1, seeds2, offset_bases, stds)
        cache[key] = meta
        return meta

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
            syre_std_scalar = group["syre_std"]
            syre_std_mode = group.get("syre_std_mode")

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

                # Per-param SYRE state. Cached per shape sub-group on the
                # group dict -- seeds / stds are already lazily drawn into
                # ``self.state[p]`` (round-tripping through ``state_dict``)
                # and the offset_base is derived from stable topology; the
                # cache avoids rebuilding the four lists every step.
                if syre_wd:
                    syre_seeds1, syre_seeds2, syre_offset_bases, syre_stds = (
                        self._get_or_build_syre_meta(
                            group, original_params, advanced_removal,
                            syre_std_scalar, syre_std_mode,
                        )
                    )
                    # When ``syre_std_mode`` is in use, the per-param
                    # sigma_0 set is only known once params are seen; fire
                    # the Theorem-3 / AR-floor warning at most once per
                    # group via a flag on the group dict so we don't
                    # re-check every step.
                    if (
                        syre_std_mode is not None
                        and not group.get("_syre_sigma0_checked", False)
                    ):
                        _check_syre_per_param_sigma0(
                            syre_stds, advanced_removal,
                            group["d_bound"],
                        )
                        group["_syre_sigma0_checked"] = True
                else:
                    syre_seeds1 = None
                    syre_seeds2 = None
                    syre_offset_bases = None
                    syre_stds = None

                yield AsyncTask(
                    aurora_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_dim,
                        syre_seeds1=syre_seeds1,
                        syre_seeds2=syre_seeds2,
                        syre_offset_bases=syre_offset_bases,
                        syre_stds=syre_stds,
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
            syre_std_scalar = group["syre_std"]
            syre_std_mode = group.get("syre_std_mode")

            # Per-param SYRE metadata, cached on the group dict. See
            # ``_get_or_build_syre_meta`` for the cache contract.
            syre_seeds1, syre_seeds2, syre_offset_bases, syre_stds = (
                self._get_or_build_syre_meta(
                    group, params, advanced_removal,
                    syre_std_scalar, syre_std_mode,
                )
            )
            if (
                syre_std_mode is not None
                and not group.get("_syre_sigma0_checked", False)
            ):
                _check_syre_per_param_sigma0(
                    syre_stds, advanced_removal, group["d_bound"],
                )
                group["_syre_sigma0_checked"] = True

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
                    syre_stds=syre_stds,
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
    advanced_removal: bool = False,
    d_bound: float = 0.01,
    syre_seeds1: Optional[List[int]] = None,
    syre_seeds2: Optional[List[int]] = None,
    syre_offset_bases: Optional[List[int]] = None,
    syre_stds: Optional[List[float]] = None,
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
            syre_stds=syre_stds,
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
    syre_stds: List[float],
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
    from .syre import syre_wd_multi_inplace

    gamma = float(base_lr) * float(weight_decay)
    adj_lr_f = float(adjusted_lr)

    # Single fused SYRE launch over the whole list. On sharded clusters
    # (W >= 4) this is where the win is -- per-param launch overhead
    # would otherwise plateau the SYRE step at ~N * 18us. See
    # ``scripts/benchmark_syre.py``.
    if gamma > 0.0:
        syre_wd_multi_inplace(
            Xs=list(X),
            gamma=gamma,
            seeds1=syre_seeds1,
            std=syre_stds,
            seeds2=syre_seeds2,
            d_bound=d_bound,
            advanced_removal=advanced_removal,
            offset_bases=syre_offset_bases,
            Us=list(U) if cautious_wd else None,
        )

    # Multi-tensor apply: one fused launch over the list instead of N
    # python-side ``sub_`` dispatches. Semantically identical.
    torch._foreach_sub_(list(X), list(U), alpha=adj_lr_f)


@torch.compile(fullgraph=True)
def _aurora_pp_init(X_t: Tensor, eps_f: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Aurora preconditioned-polar setup, compile-fused.

    Returns ``(X32, D, D * X32)`` where ``D = 1 / row_norm(X32).clamp(eps_f)``.
    Caller passes ``D * X32`` to ``base_polar`` and keeps ``X32`` / ``D``
    alive across the iteration loop so :func:`_aurora_pp_step` can update
    ``D`` and reform ``D * X32`` without re-casting.

    Four eager launches (``to``, ``norm``, ``clamp``, ``reciprocal``,
    ``mul``) collapse to one fused kernel under compile.
    """
    X32 = X_t.to(torch.float32)
    row_norm = X32.norm(dim=-1, keepdim=True).clamp(min=eps_f)
    D = 1.0 / row_norm
    return X32, D, D * X32


@torch.compile(fullgraph=True)
def _aurora_pp_step(
    U: Tensor,
    X32: Tensor,
    D: Tensor,
    target_row_sq: float,
    pp_beta: float,
    eps_sq: float,
) -> Tuple[Tensor, Tensor]:
    """Aurora preconditioned-polar iteration body, compile-fused.

    Given the prior iteration's ``U`` and the cached ``X32`` / ``D``,
    updates ``D *= (target_row_sq / ||U_i||^2)^pp_beta`` and returns
    ``(D_new, D_new * X32)`` for the next ``base_polar`` call.

    Six pointwise launches (``to``, ``pow(2)``, ``sum``, ``clamp``,
    ``div``, ``pow(pp_beta)``, ``mul``, ``mul``) collapse to one fused
    kernel under compile. Specializes per ``(target_row_sq, pp_beta,
    eps_sq)`` Python scalar combo, which is per-shape-stable.
    """
    row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp(min=eps_sq)
    D = D * (target_row_sq / row_sq).pow(pp_beta)
    return D, D * X32


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

    The pointwise interludes between ``base_polar`` calls are
    compile-fused via :func:`_aurora_pp_init` (pre-loop setup) and
    :func:`_aurora_pp_step` (per-iteration diagonal update). The
    ``base_polar`` calls themselves stay outside the compile boundary
    because they may dispatch to a Triton kernel.

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
            return base_polar(X, epsilon=epsilon)

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
        target_row_sq = nn / mm
        eps_sq = eps_f * eps_f

        X32, D, DX = _aurora_pp_init(X_t, eps_f)
        U = base_polar(DX, epsilon=epsilon)
        for _ in range(1, pp_iterations):
            D, DX = _aurora_pp_step(
                U, X32, D, target_row_sq, pp_beta, eps_sq,
            )
            U = base_polar(DX, epsilon=epsilon)
        if transposed:
            U = U.mT
        return U

    return aurora_polar
