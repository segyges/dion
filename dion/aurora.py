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
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            num_heads = self._resolve_num_heads(group)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "aurora") for p in params]
                momentums = [s["momentum"] for s in states]

                if num_heads is not None:
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

                yield AsyncTask(
                    aurora_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


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
) -> Generator[None, None, None]:
    """
    Mega-batched Aurora update. Reuses Muon's pre/post-orthogonalize stages
    and the shared megabatch communication; ``newton_schulz_func`` is the
    Aurora-wrapped polar (see ``make_aurora_polar``).
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

    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
    )


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
