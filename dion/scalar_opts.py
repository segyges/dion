import torch
from torch import Tensor
from typing import Generator, List


@torch.compile(fullgraph=True)
def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """
    AdamW optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    if cautious_wd:
        # Compute update direction (pre-LR) for CWD mask
        update_dir = M / denom

        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (update_dir * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)


@torch.compile(fullgraph=True)
def lion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    G = G.to(M.dtype)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = M.lerp(G, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    M.lerp_(G, 1 - beta2)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (U * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)


_step_tensor_cache: dict = {}


def _get_step_tensor(device: torch.device) -> Tensor:
    t = _step_tensor_cache.get(device)
    if t is None:
        t = torch.zeros((), dtype=torch.float32, device=device)
        _step_tensor_cache[device] = t
    return t


def adamw_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor or float)
    beta1: Tensor,  # Beta 1 (scalar tensor or float)
    beta2: Tensor,  # Beta 2 (scalar tensor or float)
    weight_decay: Tensor,  # Weight decay (scalar tensor or float)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """AdamW update for a list of tensors.

    Dispatches through ``torch._fused_adamw_``, which is already a
    multi-tensor-apply kernel; avoids the ~6-call ``torch._foreach_*`` chain
    that otherwise dispatches one aten op per tensor on the CPU side.

    Cautious weight decay (https://arxiv.org/pdf/2510.12402) is applied as a
    post-step correction rather than inside the kernel:

        X_std   = X - lr*wd*X - update                    (standard AdamW)
        X_cwd   = X - lr*wd*X*mask - update               (CWD)
                = X_std + lr*wd*X*(1 - mask)

    where ``mask = (sign(M_new · X_orig) >= 0)``. We add back the decay that
    was over-applied on elements where momentum and param disagree in sign.
    """
    if not X:
        return
    n = len(X)
    assert n == len(G) == len(M) == len(V)

    lr_f = float(lr)
    beta1_f = float(beta1)
    beta2_f = float(beta2)
    wd_f = float(weight_decay)
    eps_f = float(epsilon)

    do_cwd_correction = cautious_wd and wd_f > 0.0
    if do_cwd_correction:
        X_orig = [x.clone() for x in X]

    # Cache the step scalar per device. ``torch.tensor(x, device="cuda")``
    # from a Python float stages through pageable CPU memory and issues a
    # blocking ``cudaMemcpy``, which defeats the point of going fused.
    # ``fill_`` on a cached 0-d CUDA tensor is a kernel launch — async.
    step_t = _get_step_tensor(X[0].device)
    step_t.fill_(float(step))
    torch._fused_adamw_(
        X, G, M, V, [],
        [step_t] * n,
        amsgrad=False,
        beta1=beta1_f, beta2=beta2_f,
        lr=lr_f, weight_decay=wd_f, eps=eps_f,
        maximize=False,
    )

    if do_cwd_correction:
        # mask == 0  <=>  sign(M_new) * sign(X_orig) < 0  (over-decayed).
        signs = torch._foreach_mul(M, X_orig)
        undo_masks = [(s < 0).to(x.dtype) for s, x in zip(signs, X_orig)]
        correction = torch._foreach_mul(X_orig, undo_masks)
        torch._foreach_mul_(correction, lr_f * wd_f)
        torch._foreach_add_(X, correction)


@torch.compile(fullgraph=True)
def lion_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)

    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = torch._foreach_lerp(M, G, [1 - beta1] * batch_size)
    torch._foreach_sign_(U)

    # Update momentum in place with new gradient
    # M = beta2 * M + (1 - beta2) * G
    torch._foreach_lerp_(M, G, [1 - beta2] * batch_size)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    torch._foreach_mul_(U, lr)
    torch._foreach_sub_(X, U)


def adamw_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    adamw_update_foreach(
        X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd
    )
    yield


@torch.compile(fullgraph=True)
def _adamw_syre_premath(
    M: List[Tensor],
    V: List[Tensor],
    G: List[Tensor],
    beta1: float,
    beta2: float,
    bc2_sqrt: Tensor,
    epsilon: float,
) -> List[Tensor]:
    """Pre-SYRE AdamW pointwise math, compile-fused.

    Mutates ``M`` and ``V`` in place; returns the bias-corrected update
    direction ``update_dirs = M / (sqrt(V)/sqrt(bc2) + eps)``. Caller is
    responsible for casting ``G`` to ``M``'s dtype (see the
    short-circuit in :func:`adamw_update_foreach_syre`) so this helper
    sees a single-dtype list.

    The seven pointwise launches (lerp_, mul_, addcmul_, sqrt, div_,
    add_, div) collapse to one fused multi-tensor kernel under compile.

    ``bc2_sqrt`` is a 0-D Tensor, not a Python float: under
    ``fullgraph=True`` Dynamo still re-specializes on changing scalar
    values (the value enters guards even with
    ``specialize_float=False``), so passing the per-step-varying bias
    correction as a float triggers recompilation each call and trips
    the recompile limit within ~10 steps. Wrapping as a 0-D tensor
    routes the value through the data-dependent path, where Dynamo
    sees a stable tensor type and the value enters via runtime memory
    read. ``beta1`` / ``beta2`` / ``epsilon`` stay as floats: they're
    fixed per param-group and won't change call-to-call.

    List length specializes per call; Aurora groups params by shape+
    dtype on the ortho path. On the AdamW path the list length is the
    full group size, also stable across steps.
    """
    torch._foreach_lerp_(M, G, 1.0 - beta1)
    torch._foreach_mul_(V, beta2)
    torch._foreach_addcmul_(V, G, G, value=1.0 - beta2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bc2_sqrt)
    torch._foreach_add_(denom, epsilon)
    return torch._foreach_div(M, denom)


def adamw_update_foreach_syre(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool,
    syre_seeds1: List[int],
    syre_seeds2: List[int],
    syre_stds: List[float],
    syre_offset_bases: List[int],
    advanced_removal: bool,
    d_bound: float,
):
    """AdamW step with SYRE weight decay applied per param.

    Not ``torch._fused_adamw_``-backed because we need the per-param
    update direction ``update_dir = M_new / denom`` exposed to the
    cautious-SYRE Triton kernel, which gates the SYRE diff by
    ``(update_dir * X >= 0)``. The non-SYRE AdamW path stays on the
    faster fused kernel in :func:`adamw_update_foreach`.

    The AdamW pointwise math around the Triton kernel call IS compiled,
    via :func:`_adamw_syre_premath` -- seven multi-tensor launches
    collapse to one fused kernel. Only the SYRE Triton launch and the
    final param add stay outside the compile boundary.

    Replaces the AdamW decoupled weight-decay step with the SYRE pull
    ``X <- X - lr*wd*(X - theta_0)`` (optionally cautious-masked,
    optionally with SYRE-AR). Order of ops per call:

      1. ``M``/``V`` update in place (standard AdamW); compute
         ``denom = sqrt(V/bc2) + eps`` and ``update_dir = M / denom``.
         (All fused into :func:`_adamw_syre_premath`.)
      2. SYRE WD step on ``X`` -- replaces the standard
         ``X.mul_(1 - lr*wd)``. The kernel reads the pre-update ``X``
         for the cautious mask.
      3. ``X.sub_(update_dir, alpha=lr/bc1)`` -- standard AdamW
         gradient step.

    The cautious-SYRE branch matches segyges/aurora's AdamW+SYRE form:
    mask source is the bias-corrected update direction, not the SYRE
    diff itself (CWD applied to SYRE).
    """
    from .syre import syre_wd_multi_inplace

    if not X:
        return
    n = len(X)
    assert n == len(G) == len(M) == len(V)
    assert n == len(syre_seeds1) == len(syre_seeds2) == len(syre_offset_bases)

    lr_f = float(lr)
    beta1_f = float(beta1)
    beta2_f = float(beta2)
    wd_f = float(weight_decay)
    eps_f = float(epsilon)
    step_i = int(step)

    bc1 = 1.0 - beta1_f**step_i
    bc2 = 1.0 - beta2_f**step_i
    adj_lr_f = lr_f / bc1
    gamma = lr_f * wd_f
    # 0-D tensor (not Python float) -- see ``_adamw_syre_premath`` for
    # why: a per-step-varying float arg trips Dynamo recompilation
    # under ``fullgraph=True`` regardless of ``specialize_float``.
    bc2_sqrt = torch.tensor(bc2**0.5, device=M[0].device, dtype=torch.float32)

    # Cast G to M's dtype, but short-circuit when dtypes already match
    # (the common pure-bf16 and pure-fp32 cases) -- ``.to(same_dtype)``
    # is a no-op copy but still costs an aten dispatch per param. Under
    # standard autograd ``G[i].dtype == M[i].dtype`` per index (M
    # initializes from p, G is p.grad, both follow p's dtype), so
    # checking the head pair is a safe heuristic for the AdamW path
    # where the param group isn't explicitly bucketed by dtype.
    if G[0].dtype == M[0].dtype:
        G_cast = G
    else:
        G_cast = [g.to(m.dtype) for g, m in zip(G, M)]

    # Compile-fused AdamW pointwise math: M/V update + denom + update_dirs.
    # See :func:`_adamw_syre_premath`.
    update_dirs = _adamw_syre_premath(
        M, V, G_cast, beta1_f, beta2_f, bc2_sqrt, eps_f,
    )

    # SYRE WD step. Single fused Triton launch over the list -- see
    # ``syre_wd_multi_inplace`` and ``scripts/benchmark_syre.py``.
    # ``gamma == 0`` skips the call entirely (common ``weight_decay=0``
    # case avoids host overhead and the metadata-table construction).
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
            Us=list(update_dirs) if cautious_wd else None,
        )

    # X = X - adj_lr * update_dir.
    torch._foreach_add_(X, update_dirs, alpha=-adj_lr_f)


def adamw_update_foreach_syre_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool,
    syre_seeds1: List[int],
    syre_seeds2: List[int],
    syre_stds: List[float],
    syre_offset_bases: List[int],
    advanced_removal: bool,
    d_bound: float,
) -> Generator[None, None, None]:
    adamw_update_foreach_syre(
        X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon,
        cautious_wd, syre_seeds1, syre_seeds2, syre_stds, syre_offset_bases,
        advanced_removal, d_bound,
    )
    yield


def lion_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay, cautious_wd)
    yield
