"""SYRE weight decay (Ziyin et al., ICLR 2025).

Per-tensor SYRE / SYRE-AR weight decay, with an optional cautious mask.
Triton-backed; this module imports ``triton`` at module scope and will
raise ``ImportError`` on import in environments without it. Callers in
:mod:`dion.aurora` gate the import so users without triton can still
construct optimizers that do not request SYRE.

Math:

* **Basic SYRE.** ``theta <- theta - gamma * (theta - theta_0)`` where
  ``theta_0 = randn(seed1, global_offset) * std`` is regenerated from a
  stored per-parameter PRNG seed each step (no memory overhead for the
  target). The pull is element-wise; correctness on sharded params
  requires every element across the cluster to receive a unique global
  Philox offset, which the caller arranges via ``offset_base``.
* **SYRE-AR (advanced removal).** Multiplies ``(theta - theta_0)`` by
  ``d ~ Uniform(1 - d_bound, 1 + d_bound)`` (per element, keyed by
  ``seed2``) before the decay step. Breaks the continuous symmetries
  that basic SYRE / L2 cannot resolve, by giving each coordinate a
  slightly different decay coefficient. Implementation note:
  computed in *additive form* as ``diff + diff*xi`` with
  ``xi = d - 1 ~ Uniform(-d_bound, +d_bound)``, never materializing
  ``(1 + xi)`` in fp32. See "Pigeonhole avoidance" below.
* **Cautious SYRE.** Gates the SYRE diff element-wise by the
  cautious-WD mask ``(u * theta >= 0)``, where ``u`` is the post-orth
  update tensor that the optimizer is about to subtract from ``theta``
  (modulo ``lr``). The mask uses the *update direction*, not
  ``(theta - theta_0)``; this is "cautious WD applied to SYRE",
  matching the cautious-WD form already shipped in
  :func:`dion.muon.muon_update_post_orthogonalize`. SYRE-AR composes by
  multiplying the diff by ``d`` *before* masking.

The decay coefficient ``gamma`` is computed by the caller. Aurora's
contract is ``gamma = lr * weight_decay``.

Pigeonhole avoidance (additive form)
------------------------------------

The paper requires ``D_ii`` (i.e. ``1 + xi_i``) all distinct. A naive
implementation forms ``d = 1 + xi`` in fp32 and multiplies
``diff * d``. fp32 ulp at 1.0 is ~1.2e-7, so the window of
representable ``(1 + xi)`` values has only ``2*d_bound / 1.2e-7``
distinct fp32 levels -- a few hundred at typical ``d_bound`` ~ 3e-5,
far fewer than the millions of elements in a real layer. By
pigeonhole this produces massive collisions, and ~half the elements
get ``d`` rounded to *exactly* 1.0 (giving them zero AR contribution
per step, not merely sub-ulp).

We instead compute the algebraically-identical ``diff * (1 + xi)``
as ``diff + diff * xi``, never forming ``(1 + xi)``:

* ``xi`` lives in ``[-d_bound, +d_bound]`` near 0. fp32 ulp near
  ``d_bound ~ 3e-5`` is ~1.2e-12, so the window admits ~5e7 distinct
  fp32 values -- enough to keep ``xi_i`` distinct per element well
  past typical layer sizes.
* ``diff * xi`` is a multiplication, which preserves the relative
  precision of ``xi`` (rounding happens at the product's magnitude,
  not near 1.0).
* ``diff + diff*xi``: ``diff*xi`` is added at ``diff``'s magnitude,
  where ulp is much smaller than ulp(1.0). The contribution survives.

The mathematical distribution of the implicit ``D`` matrix is
identical to the naive form; the rearrangement only changes which
intermediate values the fp32 representation has to store.

Storage-dtype precision
-----------------------

The kernel does all arithmetic in fp32 (``theta_f32 = theta.to(fp32)``)
but writes the result back to the parameter's storage dtype. The
storage dtype -- *not* the gradient dtype, autocast context, or
anything else upstream -- is what determines whether SYRE's per-step
contribution survives.

Rough order-of-magnitude (with ``lr=3e-4``, ``wd=0.1``,
``sigma_0 = 0.01/sqrt(d)``, weight magnitude ``~0.03``):

* **fp32 storage** (incl. fp32-master / bf16-grad mixed precision):
  basic SYRE contributes ~250 fp32-ulps per step, AR contributes
  ~8e-3 fp32-ulps per step. Basic SYRE registers cleanly; AR is
  sub-ulp per step but accumulates deterministically (the per-element
  AR multiplier is fixed by Philox seed, not redrawn step-to-step),
  reaching ~80 ulps over one WD half-life. **Both behave as the paper
  intends.**

* **bf16 storage** (pure-bf16 training, 8-bit optimizers holding
  bf16 params): basic SYRE contributes ~4e-3 bf16-ulps per step --
  still accumulates over long horizons under round-to-nearest-even,
  but at heavily attenuated effective ``gamma``. AR contributes
  ~1e-7 bf16-ulps per step and **never crosses 1 ulp at realistic
  step counts** -- it is effectively a no-op. If you need SYRE/AR to
  behave as documented, use fp32 master weights or stochastic
  rounding on the writeback.

Numbers scale predictably with layer init magnitude and ``gamma``;
AR's signal-to-basic-SYRE ratio is ``d_bound ~ sigma_0``, which is
why bf16's ~7-bit relative ulp swallows AR regardless of ``gamma``.
"""

import math
from typing import Callable, List, Optional, Sequence, Union

import torch
import triton
import triton.language as tl
from torch import Tensor


# Per-tensor sigma_0 ("syre_std") preset resolvers. Each takes a parameter
# tensor and returns the recommended sigma_0 for it, following Ziyin et al.
# (2024) §5.4: "sigma_0 = 0.01/sqrt(d), where 1/d is the common initialization
# variance" -- i.e. sigma_0 = 0.01 * (init std). The presets differ only in
# which init scheme's std they use:
#
#   lecun_fan_in:   init std = 1/sqrt(fan_in)            (paper-strict; var=1/d)
#   kaiming_fan_in: init std = sqrt(2/fan_in)            (He-normal, ReLU/GELU)
#   xavier_normal:  init std = sqrt(2/(fan_in+fan_out))  (Glorot normal)
#
# Fan-in / fan-out follow PyTorch's nn.init conventions for ndim >= 2:
#   fan_in  = prod(shape[1:])   (= shape[-1] for 2D, = in*k*k for conv)
#   fan_out = shape[0]
# ndim < 2 is rejected -- fan-in is undefined for scalars/vectors. Users who
# want SYRE on bias / LayerNorm parameters should pass a callable instead
# (e.g. ``lambda p: 0.01 * <known_init_std_for_p>``).
def _fan_in_fan_out(p: Tensor) -> tuple:
    if p.ndim < 2:
        raise ValueError(
            f"SYRE preset requires ndim >= 2 (got shape {tuple(p.shape)}). "
            "Fan-in is undefined for scalars/vectors. Pass a callable to "
            "syre_std_mode for full control, e.g. "
            "syre_std_mode=lambda p: 0.01 * <your_init_std_for_p>."
        )
    fan_in = 1
    for d in p.shape[1:]:
        fan_in *= int(d)
    fan_out = int(p.shape[0])
    return fan_in, fan_out


def _preset_lecun_fan_in(p: Tensor) -> float:
    fan_in, _ = _fan_in_fan_out(p)
    return 0.01 / math.sqrt(fan_in)


def _preset_kaiming_fan_in(p: Tensor) -> float:
    fan_in, _ = _fan_in_fan_out(p)
    return 0.01 * math.sqrt(2.0 / fan_in)


def _preset_xavier_normal(p: Tensor) -> float:
    fan_in, fan_out = _fan_in_fan_out(p)
    return 0.01 * math.sqrt(2.0 / (fan_in + fan_out))


# Public name -> resolver mapping. Kept as a plain dict so callers can
# enumerate the supported preset names (e.g. for error messages).
SYRE_STD_PRESETS: dict = {
    "lecun_fan_in": _preset_lecun_fan_in,
    "kaiming_fan_in": _preset_kaiming_fan_in,
    "xavier_normal": _preset_xavier_normal,
}


def resolve_syre_std(p: Tensor, mode: Union[str, Callable[[Tensor], float]]) -> float:
    """Resolve a per-tensor sigma_0 from a ``syre_std_mode`` spec.

    ``mode`` is either a preset name (string in :data:`SYRE_STD_PRESETS`)
    or a callable taking the parameter and returning a positive float.
    Validates that the resolved value is a finite positive float.
    """
    if isinstance(mode, str):
        fn = SYRE_STD_PRESETS.get(mode)
        if fn is None:
            raise ValueError(
                f"Unknown syre_std_mode preset: {mode!r}. "
                f"Known presets: {sorted(SYRE_STD_PRESETS)}."
            )
    elif callable(mode):
        fn = mode
    else:
        raise TypeError(
            f"syre_std_mode must be a preset name (str) or a callable, "
            f"got {type(mode).__name__}: {mode!r}"
        )
    value = fn(p)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"syre_std_mode resolver returned non-numeric "
            f"{type(value).__name__}: {value!r}"
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"syre_std_mode resolver returned non-positive or non-finite "
            f"value {value!r} for param of shape {tuple(p.shape)}"
        )
    return value


# Storage-dtype dispatch for the multi-tensor kernel. Per-call all
# tensors must share dtype (validated in the wrapper); the dtype is
# baked in as a ``tl.constexpr`` so the kernel specializes per dtype.
_TORCH_TO_TL = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _syre_wd_kernel(
    X_ptr, U_ptr, numel, gamma, seed1, std, seed2, d_bound, offset_base,
    BLOCK_SIZE: tl.constexpr,
    ADVANCED_REMOVAL: tl.constexpr,
    CAUTIOUS: tl.constexpr,
):
    """SYRE WD step (all four variants).

    Variants are selected by the ``ADVANCED_REMOVAL`` and ``CAUTIOUS``
    ``tl.constexpr`` flags; the compiler specializes per (flag, flag)
    combination at JIT time, so dead branches cost nothing at runtime
    and ``U_ptr`` is dereferenced *only* when ``CAUTIOUS=True``. Pass
    any valid pointer (e.g. ``X_ptr`` itself) for ``U_ptr`` in the
    non-cautious case; the load is dead-code-eliminated.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    theta = tl.load(X_ptr + offsets, mask=mask)
    theta_f32 = theta.to(tl.float32)

    global_offsets = offsets + offset_base
    theta_0 = tl.randn(seed1, global_offsets) * std
    diff = theta_f32 - theta_0

    if ADVANCED_REMOVAL:
        # Additive form: compute diff*(1+xi) as diff + diff*xi rather
        # than forming (1+xi) explicitly. xi lives near 0 with full
        # fp32 precision (~1e7 distinct values in our window at typical
        # d_bound); (1+xi) computed explicitly would round many distinct
        # xi to the same fp32 value near 1.0 (ulp ~1.2e-7), collapsing
        # AR to zero for ~half the elements. See module docstring
        # "Pigeonhole avoidance" section for the full derivation.
        xi = tl.rand(seed2, global_offsets) * (2.0 * d_bound) - d_bound
        diff = diff + diff * xi

    if CAUTIOUS:
        u_f32 = tl.load(U_ptr + offsets, mask=mask).to(tl.float32)
        diff = tl.where(u_f32 * theta_f32 >= 0.0, diff, 0.0)

    result = theta_f32 - gamma * diff
    tl.store(X_ptr + offsets, result.to(theta.dtype), mask=mask)


def syre_wd_inplace(
    X: Tensor,
    gamma: float,
    seed1: int,
    std: float,
    seed2: int = 0,
    d_bound: float = 0.01,
    advanced_removal: bool = False,
    offset_base: int = 0,
    U: Optional[Tensor] = None,
):
    """Apply SYRE weight decay in-place to a single tensor.

    ``theta <- theta - gamma * (theta - theta_0)``, with optional
    advanced-removal multiplier on the diff. ``theta_0`` is generated
    deterministically from ``(seed1, offsets + offset_base)`` via
    Triton's Philox; the same ``seed1`` and ``offset_base`` always
    produce the same ``theta_0`` sequence, which is what makes seeds
    round-trippable through ``state_dict``.

    Args:
        X: parameter tensor (decayed in place). Any CUDA tensor is
            accepted; non-contiguous tensors (e.g. ``channels_last``)
            are temporarily contiguified and copied back.
        gamma: scalar decay coefficient (``lr * weight_decay`` in
            Aurora's contract).
        seed1: PRNG seed for ``theta_0`` (per-parameter, stored in
            optimizer state).
        std: scale for ``theta_0``.
        seed2: PRNG seed for the advanced-removal multiplier. Ignored
            unless ``advanced_removal=True``; pass anything (e.g. 0)
            otherwise.
        d_bound: half-width of the advanced-removal uniform interval.
            Ignored unless ``advanced_removal=True``.
        advanced_removal: SYRE-AR variant -- multiplies the per-element
            ``(theta - theta_0)`` by an ``Uniform(1 - d_bound, 1 +
            d_bound)`` factor before the decay step.
        offset_base: starting global offset for the PRNG element
            indexing. For sharded params, set this to
            ``device_rank * padded_local_size`` so every element across
            the cluster gets a unique offset. Default 0 (replicated /
            single-process).
        U: optional update tensor (same shape as ``X``). When provided,
            the cautious mask ``(U * X >= 0)`` gates the SYRE diff
            element-wise. When ``None`` (default), basic SYRE is
            applied. ``U`` is read but not modified.
    """
    numel = X.numel()
    if numel == 0 or gamma == 0.0:
        return
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    needs_copy_back = not X.is_contiguous()
    if needs_copy_back:
        X_contig = X.contiguous()
    else:
        X_contig = X
    X_flat = X_contig.view(-1)

    cautious = U is not None
    # ``reshape(-1)`` safely flattens a possibly-non-contiguous U (copies
    # if needed). When U is None we pass X_flat as a dummy pointer; the
    # ``CAUTIOUS=False`` constexpr branch never dereferences it.
    U_flat = U.reshape(-1) if cautious else X_flat

    with torch.cuda.device(X.device):
        _syre_wd_kernel[grid](
            X_flat, U_flat, numel,
            gamma, seed1, std, seed2, d_bound, offset_base,
            BLOCK_SIZE=BLOCK_SIZE,
            ADVANCED_REMOVAL=advanced_removal,
            CAUTIOUS=cautious,
        )

    if needs_copy_back:
        X.copy_(X_contig)


@triton.jit
def _syre_wd_multi_kernel(
    X_ptrs_ptr,           # *int64, length n_params, raw addresses of X[i].view(-1)
    U_ptrs_ptr,           # *int64, length n_params, raw addresses of U[i] (or dummies)
    numels_ptr,           # *int64, length n_params
    seeds1_ptr,           # *int32, length n_params
    seeds2_ptr,           # *int32, length n_params
    offset_bases_ptr,     # *int64, length n_params
    stds_ptr,             # *fp32,  length n_params (per-param sigma_0)
    block_to_param_ptr,   # *int32, length total_blocks
    block_within_ptr,     # *int32, length total_blocks
    gamma, d_bound,
    BLOCK_SIZE: tl.constexpr,
    ADVANCED_REMOVAL: tl.constexpr,
    CAUTIOUS: tl.constexpr,
    X_STORAGE_DTYPE: tl.constexpr,
    U_STORAGE_DTYPE: tl.constexpr,
):
    """Multi-tensor SYRE WD kernel.

    One program per (param_idx, block_within_param) pair. ``program_id``
    is a flat index over all blocks across all parameters; the lookup
    tables ``block_to_param`` / ``block_within`` decode it into a
    (param_idx, block_offset_within_param) pair, then per-param metadata
    (raw X pointer, numel, seeds, offset_base, std) is loaded from the
    indirection tables.

    Semantics are bit-identical to looping ``_syre_wd_kernel`` over each
    parameter individually. The fusion saves only launch overhead --
    each program does the same arithmetic.

    The per-param ``std`` (sigma_0) lives in ``stds_ptr`` so that
    different parameters can use different sigma_0 (e.g. paper-style
    per-fan-in scaling) in a single launch. When the caller wants a
    uniform sigma_0 the wrapper builds a constant table; the extra load
    is one cache hit per block and below noise.

    X and U may have different dtypes (Aurora's post-orth path holds X
    in fp32 master and U in bf16-ish update form), but all X[i] share
    ``X_STORAGE_DTYPE`` and all U[i] share ``U_STORAGE_DTYPE`` per launch
    (validated in the wrapper).
    """
    pid = tl.program_id(0)
    param_idx = tl.load(block_to_param_ptr + pid)
    block_within = tl.load(block_within_ptr + pid)

    numel = tl.load(numels_ptr + param_idx)
    seed1 = tl.load(seeds1_ptr + param_idx)
    seed2 = tl.load(seeds2_ptr + param_idx)
    offset_base = tl.load(offset_bases_ptr + param_idx)
    std = tl.load(stds_ptr + param_idx)

    offsets = block_within.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x_addr = tl.load(X_ptrs_ptr + param_idx)
    x_ptr = x_addr.to(tl.pointer_type(X_STORAGE_DTYPE))

    theta = tl.load(x_ptr + offsets, mask=mask)
    theta_f32 = theta.to(tl.float32)

    global_offsets = offsets + offset_base
    theta_0 = tl.randn(seed1, global_offsets) * std
    diff = theta_f32 - theta_0

    if ADVANCED_REMOVAL:
        # See additive-form note in _syre_wd_kernel above.
        xi = tl.rand(seed2, global_offsets) * (2.0 * d_bound) - d_bound
        diff = diff + diff * xi

    if CAUTIOUS:
        u_addr = tl.load(U_ptrs_ptr + param_idx)
        u_ptr = u_addr.to(tl.pointer_type(U_STORAGE_DTYPE))
        u_f32 = tl.load(u_ptr + offsets, mask=mask).to(tl.float32)
        diff = tl.where(u_f32 * theta_f32 >= 0.0, diff, 0.0)

    result = theta_f32 - gamma * diff
    tl.store(x_ptr + offsets, result.to(X_STORAGE_DTYPE), mask=mask)


def syre_wd_multi_inplace(
    Xs: List[Tensor],
    gamma: float,
    seeds1: List[int],
    std: Union[float, Sequence[float]],
    seeds2: List[int],
    d_bound: float,
    advanced_removal: bool,
    offset_bases: List[int],
    Us: Optional[List[Tensor]] = None,
):
    """Apply SYRE weight decay in-place to a list of tensors with one
    Triton launch.

    Semantically equivalent to looping :func:`syre_wd_inplace` over each
    ``(X, seed1, seed2, offset_base, std)`` (and ``U`` when cautious),
    but pays per-launch overhead once instead of ``N`` times. Designed
    for sharded clusters where per-rank shards are small and ``N``
    launches of ~18 us each dominate the SYRE step.

    All inputs share ``gamma``, ``d_bound``, ``advanced_removal``, and
    ``cautious`` (set by ``Us is not None``). Per-param state lives in
    the lists.

    Args:
        Xs: parameter tensors (decayed in place). All must share device
            and dtype. Non-contiguous tensors are temporarily contiguified
            and copied back, as in the single-tensor wrapper. Empty
            tensors are skipped.
        gamma: scalar decay coefficient. ``0`` is a no-op (returns
            immediately).
        seeds1: per-param ``seed1`` (theta_0 PRNG seed).
        std: scale for ``theta_0``. Either a single positive float (shared
            across all params) or a per-param sequence of positive floats
            matching ``len(Xs)``. The per-param form is how the
            ``syre_std_mode`` preset/callable plumbs through.
        seeds2: per-param ``seed2`` (AR multiplier PRNG seed). Ignored
            unless ``advanced_removal=True``; any value works otherwise.
        d_bound: half-width of the AR uniform interval. Ignored unless
            ``advanced_removal=True``.
        advanced_removal: SYRE-AR variant flag (uniform across the batch).
        offset_bases: per-param starting Philox offset for global indexing
            in sharded settings.
        Us: optional list of update tensors for the cautious mask. If
            provided, must have the same length as ``Xs`` and each
            ``U[i]`` must match ``X[i]`` in numel.
    """
    if not Xs or gamma == 0.0:
        return

    n_in = len(Xs)
    assert len(seeds1) == n_in, "seeds1 length must match Xs"
    assert len(seeds2) == n_in, "seeds2 length must match Xs"
    assert len(offset_bases) == n_in, "offset_bases length must match Xs"
    cautious = Us is not None
    if cautious:
        assert len(Us) == n_in, "Us length must match Xs"

    # Normalize ``std`` to a per-param list. Scalars broadcast; sequences
    # must match ``len(Xs)``. We accept anything indexable with ``len()``
    # to keep call sites flexible (tuple, list, 1-D Python sequence).
    if isinstance(std, (int, float)) and not isinstance(std, bool):
        stds: List[float] = [float(std)] * n_in
    else:
        try:
            n_std = len(std)  # type: ignore[arg-type]
        except TypeError as e:
            raise TypeError(
                f"std must be a float or a sequence of floats, got "
                f"{type(std).__name__}: {std!r}"
            ) from e
        if n_std != n_in:
            raise ValueError(
                f"std sequence length ({n_std}) must match len(Xs) ({n_in})"
            )
        stds = [float(s) for s in std]  # type: ignore[union-attr]

    device = Xs[0].device
    x_dtype = Xs[0].dtype
    if x_dtype not in _TORCH_TO_TL:
        raise TypeError(
            f"syre_wd_multi_inplace: unsupported dtype {x_dtype}. "
            f"Supported: {list(_TORCH_TO_TL)}."
        )
    for i, X in enumerate(Xs):
        if X.device != device:
            raise ValueError(
                f"Xs[{i}].device={X.device} != Xs[0].device={device}"
            )
        if X.dtype != x_dtype:
            raise ValueError(
                f"Xs[{i}].dtype={X.dtype} != Xs[0].dtype={x_dtype}"
            )

    # ``U`` can have a different dtype from ``X`` (Aurora's post-orth
    # update tensors are bf16-ish even when params are fp32). We allow
    # that, but require U's dtype to be uniform across the U list.
    u_dtype = None
    if cautious:
        u_dtype = Us[0].dtype
        if u_dtype not in _TORCH_TO_TL:
            raise TypeError(
                f"syre_wd_multi_inplace: unsupported U dtype {u_dtype}. "
                f"Supported: {list(_TORCH_TO_TL)}."
            )
        for i, U in enumerate(Us):
            if U.dtype != u_dtype:
                raise ValueError(
                    f"Us[{i}].dtype={U.dtype} != Us[0].dtype={u_dtype}"
                )
            if U.device != device:
                raise ValueError(
                    f"Us[{i}].device={U.device} != Xs[0].device={device}"
                )

    BLOCK_SIZE = 1024

    # Contiguify (rare for FSDP2 local shards) and flatten. Keep the
    # contig tensors alive so their storage isn't freed before launch.
    contigs: List[Tensor] = []
    copy_backs: List[tuple] = []
    flats: List[Tensor] = []
    for X in Xs:
        if X.is_contiguous():
            X_c = X
        else:
            X_c = X.contiguous()
            copy_backs.append((X, X_c))
        contigs.append(X_c)
        flats.append(X_c.view(-1))

    u_flats: List[Tensor] = []
    if cautious:
        for X, U in zip(Xs, Us):
            if U.numel() != X.numel():
                raise ValueError(
                    f"U.numel()={U.numel()} != X.numel()={X.numel()}"
                )
            u_flats.append(U.reshape(-1))

    # Filter out zero-numel params -- they would contribute 0 blocks
    # and the metadata-table construction handles them fine, but
    # carrying them through wastes work.
    keep = [i for i, f in enumerate(flats) if f.numel() > 0]
    if not keep:
        return
    if len(keep) != n_in:
        flats = [flats[i] for i in keep]
        seeds1 = [seeds1[i] for i in keep]
        seeds2 = [seeds2[i] for i in keep]
        offset_bases = [offset_bases[i] for i in keep]
        stds = [stds[i] for i in keep]
        if cautious:
            u_flats = [u_flats[i] for i in keep]

    n = len(flats)
    numels = [int(f.numel()) for f in flats]
    blocks_per_param = [(m + BLOCK_SIZE - 1) // BLOCK_SIZE for m in numels]
    total_blocks = sum(blocks_per_param)

    # Build metadata tensors. The block-to-param / block-within tables
    # are built with cumsum + arange on GPU (faster than Python list
    # comprehension for large block counts).
    x_addrs = torch.tensor(
        [f.data_ptr() for f in flats],
        dtype=torch.int64, device=device,
    )
    if cautious:
        u_addrs = torch.tensor(
            [u.data_ptr() for u in u_flats],
            dtype=torch.int64, device=device,
        )
    else:
        u_addrs = x_addrs  # dummy; kernel never dereferences

    numels_t = torch.tensor(numels, dtype=torch.int64, device=device)
    seeds1_t = torch.tensor(seeds1, dtype=torch.int32, device=device)
    seeds2_t = torch.tensor(seeds2, dtype=torch.int32, device=device)
    offset_bases_t = torch.tensor(offset_bases, dtype=torch.int64, device=device)
    stds_t = torch.tensor(stds, dtype=torch.float32, device=device)
    blocks_per_param_t = torch.tensor(
        blocks_per_param, dtype=torch.int32, device=device,
    )
    # block_to_param[k] = i  iff block k belongs to param i.
    block_to_param_t = torch.arange(n, device=device, dtype=torch.int32) \
        .repeat_interleave(blocks_per_param_t)
    # block_within[k] = k - cumulative_blocks_before_its_param.
    cum_blocks = blocks_per_param_t.cumsum(0)
    starts = torch.cat([
        torch.zeros(1, dtype=cum_blocks.dtype, device=device),
        cum_blocks[:-1],
    ])
    block_within_t = (
        torch.arange(total_blocks, device=device, dtype=torch.int32)
        - starts[block_to_param_t]
    )

    grid = (total_blocks,)
    x_tl_dtype = _TORCH_TO_TL[x_dtype]
    # When non-cautious, U pointers are dummies and the U_STORAGE_DTYPE
    # constexpr branch is dead-code-eliminated. Pass X's dtype to keep
    # the kernel cache key small (one fewer dimension when CAUTIOUS=False).
    u_tl_dtype = _TORCH_TO_TL[u_dtype] if cautious else x_tl_dtype

    with torch.cuda.device(device):
        _syre_wd_multi_kernel[grid](
            x_addrs, u_addrs,
            numels_t, seeds1_t, seeds2_t, offset_bases_t, stds_t,
            block_to_param_t, block_within_t,
            gamma, d_bound,
            BLOCK_SIZE=BLOCK_SIZE,
            ADVANCED_REMOVAL=advanced_removal,
            CAUTIOUS=cautious,
            X_STORAGE_DTYPE=x_tl_dtype,
            U_STORAGE_DTYPE=u_tl_dtype,
        )

    for X, X_c in copy_backs:
        X.copy_(X_c)
