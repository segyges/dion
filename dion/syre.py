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

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


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
