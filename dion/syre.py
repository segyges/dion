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
  slightly different decay coefficient.
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
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _syre_wd_kernel(
    X_ptr, numel, gamma, seed1, std, seed2, d_bound, offset_base,
    BLOCK_SIZE: tl.constexpr, ADVANCED_REMOVAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    theta = tl.load(X_ptr + offsets, mask=mask)
    theta_f32 = theta.to(tl.float32)

    global_offsets = offsets + offset_base
    theta_0 = tl.randn(seed1, global_offsets) * std
    diff = theta_f32 - theta_0

    if ADVANCED_REMOVAL:
        d = tl.rand(seed2, global_offsets) * (2.0 * d_bound) + (1.0 - d_bound)
        diff = diff * d

    result = theta_f32 - gamma * diff
    tl.store(X_ptr + offsets, result.to(theta.dtype), mask=mask)


@triton.jit
def _syre_wd_cautious_kernel(
    X_ptr, U_ptr, numel, gamma, seed1, std, seed2, d_bound, offset_base,
    BLOCK_SIZE: tl.constexpr, ADVANCED_REMOVAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    theta = tl.load(X_ptr + offsets, mask=mask)
    u = tl.load(U_ptr + offsets, mask=mask)
    theta_f32 = theta.to(tl.float32)
    u_f32 = u.to(tl.float32)

    global_offsets = offsets + offset_base
    theta_0 = tl.randn(seed1, global_offsets) * std
    diff = theta_f32 - theta_0

    if ADVANCED_REMOVAL:
        d = tl.rand(seed2, global_offsets) * (2.0 * d_bound) + (1.0 - d_bound)
        diff = diff * d

    cautious_mask = tl.where(u_f32 * theta_f32 >= 0.0, 1.0, 0.0)
    result = theta_f32 - gamma * diff * cautious_mask
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

    with torch.cuda.device(X.device):
        if U is not None:
            # ``reshape(-1)`` safely flattens a possibly-non-contiguous
            # U (copies if needed).
            U_flat = U.reshape(-1)
            _syre_wd_cautious_kernel[grid](
                X_flat, U_flat, numel,
                gamma, seed1, std, seed2, d_bound, offset_base,
                BLOCK_SIZE=BLOCK_SIZE,
                ADVANCED_REMOVAL=advanced_removal,
            )
        else:
            _syre_wd_kernel[grid](
                X_flat, numel,
                gamma, seed1, std, seed2, d_bound, offset_base,
                BLOCK_SIZE=BLOCK_SIZE,
                ADVANCED_REMOVAL=advanced_removal,
            )

    if needs_copy_back:
        X.copy_(X_contig)
