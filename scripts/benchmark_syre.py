"""Benchmark SYRE cost as a fraction of a full Aurora optimizer step.

Three configurations, all running real ``Aurora.step()`` on a realistic
transformer matrix-param distribution. All three use the same
``weight_decay=0.1``; SYRE replaces the decoupled WD step rather than
being added on top, so apples-to-apples is "decoupled WD vs. SYRE WD"
not "no WD vs. SYRE WD".

  1. Aurora baseline: ``weight_decay=0.1``, ``syre_wd=False`` --
     decoupled ``X.mul_(1 - lr*wd)`` after Newton-Schulz.
  2. Aurora + SYRE: ``syre_wd=True``, ``advanced_removal=False``.
  3. Aurora + SYRE-AR: ``syre_wd=True``, ``advanced_removal=True``.

For each config we report the per-step wall-clock; the delta between
(1) and (2)/(3) is the SYRE cost as a fraction of the full optimizer
step. That's the number that actually matters for "is multi-tensor
fusion worth it" -- SYRE launch overhead only matters if SYRE is a
nontrivial fraction of the step.

We also include the launch-overhead microbench from earlier as a
diagnostic (per-launch us, tiny-param dominance).

Run:
    uv run python scripts/benchmark_syre.py
"""
from __future__ import annotations

import gc
import time
from typing import List, Tuple

import torch

from dion.aurora import Aurora
from dion.syre import syre_wd_inplace, syre_wd_multi_inplace


def transformer_matrix_shapes(
    hidden: int = 1024,
    n_layers: int = 32,
    mlp_mult: int = 4,
    vocab: int = 16_000,
) -> List[Tuple[int, ...]]:
    """Matrix params (ndim >= 2) for a transformer-like model. Aurora
    only processes matrix params; LayerNorm vectors would go through an
    AdamW group, which is a separate measurement.
    """
    shapes: List[Tuple[int, ...]] = []
    shapes.append((vocab, hidden))                        # embed
    shapes.append((vocab, hidden))                        # lm_head
    for _ in range(n_layers):
        shapes.extend([(hidden, hidden)] * 4)             # Q, K, V, O
        shapes.append((mlp_mult * hidden, hidden))        # MLP up
        shapes.append((hidden, mlp_mult * hidden))        # MLP down
    return shapes


def make_params(shapes, dtype=torch.float32) -> List[torch.nn.Parameter]:
    return [
        torch.nn.Parameter(torch.randn(*s, device="cuda", dtype=dtype) * 0.02)
        for s in shapes
    ]


def time_step(opt, params, n_iters: int = 10, warmup: int = 3) -> float:
    """Time ``opt.step()`` with freshly-generated grads each iteration.

    We pre-allocate one set of grad tensors that match each param's
    shape and copy_ from a single source noise tensor each iteration --
    this keeps grad-allocation cost out of the measurement while still
    forcing the optimizer to do real per-param work.
    """
    grads = [torch.randn_like(p) * 0.01 for p in params]
    for p, g in zip(params, grads):
        p.grad = g

    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        # Re-randomize a fraction of grads to avoid pathological reuse,
        # without paying full alloc cost. (Cheap enough; doesn't affect
        # the optimizer-step timing meaningfully.)
        for g in grads[::8]:
            g.normal_(0, 0.01)
        opt.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def bench_config(label: str, shapes, lr, wd, syre_kwargs):
    """Allocate fresh params + optimizer for one config, time, tear down."""
    torch.manual_seed(0)
    params = make_params(shapes)
    n = len(params)
    numel = sum(p.numel() for p in params)
    opt = Aurora(params, lr=lr, weight_decay=wd, **syre_kwargs)
    try:
        t = time_step(opt, params)
    finally:
        # Free params + opt state explicitly so the next config has
        # the memory headroom.
        del opt, params
        gc.collect()
        torch.cuda.empty_cache()
    return label, n, numel, t


def sharded_syre_extrapolation(shapes, syre_std, gamma):
    """Simulate the SYRE-only step at different ``world_size`` values.

    Aurora's megabatch design AllGathers, runs NS on gathered tensors,
    and SYRE runs on each rank's LOCAL shard (``X_local`` in
    ``aurora_update_post_orthogonalize``). Per rank the SYRE step is
    ``N`` launches on tensors of size ``global/W`` each. We simulate
    that by allocating each param at size ``ceil(orig_numel / W)`` and
    timing the same N-launch loop. Newton-Schulz parallelizes across
    ranks differently (a subset of params per rank), so this
    extrapolation is *just* for the SYRE-step component, not the full
    optimizer.
    """
    print("--- SYRE-only step at simulated world_size ----------------------")
    print(f"  ({len(shapes)} params, varying local-shard size)")
    print()

    for W in (1, 4, 16, 64, 256):
        local_params = []
        for s in shapes:
            global_numel = 1
            for d in s:
                global_numel *= d
            local_numel = (global_numel + W - 1) // W
            local_params.append(
                torch.randn(local_numel, device="cuda", dtype=torch.float32)
            )
        n = len(local_params)
        seeds1 = list(range(1, n + 1))
        seeds2 = list(range(10001, 10001 + n))
        offs = [0] * n

        def step_loop_basic():
            for p, s1, s2, o in zip(local_params, seeds1, seeds2, offs):
                syre_wd_inplace(
                    p, gamma=gamma, seed1=s1, std=syre_std,
                    seed2=s2, d_bound=0.1 * syre_std,
                    advanced_removal=False, offset_base=o,
                )

        def step_loop_ar():
            for p, s1, s2, o in zip(local_params, seeds1, seeds2, offs):
                syre_wd_inplace(
                    p, gamma=gamma, seed1=s1, std=syre_std,
                    seed2=s2, d_bound=0.1 * syre_std,
                    advanced_removal=True, offset_base=o,
                )

        def step_multi_basic():
            syre_wd_multi_inplace(
                local_params, gamma=gamma, seeds1=seeds1, std=syre_std,
                seeds2=seeds2, d_bound=0.1 * syre_std,
                advanced_removal=False, offset_bases=offs,
            )

        def step_multi_ar():
            syre_wd_multi_inplace(
                local_params, gamma=gamma, seeds1=seeds1, std=syre_std,
                seeds2=seeds2, d_bound=0.1 * syre_std,
                advanced_removal=True, offset_bases=offs,
            )

        # Warm + time.
        for _ in range(3):
            step_loop_basic(); step_loop_ar()
            step_multi_basic(); step_multi_ar()
        torch.cuda.synchronize()

        def time_n(fn, n=20):
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n

        t_loop_basic = time_n(step_loop_basic)
        t_loop_ar    = time_n(step_loop_ar)
        t_multi_basic = time_n(step_multi_basic)
        t_multi_ar    = time_n(step_multi_ar)

        local_total = sum(p.numel() for p in local_params)
        print(f"  W={W:>4}:  local shard total = {local_total/1e6:7.1f}M elements")
        print(f"           per-param  SYRE      : {t_loop_basic*1e3:7.3f} ms  "
              f"-> multi: {t_multi_basic*1e3:7.3f} ms  "
              f"(speedup {t_loop_basic/t_multi_basic:5.1f}x)")
        print(f"           per-param  SYRE-AR   : {t_loop_ar*1e3:7.3f} ms  "
              f"-> multi: {t_multi_ar*1e3:7.3f} ms  "
              f"(speedup {t_loop_ar/t_multi_ar:5.1f}x)")

        del local_params
        gc.collect()
        torch.cuda.empty_cache()
    print()


def launch_overhead_microbench():
    """Carry over the per-param-launch overhead microbench."""
    print("--- Launch-overhead microbench ----------------------------------")
    n_tensors = 64
    size = 4096
    tiny = [torch.randn(size, device="cuda") for _ in range(n_tensors)]
    big = torch.randn(n_tensors * size, device="cuda")
    seeds1 = list(range(1, n_tensors + 1))
    seeds2 = list(range(101, n_tensors + 101))
    offs = [0] * n_tensors

    def per_param():
        for p, s1, s2, o in zip(tiny, seeds1, seeds2, offs):
            syre_wd_inplace(
                p, gamma=1e-5, seed1=s1, std=1e-4,
                seed2=s2, d_bound=1e-5,
                advanced_removal=False, offset_base=o,
            )

    def single():
        syre_wd_inplace(
            big, gamma=1e-5, seed1=1, std=1e-4,
            seed2=2, d_bound=1e-5,
            advanced_removal=False, offset_base=0,
        )

    for _ in range(3):
        per_param(); single()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        per_param()
    torch.cuda.synchronize()
    t_per = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        single()
    torch.cuda.synchronize()
    t_one = (time.perf_counter() - t0) / 20

    delta = t_per - t_one
    print(f"  {n_tensors} x [{size}] fp32:")
    print(f"    per-param launches: {t_per*1e3:8.3f} ms")
    print(f"    single launch:      {t_one*1e3:8.3f} ms")
    print(f"    per-launch overhead: {delta*1e6/n_tensors:6.2f} us")
    print()


def main():
    assert torch.cuda.is_available(), "CUDA required"

    shapes = transformer_matrix_shapes()
    n = len(shapes)
    total_numel = sum(int(torch.tensor(s).prod().item()) for s in shapes)
    print(f"Param-shape distribution (matrix params only):")
    print(f"  count:        {n}")
    print(f"  total numel:  {total_numel:,}  ({total_numel/1e6:.0f}M elements)")
    print(f"  storage:      {total_numel * 4 / 1e9:.2f} GB (fp32)")
    print()

    lr = 3e-4
    wd = 0.1
    syre_std = 0.01 / (1024 ** 0.5)   # paper recommendation for hidden=1024

    configs = [
        ("Aurora baseline (decoupled WD)", {}),
        ("Aurora + SYRE",                  {"syre_wd": True, "syre_std": syre_std}),
        ("Aurora + SYRE-AR",               {"syre_wd": True, "syre_std": syre_std,
                                            "advanced_removal": True}),
    ]

    print("--- Full Aurora.step() wall-clock -------------------------------")
    results = []
    for label, kwargs in configs:
        r = bench_config(label, shapes, lr, wd, kwargs)
        results.append(r)
        print(f"  {label:35s}: {r[3]*1e3:7.2f} ms / step")

    baseline_t = results[0][3]
    print()
    print("  Delta vs. baseline (= SYRE cost on top of decoupled WD):")
    for label, _, _, t in results[1:]:
        delta = t - baseline_t
        pct = 100 * delta / t
        print(f"    {label:33s}: +{delta*1e3:6.2f} ms ({pct:4.1f}% of step)")
    print()

    gamma = lr * wd
    sharded_syre_extrapolation(shapes, syre_std, gamma)
    launch_overhead_microbench()


if __name__ == "__main__":
    main()
