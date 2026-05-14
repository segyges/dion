"""Cache-interaction edge cases for the two metadata caches added in
commit 0185260 ("tighten compile coverage and metadata-rebuild
overhead"):

  * ``dion.aurora._syre_meta_cache``   -- per param-group dict, holds
    ``(seeds1, seeds2, offset_bases, stds)`` keyed by
    ``tuple(id(p) for p in params)``.

  * ``dion.syre._SYRE_METADATA_CACHE`` -- module-level, holds the GPU
    metadata tensors for ``syre_wd_multi_inplace``, keyed by the full
    ``(data_ptr, numels, seeds, offset_bases, stds)`` tuple.

These tests probe four categories of edge case:

  1. Safe behavior under common mutation patterns (grad-filter toggle,
     overflow, distinct param sets, seed change between calls).
  2. Stale-data hazards under unusual but legal mutation patterns
     (mid-training ``state[p]['syre_std']`` mutation,
     ``load_state_dict`` after the cache has filled). These are
     marked ``xfail(strict=True)`` -- they document known bugs and
     will turn into failures if the underlying behavior changes (e.g.,
     someone adds cache invalidation in ``load_state_dict``), forcing
     a deliberate update of these tests.
  3. Cap-overflow behavior (wholesale clear).
  4. Result-equivalence: cache-hit values must equal a fresh build.
"""

from __future__ import annotations

import importlib.util
import pytest
import torch

from dion.aurora import Aurora, _SYRE_META_CACHE_MAXSIZE
import dion.syre as syre_module


CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
HAS_TRITON_GPU = CUDA_AVAILABLE and TRITON_AVAILABLE

gpu_only = pytest.mark.skipif(
    not HAS_TRITON_GPU, reason="SYRE requires CUDA + triton"
)


@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Module-level _SYRE_METADATA_CACHE leaks across tests; clear before
    and after each one so tests don't see stale entries from neighbors.
    """
    syre_module._SYRE_METADATA_CACHE.clear()
    yield
    syre_module._SYRE_METADATA_CACHE.clear()


def _adamw_opt(p_init, **kw):
    """One-line Aurora-AdamW-with-SYRE optimizer for cache tests."""
    p = torch.nn.Parameter(p_init.detach().clone())
    opt = Aurora(
        [{"params": [p], "algorithm": "adamw"}],
        lr=0.05, weight_decay=0.05, syre_wd=True, syre_std=0.04, **kw,
    )
    return p, opt


# ---------------------------------------------------------------------
# Aurora _syre_meta_cache: safe behaviors
# ---------------------------------------------------------------------


@gpu_only
def test_aurora_meta_cache_fills_after_step():
    """Smoke test: after the first ``opt.step()``, the group dict has a
    ``_syre_meta_cache`` entry. Establishes that subsequent tests are
    actually exercising the cache path.
    """
    torch.manual_seed(0)
    p_init = torch.randn(16, device="cuda") * 0.1
    p, opt = _adamw_opt(p_init)
    p.grad = torch.ones_like(p) * 0.1
    opt.step()
    cache = opt.param_groups[0].get("_syre_meta_cache")
    assert cache is not None
    assert len(cache) == 1
    # cache value shape: (seeds1, seeds2, offset_bases, stds)
    seeds1, seeds2, offset_bases, stds = next(iter(cache.values()))
    assert len(seeds1) == 1
    assert stds == [0.04]


@gpu_only
def test_aurora_meta_cache_grad_filter_toggle_creates_separate_entries():
    """``group_params = [p for p in group['params'] if p.grad is not None]``
    runs before cache lookup, so a step that skips some params (None
    grads) gets a distinct cache key and a fresh entry. Both subsets
    must keep their own entries side-by-side; toggling back must hit
    the existing entry, not create a third.
    """
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(16, device="cuda") * 0.1)
    p2 = torch.nn.Parameter(torch.randn(16, device="cuda") * 0.1)
    opt = Aurora(
        [{"params": [p1, p2], "algorithm": "adamw"}],
        lr=0.05, weight_decay=0.05, syre_wd=True, syre_std=0.04,
    )

    # Step 1: both have grads
    p1.grad = torch.ones_like(p1) * 0.1
    p2.grad = torch.ones_like(p2) * 0.1
    opt.step()

    # Step 2: only p1 has a grad
    p1.grad = torch.ones_like(p1) * 0.1
    p2.grad = None
    opt.step()

    # Step 3: both again -- should reuse the step-1 entry, not create new
    p1.grad = torch.ones_like(p1) * 0.1
    p2.grad = torch.ones_like(p2) * 0.1
    opt.step()

    cache = opt.param_groups[0]["_syre_meta_cache"]
    assert len(cache) == 2, (
        f"Expected 2 cache entries (both-grads, p1-only), got {len(cache)}"
    )


@gpu_only
def test_aurora_meta_cache_overflow_wholesale_clears():
    """``_SYRE_META_CACHE_MAXSIZE`` is a wholesale-clear cap. When the
    Nth-plus-1th distinct subset is requested, the cache is dropped and
    rebuilt from scratch. Verify the cache stays bounded and results
    are still correct after a clear.
    """
    cap = _SYRE_META_CACHE_MAXSIZE  # 16
    torch.manual_seed(0)
    # ``cap + 2`` params; each step we'll pass a different SINGLE-param
    # subset to force a unique cache key per step.
    params = [
        torch.nn.Parameter(torch.randn(8, device="cuda") * 0.1)
        for _ in range(cap + 2)
    ]
    opt = Aurora(
        [{"params": params, "algorithm": "adamw"}],
        lr=0.05, weight_decay=0.05, syre_wd=True, syre_std=0.04,
    )

    # Step every param individually so each step's filtered subset is
    # a fresh singleton -> fresh cache key.
    for i, p in enumerate(params):
        for q in params:
            q.grad = None
        p.grad = torch.ones_like(p) * 0.1
        opt.step()

    cache = opt.param_groups[0]["_syre_meta_cache"]
    assert len(cache) <= cap, (
        f"Cache grew past cap: {len(cache)} > {cap}"
    )

    # Final all-grads step: results must still be correct -- if the
    # cache silently corrupted seeds/stds during a clear-and-rebuild,
    # we'd see NaN or way-off values here.
    for p in params:
        p.grad = torch.ones_like(p) * 0.05
    opt.step()
    for p in params:
        assert torch.isfinite(p.detach()).all()


# ---------------------------------------------------------------------
# Aurora _syre_meta_cache: stale-data hazards (xfail strict)
# ---------------------------------------------------------------------


@gpu_only
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation: ``_syre_meta_cache`` is keyed by id(p) and "
        "snapshots (seeds1, seeds2, offset_bases, stds) on first build. "
        "``load_state_dict`` does not invalidate the cache, so a resume "
        "into an already-stepped optimizer reuses pre-load seeds/stds. "
        "Fresh-optimizer load_state_dict (the documented resume path) is "
        "safe -- see test_syre_adamw_state_dict_round_trip."
    ),
)
def test_aurora_meta_cache_invalidated_on_load_state_dict_same_opt():
    """``load_state_dict`` mid-run must not leave the cache pointing at
    pre-load seeds/stds. Construct two optimizers with identical config,
    fill the cache in the first by stepping, then load a state dict
    that should change the seeds. The next step must use the loaded
    seeds, not the cached pre-load seeds.
    """
    torch.manual_seed(0)
    p_init = torch.randn(16, device="cuda") * 0.1
    g = torch.ones(16, device="cuda") * 0.1

    # Fresh-opt baseline: load_state_dict before any step.
    pA, optA = _adamw_opt(p_init)
    # Plant known seeds/std directly so the comparison is reproducible.
    optA.state[pA]["syre_seed1"] = 12345
    optA.state[pA]["syre_seed2"] = 0
    optA.state[pA]["syre_std"] = 0.04
    pA.grad = g.clone()
    optA.step()
    sd = optA.state_dict()
    expected = pA.detach().clone()

    # Same-opt resume: fill cache with DIFFERENT seeds, then
    # load_state_dict expecting the cache to be refreshed.
    pB, optB = _adamw_opt(p_init)
    # First step with an unrelated seed -- this fills the cache.
    optB.state[pB]["syre_seed1"] = 99999
    optB.state[pB]["syre_seed2"] = 0
    optB.state[pB]["syre_std"] = 0.04
    pB.grad = g.clone()
    optB.step()
    # Now reset pB to p_init and load the state dict from optA. After
    # the load, ``state[pB]['syre_seed1']`` should be 12345 -- and the
    # next step should match optA.
    with torch.no_grad():
        pB.copy_(p_init)
    optB.load_state_dict(sd)
    pB.grad = g.clone()
    optB.step()
    got = pB.detach().clone()

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


@gpu_only
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation: mutating ``state[p]['syre_std']`` after the "
        "first ``opt.step()`` does not flow through to subsequent steps "
        "because ``_syre_meta_cache`` snapshots the std list at first "
        "build. Same applies to ``state[p]['syre_seed1' / 'syre_seed2']``. "
        "If/when cache invalidation is added, drop the xfail marker."
    ),
)
def test_aurora_meta_cache_picks_up_state_std_mutation():
    """If a user mutates ``state[p]['syre_std']`` between steps, the
    next step should use the new value. With the current cache, it
    doesn't.
    """
    torch.manual_seed(0)
    p_init = torch.randn(16, device="cuda") * 0.1
    p, opt = _adamw_opt(p_init)
    p.grad = torch.ones_like(p) * 0.1
    opt.step()

    # Mutate std on the state dict -- the user-facing surface for
    # changing SYRE behavior after construction.
    opt.state[p]["syre_std"] = 0.999

    cache = opt.param_groups[0]["_syre_meta_cache"]
    _, _, _, cached_stds = next(iter(cache.values()))
    assert cached_stds == [0.999], (
        f"Cache should reflect mutated state[p]['syre_std']=0.999, "
        f"got cached_stds={cached_stds}"
    )


# ---------------------------------------------------------------------
# Module-level _SYRE_METADATA_CACHE: safe behaviors
# ---------------------------------------------------------------------


@gpu_only
def test_syre_static_cache_seed_change_rebuilds():
    """``seeds1`` enters the cache key as a Python tuple, so calling
    ``syre_wd_multi_inplace`` with the same param at the same storage
    but DIFFERENT seeds must produce two distinct cache entries (no
    silent reuse of the old metadata tensors).
    """
    from dion.syre import syre_wd_multi_inplace

    X = torch.randn(64, device="cuda") * 0.1
    X1, X2 = X.clone(), X.clone()
    syre_wd_multi_inplace(
        [X1], gamma=0.01,
        seeds1=[42], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )
    n_after_1 = len(syre_module._SYRE_METADATA_CACHE)
    syre_wd_multi_inplace(
        [X2], gamma=0.01,
        seeds1=[99], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )
    n_after_2 = len(syre_module._SYRE_METADATA_CACHE)
    assert n_after_2 == n_after_1 + 1, (
        f"Different seeds must produce distinct cache entries: "
        f"got {n_after_1} -> {n_after_2}"
    )
    # Sanity: the two outputs must differ because seeds drive the
    # per-element pull.
    assert not torch.equal(X1, X2)


@gpu_only
def test_syre_static_cache_overflow_clears():
    """Module cache uses a wholesale-clear cap at
    ``_SYRE_METADATA_CACHE_MAXSIZE``. Generate >cap distinct entries by
    walking seeds; cache must stay bounded and the final call must
    return correct values (not stale tensors).
    """
    from dion.syre import syre_wd_multi_inplace

    cap = syre_module._SYRE_METADATA_CACHE_MAXSIZE  # 64
    X = torch.randn(64, device="cuda") * 0.1
    # Each call uses a fresh seed -> fresh cache key (data_ptr is the
    # same for cloned tensors but the seed tuple varies).
    for seed in range(cap + 5):
        Xi = X.clone()
        syre_wd_multi_inplace(
            [Xi], gamma=0.005,
            seeds1=[seed], std=[0.04], seeds2=[0], offset_bases=[0],
            advanced_removal=False, d_bound=0.01,
        )
    assert len(syre_module._SYRE_METADATA_CACHE) <= cap

    # Final correctness check: a known-seed call after overflow must
    # produce the same result as the same call from a cold cache.
    syre_module._SYRE_METADATA_CACHE.clear()
    Xa = X.clone()
    syre_wd_multi_inplace(
        [Xa], gamma=0.005,
        seeds1=[7777], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )
    # Re-populate beyond cap, then call with seed=7777 again -- must
    # match the cold-cache result.
    for seed in range(cap + 5):
        Xi = X.clone()
        syre_wd_multi_inplace(
            [Xi], gamma=0.005,
            seeds1=[seed + 10_000], std=[0.04], seeds2=[0],
            offset_bases=[0], advanced_removal=False, d_bound=0.01,
        )
    Xb = X.clone()
    syre_wd_multi_inplace(
        [Xb], gamma=0.005,
        seeds1=[7777], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )
    torch.testing.assert_close(Xa, Xb, atol=0.0, rtol=0.0)


@gpu_only
def test_syre_static_cache_hit_equals_fresh_build():
    """The punchline: cache hits must return values bit-identical to a
    fresh, no-cache build. Same param + same metadata called twice
    in a row: second call hits the cache. Compare to a third call
    after clearing the cache. All three results must match exactly.
    """
    from dion.syre import syre_wd_multi_inplace

    X = torch.randn(128, device="cuda") * 0.1
    X_a = X.clone()
    X_b = X.clone()
    X_c = X.clone()

    # Call 1: cold cache -> build
    syre_wd_multi_inplace(
        [X_a], gamma=0.01,
        seeds1=[31415], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )
    # Call 2: same args -> cache hit
    syre_wd_multi_inplace(
        [X_b], gamma=0.01,
        seeds1=[31415], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )
    # Call 3: cache cleared, fresh build
    syre_module._SYRE_METADATA_CACHE.clear()
    syre_wd_multi_inplace(
        [X_c], gamma=0.01,
        seeds1=[31415], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )

    assert torch.equal(X_a, X_b), "cache hit diverges from initial build"
    assert torch.equal(X_a, X_c), "cache hit diverges from post-clear build"


@gpu_only
def test_syre_static_cache_data_ptr_reuse_with_same_metadata_safe():
    """Worst-case for data_ptr keying: a flat is freed, a new flat
    allocated at the same address with identical numels/seeds/stds.
    Because the cache key includes seeds/numels/stds (not just
    data_ptr), the cache should still produce correct output -- the
    metadata tensors don't actually depend on data_ptr beyond identity.

    Construct the situation explicitly: call once, drop the tensor,
    allocate a fresh one (CUDA caching allocator very likely reuses
    the address), call again with the same numels/seeds/stds. Result
    must match a from-scratch run.
    """
    from dion.syre import syre_wd_multi_inplace, syre_wd_inplace

    n = 64
    X1 = torch.randn(n, device="cuda") * 0.1
    template = X1.clone()
    syre_wd_multi_inplace(
        [X1], gamma=0.01,
        seeds1=[2024], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )
    # Drop X1; allocate a new tensor of same numel/dtype/device.
    del X1
    torch.cuda.synchronize()
    X2 = template.clone()  # same content, likely same allocator slot
    syre_wd_multi_inplace(
        [X2], gamma=0.01,
        seeds1=[2024], std=[0.04], seeds2=[0], offset_bases=[0],
        advanced_removal=False, d_bound=0.01,
    )

    # Independent reference path: the single-tensor kernel takes the
    # same seed/std/offset_base and is not cached.
    X_ref = template.clone()
    syre_wd_inplace(
        X_ref, gamma=0.01, seed1=2024, std=0.04,
        seed2=0, d_bound=0.01, advanced_removal=False, offset_base=0,
    )
    torch.testing.assert_close(X2, X_ref, atol=0.0, rtol=0.0)
