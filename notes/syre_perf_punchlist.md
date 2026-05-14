# SYRE performance punchlist

Audit of compile coverage and per-step Python overhead across the SYRE
stack (`dion/syre.py`, `dion/syre_config.py`, `dion/aurora.py`,
`dion/scalar_opts.py`). Snapshot date: 2026-05-13.

## Compile coverage today

| function | status | notes |
|---|---|---|
| `adamw_update` (single-tensor) | `@torch.compile(fullgraph=True)` | good |
| `adamw_update_foreach` (no-SYRE) | uses `torch._fused_adamw_` | best possible |
| `lion_update`, `lion_update_foreach` | `@torch.compile` | good |
| `_syre_wd_kernel`, `_syre_wd_multi_kernel` | `@triton.jit` | the multi-tensor kernel fuses N params into 1 launch |
| **`adamw_update_foreach_syre`** | **eager** | the comment says "not fused because Triton" — half-true; the AdamW math around the kernel can still be compiled |
| **`aurora_update_post_orthogonalize`** | eager, python `for x, u in zip(X, U)` final loop | one foreach call away |
| **`make_aurora_polar`** inner pointwise math | eager | mid-value compile candidate |

## Punchlist (ranked by value-to-effort)

### 1. Compile-fuse the AdamW pointwise math around the SYRE kernel ★

In `adamw_update_foreach_syre`, the function fires **9 kernel launches per
step**:

```
lerp_(M, G) | mul_(V) | addcmul_(V) | sqrt(V) | div_(denom) | add_(denom) | div(M,denom)
└── 7 launches of AdamW pointwise math ──────────────────────────────────┘
  + 1 SYRE Triton kernel
  + 1 foreach_add_(X, update_dirs)
```

The "eager because of Triton" comment is misleading: `torch.compile`
doesn't need to swallow the Triton call. Factor the M/V update + denom +
update_dirs computation into a `@torch.compile(fullgraph=True)` helper
that returns `update_dirs` (and mutates M/V in place); the SYRE call and
the final foreach_add_ stay eager. The 7 AdamW-math launches collapse
to 1 fused launch. Same savings profile as `adamw_update_foreach` (fused)
vs. a hand-rolled foreach chain.

Caveat: `torch.compile` over `torch._foreach_*` can recompile if list
length changes call-to-call. Aurora groups params by shape+dtype, so
within a `_create_adamw_tasks` yield the list length is stable for a
given param-group set. The first step pays compilation; subsequent
steps amortize.

### 2. Replace the python tail loop in `aurora_update_post_orthogonalize` ★

```python
for x, u in zip(X, U):
    x.sub_(u, alpha=adj_lr_f)
```
→
```python
torch._foreach_sub_(X, U, alpha=adj_lr_f)
```

N python dispatches → 1 multi-tensor-apply launch. Free.

### 3. Short-circuit `G_cast` when dtypes already match ★

In `adamw_update_foreach_syre`:

```python
G_cast = [g.to(m.dtype) for g, m in zip(G, M)]
```

`.to(same_dtype)` returns the same tensor with no copy, but still costs
an aten dispatch per param. When `G[i].dtype == M[i].dtype` for all i
(the common bf16-grad-bf16-momentum / fp32-grad-fp32-momentum cases),
the comprehension can short-circuit. Aurora's param-group construction
already constrains dtypes within a group, so checking `G[0].dtype ==
M[0].dtype` is sufficient.

### 4. Cache `syre_wd_multi_inplace` metadata tables across steps

Every call rebuilds and uploads 7 metadata tensors from python lists
(addrs, numels, seeds1, seeds2, offset_bases, stds, blocks_per_param).
Each `torch.tensor(python_list, device=cuda)` stages through pageable
host memory.

Seeds, stds, offset_bases, numels are stable across steps for a given
param set — only `data_ptr()`s can in principle change. Cache keyed by
`tuple(id(x) for x in Xs)` and rebuild only on miss. Lives one level up:
the optimizer holds the cache; the wrapper takes pre-built tables.

Risk: subtle bugs if a user mutates parameter storages (`param.data =
new_tensor`). FSDP2's unshard/reshard does *not* swap `data_ptr` on the
original Parameter, so safe in practice — defensive assertion warranted.

Value: %-meaningful on small clusters where N is small and overhead is a
larger share; on large clusters the kernel work dominates.

### 5. Cache per-param SYRE metadata in the optimizer state

The blocks in `_create_ortho_tasks` / `_create_adamw_tasks`:

```python
for p in original_params:
    s1, s2 = self._get_or_init_syre_seeds(p, advanced_removal)
    syre_seeds1.append(s1); ...
    syre_offset_bases.append(self._compute_syre_offset_base(p))
    syre_stds.append(self._get_or_init_syre_std(p, ...))
```

rebuild four python lists per step. After the first step the values are
stable (cached in `self.state[p]`), but we still pay dict lookups and
method-call overhead. Pre-build per-group `(seeds1, seeds2,
offset_bases, stds)` once at first-step time; invalidate only when the
group's param set changes. Tiny win per call, composes with (4).

### 6. Compile pointwise math inside `make_aurora_polar`'s `aurora_polar` body

```python
row_sq = U.to(fp32).pow(2).sum(dim=-1, keepdim=True).clamp(min=eps_sq)
D = D * (target_row_sq / row_sq).pow(pp_beta)
```

Runs `pp_iterations` times between `base_polar` calls. Each line is 2-3
launches; compile-fusing the chunks *between* base_polar calls saves a
few launches per polar iteration per param-shape-group. Subtle because
`base_polar` is sometimes itself a compiled Triton kernel — have to
compile only the interludes, not the whole loop body.

## What was checked and intentionally left alone

- **`_check_syre_triton_available`** uses `importlib.import_module` on
  every `add_param_group`. After the first success, `sys.modules`
  serves it instantly.
- **`_validate_syre_kwargs`** lazy-imports `SYRE_STD_PRESETS` only in
  error paths. Cold path; no work in steady state.
- **The single-tensor `syre_wd_inplace` wrapper** is only used by tests;
  the optimizer hot path is exclusively `syre_wd_multi_inplace`.
- **The kernel itself.** The four-way constexpr fan-out
  (`ADVANCED_REMOVAL` × `CAUTIOUS`) already specializes; dead branches
  DCE.

## Status

- [x] 1: compile-fuse AdamW pre-SYRE math
- [x] 2: foreach_sub_ in aurora_update_post_orthogonalize
- [x] 3: short-circuit G_cast on matching dtypes
- [x] 4: cache `syre_wd_multi_inplace` metadata (static-only; addresses
      always rebuilt because `data_ptr()` can change under
      `param.data = new_tensor` or non-contig contigification)
- [x] 5: cache per-param SYRE metadata in optimizer state
- [x] 6: compile pointwise math in `aurora_polar`

All items landed. Still want to extend `scripts/benchmark_syre.py` to
cover the AdamW-SYRE path before/after so we have a quantitative number
for the wall-time win.
