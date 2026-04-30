# Profile-driven Phase 2 — results & analysis

## What this branch ships

1. **Microbench harness** (`examples/c++/baobzi_microbench.cpp` + nanobench
   FetchContent dep) — sweeps {1D, 2D, 3D} × scientific kernels × {deg 6,
   8, 10} × N ∈ {1, 32, 1024, 10⁶}, reports MEvals/s, ns/eval, cyc/eval,
   IPC, branch-miss%, MdAPE.
2. **xsimd fork wired in** (`DiamonDinoia/xsimd:feat/dynamic-masks` via
   `CPM_xsimd_SOURCE` override, since polyfit fetches xsimd through CPM)
   — gives runtime / compile-time `batch_bool` masked-load primitives
   on AVX2 + AVX-512.
3. **Three perf-confirmed eval-pipeline changes** to
   `include/baobzi/detail/function_impl.hpp` (Phase 2.3, 2.4, and a
   profile-discovered prefetch removal).

## Profile-driven workflow (per simdref `references/workflow.md`)

```bash
# 1. Compile baobzi_microbench Release with -g -fno-omit-frame-pointer
# 2. Profile run
simdref profile run --target build-new/baobzi_microbench \
    --adapter perf --event "cycles:u,instructions:u" \
    --duration 60 --arch alderlake --top 5 -o report/

# 3. Hot-symbol triage via perf report -F overhead,sample,symbol
# 4. perf annotate --symbol "<hot baobzi function>"
# 5. simdref show <mnemonic> --arch alderlake  (lat/cpi citations)
```

Resolved microarch: **alderlake** (Intel Core Ultra 7 155H, P-core,
AVX2 W=4). simdref's measured ADL-P payloads cited inline below.

## Iteration 1 — perf identified two hotspots

The hot baobzi function in the microbench is
`baobzi::Function<deg, F>::operator()(double const*, double*, size_t)`,
which inlines into the bench's `sweep_*` lambdas. `perf annotate`
showed two clear targets in the `for (i…) leaf_ids[i] = …` traversal
loop (1D, log1p kernel, depth ~6):

| addr | % cycles | instruction | meaning |
|------|---------:|-------------|---------|
| 88b1c | **14.22%** | `prefetcht0 (%rdx)` | descent prefetch |
| 88b06 |   5.59% | `mov 0x10(%rdx),%edx` | load `first_child_idx` |
| 88b0c |   9.31% | `add %rdx,%rax` | `next = first_child_idx + child_idx` |
| 88b13 |   5.46% | `lea (%rsi,%rdx,8),%rdx` | `&node_pointers_[idx]` |
| 88b17 |   5.24% | `cmpq $-1,0x8(%rdx)` | `is_leaf()` via `poly_eval_id` sentinel |
| 88b24 |   7.95% | `mov 0x98(%r9),%rsi` | reload `node_pointers_.data()` |

`simdref show prefetcht0 --arch alderlake` → `lat=- cpi=1.0 ports=p23A`.
`simdref show vcomisd  --arch alderlake` → `lat=3c cpi=1.0`.

The prefetch is **port 23A** (load port). Each descent step's
`nodes_[next]` load is strictly dependent on the previous level's
`child_idx`, so the prefetch's address arrives *after* the demand
load it was meant to hide. It just steals a load-port slot per
descent step. Hence 14.22 % of the symbol burned on a no-op.

### Change A — drop the post-descent two-load indirection (Plan §2.3)

Replace `node_pointers_[idx]->poly_eval_id` (vector of `node_t*` →
deref to read `poly_eval_id`) with `leaf_index_by_global_node_[idx]`
(`std::vector<std::uint32_t>`, single load). Built and benched in the
same session vs pristine baseline (both on the xsimd fork): clean
+5–17 % across 1D N=1/32/1024 cells with err% < 3 %.

### Change B — remove the descent prefetch

```cpp
// before
const index_t next = nodes_[curr_index].first_child_idx + child_idx;
__builtin_prefetch(&nodes_[next]);
curr_index = next;

// after
curr_index = nodes_[curr_index].first_child_idx + child_idx;
```

A re-profile after this change confirmed `prefetcht0` no longer appears
in the function's annotated listing.

## Iteration 2 — descent now has 2 dependent loads per step

Re-profile of the post-A+B binary showed the descent loop tightened to:

```
mov 0x10(%rdx),%edx     ; load first_child_idx           — 7.18%
add %rdx,%rax           ; next = first_child_idx + idx   — 9.79%
lea (%rax,%rax,2),%rdx  ; ×3
lea (%r12,%rdx,8),%rdx  ; rdx = &nodes_[next]
cmpq $-1,0x8(%rdx)      ; is_leaf via poly_eval_id (off 8) — 4.16%
je   loop               ;                                — 13.20%
```

The `cmpq $-1, 0x8(%rdx)` test reads `poly_eval_id` at a *different*
offset than `first_child_idx`. Two dependent loads per descent step.

### Change C — reorder `Node` so `is_leaf` fuses with `first_child_idx` (Plan §2.4)

```cpp
struct Node {
    Value<value_type, input_dim> center;
    std::uint32_t first_child_idx = max32;
    std::uint32_t _pad_           = 0;
    std::uint64_t poly_eval_id    = max64;

    bool is_leaf() const {
        return first_child_idx == max32;   // same sentinel + same load
    }
};
```

`_pad_` keeps the total node size unchanged (24 B for 1D, 32 B for 2D,
40 B for 3D — i.e. existing footprint preserved). The descent loop
collapses to a *single* dependent load per level: load
`first_child_idx` once, use both for the leaf test and the
`next = first_child_idx + child_idx` step.

A subsequent re-profile after A+B+C shows the descent body is now:

```
vcomisd (%rsi),%xmm0    ; descent compare
seta %dl                ; child_idx
add  %rax,%rdx          ; next = …
lea  (%rdx,%rdx,2),%rsi ; ×3
lea  (%rcx,%rsi,8),%rsi ; &nodes_[next]
mov  0x8(%rsi),%eax     ; load first_child_idx (one load now)
cmp  $-1,%eax           ; is_leaf
jne  loop
```

## Tried & rejected — Plan §2.5 (axis_stride hoist)

I prototyped Plan §2.5 (precompute prefix-product strides for ND
`get_linear_bin`). Re-bench under nanobench showed it regressed ND
cells by 5–18 %, almost certainly because adding the new
`Value<size_t, dim>` member shifted `polyfits_`' cacheline residency
in the `Function` class layout. Reverted.

## Final delta vs pristine xsimd-fork baseline

`bench/v3_final_compare.txt` — pristine (xsimd fork) vs Phase 2.3 + 2.4
+ no-prefetch (xsimd fork), same session, interleaved builds, one nanobench
sample each. The wins are reproducible per intermediate compares
(`bench/v3_phase24_compare.txt` shows the cleanest signal at +5–20 %
across cells when the bench was re-run during the iteration loop).

The single-sample comparisons cap at ~1.7 % MdAPE on most large-N cells
and the day-to-day variance on this hybrid Meteor Lake host is ~5 %, so
small-N ratios should be averaged across multiple runs before being read
as definitive.

## Honest caveats

- The host (Meteor Lake P-core) is the noisier validation environment.
  AVX-512 hosts (Sapphire Rapids, Zen4-5) are where the next-step Phase 3
  SIMD-parallel descent would shine — the descent here is dependent-load
  bound, exactly as the plan diagnosed.
- The xsimd fork's masked-load APIs are wired in but **not yet used by
  baobzi**. The infrastructure is ready for Phase 2.1/2.2 SIMD bounds +
  child-bit (skipped because perf showed they aren't on the critical
  path on AVX2) and for Phase 3 vector descent (the actual next move).
- Phase 1.5 file split deferred — pure refactor, no perf delta.

## Files

- `include/baobzi/detail/function_impl.hpp` — Phase 2.3 + 2.4 + prefetch removal.
- `cmake/baobzi_deps.cmake` — xsimd fork override + nanobench fetch.
- `cmake/baobzi_examples.cmake` — `baobzi_microbench` target.
- `examples/c++/baobzi_microbench.cpp` — nanobench-driven sweep.
- `bench/compare_nb.py` — A/B parser for nanobench markdown.
- `bench/baseline_v3_nb.txt`, `bench/final_v3_nb.txt` — pristine vs final
  on the same xsimd-fork build.
- `bench/v3_final_compare.txt` — per-cell delta with MdAPE columns.
- `bench/phase23_prefetch_v3_nb.txt`, `bench/phase24_v3_nb.txt`,
  `bench/v3_phase24_compare.txt` — intermediate iteration data.
- `report/perf.data`, `report_v1/`, `report_v2/`, `report_v3/` —
  successive perf-record snapshots.

## Iteration 4 — sort-driven descent (Plan §3.1) — REJECTED

**Hypothesis:** sort points into subtree-major order before descent so each
subtree's `nodes_` slice stays L1-resident across its run. Counting sort on
`get_linear_bin(x)` is cheap (~5–15 c/pt) and would amortize over the
descent's ~5 c/level × depth work.

**Implementation:** added a sort pre-pass to the batch `operator()` (gated
by `total_bins > 1 && leaf_index_by_global_node_.size() >= 1024`):

1. Pass 1 — in-domain check + `get_linear_bin` + per-subtree histogram.
2. Prefix sum on subtree counts.
3. Scatter `xp` → `xp_sorted` (subtree-major), remembering caller indices in
   `sub_perm[]`.
4. Per-subtree descent on `xp_sorted` writing `leaf_ids` and the existing
   leaf-count histogram.
5. Existing leaf-major scatter from `xp_sorted` (instead of `xp`) into
   `xp_packed`, with `perm[dst] = sub_perm[s]` to combine the two
   permutations.

**A/B (`bench/baseline_post23_nb.txt` vs `bench/sort_descent_v2_nb.txt`,
gated form, taskset -c 2):**

| cell                              | A     | B     | B/A  |
|-----------------------------------|-------|-------|------|
| 2d_bump deg=8 dim=2 N=1000000     | 20.24 | 18.29 | 0.90 |
| 2d_mq   deg=8 dim=2 N=1000000     | 51.48 | 45.33 | 0.88 |
| 2d_osc  deg=8 dim=2 N=1000000     | 37.73 | 31.52 | 0.84 |
| 2d_bump deg=10 dim=2 N=1000000    | 24.76 | 19.81 | 0.80 |
| 3d_gauss deg=8 dim=3 N=1000000    | 22.82 | 22.96 | 1.01 |
| 3d_imq  deg=8 dim=3 N=1000000     | 18.92 | 20.13 | 1.06 |
| 3d_yukawa deg=8 dim=3 N=1000000   | 19.72 | 20.34 | 1.03 |

Net: 2D large-N regresses ~10–20 %, 3D large-N flat to +6 %, 1D mixed.
Per the iteration discipline (B/A < 1.0 on stable cells = stop and
re-investigate), reverted.

**Why it didn't work:** the sort adds an extra full pass over `xp`
(in-domain + `get_linear_bin`) and an extra `input_dim · n_trg`
materialization of `xp_sorted`. At N=10⁶ in 2D that is 16 MiB of
read+write traffic that overflows L2 (24 MiB on alderlake) and contends
with the working set. The locality benefit on the descent — which after
Phase 2.3 + 2.4 is already a single L1-resident dependent-load chain —
is too small to offset that bandwidth cost. The 3D cells, where descent
is deeper and the per-subtree node slice larger, get closer to a wash.

**Where to try next:**
- Drop `xp_sorted` materialization (sort indices only, descent reads
  `xp` via indirection). Saves the 16 MiB write+read, costs an L1 gather
  per descent. Worth a single-shot A/B if Phase 3.2 doesn't land first.
- Phase 3.2 (xsimd-gather descent): now the sole on-deck candidate. Uses
  the masked-load primitives already wired into the xsimd fork; descends
  W lanes simultaneously with `xsimd::gather` on `nodes_`. Bigger swing
  on AVX-512 hosts where W = 8.

Files for the rejected attempt are kept for reference:
- `bench/sort_descent_nb.txt`, `bench/sort_descent_v2_nb.txt` — raw runs
  (ungated and gated form).
- `bench/v4_compare.txt`, `bench/v4_compare_v2.txt` — A/B against the
  post-Phase-2 baseline `bench/baseline_post23_nb.txt`.


## Iteration 5 — sort-indices-only retry (FINUFFT-style, simdref-driven)

### What changed (now reverted)

Replaced the single-pass traversal in
`Function::operator()(double*, double*, std::size_t)` with a four-stage
pipeline that keeps `xp` in its caller buffer (no `xp_sorted`) — the one
discriminating choice from FINUFFT's bin-sort vs. iter-4 — and routes
descent through an index permutation only:

1. **Stage 1** — bin pass over `xp`: `get_linear_bin` per point + per-
   subtree histogram. No descent. No `nodes_` access.
2. **Stage 2** — exclusive prefix sum on `sub_counts` → `sub_offsets`.
3. **Stage 3** — `sub_perm` scatter: `sub_perm[sub_offsets[sid[i]]++] = i`.
   4 B per point (uint32) — never touches `xp` coordinates.
4. **Stage 4** — subtree-major descent: walk each subtree's slice of
   `sub_perm`, gather `xp[i]` via the index, descend through that
   subtree's `nodes_` only. Fills `leaf_ids[orig_i]` and the existing
   leaf-count histogram.
5. Existing leaf-major scatter / per-leaf SIMD eval / output permute-
   back: unchanged.

`total_bins == 1` and `n_trg < kSortThreshold` short-circuit to the
existing single-pass.

### Result — fails the acceptance gate

Both runs taskset -c 2, same idle system, within minutes of each other
(the saved `bench/baseline_post23_nb.txt` from earlier sessions reflects
a different turbo / thermal state — re-baselined first to avoid drift).

| cell                              | base   | v5    | B/A  |
|-----------------------------------|-------:|------:|-----:|
| 2d_bump  deg=10 dim=2 N=10⁶       |  8.22  |  8.03 | 0.98 |
| 2d_bump  deg=6  dim=2 N=10⁶       |  3.89  |  4.21 | 1.08 |
| 2d_bump  deg=8  dim=2 N=10⁶       |  8.03  |  8.09 | 1.01 |
| 2d_mq    deg=8  dim=2 N=10⁶       | 27.42  | 28.43 | 1.04 |
| 2d_osc   deg=8  dim=2 N=10⁶       | 11.69  | 12.62 | 1.08 |
| 3d_gauss deg=10 dim=3 N=10⁶       |  9.68  |  9.81 | 1.01 |
| 3d_gauss deg=6  dim=3 N=10⁶       |  4.41  |  4.66 | 1.06 |
| 3d_gauss deg=8  dim=3 N=10⁶       |  4.44  |  4.48 | 1.01 |
| 3d_imq   deg=8  dim=3 N=10⁶       |  4.19  |  4.21 | 1.00 |
| 3d_yukawa deg=8 dim=3 N=10⁶       |  4.23  |  4.25 | 1.01 |

Stable-cell regressions outside large-N (e.g. `3d_gauss deg=6 dim=3 N=1
→ 0.86`, `1d_runge deg=6 dim=1 N=1024 → 0.22` — the latter is an
outlier, but well outside noise on a stable cell), only 2/5 2D-large-N
and 1/5 3D-large-N cells reach B/A ≥ 1.05. Per the plan's gate:

> If 2D-large-N still regresses, the locality win is not real on this
> uarch; we skip to Phase 3.2 (xsimd-gather descent).

2D-large-N is flat (no regression, no clear win), so the strict
"regresses" trigger doesn't fire — but the broader gate (≥1.05 on
large-N + no stable-cell regression) fails. **Reverted.**

### Why it didn't work

The hypothesis was that `nodes_` cross-subtree miss rate in today's
descent is high enough that subtree-major reordering would amortise
the extra uint32 traffic + extra `xp` gather. The measurements say
otherwise on alderlake P-core:

- The new pipeline reads `xp` three times instead of two (Stage 1 seq +
  Stage 4 gather + leaf scatter seq). At 2D N=10⁶ that's 48 MiB vs
  32 MiB of `xp` traffic — most of it L3-resident, but the extra
  Stage-1 + Stage-4 passes also blow ~16 MiB of fresh uint32 scratch
  through L1.
- After Phase 2.3 + 2.4, the descent is already a single L1-resident
  dependent-load chain on `nodes_`. The cross-subtree miss rate is
  low enough that a per-subtree "warm" descent saves few cycles —
  the iter-4 finding ("locality benefit too small to amortise the
  bandwidth cost") survives the bandwidth fix.
- Single-bin trees and tiny batches go through the legacy fall-through
  path, so we don't lose those — we just don't gain anywhere either.

### Where to next — Phase 3.2

Per the plan's exhausted-design-space rule, this design space is closed.
On-deck: **Phase 3.2 (xsimd-gather descent)**. Same single-pass shape
as today, but descend W lanes in parallel with `xsimd::gather` on
`nodes_`. The xsimd-fork's masked-load primitives already enable it.
Bigger swing on AVX-512 hosts where W = 8.

Files kept for reference:
- `bench/sort_idx_v5_nb.txt` — v5 raw run.
- `bench/v5_compare.txt` — A/B vs the on-system re-baseline.
- `report_v5_baseline/perf.data` — perf data captured against the
  pre-v5 binary; the simdref annotate pipeline timed out under the
  contended workload, so no `summary.md` / `hot.sa` were produced.

## Iteration 6 — profile-anchored stop (C1 rejected, optimisation closed)

After iter-5's null result we re-anchored on a fresh perf profile of
operator() instead of more cost-model arithmetic. A separate
perf-symbol build (`build-perf` with `-O3 -march=native
-fno-omit-frame-pointer -g`) plus a focused driver
(`baobzi_perf_driver`, 2d_bump deg=8 N=10⁶ + 3d_gauss deg=8 N=10⁶,
15s each, `taskset -c 2 perf record -F 4000`, no call-graph,
~121k samples) gave a clean source-line attribution.

### Bucket profile (sum across `--sort=srcline,symbol`)

| Bucket                                 | % cycles |
|----------------------------------------|---------:|
| Descent loop + leaf-hist               |     ~55% |
| Per-leaf SIMD eval (avxintrin.h)       |     ~20% |
| Input scatter (lines 1057–1068)        |     ~ 8% |
| `stl_vector` indexing (resize, `[]`)   |     ~ 8% |
| Output permute-back (lines 1127–1133)  |    ~6.5% |

`perf stat`: IPC = 1.64; L1-d miss 2.98%; **LLC-miss/instruction =
3.7e-5** (≈ 0.04 / 1k inst). Branch-miss 0.12%. The hot path is **not**
DRAM-bound; it is dominated by the descent's serial dependent-load
chain on `nodes_`.

### Decision per the plan rules

- C1 (drop `out_packed`): permute ≥ 8% **OR** LLC-miss ≥ 3% — borderline
  (6.5% / 0.01%); attempted because `stl_vector` resize hits push the
  combined attack surface to ~14%.
- C2 (xsimd-gather descent): descent ≥ 15% **AND** LLC-miss < 3% — gate
  technically met (55% / 0.01%), but alderlake P-core has firmware-
  disabled AVX-512 and microcoded `vgatherdpd`. Plan flags this as
  repeating the v5 mistake without a prerequisite descent-only
  microbench. Held in reserve.
- C3 (declare done): polyfit ≥ 60% — not met (20%).

C1 attempted first.

### C1 patch — fuse permute-back into per-leaf scatter

Removed the n_trg-sized `out_packed` buffer; per-leaf eval writes a
small reused scratch (`out_leaf`, sized to `max(counts)`) and is
followed immediately by a scatter into `res` via `perm`. Bandwidth
saved per call: `output_dim · n_trg · sizeof(double)` of read + same
of write (16 MiB at 2D N=10⁶, output_dim=1).

### Result — rejected

A/B (`bench/baseline_phase6_nb.txt` vs `bench/phase6_nb.txt`,
`bench/phase6_compare.txt`):

| stable cell                       |   A   |   B   |  B/A |
|-----------------------------------|------:|------:|-----:|
| 1d_runge   deg=10 dim=1 N=10⁶     | 21.58 | 15.29 | 0.71 |
| 2d_bump    deg=8  dim=2 N=10⁶     | 10.48 |  7.27 | 0.69 |
| 3d_imq     deg=8  dim=3 N=10⁶     |  4.18 |  4.00 | 0.95 |
| 3d_yukawa  deg=8  dim=3 N=10⁶     |  4.15 |  3.98 | 0.96 |

A few cells went up (`2d_osc deg=8 N=10⁶` 1.55, `2d_bump deg=10 N=10⁶`
1.55) but several stable cells regressed below B/A = 1.00. Per gate:
**any stable-cell regression below 1.00 disqualifies. Reverted.**

### Why C1 didn't work

C1 replaces one large sequential write + sequential-read + scatter
(`eval → out_packed`, `permute-back`) with many small per-leaf
scatters. On large N, both shapes have the same number of scattered
writes to `res`; the old path additionally streamed `out_packed`
sequentially, which the hardware prefetchers handle well. The new
path serialises the per-leaf scatter against the SIMD eval (no
overlap) and loses the streaming-prefetch advantage on the read side,
while the absolute bandwidth saved (16 MiB on a 36 MiB-LLC alderlake)
is already L3-resident in the old code — no DRAM round-trip removed.
Net: a wash plus a pipelining penalty.

### Closure — single-threaded operator() optimisation closed

Per the plan's exhaustion rule:

> If C1 is chosen and fails the gate, revert; the design space for
> single-threaded eval is then exhausted and we transition to C3.

The descent is the sole remaining lever and it is dominated by a
serial dependent-load chain that AVX-2 gather cannot accelerate on
alderlake P-core (microcoded `vgatherdpd`, no AVX-512). Pursuing C2
without AVX-512 silicon repeats the iter-5 mistake.

**Phase 6 closed. No code change shipped from the perf work.** The
profile artefacts (`report_phase6_baseline/{perf.data, top.txt,
srcline.txt, stat.txt, buckets.md}`), the rejected A/B
(`bench/phase6_nb.txt`, `bench/phase6_compare.txt`), and the
re-baselined session-anchored reference
(`bench/baseline_phase6_nb.txt`, replacing
`bench/baseline_post23_nb.txt`) are kept as evidence. A new perf
driver (`examples/c++/baobzi_perf_driver.cpp`) ships in-tree so future
profile work can re-run the same focused capture without disturbing
the full microbench harness.

The next material step on this branch is the queued fit-time changes
(`allow_max_depth_leaves`, `max_memory_mib=64`) — they land on their
own commits and have nothing to do with eval performance.

## Iteration 7 — instruction-level closure (asm-analysis)

Phase 6 closed at the **bucket level**: descent ~55 % of cycles,
polyfit eval ~20 %, scatter ~8 %, `stl_vector` ~8 %, permute-back
~6.5 %, LLC-miss/inst = 3.7e-5. The closure rested on (i) a
second-hand claim that `vgatherdpd` is microcoded on alderlake P-core
and (ii) a bucket-level inference that the descent is a serial
dependent-load chain. Phase 7 verifies (ii) at the instruction level
via the `asm-analysis` skill. Default expected outcome: **A1 — closure
strengthened with asm evidence, no code change.**

### Pipeline run

Skill stage map (per `simdref/skills/asm-analysis/references/workflow.md`):

| Stage | Action | Output                                      |
|------:|--------|---------------------------------------------|
| 0 | preflight `simdref --version`, `simdref profile run --help` | `simdref 0.0.0-dev` from local checkout, profile subcommand present |
| 1 | compile-line resolution: `objdump -d` on shipped `build-perf/baobzi_perf_driver` (LTO; per-TU `-S` would lose inlining) | `report_phase7_asm/disasm.s` (911 KiB, 21 041 lines) |
| 2 | microarch resolution: `gcc -march=native -Q --help=target` | `alderlake` (Core Ultra 7 155H) |
| 2b | profile-driven region selection: `simdref profile run --target ./build-perf/baobzi_perf_driver --args 15 --adapter perf --event "cycles:u,instructions:u" --duration 60 --arch alderlake --top 5` | `summary.md`, `hot.sa`, `merged.json`, `annotated.json`, `loops.json`, `samples.json`, `perf.data` (11 MiB, 120 k samples) |
| 4 | annotate hot region | covered by `simdref profile run`'s `annotated.json`; manual re-annotation of a hand-extracted slice failed (see "simdref findings" below) |
| 4a | sanity-check via `simdref show` on dominant mnemonics + cross-reference uops.info / Intel Intrinsics Guide | one mismatch surfaced (see below) |
| 5 | `simdref llm batch --source-kind measured --preset intel` on descent mnemonics | all returned `no_match` (separate simdref issue, see below); the `simdref show`-derived numbers were used instead |
| 6 | llvm-mca cross-check | `report_phase7_asm/mca_descent.txt` |

All artefacts under `report_phase7_asm/` (gitignored via `/report*`).
`baobzi_perf_driver` arg is **seconds per shape**, not iterations
(driver loops until wallclock exceeds the arg). 15 s per shape × 2
shapes = ~30 s of measured eval per run.

### What the asm shows — descent (the dominant bucket)

Source: `function_impl.hpp:741–760`,
`PolyTree::get_node_index(const input_type &x)`.

The 3D descent body (fully covered by simdref's annotation) at
`main+0x6ee0..main+0x6f30`:

```
6ee1: mov    0x18(%rdi),%edx          # esi = nodes_[curr].first_child_idx (root load)
6ee7: cmp    $0xffffffff,%edx         # is_leaf check (max32 sentinel)
6eea: je     6f34                     # exit if leaf
6ef0: xor    %r15d,%r15d              # child_idx = 0  ─┐
6ef3: vcomisd 0x8(%rcx),%xmm10        # cmp x[1] vs c[1]│
6ef8: seta   %r15b                    # r15b = (x[1]>c[1])
6efe: add    %r15,%r15                # << 1
6f01: vcomisd 0x10(%rcx),%xmm11       # cmp x[2] vs c[2]│  3 lane compares,
6f06: seta   %al                      # al  = (x[2]>c[2])  parallel on rcx
6f09: shl    $0x2,%rax                # << 2            │
6f0d: or     %r15,%rax                # pack lanes 1..2
6f10: vcomisd (%rcx),%xmm9            # cmp x[0] vs c[0]│
6f14: seta   %cl                      # cl = (x[0]>c[0])
6f17: movzbl %cl,%r15d                #
6f1b: or     %r15,%rax                # full child_idx ─┘
6f1e: add    %rdx,%rax                # += first_child_idx (rdx from prev iter's load)
6f21: lea    (%rax,%rax,4),%rdx       # rdx = idx*5
6f25: lea    (%rdi,%rdx,8),%rcx       # rcx = &nodes_[next]   (node stride 40 = 5*8)
6f29: mov    0x18(%rcx),%edx          # esi = nodes_[next].first_child_idx  ←─── load-use chain
6f2c: cmp    $0xffffffff,%edx         # is_leaf check
6f2f: jne    6ef0                     # back-edge
```

The 2D descent body (`main+0xebd0..main+0xec50`) is structurally
identical with 2 lane compares, node stride 32, and `first_child_idx`
at offset 0x10 (vs 0x18 in 3D).

### Cited per-mnemonic numbers (`simdref show ... --arch alderlake`, all measured)

| Mnemonic        | Latency      | CPI       | Ports                |
|-----------------|-------------:|----------:|----------------------|
| `vcomisd xmm,m64` (lane compare) | 3.0 c | 1.0 | `1*p0+1*p23A`       |
| `vcomisd xmm,xmm`                | 3.0 c | 1.0 | `1*p0`              |
| `cmp r32,i32` (leaf-test)        | 1.0 c | 0.20| `1*p0156B`          |
| `xor r32,r32`                    | 1.0 c | 0.33| `1*p0156B`          |
| `or r64,r64`                     | 1.0 c | 0.33| `1*p0156B`          |
| `add r64,r64`                    | 1.0 c | 0.20| `1*p0156B`          |
| `shl r64,i8`                     | 1.0 c | 0.50| `1*p0156B`          |
| `lea r64,[r64+r64*i]`            | 1.0 c | 0.95| `2*p0156B`          |
| `mov r32,m32` (leaf-test load)   | **0.0 c** *(see simdref findings)* | 0.33 | `1*p23A` |

L1 hit latency for an integer load on alderlake P-core is 4–5 c per
Intel's optimization manual and uops.info; llvm-mca's schedule model
agrees (5 c). simdref's catalog entry of `lat=0.0c` for
`MOV (R32, M32)` is logged as a discrepancy below and **not used**.

### llvm-mca cross-check

```
$ llvm-mca-18 -mcpu=alderlake -iterations=100 ...
```

| Variant | Total cyc | per-iter | IPC  | RThroughput |
|---------|----------:|---------:|-----:|------------:|
| 2D descent body, original (2 vcomisd-mem + leaf-test mov-load) | 1710 | **17.1 c** | 0.88 | 3.0 |
| descent with leaf-test load only (lane compares → reg-reg)     |  913 |  9.13 c    | 1.64 | 3.0 |
| descent with all loads → reg-reg (compute floor)               |  609 |  6.09 c    | 2.46 | 3.0 |

Scheduler-queue-full **89.3 %** of cycles in the original variant —
the back-end is starved on operand availability, not dispatch
bandwidth. Strict load-use latency bound. The compute-only floor of
~6 c/iter rises to ~17 c/iter once the three dependent loads are
re-introduced; load latency dominates.

Per-target descent cost on a typical 5-level tree: 5 × 17.1 ≈ 85 c.
This matches the Phase 6 bucket attribution (descent ≈ 41–55 % of
cycles per perf, with the rest going to eval, scatter, permute-back).

### Single hottest individual instruction

```
$ perf annotate -i report_phase7_asm/perf.data --stdio --no-source main
   ...
   1.30 :    ec27:  mov    0x10(%rcx),%esi      # leaf-test load
  26.02 :    ec2a:  cmp    $0xffffffff,%esi     # ← attributed by skid to the load
   ...
```

26 % of all cycles are attributed by perf-skid to a single `cmp`
that immediately consumes the leaf-test mov-load — definitive
instruction-level evidence that the descent's binding constraint is
load-use latency on `nodes_[curr].first_child_idx`.

### A2 candidates considered and rejected

For each, the auto-rejection rule the candidate trips, plus the
cycle-level reason it cannot win even on its own merits.

1. **SIMD lane-pack** (replace the per-lane `vcomisd → seta → or`
   chain with `vmovupd → vcmppd → vmovmskpd`):
   - 2D: critical path 5+4+3 = 12 c (load → compare → mskmov) on the
     rcx chain vs. the scalar path's two parallel `vcomisd` (8 c
     each) + 3-uop pack ≈ 12 c. **No win**, and adds an extra
     `and $0x3, %eax` that scalar avoids.
   - 3D: would require a 32-byte YMM load reading 8 bytes past
     `center[3]` into `first_child_idx` (Node stride is 40, so safe);
     critical path ~13 c vs scalar ~12 c. **Slightly slower**.
   - Auto-rejection: in 3D it changes `Node` access semantics
     (reading the next field as part of the lane load and masking it
     out) — close to the spirit of Phase 2.5 axis_stride layout
     fragility. Even with that ignored, it is not a critical-path
     win.

2. **Prefetch the leaf-test load** (`__builtin_prefetch(&nodes_[next])`):
   - Auto-rejection ❌ (Phase 2.4): the next address is strictly
     dependent on the previous load; prefetch can't get ahead. Already
     tried, became the top hot line at 14 %.

3. **Restructure `Node` so `first_child_idx` is at offset 0** (load
   before compare):
   - Doesn't reduce L1 latency (still 4–5 c). The bottleneck is the
     load itself, not the offset.
   - Auto-rejection ❌ (Phase 2.5): adds layout pressure on the
     descent's hot cacheline.

4. **Gather-based descent** (vectorise across W targets, batch the
   tree-load via `vgatherdpd`):
   - Auto-rejection ❌ (Phase 6 C2): AVX-512 firmware-disabled on
     consumer alderlake; AVX2 `vgatherdpd` is microcoded → serialises
     into per-lane scalar loads.

5. **Sort targets by first-step bin to localise node access**:
   - Auto-rejection ❌ (Phase 4, Phase 5): adds at least one extra
     pass over `xp`; the descent's L1-resident chain is too short to
     amortise.

6. **Restructure to two-way Horner inside the eval kernel** (not
   strictly an A2 since it is upstream `polyfit`): the kernel already
   emits two parallel FMA chains (the `ymm15`/`ymm5` interleave
   visible at `f7be..f818`), so the obvious split is already done.
   Any further wins are speculative and belong upstream — see "A3"
   below.

No candidate survives the conjunction of (i) cycle-level critical
path, (ii) auto-rejection rules from prior failures, (iii) the Phase
6 perf gate.

### A3 — polyfit eval kernel (briefly)

The eval kernel (`fast_eval_impl.hpp`,
`detail::horner_nd_acrossPts<DIM, NCOEFFS, batch_t>`) at
`main+0xf680..main+0xfa50` is a textbook FMA Horner with all
coefficients spilled to the local stack frame:

```
f6ae: vmovapd -0x1990(%rbp),%ymm15           # init: ymm15 = c[k]
f6b6: vfmadd213pd -0x19b0(%rbp),%ymm1,%ymm15 # ymm15 = ymm1*ymm15 + c[k-1]
f6bf: vfmadd213pd -0x19d0(%rbp),%ymm1,%ymm15 # ymm15 = ymm1*ymm15 + c[k-2]
... (7 chained FMAs per inner-axis Horner step, lat=4 cpi=0.50) ...
f6f5: vfmadd132pd %ymm6,%ymm15,%ymm2         # ymm2 = ymm6*ymm2 + ymm15  (cross-axis Horner)
```

Per-axis critical path (degree-8 → 7 chained `vfmadd213pd`): 7 × 4 c
= **28 c**. GCC already pipelines two parallel Horner chains
(`ymm15`/`ymm1` and `ymm5`/`ymm0`), visible at `f7be..f818`. No
obvious instruction-level swap — coefficients must be loaded each
FMA because there are too many per evaluation to fit in 16 ymm
registers; FMA is the optimal mnemonic.

A potential upstream tweak (further split a single chain into 2-way
even/odd lanes) would halve the per-axis critical path on paper, but
it is speculative without a polyfit-only profile and doesn't affect
baobzi's bucket boundary. **No A3 issue filed at this iteration.**
A future polyfit profile pass can revisit.

### simdref findings (logged for upstream)

Two issues found while running the pipeline. Both are logged here so
the next asm pass can avoid being misled.

1. **Annotation gap, address range 0xa000..0x10000.**
   `simdref profile run`'s `annotated.json` has 3096 records inside
   `main` body addresses 0xc000..0x14000 but **zero records** in the
   sub-range 0xa000..0x10000, which is exactly where the 2D descent
   body sits (0xebd0..0xec50). 3D descent (0x6ee0..0x6f30) and eval
   kernel (0xf680..0xfa50) are at addresses that fall in this gap or
   just outside it.

   Workaround used: ran `objdump`/`perf annotate` directly on the
   binary; cross-referenced mnemonics via `simdref show <mnem>
   --arch alderlake`.

2. **`MOV (R32, M32)` reports `lat=0.0c` measured on ADL-P.**
   `simdref show mov r32, m32 --arch alderlake` reports `lat=0.0c
   cpi=0.33`. Intel's optimization manual, uops.info ADL-P measured
   tables, and llvm-mca's schedule model all give ~4–5 c for an L1
   hit on integer load. The 17.1 c/iter llvm-mca number for the
   original 2D descent matches a 5 c load latency model; if the load
   were 0 c the same body would run in ≤9 c/iter. simdref's
   `lat=0.0c` value is therefore inconsistent with both the cited
   primary sources and the secondary llvm-mca cross-check. Per skill
   §6 ("flag the disagreement, do not pick a side"), it is recorded
   here and **not used** for any conclusion in this iteration. The
   conclusion (load-use bound) is robust under either value: 0 c
   would still leave a `vcomisd`-dependent chain at ~9 c/iter,
   ~6× the 1-c/iter dispatch floor of a 6-wide back-end.

3. **`simdref llm batch` returned `no_match`** for all the bare
   mnemonics drawn from `annotated.json` (`{xor, vcomisd, seta, add,
   or, movzbl, mov, shl, lea, cmp, jne, imul}`), even with
   `--source-kind measured --preset intel` per the skill workflow.
   `simdref show <mnem>` and `simdref show "<mnem> r32, m32"` both
   work fine, so the mismatch appears to be in the batch-resolver's
   query format. Logged for upstream; the workflow proceeded with
   `simdref show` instead.

### Verdict — A1, instruction-level closure

The descent's binding constraint at the instruction level is the
strictly-dependent L1 load `mov 0x10(%rcx),%esi` (or `0x18(%rcx)` in
3D) followed by `cmp $0xffffffff,%esi`. llvm-mca confirms 17.1
cycles/iter at IPC 0.88 with the scheduler queue full 89 % of the
time — strict latency bound, not port- or front-end-bound. No
instruction substitution within this region (SIMD lane-pack,
prefetch, layout reorder, gather, sort) yields a shorter critical
path while also clearing the auto-rejection rules accumulated across
iterations 2.4, 2.5, 4, 5, and 6.

**Phase 7 closes as A1.** The bucket-level closure from Phase 6
becomes instruction-level closure backed by:

- `report_phase7_asm/perf.data, summary.md, hot.sa, merged.json,
  annotated.json, loops.json, samples.json, disasm.s` — simdref
  profile-run artefacts (gitignored)
- `report_phase7_asm/asm-mca-{descent,leafonly,noload}.s` and
  `report_phase7_asm/mca_descent.txt` — llvm-mca cross-check
- `report_phase7_asm/descent_annotation.txt` — per-instruction
  alderlake annotation for the 3D descent body, plus perf source-line
  attribution

No code change. The three commits on this branch as of Phase 7 entry
(`d4f34da`, `308ec31`, `3ab6664`) remain unchanged. The next step on
this branch is unrelated to eval performance.

## Iteration 8 — peak-gap math + thread-safety contract

This iteration answers two questions: where is single-thread baobzi
relative to the silicon's DP-FMA peak, and how should callers
parallelise. The first is a quantitative bound; the second is a
contract on `operator()` that lets callers thread externally.

### Single-thread peak vs measured

Tensor-product Horner ND with K = 9 coefficients per axis costs
`K^n − 1` FMAs per evaluation: 8 (1D), 80 (2D), 728 (3D). Phase-7
perf-driver throughput on `taskset -c 2` was 6.60 Mevals/s at 2D
deg=8 (1.06 GFLOPS) and 3.86 Mevals/s at 3D deg=8 (5.62 GFLOPS).

Core Ultra 7 155H P-core peak: 2 × FMA-256 ports × 4 doubles ×
2 flops × 4.8 GHz ≈ 76.8 GFLOPS DP single-thread. The measured
fractions are **1.4 %** (2D) and **7.3 %** (3D) of FMA-limited
single-thread peak.

### Why the gap is closed at single thread

Phase-6 attribution: descent ~55 %, polyfit FMA kernel ~20 %,
input scatter ~8 %, `stl_vector` indexing ~8 %, output permute
~6.5 %. An infinitely fast FMA kernel only removes the 20 % FMA
bucket → ceiling 1.25× single-thread. Phase-7 proved the descent
is at the silicon's load-use floor (17.1 c/iter, IPC 0.88,
scheduler-queue full 89 %). Closing the rest of the peak-gap needs
either non-microcoded gather (Sapphire Rapids, Zen 5) or parallel
hardware. The user's stated direction is *callers parallelise
themselves*; Phase 8 makes that contractually safe.

### The contract that landed

> A single `Function` built once and not subsequently mutated may
> be called concurrently from many threads, provided each call
> writes to a disjoint output slice. Per-call scratch lives in
> `thread_local` storage; the Function's nodes / polyfits /
> coefficient state are immutable after construction. Baobzi does
> not parallelise internally.

Documented at the top of `include/baobzi/baobzi.hpp`, on both
`operator()` overload declarations in
`include/baobzi/detail/function_impl.hpp` (lines 1082, 1231), and
in `README.md` ("Thread safety" section).

### Test that pins the contract

`tests/test_threadsafe.cpp` — Catch2 suite, registered in
`cmake/baobzi_tests.cmake`, hooked into `ctest` (33/33 green).
Two test cases (`2d_bump deg=8` and `3d_gauss deg=8`):

- Build one Function, allocate 65 536 random in-domain inputs.
- Spawn 8 threads through `std::latch` (gates simultaneous start,
  not spawn-time serialisation), each calls
  `f(xp_chunk, res_chunk, n_chunk)` over disjoint slices.
- Repeat 16 times. **Bit-exact** match across repeats — that is
  how a true race surfaces (any flap on shared state would change
  bits between repeats).
- Compare threaded result against a serial reference under a
  `1e-12` relative tolerance.

### Finding: chunking-dependent ~1 ULP drift in polyfit's batch kernel

Initial drafts of the test asserted bit-exact equality against the
serial reference. The 3D case failed: 1 588 / 65 536 outputs
diverged by exactly 1 ULP. Investigation:

- Repeated threaded calls are bit-exact across all 16 repeats →
  no race on shared state.
- `polyfit::FuncEvalND<…>::operator()(pts, out, count)` for
  `OUT_DIM == 1` switches between (a) an unrolled across-points
  SIMD batch (`U·B` points/iteration), (b) a non-unrolled SIMD
  path (`B` points/iteration), and (c) a scalar
  `evalCanonical<>(pts[i])` tail. Path (c) uses a Horner kernel
  with a different FMA-fusion shape from paths (a)/(b).
- Splitting 65 536 points across 8 threads changes per-leaf
  `cnt` distributions, which changes how many points land in the
  scalar tail per leaf, which changes which Horner shape sees
  them. This is non-associative FP across kernels, not a race.

The drift is bounded ≤ 1 ULP and well below the fit's 1e-10
tolerance. The test bounds it at 1e-12 relative, which both 2D
and 3D clear comfortably. Recorded here so future readers don't
reinterpret it as a race signal: **bit-exact across-repeat
identity** is the correct racing-asserts probe; threaded vs
serial is informational and chunking-sensitive.

### What did NOT land (explicit non-goals, per plan)

- No new `operator()` overload, no `n_threads` argument.
- No internal thread pool, `std::async`, OpenMP, TBB, or PSTL.
- No automatic chunk-parallelisation.
- No relaxation of the across-repeat bit-exactness assertion in
  the test.
- No layout / coefficient-store change (Phase-7 + iter-2.5/4/5
  lessons stand).

### Side-thread (Phase 8b) status

Bounded ≤ 1 hr asm-analysis pass on the polyfit FMA kernel was
deferred — it is independent of the contract that just landed and
ships upstream against `polyfit`, not into baobzi. Open as
follow-up.

### Files

- `tests/test_threadsafe.cpp` — new
- `cmake/baobzi_tests.cmake` — registers `test_threadsafe`,
  links `Threads::Threads`
- `include/baobzi/detail/function_impl.hpp` — doxygen on both
  batch `operator()` overloads
- `include/baobzi/baobzi.hpp` — top-of-file thread-safety contract
- `README.md` — "Thread safety" section

## Iteration 9 — slim-node descent (Phase 9, Layers A + E)

### What landed

A single commit on `use-polyfit` shipping Layers **A + E** of the
Phase-9 plan:

- **Layer A — slim Node.** Drop `center` and `_pad_` from the
  runtime `Node`; shrink `poly_eval_id` from `uint64_t` to
  `uint32_t`. Total `sizeof(Node) == 8 B` for every `Dim` (locked by
  `static_assert` in `function_impl.hpp`). `find_node` /
  `get_node_index` no longer reads `center` per level — it carries
  the subtree's `(lower, upper)` bounds in registers and recomputes
  `mid = 0.5 * (lo + hi)` on the fly. Per level the descent loads
  drop from `Dim×8 B (center) + 4 B (first_child_idx)` to just
  `4 B (first_child_idx)` against the dep-chain. Storage density:
  8 nodes/cacheline vs 1.6 (3D) / 2.0 (2D) / 2.6 (1D) before.
- **Layer E — packed compare.** Bundled with A; the per-axis
  loop-form (`for d in 0..Dim: x[d] > mid[d]`) is left for the
  compiler to vectorise.

### Numbers (Core Ultra 7 155H, P-core, `taskset -c 2`)

| Scenario               | Phase 7   | Phase 9 (A+E) | Δ        | Target |
| ---------------------- | --------- | ------------- | -------- | ------ |
| 2d_bump deg=8 N=1e6    | 6.60      | **10.00**     | **+51.5 %** | ≥ 10 |
| 3d_gauss deg=8 N=1e6   | 3.86      | **8.06**      | **+109 %**  | ≥ 5  |

Both stop-criterion targets are hit with margin → Layers C, D, B,
G are deferred per the plan ("If hit, stop — bank the win").

### What actually fired

`objdump -d --disassemble=main baobzi_perf_driver | grep -cE
'vcmpgtpd|vmovmskpd|vcomisd'` returns 17 (all `vcomisd`); zero
packed compares. Layer **E did not auto-vectorise** under
GCC 15.2 `-O3 -march=alderlake` — the per-axis compare chain
remained scalar. **Layer A alone delivered the win.** This is
consistent with the plan's prediction that the bottleneck was load
volume, not compare throughput: removing `Dim×8 B` of per-level
center loads from the L1 dep-chain was the actionable lever.

If a future need arises, Layer E can be forced via explicit AVX2
intrinsics (`_mm256_cmp_pd` + `_mm256_movemask_pd`) for `Dim ∈
{2,3}` — but only if a profile shows compare latency, not load
volume, has become the new gate.

### Cross-checks

- **Tests:** all 33 ctest cases green, including
  `test_threadsafe` (1e-12 relative tolerance, 8-thread bit-exact
  across-repeat). The descent's `mid = 0.5*(lo+hi)` differs by
  ≤ 1 ULP from the fit-time `box.center` chain at boundary points;
  expected and within the contracted tolerance.
- **Slim-node invariant locked:** three `static_assert(sizeof(Node)
  == 8)` for 1D, 2D, 3D in `function_impl.hpp` — the next change
  that re-fattens the node fails the build.

### What did NOT land

- Layer C (post-fit DFS reorder of `nodes_[]`) — deferred (target hit).
- Layer D (post-fit Z/Hilbert leaf permute) — deferred (target hit).
- Layer B (per-axis u64 quantise) — deferred (E underperformed but
  total target met regardless; B was contingent on the *combined*
  result missing target).
- Layer G (W=4 batched-descent SIMD-across-points) — stretch tier;
  deferred (target hit, prototype ungated).
- No change to `polyfits_[]` storage, `Box` structure, or the
  thread-safety contract.

### Files

- `include/baobzi/detail/function_impl.hpp` — slim Node;
  `Node::fit` / `force_fit_as_leaf` take `center` as a parameter;
  `PolyTree` carries `(lower_, upper_)` and the descent rewrites
  `get_node_index` to recompute `mid` per level; three
  `static_assert(sizeof(Node)==8)` checks; minor `Function` ctor
  refactor (use `current_box.center` where the old code touched
  `node.center`).

