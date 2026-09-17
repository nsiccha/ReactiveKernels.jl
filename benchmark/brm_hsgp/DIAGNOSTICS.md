# What work accounts for the motorcycle timings?

The full v2 comparison verifies the same normalized density and all 44
derivatives against both references. That establishes mathematical parity;
it does not require the compilers to execute identical operations.

The longer full benchmark puts native Enzyme within 1.02–1.20× the emitted
Stan target. The first, shorter run overstated that gap. Reactant still costs
1.50–1.98× Stan with resident inputs and synchronized results.

## Controlled measurements

Both diagnostic scripts use the same data, NCP posterior column 5000, Float64,
and one BLAS thread. They ran pinned to strato2 CPU7, while the full reference
benchmark ran on CPU6. The shared host was not reserved. Each measurement has
a two-second budget and at most 100,000 samples, with `evals=1`. Compare controls
within their run; differences between separately timed calls are not an
additive profile of one execution.

```sh
KB_COMPACT_KEEP_LOG=1 kb-run-compact taskset -c 7 \
  julia --startup-file=no --project=benchmark/brm_hsgp \
  benchmark/brm_hsgp/diagnose.jl "$TMPDIR/brm-benchmark" "$TMPDIR/brm-diagnostics"
KB_COMPACT_KEEP_LOG=1 kb-run-compact taskset -c 7 \
  julia --startup-file=no --project=benchmark/brm_hsgp \
  benchmark/brm_hsgp/diagnose_matvec.jl "$TMPDIR/brm-benchmark" "$TMPDIR/brm-matvec-grouped-diagnostics"
```

The first command exited 0 in 152 s (`kb-run-compact.RkTFs2`); the second
exited 0 in 57 s (`kb-run-compact.CkZlq0`). Receipts retain the script hashes:

- [Preparation/allocation/call controls](../receipts/brm-hsgp-diagnostics-v1.toml)
- [Dense-projection controls](../receipts/brm-hsgp-matvec-diagnostics-v1.toml)

| Value + gradient control | Native median μs | Reactant median μs |
|---|---:|---:|
| Model, live centeredness | 8.42 (Enzyme) | 16.26 |
| Model, bound NCP centeredness | 6.95 (Enzyme) | 14.76 |
| Cheap scalar + 44-vector output | — | 5.76 |
| Quadratic, two separate projections | 1.37 (analytic) | 15.98 |
| Same quadratic, grouped right-hand sides | — | 15.67 |

The quadratic is a control containing the two `133×20` projections and two
reverse projections, with the same scalar-plus-44-vector output shape. Its
explicit derivative is used only for this control, never as a replacement
model or an acceptance reference. Native and Reactant control values and
gradients are checked against each other.

## What the code and profiles establish

BRM emits an NCP-specific Stan model with standard-normal weight priors and
no partial-centering Jacobian. RK keeps the 40 centeredness values live so one
executable supports changing frames. Both prepare the observed-data basis once.
The optimized RK primal StableHLO contains a constant `133×20` basis and no
sine operation. The data basis is not being recomputed in either gradient path.

The emitted RK primal has two matrix-vector projections, two weight-prior
dot products, and two centeredness-Jacobian dot products. Binding NCP
centeredness removes the latter two, but zero-times-log-scale terms feeding
exponentials remain in this StableHLO under its floating-point semantics.
Thus binding controls alone does not produce the same NCP algebra as the
specialized Stan source. These counts describe optimized StableHLO, not final
machine instructions.

The native Enzyme allocation profile samples every allocation across 20 calls:
800 arrays total, or 40 per call. Their stack traces identify eight slice
arrays, 28 broadcast-result arrays, and four matrix-product arrays per call.
There are 36 length-20 arrays and four length-133 arrays per call. BenchmarkTools
reports 12,736 Julia bytes per call; binding NCP centeredness reduces this to
10,944 bytes and 32 allocations. Preparation reuses AD setup and caller-owned
gradient output, but does not preallocate these intermediates. A CPU profile
also samples array allocation, broadcast work, and BLAS matrix-vector calls.

The existing reusable-buffer primal path reduces Julia allocation from
5,472 to 1,536 bytes, but remains at 14 allocations and takes 2.52 μs versus
1.79 μs for the ordinary primal in that control. It is not an allocation-free
solution for this graph as currently authored, and this diagnostic does not
claim a prepared AD path for that separate execution type.

Reactant's cheap control demonstrates a material synchronous call cost.
The projection-only control reproduces nearly the full model latency, and
grouping right-hand sides barely changes it. This implicates the small dense
linear-algebra workload under XLA as a substantial additional cost; it does
not isolate a particular XLA runtime routine or prove a general compiler limit.
Julia allocation counters also exclude both XLA-managed storage and Stan's
C++ allocations, so they are not a cross-backend memory comparison.

The scripts regenerate full allocation traces, CPU profiles, and StableHLO
in their output directories. The committed receipts contain the compact
measurements; the output traces remain producer-local diagnostic evidence.
