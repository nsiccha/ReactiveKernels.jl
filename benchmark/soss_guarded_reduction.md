# Guarded scalar summaries for repeated evaluation

The follow-up to `soss_symbolic_reduction.jl` preserves unexpanded separable
terms and adds a condition diagnostic before using their summaries. It catches
the tested expansion, aggregation-overflow and cancellation failures while
retaining useful repeated-gradient speedups. It does **not** establish a
generally stable lowering: the diagnostic is heuristic, and the direct path
can itself have an inaccurate gradient under extreme cancellation.

This is a benchmark-only experiment. No production RK lowering, selection
policy, dependency or public API changes. The earlier separator gains an
optional `expand=false`; its default remains the earlier expanded experiment.

## Candidate and diagnostic

`guarded_spec` partitions an explicit, pure scalar SymbolicUtils expression
into `coefficient(a,b) * basis(y)` terms without calling `SU.expand`. SymbolicUtils
still canonicalizes the expression during its construction; this is not a
promise to preserve every operation of arbitrary authored Julia. Mixed factors
such as `(a*y-b)^2` or `exp(a*y)` decline to the direct RK plate. The generic
polynomial/log case keeps `(y+1)^2` as one data basis instead of expanding it.

Binding evaluates each data basis with an ordinary RK callable and Julia's
runtime `map`/`sum`, then prepares a residual RK graph with scalar statistic
ports bound through `prepare`. The original bound direct kernel is retained.
Data, lengths, summaries and the diagnostic threshold remain runtime fields.
Source construction depends only on the scalar expression's structure.

The diagnostic checks finiteness and cancellation for each data reduction,
then for the value and both parameter sensitivities of the scalar residual.
For contributions `t`, it requires either all-zero scale or
`abs(sum(t)) > threshold * sum(abs,t)`. The experimental default threshold is
`sqrt(eps(Float64))`. It is an adjustable diagnostic, **not a proved error
tolerance or a new RK requirement**. Each coefficient has one primal;
ForwardDiff obtains its diagnostic sensitivities. Enzyme differentiates the
returned value through its ordinary reverse path, including fallback selection.
No handwritten derivative, activity override or backend rule was added.

An empty input returns zero without evaluating live coefficients. A failed
data diagnostic bypasses coefficient diagnostics. A failed runtime diagnostic
selects the direct kernel with ordinary lazy control flow. The tested negative
`log(y)` input keeps its `DomainError` at evaluation instead of moving it to
binding. These checks concern the explicit pure scalar fixture, not arbitrary
effectful user functions or branch-dependent model outputs.

## Validation and precision boundary

The final run passed **260/260 checks**, exit 0, on 2026-11-13 from
10:51:28Z to 10:52:43Z (75 seconds including startup). It covered:

- Five expressions, lengths 0/1/32/4096, and three parameter pairs, checked
  against native ordinary Enzyme gradients and independent 256-bit BigFloat
  values and ForwardDiff gradients.
- Empty reductions with infinite parameters, same-array mutation followed by
  explicit rebinding, and unchanged residual expression/callable type across
  lengths and data values.
- Both earlier expanded-square failures: `(a*y-b)^2` at `(1e8,1e8+1)` and
  `(1e160,1e160)` now decline summary construction and return 32 and 0.
- Aggregation overflow: `a*y` on two `1e308` rows at `a=1e-308` has an infinite
  unguarded summary value; fallback returns `1.9999999999999998`. Its derivative
  with respect to `a` exceeds Float64 range, so both gradients are infinite.
- Value cancellation: `a*y-b`, with one `nextfloat(1.0)` and 31 ones at
  `a=b=1e16`, gives 0 through the unguarded summary and 2 through fallback.
- Gradient cancellation and preservation of the tested source-domain error.

The gradient probe is an important limit on the conclusion. For
`a*y+exp(a)`, `a=40`, with one `nextfloat(-exp(40.0))` and 31 copies of
`-exp(40.0)`, the value is about `-2.94e20` and its relative cancellation
diagnostic passes. Its derivative is a tiny difference between large terms,
so the sensitivity diagnostic selects fallback. The measured derivatives are:

| Evaluation | Derivative with respect to `a` |
| --- | ---: |
| Direct RK, ordinary Enzyme | 3072 |
| Unguarded summary RK, ordinary Enzyme | 3072 |
| Native row loop, ordinary Enzyme | 2752 |
| Native row loop, Float64 ForwardDiff | 32 |
| Native row loop, 256-bit BigFloat ForwardDiff | -434.947202856… |

The high-precision reference differentiates the real expression at the exact
stored Float64 data and parameters; those data include already-rounded
`exp(40.0)`. The sensitivity check detects the conditioning problem, but
fallback preserves the direct RK result, not the high-precision derivative.
The test intentionally asserts fallback equivalence here, **not accuracy**.
This does not establish a backend defect: all low-precision paths are operating
on an extremely ill-conditioned calculation with differing arithmetic.

The diagnostic also supplies no error bound on basis evaluation, summation,
coefficient arithmetic or backend derivatives. Passing these finite probes is
not a certificate for unseen states. A production stable summary would need
justified numerical error control or an appropriately stable primal, with both
value and gradient evidence. Merely falling back to an unchecked direct
formula is insufficient for that stronger claim.

## Repeated-evaluation costs

The benchmark cycles through four runtime parameter pairs for 1, 100 and
10,000 calls. Both measured closures specialize on their kernel before timing;
parameters stay in an array and counts remain runtime inputs. These are
evaluation workloads, not MCMC chains: no sampling efficiency, posterior
accuracy or real-chain fallback frequency was measured. All four timed
parameter pairs selected summaries in the reducible cases.

The table gives **one warm binding plus 10,000 ordinary gradient calls**, in
milliseconds. Julia compilation and symbolic graph construction are excluded.
Minimum BenchmarkTools samples were taken on the shared `gordito` host, capped
at one CPU, with load 27.25/22.77/22.42 at admission. Small timing differences
are not reliable rankings. The mixed-exponential control remains direct and
shows only normal wrapper/noise differences.

| Expression | Rows | Direct total ms | Guarded total ms |
| --- | ---: | ---: | ---: |
| Poisson parameter terms | 32 | 0.386 | 0.412 |
| Exponential log-rate | 32 | 0.265 | 0.411 |
| Gamma-like parameter terms | 32 | 2.184 | 0.644 |
| Generic polynomial/log | 32 | 2.982 | 0.555 |
| Poisson parameter terms | 4096 | 2.188 | 0.417 |
| Exponential log-rate | 4096 | 2.079 | 0.400 |
| Gamma-like parameter terms | 4096 | 319.456 | 0.632 |
| Generic polynomial/log | 4096 | 502.875 | 0.573 |

The Gamma-like fixture is the earlier parameter-terms expression, not a
complete Gamma log density. All warm repeated-call measurements allocate zero
bytes on both sides. Binding allocates, including temporary data-basis arrays.

Using steady per-call costs from the 10,000-call batches, the extra warm
preparation cost is recovered after roughly 83 gradient calls for the 32-row
Gamma-like case and 50 for the polynomial/log case. At 4096 rows, Poisson and
exponential recover it after about 82 and 78 calls; the more expensive cases
recover it within one call. These are approximate workload-specific estimates.
At 32 rows, Poisson shows no clear gain and exponential's diagnostic costs more
per call than it saves. More reuse cannot amortize a recurring per-call loss.

The distinction matters for MCMC: repeated evaluation makes a one-time
preparation cost plausible to recover, including on small datasets, but it
does not remove per-call diagnostic costs or repair numerical instability.
Cold compilation cost and fallback frequency would also matter in an actual
end-to-end sampler comparison.

Retaining the direct fallback means storage remains O(n), even though the
summary residual uses only a few scalars. The four guarded kernels retain
approximately 66–90 KB at 32 rows and 99–123 KB at 4096 rows, including graph
metadata and the original data. This experiment makes no O(1)-storage claim.

## Reproduction and evidence

Source baseline: `c7d5b238631ad74ce4ef2cf9953f1f4a59dd972c`; production source
unchanged from `6358fa81faa0d11322993a512ce462397810c14f`. Consumer environment:
Julia 1.10.12, RK 0.1.222, SymbolicUtils 3.32.0, Enzyme 0.13.219,
ForwardDiff 1.4.7. The evidence includes the complete resolved Project and
Manifest; it adds no dependency to RK's own project.

```bash
KB_COMPACT_KEEP_LOG=1 \
KB_COMPACT_LOG_PATH=/home/niko/scratch/soss-guarded-prototype-20261113/run-2.log \
kb-acquire-compute-token --here --why 'Guarded summary experiment' -- \
kb-run-compact -- timeout 900 julia --startup-file=no \
  --project=/home/niko/scratch/soss-guarded-prototype-20261113/env \
  benchmark/soss_guarded_reduction.jl
```

The durable evidence archive contains the final source and commit patch,
environment, resolver, final `run-2.log`, structured CSV/JSON metrics, unmodified
actual `code_expr` displays across lengths, and checksums. The earlier
`run-1.log` passed 177 checks but had field-access boxing in its measurement
setup; only `run-2.log` supplies the reported timings.

The experiment supports investigating automatically discovered, numerically
justified summaries. It does not select a production architecture, authorize
unsafe expansion, or establish this heuristic as RK's admission policy.
