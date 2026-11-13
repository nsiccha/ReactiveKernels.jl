# Soss-style symbolic reduction: bounded RK experiment

The executable prototype discovers reusable data summaries for four scalar
expressions, prepares those summaries through RK's existing `bound` machinery,
and differentiates the residual through ordinary Enzyme Reverse. It also
demonstrates why unrestricted algebraic expansion is not a safe production
lowering: cancellation changes an exact result of `32` to `0`, and overflow
changes an exact result of `0` to `NaN`.

This experiment supports generating a summary-based candidate alongside the
original computation. It does not establish a safe default admission rule or
choose a production architecture.

## Mechanism and scope

[`soss_symbolic_reduction.jl`](soss_symbolic_reduction.jl) takes one symbolic
scalar row expression. SymbolicUtils supplies expansion and symbolic occurrence
queries; a small adapter groups additive factors into data bases and live
coefficients, then constructs ordinary RK graphs. No probability-law registry,
symbolic derivative generator, custom AD rule, or runtime `eval` is involved.

| Row expression | Automatically discovered summaries |
| --- | --- |
| `y*a - exp(a)` | count, sum of `y` |
| `a - exp(a)*y` | count, sum of `y` |
| `(exp(a)-1)*log(y) - exp(b)*y + exp(a)*b` | count, sum of `y`, sum of `log(y)` |
| `sin(a)*(y+1)^2 + cos(b)*log(y) + a*b` | count, sum of `y`, sum of `y^2`, sum of `log(y)` |
| `exp(a*y) + b*y` | declined; the mixed exponential stays in the direct graph |

The first and third expressions are representative probability-related terms,
not complete Poisson/Gamma log densities. The Gamma-like expression omits the
shape normalizer. The second is the exponential log density on positive data;
this fixture supplies positive observations and does not represent support
guards. The fourth is general numerical arithmetic.

Count uses runtime `length`; the identity basis uses Julia's `sum`; other data
bases use RK reduced plates. Each basis is prepared separately, so several
bases can scan the same data several times. RK partial evaluation computes
these data-only nodes once per binding. The residual receives their values
through its constants table. Neither data values nor lengths are inserted into
source. A runtime empty-input wrapper returns zero without evaluating live
coefficients, including when the supplied parameters are infinite.

The adapter accepts these explicit, straight-line scalar expressions and a
total-only demand. It does not extract expressions from arbitrary RK/RKPPL
source, encode authored branches, provide distribution support checks, or
handle extra requested row outputs. Its two scalar live symbols are structural;
it does not expand a data-sized parameter vector. The existing least-squares
implementation and its separate repair/comparison are unchanged.

## Validation and measurements

The recorded environment uses Julia `1.10.12`, RK `0.1.222` at
`6358fa81faa0d11322993a512ce462397810c14f`, SymbolicUtils `3.32.0`, registered
Enzyme `0.13.219`, and Enzyme_jll `0.0.307+0`. The experiment adds no dependency
to RK's package project. Its consumer environment was created with
`consumer_env_init` and `resolve_script` from the workspace resolver; the
evidence archive retains the generated program, Project, and Manifest.

The checks cover lengths `0, 1, 32, 4096, 65536`, three parameter pairs per
case, 256-bit BigFloat values, and ordinary reverse gradients of direct RK,
reduced RK, and a plain Julia row loop. Rebinding an in-place modified array
must recompute the summaries. The actual residual `code_expr` and compiled
callable type must stay identical across different lengths and values. These
tests establish the bounded fixture's behavior, not general numerical admission.

Timing uses BenchmarkTools minima, at most 300 samples per measurement,
`evals=1`, on shared host `gordito` under a one-CPU compute lane. Parameters are
loaded from a runtime vector to avoid folding fixed scalar coefficients into
the benchmark closure. Tiny nanosecond differences are below a useful decision
threshold. Binding times include RK preparation reuse and summary execution;
they exclude first-time graph construction and compilation. Gradients are
gradient-only, not full inference or value-and-gradient throughput.

The log records first-in-process construction, binding, value and gradient
times separately. These include warm-up/compiler costs and share prior work;
they are not an independent cold-start comparison between alternatives.
It also records constants-table size and full reachable kernel size separately.
The latter includes graph metadata; process compiler caches are excluded.

The final run passed **370/370 checks**, exited `0`, and took 101 seconds
(`2026-11-13T10:17:57Z` to `10:19:38Z`). At 65,536 rows, all warm primal and
gradient measurements below allocated zero bytes. Times are microseconds:

| Case | Primal direct / candidate | Gradient direct / candidate | Rebind direct / candidate |
| --- | ---: | ---: | ---: |
| Poisson terms | 24.60 / 0.02 | 3.66 / 0.03 | 5.45 / 9.97 |
| Exponential | 24.40 / 0.02 | 3.70 / 0.03 | 5.03 / 10.65 |
| Gamma-like terms | 356.99 / 0.02 | 768.18 / 0.04 | 11.09 / 322.63 |
| Polynomial/log arithmetic | 488.14 / 0.02 | 868.12 / 0.05 | 5.64 / 489.03 |
| Mixed exponential control | 148.04 / 148.29 | 259.40 / 259.71 | 5.01 / 5.27 |

For these large reducible fixtures, roughly one repeated primal evaluation
amortizes the additional warm binding work. At 32 rows the estimate is tens to
hundreds of calls and is sensitive to timer noise; empty/singleton inputs have
little or no measurable repeated-evaluation benefit. These are observations
from these expressions and this host, not selector thresholds.

Reduced constants tables occupy 24–40 bytes and complete reachable kernels
32,368–78,442 bytes, independent of the tested data length. At 65,536 rows the
direct constants tables occupy 524,336 bytes and complete kernels
560,770–580,826 bytes. For small inputs, extra graph metadata can make the
candidate larger: the polynomial/log candidate occupies 78,442 bytes versus
56,538 bytes for its empty direct kernel. The mixed control retains its array
and loop. The residual expression sizes are fixed at 302, 434 or 496 printed
characters for reducible cases, and 2,352 for the unreduced control.

The earlier `run-5.log` also passed 370 checks but used literal benchmark
parameters. Its timings are superseded by `run-final.log`. Development runs
1–3 failed on script parse/API setup; run 4 exposed the identity-plate boundary
below after 15 passing checks. The archive retains these logs rather than
presenting them as accepted runs.

## Numerical boundaries

For 32 observations all equal to one and the row expression `(a*y-b)^2`:

| Parameters | BigFloat and direct RK | Expanded candidate | Ordinary gradients, direct / candidate |
| --- | --- | --- | --- |
| `a=1e8`, `b=1e8+1` | `32` | `0` | `(-64,64)` / `(-64,64)` |
| `a=b=1e160` | `0` | `NaN` | `(0,-0)` / `(0,0)` |

These are printed counterexamples, not accepted transformations or weakened
correctness tests. A finite-result fallback would miss the first error.
Agreement of gradients also does not establish a correct primal. A useful
candidate needs a stable representation and a justified admission/fallback
condition; exact real-algebra equivalence alone supplies neither.

At this RK pin, a separate recipe-free plate returning its batched HAVE port
has another boundary: its actual generated loop adds the whole array to its
scalar accumulator. Empty data returns zero, while `[1.0]` and `[1.0,2.0]`
raise `MethodError: +(::Float64, ::Vector{Float64})`. Reproduce after including
the experiment with `row_kernel(y; data_only=true)([1.0])`. The evidence contains
the unchanged `code_expr` and exact output. The experiment uses the existing
`sum` operation for this basis, and the finding was sent to the RK owner.

## Reproduction and interpretation

Run from the RK checkout, using the evidence consumer environment or a fresh
resolver-created environment declaring ReactiveKernels, SymbolicUtils 3,
Enzyme 0.13, and BenchmarkTools:

```sh
julia --startup-file=no --project=/path/to/consumer/env benchmark/soss_symbolic_reduction.jl
```

The evidence archive records source, exact dependency provenance, complete
logs and exit receipts, parsed measurements, and checksums. Its absolute
development path refers to the original checkout; on another host, regenerate
the consumer environment with the resolver against the recorded source pin.

RK can reuse established symbolic algebra to discover bases, and its existing
binding and AD paths can execute the result. Before production integration,
the candidate must preserve demanded outputs and source guards, pass numerical
admission, and compete on binding cost, repeated-evaluation cost, and retained
memory. Small datasets may never amortize the extra preparation. More bases
also increase preparation work and metadata; these scalar fixtures do not
establish behavior for high-dimensional or grouped models.

The historical source motivation is Soss `0.18.3`'s
[symbolic pipeline](https://github.com/cscherrer/Soss.jl/blob/23b090e7b29fd523f29cc1f487ffb4f8e0edf810/src/symbolic/symbolic.jl)
and SymbolicCodegen `0.2.2`'s data folding. The earlier historical environment
failed during package loading and did not reproduce that demo. This experiment
is a new RK implementation of the transferable idea, with its own measured
evidence.
