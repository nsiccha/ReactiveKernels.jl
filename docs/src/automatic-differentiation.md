# Automatic differentiation

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

ReactiveKernels exposes automatic differentiation through
`DifferentiationInterface`. The package owns the prepared-kernel boundary, not
a concrete differentiation engine: Enzyme is an optional test and example
dependency, and core package source never imports it.

`prepare_ad` resolves one active HAVE port and one scalar WANT once. Every
other selected HAVE is supplied to the backend as a freshly rebound
`Constant`, so preparation fixes types and shapes without freezing the values
used by later calls.

## Prepare once, then request gradients or value-and-gradient

This build-executed example shows the actual kernel definition and both ways to
interact with its prepared differentiation boundary:

```@example automatic_differentiation
using ReactiveKernels
using DifferentiationInterface
import Enzyme

backend = AutoEnzyme(; mode = Enzyme.Reverse)

@kernel objective(q::Vector{Float64}, scale::Float64 = 1.25;
                  data::Vector{Float64}, offset::Float64 = 0.0) = begin
    density::Float64 =
        sum(q .* data) - scale * sum(abs2, q) + offset
end

parameters = [0.3, -0.4, 0.2]
data = [2.0, -1.0, 0.5]
prepared = prepare_ad(
    objective, backend, parameters;
    data, active = :q, want = :density,
)

gradient = ad_gradient(prepared, parameters; data)

gradient_buffer = similar(parameters)
value, returned_gradient = ad_value_and_gradient!(
    prepared, gradient_buffer, parameters; data,
)

@assert returned_gradient === gradient_buffer
@assert gradient ≈ data .- 2(1.25) .* parameters
@assert gradient_buffer ≈ gradient

(; value, gradient = copy(gradient_buffer), caller_owned = true)
```

`ad_gradient` returns only the derivative. The prepared-only
`ad_value_and_gradient!` returns `(value, gradient)` and fills the caller-owned
destination in place. Both calls preserve the authored positional defaults and
keyword interface while rebuilding the inactive `Constant` context from the
current arguments.

The boundary is deliberately narrow:

- the requested WANT must be scalar and explicit when the `KernelSpec` has more
  than one possible output;
- exactly one HAVE port is active;
- integer active ports and aliased active boundaries reject;
- an inactive HAVE downstream of the active port rejects, because treating it
  as constant would cut a real derivative path; and
- the stored backend preparation is reusable but not thread-safe, so concurrent
  callers use separate prepared objects.

Plain `AutoEnzyme(mode = Enzyme.Reverse)` is the supported example
configuration. Runtime-activity mode and function annotations are not part of
the ReactiveKernels boundary.

## One reverse pass over a plated objective

The same API applies to an authored likelihood whose `plate` result is summed.
The build-executed source below extends the exact primal source from
[Batched log densities](batched.md): the canonical `normal` object, the one
authored graph, its return/pointwise/both query boundaries, and the prepared
reverse pass. One `ad_gradient` call differentiates the distinguished return
with respect to the whole observation vector, while location and scale are
constants.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BatchedExamples.BATCHED_AD_SOURCE,
)
```

The resulting gradient is checked against the analytic score
`-(xᵢ - location)/scale²`. The generated-kernel pane comes from the exact
prepared plan executed during this docs build; it is not a parallel
illustrative copy.

## Distribution gradient latency and allocation

The following receipt reuses exactly the three benchmark inventories on the
[Distribution kernels](distributions.md) page: the Normal plate sizes, all seven
scalar-gallery families and sizes, and the covariance-Cholesky MVN sizes. AR(1)
remains outside this benchmark for the same reason it is absent there. This is
deliberately a distribution-only allocation claim; the broader bijector and PPL
coverage below remains correctness evidence.

Continuous scalar families, Normal, and MVN differentiate the observation port
`x`. Bernoulli and Geometric differentiate their scalar logit ports because
integer observations cannot be active. Each row is checked against an analytic
gradient before timing.

```@eval
Main.ReactiveKernelsDocs.render_distribution_gradient_benchmarks()
```

Every cell is the median of five minimum-time measurements after preparation.
The two timing series perform different documented work: `ad_gradient` returns
the gradient only and owns any vector it returns;
`ad_value_and_gradient!` computes both the value and gradient while filling a
caller-owned vector. Scalar logit gradients are returned as isbits `Float64`
values, so no mutable destination is needed. The checked-in
[gradient benchmark receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/distribution-gradient-v1.toml)
retains all raw times, allocation bytes/counts, analytic errors, source-receipt
links, and exact package pins.

## Eight Schools model gradient matrix

This is the AD-only companion to the
[Eight Schools primal matrix](eight-schools.md). It prepares scalar selections
of that page's exact published
[`eight_schools.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/eight_schools.jl)
graph; no prior, likelihood, transform, or distribution formula is copied into
an AD-specific RK evaluator. The optimized Turing model and manual Julia control
are loaded from the primal benchmark authority as well.

For a native prepared plate, `prepare_ad` differentiates the exact generated
callable and operation table used by primal execution; it does not re-lower an
AD-specific kernel. The one native lowering emits an unconditional scalar loop
for common vector plates and retains dependency scheduling only where a
higher-dimensional broadcast can reuse a partial-axis invariant.
`code_expr(prepared_ad_kernel) === code_expr(prepared_ad_kernel.kernel)`, so the
publicly inspectable AST is the exact body differentiated by Enzyme. It
materializes no pointwise buffer for a summed WANT.

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_ad_benchmarks()
```

The four scalar gradient cells are the packed unconstrained joint, prior, and
likelihood plus the minimal θ-only likelihood. Each uses a prepared,
caller-owned value-and-gradient path and is checked against central finite
differences before timing. The constrained parameter object is not a supported
active storage type for the public RK AD boundary. Pointwise output remains
blank because neither compared public surface offers a useful matched
Jacobian/VJP contract; the benchmark does not invent a fused pointwise-plus-total
surrogate.

### Baseline implementations

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_ad_baselines()
```

The checked-in
[Eight Schools AD receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-ad-v1.toml)
retains ten raw rounds, source and primal-receipt pins, parity errors,
preparation/first-execution costs, and steady-state timing and allocation data.
Reproduce and validate it with:

```sh
julia --startup-file=no benchmark/eight_schools_ad_comparison.jl \
  --output=benchmark/receipts/eight-schools-ad-v1.toml
julia --startup-file=no benchmark/receipts/validate_eight_schools_ad.jl \
  benchmark/receipts/eight-schools-ad-v1.toml
```

## MNIST full-data model gradients

The same AD-only boundary now covers the
[MNIST multinomial-logistic model](mnist-logistic.md). The benchmark imports the
exact published `MNIST_LOGISTIC_SOURCE` graph and loads the exact manual Julia
and Turing comparator definitions from its primal benchmark; no model,
distribution, or comparator formula is copied into an AD-specific evaluator.
All graph-independent data (`X`, `y`, and the class count) are rebound as
`Constant`s, while the sampler-relevant packed `[vec(W); b]` vector is the one
active port.

The complete primal 2×4 inventory remains visible. Joint, prior, and likelihood
are scalar packed-vector gradients. The structured `(W, b)` boundary remains
unsupported because it consists of two active HAVE ports and the public RK AD
contract deliberately selects exactly one. Pointwise remains unsupported
because the compared public surfaces do not share a matched Jacobian/VJP
contract. Neither blank region is replaced by a benchmark-only API.

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_ad_benchmarks()
```

The authoritative receipt uses all 60,000 training images, 784 features, ten
classes, and 7,065 active coefficients. An independent analytic
reference-class softmax score checks all nine RK/manual/Turing gradients before
timing. The dense joint and likelihood paths retain their real
linear-predictor and reverse-pass storage; only the pruned RK and manual prior
paths are zero-allocation. This keeps the allocation plot a measurement rather
than extending the zero-allocation claim beyond what the generated callable
currently does.

The checked-in
[MNIST AD receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-ad-v1.toml)
retains ten raw rounds, source and primal-receipt pins, analytic parity errors,
preparation/first-execution costs, and steady-state timing and allocations.
Reproduce and validate it with:

```sh
julia --startup-file=no benchmark/mnist_logistic_ad_comparison.jl \
  --output=benchmark/receipts/mnist-logistic-ad-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_logistic_ad.jl \
  benchmark/receipts/mnist-logistic-ad-v1.toml
```

## Owned storage, not borrowed caches

Do not differentiate a `NonAllocatingKernel` whose recipe caches are borrowed
and overwritten on every call. A reverse pass needs the forward intermediates
to remain valid until the backward pass consumes them, so cache reuse at that
boundary can silently corrupt derivatives.

When a derivative needs reusable batch storage, keep the primal operation
explicit and give DifferentiationInterface an owned `Cache` for that buffer.
The focused executable authority is
[`packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl).
It checks the caller-owned buffer, the backend cache, the analytic score, and
zero steady-state allocations. Reproduce that boundary from the repository
root with:

```sh
julia --startup-file=no test/run_nonallocating_integration.jl
```

## Coverage across the example corpus

The same prepared boundary is exercised without making a concrete backend a
core dependency:

- scalar, plated, multivariate, and time-series log-density kernels are checked
  in
  [`packages/ReactiveKernelsKernelExamples/test/test_distributions_example.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsKernelExamples/test/test_distributions_example.jl);
- constrained transforms and fused Jacobians are checked in
  [`packages/ReactiveKernelsKernelExamples/test/test_bijectors_enzyme.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsKernelExamples/test/test_bijectors_enzyme.jl); and
- every declarative PPL walkthrough is checked in
  [`packages/ReactiveKernelsPPLExamples/test/test_ppl_enzyme.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/test/test_ppl_enzyme.jl).

For matched primal, gradient, and generated-quantity measurements against
Turing.jl, continue to [Evaluation latency and batched throughput](eval-throughput.md).

## Reactant/XLA-compiled gradients

When the optional Reactant weak dependency is loaded, the same prepared
differentiation boundary can be compiled through Reactant/XLA — the automatic
differentiation analog of compiling the primal kernel with
`@compile sync = true kernel(traced_args...)`. `compile_ad_gradient` and
`compile_ad_value_and_gradient` take a [`PreparedADKernel`](@ref) — so all of the
scalar-`want`, single-active-port validation and the authored-`have`-order reorder
above are reused unchanged — and return a compiled callable over the traced HAVE
boundary. The differentiation engine stays the caller's `DifferentiationInterface`
backend; `AutoEnzyme(mode = Enzyme.Reverse)` traces through Reactant and matches
the native reverse pass bit-for-bit on the boundaries that compile.

This example is not executed during the docs build (Reactant is not a docs
dependency); it is the exact shape exercised by
[`test/test_ad_reactant.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/test/test_ad_reactant.jl):

```julia
using ReactiveKernels
using Reactant
import Enzyme
using DifferentiationInterface: AutoEnzyme

backend = AutoEnzyme(; mode = Enzyme.Reverse)

# Prepare exactly as on the native path — this owns all the validation.
prepared = prepare_ad(objective, backend, parameters; data, active = :q, want = :density)

# Trace the selected HAVE boundary (authored order), then compile.
traced = map(Reactant.to_rarray, (parameters, data))
compiled_gradient = compile_ad_gradient(prepared, traced...)
gradient = compiled_gradient(traced...)                     # XLA-compiled gradient

compiled_both = compile_ad_value_and_gradient(prepared, traced...)
value, gradient = compiled_both(traced...)                  # potential + gradient in one call
```

The boundary is the same one Reactant already imposes on the primal kernel:

- a Reactant-compiled gradient is only possible where the **primal kernel itself
  compiles through Reactant**. Where it does not (for example the Eight Schools
  `packed_unconstrained/{joint,prior}` boundaries, which fail with
  `"Scalar indexing is disallowed."`), the underlying `@compile` error propagates
  unchanged — the capability neither degrades silently nor tries to fix the primal
  lowering; and
- a host (non-traced) active argument, or an argument count that does not match
  the selected HAVE boundary, raises a clear `ArgumentError` rather than a bare
  `MethodError`.

`compile_ad_value_and_gradient` returns the `(value, gradient)` pair — one
compiled call yields both the potential and its gradient, which is the shape a
sampler wants — while `compile_ad_gradient` returns the gradient alone, mirroring
[`ad_gradient`](@ref) and [`ad_value_and_gradient!`](@ref) on the native path.
