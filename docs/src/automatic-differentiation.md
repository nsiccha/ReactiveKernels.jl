# Automatic differentiation

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

## Owned storage, not borrowed caches

Do not differentiate a `NonAllocatingKernel` whose recipe caches are borrowed
and overwritten on every call. A reverse pass needs the forward intermediates
to remain valid until the backward pass consumes them, so cache reuse at that
boundary can silently corrupt derivatives.

For a zero-allocation derivative, keep the primal operation explicit and give
DifferentiationInterface an owned `Cache` for its reusable batch buffer. The
focused executable authority is
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
