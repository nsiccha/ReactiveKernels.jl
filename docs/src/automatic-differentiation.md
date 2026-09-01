# Prepared automatic differentiation

ReactiveKernels exposes automatic differentiation through
`DifferentiationInterface`. The package owns the prepared-kernel boundary, not
a concrete differentiation engine: Enzyme is an optional test and example
dependency, and core package source never imports it.

`prepare_ad` resolves one active HAVE port and one scalar WANT once.
`prepare_ad_pullback` applies the same boundary to a scalar or non-scalar WANT
and prepares one reverse output-cotangent direction. Every other selected HAVE
is supplied to the backend as a freshly rebound `Constant`, so preparation
fixes types and shapes without freezing values used by later calls. Both paths
differentiate the exact primal callable, operation table, and inspectable AST;
RK does not generate an AD-specific kernel.

## Prepare once, then request gradients or value-and-gradient

This build-executed example shows the kernel definition and the two native
prepared interfaces:

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
`ad_value_and_gradient` preserves structured results such as a `NamedTuple`,
while `ad_value_and_gradient!` returns `(value, gradient)` and fills
caller-owned array storage. All preserve authored positional defaults and
keyword interfaces while rebuilding inactive constants from the current call.

## Structured gradients and reverse pullbacks

One active HAVE may be a recursively differentiable `NamedTuple` of floating
scalars, arrays, tuples, and nested NamedTuples. `ad_gradient` and
`ad_value_and_gradient` preserve that structure in the returned sensitivity.
DI's Enzyme backend does not currently provide a caller-owned mutation contract
for such structured destinations, so RK does not pretend that
`ad_value_and_gradient!` supports them.

For a vector-valued WANT, prepare a reverse pullback with an exemplar output
cotangent and reuse it with current arguments and cotangents:

```julia
prepared_vjp = prepare_ad_pullback(
    pointwise_kernel, backend, output_cotangent, parameters, data;
    active = :parameters,
)
value, vjp = ad_value_and_pullback(
    prepared_vjp, output_cotangent, parameters, data,
)
```

The returned sensitivity is `J' * output_cotangent`, computed in one reverse
pass. It is not a full Jacobian. `ad_value_and_pullback!` accepts caller-owned
array cotangent storage when the backend supports it.

## Freeze data-only work during preparation

When data stay fixed across many derivative calls, pass them in the named
`bound` NamedTuple instead of rebinding them as `Constant` contexts. The
data-only prefix executes exactly once during preparation; its results become
constants in the residual kernel, and the prepared derivative boundary accepts
only the remaining HAVE ports. Rebind by preparing again from the original
kernel specification.

```@example automatic_differentiation
bound_prepared = prepare_ad(
    objective, backend, parameters;
    active = :q, want = :density,
    bound = (; data),
)

bound_gradient = ad_gradient(bound_prepared, parameters)
@assert bound_gradient ≈ gradient

(; bound_gradient)
```

`active` and the positional preparation exemplars refer to the remaining
ports. Authored keyword arguments do not apply to a bound preparation, because
their values have either been fixed or removed with the hoisted prefix.

## Accepted boundary

- Gradient preparation requires a scalar WANT. Pullback preparation accepts one
  scalar or non-scalar WANT and an output-cotangent exemplar.
- Exactly one HAVE port is active.
- Integer active ports and aliased active boundaries reject.
- An inactive HAVE downstream of the active port rejects rather than cutting a
  real derivative path.
- Stored backend preparation is reusable but not thread-safe; concurrent callers
  use separate prepared objects.
- Plain `AutoEnzyme(mode = Enzyme.Reverse)` is the supported example
  configuration. Runtime-activity mode and function annotations are outside the
  ReactiveKernels boundary.

## Ownership and reusable storage

Do not differentiate a `NonAllocatingKernel` whose borrowed recipe caches are
overwritten on every call. A reverse pass needs forward intermediates to remain
valid until the backward pass consumes them. For reusable batch storage, keep
the primal operation explicit and give DifferentiationInterface an owned
`Cache`.

The focused executable authority is
[`test_batched_nonallocating.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl).

## Results and compiler integrations

- [Distribution AD: scalar and batched](distributions-ad.md) contains the plated
  objective and distribution gradient receipts.
- [PPL automatic differentiation](ppl-ad.md) puts reviewed Eight Schools and
  MNIST model gradients first.
- [Automatic differentiation through Reactant](reactant-ad.md) documents the
  compiled prepared-AD boundary without making Reactant a docs dependency.

The same native prepared boundary is also checked across bijectors and the
remaining PPL walkthroughs. These checks establish correctness coverage; they do
not create additional benchmark claims.
