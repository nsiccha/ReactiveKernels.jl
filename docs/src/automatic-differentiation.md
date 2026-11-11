# Prepared differentiation

Use `prepare_ad` to select a scalar objective and its active input, then reuse the prepared derivative calculation with changing parameter values. ReactiveKernels supplies the graph boundary through DifferentiationInterface; a loaded backend performs differentiation.

## Active input

This example uses Enzyme. The data and scale are ordinary runtime inputs but only the parameter vector is active.

```@example gradients
using ReactiveKernels, DifferentiationInterface
import Enzyme

@kernel objective(q::Vector{Float64}, data::Vector{Float64}, scale::Float64) = begin
    density::Float64 = sum(q .* data) - scale * sum(abs2, q)
    return density
end

q = [0.3, -0.4, 0.2]
data = [2.0, -1.0, 0.5]
scale = 1.25
backend = AutoEnzyme(; mode=Enzyme.Reverse)
ad = prepare_ad(objective, backend, q, data, scale;
    active=:q, want=:density)
gradient = ad_gradient(ad, q, data, scale)
reference = data - 2scale * q
@assert gradient ≈ reference
(; gradient, reference)
```

Inactive arguments remain arguments at evaluation time. Preparing with one data vector does not freeze that vector's values. Bind genuinely fixed data through `bound` when its preprocessing should belong to the prepared object.

## Caller-owned output

The mutating interface fills storage supplied by the caller and returns the objective with that gradient. Copy borrowed results when retaining them across later calls.

```@example gradients
buffer = similar(q)
value, returned = ad_value_and_gradient!(ad, buffer, q, data, scale)
@assert returned === buffer
@assert buffer ≈ reference
(; value, gradient=copy(buffer), caller_owned=returned === buffer)
```

A prepared pullback handles an output cotangent for scalar, array or selected multiple outputs. Pushforwards and Hessian-vector products have separate preparation APIs. Their supported active structures and backend requirements are stated in the [differentiation reference](api.md); choose the operator needed rather than constructing a dense Jacobian unnecessarily.

Derivative rules for an owned mathematical primitive belong to one [derivative graph](manual-derivative-rules.md). The native primal and the derivative use that graph's selected computations.

The [Eight-schools example](eight-schools.md) explains the density and coordinate boundary that a sampler differentiates.
