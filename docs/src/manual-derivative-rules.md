# Derivative graphs

Use a derivative graph to define an owned primitive's value and derivatives together. ReactiveKernels selects the relevant graph outputs and generates backend adapters from that mathematical authority.

## Scalar partials

This graph offers a value-only recipe and a joint value-and-partial recipe. A value query can use `sin`; a query that needs both results can use `sincos`.

```@example rules
using ReactiveKernels

@kernel sine_graph(x::Float64) = begin
    y::Float64 = sin(x)
    (y, dy::Float64) = sincos(x)
    return y, dy
end

sine = scalar_derivative_rule(sine_graph; primal=:y, partials=(x=:dy,))
x = 0.4
(; value=sine(x), reference=sin(x))
```

The rule owns a callable. It does not attach a method to somebody else's primitive or replace a foreign backend rule. An active-input selection determines which partials the adapter needs.

## Vector directions

A general derivative graph names forward directions, the output tangent, the output covector and input cotangents. Fixed data is marked with `nothing` in both derivative mappings.

```@example rules
struct Geometry
    scale::Float64
end

@kernel scaled_graph(x::Float64, geometry::Geometry,
                     x_dot::Float64, y_bar::Float64) = begin
    y::Float64 = x * geometry.scale
    y_dot::Float64 = x_dot * geometry.scale
    x_bar::Float64 = y_bar * geometry.scale
    return y, y_dot, x_bar
end

scaled = derivative_rule(scaled_graph; primal=:y,
    directions=(x=:x_dot, geometry=nothing), tangent=:y_dot,
    covector=:y_bar, cotangents=(x=:x_bar, geometry=nothing))
geometry = Geometry(3.0)
forward = forward_cut(scaled, 2.0, geometry, 1.0)
reverse = reverse_cut(scaled, Val(1), 2.0, geometry, 1.0)
@assert forward == (6.0, 3.0)
@assert reverse == (3.0,)
(; forward, reverse)
```

The reverse mask uses original input positions. A bit for a data-only input is invalid. Data remains available to the primal and any derivative that reads it, but has no authored derivative port.

Under Enzyme reverse mode, the adapter keeps what the reverse branch reads between the two passes. It reads an input in place when Enzyme proves that nothing writes it before the reverse pass and the result cannot expose it: the result is a plain-bits value such as a number, or an array of numbers whose memory does not overlap the input's. Otherwise it copies the input once per call. Enzyme proves this for an array passed directly, unless the caller writes, after the call, to memory Enzyme cannot show is apart from it, such as an output array it also received. For arrays held inside a value such as a struct, tuple or view, Enzyme dogfood with the `overwritten_reachable_data` repair is required (`Enzyme.has_dogfood_repair(:overwritten_reachable_data)`). Without that repair, such arrays are copied on every reverse call, so the cost grows with their length.

A prepared program writes after the rule's call: a plate stores each result in its cells, and a prepared pullback keeps the port's value. Two further dogfood repairs let Enzyme show that those writes miss the input. With `unexposed_storage_barriers`, a plate reads the input in place on Julia 1.10. With `later_store_targets`, a prepared pullback reads it in place on Julia 1.10, 1.12 and 1.13, also when the input holds a `Dict`. A plate on Julia 1.12 and 1.13 still copies the input on each call. `benchmark/repro_prepared_rule_overwritten.jl` prints the bytes per call for each case.

`jax_source` generates `custom_jvp` adapters for scalar rules from these same
activity-selected graph cuts. Joint value/partials recipes are shared across
calls and activity patterns. General rules generate their forward and staged
reverse adapters from the authored branches too. Host data preparation uses
the primal cut. See
[JAX runtime and derivatives](jax-runtime.md).

For adapter and stage APIs, see the [core reference](api.md). The [bound-regression example](bound-regression.md) illustrates the same graph-selection principle for primal outputs.
