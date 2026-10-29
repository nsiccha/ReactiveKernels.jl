# Prepared automatic differentiation

ReactiveKernels exposes automatic differentiation through
`DifferentiationInterface`. The package owns the prepared-kernel boundary, not
a concrete differentiation engine: Enzyme is an optional test and example
dependency, and core package source never imports it.

`prepare_ad` resolves one active HAVE port and one scalar WANT once.
`prepare_ad_pullback` applies the same boundary to a scalar or non-scalar WANT,
or to several WANT ports at once, and prepares one reverse output-cotangent
direction; `prepare_ad_pushforward` and `prepare_ad_hvp` prepare
Jacobian-vector and Hessian-vector products on it (see below). Every other selected HAVE
is supplied to the backend as a freshly rebound `Constant`, so preparation
fixes types and shapes without freezing values used by later calls. Both paths
differentiate the exact primal callable, operation table, and inspectable AST;
RK does not generate an AD-specific kernel.

## Prepare once, then request gradients or value-and-gradient

The example below shows the kernel definition and the two native prepared
interfaces. Its assertions are exercised by the package test suite rather than
the docs build, so the published documentation carries no Enzyme/LLVM autodiff
toolchain:

```julia
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

### Several WANT ports in one sweep

A pullback or pushforward may select several WANT ports, `want = (:a, :b, ...)`
or a low-level kernel prepared with several, the way
`prepare(spec; want = (:a, :b, ...))` evaluates them in one sweep. The ports
are differentiated together as the kernel's tuple result, so the graph they
share (index gathers, design products) is computed once per call rather than
once per port:

```julia
@kernel tangent_ports(q::Vector{Float64}, s::Float64;
                      data::Vector{Float64}) = begin
    shared = exp.(s .* q)
    mean = data .* shared
    total::Float64 = sum(shared) + s^2
end

vjp = prepare_ad_pullback(tangent_ports, backend, (mean_cotangent, 1.0),
                          parameters, 0.7; data, active = :q,
                          want = (:mean, :total))
(mean, total), cotangent = ad_value_and_pullback(
    vjp, (mean_cotangent, 1.0), parameters, 0.7; data)

jvp = prepare_ad_pushforward(tangent_ports, AutoEnzyme(; mode = Enzyme.Forward),
                             (v,), parameters, 0.7; data, active = :q,
                             want = (:mean, :total))
(mean, total), ((mean_tangent, total_tangent),) =
    ad_value_and_pushforward(jvp, (v,), parameters, 0.7; data)
```

The pullback's seed holds one cotangent per port, in WANT order, each shaped
like its port's value, and the result is the sum of the ports'
vector-Jacobian products. RK computes it as the gradient of the seeded inner
product `Σᵢ ⟨seedᵢ, yᵢ⟩`, one primal and one reverse sweep through the
backend's gradient: DifferentiationInterface's Enzyme pullback seeds only a
scalar or array output. `ad_value_and_pullback` returns the tuple of port
values, from the same sweep when the backend keeps it
(`ad_retains_primal_sweep`, reverse-mode `AutoEnzyme`) and from one separate
primal evaluation otherwise. Each pushforward direction yields a tuple with
one tangent per port; forward-mode `AutoEnzyme` differentiates the
tuple-valued kernel directly. A single WANT keeps its untupled value,
cotangent and tangents.

### Ports that no active port reaches

A selected WANT port may not depend on any active HAVE port. Examples are a
gather of bound data that `bound` partial evaluation folds into a constant
(`loc[rows]` or `view(loc, rows)`), a value computed from inactive HAVE values
alone, or an inactive HAVE port itself. Its derivative is zero. RK finds such
ports in the graph and leaves them out of the call the backend
differentiates. The pullback adds nothing for them, each pushforward
direction carries a zero tangent for them, and their values come from the
same sweep. When every selected port is such a port, the pullback is zero and
the tangents are zero without a backend call. The same holds for a port named
in `retain` (below). Returning such a port from the differentiated call would
hand the backend constant memory as a differentiable output. Plain Enzyme
refuses that, and under runtime activity the pullback would add its seed into
the constant, the prepared kernel's own bound data. `test_ad_constant_ports.jl`
covers bound, folded, view and inactive-input ports, alone and beside ports
that read the point.

## Retain other WANT values from the gradient sweep

A caller that needs both a gradient and other values of the same point, such
as per-observation densities beside the gradient of their sum, can name those
WANT ports in `retain`. The prepared kernel then computes the objective and
every retained WANT in one primal sweep. Only the objective is differentiated,
and the retained values of that sweep come back with the gradient, with no
second primal call:

```julia
@kernel scored(q::Vector{Float64}; data::Vector{Float64}) = begin
    terms = plate(q, data) do qi, di
        term::Float64 = -(di - qi)^2 / 2
        return term
    end
    density::Float64 = sum(terms)
end

retaining = prepare_ad(
    scored, backend, parameters;
    data, active = :q, want = :density, retain = (:terms,),
)
value, gradient, retained = ad_value_gradient_and_retained!(
    retaining, similar(parameters), parameters; data,
)
retained.terms   # the plate values the density summed
```

`retained` is a `NamedTuple` keyed by the retained ports. Each call returns
freshly computed values. A low-level `PreparedKernel` prepared with several
WANT ports takes the same `retain`; its one other WANT is the objective.
`ad_value_gradient_and_retained` is the out-of-place form.

The retained values are those of the gradient's own primal sweep, so
retention needs a backend that evaluates the objective once at the point and
keeps that sweep's writes. `ad_retains_primal_sweep(backend)` states it: core
RK knows no engine and answers `false`, and the Enzyme extension declares
reverse-mode `AutoEnzyme`. Preparation refuses other backends, which may
evaluate perturbed points (finite differences), carry dual numbers, or
restore mutated memory in the reverse pass. Retention is native: a
`NonAllocatingKernel` and compiled (Reactant) staging do not retain values
yet. `test/test_ad_retained_outputs.jl` checks the values, gradients and
refusals.

## Jacobian-vector and Hessian-vector products

`prepare_ad_pushforward` and `prepare_ad_hvp` prepare the forward and
second-order operators on the same boundary: one active HAVE, every other HAVE
rebound as a `Constant` per call, `bound` partial evaluation, and the exact
primal body. Both take input tangents as a tuple of directions, one entry per
direction, so `(v,)` is one direction and `(v1, v2, v3)` evaluates three
together. Results come back as a tuple in the same order.

```julia
second_order = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
                           AutoEnzyme(; mode = Enzyme.Forward))

@kernel tangent_density(q::Vector{Float64}, s::Float64;
                        data::Vector{Float64}) = begin
    density::Float64 = sum(data .* exp.(s .* q)) - 0.5 * sum(abs2, q) - s^2
end

v = [1.0, 0.5, -2.0]
hvp = prepare_ad_hvp(tangent_density, second_order, (v,), parameters, 0.7;
                     data, active = :q, want = :density)
(hv,) = ad_hvp(hvp, (v,), parameters, 0.7; data)
gradient, (hv,) = ad_gradient_and_hvp(hvp, (v,), parameters, 0.7; data)

@kernel tangent_mean(q::Vector{Float64}, s::Float64;
                     data::Vector{Float64}) = begin
    mean = data .* exp.(s .* q)
end

jvp = prepare_ad_pushforward(tangent_mean, AutoEnzyme(; mode = Enzyme.Forward),
                             (v,), parameters, 0.7;
                             data, active = :q, want = :mean)
value, (jv,) = ad_value_and_pushforward(jvp, (v,), parameters, 0.7; data)
```

A Hessian-vector product needs a scalar WANT and a second-order backend.
Use reverse over forward, as above. Its preparation holds memory linear in
the number of active coordinates `n`: RK prepares the reverse gradient of the
directional derivative `x -> Df(x)[v]` itself, without the forward-mode
gradient that DifferentiationInterface's own reverse-over-forward preparation
adds for `gradient_and_hvp`, whose Enzyme preparation holds `n` one-hot
vectors of length `n` (8n² bytes, 187 GiB at 158,343 coordinates).
`ad_gradient_and_hvp` takes its gradient from the reverse backend in one
sweep. Native Enzyme forward over reverse fails
static activity analysis when the objective multiplies a constant matrix by an
active vector, as a design-matrix linear predictor does
(`benchmark/repro_enzyme_forward_over_reverse_const_array.jl`); it works on
objectives without such a product. A pushforward accepts any WANT,
including arrays and `NamedTuple`s of arrays. `ad_hvp!` and
`ad_gradient_and_hvp!` write into caller-owned destinations, one array per
direction.

Several directions with disjoint supports compress a structured Jacobian or
Hessian. When active coordinates fall into groups that never interact, such as
the per-subject effects of a population model, one direction per coordinate
within a group, summed across all groups, recovers every group's block. A
population with `K` effects per subject needs `K` directions, whatever the
number of subjects. The focused authority is
[`test_ad_tangent_operators.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/test/test_ad_tangent_operators.jl).

A tuple `active` selector works for pushforwards (each direction is a tuple of
component tangents) and for reverse-over-forward Hessian-vector products (each
direction and each result is a tuple of components). DifferentiationInterface
0.7.21 cannot yet take a forward-over-reverse Hessian-vector product at such
a structured point. A `NonAllocatingKernel` supports pushforwards; its
Hessian-vector products fail Enzyme's static activity analysis, so take them
from the dataflow kernel. Reactant-compiled pushforwards and Hessian-vector
products are not provided; these operators run natively.

### Third-order directional derivatives

`prepare_ad_third` prepares, per direction `v`, the gradient of the Hessian
quadratic form, `∇(vᵀ H v) = D³f[v, v, ·]`, a vector shaped like the active
point. The gradient of a Laplace-approximate marginal needs these
contractions: the derivative of `log det(-H)` is a sum of them over the
columns of an inverse Cholesky factor. It is the Hessian-vector product of
the directional derivative `x -> Df(x)[v]`, so its backend is
`SecondOrder(hvp_backend, pushforward_backend)`, and reverse over forward over
forward extends the reverse-over-forward composition above by one forward
layer:

```julia
third_order = SecondOrder(second_order, AutoEnzyme(; mode = Enzyme.Forward))
third = prepare_ad_third(tangent_density, third_order, (v,), parameters, 0.7;
                         data, active = :q, want = :density)
(tv,) = ad_third(third, (v,), parameters, 0.7; data)
gradient, (hv,), (tv,) = ad_gradient_hvp_and_third(third, (v,), parameters, 0.7;
                                                    data)
```

Each direction costs one reverse sweep over a doubly forward-differentiated
primal. The outermost pass is plain reverse mode, so objectives that multiply
a constant matrix by the active vector differentiate, and so do the RK
derivative rules the primal calls (such as the Normal identity-link GLM
density). `ad_gradient_hvp_and_third` also returns the gradient, from one
reverse sweep, and the Hessian-vector products, from the Hessian-vector
backend, at the same point. `ad_third!` and `ad_gradient_hvp_and_third!`
write into caller-owned destinations, one array per direction.

Colored directions compress third-order contractions as they compress
Hessians: when active coordinates fall into groups that never interact, a
direction summed across groups, `v = Σᵢ vᵢ`, returns `Σᵢ D³f[vᵢ, vᵢ, ·]`,
because every cross term `D³f[vᵢ, vⱼ, ·]` vanishes. A population with `K`
effects per subject therefore needs `K` directions for every subject's
contraction, whatever the number of subjects. A bilinear contraction follows
by polarization, `D³f[w, v, ·] = (D³f[w + v, w + v, ·] - D³f[w - v, w - v, ·]) / 4`.
The tuple-selector, non-allocating and Reactant boundaries of Hessian-vector
products apply unchanged: on Julia 1.10 a `NonAllocatingKernel`'s third-order
derivatives fail Enzyme's static activity analysis, as its Hessian-vector
products do.

Registered Enzyme 0.13.216's forward mode over forward mode returns NaN on
some calls through a dense matrix product, and identical calls disagree
(`benchmark/repro_enzyme_forward_over_forward_view_matvec.jl`, Enzyme only).
Third-order derivatives of an objective that multiplies a constant matrix by
the active vector, as an RKPPL design matrix does, can therefore be NaN:
through a matrix-vector product on Julia 1.12 and later, through a
matrix-matrix product on every Julia version. The local Enzyme `dogfood` line
fixes it from libEnzyme build `4be29b19`. The section on current backend
limitations in `constraints.md` tracks it.

## Freeze data-only work during preparation

When data stay fixed across many derivative calls, pass them in the named
`bound` NamedTuple instead of rebinding them as `Constant` contexts. The
data-only prefix executes exactly once during preparation; its results become
constants in the residual kernel, and the prepared derivative boundary accepts
only the remaining HAVE ports. Rebind by preparing again from the original
kernel specification.

Bound data may also contain tuples or named tuples of arrays. Native AD
preserves the record shape and copies nested array views into ordinary arrays
at preparation. Rebinding such records uses the same preparation step.

```julia
bound_prepared = prepare_ad(
    objective, backend, parameters, 1.25, 0.0;
    active = :q, want = :density,
    bound = (; data),
)

bound_gradient = ad_gradient(bound_prepared, parameters, 1.25, 0.0)
@assert bound_gradient ≈ gradient

(; bound_gradient)
```

`active` and the positional preparation exemplars refer to the remaining
ports. Authored signature defaults and keyword arguments do not apply to a
bound preparation; the example therefore supplies the remaining `scale` and
`offset` values positionally.

The same preparation step can cache named data-only recipes inside an authored
plate whose remaining work depends on the parameter:

```julia
@kernel plated_objective(q::Vector{Float64}, data::Vector{Float64}) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(data, parameter) do d, theta
        transformed::Float64 = log(d)
        result::Float64 = transformed * theta
        result
    end
    total::Float64 = sum(pointwise)
end

plate_data = [1.0, 2.0, 4.0]
prepared_plate = prepare_ad(
    plated_objective, backend, parameters;
    active = :q, want = :total, bound = (; data = plate_data),
)
plate_gradient = ad_gradient(prepared_plate, parameters)
```

Here `log(d)` runs during preparation and its cached values feed the live plate.
Each data-only recipe uses its own broadcast coordinates, including singleton
axes. Rebinding rebuilds the cache. Native execution, prepared AD, and Reactant
consume the same residual graph. The focused native and Reactant tests exercise
this example without adding an AD toolchain to the documentation build.

Caching adds preparation work and storage; cheap arithmetic may gain nothing or
run slower. Original data arguments remain retained for shape validation even
when their elements are no longer read by the residual. Empty bound domains
keep their original execution. A live array with a declared rank, whatever
its declared element type (`AbstractVector` qualifies like `Vector{Float64}`,
while `AbstractArray{Float64}` leaves the rank open), and bound axes longer
than one in every dimension it can supply cannot change which cells run;
scalar and explicit atomic inputs add no dimensions. Any other live input
could leave the plate fewer cells than its bound inputs span, so the cached
values are computed over the bound domain and the cell keeps its original
execution if that computation throws. Non-concrete intermediate results also
keep their original execution. A nested plate, scan or prepared kernel in the cell is never run
during preparation: it stays in the cell, lowered as before, and the cell's
other bound-only values are still cached, such as a per-subject index list
beside a recurrence over that subject's operations. Cached cell values are
`Bool`, standard 8–64-bit integers, and `Float16`/`Float32`/`Float64`, or
dense `Array`s of those element types: a per-cell index list such as
`findall(isone, kinds)` is cached as an array of arrays, read-only and shared
by later calls like any bound value. A tuple, struct, view or range value
(a named-tuple or static-vector scan seed, for example) is not cached: the
recipe producing it stays in the cell, and the bound-only values it reads
are cached instead. A plate whose cell result reads only bound data, and whose
other inputs are all atomic, is evaluated whole at preparation and replaced by
its result; recipes that then read only bind-time values, such as a
`reduce(vcat, …)` over its cells, are evaluated with it. Such an array result
beside a live non-atomic input keeps its
original execution, so a plate's output never stores the shared cached arrays.
Under Reactant a tensorized plate still needs rectangular
per-lane arrays; ragged lanes are refused there with or without the cache.
Recipes still follow the pure-operation contract.
An inline expression such as `theta * log(d)` is split along its authored
structure: `log(d)` is cached and the cell computes `theta *` that value (see
[Data-only subexpressions](compiler.md#data-only-subexpressions)).

## Accepted boundary

- Gradient and Hessian-vector-product preparation require a scalar WANT.
  Pullback preparation accepts one scalar or non-scalar WANT, or several, and
  an output-cotangent exemplar (one per port for several); pushforward
  preparation accepts the same WANT selection and a tuple of input-tangent
  exemplars. Further WANT ports may be retained, not differentiated, with a
  gradient's `retain`.
- Exactly one HAVE port is active.
- Integer active ports and aliased active boundaries reject.
- An inactive HAVE downstream of the active port rejects rather than cutting a
  real derivative path.
- Stored backend preparation is reusable but not thread-safe; concurrent callers
  use separate prepared objects.
- Plain `AutoEnzyme(mode = Enzyme.Reverse)` is the supported example
  configuration. Runtime-activity mode and function annotations are outside the
  ReactiveKernels boundary.
- Preparation warns about each value whose type does not infer concretely
  from the exemplar arguments, and continues. Make such a value inferable
  before differentiating natively; see
  [Concretely typed programs](compiler.md#concretely-typed-programs).

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
