# Writing models with `@rkppl`

`ReactiveKernelsPPL` is a small probabilistic-programming layer on top of
ReactiveKernels. It lives in the nested package
`packages/ReactiveKernelsPPL`, and ReactiveKernels core does not depend on it.
A model is an ordinary Julia block of `~` statements and assignments. The layer
turns it into a `@kernel` graph with a packed unconstrained parameter vector,
the transforms and their log-Jacobians, data preprocessing done inside the
graph, and the canonical `likelihood` / `prior` / `log_jacobian` / `posterior`
nodes.

The same language is what BayesianRegressionModels' `RKBRMI` backend emits for
an `@brm` formula. Hand-written and BRM-emitted models go through one pipeline:

```
@rkppl block ─ lower_rkppl ─▶ StructuralPlan ─ bind_data ─▶ bound plan
   ─ build_kernel ─▶ (; spec, layout) ─ prepare_query / prepare_sampler
```

Every example panel on this page runs at docs-build time, exactly as displayed. The
canonical programs come verbatim from the package's corpus
(`packages/ReactiveKernelsPPL/test/corpus/`), which also pins how each one
lowers.

## A first model

Bind inputs by keyword, then observe responses with `model(; x) | (; y)`.
`condition(model(; x); y)` is the same operation. A call keyword naming a
declared variable pins it: `model(; x, sigma = 0.3) | (; y)` removes the
`sigma` declaration and its density. Observing it instead,
`model(; x) | (; y, sigma = 0.3)`, keeps its density as likelihood.
`build_kernel` generates the program, and `prepare_query(built, plan, :sampler)`
returns the log-posterior over the packed unconstrained coordinates.

An explicitly observed empty vector has zero observations and contributes
zero log-likelihood. Scalar priors and their transforms still contribute as
authored. An elementwise declaration such as `z[1:0] .~ Normal.(0, 1)` is an
empty vector with no coordinates and zero prior and log-Jacobian. The same
identity applies to empty matrix axes and empty data-sized elementwise arrays;
negative declared sizes and out-of-bounds gathers remain errors.
Native gradients also support a completely empty coordinate pack. Compiled
values support it; compiled gradients currently fail at the backend's empty
tensor export boundary (see [Core constraints](constraints.md)).

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("01_gaussian.jl", :rkppl_first_model; preamble = "using ReactiveKernelsPPL")
```

This is the `@kernel` program `build_kernel` generates for the model above —
the graph that the sampler cut is prepared from:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_kernel_program("01_gaussian.jl")
```

Elementwise observations follow Julia broadcasting. A singleton input such as
`x = [0.5]` can serve a longer response, and matrix or tensor responses keep
their axes. For `mu = a .+ b .* x; y .~ Normal.(mu, 1)`, an `x` of size
`(2, 1)` broadcasts beside a `y` of size `(2, 3)`. The likelihood sums every
broadcast cell. Incompatible non-singleton dimensions fail during binding.
Shared singleton inputs can also serve independent responses of different
sizes. Explicit `@plate` indexing keeps its authored indexing requirements;
`x[i]` does not stretch a singleton `x`.

## Every parameter is declared

Declarations are strict: every parameter needs an explicit prior statement. A
name that is not a data column, a definition, or a declared parameter is an
error, so a typo can never become a parameter:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_strict_error()
```

Coefficient vectors are declared the same way, sized by their design matrix:
`b[axes(X, 2)] .~ Normal.(0, 1)`.

## Parameter priors

Scalar parameter priors include `Normal`, `Cauchy`, `Exponential`, `Gamma`,
`LogNormal`, `Beta`, `InverseGamma`, `StudentT`, `TDist`, `Laplace`, `Logistic`,
`Uniform` and `Weibull`. Arguments can read data, sampled parameters and
ordinary definitions; scalar expressions such as `1 + exp(a)` are values too.
Omitted positional arguments use Distributions.jl defaults: `Normal()` is
standard normal, `Gamma(k)` has unit scale, `Beta(k)` has two equal shape
parameters, and `TDist(nu)` is the standard Student t distribution.

Positive priors have the same normalized meaning in every slot:
`HalfNormal(s)`, `HalfCauchy(s)` and their equivalent truncated Normal/Cauchy
forms work for sampled parameters and the built-in varying and smooth scales. A bare
`Normal(0, s)` or `Cauchy(0, s)` keeps its full support and is rejected in a
positive scale slot. Write the explicit half or truncation instead.

Support keywords on distribution constructors, such as
`Normal(0, s; lower=0)`, are rejected. Use normalized
`truncated(Normal(0, s), 0, Inf)` or the density-preserving `restricted` form
below according to the intended model.
`Flat()` is an improper real prior and accepts no support keywords; choose
`Exponential(s)` for a positive prior or `Uniform(lo, hi)` for a bounded
uniform. These are proper densities and change an improper prior's model.

`x ~ truncated(D, lo, hi)` uses the normalized Distributions.jl density for
any of these univariate families. Bounds may be literals, model-level data,
sampled values or scalar expressions; use `-Inf` or `Inf` for an open end.
The transform uses the intersection with the base distribution's support.
A parameter-dependent bound also determines `constrain`, `unconstrain` and
`logjac`, independent of declaration order. Array and plate priors accept
shared scalar bounds. Per-element bounds are not built yet.

```julia
a ~ Normal(0, 1)
x ~ truncated(Weibull(2 + exp(a), 1), exp(a), 3 + exp(a))
p ~ Dirichlet(3, exp(a))
y .~ Normal.(x, 1)
```

`x ~ restricted(D, lo, hi)` declares the same support intersection and
coordinate transform, while preserving `D`'s original log density. It does
not divide by the probability mass inside the bounds. This is a dedicated
`@rkppl` prior form for bounded density kernels, distinct from normalized
`truncated`. The original family arguments and literal, data or sampled
scalar bounds stay live. The existing normalized half prior retains its
`log(2)` constant when used as `D`.

```julia
scale ~ restricted(Normal(0, 1), 0, Inf)
invdf ~ restricted(Exponential(0.125), 0, 0.5)
```

At the same physical point these priors have the original Normal and
Exponential densities. Normalized truncation instead adds `log(2)` and
`-log1p(-exp(-4))`, respectively. Both forms have the same coordinates and
Jacobians; a live bound or family argument can make the normalization
parameter dependent, so their gradients need not agree in general.

`restricted(Flat(), lo, hi)` has zero log density inside its support and
uses only the transform Jacobian. It supplies an improper flat density
kernel, without a uniform normalizer. Scalar conditioning keeps the original
density and returns `-Inf` outside the declared support; pinning removes the
density as usual. Array and plate priors accept shared scalar bounds with
`restricted.(D.(args...), lo, hi)` or a per-cell `restricted(D, lo, hi)`.
Assign a computed array bound to a name first (`hi = 2 + exp(a)`), as
with `truncated` array priors. Per-element bounds and nested
restrictions/truncations are not built yet.

`Dirichlet(alpha)` takes a concentration vector from data, a sampled array,
a vector literal containing live values, or an array-valued definition.
`Dirichlet(K, a)` takes a literal dimension and a live scalar concentration.
The concentration shape determines the simplex size at binding; it does not
replicate the density body. Concentrations must be positive.

Dynamic Normal, Cauchy and Weibull truncation supports native and Reactant
primal and reverse execution. Gamma, Beta and InverseGamma truncation calls
incomplete gamma or beta functions whose traced methods are still unavailable;
those normalizers support native execution. Hierarchical Dirichlet supports
native and compiled primal and reverse execution.

## Distribution arguments are values

Locations, scales, StudentT degrees of freedom and zero-inflation
probabilities may be scalar or per-observation expressions. Naming an
expression keeps the same density. Scalar calls stay scalar, and vector
operations use Julia's dots:

```julia
a ~ Normal(0, 1)
b ~ Normal(0, 1)
s ~ Normal(0, 1)
nu ~ Exponential(1)
mu = exp.(a .+ b .* x)
y .~ StudentT.(2 + nu, mu, exp(s))
```

Data-computed scales such as `exp.(x)` and mixed plate-latent locations
such as `theta .+ b .* x` also use their written values. Literal and named
scalar offsets add directly to a predictor. Gaussian mixture components also
accept these value locations, including scalar aliases; explicit component
links such as `Poisson.(exp.(a))` apply to sampled scalars.

`Beta.(alpha, beta)` accepts two ordinary shape values. Circular
VonMises endpoints may be named, bound as data or sampled; they must define
a finite interval of width `2pi`. A live lower endpoint can use
`hi = lo + 2pi`. The interval and support are checked before wrapping the
location or evaluating the density.

An ordered vector may use Normal, Cauchy, Laplace, Logistic or StudentT
element priors with live scalar arguments, for example
`c ~ Ordered(Normal(m, 2s), 2)`. Its density is the sum of the element log
densities on increasing vectors, with no factorial normalizer.

Ordered extents may be data-only expressions, including
`length(levels(x)) - 1`, `length(levels(y)) - 2` and
`length(unique(y)) - 1`. Binding evaluates the actual nonnegative integer
extent even when a Gaussian response reads the vector. An ordinal response
has support `1:length(c)+1`; an observed category outside it fails at bind.
`levels` includes a declared DataAPI level pool; `unique` counts observed
values. The prior and support come from the declaration: cumulative
cutpoints may have an ordinary Normal vector prior (non-increasing values
give `-Inf`), and stopping-ratio thresholds may have an Ordered or ordinary
Normal/Cauchy/Laplace/Logistic/StudentT prior.

A per-cell `Flat()` prior contributes zero density and retains its layout
coordinates. Its posterior may be proper through the likelihood; density
evaluation requests no draw from the unnormalizable prior.

Scalar observations also keep their declaration's density. For
`m = @rkppl begin theta ~ Beta(1, 1); k ~ Binomial(n, theta) end`,
`m(; n=5) | (; k=2)` packs only `theta`. Direct lowering with
`conditioned=(:k,)` has the same density and Jacobian. Trials must be a
nonnegative integer value, and the observed `k` must be scalar; use `.~`
for an observation vector.

Uniform bounds may be live scalar values, including in factor arrays:
`lo ~ Normal(0, 1); c[levels(g)] .~ Uniform.(lo, lo + 3)`.

## Caller-owned sampling RHS definitions

Sampling RHS definitions are open Julia bindings. Built-in spellings select
the existing specialized lowerings; other bindings use the following public
protocol. An observation needs only a constrained-value log density:

```julia
struct MyNormal{T}
    mu::T
end
ReactiveKernelsPPL.sampling_logdensity(d::MyNormal, x) =
    -log(2pi)/2 - (x-d.mu)^2/2

normal_lpdf(x, mu) = -log(2pi)/2 - (x-mu)^2/2
# Inside an @rkppl model with a declared parameter a:
y .~ MyNormal.(a)
# The convenience wrapper has the same density:
y .~ LogDensity.(normal_lpdf, a)
```

With `Distributions` loaded, any `Distributions.Distribution` object uses its
ordinary `Distributions.logpdf` method automatically, including custom types.
Other types extend `sampling_logdensity`. A module-visible RHS object can be
used directly; a definition can also construct an RHS value. Undotted `~`
evaluates one whole-event density; `.~` applies it over Julia broadcast cells,
retains the broadcast axes in `:pointwise`, and sums those cells. Independent
custom broadcasts need not share another response's row count. Observations
acquire no packed coordinates and need no transform or random generator. As
under a built-in family, an observed number is one observation, and an
observed definition reading only data (`y = v[positions(rows)]`,
`y = sum(f(raw))`) is evaluated once by `bind_data` and validated as the
response, exactly like the same value bound as data.

A parameter declaration additionally needs structural geometry. The method
receives the constructor binding, authored argument expressions and declared
constrained shape; it must not evaluate active statistical arguments. For
example, a positive exponential residual above a live lower bound:

```julia
struct ShiftedExp{T}
    lower::T
end
ReactiveKernelsPPL.sampling_logdensity(d::ShiftedExp, x) =
    x > d.lower ? -(x-d.lower) : -Inf
function ReactiveKernelsPPL.sampling_geometry(::Type{ShiftedExp}, args, shape)
    ParameterGeometry(shape, 1; support=(:lower, args[1]),
        constrain=(u, shape, lower)->lower + exp(u[1]),
        unconstrain=(x, shape, lower)->[log(x-lower)],
        logjac=(u, shape, lower)->u[1])
end
# Inside a model:
a ~ Normal(0, 1)
b ~ ShiftedExp(a + 0.7)
```

The endpoints receive current argument **values**, including sampled parent
values, on both the host layout path and the generated kernel path.
`ParameterGeometry` specifies constrained `shape`, packed dimension,
support metadata, constrain, inverse and log absolute Jacobian. Shape and
packed dimension may differ: a two-component simplex can pack one coordinate.
Declare constrained array dimensions on the LHS, for example `p[1:2] ~ D()`;
dimensions and packed extents may depend on bound data, never sampled values.
Optional `coordinate_names` names every packed coordinate. A scalar `@plate`
cell uses scalar geometry with one packed coordinate. The density and inverse
must enforce the declared support; metadata alone adds no support check.

An external frontend can supply a submodel without defining it through
`@rkppl`: extend `sampling_fragment(binding)` to return a stable
`RKPPLSubmodel(name, argument_names, body_ast, defining_module)`. Its ordinary
sampling statements, definitions and authored loops then follow the same
scope, nesting and conditioning expansion as RKPPL submodels. The adapter
does not execute an opaque model or modify compiler internals. It returns
`nothing` for bindings that are densities rather than model fragments.

Density bodies and geometry endpoints are pure ordinary Julia and use standard
backend AD. Statistical bodies remain caller-owned. Model-sized iteration
belongs in visible submodel statements or retained plates/scans, following
[Core constraints](constraints.md). The initial custom RHS acceptance tests
cover native values, ordinary Enzyme Reverse, host/kernel transforms and
printed-source replay; compiled backend support depends on the caller's Julia
operations and requires its own retained-loop acceptance. Custom scan-state
geometry is not yet implemented. Ordinary RHS keyword arguments are preserved;
geometry endpoints receive those same keyword values. Structural argument
expressions retain Julia's `Expr(:parameters, ...)` keyword representation.

This protocol currently computes log densities. Generated-only draws and RNG
execution remain future work; they will require a draw capability separately
from parameter geometry, without adding generated values to sampler coordinates.

## Julia semantics, written out

- Use Distributions.jl constructors (`Normal`, `Exponential`, `Gamma`, …).
- Response constructor defaults have their ordinary meaning: `Normal.(mu)`
  and `LogNormal.(mu)` use unit scale, `NegativeBinomial.(exp.(eta))` uses
  probability `0.5`, `InverseGaussian.(exp.(eta))` uses unit shape,
  `Weibull.(k)` uses unit scale, and `VonMises.(exp.(eta))` has zero mean
  and concentration `exp.(eta)`. Plate observations accept `Normal.(mu)` too.
  Live VonMises concentration supports native values and derivatives;
  Reactant compilation still encounters its existing traced `besseli` gap.
- `Bool` response values retain their numeric meaning, zero or one.
  Responses with a strictly positive data domain require `true`.
- `MixtureModel.(vcat.(C1, C2))` gives the components equal weights.
  Explicit shared weights use `MixtureModel.(vcat.(C1, C2), Ref(w))`.
- `Beta.(alpha, beta)` preserves argument order. Swapping the arguments in
  `Beta.(logistic.(eta) .* k, (1 .- logistic.(eta)) .* k)` changes its mean
  from `logistic.(eta)` to `1 .- logistic.(eta)`.
- Broadcasting is explicit: `mu = a .+ b .* x`.
- Distribution arguments use their ordinary values. `exp.`, `logistic.`,
  `normcdf.` and `cexpexp.` apply at each use, including means, rates,
  probabilities, scales, shapes, Student degrees of freedom and zero
  inflation. One predictor may feed several slots or responses under
  different links, with its original parameter names and priors.
  Values outside parameter support contribute `-Inf` through a lazy
  density branch. Mixture components may use independent links.
  A computed coefficient such as `b = z * lambda * tau` reads its
  declared parameters, so using it as a scale retains the same priors.
  Native value and reverse checks cover this form. With Reactant 0.2.290,
  some shared scalar density guards fail during tracing with
  `isless(::Int64, ::Reactant.EnsureReturnType{Any})`, before the compiled
  primal runs. The generic RK reproducer is
  `benchmark/repro_reactant_shared_scale_guard.jl`; this also reproduces
  outside the PPL and predates the construct removal.
- Plate cells accept `BernoulliLogit.(eta)` and `PoissonLog.(eta)` directly
  on the logit and log-rate scales. A bare modeled `VonMises.(kappa)` uses
  zero mean. Live concentration supports native density and AD; compiled
  execution remains gated by the missing traced `besseli` method.
- A vector response uses the dotted tilde, `y .~ Normal.(mu, sigma)`. A plain
  `y ~ Normal(...)` on a data vector is rejected.
- A `~` whose left-hand side is a bound data column is an observation; every
  other `~` declares a parameter.
- `Ordinal` permits an intercept alongside its thresholds, for both
  cumulative and stopping-ratio responses. Both declarations and their
  priors are translated as written.
  `OrderedLogistic` and `Ordinal` require an explicit threshold argument
  such as `Ref(c)`; declare the modeled vector and its prior in the body.
  Omitting the argument is rejected.
- Single assignment, no `if`, no `target +=`. Loops are written as
  `@plate` cells or `@scan` recurrences (see [Plates](#Plates)).

Joint `MvNormalCholesky` responses use explicitly declared covariance pieces:

```julia
sd[1:2] .~ Exponential.(1)
C ~ LKJCholesky(2, 2)
F = sd .* C
[y1, y2] ~ MvNormalCholesky([mu1, mu2], F)
```

`LKJCovarianceFactor` no longer creates implicit priors or names. The scale
prior and factor names are the author's; ordinary positive-support scale
priors and a sampled `LKJCholesky` shape are supported.

The implicit `r2d2(...)` statement and `b ~ Horseshoe(...)` coefficient
shortcut are retired. State priors and coefficient arithmetic explicitly,
or obtain the BRM-owned bodies with
`BayesianRegressionModels.rkppl_model(:r2d2_coefs)` and
`BayesianRegressionModels.rkppl_model(:horseshoe_coefs)`. Their statistical
preparation and model construction belong to BRM.

A definition may call a function-shaped ReactiveKernels `@kernel` in the model
module. A positional call such as `loc = recurrence(x, a)` splices the child's
graph into the generated model, including its authored plates and scans.
Imported, aliased and qualified bindings use the same composition. Omitted
child port annotations accept the caller's declared types; declared boundaries
keep their types. The emitted `kernel_expr` remains the source of the graph,
so evaluating that expression retains the same child operations.

A kernel with several outputs (`return PHI, omega2`) composes through Julia's
destructuring: `(P, om) = basis(x)` stays one statement in the generated
program and splices the child at its whole output boundary. Destructuring an
ordinary function likewise calls it once per evaluation. Binding the call to
one name, `bt = basis(x)`, or indexing it in place, `P = basis(x)[1]`, splices
the same child and binds the tuple the call returns, so every output is
computed even when one element is read.

Composition exposes the child's execution capabilities. Ordinary native
Enzyme reverse covers empty and nonempty child scans, including inside a bound
`eachcol` subject plate. The default compiled backend expands small subject
plates into copies of the child scan, so those shapes still lack retained-loop
structural acceptance. The composition tests keep that gap visible; larger
compiled subject plates are covered separately.

A data-only call used only by a parameter-dependent function runs once when
`prepare_query` or `prepare_sampler` prepares the graph. It may return a tuple or
named tuple containing arrays. Named definitions, aliases and inline
calls follow the same rule, including beside declared vector or matrix
parameters. A value also needed during binding, such as a prior argument or an
array dimension, keeps its bind-time evaluation and validation.

## Factor levels

`c[levels(g)] .~ Normal.(0, 2)` declares one coefficient per level of `g`, and
`c[g]` gathers them per observation. A full-cover factor may appear alongside
an intercept, with the fixed or hierarchical priors written in the model.
RK-PPL translates these declarations without imposing an identifiability or
posterior-propriety test. Offsets are plain data or scalar value summands.
A grouping axis may be computed from data, for example
`z = x .+ 1; c[levels(z)] .~ Normal.(0, s); mu = c[z]`. Binding computes
these grouping values once and preserves the caller's data.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("10_levels_prior.jl", :rkppl_levels)
```

## Group-level effects

Statistical varying-effect bodies and their defaults are owned by
BayesianRegressionModels. Obtain a body with
`BayesianRegressionModels.rkppl_model(:varying_coefs)` or its correlated,
centered, multi-membership or stratified counterpart, then bind that ordinary
submodel in the defining module. RK-PPL compiles the author's declared arrays,
priors, gathers and matrix arithmetic.

For hand-authored arrays, `b[levels(g)] .~ Normal.(0, scale)` declares one
value per group and `b[g]` gathers it per observation. Multi-membership can
use `gg = vcat(g1, g2)` as a data-only definition, declare one array over
`levels(gg)`, and combine its gathers with the authored weights.

Multivariate rows use
`eachrow(B[levels(g), 1:K]) .~ MvNormalCholesky(mu, F)`;
columns use `eachcol(B[1:K, levels(g)]) .~ MvNormalCholesky(mu, F)`.
The mean, factor and their priors are declared values. Read rows with
`B[g, 1]`, or columns with `B[:, g]`. Integer axes use positive integer
positions; level axes map the authored labels.

Valid multivariate priors compile with Reactant when prepared with
`on_error = :ignore`. This explicit preparation policy strips visible
throws and assertions; native preparation checks the factor or covariance.
The shared solve retains its loops and ordinary reverse AD. Compiled empty
gradients remain subject to the [Reactant export limitation](constraints.md#acceptance-and-existing-limitations).

## Declared arrays as values

`z[levels(g), 1:K] .~ Normal.(0, 1)` declares a groups × K matrix;
`L ~ LKJCholesky(K, eta)` declares a lower-triangular matrix; an explicit
third argument `'U'` returns its upper-triangular counterpart. Sized vectors
(`sd[1:K] .~ HalfNormal.(1)`) and multivariate rows
(`eachrow(c[levels(g), 1:K]) .~ MvNormalCholesky(zeros(K), F)`) are also plain
Julia values. Their priors are explicit, and their dimensions determine their
packed coordinates.

The LKJ shape `eta` is a scalar value: it can be a positive literal, bound
data, a sampled parameter, or a definition using those values. A per-level
LKJ prior can share that same value across all its factors.

Compiled reverse mode currently fails for a data-sized whole factor with a
live shape when its retained diagonal prior and shared response expressions
both read the factor. Native density and gradients and compiled primal pass.
The literal two-dimensional and per-level forms have compiled reverse coverage;
see the [backend limitation](constraints.md#acceptance-and-existing-limitations).

Joint correlated responses can also use explicit factor priors:
declare `sd[1:2] .~ Gamma.(2, 1)` and `C ~ LKJCholesky(2, eta)`,
then bind `F = sd .* C` and use
`[y1, y2] ~ MvNormalCholesky([mu1, mu2], F)`. The scale vector has
positive support and the factor width matches the outcome count.

Array-valued definitions retain known axes: `b = z * (sd .* L)'` is groups ×
K, so `b[g, 1]` gathers one margin per observation. Positional reads such as
`L[2, 1]` and `M[:, 1]` remain ordinary Julia reads. A submodel's returned
array follows the same rule, and so does the expression written inline:
`(z * (sd .* L)')[g, 1]` gathers exactly as `b[g, 1]`, level lookup included. A data-only definition can size a declared array
through `levels(gg)`, as in the multi-membership example above.

Level axes preserve the order of their source. `z[levels(g)]` uses the
`DataAPI.levels` pool, including unobserved categorical levels;
`z[unique(g)]` uses first occurrence order. A supplied vector or range
`lv`, or a data-only definition such as `lv = reverse(unique(g))`, can
declare `z[lv]` in that order. Selections such as
`levels(g)[1:2:end]` and `unique(g)[3:-2:1]` select positions in the
source pool. Labels compare with `isequal`, including `missing`, `NaN`,
and array values. A column read only as labels may contain `missing`;
numeric responses follow the provisional automatic missing-observation
handling described under Plates.

For paired crossed effects, index each axis by one observation's label
inside a plate: `mu[i] = a + b[g[i], h[i]]`. Julia's `b[g, h]` with two
vectors selects a Cartesian matrix.

BRM's `StatisticalPreparation.gp_exp_quad_cov` and `gp_periodic_cov` functions
build covariance
with an RK plate over two location axes. Data-only locations cache pair
distances during preparation; locations derived from parameters retain live
distance calculations. The covariance matrix remains dense and diagonal jitter
depends on position, including when two locations are equal. Dense Cholesky in
`gp_chol_latent` delegates its factorization to the general RK-proper
`rk_cholesky_lower` callable, preserving native gradient support;
compiled covariance support does not imply compiled Cholesky gradients.
Import these helpers into the model module as ordinary Julia functions. For
explicit graph composition, load RK and RKPPL and use
`BayesianRegressionModels.rk_model(:gp_exp_quad_cov)` or
`rk_model(:gp_periodic_cov)`; an assigned graph call follows the general
composition path. The adopted owner API is available at BRM
`1f296dac2c086347525194887500948edb213228`.
Opaque prepared covariance callbacks with bound locations currently reach an
Enzyme activity error on Julia 1.10.12 / Enzyme 0.13.210. Assigned owned graph
calls pass native Reverse with bound or live locations. The generic
`benchmark/repro_enzyme_prepared_pair_callback.jl` reproduces the callback
boundary using a synthetic pair grid; this limitation precedes the ownership
cleanup.

A single index on a multi-axis definition is linear and positional:
`b = (z * sd)'` makes a row, and `b[g]` reads its column-major positions.
A module function's result has no tracked level axes: `b = identity(z)`
followed by `b[g, 1]` uses integer positions too. Binding checks that positional
indices are positive integers and within known sizes. If the result's size
depends on a function call, Julia checks its bounds when the result is read.
Linear indexing of an adjoint row currently supports native execution and
Enzyme gradients; Reactant tracing fails on that wrapper. Linear indexing of
a plain matrix and positional gathers from module-produced arrays compile.
See [backend limitations](constraints.md#acceptance-and-existing-limitations).

Gathering by a derived column is not supported yet. Pass a declared array
whole to a model-level Julia function when it needs the full value; a bare
array combined directly with observation data must first be indexed to the
observation axis.

## Design matrices

Bind the matrix once with `hcat` and size the coefficients with an axes prior.
Use `ones(length(x1))` for an intercept column. Scalar prior arguments are
shared across elements; literal vectors give one value per element.

Direct `X * b` reads can use the affine design path. A product passed to a
function, such as `f(X * b)`, reads the matrix as an ordinary Julia value.
Naming it first (`p = X * b; mu = f(p)`) or returning it from a submodel
preserves the inline expression's density, priors and coefficient coordinates.
A direct product can also be an affine component in a composed predictor,
such as `p = X * b; mu = p .+ q` with another component `q`.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("46_matrix_gaussian.jl", :rkppl_matrix)
```

Spline and approximate GP preparation and model bodies belong to BRM.
RK-PPL accepts their supplied matrices and ordinary declared coefficient
priors. See [Statistical model ownership](rkppl-smooths.md).

## Plates

`@plate for i in R … end` writes a loop whose cells each mean one iteration
of that Julia loop. Scalar arithmetic needs no dots; vector and matrix
intermediates use ordinary Julia broadcasts and matrix products.
Shapes come from named data (the range, a data index column), never from a
separate size argument. A cell holds observations, per-cell latents, per-cell
submodel calls and cell locals. The whole loop lowers at once, exactly like its
broadcast spelling.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("99_plate_32_gaussian.jl", :rkppl_plate)
```

Observation statements cover their whole response. Write `y .~ Normal.(mu, sigma)`
or a plate over the full response indices. Binding automatically skips entries
equal to `missing`; authors do not select the present rows manually. An authored
subset such as `@plate for i in 2:6` over a six-element response is refused,
including when the omitted entry is missing. An empty loop covers only an empty
response. Indexing outside the bound array still fails.

This missing-observation behavior is **provisional** (user decision `1uhcm3b`,
October 5, 2026). Binding owns concrete numeric response arrays and presence masks;
the likelihood plate skips absent entries, leaving zero at those positions in
pointwise output. Response axes, intermediate values and declared parameter sizes
remain unchanged, and caller arrays are not modified. No missing-response latent
is introduced. Missing responses read elsewhere as model values and missing
components of joint outcomes still need richer handling; those uses fail rather
than silently inventing values or discarding supplied joint components.

The current default Reactant runtime can fail compilation when a matrix-vector
predictor feeds a strided presence gather. The backend-only reproducer is
`benchmark/repro_reactant_strided_dot_gather.jl`; native execution remains
available. This compiler repair is tracked separately from missing-response
binding. Successful compiled checks for other shapes do not establish support
for this affected shape.

Some short guarded likelihood batches expand into repeated scalar work during
backend optimization. RK must emit retained loops or batched array structure;
subsequent semantics-preserving backend unrolling is diagnostic, following the
user's October 5 scope clarification in [Core constraints](constraints.md).
The Beta and Binomial checks record emitted traces, optimized MLIR and actual
executable HLO alongside values and ordinary reverse results.

Responses may have different row counts. Each statement reads columns on its
own observation axis; statements that read a common observation column must
agree on its rows. A latent plate follows its authored iterator's own extent,
including an iterator over a definition, a scan trajectory or a declared
parameter. Binding resolves declared axes and data-only call results; a scan
trajectory uses its authored bound. These extents do not come from an unrelated
response. Indexed prior arguments select those authored cells, and rebinding
recomputes their sizes from the new data.

An opaque parameter-dependent iterator whose shape cannot be inferred from
bound data remains a sizing capability gap. Binding names the iterator and
unavailable extent, rather than silently borrowing response rows or evaluating
sampled values on the host. The historical scan form with an unsupplied length
name still uses its direct response consumers; a latent iterator over that scan
needs an established bound.

Design matrices follow their input rows.
Declared `axes(X, 1)` arrays have X's rows; `axes(X, 2)` coefficient vectors
have X's width. The total `n_obs` does not size these values. A trajectory used
by responses of different lengths fails binding because its axis is ambiguous.
The legacy panel sampling do-block and `@plate result for ...` forms are
retired. Write indexed observations and explicit array dimensions; binding
no longer infers panel shapes from `dims` keys.

Per-level and per-observation cells compose through declared arrays. `L[k] ~ LKJCholesky(K, eta)` inside a plate over `levels(s)` declares
one factor per stratum. An observation cell can then read `sd[s[i], :]`,
`L[s[i]]` and `z[g[i], :]`, compute a matrix-vector product, and name a
scalar output with `r[i] = ...` or a row with `b[i, 1:K] = ...`. Shared arrays
enter the generated RK plates through `Ref`; increasing observations or
strata does not multiply their compiled cell regions.

A per-level row statement, `b[j, :] ~ MvNormalCholesky(zeros(2), F)` inside a
plate over `levels(g)`, is equivalent to
`eachrow(b[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), F)`.
`b[j, 1:2]` and `MvNormal` also work; these multivariate priors use shared
arguments. Scalar priors can vary by level, and a level cell can define
deterministic values:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("101_level_plate_values.jl", :rkppl_level_values)
```

`c[k] ~ Normal(m[k], 1)` also works with a bound Julia array `m`.
Here `m[k]` uses the authored Julia index; a declared `z[levels(g)]` instead
uses its level axis, so noncontiguous and string labels gather the correct
coordinate. Deterministic outputs retain that axis for later reads such as
`d[g]`. Each cell remains one RK plate body as the number of levels grows.
Declared array reads inside a level cell are aligned by label before entering
the plate. With `z[levels(g)[2:end]]`, `z[k]` is zero for the omitted first
level and reads the selected coordinate for every other level. A declaration
on `levels(h)` also works when every loop label occurs on that full axis;
the order and number of its coordinates can differ from `levels(g)`.
Matrix row reads follow the same rule. Raw bound-array reads and lazy
branches keep their authored Julia semantics.
Reactant's remaining limitation for an invalid host-constant index inside
an inactive live branch is recorded in the [core constraints](constraints.md).

In an array cell, an observation must read a named per-index output
(`y[i] ~ Normal(r[i], sigma)`), rather than a cell local directly.

`@scan begin … end` writes a sequential recurrence, such as an AR(1) state
(corpus `38_scan_ar.jl`).

### One array per index

A dotted cell broadcasts over its own iteration's values, as the loop it
writes does in Julia. When the response holds one array per index, such as
ragged groups including empty ones, each cell observes that index's entries:

```julia
# y = [[0.7, 0.3, 0.1], [1.4, 0.6], Float64[]]; x has the same shape
@plate for i in eachindex(y)
    y[i] .~ Normal.(a .+ b .* x[i], sigma)
end
```

Values the cell reads per index supply that index's array or number. Examples
are `x[i]`, a per-group scalar `mu[i]`, or the per-group result `loc[i]` of a
function-shaped kernel. Every other value is shared by all indices, and each
per-index value broadcasts against its response array as Julia requires. The
observation lowers to RK's nested group and observation plates: one retained
observation plate runs inside the group plate, with no copy per group. The
`:pointwise` query returns one array of densities per index, and empty arrays
contribute zero. Native values and ordinary Enzyme reverse gradients are
supported; RK does not implement compiled nested plate regions.

Every observation a flat response admits runs on each index's entries the
same way. This includes link families such as `BernoulliLogit.(eta[i])` and
caller-owned sampling laws such as `LogDensity.(score, loc[i], sigma)`, where
a visible `KernelSpec` law runs inside the inner observation plate. A
per-index value may combine with shared values in any distribution argument,
as in `Normal.(loc[i], dose[i] * sigma)`. The response may also be a
definition that reads only data, such as `y = group_cells(raw, rows)` or a
gather `y = cells[perm]`; binding evaluates it once and validates its arrays
as the response. The data the definition reads are its inputs, not
observation operands.

Outside a dotted cell, Julia refuses this shape, and so does binding.
`y .~ Normal.(loc, sigma)` broadcasts `Normal` over the arrays of `y`, and an
undotted cell `y[i] ~ Normal(…)` observes the array `y[i]` with a univariate
distribution; the error names the dotted cell. Missing entries inside the
arrays and `mi()` packing are not built yet.

## Submodels

`@rkppl name(args...) = begin … end` defines a reusable block. Using it,
`sigma ~ half_scale(1.0)`, expands it inline under the left-hand side's name, so
it lowers exactly like the hand-inlined program. A submodel whose result is a
response pointer is used as an observation stream: `y ~ stream(x, g)`.
`Base.merge(model, override)` replaces or appends statements by name.
Submodels also accept declared keyword defaults and statement replacements.
For a caller-defined `half_scale`, `custom = merge(half_scale,
:(tau ~ HalfCauchy(0.5)))` derives a reusable variant.

Submodel calls accept only their declared keywords. The legacy undeclared
`predictor = name` shortcut is rejected. Bind a returned quantity with an
ordinary assignment or declaration to give it an explicit use-site name;
a declared keyword named `predictor` keeps its ordinary argument meaning.
Catalogue spellings such as `r2d2`, `spline_basis`, and `dummy` are also
ordinary quantity names in declarations and assignments.

Each call owns a lexical namespace. For `z ~ sm(x)`, `z.b` reads the
submodel's local `b`, and bare `z` is the actual returned Julia value in
arithmetic and function arguments. Nested calls compose paths: `z.w.b`.
Locals never create caller bindings, so `a.b_c`, `a_b.c`, and a caller
parameter or data value `a_b_c` coexist.

Property access checks the local namespace first. If the scope has no local
with that name, it reads a property of the returned value. Explicit
`getproperty(z, :b)` always reads the returned value, even when `z.b` names
a local. The caller can read sampled and deterministic locals, including
data-only definitions. Ordinary caller assignments cannot redefine them;
explicit `merge` operations replace their declarations. Loop variables remain local
to their loop. Per-cell calls support `theta[i].b` and the whole local array
`theta.b`; their scalar-body restrictions still apply.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_submodel_example()
```

### Consumer-owned models

Statistical model bodies belong to BayesianRegressionModels, and PK-specific
bodies belong to downstream RKPPLBench. Import the consumer-owned body into
the model module and use it as an ordinary submodel. RK-PPL supplies lexical
namespaces, declared priors, arrays, plates, scans and standard AD.

## Replace, pin, and condition

Each operation returns a new model. The original program and shared submodel
remain reusable. A statement replacement changes the declaration in place;
an indexed replacement must match its complete left-hand side.

For example, `merge(model, :(b[axes(X, 2)] .~ Normal.(0, 2)))` changes an
array prior, and `merge(model, :(z.w.tau ~ HalfCauchy(0.5)))` changes a
nested local's prior.

Scoped replacements use the same dotted paths as ordinary reads. Derive a
reusable submodel variant with `merge(submodel, :(tau ~ HalfNormal(2)))`;
relative paths in a nested variant are expanded separately at each call.

A pin removes the entire declaration and stores its number or array as input
data. It adds no density and no sampled coordinate. A call keyword naming a
declaration has the same meaning.

Scoped keyword pins use `var"z.w.tau" = 0.3`.

Conditioning keeps the sampling statement. Its density at the given
constrained value contributes to `likelihood`; the observed variable has no
sampled coordinate and no transform Jacobian. For example, observing
`tau ~ HalfNormal(1)` at `0.3` contributes
`logpdf(truncated(Normal(), 0, Inf), 0.3)`. A pin of `tau` contributes nothing.

`model(; x) | (; y, tau = 0.3)` and
`condition(model(; x); y, tau = 0.3)` are equivalent.
`condition(plan; tau = 0.6)` rebinds that observation in a new bound plan.

This example executes all four operations and checks the density retained by
conditioning against Distributions.jl:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_rewrites_example()
```

Direct lowering declares observation roles separately from supplied data:
`lower_rkppl(ast, data; conditioned = (:y, :tau))`, then `bind_data(plan, data)`.
Both calls accept a `NamedTuple` or a symbol-keyed `AbstractDict` of data values;
the same container can pass through the pipeline. Binding returns a new plan
and also supports rebinding an already-bound plan with either container.
Pass a captured model to `lower_rkppl` when it carries scoped merge edits.
Sized array observations must have the declaration's shape. Partial indexed
writes do not replace a complete declaration and fail explicitly.

## Querying and fitting

`prepare_sampler` adds a reverse-mode gradient over the same cut. The
gradient's executable authority is the package test suite, so it is shown here
as source:

```julia
using Enzyme
using DifferentiationInterface: AutoEnzyme

q = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
q(u)                                   # log posterior (world-age safe)
g = similar(u)
value, _ = sampler_value_and_gradient!(q, g, u)
prepare_query(built, plan, :likelihood)  # presets: :sampler | :likelihood | :prior | :log_jacobian
constrain(built.layout, u)               # named constrained values
restore_draws(built.layout, U)           # U: layout.total × draws
```

- `:sampler` is the posterior preset; `:posterior` is the generated node's
  name, not a preset.
- `constrain`, `unconstrain` and `logjac` follow the number type of their
  input: `BigFloat` input gives `BigFloat` values, and other number types,
  such as dual numbers, pass through the transforms unchanged. Plain reals
  give `Float64` values, as before. Caller-owned parameter geometry receives
  the same numbers in its endpoints. Generic number support does not establish
  native Enzyme support for these layout-based calls; use `prepare_sampler`
  for posterior gradients through the model graph.
- Explicit parameter declarations keep their authored names. For
  `a ~ Normal(0, 1); b ~ Normal(0, 2); mu = a .+ b .* x`, the coordinates
  are `a`, `b` and the constrained values are `nt.a`, `nt.b`.
  A level-sized declaration `c[levels(g)]` gives `c.1`, `c.2`, … and `nt.c`.
  Adding another reader preserves those names and the single declared prior.
  A submodel parameter `b` in call `z` has coordinate `z.b` (a vector has
  `z.b.1`, `z.b.2`, …) and constrained value `nt.z.b`. Nested calls return
  nested NamedTuples, such as `nt.z.w.b`; `unconstrain` accepts that same
  structure, and `restore_draws` stacks its leaves across draws. These
  containers include sampled locals; deterministic locals, observed slots,
  and submodel return values are read in the model itself.
  Generated draw blocks keep their existing names under the call. If an
  author local claims that name, the generated block gets the first free
  numeric suffix; the author local keeps its name. Read `coordinate_names`
  for the resulting block labels.
  Rebuild prepared layouts and draw mappings when migrating from old
  flattened names such as `z_b`. Unusual identifiers use Julia's `var"…"`
  spelling in coordinate labels, so a literal name `var"z.b"` stays distinct
  from the scoped path `z.b`.
  Read `coordinate_names(built.layout)` and `constrain`; rebuild old
  prepared models and packed-draw mappings when migrating.
- A kernel returned by `prepare_query` closes over code generated at build time.
  Call it from top level or through `Base.invokelatest`; `SamplerQuery` calls
  already carry that barrier.

### One cell of an observation

`prepare_cell_query` evaluates ONE cell of the plates that observe the named
observations: one iteration of an `@plate for` loop, such as one group's or one
subject's observations, or one entry of an elementwise observation `y .~ …`.
Only that cell runs, so its cost is what one iteration costs, plus any value
the cell reads whole (such as a predictor vector indexed `log_k[i]`).

```julia
q = prepare_cell_query(built, plan, :y)          # or (:y, :z): one loop's observations
q(u, i)                                          # Σ of iteration i's densities
s = prepare_cell_sampler(built, plan, :y, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
value, g = cell_value_and_gradient!(s, similar(u), u, i)
```

- Cell `i` is entry `i` of the observation's `:pointwise` array: summing the
  cells gives that observation's likelihood. For an `@plate for` loop holding
  one array per index, entry `i` is the sum of iteration `i`'s densities.
- The gradient is with respect to the whole packed vector; it is nonzero only
  on the coordinates cell `i` reads.
- Observations without a pointwise plate (a scalar `~` observation, a joint
  multivariate response) are refused, naming the observation.
- Queries are not thread-safe: prepare one per thread.

## Programs emitted by BRM

`RKBRMI(brmi)` lowers an `@brm` formula to this same surface. The emitted
program is ordinary `@rkppl` source: shared submodel definitions such as
`popefs_normal_i_c_r(...)` for an intercept + slope + group effect, plus a main
block. For plain population, group-level and GLM models, evaluating the printed
source by hand reproduces the backend's density exactly. Some BRM features
(ordinal discrimination, per-threshold coefficients, missing-data `mi()`) are
still patched into the lowered plan after lowering, so for those the printed
source is not yet the whole model. See the BRM documentation for the formula
side.

## Debugging

- `model_view(built)` displays an existing build: its coordinates, the built
  ReactiveKernels program (`readable_code(built.spec)`) and its structural
  graph (`kernel_graph(built.spec)`), as labeled sections in plain text or
  HTML. `built` is the `(; spec, layout)` returned by `build_kernel`, including
  the built model a BRM `RKBRMI` retains as `backend.model`. Pass
  `bound = plan` to add the generated pre-build `@kernel` program and
  `query = prepare_query(...)` to add that prepared program. The view only
  reads these values; nothing is lowered, bound, built or evaluated again.
- `recipe_inventory` lists a program's plates and scans with their nesting, so a
  structure check reads the public contract rather than internal operation
  types. For a model whose subject plate holds a child scan, beside its
  observation plate:

  ```julia
  structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
                        if e.kind !== :ordinary]
  structure(built.spec)                         # [(:plate, 0), (:scan, 1), (:plate, 0)]
  structure(prepare_sampler(built, plan, u; backend).kernel)   # the same, data bound
  ```

  `recipe_kind(recipe)` classifies one recipe, and `plate_body`/`scan_body`
  return a body plan; see the ReactiveKernels compiler page.
- `kernel_expr(plan, built.layout)` returns the generated `@kernel` program
  shown above. It reads with the model's names: a plate cell names each
  argument after the value it iterates (`y .~ Normal.(loc, s)` is
  `plate(y, loc, s) do y, loc, s ... normal(loc, s).logpdf(y)`), a prior cell
  reads an argument value it generated by its role
  (`normal(location, scale).logpdf(c)`), and a value computed for a
  distribution argument is named after its owner and role
  (`y .~ Normal.(loc, f(x))` defines `y_scale = f(x)`; `b ~ Normal(0, 2s)`
  defines `b_scale = 2s`). A name the model or its data already use takes the
  first free `_k`. Densities, gradients and coordinates do not depend on these
  names.
- `packages/ReactiveKernelsPPL/report/transpile_report.jl --surface model.jl
  --data data.jl` writes a markdown report of the real pipeline: the plan, the
  generated program, and the posterior at a probe point. With
  `--artifact model.jls` it reports on a serialized BRM emission instead.
- `SurfaceLoweringError` means the spelling is not admitted, and its message
  names the fix. `ContractValidationError` comes from the IR or binding layer.
