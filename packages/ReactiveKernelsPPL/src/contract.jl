# Emitter → thin-layer input contract (slice 1).
#
# Agreed text: contract v1+v2, the link-triple pin, positional-args +
# scalar-assignments micro-pins, and the ref-index pin (see contract leaf
# 2026-09-17T23-42-44-576-14qsrik comments), plus contract v3 (vector
# assignments: in-graph derived columns, 2026-09-18). This file is normative
# for the thin side; the BRM emitter mirrors admission emitter-side with
# BRM-side attribution (R8), and this validation stays as loud defense in
# depth.

"""
    ContractValidationError <: Exception

Thrown by [`validate_plan`](@ref) when a structural plan violates the
slice-1 input contract. Loud, never silent: every message names the
offending node label for BRM-side attribution.
"""
struct ContractValidationError <: Exception
    message::String
end
Base.showerror(io::IO, e::ContractValidationError) =
    print(io, "ContractValidationError: ", e.message)

"""Reference to a raw data column: a key into `StructuralPlan.columns`."""
const ColumnRef = Symbol
"""Reference to a sampled parameter or scalar assignment by name."""
const ParamName = Symbol

"""Slice-1 likelihood families (D3 narrow slice), plus the leveled slice-2
families (categorical / ordinal / multinomial): reference-coded
multi-logit categorical, cumulative-logit ordinal with ordered cutpoints,
general typed ordinal (2 structures × 3 links), shared-simplex multinomial
over a count matrix, and plain categorical over simplex probabilities."""
@enum LikelihoodFamily::UInt8 begin
    GaussianFam
    BernoulliLogitFam
    PoissonLogFam
    BinomialLogitFam
    NegativeBinomial2Fam
    GammaLogFam
    BernoulliProbitFam
    BernoulliCloglogFam
    BinomialProbitFam
    BinomialCloglogFam
    BetaLogitFam
    CategoricalLogitFam
    OrderedLogisticFam
    OrdinalFam
    MultinomialFam
    CategoricalFam
end

"""Link functions. The enum crosses the boundary; the thin layer owns the
inverse-link numerics (R1 counter)."""
@enum LinkFunction::UInt8 begin
    IdentityLink
    LogitLink
    LogLink
    ProbitLink
    CloglogLink
end

"""Slice-1 predictor term kinds (core set; stretch adds variants later).
`LatentTerm` carries a per-cell latent parameter vector (a [`PlateParameter`](@ref))
as the whole linear predictor (`lp = theta`, identity design) — the
random-effects / per-observation-latent location."""
@enum TermKind::UInt8 begin
    InterceptTerm
    ContinuousTerm
    FactorTerm
    OffsetTerm
    LatentTerm
    RanefGatherTerm
    SplineSummandTerm
    HSGPSummandTerm
    MonotonicTerm
    MonotonicSummandTerm
end

"""
    ResponseEvidence(kind, lower, upper)

Censoring/truncation evidence wrapper on an admitted family (D3 response
evidence). Bounds are literals or [`ColumnRef`](@ref)s. For
`:interval_censored` the response itself is the lower endpoint, so `lower`
must be `nothing` and `upper` is required.
"""
struct ResponseEvidence
    kind::Symbol # :none | :truncated | :censored | :interval_censored
    lower::Union{Nothing,Real,ColumnRef}
    upper::Union{Nothing,Real,ColumnRef}
end

"""
    LikelihoodSpec(family, link, response, predictor, scale, weights, evidence, label[, trials[, range]])

One independent response. `scale` is the response's scalar auxiliary —
Gaussian sigma, NB2 dispersion phi, Gamma shape alpha (parameter,
assignment, or folded literal) — and must be `nothing` otherwise. (One
slot covers every admitted family; a two-auxiliary family such as Beta
needs a new field — noted, not built.) `weights` is a
frequency/power-objective column (D1); analytic/precision weights fail
closed emitter-side. `trials` is the Binomial trial count (Int column or
Int literal), `nothing` otherwise. `range` carries a literal `y[1:N]`
response range (`nothing` = whole column: bare `.~`, `eachindex`,
`axes`); it must cover `1:n_obs` exactly (checked at bind).

Leveled families (categorical / ordinal / multinomial) use the trailing
fields, built with keywords (`n_levels=`, `thresholds=`,
`extra_predictors=`, `count_columns=`, `ordinal_structure=`,
`discrimination=`, `threshold_columns=`); every other family leaves them
at their defaults:

- `n_levels`: category count K (`nothing` = infer at bind: from the
  response column for OrderedLogistic/Ordinal/Categorical, structurally
  — 1 + predictor count — for CategoricalLogit, structurally — column
  count — for Multinomial).
- `thresholds`: ordered/vector threshold parameter ([`VectorParameter`](@ref))
  for OrderedLogistic/Ordinal, `nothing` otherwise.
- `extra_predictors`: CategoricalLogit non-reference predictors after
  `predictor` (class order 2..K is `[predictor; extra_predictors...]`,
  K−1 total); empty otherwise.
- `count_columns`: Multinomial count columns after `response`
  (category order 1..K is `[response; count_columns...]`); empty
  otherwise.
- `ordinal_structure`: `:cumulative`/`:stopping` for OrdinalFam,
  `nothing` otherwise.
- `discrimination`: OrdinalFam positive latent scale
  (`nothing` = 1.0), `nothing` otherwise: a positive Real literal, a
  finite-positive data column, or a modeled scale naming a LogLink plan
  predictor (positivity is structural via `exp` — the `log(disc)`
  recipe; any other link fails closed).
- `threshold_columns`: OrdinalFam per-threshold design columns
  (StoppingRatio only), empty otherwise.
- `threshold_coefs`: the (K−1)×p threshold-coefficient matrix packed as a
  `:vector_normal` [`VectorParameter`](@ref) (required exactly when
  `threshold_columns` is non-empty), `nothing` otherwise. Stage-major:
  stage j occupies entries `(j−1)*p+1 .. j*p`.

Multinomial/Categorical responses name their shared-simplex
[`VectorParameter`](@ref) in `predictor` (no linear predictor — the
scan-state precedent).
"""
struct LikelihoodSpec
    family::LikelihoodFamily
    link::LinkFunction
    response::ColumnRef
    predictor::Symbol
    scale::Union{Nothing,ParamName,Real}
    weights::Union{Nothing,ColumnRef}
    evidence::ResponseEvidence
    label::Symbol
    trials::Union{Nothing,ColumnRef,Int}
    range::Union{Nothing,UnitRange{Int}}
    n_levels::Union{Nothing,Int}
    thresholds::Union{Nothing,ParamName}
    extra_predictors::Vector{Symbol}
    count_columns::Vector{ColumnRef}
    ordinal_structure::Union{Nothing,Symbol}
    discrimination::Union{Nothing,Real,ColumnRef}
    threshold_columns::Vector{ColumnRef}
    threshold_coefs::Union{Nothing,ParamName}
end
LikelihoodSpec(family, link, response, predictor, scale, weights, evidence,
    label) =
    LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, nothing, nothing)
LikelihoodSpec(family, link, response, predictor, scale, weights, evidence,
    label, range) =
    LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, nothing, range)
# Full positional (pre-leveled 10-arg) with optional leveled keywords:
# existing 10-arg call sites keep working (leveled fields default); new
# leveled call sites pass keywords.
function LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, trials, range; n_levels::Union{Nothing,Int} = nothing,
        thresholds::Union{Nothing,ParamName} = nothing,
        extra_predictors::Vector{Symbol} = Symbol[],
        count_columns::Vector{ColumnRef} = Symbol[],
        ordinal_structure::Union{Nothing,Symbol} = nothing,
        discrimination::Union{Nothing,Real,ColumnRef} = nothing,
        threshold_columns::Vector{ColumnRef} = Symbol[],
        threshold_coefs::Union{Nothing,ParamName} = nothing)
    return LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, trials, range, n_levels, thresholds,
        extra_predictors, count_columns, ordinal_structure, discrimination,
        threshold_columns, threshold_coefs)
end

"""
    TermSpec(kind, columns, options, addressee, label)

One additive predictor term: structure only, never materialized designs
(D5a). `addressee` is the prior address (source column or `:Intercept`),
never a per-level label. Terms take no options: factor sizing lives in
the plan's [`LevelMap`](@ref)s (full-rank over exactly the mapped
levels; no contrasts, no reference dropping — that machinery was
BRM-specific and is gone).
"""
struct TermSpec
    kind::TermKind
    columns::Vector{ColumnRef}
    options::NamedTuple
    addressee::Symbol
    label::Symbol
end

"""One named linear predictor: additive terms over raw columns."""
struct PredictorSpec
    name::Symbol
    link::LinkFunction
    terms::Vector{TermSpec}
    label::Symbol
end

"""
    PopulationPrior(predictor, addressee, location, scale)

Normal-only population prior (slice 1) addressed by
`(predictor, column|:Intercept)`. A factor source column address applies one
shared Normal across its full-rank level block (one coefficient per mapped
level). The emitter fills `Normal(0,1)` defaults so coverage is complete by
construction — except factor coefficients, whose broadcast prior also sizes
the block and is therefore required, never defaulted.
"""
struct PopulationPrior
    predictor::Symbol
    addressee::Symbol
    location::Real
    scale::Real
end

"""
    SupportOverride

A latent's support override: `nothing` (infer from the family), the bare
`Symbol` `:positive` (half-Normal/half-Cauchy truncation of a real-support
family, exact +log(2) at a literal-zero location), or the tuple
`(:interval, lo, hi)` (a two-sided finite truncation `truncated(Normal(mu, s),
lo, hi)` — an affine-logistic constrained transform onto `(lo, hi)` with the
renormalized truncated density). Shared by scalar [`SampledParameter`](@ref)s
and per-cell [`PlateParameter`](@ref)s.
"""
const SupportOverride = Union{Nothing,Symbol,Tuple{Symbol,Float64,Float64}}

"""
    SampledParameter(name, family, args, support_override, label)

One non-coefficient latent (scalar, slice 1). `args` use POSITIONAL keys
`(arg1, arg2, …)` in Distributions.jl constructor order with Distributions.jl
semantics (`Exponential(θ)` = scale θ). Values are literals or
[`ParamName`](@ref)s (hierarchical OK, cycles rejected). `support_override`
is a [`SupportOverride`](@ref): `nothing` (infer from family), `:positive`
(half-Normal/half-Cauchy), or `(:interval, lo, hi)` (a finite truncated
interval).
"""
struct SampledParameter
    name::ParamName
    family::Symbol
    args::NamedTuple
    support_override::SupportOverride
    label::Symbol
end

"""
    PlateParameter(name, family, args, support_override, range, label)

One per-cell latent parameter VECTOR (`@plate` sampling statement, e.g.
`theta ~ Normal(mu, tau)`): `size` independent draws from `family`, one per
plate cell, packed as one contiguous block. `family`/`args`/`support_override`
follow [`SampledParameter`](@ref) exactly (POSITIONAL `(arg1, …)` keys,
Distributions.jl semantics), except the args are SHARED across cells — literals
or scalar parameter/assignment names (per-cell vector args are a later
increment). `range` is `nothing` (size = `n_obs`, the `eachindex`/`axes` /
bare-plate case) or a literal `UnitRange{Int}` that must cover `1:n_obs` exactly
(the `1:N` case), mirroring [`LikelihoodSpec`](@ref)'s response range. A
real-support prior (`normal`/`cauchy`/half-versions) lays out identity (an
unconstrained block); positive/unit support constrains per element.
"""
struct PlateParameter
    name::ParamName
    family::Symbol
    args::NamedTuple
    support_override::SupportOverride
    range::Union{Nothing,UnitRange{Int}}
    label::Symbol
end
"""Provenance/range default to a whole-column (n_obs) plate under the name."""
PlateParameter(name::ParamName, family::Symbol, args::NamedTuple,
    support_override::SupportOverride) =
    PlateParameter(name, family, args, support_override, nothing, name)
PlateParameter(name::ParamName, family::Symbol, args::NamedTuple,
    support_override::SupportOverride, range::Union{Nothing,UnitRange{Int}}) =
    PlateParameter(name, family, args, support_override, range, name)

"""
    RanefZRecipe(kind, column, level)

One random-effect design (Z) column recipe: structure only, never a
materialized vector pre-codegen. `kind` is `:ones` (intercept — `column`
is `:none`, `level` is `nothing`), `:column` (continuous raw column),
or `:dummy` (indicator over `column`: `level` is the level VALUE for
`Int`, exact match for `AbstractString`). The thin layer performs NO
coding inference — treatment/cell-means decisions arrive as explicit
`dummy` recipes from the emitter.
"""
struct RanefZRecipe
    kind::Symbol
    column::Symbol
    level::Union{Nothing,Int,AbstractString}
end

"""
    RanefMargin(predictor, coefficient, z)

One random-effect margin in `ranefcoefnames` order within its target
slice: `predictor` owns the slice, `coefficient` is the margin address
(`:Intercept`, a column, or a dummy label), `z` is the [`RanefZRecipe`](@ref)
for its Z column.
"""
struct RanefMargin
    predictor::Symbol
    coefficient::Symbol
    z::RanefZRecipe
end

"""
    RanefBucket(id, group, kind, margins, slices, lkj_eta, label)

One shared random-effect draws block (SB mirror): non-centered geometry
over `K = length(margins)` margins in `G` groups of raw column `group`.
`id` is `nothing` for a plain `(x|g)` single-slice bucket, else the `|ID|`
label. `kind` is `:intercept1` (plain single-`1`: log-scale/xi geometry),
`:slope1` (plain single slope: tau/xi geometry), or `:correlated` (LKJ +
tau + z_flat; ALL ID buckets, even K=1 — SB's ID path has no K=1 fast
path). `slices` maps each target predictor to its static column range;
the ranges partition `1:K` contiguously in body order. `lkj_eta` is the
LKJ shape (correlated only; `NaN` otherwise). `label` is `:bucket_<suffix>`
(`<ID>_<group>` or plain `<group>`).
"""
struct RanefBucket
    id::Union{Nothing,Symbol}
    group::ColumnRef
    kind::Symbol
    margins::Vector{RanefMargin}
    slices::Vector{Tuple{Symbol,UnitRange{Int}}}
    lkj_eta::Float64
    label::Symbol
end

"""
    VectorParameter(name, family, args, size[, label])

One constrained VECTOR parameter for a leveled response (cutpoints,
thresholds, or a shared simplex), packed as one contiguous block:

- `:ordered_normal` — an ordered cutpoint/threshold vector with an
  elementwise `Normal(arg1, arg2)` prior (Stan `ordered` semantics: no
  factorial normalizer) + the ordered-transform Jacobian.
- `:vector_normal` — a plain (unconstrained, identity-transform) vector
  with an elementwise `Normal(arg1, arg2)` prior (stopping-ratio stage
  thresholds).
- `:simplex_dirichlet` — a simplex with a `Dirichlet(arg1)` prior
  (`arg1` a literal concentration vector — frozen data, the
  coefficient-prior precedent) + the stick-breaking Jacobian.

`size` is the constrained length (K−1 for thresholds, K for a simplex;
`nothing` = infer at bind from the linked leveled response). Args are
LITERALS only (hierarchical threshold/Dirichlet concentrations fail
closed — planned). K=1 is uniform: zero-length threshold vectors carry
no statements/prior/Jacobian, and a 1-simplex is the constant `[1.0]`.
"""
struct VectorParameter
    name::ParamName
    family::Symbol
    args::NamedTuple
    size::Union{Nothing,Int}
    label::Symbol
end
"""Provenance defaults to the parameter's own name."""
VectorParameter(name::ParamName, family::Symbol, args::NamedTuple,
    size::Union{Nothing,Int}) =
    VectorParameter(name, family, args, size, name)

"""
    SplineBasisBlock(name, width, columns)

One fitted basis block (`:fixed`, `:pen`, `:rr`, `:rn`, `:nr`): `width`
is static from `k` (known at lowering); `columns` holds the materialized
bound-vector names, filled at bind (empty pre-bind).
"""
struct SplineBasisBlock
    name::Symbol
    width::Int
    columns::Vector{Symbol}
end

"""
    SplineBasis(id, kind, axes, k, blocks, label)

One spline term's basis recipe (SB mirror): `kind` is `:tps` (`s(x)`,
one axis) or `:t2` (`t2(x,z)`, two axes); `k` is the basis dimension
(`Int` for `s`, `(Int,Int)` for `t2`). `blocks` are static in
name/width (`:fixed`/`:pen` for `s`, `:fixed`/`:rr`/`:rn`/`:nr` for
`t2`); bind fits the basis from the raw `axes` columns (host-side
transformed-data mirror — eigen is inexpressible in-graph) and fills
each block's materialized `columns`. A term feeds exactly one predictor
(the `spline(:id)` use-site).
"""
struct SplineBasis
    id::Symbol
    kind::Symbol
    axes::Vector{Symbol}
    k::Union{Int,Tuple{Int,Int}}
    blocks::Vector{SplineBasisBlock}
    label::Symbol
end

"""
    SplineVector(name, family, args, support_override, width, basis, label)

One free spline coefficient block (SB `_sb_s_generic`/`_sb_t2_generic`):
flat `b_fixed`, standard-normal `b_*_raw`, half-normal `sd` (`:normal`
+ `:positive`). `width` is static from `k`; `basis` is the owning
[`SplineBasis`](@ref) id. Layout packs each as one contiguous block
(plate-shaped); priors broadcast over cells.
"""
struct SplineVector
    name::Symbol
    family::Symbol
    args::NamedTuple
    support_override::SupportOverride
    width::Int
    basis::Symbol
    label::Symbol
end

"""
    HSGPBasis(id, axes, K, c, iso, fits, label)

One Hilbert-space GP basis (SB `_sb_hsgp`): `axes` raw data columns,
`K` modes per axis, `c` boundary factors per axis (`L =
c*max|x-mu|`, `c > 1`), `iso` length-scale sharing. `fits` holds the
bind-time `(mu, L)` per axis (empty pre-bind); the basis itself is
evaluated in-graph from the raw columns + frozen fits (trig is
elementwise-expressible — unlike spline eigen, no bind-time
materialization). `M = prod(K)` basis functions; the term owns
`beta_raw_<id>` (M-vector), `rho_<id>` (iso scalar) or
`rho_<id>_1..d` (aniso scalars), `sigma_<id>` (scalar).
"""
struct HSGPBasis
    id::Symbol
    axes::Vector{Symbol}
    K::Vector{Int}
    c::Vector{Float64}
    iso::Bool
    fits::Vector{Tuple{Float64,Float64}}
    label::Symbol
end

"""
    KernelPlate(result, subjects, timepoints, slices, assignments, obs, collected, label)

One panel kernel (BRM `_RKKernelPlan`, panel v1): `result` is the collected
per-subject name (the plate LHS); `subjects` the subject count (integer
literal, or a dims-key `Symbol` resolved at bind); `timepoints` the
per-subject timepoint count (`nothing` for all-scalar models, an integer,
or a dims-key `Symbol` resolved at bind); `slices` the
`(data column, cell param, kind)` triples with `kind ∈ (:vector, :scalar,
:unknown)` (`:unknown` pre-bind — kinds resolve from lengths at bind);
`assignments` the cell-local `name => expr` pairs in cell order (flat
elementwise vocabulary); `obs` the single in-cell observation
`(response, family, location, scale)` with `response` a cell param;
`collected` the trailing collected cell name. Grouping is ABSENT for
panel (implicit 1:n subjects — structural, no sentinel).
"""
struct KernelPlate
    result::Symbol
    subjects::Union{Int,Symbol}
    timepoints::Union{Nothing,Int,Symbol}
    slices::Vector{Tuple{Symbol,Symbol,Symbol}}
    assignments::Vector{Pair{Symbol,Any}}
    obs::NamedTuple{(:response, :family, :location, :scale)}
    collected::Symbol
    label::Symbol
end

"""Flat length of a resolved kernel plate: `n_sub * T` (`T = 1` all-scalar)."""
_kernel_flat_length(n_sub::Int, T::Union{Nothing,Int}) =
    T === nothing ? n_sub : n_sub * T

"""Materialized flat-expansion column for a scalar slice (spline-blocks
precedent: deterministic bind product, caller collisions fail closed)."""
_kexp_name(result::Symbol, col::Symbol) = Symbol("$(result)_kexp_$(col)")

"""Columns a kernel plate manages (slice columns + scalar expansions):
exempt from the uniform-`n_obs` rule, validated under kernel rules.
Expansions exist only for resolved vector models (`T isa Int`); gating
on that keeps a stray caller column from hiding behind the exemption."""
function _kernel_managed_columns(kp::KernelPlate)
    out = Set{Symbol}()
    for (col, _, kind) in kp.slices
        push!(out, col)
        kind === :scalar && kp.timepoints isa Int &&
            push!(out, _kexp_name(kp.result, col))
    end
    return out
end

"""
    AssignmentSpec(name, expr)

One scalar temporary (slice 1): `expr` may reference scalar names and
whole-column reductions only. Row-varying (elementwise) assignments fail
closed emitter-side with "precompute the column". A bare `Symbol` aliases
another scalar name; a bare `Real` is a folded literal.
"""
struct AssignmentSpec
    name::Symbol
    expr::Union{Expr,Symbol,Real}
    label::Symbol
end
"""Provenance defaults to the assignment's own name."""
AssignmentSpec(name::Symbol, expr::Union{Expr,Symbol,Real}) =
    AssignmentSpec(name, expr, name)

"""
    VectorAssignmentSpec(name, expr)

One in-graph derived column (contract v3): `expr` is elementwise over bound
raw columns (dotted arithmetic/comparisons/math, `ifelse`, bare-column
reductions as scalar subterms) plus scalar parameter/assignment references.
Length is `n_obs` by construction (every admitted form preserves length);
the generator emits the computation into the kernel body and `bound=`
partial evaluation folds data-only derivations exactly once (the StanBlocks
transformed-data analog). A bare-`Symbol` `expr` aliases another column
(raw or derived). Factors take raw grouping columns only (levels need
pre-evaluation knowledge); continuous/offset terms and reduction arguments
admit derived names.
"""
struct VectorAssignmentSpec
    name::Symbol
    expr::Union{Expr,Symbol}
    label::Symbol
end
"""Provenance defaults to the derived column's own name."""
VectorAssignmentSpec(name::Symbol, expr::Union{Expr,Symbol}) =
    VectorAssignmentSpec(name, expr, name)

"""One literal-index seed fill of a `@scan` carried array: `state[index] ~ Dist(args…)`.
`family` is an internal family symbol (see the surface's `_PARAM_FAMILIES`); `args`
are the positional distribution-argument expressions (literals in a seed fill)."""
struct ScanSetup
    index::Int
    family::Symbol
    args::Vector{Any}
end

"""
    ScanStep(kind, target, indexed, family, args, expr)

One `@scan` recurrence-body statement.

- `kind === :sample` — a `~` statement: the carried array at the loop index
  (`indexed = true`, `target === state`, centered form) or a fresh per-step local
  innovation (`indexed = false`, non-centered form). `family`/`args` describe the
  distribution; `expr` is `nothing`.
- `kind === :assign` — a `=` statement: the carried array's deterministic write
  (`indexed = true`) or a per-step deterministic local (`indexed = false`).
  `expr` is the RHS AST; `family`/`args` are `nothing`.
"""
struct ScanStep
    kind::Symbol
    target::Symbol
    indexed::Bool
    family::Union{Symbol,Nothing}
    args::Union{Vector{Any},Nothing}
    expr::Any
end

"""
    ScanSpec(state, loopvar, lo, hi, setup, step, maxlag, label)

One sequential recurrence (`@scan begin <setup>; for loopvar in lo:hi … end end`):
the carried array `state`, the loop variable and range `lo:hi` (`hi` a literal `Int`
or a data length `Symbol`), the ordered `setup` seed fills, the ordered recurrence
`step`s, and the maximum backward lag read. `state` is the plan-level latent the
recurrence produces (the value visible after the block); the per-step locals and
`loopvar` are scoped to the recurrence and never enter the plan name table.
"""
struct ScanSpec
    state::Symbol
    loopvar::Symbol
    lo::Int
    hi::Union{Symbol,Int}
    setup::Vector{ScanSetup}
    step::Vector{ScanStep}
    maxlag::Int
    label::Symbol
end

"""
    LevelMap(predictor, column, values, source, subset)

Ordered level values sizing one full-rank factor term: `values` is the
coefficient position↔level mapping (binder-evaluated from the grouping
column, `[]` pre-bind). `source` is the levels function (`:levels`
only). `subset` selects from `sort(unique(column))`: `:` (full cover),
a literal `UnitRange{Int}`, a literal `Vector{Int}` of positions, or
`(lo, :end)`. Keyed by `(predictor, column)` — one map per factor term.
"""
struct LevelMap
    predictor::Symbol
    column::ColumnRef
    values::Vector
    source::Symbol
    subset::Union{Colon,UnitRange{Int},Vector{Int},Tuple{Int,Symbol}}
end

"""
    StructuralPlan(responses, predictors, population_priors, parameters,
                   assignments, derived, columns, n_obs)

Complete emitter→thin-layer input. `columns` maps RAW-column names to plain
vectors (derived columns are never bound — they compute in-graph from
`derived`); `parameters`, `assignments`, `derived`, each `plate_parameters`
latent, each `vector_parameters` latent, and each `scans` state share one
name table (duplicates rejected). N≥1 independent responses; shared
predictor Symbols allowed. `levelmaps` sizes every factor term
(binder-evaluated values); `plate_parameters` carries per-cell latents,
`vector_parameters` leveled-response latents (cutpoints/thresholds/simplexes),
`scans` sequential-recurrence latents, and `ranef_buckets` random-effect
draws blocks (empty for a plain population-GLM plan).
"""
struct StructuralPlan
    responses::Vector{LikelihoodSpec}
    predictors::Vector{PredictorSpec}
    population_priors::Vector{PopulationPrior}
    parameters::Vector{SampledParameter}
    assignments::Vector{AssignmentSpec}
    derived::Vector{VectorAssignmentSpec}
    columns::Dict{Symbol,AbstractVector}
    n_obs::Int
    roles::Dict{Symbol,Symbol}
    levelmaps::Vector{LevelMap}
    plate_parameters::Vector{PlateParameter}
    scans::Vector{ScanSpec}
    ranef_buckets::Vector{RanefBucket}
    vector_parameters::Vector{VectorParameter}
    spline_bases::Vector{SplineBasis}
    spline_vectors::Vector{SplineVector}
    hsgp_bases::Vector{HSGPBasis}
    kernel_plates::Vector{KernelPlate}
end

# Pre-extension full-positional constructor (9-arg): callers that built a plan
# before `levelmaps`/`scans`/`ranef_buckets`/`vector_parameters`/spline/hsgp
# nodes existed keep working with all empty.
StructuralPlan(
    responses::Vector{LikelihoodSpec},
    predictors::Vector{PredictorSpec},
    population_priors::Vector{PopulationPrior},
    parameters::Vector{SampledParameter},
    assignments::Vector{AssignmentSpec},
    derived::Vector{VectorAssignmentSpec},
    columns::Dict{Symbol,AbstractVector},
    n_obs::Int,
    roles::Dict{Symbol,Symbol}) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, LevelMap[], ScanSpec[],
        RanefBucket[], VectorParameter[], SplineBasis[], SplineVector[],
        HSGPBasis[], KernelPlate[])

"""Column roles: what a bound column IS (CV travel + program transforms read
this). Inferred at bind; explicit roles override. Unbound plans carry none."""
const COLUMN_ROLES =
    (:response, :predictor, :weight, :evidence, :trials, :group, :data)

# Compatibility constructor: 7-arg positional construction (pre-roles,
# pre-derived, pre-levelmaps) keeps working with empties; the
# emitter/serializer path is unaffected.
function StructuralPlan(
        responses::Vector{LikelihoodSpec},
        predictors::Vector{PredictorSpec},
        population_priors::Vector{PopulationPrior},
        parameters::Vector{SampledParameter},
        assignments::Vector{AssignmentSpec},
        columns::Dict{Symbol,AbstractVector},
        n_obs::Int;
        roles::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
        derived::Vector{VectorAssignmentSpec} = VectorAssignmentSpec[],
        levelmaps::Vector{LevelMap} = LevelMap[],
        plate_parameters::Vector{PlateParameter} = PlateParameter[],
        scans::Vector{ScanSpec} = ScanSpec[],
        ranef_buckets::Vector{RanefBucket} = RanefBucket[],
        vector_parameters::Vector{VectorParameter} = VectorParameter[],
        spline_bases::Vector{SplineBasis} = SplineBasis[],
        spline_vectors::Vector{SplineVector} = SplineVector[],
        hsgp_bases::Vector{HSGPBasis} = HSGPBasis[],
        kernel_plates::Vector{KernelPlate} = KernelPlate[])
    return StructuralPlan(responses, predictors, population_priors,
        parameters, assignments, derived, columns, n_obs, roles, levelmaps,
        plate_parameters, scans, ranef_buckets, vector_parameters,
        spline_bases, spline_vectors, hsgp_bases, kernel_plates)
end

"""Bound ⟺ columns attached. Unbound plans (empty columns) carry structure
only; [`bind_data`](@ref) attaches data (+ roles). Rebinding replaces."""
isbound(plan::StructuralPlan) = !isempty(plan.columns)

"""Admitted (family, likelihood-link, predictor-link) triples (triple pin).
Triples 2 and 3 lower identically; the triple is admission key + lowering
selector, never pairwise link equality. Multinomial/Categorical name a
simplex vector parameter instead of a linear predictor, so they skip the
triple (the scan-state precedent) and validate on the simplex path."""
const ADMITTED_TRIPLES = (
    (GaussianFam, IdentityLink, IdentityLink),
    (BernoulliLogitFam, LogitLink, IdentityLink),
    (BernoulliLogitFam, LogitLink, LogitLink),
    (PoissonLogFam, LogLink, LogLink),
    (BinomialLogitFam, LogitLink, IdentityLink),
    (NegativeBinomial2Fam, LogLink, LogLink),
    (GammaLogFam, LogLink, LogLink),
    (BernoulliProbitFam, ProbitLink, IdentityLink),
    (BernoulliCloglogFam, CloglogLink, IdentityLink),
    (BinomialProbitFam, ProbitLink, IdentityLink),
    (BinomialCloglogFam, CloglogLink, IdentityLink),
    (BetaLogitFam, LogitLink, IdentityLink),
    (CategoricalLogitFam, LogitLink, IdentityLink),
    (OrderedLogisticFam, LogitLink, IdentityLink),
    (OrdinalFam, LogitLink, IdentityLink),
    (OrdinalFam, ProbitLink, IdentityLink),
    (OrdinalFam, CloglogLink, IdentityLink),
)

"""Positional arity per sampled family (Distributions.jl order)."""
const SAMPLED_ARITY = Dict{Symbol,Int}(
    :normal => 2,
    :cauchy => 2,
    :exponential => 1,
    :gamma => 2,
    :lognormal => 2,
    :beta => 2,
    :inverse_gamma => 2,
    :flat => 0,
)

"""Inferred unconstrained support per sampled family (`:flat` = real)."""
const SAMPLED_SUPPORT = Dict{Symbol,Symbol}(
    :normal => :real,
    :cauchy => :real,
    :flat => :real,
    :exponential => :positive,
    :gamma => :positive,
    :lognormal => :positive,
    :inverse_gamma => :positive,
    :beta => :unit,
)

# Spline block/width/vector/name rules, derived purely from (kind, k):
# the single source of truth shared by surface lowering (which builds the
# nodes) and contract validation (which re-derives and compares). Widths
# are static — the fit can only confirm them at bind, never change them.
# Block order is SB's data order (:fixed first, then pen/rr/rn/nr); the t2
# sd index follows the pen-block position (rr→1, rn→2, nr→3).
function _spline_blocks(kind::Symbol, k::Union{Int,Tuple{Int,Int}})
    if kind === :tps
        k isa Int ||
            _fail(:plan, "tps spline k must be an Int, got $(repr(k))")
        return [(:fixed, 2), (:pen, k - 2)]
    elseif kind === :t2
        k isa Tuple{Int,Int} ||
            _fail(:plan, "t2 spline k must be an (Int, Int) tuple, got " *
                  repr(k))
        k1, k2 = k
        return [(:fixed, 3), (:rr, (k1 - 2) * (k2 - 2)),
            (:rn, (k1 - 2) * 2), (:nr, 2 * (k2 - 2))]
    end
    return _fail(:plan, "spline kind must be :tps or :t2, got $(repr(kind))")
end

_spline_block_names(id::Symbol, block::Symbol, width::Int) =
    [Symbol("$(id)_$(block)_$j") for j in 1:width]

function _spline_basis_columns(id::Symbol, kind::Symbol, k)
    return [(name, _spline_block_names(id, name == :fixed ?
        (kind === :tps ? :Xnull : :Xfixed) : Symbol(:Z, name), w))
            for (name, w) in _spline_blocks(kind, k)]
end

# Per-block emission roles + the shared sd-vector name: `(roles, sd)`
# with roles `(block, coefficient-vector name, sd index or nothing)`.
# The sd index follows the pen-block position (tps: the one pen block →
# 1; t2: rr/rn/nr → 1/2/3, SB's `vector[3]` order).
function _spline_block_roles(id::Symbol, kind::Symbol, k)
    roles = Tuple{Symbol,Symbol,Union{Nothing,Int}}[]
    sdpos = 0
    for (name, _) in _spline_blocks(kind, k)
        if name === :fixed
            push!(roles, (name, Symbol("b_$(id)_fixed"), nothing))
        else
            sdpos += 1
            tag = kind === :tps ? :raw : Symbol("$(name)_raw")
            push!(roles, (name, Symbol("b_$(id)_$tag"), sdpos))
        end
    end
    return roles, Symbol("sd_$id")
end

function _spline_vector_specs(id::Symbol, kind::Symbol, k)
    specs = Tuple{Symbol,Symbol,NamedTuple,SupportOverride,Int}[]
    widths = Dict(first(b) => last(b) for b in _spline_blocks(kind, k))
    roles, sd = _spline_block_roles(id, kind, k)
    for (name, coef, sdidx) in roles
        if sdidx === nothing
            push!(specs, (coef, :flat, NamedTuple(), nothing, widths[name]))
        else
            push!(specs, (coef, :normal, (arg1=0, arg2=1), nothing,
                widths[name]))
        end
    end
    nsd = kind === :tps ? 1 : 3
    push!(specs, (sd, :normal, (arg1=0, arg2=1), :positive, nsd))
    return specs
end

# HSGP sampled names, derived purely from the basis id (SB `_sb_hsgp`
# vocabulary, basis-qualified): the standardized coefficients
# (`beta_raw_<id>`, M-vector), the length scale(s) (`rho_<id>` iso
# scalar, `rho_<id>_1..d` aniso scalars — d scalars, never a
# floors-vector), and the marginal scale (`sigma_<id>` scalar).
# Single source for surface claims, name tables, layout, and the
# generator (Stage B).
function _hsgp_names(hb::HSGPBasis)
    id = hb.id
    beta = Symbol("beta_raw_", id)
    sigma = Symbol("sigma_", id)
    rhos = hb.iso ? [Symbol("rho_", id)] :
        [Symbol("rho_", id, :_, j) for j in 1:length(hb.axes)]
    return (beta = beta, rhos = rhos, sigma = sigma)
end

"""Basis-function count `M = prod(K)` for an [`HSGPBasis`](@ref)."""
_hsgp_n_basis(hb::HSGPBasis) = prod(hb.K)

# Flat sampled-name list for name tables + claims (beta, rhos, sigma).
function _hsgp_all_names(hb::HSGPBasis)
    n = _hsgp_names(hb)
    return Symbol[n.beta, n.sigma, n.rhos...]
end

# Every model-scope name a kernel plate introduces (cell names become flat
# model-scope locals at codegen): the result, slice params, and cell-local
# assignment names. Single source for the global name-table gate.
function _kernel_all_names(kp::KernelPlate)
    names = Symbol[kp.result]
    for (_, p, _) in kp.slices
        push!(names, p)
    end
    for (nm, _) in kp.assignments
        push!(names, nm)
    end
    return names
end

"""
    _hsgp_floors(K, fits, iso) -> Vector{Float64}

Length-scale floors from bound fits (SB `_brm_hsgp_rho_lower[_s]`
verbatim): `(4L/pi)*sqrt(log(100)/(K^2-1))` per axis (`K=1` →
`0.0`, unbounded); iso takes the max. Pure function of fits — no
storage on the IR (layout calls it on bound fits in Stage B).
"""
function _hsgp_floors(K::Vector{Int}, fits::Vector{Tuple{Float64,Float64}},
        iso::Bool)
    per = Float64[
        k == 1 ? 0.0 :
            (4 * L / pi) * sqrt(log(100.0) / (k * k - 1))
        for (k, (_, L)) in zip(K, fits)]
    return iso ? [maximum(per)] : per
end

"""Slice-1 term-name vocabulary (emitter-side admission keys; `:ranef_gather`,
`:spline_summand`, `:monotonic`, and `:monotonic_summand` joined with their
slices)."""
const TERM_NAMES = Dict{Symbol,TermKind}(
    :intercept => InterceptTerm,
    :continuous => ContinuousTerm,
    :factor => FactorTerm,
    :offset => OffsetTerm,
    :ranef_gather => RanefGatherTerm,
    :spline_summand => SplineSummandTerm,
    :hsgp_summand => HSGPSummandTerm,
    :monotonic => MonotonicTerm,
    :monotonic_summand => MonotonicSummandTerm,
)

"""Allowlisted assignment functions (slice 1: scalar ops + whole-column
reductions; elementwise math over columns deferred with vector assignments)."""
const ASSIGNMENT_FNS = (
    :+, :-, :*, :/, :^,
    :log, :log10, :log1p, :exp, :expm1, :sqrt, :abs,
    :sum, :mean, :std, :var, :minimum, :maximum, :length,
)

"""Vector-returning whole-column functions (exact-GP slice): admitted in
derived columns and predictor locations only; always vector-shaped."""
const VECTOR_FNS = (:gp_exp_quad_cov, :gp_chol_latent)

"""Whole-column reductions (their single argument must be a bare column)."""
const REDUCTION_FNS = (:sum, :mean, :std, :var, :minimum, :maximum, :length)

"""Dotted operators admitted in derived-column expressions (elementwise)."""
const ELEMENTWISE_OPS =
    (:.+, :.-, :.*, :./, :.^, :.%, :.==, :.!=, :.<, :.>, :.<=, :.>=)

"""Dotted comparisons (the `ifelse` condition vocabulary)."""
const ELEMENTWISE_COMPARISONS = (:.==, :.!=, :.<, :.>, :.<=, :.>=)

"""Dotted math functions admitted in derived columns (`f.(x)` parses to
`Expr(:., f, ...)`; single-argument, mirroring the scalar math subset)."""
const ELEMENTWISE_FNS = (:log, :log10, :log1p, :exp, :expm1, :sqrt, :abs)

"""Families the thin layer can lower (ext handshake predicate)."""
admitted_families() = (GaussianFam, BernoulliLogitFam, PoissonLogFam,
    BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam,
    BernoulliProbitFam, BernoulliCloglogFam, BinomialProbitFam,
    BinomialCloglogFam, BetaLogitFam, CategoricalLogitFam,
    OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam)

"""Term kinds the thin layer can lower (ext handshake predicate)."""
admitted_terms() = (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm,
    RanefGatherTerm, SplineSummandTerm, HSGPSummandTerm, MonotonicTerm,
    MonotonicSummandTerm)

"""Assignment functions the thin layer can lower (ext handshake predicate):
scalar/reduction vocabulary plus vector-returning whole-column functions."""
admitted_functions() = (ASSIGNMENT_FNS..., VECTOR_FNS...)

"""Elementwise vocabulary the thin layer can lower in derived columns:
`(dotted operators, dotted math functions)` (ext handshake predicate)."""
admitted_elementwise() = (ELEMENTWISE_OPS, ELEMENTWISE_FNS)

"""
    supports_term(name) -> Bool

Stretch handshake: does the thin layer lower the named term kind yet?
The emitter fails closed on `false` (`:zscale`/`:center`/`:standardize`/
`:protect` until declared here).
"""
supports_term(name::Symbol) = haskey(TERM_NAMES, name)

"""In-graph coefficient-block name for a predictor (`mu` → `mu_coef`)."""
block_name(predictor::Symbol) = Symbol(string(predictor) * "_coef")

_fail(label, msg) = throw(ContractValidationError("[$label] $msg"))

"""
    validate_plan(plan) -> nothing

Defensive thin-side validation of a [`StructuralPlan`](@ref) (R8): every
unknown kind, dangling reference, arity/shape violation, and ordering
failure throws [`ContractValidationError`](@ref) — loud, never silent.
Returns `nothing` on success.
"""
function validate_plan(plan::StructuralPlan)
    validate_structure(plan)
    isbound(plan) && validate_data(plan)
    return nothing
end

"""Structure checks: everything provable without data. Runs on bound and
unbound plans alike (the macro lowering + emitter-AST path call this)."""
function validate_structure(plan::StructuralPlan)
    _validate_name_tables(plan)
    _validate_scans(plan)
    _validate_assignments_structure(plan)
    _validate_vector_structure(plan)
    _validate_parameters(plan)
    _validate_plate_parameters(plan)
    _validate_vector_parameters(plan)
    _validate_topo_order(plan)
    _validate_predictors(plan)
    _validate_levelmaps(plan)
    _validate_priors(plan)
    _validate_kernels(plan)
    _validate_responses(plan)
    _validate_ranef_buckets(plan)
    _validate_splines(plan)
    _validate_hsgp(plan)
    return nothing
end

"""Data checks: columns, eltypes, levels, bounds. Requires a bound plan;
[`bind_data`](@ref) runs this after attaching columns."""
function validate_data(plan::StructuralPlan)
    isbound(plan) || throw(ContractValidationError(
        "[bind] validate_data requires a bound plan (bind_data first)"))
    _validate_columns(plan)
    _validate_column_names(plan)
    _validate_assignments_data(plan)
    _validate_vector_data(plan)
    _validate_predictor_columns(plan)
    _validate_levelmaps_data(plan)
    _validate_response_data(plan)
    _validate_plate_parameters_data(plan)
    _validate_ranef_buckets_data(plan)
    _validate_splines_data(plan)
    _validate_hsgp_data(plan)
    _validate_kernels_data(plan)
    return nothing
end

# Bucket keys, kinds, margins, slices, and gather linkage: everything
# provable without data. Slice ranges partition 1:K contiguously in body
# order (SB's static per-target ranges); every slice is gathered at least
# once (a dangling bucket samples dead parameters).
# K=1 bucket suffix (SB `r_<target>_<suffix>` vocabulary): the group for
# plain buckets, `id_group` for `|ID|` buckets. K=1 kinds are always
# plain, so their suffix is always the group — but the rule is total.
_ranef_bucket_suffix(b::RanefBucket) =
    b.id === nothing ? string(b.group) : string(b.id) * "_" * string(b.group)

# K=1 sampled names, derived purely from the bucket key + kind (SB
# `ranef_intercept`/`ranef_slope` vocabulary): the scalar scale
# (`log_scale_<s>` / `tau_<s>`) and the G-vector (`xi_<s>`). Single
# source for surface claims, name tables, layout, and the generator.
# Correlated sampled names, derived purely from the bucket key (SB
# `ranef_correlated_draws` vocabulary, bucket-qualified): the LKJ
# Cholesky factor (`L_<s>`, KxK), the marginal-scale vector
# (`tau_<s>`, K — a VECTOR, unlike the Stage-B scalar), and the
# standardized draws (`z_flat_<s>`, K*G column-major). Single source
# for surface claims, name tables, layout, and the generator. K=1 ID
# buckets own the same three names (`L` packs zero coords).
function _ranef_corr_names(b::RanefBucket)
    b.kind === :correlated ||
        _fail(b.label, "bucket kind $(b.kind) owns no correlated " *
              "sampled names (K=1 plain geometry has its own names)")
    s = _ranef_bucket_suffix(b)
    return (Symbol("L_", s), Symbol("tau_", s), Symbol("z_flat_", s))
end

function _ranef_k1_names(b::RanefBucket)
    (b.kind === :intercept1 || b.kind === :slope1) ||
        _fail(b.label, "bucket kind $(b.kind) owns no K=1 sampled names " *
              "(correlated buckets own L/tau/z names instead)")
    s = _ranef_bucket_suffix(b)
    scale = b.kind === :intercept1 ? Symbol("log_scale_", s) :
        Symbol("tau_", s)
    return (scale, Symbol("xi_", s))
end

function _validate_ranef_buckets(plan::StructuralPlan)
    buckets = plan.ranef_buckets
    keys = [(b.id, b.group) for b in buckets]
    length(unique(keys)) == length(keys) ||
        _fail(:plan, "duplicate ranef bucket keys (one bucket per (id, group))")
    labels = [b.label for b in buckets]
    length(unique(labels)) == length(labels) ||
        _fail(:plan, "duplicate ranef bucket labels (rename the colliding |ID|)")
    prednames = Set{Symbol}(p.name for p in plan.predictors)
    for b in buckets
        _validate_bucket_shape(b, prednames)
    end
    # Gather linkage, jointly over predictors + buckets.
    gathered = Set{Tuple{Symbol,Union{Nothing,Symbol},Symbol}}()
    for pred in plan.predictors
        for t in pred.terms
            t.kind === RanefGatherTerm || continue
            o = t.options
            key = (o.bucket_id, o.bucket_group)
            key in keys ||
                _fail(t.label, "gather in predictor $(pred.name) references " *
                      "unknown bucket $key (no such ranef_bucket)")
            (pred.name, o.bucket_id, o.bucket_group) in gathered &&
                _fail(t.label, "duplicate gather of bucket $key in " *
                      "predictor $(pred.name) (one gather per bucket per predictor)")
            push!(gathered, (pred.name, o.bucket_id, o.bucket_group))
            bi = findfirst(k -> k == key, keys)
            any(s -> s[1] === pred.name, buckets[bi].slices) ||
                _fail(t.label, "bucket $key carries no slice for " *
                      "predictor $(pred.name)")
        end
    end
    for b in buckets
        for (target, _) in b.slices
            (target, b.id, b.group) in gathered ||
                _fail(b.label, "bucket $((b.id, b.group)) slice for " *
                      "predictor $target is never gathered (dangling slice " *
                      "samples dead parameters — gather it or drop it)")
        end
    end
    return nothing
end

function _validate_bucket_shape(b::RanefBucket, prednames::Set{Symbol})
    b.kind in (:intercept1, :slope1, :correlated) ||
        _fail(b.label, "bucket kind must be :intercept1, :slope1, or " *
              ":correlated, got $(repr(b.kind))")
    K = length(b.margins)
    K >= 1 || _fail(b.label, "bucket has zero margins")
    for m in b.margins
        _validate_margin(m, b.label)
    end
    targets = [s[1] for s in b.slices]
    length(unique(targets)) == length(targets) ||
        _fail(b.label, "bucket lists a target twice (one margin list per target)")
    for t in targets
        t in prednames ||
            _fail(b.label, "bucket slice for unknown predictor $t")
    end
    # Ranges partition 1:K contiguously in body order (and no range is
    # empty — an empty slice would emit a vacuous gather).
    lo = 1
    for (t, r) in b.slices
        first(r) <= last(r) ||
            _fail(b.label, "slice for $t is empty (each slice carries " *
                  "at least one margin)")
        first(r) == lo ||
            _fail(b.label, "slice for $t starts at $(first(r)), want $lo " *
                  "(slices partition 1:$K contiguously)")
        lo = last(r) + 1
    end
    lo - 1 == K ||
        _fail(b.label, "slices cover $(lo - 1) margins but the bucket " *
              "carries $K")
    for (t, r) in b.slices
        for i in r
            b.margins[i].predictor === t ||
                _fail(b.label, "margin $i belongs to " *
                      "$(b.margins[i].predictor) but slice $r belongs to $t")
        end
    end
    # Kind dispatch (mirror of SB's plain/ID split): plain single-`1` is
    # :intercept1, plain single slope is :slope1, everything else —
    # including ALL ID buckets — is :correlated.
    want = if b.id !== nothing
        :correlated
    elseif K == 1 && _is_ones_margin(first(b.margins))
        :intercept1
    elseif K == 1
        :slope1
    else
        :correlated
    end
    b.kind === want ||
        _fail(b.label, "bucket kind $(b.kind) mismatches its margins " *
              "(want $want)")
    if want === :correlated
        isfinite(b.lkj_eta) && b.lkj_eta > 0 ||
            _fail(b.label, "correlated bucket needs a positive LKJ eta, " *
                  "got $(b.lkj_eta)")
    else
        isnan(b.lkj_eta) ||
            _fail(b.label, "K=1 plain bucket takes no LKJ eta (no " *
                  "correlation to parameterize), got $(b.lkj_eta)")
    end
    return nothing
end

_is_ones_margin(m::RanefMargin) =
    m.z.kind === :ones && m.coefficient === :Intercept

function _validate_margin(m::RanefMargin, label::Symbol)
    z = m.z
    z.kind in (:ones, :column, :dummy) ||
        _fail(label, "margin $(m.predictor).$(m.coefficient): Z recipe " *
              "kind must be :ones, :column, or :dummy, got $(repr(z.kind))")
    if z.kind === :ones
        z.column === :none && z.level === nothing ||
            _fail(label, "margin $(m.predictor).$(m.coefficient): :ones " *
                  "recipe carries no column/level")
        m.coefficient === :Intercept ||
            _fail(label, "margin $(m.predictor).$(m.coefficient): :ones " *
                  "recipe addresses :Intercept")
    elseif z.kind === :column
        z.level === nothing ||
            _fail(label, "margin $(m.predictor).$(m.coefficient): :column " *
                  "recipe carries no level")
        m.coefficient === z.column ||
            _fail(label, "margin $(m.predictor).$(m.coefficient): :column " *
                  "recipe addresses its column $(z.column)")
    else
        z.level !== nothing ||
            _fail(label, "margin $(m.predictor).$(m.coefficient): :dummy " *
                  "recipe needs a level value")
    end
    return nothing
end

# A literal plate range covers 1:n_obs exactly (the size the latent vector
# packs), mirroring the response-range cover check.
function _validate_plate_parameters_data(plan::StructuralPlan)
    scalarnames = _union_names(plan)
    derivednames = Set{Symbol}(d.name for d in plan.derived)
    for p in plan.plate_parameters
        # A per-cell prior arg that is neither a scalar name nor a derived
        # column must be a bound raw data column (validated now that data is
        # attached); anything else is a genuine unknown name.
        for (k, v) in pairs(p.args)
            v isa Symbol || continue
            (v in scalarnames || v in derivednames || haskey(plan.columns, v)) ||
                _fail(p.label, "arg $k references unknown name $v")
        end
        p.range === nothing && continue
        last(p.range) == plan.n_obs || _fail(p.label,
            "plate range $(p.range) covers $(length(p.range)) cells " *
            "but n_obs is $(plan.n_obs) — ranges cover eachindex exactly")
    end
    return nothing
end

# Grouping columns are raw (level knowledge needs values), Z continuous
# columns mirror the population rule (raw numeric or derived, whose eltype
# is unknown statically), and dummy levels must be members of the column's
# grouping levels (Int value / string exact match).
function _validate_ranef_buckets_data(plan::StructuralPlan)
    for b in plan.ranef_buckets
        haskey(plan.columns, b.group) ||
            _fail(b.label, "grouping column $(b.group) is not bound")
        _is_derived(plan, b.group) &&
            _fail(b.label, "grouping column $(b.group) must be raw data " *
                  "(level knowledge needs bound values)")
        levels = _grouping_levels(plan.columns[b.group])
        length(levels) >= 1 ||
            _fail(b.label, "grouping column $(b.group) has no levels")
        for m in b.margins
            _validate_margin_data(m, b.label, plan)
        end
    end
    return nothing
end

function _validate_margin_data(m::RanefMargin, label::Symbol, plan::StructuralPlan)
    z = m.z
    z.kind === :ones && return nothing
    haskey(plan.columns, z.column) || _is_derived(plan, z.column) ||
        _fail(label, "margin $(m.predictor).$(m.coefficient): Z column " *
              "$(z.column) is not bound")
    if z.kind === :column
        _is_derived(plan, z.column) && return nothing
        eltype(plan.columns[z.column]) <: Real ||
            _fail(label, "margin $(m.predictor).$(m.coefficient): Z column " *
                  "$(z.column) must be numeric (a categorical slope needs " *
                  "explicit `dummy($(z.column), k)` recipes)")
    else
        _is_derived(plan, z.column) &&
            _fail(label, "margin $(m.predictor).$(m.coefficient): :dummy " *
                  "needs a raw column (level membership needs bound values)")
        z.level in _grouping_levels(plan.columns[z.column]) ||
            _fail(label, "margin $(m.predictor).$(m.coefficient): dummy " *
                  "level $(repr(z.level)) is not a level of $(z.column)")
    end
    return nothing
end

function _validate_columns(plan::StructuralPlan)
    plan.n_obs > 0 || _fail(:plan, "n_obs must be positive, got $(plan.n_obs)")
    managed = Set{Symbol}()
    for kp in plan.kernel_plates
        union!(managed, _kernel_managed_columns(kp))
    end
    for (name, col) in plan.columns
        # Kernel-managed columns (slices + scalar expansions) carry two
        # lengths by design — they validate under kernel rules, not here.
        name in managed || length(col) == plan.n_obs ||
            _fail(name, "column length $(length(col)) ≠ n_obs $(plan.n_obs)")
        !any(ismissing, col) ||
            _fail(name, "column contains missing (slice 1 has no missingness machinery)")
    end
    return nothing
end

function _validate_splines(plan::StructuralPlan)
    ids = [sb.id for sb in plan.spline_bases]
    length(unique(ids)) == length(ids) ||
        _fail(:plan, "duplicate spline basis ids")
    labels = [sb.label for sb in plan.spline_bases]
    length(unique(labels)) == length(labels) ||
        _fail(:plan, "duplicate spline basis labels")
    for sb in plan.spline_bases
        sb.kind === :tps || sb.kind === :t2 ||
            _fail(:plan, "spline :$(sb.id): kind must be :tps or :t2, " *
                  "got $(repr(sb.kind))")
        if sb.kind === :tps
            sb.k isa Int ||
                _fail(:plan, "tps spline :$(sb.id): k must be an Int, " *
                      "got $(repr(sb.k))")
            sb.k > 2 ||
                _fail(:plan, "tps spline :$(sb.id): k must exceed 2, " *
                      "got $(sb.k)")
            length(sb.axes) == 1 ||
                _fail(:plan, "tps spline :$(sb.id): takes exactly one " *
                      "axis column, got $(sb.axes)")
        else
            sb.k isa Tuple{Int,Int} ||
                _fail(:plan, "t2 spline :$(sb.id): k must be an " *
                      "(Int, Int) tuple, got $(repr(sb.k))")
            all(k -> k > 2, sb.k) ||
                _fail(:plan, "t2 spline :$(sb.id): k entries must " *
                      "exceed 2, got $(repr(sb.k))")
            length(sb.axes) == 2 ||
                _fail(:plan, "t2 spline :$(sb.id): takes exactly two " *
                      "axis columns, got $(sb.axes)")
        end
        want = _spline_blocks(sb.kind, sb.k)
        got = [(b.name, b.width) for b in sb.blocks]
        got == want || _fail(:plan,
            "spline :$(sb.id): blocks are determined by (kind, k) alone " *
            "— expected $want, got $got")
        wantvec = _spline_vector_specs(sb.id, sb.kind, sb.k)
        gotvec = [v.name for v in plan.spline_vectors if v.basis === sb.id]
        sort!(gotvec)
        wantnames = sort!([first(s) for s in wantvec])
        gotvec == wantnames || _fail(:plan,
            "spline :$(sb.id): spline-vectors must be exactly " *
            "$wantnames, got $gotvec")
        byname = Dict{Symbol,SplineVector}(v.name => v
            for v in plan.spline_vectors if v.basis === sb.id)
        for (vname, vfamily, vargs, vsupport, vwidth) in wantvec
            v = byname[vname]
            (v.family === vfamily && v.args == vargs &&
             v.support_override === vsupport && v.width == vwidth) ||
                _fail(:plan, "spline :$(sb.id): vector :$vname must be " *
                      "$vfamily$(vargs) with support $(repr(vsupport)) " *
                      "and width $vwidth — the prior structure is part " *
                      "of the contract, not emitter's choice")
        end
        # Materialized <id>_<block>_<j> names are computable pre-bind
        # (widths are static), so the sampler-scope clash check runs here
        # rather than at bind. Names are unique across bases by
        # construction (right-parse _<j> then the fixed block tag is
        # unambiguous, and ids are unique), so no pairwise check.
        wantcols = reduce(vcat, (last(b) for b in
            _spline_basis_columns(sb.id, sb.kind, sb.k)); init=Symbol[])
        union = _union_names(plan)
        clash = filter(c -> c in union, wantcols)
        isempty(clash) || _fail(:plan,
            "spline :$(sb.id): materialized basis columns $clash collide " *
            "with parameter/assignment names — rename the spline id")
    end
    idset = Set{Symbol}(ids)
    for v in plan.spline_vectors
        v.basis in idset ||
            _fail(:plan, "spline vector :$(v.name) addresses unknown " *
                  "basis :$(v.basis)")
    end
    # Basis linkage: every summand names an existing basis; every basis
    # feeds exactly one summand (SB: a smooth has one target; a dangling
    # basis would sample dead parameters, a double use double-counts).
    uses = Dict{Symbol,Int}(id => 0 for id in ids)
    for pred in plan.predictors, t in pred.terms
        t.kind === SplineSummandTerm || continue
        sid = t.options.spline_id
        haskey(uses, sid) ||
            _fail(t.label, "spline summand addresses unknown basis :$sid")
        uses[sid] += 1
    end
    for (id, n) in uses
        n == 1 || _fail(:plan,
            "spline :$id is used by $n summands — exactly one " *
            "(one target per smooth)")
    end
    return nothing
end

function _validate_splines_data(plan::StructuralPlan)
    for sb in plan.spline_bases
        for c in sb.axes
            haskey(plan.columns, c) ||
                _fail(sb.label, "spline :$(sb.id): axis column $c is " *
                      "not bound")
            _is_derived(plan, c) &&
                _fail(sb.label, "spline :$(sb.id): axis column $c must " *
                      "be raw data (the bind-time fit needs bound values)")
            eltype(plan.columns[c]) <: Real ||
                _fail(sb.label, "spline :$(sb.id): axis column $c must " *
                      "be numeric, got $(eltype(plan.columns[c]))")
        end
        for b in sb.blocks, c in b.columns
            haskey(plan.columns, c) ||
                _fail(sb.label, "spline :$(sb.id): materialized basis " *
                      "column $c (block :$(b.name)) is not bound")
        end
    end
    return nothing
end

function _validate_hsgp(plan::StructuralPlan)
    ids = [hb.id for hb in plan.hsgp_bases]
    length(unique(ids)) == length(ids) ||
        _fail(:plan, "duplicate hsgp basis ids")
    labels = [hb.label for hb in plan.hsgp_bases]
    length(unique(labels)) == length(labels) ||
        _fail(:plan, "duplicate hsgp basis labels")
    for hb in plan.hsgp_bases
        d = length(hb.axes)
        d >= 1 ||
            _fail(:plan, "hsgp :$(hb.id): takes at least one axis column")
        length(hb.axes) == length(unique(hb.axes)) ||
            _fail(:plan, "hsgp :$(hb.id): duplicate axis columns $(hb.axes)")
        length(hb.K) == d ||
            _fail(:plan, "hsgp :$(hb.id): K has $(length(hb.K)) entries " *
                  "for $d axes (one mode count per axis)")
        all(k -> k isa Int && k >= 1, hb.K) ||
            _fail(:plan, "hsgp :$(hb.id): K must be positive integers, " *
                  "got $(hb.K)")
        length(hb.c) == d ||
            _fail(:plan, "hsgp :$(hb.id): c has $(length(hb.c)) entries " *
                  "for $d axes (one boundary factor per axis)")
        all(c -> c isa Real && isfinite(Float64(c)) && Float64(c) > 1,
            hb.c) ||
            _fail(:plan, "hsgp :$(hb.id): c must be finite and exceed 1 " *
                  "(L = c*max|x-mu| must cover the data), got $(hb.c)")
        hb.iso isa Bool ||
            _fail(:plan, "hsgp :$(hb.id): iso must be Bool, " *
                  "got $(repr(hb.iso))")
        # Fits are bind products (empty pre-bind); a hand-built bound plan
        # carries one finite (mu, L) per axis with L > 0.
        isempty(hb.fits) || length(hb.fits) == d ||
            _fail(:plan, "hsgp :$(hb.id): fits has $(length(hb.fits)) " *
                  "entries for $d axes (bind fills one (mu, L) per axis)")
        for (mu, L) in hb.fits
            isfinite(mu) && isfinite(L) && L > 0 ||
                _fail(:plan, "hsgp :$(hb.id): fit (mu, L) must be finite " *
                      "with L > 0, got ($mu, $L)")
        end
    end
    # Basis linkage: every summand names an existing basis; every basis
    # feeds exactly one summand (one target per basis — a dangling basis
    # would sample dead parameters, a double use double-counts).
    uses = Dict{Symbol,Int}(id => 0 for id in ids)
    for pred in plan.predictors, t in pred.terms
        t.kind === HSGPSummandTerm || continue
        haskey(t.options, :hsgp_id) ||
            _fail(t.label, "hsgp summand carries no hsgp_id option")
        sid = t.options.hsgp_id
        haskey(uses, sid) ||
            _fail(t.label, "hsgp summand addresses unknown basis :$sid")
        uses[sid] += 1
    end
    for (id, n) in uses
        n == 1 || _fail(:plan,
            "hsgp :$id is used by $n summands — exactly one " *
            "(one target per basis)")
    end
    return nothing
end

function _validate_hsgp_data(plan::StructuralPlan)
    for hb in plan.hsgp_bases
        for c in hb.axes
            haskey(plan.columns, c) ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c is " *
                      "not bound")
            _is_derived(plan, c) &&
                _fail(hb.label, "hsgp :$(hb.id): axis column $c must " *
                      "be raw data (the bind-time fit needs bound values)")
            eltype(plan.columns[c]) <: Real ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c must " *
                      "be numeric, got $(eltype(plan.columns[c]))")
        end
        length(hb.fits) == length(hb.axes) ||
            _fail(hb.label, "hsgp :$(hb.id): fits not filled at bind " *
                  "(one (mu, L) per axis)")
    end
    return nothing
end

# Panel-kernel (KernelPlate) structure: everything provable without data.
# Panel v1: at most one plate per model; a plate carries the ONLY
# likelihood (no top-level responses alongside — BRM routes kernel models
# away from the GLM flow); grouping is ABSENT (implicit 1:n subjects —
# structural, no sentinel, pinned here + tests).
function _validate_kernels(plan::StructuralPlan)
    plates = plan.kernel_plates
    length(plates) <= 1 ||
        _fail(:plan, "panel v1 admits at most one kernel plate per model " *
              "(got $(length(plates)))")
    isempty(plates) && return nothing
    kp = only(plates)
    isempty(plan.responses) ||
        _fail(kp.label, "a kernel plate carries the only likelihood " *
              "(panel v1: no top-level responses alongside `$(kp.result)`)")
    # Name hygiene + collisions (result, slice params, cell locals) live in
    # the global `_validate_name_tables` gate via `_kernel_all_names`.
    _check_name_hygiene(kp.label)
    if kp.subjects isa Int
        kp.subjects > 0 ||
            _fail(kp.label, "subject count must be a positive integer, " *
                  "got $(kp.subjects)")
    end
    if kp.timepoints isa Int
        kp.timepoints > 0 ||
            _fail(kp.label, "timepoint count must be a positive integer, " *
                  "got $(kp.timepoints)")
    end
    slices = kp.slices
    isempty(slices) &&
        _fail(kp.label, "kernel plate `$(kp.result)` takes at least one slice")
    cols = [c for (c, _, _) in slices]
    length(unique(cols)) == length(cols) ||
        _fail(kp.label, "duplicate slice columns $(cols)")
    params = [p for (_, p, _) in slices]
    for (_, _, kind) in slices
        kind in (:vector, :scalar, :unknown) ||
            _fail(kp.label, "slice kind must be :vector, :scalar, or " *
                  ":unknown (pre-bind), got $(repr(kind))")
    end
    # Cell assignments in order: each RHS sees slice params + earlier
    # locals + model-scope scalars only (cross-cell refs fail closed).
    known = union(Set{Symbol}(params), _union_names(plan))
    cell_locals = Set{Symbol}()
    for (nm, ex) in kp.assignments
        _collect_kernel_cell_refs!(Symbol[], ex, kp, known)
        push!(known, nm)
        push!(cell_locals, nm)
    end
    # The single in-cell observation (Gaussian-identity v1).
    obs = kp.obs
    obs.response in params ||
        _fail(kp.label, "kernel obs response `$(obs.response)` is not a " *
              "slice param (responses enter the cell as slices)")
    obs.family === GaussianFam ||
        _fail(kp.label, "panel v1 admits a Gaussian in-cell observation " *
              "only, got $(obs.family)")
    for (nm, ref) in ((:location, obs.location), (:scale, obs.scale))
        if ref isa Number && !(ref isa Bool)
            (isfinite(ref) && (nm === :location || ref > 0)) ||
                _fail(kp.label, "kernel obs $nm literal must be finite" *
                      (nm === :scale ? " positive" : "") * ", got $ref")
        elseif ref isa Symbol
            ref in known ||
                _fail(kp.label, "kernel obs $nm `$ref` is neither a cell " *
                      "name nor a model-level scalar (cross-cell refs " *
                      "fail closed)")
        else
            _fail(kp.label, "kernel obs $nm must be a cell/model name or " *
                  "a numeric literal, got $(repr(ref))")
        end
    end
    kp.collected in union(Set{Symbol}(params), cell_locals) ||
        _fail(kp.label, "collected result `$(kp.collected)` is not a cell " *
              "name (slice param or cell-local assignment)")
    return nothing
end

# Kernel cell vocabulary: the derived-column elementwise walker with a
# cell name environment (slice params + earlier locals + model scalars).
# Dotted ops/math, undotted arithmetic (canonicalized at bind),
# `ifelse`, bare names and numeric literals; reductions, whole-column
# functions, indexing, loops, branches, nested observations, and unknown
# names fail closed.
function _collect_kernel_cell_refs!(refs, ex, kp::KernelPlate, known::Set{Symbol})
    label = kp.label
    ex isa Number && return nothing
    ex isa LineNumberNode && return nothing
    if ex isa Symbol
        ex in known ||
            _fail(label, "cell expression references unknown name `$ex` " *
                  "(slices + earlier cell locals + model-level scalars only)")
        push!(refs, ex)
        return nothing
    end
    ex isa Expr ||
        _fail(label, "unsupported literal $(repr(ex)) (numeric literals only)")
    head = ex.head
    if head === :call
        fn = ex.args[1]
        if fn isa Symbol && fn in ELEMENTWISE_OPS
            for arg in ex.args[2:end]
                _collect_kernel_cell_refs!(refs, arg, kp, known)
            end
            return nothing
        end
        if fn isa Symbol && fn in REDUCTION_FNS
            # A reduction over a series is cross-timepoint (per-subject
            # aggregation) — not flat-lowerable in panel v1 (P3 needs real
            # per-subject loop codegen); over a cell scalar it is
            # degenerate. Either way it does not lower in a cell.
            return _fail(label, "reduction `$fn` does not lower in a cell " *
                                "(series reductions are cross-timepoint — P3)")
        end
        if fn isa Symbol && fn in VECTOR_FNS
            return _fail(label, "whole-column `$fn` does not lower in a " *
                                "cell (whole-model constructs only)")
        end
        if fn isa Symbol && fn in ASSIGNMENT_FNS
            # Undotted arithmetic is admitted syntactically here (the
            # emitter passes scalar-context user code verbatim — Ex1's
            # `ke = CLi / Vci`); bind canonicalizes with slice-kind
            # provenance (dotify over flat vectors, fail closed over 2+
            # genuinely-vector operands). Unary +/- stay as-is (valid on
            # vectors and scalars alike).
            for arg in ex.args[2:end]
                _collect_kernel_cell_refs!(refs, arg, kp, known)
            end
            return nothing
        end
        fn isa Symbol && startswith(string(fn), ".") &&
            _fail(label, "dotted operator $fn is not in the panel-v1 " *
                         "cell vocabulary")
        return _fail(label, "call `$fn` is not in the panel-v1 cell " *
                            "vocabulary (elementwise + reductions only)")
    end
    head === :. && return _collect_kernel_cell_dot!(refs, ex, kp, known)
    head === :ref &&
        _fail(label, "indexing does not lower in a cell (flat vectors " *
                     "keep full length — no `[...]`)")
    head === :(=) && _fail(label, "nested assignment does not lower in a cell")
    head === :kw &&
        _fail(label, "keyword arguments do not lower in a cell")
    return _fail(label, "unsupported expression head $head in a cell " *
                        "(elementwise expressions only)")
end

function _collect_kernel_cell_dot!(refs, ex, kp::KernelPlate, known::Set{Symbol})
    label = kp.label
    length(ex.args) == 2 && ex.args[1] isa Symbol && ex.args[2] isa Expr &&
        ex.args[2].head === :tuple ||
        return _fail(label, "field access does not lower in a cell " *
                            "(dotted calls take `f.(...)`)")
    f = ex.args[1]
    args = ex.args[2].args
    if f === :ifelse
        length(args) == 3 ||
            _fail(label, "`ifelse` takes `ifelse.(condition, x, y)`")
        _collect_kernel_cell_condition!(refs, args[1], kp, known)
        for arg in args[2:end]
            _collect_kernel_cell_refs!(refs, arg, kp, known)
        end
        return nothing
    end
    f in ELEMENTWISE_FNS ||
        _fail(label, "dotted call `$f.(...)` is not in the panel-v1 cell " *
                     "vocabulary")
    for arg in args
        _collect_kernel_cell_refs!(refs, arg, kp, known)
    end
    return nothing
end

# Bind-time cell canonicalization (slice-kind provenance): undotted
# arithmetic over flat vectors takes dotted-canonical form (the surface
# `_canonical_expr` precedent — the emitter passes scalar-context user
# code verbatim, e.g. Ex1's `ke = CLi / Vci`); pure-scalar
# (global/literal) combos stay as-is; undotted operators over 2+
# genuinely-vector operands fail closed naming the dotted fix (vector
# `*`/`/` is meaningless in the cell). Shapes are CELL shapes
# (:scalar for scalar slices + scalar-shaped locals, :vector for vector
# slices + vector-shaped locals); derivation (slice/cell vs pure scalar)
# drives dotify. Idempotent: bind applies it for early errors, the
# generator re-applies it for hand-bound plans.
const _KERNEL_UNDOTTED_ARITHMETIC = (:+, :-, :*, :/, :^)

function _kernel_cell_shapes(kp::KernelPlate)
    shapes = Dict{Symbol,Symbol}()
    for (_, p, kind) in kp.slices
        kind in (:vector, :scalar) ||
            _fail(kp.label, "slice `$p` kind unresolved " *
                  "(bind_data resolves :unknown from lengths)")
        shapes[p] = kind
    end
    return shapes
end

function _canonicalize_kernel_assignments(kp::KernelPlate)
    shapes = _kernel_cell_shapes(kp)
    out = Pair{Symbol,Any}[]
    for (nm, ex) in kp.assignments
        canon, shape = _canonicalize_kernel_cell(ex, kp, shapes)
        shapes[nm] = shape
        push!(out, nm => canon)
    end
    return out
end

function _canonicalize_kernel_cell(ex, kp::KernelPlate, shapes::Dict{Symbol,Symbol})
    label = kp.label
    ex isa Number && return (ex, :scalar)
    ex isa LineNumberNode && return (ex, :scalar)
    ex isa Symbol && return (ex, get(shapes, ex, :scalar))
    ex isa Expr || _fail(label, "unsupported literal $(repr(ex)) (numeric literals only)")
    head = ex.head
    if head === :call
        isempty(ex.args) && _fail(label, "operator needs operands")
        fn = ex.args[1]
        if fn isa Symbol && fn in ELEMENTWISE_OPS
            cargs = Any[]
            shape = :scalar
            for arg in ex.args[2:end]
                carg, ashape = _canonicalize_kernel_cell(arg, kp, shapes)
                push!(cargs, carg)
                ashape === :vector && (shape = :vector)
            end
            return (Expr(:call, fn, cargs...), shape)
        end
        if fn isa Symbol && fn in _KERNEL_UNDOTTED_ARITHMETIC
            operands = ex.args[2:end]
            isempty(operands) && _fail(label, "operator `$fn` needs operands")
            # Unary +/- stay as-is (valid on vectors and scalars alike).
            if length(operands) == 1
                carg, shape = _canonicalize_kernel_cell(only(operands), kp, shapes)
                return (Expr(:call, fn, carg), shape)
            end
            cargs = Any[]
            nvec = 0
            derived = false
            for arg in operands
                carg, ashape = _canonicalize_kernel_cell(arg, kp, shapes)
                push!(cargs, carg)
                ashape === :vector && (nvec += 1)
                derived |= _kernel_operand_derived(arg, shapes)
            end
            nvec >= 2 && _fail(label, "undotted `$fn` over vector series " *
                                      "does not lower — write the dotted " *
                                      "form (`$(Symbol(:., fn))`)")
            shape = nvec >= 1 ? :vector : :scalar
            derived || return (Expr(:call, fn, cargs...), shape)
            return (Expr(:call, Symbol(:., fn), cargs...), shape)
        end
        if fn isa Symbol && fn in ELEMENTWISE_FNS
            cargs = Any[]
            shape = :scalar
            derived = false
            for arg in ex.args[2:end]
                carg, ashape = _canonicalize_kernel_cell(arg, kp, shapes)
                push!(cargs, carg)
                ashape === :vector && (shape = :vector)
                derived |= _kernel_operand_derived(arg, shapes)
            end
            derived || return (Expr(:call, fn, cargs...), shape)
            return (Expr(:., fn, Expr(:tuple, cargs...)), shape)
        end
        if fn isa Symbol && fn in REDUCTION_FNS
            return _fail(label, "reduction `$fn` does not lower in a cell " *
                                "(series reductions are cross-timepoint — P3)")
        end
        return _fail(label, "call `$fn` is not in the panel-v1 cell vocabulary")
    end
    if head === :.
        length(ex.args) == 2 && ex.args[1] isa Symbol && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple ||
            _fail(label, "field access does not lower in a cell " *
                         "(dotted calls take `f.(...)`)")
        f = ex.args[1]
        (f === :ifelse || f in ELEMENTWISE_FNS) ||
            _fail(label, "dotted call `$f.(...)` is not in the panel-v1 " *
                         "cell vocabulary")
        f === :ifelse &&
            length(ex.args[2].args) != 3 &&
            _fail(label, "`ifelse` takes `ifelse.(condition, x, y)`")
        cargs = Any[]
        shape = :scalar
        for arg in ex.args[2].args
            carg, ashape = _canonicalize_kernel_cell(arg, kp, shapes)
            push!(cargs, carg)
            ashape === :vector && (shape = :vector)
        end
        return (Expr(:., f, Expr(:tuple, cargs...)), shape)
    end
    return _fail(label, "unsupported expression head $head in a cell " *
                        "(elementwise expressions only)")
end

# An operand is slice/cell-derived (a flat vector at codegen) iff it
# mentions a slice param or cell local; pure global/literal subtrees stay
# scalar and their undotted operators are kept as-is.
function _kernel_operand_derived(ex, shapes::Dict{Symbol,Symbol})
    ex isa Number && return false
    ex isa LineNumberNode && return false
    ex isa Symbol && return haskey(shapes, ex)
    ex isa Expr || return false
    return any(a -> _kernel_operand_derived(a, shapes), ex.args)
end

function _collect_kernel_cell_condition!(refs, ex, kp::KernelPlate, known::Set{Symbol})
    label = kp.label
    if ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] in ELEMENTWISE_COMPARISONS
        return _collect_kernel_cell_refs!(refs, ex, kp, known)
    end
    ex isa Symbol || return _fail(label, "`ifelse` condition must be a " *
                                          "comparison (`x .< y`) or a bare " *
                                          "cell/model boolean name")
    ex in known ||
        _fail(label, "`ifelse` condition `$ex` is not a cell or " *
                     "model-level name")
    push!(refs, ex)
    return nothing
end

# Kernel bind checks: resolved dims, total kinds, flat-T-blocked lengths,
# subjects coverage. Runs on bound plans (bind resolves Symbol dims via
# the `dims` map first; hand-bound plans carry Ints directly).
function _validate_kernels_data(plan::StructuralPlan)
    isempty(plan.kernel_plates) && return nothing
    kp = only(plan.kernel_plates)
    kp.subjects isa Int ||
        _fail(kp.label, "subjects dims key `$(kp.subjects)` unresolved " *
              "(bind_data with dims first)")
    n_sub = kp.subjects
    T = kp.timepoints
    T isa Symbol &&
        _fail(kp.label, "timepoints dims key `$T` unresolved " *
              "(bind_data with dims first)")
    flat = _kernel_flat_length(n_sub, T)
    plan.n_obs == flat ||
        _fail(kp.label, "n_obs $(plan.n_obs) ≠ kernel flat length $flat " *
              "(n_sub=$n_sub$((T === nothing ? "" : ", T=$T")))")
    if T === nothing
        all(s -> s[3] === :scalar, kp.slices) ||
            _fail(kp.label, "a vector slice needs T (bind the " *
                  "`kernel_T_<result>` dims key)")
    elseif T > 1
        any(s -> s[3] === :vector, kp.slices) ||
            _fail(kp.label, "T=$T bound but no vector slice uses it " *
                  "(scalar models omit the timepoints dims key)")
    end
    # T == 1 with all-scalar slices is the unobservable-kinds case
    # (recovered scalar by scalar-first inference) — admitted.
    for (col, param, kind) in kp.slices
        kind in (:vector, :scalar) ||
            _fail(kp.label, "slice `$param` kind unresolved " *
                  "(bind_data resolves :unknown from lengths)")
        haskey(plan.columns, col) ||
            _fail(kp.label, "slice column `$col` is not bound")
        colv = plan.columns[col]
        eltype(colv) <: Real ||
            _fail(kp.label, "slice column `$col` must be numeric, " *
                  "got $(eltype(colv))")
        all(isfinite, colv) ||
            _fail(kp.label, "slice column `$col` must be finite")
        want = kind === :vector ? flat : n_sub
        length(colv) == want ||
            _fail(kp.label, "slice `$param` ($kind) column `$col` has " *
                  "length $(length(colv)), want $want " *
                  (kind === :vector ? "(n_sub*T flat T-blocked)" :
                   "(n_sub per-subject)"))
        if kind === :scalar && T !== nothing
            # Scalar slices expand to flat T-blocks at bind; a hand-bound
            # plan must carry the same expansion (verified, not trusted).
            exp = _kexp_name(kp.result, col)
            haskey(plan.columns, exp) ||
                _fail(kp.label, "scalar slice `$param` expansion `$exp` " *
                      "missing (bind_data materializes flat T-blocks)")
            expv = plan.columns[exp]
            length(expv) == flat ||
                _fail(kp.label, "expansion `$exp` has length " *
                      "$(length(expv)), want flat $flat")
            expv == repeat(Vector{Float64}(colv); inner = T) ||
                _fail(kp.label, "expansion `$exp` is not the flat " *
                      "T-block repeat of `$col`")
        end
    end
    return nothing
end

function _validate_name_tables(plan::StructuralPlan)
    pnames = [p.name for p in plan.predictors]
    length(unique(pnames)) == length(pnames) ||
        _fail(:plan, "duplicate predictor names")
    params = [p.name for p in plan.parameters]
    assigns = [a.name for a in plan.assignments]
    deriveds = [d.name for d in plan.derived]
    plates = [p.name for p in plan.plate_parameters]
    scanstates = [s.state for s in plan.scans]
    vectors = [p.name for p in plan.vector_parameters]
    svec = [v.name for v in plan.spline_vectors]
    k1 = Symbol[nm for b in plan.ranef_buckets
        if b.kind === :intercept1 || b.kind === :slope1
        for nm in _ranef_k1_names(b)]
    corr = Symbol[nm for b in plan.ranef_buckets
        if b.kind === :correlated
        for nm in _ranef_corr_names(b)]
    hsgp = Symbol[nm for hb in plan.hsgp_bases for nm in _hsgp_all_names(hb)]
    kern = Symbol[nm for kp in plan.kernel_plates for nm in _kernel_all_names(kp)]
    length(unique(params)) == length(params) || _fail(:plan, "duplicate parameter names")
    length(unique(assigns)) == length(assigns) ||
        _fail(:plan, "duplicate assignment names")
    length(unique(deriveds)) == length(deriveds) ||
        _fail(:plan, "duplicate derived-column names")
    length(unique(plates)) == length(plates) ||
        _fail(:plan, "duplicate plate-parameter names")
    length(unique(scanstates)) == length(scanstates) ||
        _fail(:plan, "duplicate scan-state names")
    length(unique(vectors)) == length(vectors) ||
        _fail(:plan, "duplicate vector-parameter names")
    length(unique(svec)) == length(svec) ||
        _fail(:plan, "duplicate spline-vector names")
    length(unique(k1)) == length(k1) ||
        _fail(:plan, "duplicate K=1 ranef names")
    length(unique(corr)) == length(corr) ||
        _fail(:plan, "duplicate correlated ranef names")
    length(unique(hsgp)) == length(hsgp) ||
        _fail(:plan, "duplicate hsgp names")
    length(unique(kern)) == length(kern) ||
        _fail(:plan, "duplicate kernel-plate names")
    for (l, r, what) in ((params, assigns, "parameters and assignments"),
        (params, deriveds, "parameters and derived columns"),
        (assigns, deriveds, "assignments and derived columns"),
        (plates, params, "plate parameters and parameters"),
        (plates, assigns, "plate parameters and assignments"),
        (plates, deriveds, "plate parameters and derived columns"),
        (params, scanstates, "parameters and scan states"),
        (assigns, scanstates, "assignments and scan states"),
        (deriveds, scanstates, "derived columns and scan states"),
        (plates, scanstates, "plate parameters and scan states"),
        (vectors, params, "vector parameters and parameters"),
        (vectors, assigns, "vector parameters and assignments"),
        (vectors, deriveds, "vector parameters and derived columns"),
        (vectors, plates, "vector parameters and plate parameters"),
        (vectors, scanstates, "vector parameters and scan states"),
        (svec, params, "spline vectors and parameters"),
        (svec, assigns, "spline vectors and assignments"),
        (svec, deriveds, "spline vectors and derived columns"),
        (svec, plates, "spline vectors and plate parameters"),
        (svec, scanstates, "spline vectors and scan states"),
        (vectors, svec, "vector parameters and spline vectors"),
        (k1, params, "K=1 ranef names and parameters"),
        (k1, assigns, "K=1 ranef names and assignments"),
        (k1, deriveds, "K=1 ranef names and derived columns"),
        (k1, plates, "K=1 ranef names and plate parameters"),
        (k1, scanstates, "K=1 ranef names and scan states"),
        (k1, vectors, "K=1 ranef names and vector parameters"),
        (k1, svec, "K=1 ranef names and spline vectors"),
        (corr, params, "correlated ranef names and parameters"),
        (corr, assigns, "correlated ranef names and assignments"),
        (corr, deriveds, "correlated ranef names and derived columns"),
        (corr, plates, "correlated ranef names and plate parameters"),
        (corr, scanstates, "correlated ranef names and scan states"),
        (corr, vectors, "correlated ranef names and vector parameters"),
        (corr, svec, "correlated ranef names and spline vectors"),
        (corr, k1, "correlated ranef names and K=1 ranef names"),
        (hsgp, params, "hsgp names and parameters"),
        (hsgp, assigns, "hsgp names and assignments"),
        (hsgp, deriveds, "hsgp names and derived columns"),
        (hsgp, plates, "hsgp names and plate parameters"),
        (hsgp, scanstates, "hsgp names and scan states"),
        (hsgp, vectors, "hsgp names and vector parameters"),
        (hsgp, svec, "hsgp names and spline vectors"),
        (hsgp, k1, "hsgp names and K=1 ranef names"),
        (hsgp, corr, "hsgp names and correlated ranef names"),
        (kern, params, "kernel-plate names and parameters"),
        (kern, assigns, "kernel-plate names and assignments"),
        (kern, deriveds, "kernel-plate names and derived columns"),
        (kern, plates, "kernel-plate names and plate parameters"),
        (kern, scanstates, "kernel-plate names and scan states"),
        (kern, vectors, "kernel-plate names and vector parameters"),
        (kern, svec, "kernel-plate names and spline vectors"),
        (kern, k1, "kernel-plate names and K=1 ranef names"),
        (kern, corr, "kernel-plate names and correlated ranef names"),
        (kern, hsgp, "kernel-plate names and hsgp names"))
        overlap = intersect(l, r)
        isempty(overlap) ||
            _fail(:plan, "names in both $what: $(join(overlap, ", "))")
    end
    allnames = union(params, assigns, deriveds, plates, scanstates, vectors,
        svec, k1, corr, hsgp, kern)
    for pn in pnames
        pn in allnames && _fail(
            :plan,
            "predictor $pn collides with a parameter/assignment/derived/plate/scan/vector/spline/ranef/kernel name",
        )
        block_name(pn) in allnames && _fail(
            :plan,
            "parameter/assignment/derived/plate/scan/vector/spline/ranef/kernel $(block_name(pn)) collides with predictor $pn block name",
        )
    end
    for n in Iterators.flatten((pnames, params, assigns, deriveds, plates, scanstates, vectors, svec, k1, corr, hsgp, kern))
        _check_name_hygiene(n)
    end
    return nothing
end

# Structural invariants of each sequential recurrence. The surface parser
# (`parse_scan_block`) already enforces these; this is defense-in-depth for a
# hand-built plan and the invariants the layout/emitter will rely on.
function _validate_scans(plan::StructuralPlan)
    for s in plan.scans
        s.lo == length(s.setup) + 1 || _fail(s.label,
            "scan loop start $(s.lo) must be one past the $(length(s.setup)) " *
            "seed fill(s)")
        for (k, f) in enumerate(s.setup)
            f.index == k || _fail(s.label,
                "scan seed fills must be contiguous 1..$(length(s.setup)); " *
                "entry $k has index $(f.index)")
        end
        s.maxlag >= 1 || _fail(s.label,
            "a scan must read a backward lag of its carried state (maxlag ≥ 1)")
        length(s.setup) >= s.maxlag || _fail(s.label,
            "scan maxlag $(s.maxlag) exceeds the $(length(s.setup)) seeded value(s)")
        any(st -> st.indexed, s.step) || _fail(s.label,
            "scan recurrence never writes its carried state $(s.state)")
        (s.hi isa Int || s.hi isa Symbol) || _fail(s.label,
            "scan loop bound must be a literal Int or a data length Symbol, " *
            "got $(repr(s.hi))")
    end
    return nothing
end

function _check_name_hygiene(n::Symbol)
    startswith(string(n), "_ppl_") && _fail(
        :plan,
        "name $n uses the reserved _ppl_ prefix (transform intermediates)",
    )
    n in RESERVED_NODES && _fail(
        :plan,
        "name $n collides with a canonical node (prior/likelihood/log_jacobian/posterior/unconstrained)",
    )
    return nothing
end

function _validate_column_names(plan::StructuralPlan)
    col_overlap = filter(
        n -> haskey(plan.columns, n),
        union([p.name for p in plan.parameters],
            [a.name for a in plan.assignments],
            [d.name for d in plan.derived],
            [p.name for p in plan.plate_parameters]),
    )
    isempty(col_overlap) || _fail(
        :plan,
        "parameter/assignment/derived/plate names collide with raw columns: $(join(col_overlap, ", "))",
    )
    for n in keys(plan.columns)
        _check_name_hygiene(n)
    end
    return nothing
end

function _validate_assignments_structure(plan::StructuralPlan)
    for a in plan.assignments
        _collect_assignment_refs!(Symbol[], a.expr, plan, a.label, false)
    end
    return nothing
end

function _validate_assignments_data(plan::StructuralPlan)
    for a in plan.assignments
        refs = Symbol[]
        _collect_assignment_refs!(refs, a.expr, plan, a.label, true)
        for r in refs
            r in _all_names(plan) ||
                _fail(a.label, "assignment references unknown name $r")
        end
    end
    return nothing
end

function _validate_vector_structure(plan::StructuralPlan)
    for d in plan.derived
        _validate_vector_alias(d, plan, false)
        d.expr isa Symbol && continue
        refs = Symbol[]
        _collect_vector_refs!(refs, d.expr, plan, d.label, false)
        _is_vector_valued(d.expr, plan) || _fail(d.label,
            "derived column is scalar-valued — write it as a scalar " *
            "assignment instead")
    end
    return nothing
end

function _validate_vector_data(plan::StructuralPlan)
    for d in plan.derived
        _validate_vector_alias(d, plan, true)
        d.expr isa Symbol && continue
        refs = Symbol[]
        _collect_vector_refs!(refs, d.expr, plan, d.label, true)
        for r in refs
            r in _all_names(plan) ||
                _fail(d.label, "derived column references unknown name $r")
        end
        _is_vector_valued(d.expr, plan) || _fail(d.label,
            "derived column is scalar-valued — write it as a scalar " *
            "assignment instead")
    end
    return nothing
end

# A bare-Symbol derived expression aliases a column (raw or derived) —
# never a scalar name (length mismatch).
function _validate_vector_alias(d::VectorAssignmentSpec, plan, bound::Bool)
    d.expr isa Symbol || return nothing
    target = d.expr
    _is_derived(plan, target) && return nothing
    target in _union_names(plan) &&
        _fail(d.label, "alias target $target is a scalar name — aliases " *
                       "take columns only")
    bound || return nothing
    haskey(plan.columns, target) ||
        _fail(d.label, "derived column aliases unknown column $target")
    return nothing
end

_union_names(plan::StructuralPlan) =
    union([p.name for p in plan.parameters], [a.name for a in plan.assignments])

"""Full name table: scalar names plus derived columns plus per-cell latent
(plate) parameter VECTORS (dependency edges and bind-time reference checks
admit all; sampled-arg positions stay scalar-only via [`_union_names`](@ref),
so a scalar arg can never reference a latent vector)."""
_all_names(plan::StructuralPlan) =
    union(_union_names(plan), [d.name for d in plan.derived],
        [p.name for p in plan.plate_parameters])

_is_derived(plan::StructuralPlan, name::Symbol) =
    any(d -> d.name === name, plan.derived)

_is_plate_param(plan::StructuralPlan, name::Symbol) =
    any(p -> p.name === name, plan.plate_parameters)

# With bound=false (structure), bare Symbols are opaque refs and column
# checks are skipped — classification needs columns. With bound=true, bare
# columns fail (row-varying outside a reduction) and reduction args must be
# bound columns or derived names. Derived names are known in both states,
# so derived-outside-a-reduction fails at structure already.
function _collect_assignment_refs!(refs, ex, plan, label, bound::Bool)
    ex isa Number && return nothing
    ex isa LineNumberNode && return nothing
    if ex isa Symbol
        _is_derived(plan, ex) && _fail(
            label,
            "$ex is a derived column: reference it inside reductions " *
            "(`mean($ex)`) or elementwise in a derived definition " *
            "(`log.($ex)`-style); scalar positions take scalars",
        )
        bound && haskey(plan.columns, ex) && _fail(
            label,
            "row-varying column $ex outside a reduction (derive it " *
            "elementwise in a `name = ...` definition, e.g. `log.($ex)`)",
        )
        push!(refs, ex)
        return nothing
    end
    ex isa Expr || _fail(label, "unsupported literal $(repr(ex)) (numeric literals only)")
    head = ex.head
    if head === :call
        fn = ex.args[1]
        fn isa Symbol && startswith(string(fn), ".") && _fail(
            label,
            "dotted subexpression `$(repr(ex))` is row-varying — stage it " *
            "as its own derived `name = ...` first",
        )
        fn isa Symbol && fn in ASSIGNMENT_FNS ||
            _fail(label, "function $fn not in the slice-1 assignment allowlist")
        if fn in REDUCTION_FNS
            length(ex.args) == 2 || _fail(
                label,
                "reduction $fn takes exactly one bare column or derived name",
            )
            arg = ex.args[2]
            arg isa Symbol || _fail(
                label,
                "reduction $fn argument must be a bare column or derived " *
                "name (stage nested transforms as their own `name = ...` first)",
            )
            (!bound || haskey(plan.columns, arg) || _is_derived(plan, arg)) ||
                _fail(
                    label,
                    "reduction $fn argument $arg is not a bound column or " *
                    "derived name",
                )
            _is_derived(plan, arg) && push!(refs, arg)
            return nothing
        end
        for arg in ex.args[2:end]
            _collect_assignment_refs!(refs, arg, plan, label, bound)
        end
        return nothing
    end
    head === :. && _fail(
        label,
        "broadcast expressions are row-varying — stage them as their own " *
        "derived `name = ...` first",
    )
    return _fail(label, "unsupported expression head $head (pure calls only)")
end

# Elementwise walker for derived columns (contract v3). Vector mode admits
# dotted operators, dotted math, `ifelse`, reductions over bare names, and
# bare names/literals; scalar subterms (undotted allowlist calls) delegate
# to the scalar collector. With bound=true, direct column references in
# math positions must be numeric and bare-Symbol `ifelse` conditions must be
# Bool columns.
function _collect_vector_refs!(refs, ex, plan, label, bound::Bool)
    ex isa Number && return nothing
    ex isa LineNumberNode && return nothing
    if ex isa Symbol
        # Per-cell latent (plate) parameters are vectors, so a derived column
        # may transform one (`theta = mu .+ tau .* z`) — the non-centered shape.
        if _is_derived(plan, ex) || _is_plate_param(plan, ex) ||
                ex in _union_names(plan)
            push!(refs, ex)
            return nothing
        end
        bound || return nothing
        haskey(plan.columns, ex) && return nothing
        return _fail(label, "derived column references unknown name $ex")
    end
    ex isa Expr || _fail(label, "unsupported literal $(repr(ex)) (numeric literals only)")
    head = ex.head
    if head === :call
        fn = ex.args[1]
        if fn isa Symbol && fn in ELEMENTWISE_OPS
            _check_numeric_position!(ex.args[2:end], plan, label, bound)
            for arg in ex.args[2:end]
                _collect_vector_refs!(refs, arg, plan, label, bound)
            end
            return nothing
        end
        if fn isa Symbol && fn in REDUCTION_FNS
            _collect_vector_reduction!(refs, ex, plan, label, bound)
            return nothing
        end
        if fn isa Symbol && fn in VECTOR_FNS
            for arg in ex.args[2:end]
                _collect_vector_refs!(refs, arg, plan, label, bound)
            end
            return nothing
        end
        if fn isa Symbol && fn in ASSIGNMENT_FNS
            for arg in ex.args[2:end]
                _collect_assignment_refs!(refs, arg, plan, label, bound)
            end
            return nothing
        end
        fn isa Symbol && startswith(string(fn), ".") &&
            _fail(label, "dotted operator $fn is not in the slice-1 " *
                         "elementwise vocabulary")
        return _fail(label, "call `$fn` is not in the slice-1 elementwise " *
                            "vocabulary — arbitrary Julia functions are " *
                            "planned (no-@deffun-ceremony direction) but need " *
                            "IR/contract growth")
    end
    head === :. && return _collect_vector_dot!(refs, ex, plan, label, bound)
    head === :ref && return _fail(label, "indexing changes length — " *
                                          "derived columns keep n_obs " *
                                          "(no `[...]` in vector expressions)")
    head === :(=) && return _fail(label, "nested assignment does not lower")
    return _fail(label, "unsupported expression head $head in a vector expression")
end

function _collect_vector_reduction!(refs, ex, plan, label, bound::Bool)
    fn = ex.args[1]
    length(ex.args) == 2 || _fail(
        label,
        "reduction $fn takes exactly one bare column or derived name",
    )
    arg = ex.args[2]
    arg isa Symbol || _fail(
        label,
        "reduction $fn argument must be a bare column or derived name " *
        "(stage nested transforms as their own `name = ...` first)",
    )
    if _is_derived(plan, arg)
        push!(refs, arg)
        return nothing
    end
    (!bound || haskey(plan.columns, arg)) || _fail(
        label,
        "reduction $fn argument $arg is not a bound column or derived name",
    )
    return nothing
end

function _collect_vector_dot!(refs, ex, plan, label, bound::Bool)
    length(ex.args) == 2 && ex.args[1] isa Symbol && ex.args[2] isa Expr &&
        ex.args[2].head === :tuple ||
        return _fail(label, "field access does not lower in vector " *
                            "expressions (dotted calls take `f.(...)`)")
    f = ex.args[1]
    args = ex.args[2].args
    if f === :ifelse
        length(args) == 3 ||
            _fail(label, "`ifelse` takes `ifelse.(condition, x, y)`")
        _collect_vector_condition!(refs, args[1], plan, label, bound)
        for arg in args[2:end]
            _collect_vector_refs!(refs, arg, plan, label, bound)
        end
        return nothing
    end
    f in ELEMENTWISE_FNS || return _fail(
        label,
        "`$f.` is not in the slice-1 elementwise vocabulary — arbitrary " *
        "Julia functions are planned (no-@deffun-ceremony direction) but " *
        "need IR/contract growth",
    )
    length(args) == 1 ||
        _fail(label, "`$f.` takes exactly one argument")
    _check_numeric_position!(args, plan, label, bound)
    for arg in args
        _collect_vector_refs!(refs, arg, plan, label, bound)
    end
    return nothing
end

function _collect_vector_condition!(refs, ex, plan, label, bound::Bool)
    if ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] in ELEMENTWISE_COMPARISONS
        return _collect_vector_refs!(refs, ex, plan, label, bound)
    end
    ex isa Symbol || return _fail(label, "`ifelse` condition must be a " *
                                          "comparison (`x .< y`) or a bare " *
                                          "Bool column / scalar name")
    _is_derived(plan, ex) && return _fail(
        label,
        "`ifelse` condition over the derived column $ex needs static Bool " *
        "knowledge — inline the comparison",
    )
    if bound && haskey(plan.columns, ex)
        eltype(plan.columns[ex]) === Bool ||
            _fail(label, "`ifelse` condition column $ex must be Bool")
        return nothing
    end
    push!(refs, ex)
    return nothing
end

# Direct column references in math positions must be numeric (derived and
# nested references are validated where they resolve).
function _check_numeric_position!(args, plan, label, bound::Bool)
    bound || return nothing
    for arg in args
        arg isa Symbol && haskey(plan.columns, arg) &&
            !(eltype(plan.columns[arg]) <: Real) &&
            _fail(label, "column $arg is not numeric (elementwise math " *
                         "needs numeric columns)")
    end
    return nothing
end

# A derived column must be row-varying by value: some column-or-derived
# reference in a length-propagating position. Dotted forms propagate,
# reductions and scalar calls collapse; unbound unknown symbols count as
# (possibly-column) evidence and resolve at bind.
function _is_vector_valued(ex, plan::StructuralPlan)
    ex isa Symbol && return !(ex in _union_names(plan))
    ex isa Number && return false
    ex isa LineNumberNode && return false
    ex isa Expr || return false
    head = ex.head
    if head === :call
        isempty(ex.args) && return false
        fn = ex.args[1]
        fn in REDUCTION_FNS && return false
        fn isa Symbol && fn in VECTOR_FNS && return true
        fn isa Symbol && (fn in ELEMENTWISE_OPS || fn in ASSIGNMENT_FNS) &&
            return any(a -> _is_vector_valued(a, plan), ex.args[2:end])
        return false
    end
    if head === :.
        length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple ||
            return false
        return any(a -> _is_vector_valued(a, plan), ex.args[2].args)
    end
    return false
end

function _validate_parameters(plan::StructuralPlan)
    names = _union_names(plan)
    for p in plan.parameters
        haskey(SAMPLED_ARITY, p.family) || _fail(
            p.label,
            "sampled family $(p.family) not in the slice-1 set " *
            "($(join(sort!(collect(keys(SAMPLED_ARITY))), ", ")))",
        )
        arity = SAMPLED_ARITY[p.family]
        expected_keys = ntuple(i -> Symbol(:arg, i), arity)
        Tuple(keys(p.args)) == expected_keys || _fail(
            p.label,
            "family $(p.family) takes positional keys $expected_keys, " *
            "got $(Tuple(keys(p.args)))",
        )
        for (k, v) in pairs(p.args)
            v isa Number && continue
            v isa Symbol || _fail(
                p.label,
                "arg $k must be a literal or a parameter/assignment name",
            )
            v in names ||
                _fail(p.label, "arg $k references unknown name $v")
        end
        _validate_support_override(p.label, p.family, p.support_override, p.args)
    end
    return nothing
end

# Shared support-override rule for scalar and per-cell latent parameters:
# `:positive` half-truncates a real-support (normal/cauchy) family, and the
# +log(2) renormalization is exact only for a literal zero location.
# `(:interval, lo, hi)` is a two-sided finite truncation with finite lo < hi;
# the family must be real-support (a truncated Normal), and the density carries
# the exact -log(cdf(hi) - cdf(lo)) renormalization at any location.
function _validate_support_override(label, family::Symbol,
        ov::SupportOverride, args::NamedTuple)
    ov === nothing && return nothing
    if ov isa Tuple
        ov[1] === :interval || _fail(label,
            "tuple support override must be (:interval, lo, hi), got $ov")
        family === :normal || _fail(label,
            "an :interval override is a truncated Normal in slice 1 " *
            "(`truncated(Normal(mu, s), lo, hi)`); got $family")
        lo, hi = ov[2], ov[3]
        (isfinite(lo) && isfinite(hi)) || _fail(label,
            ":interval bounds must be finite (a one-sided or half truncation " *
            "uses :positive); got ($lo, $hi)")
        lo < hi || _fail(label,
            ":interval lower bound must be < upper bound; got ($lo, $hi)")
        return nothing
    end
    ov === :positive || _fail(label, "support override must be :positive, got $ov")
    (family === :normal || family === :cauchy) || _fail(label,
        ":positive override only applies to normal/cauchy " *
        "(half-Normal/half-Cauchy); got $family")
    loc = first(values(args))
    loc isa Real && loc == 0 || _fail(label,
        ":positive override requires literal zero location " *
        "(+log(2) is exact only by symmetry at 0); got $(repr(loc))")
    return nothing
end

# Per-cell latent (plate) parameters: same family/arity/support grammar as
# scalar SampledParameters. Prior args are either SHARED across cells (a
# literal or a scalar parameter/assignment name) or PER-CELL (a derived column,
# giving a varying prior mean/scale — the varying-intercept shape); never
# another latent vector. `range` is `nothing` (whole-column, size n_obs) or a
# literal UnitRange (validated to cover 1:n_obs at bind, mirroring responses).
function _validate_plate_parameters(plan::StructuralPlan)
    for p in plan.plate_parameters
        haskey(SAMPLED_ARITY, p.family) || _fail(p.label,
            "plate family $(p.family) not in the slice-1 set " *
            "($(join(sort!(collect(keys(SAMPLED_ARITY))), ", ")))")
        p.family === :flat && _fail(p.label,
            "a per-cell `flat()` latent has no proper prior to draw a cell " *
            "from — give the plate parameter a proper family")
        arity = SAMPLED_ARITY[p.family]
        expected_keys = ntuple(i -> Symbol(:arg, i), arity)
        Tuple(keys(p.args)) == expected_keys || _fail(p.label,
            "family $(p.family) takes positional keys $expected_keys, " *
            "got $(Tuple(keys(p.args)))")
        for (k, v) in pairs(p.args)
            v isa Number && continue
            v isa Symbol || _fail(p.label,
                "arg $k must be a literal, a scalar parameter/assignment name, " *
                "or a per-cell column (data or derived)")
            _is_plate_param(plan, v) && _fail(p.label,
                "arg $k is the latent vector $v — a per-cell latent's prior args " *
                "are shared scalars or per-cell columns, never another latent")
            # scalar (shared) and derived (per-cell) resolve structurally; a raw
            # data-column arg (per-cell) resolves at bind, like any data ref.
        end
        _validate_support_override(p.label, p.family, p.support_override, p.args)
        if p.range !== nothing
            r = p.range
            (first(r) == 1 && last(r) >= 1) || _fail(p.label,
                "plate range must start at 1 (`1:N`), got $(first(r)):$(last(r))")
        end
    end
    return nothing
end

# Leveled-family predicates (response/trials/link rules are per group).
_is_ordered_family(f) = f === OrderedLogisticFam || f === OrdinalFam
_is_simplex_family(f) = f === MultinomialFam || f === CategoricalFam
_is_leveled_family(f) = f === CategoricalLogitFam || _is_ordered_family(f) ||
    _is_simplex_family(f)

"""Vector-parameter families and their positional arg keys."""
const VECTOR_ARITY = Dict{Symbol,Tuple{Vararg{Symbol}}}(
    :ordered_normal => (:arg1, :arg2),
    :vector_normal => (:arg1, :arg2),
    :simplex_dirichlet => (:arg1,),
)

# Constrained vector (cutpoint/threshold/simplex) parameters: family/arity
# plus literal-only args (a threshold Normal takes literal location/scale; a
# Dirichlet takes a literal concentration vector — hierarchical args fail
# closed). Sizes resolve at bind (`nothing` = infer from the linked leveled
# response, or from the concentration length for a monotonic-linked
# simplex); an explicit size is bounds-checked here and linked-checked in
# `_validate_responses`. Each vector parameter serves exactly one response
# or one monotonic term (SB allocates per response and per `mo` term;
# sharing fails closed).
function _validate_vector_parameters(plan::StructuralPlan)
    for p in plan.vector_parameters
        haskey(VECTOR_ARITY, p.family) || _fail(p.label,
            "vector family $(p.family) unknown (admitted: ordered_normal, " *
            "vector_normal, simplex_dirichlet)")
        expected = VECTOR_ARITY[p.family]
        Tuple(keys(p.args)) == expected || _fail(p.label,
            "family $(p.family) takes positional keys $expected, got " *
            "$(Tuple(keys(p.args)))")
        if p.family === :simplex_dirichlet
            alpha = p.args.arg1
            alpha isa AbstractVector || _fail(p.label,
                "simplex_dirichlet takes a literal concentration vector " *
                "`Dirichlet(alpha)` or symmetric `Dirichlet(K, a)` resolved " *
                "to a vector; got $(repr(alpha))")
            all(x -> x isa Real && isfinite(x) && x > 0, alpha) || _fail(
                p.label,
                "Dirichlet concentrations must be finite and strictly " *
                "positive, got $(repr(alpha))")
            p.size === nothing || p.size == length(alpha) || _fail(p.label,
                "simplex size $(p.size) disagrees with its concentration " *
                "length $(length(alpha))")
            p.size === nothing || p.size >= 1 || _fail(p.label,
                "simplex size must be ≥ 1, got $(p.size)")
        else
            mu, s = p.args.arg1, p.args.arg2
            (mu isa Real && isfinite(mu)) || _fail(p.label,
                "threshold location must be a finite literal, got $(repr(mu))")
            (s isa Real && isfinite(s) && s > 0) || _fail(p.label,
                "threshold scale must be a finite positive literal, got " *
                "$(repr(s))")
            p.size === nothing || p.size >= 0 || _fail(p.label,
                "threshold size must be ≥ 0, got $(p.size)")
        end
    end
    # Linkage: each vector parameter is referenced by exactly one response
    # (as `thresholds` for ordered families, as `threshold_coefs` for
    # per-threshold Ordinal, or as the simplex `predictor`) or by exactly
    # one monotonic term (as its `increments` simplex) — never both, never
    # shared.
    refs = Dict{Symbol,Vector{Symbol}}(
        p.name => Symbol[] for p in plan.vector_parameters)
    for r in plan.responses
        if r.thresholds !== nothing
            haskey(refs, r.thresholds) || _fail(r.label,
                "thresholds parameter $(r.thresholds) is not a vector parameter")
            push!(refs[r.thresholds], r.label)
        end
        if r.threshold_coefs !== nothing
            haskey(refs, r.threshold_coefs) || _fail(r.label,
                "threshold_coefs parameter $(r.threshold_coefs) is not a " *
                "vector parameter")
            push!(refs[r.threshold_coefs], r.label)
        end
        if _is_simplex_family(r.family) && haskey(refs, r.predictor)
            push!(refs[r.predictor], r.label)
        end
    end
    for pred in plan.predictors, t in pred.terms
        (t.kind === MonotonicTerm || t.kind === MonotonicSummandTerm) || continue
        incr = _monotonic_options(t)
        haskey(refs, incr) || _fail(t.label,
            "monotonic increments $incr is not a vector parameter")
        p = only(q for q in plan.vector_parameters if q.name === incr)
        p.family === :simplex_dirichlet || _fail(t.label,
            "monotonic increments $incr must be a " *
            ":simplex_dirichlet vector parameter, got $(p.family)")
        push!(refs[incr], t.label)
    end
    for p in plan.vector_parameters
        got = refs[p.name]
        isempty(got) && _fail(p.label,
            "vector parameter $(p.name) unused by any response or " *
            "monotonic term")
        length(got) == 1 || _fail(p.label,
            "vector parameter $(p.name) shared by " *
            "$(join(got, ", ")) — one vector parameter per response or " *
            "monotonic term")
    end
    return nothing
end

"""Canonical in-graph node names: reserved across every plan namespace."""
const RESERVED_NODES = (:prior, :likelihood, :log_jacobian, :posterior, :unconstrained)

"""
    topological_order(plan) -> Vector{Symbol}

Evaluation order over parameters ∪ assignments ∪ derived columns (Kahn's
algorithm). Loud on cycles (and, on bound plans, unknown references —
unbound plans defer name resolution to bind). Shared by validation and the
generator.
"""
function topological_order(plan::StructuralPlan)
    names = _union_names(plan)
    allnames = _all_names(plan)
    deps = Dict{Symbol,Set{Symbol}}()
    for p in plan.parameters
        refs = Set{Symbol}()
        for v in values(p.args)
            v isa Symbol || continue
            v in names ||
                _fail(p.label, "arg references unknown name $v")
            push!(refs, v)
        end
        deps[p.name] = refs
    end
    # Per-cell latent (plate) parameters: prior args are shared scalars,
    # per-cell derived columns, or raw data columns (never another latent).
    # They constrain in the layout transforms like scalar params; a
    # param/assignment/derived arg must be computed before the prior, so it is
    # a real dependency edge. A raw data-column arg is always available (not a
    # node) and is validated at bind, so it adds no edge here.
    for p in plan.plate_parameters
        refs = Set{Symbol}()
        for v in values(p.args)
            v isa Symbol || continue
            v in allnames && push!(refs, v)
        end
        deps[p.name] = refs
    end
    for a in plan.assignments
        refs = Symbol[]
        _collect_assignment_refs!(refs, a.expr, plan, a.label, isbound(plan))
        if isbound(plan)
            for r in refs
                r in allnames || _fail(a.label, "assignment references unknown name $r")
            end
        end
        deps[a.name] = Set{Symbol}(r for r in refs if r in allnames)
    end
    for d in plan.derived
        refs = Symbol[]
        if d.expr isa Symbol
            _validate_vector_alias(d, plan, isbound(plan))
            _is_derived(plan, d.expr) && push!(refs, d.expr)
        else
            _collect_vector_refs!(refs, d.expr, plan, d.label, isbound(plan))
        end
        if isbound(plan)
            for r in refs
                r in allnames || _fail(d.label, "derived column references unknown name $r")
            end
        end
        deps[d.name] = Set{Symbol}(r for r in refs if r in allnames)
    end
    remaining = Dict{Symbol,Int}(name => length(d) for (name, d) in deps)
    dependents = Dict{Symbol,Vector{Symbol}}(name => Symbol[] for name in keys(deps))
    for (name, ds) in deps, d in ds
        push!(dependents[d], name)
    end
    order = Symbol[]
    ready = [name for (name, n) in remaining if n == 0]
    while !isempty(ready)
        name = pop!(ready)
        push!(order, name)
        for m in dependents[name]
            remaining[m] -= 1
            remaining[m] == 0 && push!(ready, m)
        end
    end
    cyclic = sort!([name for (name, n) in remaining if n > 0])
    isempty(cyclic) ||
        _fail(:plan, "cyclic parameter/assignment dependency: $(join(cyclic, ", "))")
    return order
end

function _validate_topo_order(plan::StructuralPlan)
    topological_order(plan)
    return nothing
end

function _validate_predictors(plan::StructuralPlan)
    for pred in plan.predictors
        isempty(pred.terms) &&
            _fail(pred.label, "predictor $(pred.name) has no terms (empty design)")
        for t in pred.terms
            _validate_term(t, pred, plan)
        end
    end
    return nothing
end

function _validate_term(t::TermSpec, pred::PredictorSpec, plan::StructuralPlan)
    if t.kind === RanefGatherTerm
        _validate_gather_term(t, pred)
        return nothing
    end
    if t.kind === MonotonicTerm || t.kind === MonotonicSummandTerm
        _validate_monotonic_term(t, pred)
        return nothing
    end
    if t.kind === SplineSummandTerm
        _validate_spline_term(t, pred)
        return nothing
    end
    if t.kind === HSGPSummandTerm
        _validate_hsgp_term(t, pred)
        return nothing
    end
    t.options == NamedTuple() ||
        _fail(t.label, "terms take no options (slice 1: factor sizing " *
                       "lives in LevelMap)")
    if t.kind === FactorTerm
        length(t.columns) == 1 ||
            _fail(t.label, "factor term takes exactly one grouping column")
        for c in t.columns
            _is_derived(plan, c) && _fail(t.label,
                "factor over the derived column $c needs pre-evaluation " *
                "level knowledge — factors take raw grouping columns in slice 1")
        end
    elseif t.kind === LatentTerm
        length(t.columns) == 1 ||
            _fail(t.label, "latent term takes exactly one latent-vector name")
        c = only(t.columns)
        (_is_plate_param(plan, c) || _is_derived(plan, c)) || _fail(t.label,
            "latent term over $c needs a per-cell latent parameter " *
            "(a `PlateParameter`) or a derived column that transforms one")
    else
        if t.kind === ContinuousTerm || t.kind === OffsetTerm
            length(t.columns) == 1 ||
                _fail(t.label, "term takes exactly one column")
        elseif t.kind === InterceptTerm
            isempty(t.columns) ||
                _fail(t.label, "intercept term takes no columns")
        end
    end
    return nothing
end

# A gather term names its bucket by (id, group) pair in `options` (never a
# lossy suffix parse) and carries exactly the grouping column; its addressee
# is its own label (self-addressed: gathers take no PopulationPrior).
# Bucket linkage (existence, slices, dangling) is checked jointly in
# `_validate_ranef_buckets`, which sees predictors and buckets together.
function _validate_gather_term(t::TermSpec, pred::PredictorSpec)
    o = t.options
    Tuple(keys(o)) == (:bucket_id, :bucket_group) ||
        _fail(t.label, "ranef gather options must be exactly " *
              "`(bucket_id, bucket_group)`, got $(Tuple(keys(o)))")
    (o.bucket_id === nothing || o.bucket_id isa Symbol) ||
        _fail(t.label, "gather bucket_id must be a Symbol or nothing, " *
              "got $(repr(o.bucket_id))")
    o.bucket_group isa Symbol ||
        _fail(t.label, "gather bucket_group must be a Symbol, " *
              "got $(repr(o.bucket_group))")
    t.columns == ColumnRef[o.bucket_group] ||
        _fail(t.label, "ranef gather columns must be exactly the grouping " *
              "column [$(o.bucket_group)], got $(t.columns)")
    t.addressee === t.label ||
        _fail(t.label, "ranef gather addressee must be its own label " *
              "(self-addressed, no population prior), got $(t.addressee)")
    return nothing
end

# Monotonic options precondition, shared by term validation and the
# vector-parameter linkage (which reads `options.increments` before
# `_validate_predictors` runs, so it must establish the shape itself).
function _monotonic_options(t::TermSpec)
    o = t.options
    Tuple(keys(o)) == (:increments,) ||
        _fail(t.label, "monotonic term options must be exactly " *
              "`(increments,)`, got $(Tuple(keys(o)))")
    o.increments isa Symbol ||
        _fail(t.label, "monotonic increments must name a simplex vector " *
              "parameter, got $(repr(o.increments))")
    return o.increments
end

# A monotonic term (SB `mo(c)` / `mo1(c)`) carries exactly the bound index
# column (integer level codes 1..K, checked at bind) and names its
# increment simplex in `options` (`(increments,)` — a `:simplex_dirichlet`
# vector parameter, linked in `_validate_vector_parameters`). `mo` takes a
# free coefficient (addressee is the column, like a continuous term, so a
# PopulationPrior covers its beta); `mo1` is beta-free and self-addressed
# (no population prior, like the spline/gather summands).
function _validate_monotonic_term(t::TermSpec, pred::PredictorSpec)
    _monotonic_options(t)
    length(t.columns) == 1 ||
        _fail(t.label, "monotonic term takes exactly one index column")
    if t.kind === MonotonicSummandTerm
        t.addressee === t.label ||
            _fail(t.label, "monotonic summand addressee must be its own " *
                  "label (self-addressed, no population prior), got " *
                  "$(t.addressee)")
    end
    return nothing
end

# A spline summand names its basis by id in `options` and carries no
# columns (basis vectors materialize at bind — unnameable pre-bind); its
# addressee is its own label (self-addressed: no population prior).
# Basis linkage (existence, single target, no dangling) is checked jointly
# in `_validate_splines`, which sees predictors and bases together.
function _validate_spline_term(t::TermSpec, pred::PredictorSpec)
    o = t.options
    Tuple(keys(o)) == (:spline_id,) ||
        _fail(t.label, "spline summand options must be exactly " *
              "`(spline_id,)`, got $(Tuple(keys(o)))")
    o.spline_id isa Symbol ||
        _fail(t.label, "spline summand spline_id must be a Symbol, " *
              "got $(repr(o.spline_id))")
    isempty(t.columns) ||
        _fail(t.label, "spline summand carries no columns (basis vectors " *
              "materialize at bind), got $(t.columns)")
    t.addressee === t.label ||
        _fail(t.label, "spline summand addressee must be its own label " *
              "(self-addressed, no population prior), got $(t.addressee)")
    return nothing
end

function _validate_hsgp_term(t::TermSpec, pred::PredictorSpec)
    o = t.options
    Tuple(keys(o)) == (:hsgp_id,) ||
        _fail(t.label, "hsgp summand options must be exactly " *
              "`(hsgp_id,)`, got $(Tuple(keys(o)))")
    o.hsgp_id isa Symbol ||
        _fail(t.label, "hsgp summand hsgp_id must be a Symbol, " *
              "got $(repr(o.hsgp_id))")
    isempty(t.columns) ||
        _fail(t.label, "hsgp summand carries no columns (the basis is " *
              "evaluated in-graph in Stage B), got $(t.columns)")
    t.addressee === t.label ||
        _fail(t.label, "hsgp summand addressee must be its own label " *
              "(self-addressed, no population prior), got $(t.addressee)")
    return nothing
end

function _validate_predictor_columns(plan::StructuralPlan)
    for pred in plan.predictors
        for t in pred.terms
            _validate_term_columns(t, pred, plan)
        end
    end
    return nothing
end

function _validate_term_columns(t::TermSpec, pred::PredictorSpec, plan::StructuralPlan)
    # Latent terms name a per-cell latent VECTOR (a PlateParameter), not a
    # raw/derived data column; structure validation checked its presence.
    t.kind === LatentTerm && return nothing
    for c in t.columns
        haskey(plan.columns, c) || _is_derived(plan, c) ||
            _fail(t.label, "term references missing column $c")
    end
    # Gather terms name the raw grouping column (strings included — the
    # encoder maps levels to codes); presence above is the whole check.
    t.kind === RanefGatherTerm && return nothing
    # FactorTerm: column presence is checked by the loop above; level
    # coverage is a LevelMap concern (_validate_levelmaps_data).
    if t.kind === ContinuousTerm || t.kind === OffsetTerm
        c = only(t.columns)
        # Derived columns are length-n by construction; their eltype is
        # unknown statically (in-graph Julia errors are loud).
        _is_derived(plan, c) && return nothing
        col = plan.columns[c]
        eltype(col) <: Real ||
            _fail(t.label, "column $c must be numeric")
    end
    if t.kind === MonotonicTerm || t.kind === MonotonicSummandTerm
        _validate_monotonic_columns(t, plan)
    end
    return nothing
end

# Monotonic index data (SB `_sb_mo`'s `<c>_idx`): the emitter binds integer
# level codes, so the thin layer takes them as-is — a BOUND raw column of
# integers 1..K, where K − 1 is the linked increments simplex's
# concentration length (unobserved levels are allowed, like unobserved
# factor levels; out-of-range codes fail closed).
function _validate_monotonic_columns(t::TermSpec, plan::StructuralPlan)
    c = only(t.columns)
    _is_derived(plan, c) && _fail(t.label,
        "monotonic index $c must be a bound raw column of level codes " *
        "(the emitter binds integer codes 1..K, SB's `<c>_idx`)")
    col = plan.columns[c]
    (eltype(col) <: Integer && eltype(col) !== Bool) ||
        _fail(t.label, "monotonic index $c must hold integer level codes " *
              "1..K, got eltype $(eltype(col))")
    i = findfirst(p -> p.name === t.options.increments, plan.vector_parameters)
    i === nothing && _fail(t.label,
        "internal: monotonic increments $(t.options.increments) unlinked")
    K = length(plan.vector_parameters[i].args.arg1) + 1
    all(v -> 1 <= v <= K, col) ||
        _fail(t.label, "monotonic index $c holds codes outside 1..$K " *
              "(K − 1 = $(K - 1) is the linked increments simplex size)")
    return nothing
end

"""Grouping levels for a raw factor column: sort-ordered uniques, with
non-`String` strings (e.g. `CategoricalString`) and categorical values
normalized via `string` (zero-dep duck-typing — `String`/`Number`/`Symbol`/
`Bool`/`Char` behavior is unchanged)."""
function _grouping_levels(col::AbstractVector)
    isempty(col) && return []
    v = first(col)
    if (v isa AbstractString && !isa(v, String)) ||
            string(nameof(typeof(v))) == "CategoricalValue"
        return sort!(unique!(string.(col)))
    end
    return sort(unique(col))
end

# One map per factor term, keyed (predictor, column); duplicate keys mean
# two factor terms over one column in one predictor (unidentified sums —
# merge them).
_map_key(m::LevelMap) = (m.predictor, m.column)

function _validate_levelmaps(plan::StructuralPlan)
    keys = _map_key.(plan.levelmaps)
    length(unique(keys)) == length(keys) ||
        _fail(:plan, "duplicate LevelMap keys (one map per factor term)")
    for pred in plan.predictors
        for t in pred.terms
            t.kind === FactorTerm || continue
            col = only(t.columns)
            nm = count(m -> m.predictor === pred.name && m.column === col,
                plan.levelmaps)
            nm == 1 || _fail(t.label,
                "factor term over $col in predictor $(pred.name) has no " *
                "LevelMap (surface: size it with a `c[levels($col)]` prior)")
        end
        if any(t -> t.kind === InterceptTerm, pred.terms)
            for t in pred.terms
                t.kind === FactorTerm || continue
                m = _find_levelmap(plan.levelmaps, pred.name, only(t.columns))
                m !== nothing && m.subset === Colon() && _fail(pred.label,
                    "predictor $(pred.name) is unidentified: intercept + " *
                    "full-cover factor over $(only(t.columns)) (drop the " *
                    "intercept or index a strict subset of levels)")
            end
        end
    end
    for m in plan.levelmaps
        m.source === :levels ||
            _fail(:plan, "LevelMap source must be :levels (only admitted " *
                         "levels function), got $(repr(m.source))")
        _validate_subset_shape(m)
    end
    return nothing
end

function _find_levelmap(maps::Vector{LevelMap}, pred::Symbol, col::Symbol)
    idx = findfirst(m -> m.predictor === pred && m.column === col, maps)
    return idx === nothing ? nothing : maps[idx]
end

function _validate_subset_shape(m::LevelMap)
    s = m.subset
    s === Colon() && return nothing
    s isa UnitRange{Int} ||
        s isa Vector{Int} ||
        (s isa Tuple && length(s) == 2 && s[1] isa Int && s[2] === :end) ||
        return _fail(:plan, "LevelMap subset must be `:`, a UnitRange, " *
                            "a Vector{Int}, or (lo, :end) — got $(repr(s))")
    if s isa UnitRange{Int}
        first(s) >= 1 && first(s) <= last(s) ||
            return _fail(:plan, "LevelMap range $(repr(s)) is empty or " *
                                "starts below 1")
    elseif s isa Vector{Int}
        !isempty(s) && all(>=(1), s) ||
            return _fail(:plan, "LevelMap index list must be non-empty " *
                                "1-based positions — got $(repr(s))")
    else
        s[1] >= 1 ||
            return _fail(:plan, "LevelMap (lo, :end) needs lo ≥ 1 — " *
                                "got $(repr(s))")
    end
    return nothing
end

# Binder evaluation: sort-ordered uniques, then the subset selection
# (bounds-checked against the observed count).
function _eval_levelmaps(levelmaps::Vector{LevelMap},
        columns::Dict{Symbol,AbstractVector})
    out = LevelMap[]
    for m in levelmaps
        haskey(columns, m.column) ||
            _fail(:plan, "LevelMap addresses missing column $(m.column)")
        levels =
            try
                _grouping_levels(columns[m.column])
            catch err
                _fail(:plan, "grouping column $(m.column) levels not " *
                             "orderable ($err)")
            end
        push!(out, LevelMap(m.predictor, m.column,
            _apply_subset(levels, m), m.source, m.subset))
    end
    return out
end

function _apply_subset(levels::Vector, m::LevelMap)
    K = length(levels)
    s = m.subset
    vals = if s === Colon()
        levels
    elseif s isa UnitRange{Int}
        last(s) <= K || _fail(:plan,
            "LevelMap range $(repr(s)) exceeds $K observed levels of " *
            "$(m.column)")
        levels[s]
    elseif s isa Vector{Int}
        all(i -> 1 <= i <= K, s) || _fail(:plan,
            "LevelMap indices $(repr(s)) exceed $K observed levels of " *
            "$(m.column)")
        levels[s]
    else
        lo = s[1]::Int
        lo <= K || _fail(:plan,
            "LevelMap ($(lo), :end) exceeds $K observed levels of " *
            "$(m.column)")
        levels[lo:end]
    end
    length(unique(vals)) == length(vals) ||
        _fail(:plan, "LevelMap selects duplicate positions of " *
                     "$(m.column) — got $(repr(vals))")
    return collect(vals)
end

function _validate_levelmaps_data(plan::StructuralPlan)
    # Rows whose codes fall outside the mapped levels contribute 0 (the
    # subset is explicit on the page — e.g. reference rows under an
    # intercept); unobserved mapped levels are allowed like any
    # zero-variance column. The one failure is unfilled values.
    for m in plan.levelmaps
        isempty(m.values) && _fail(:plan,
            "LevelMap for ($(m.predictor), $(m.column)) has no evaluated " *
            "values (bind_data fills these — hand-built bound plans must too)")
    end
    return nothing
end

function _validate_priors(plan::StructuralPlan)
    seen = Set{Tuple{Symbol,Symbol}}()
    for pr in plan.population_priors
        any(p -> p.name === pr.predictor, plan.predictors) ||
            _fail(:plan, "prior addresses unknown predictor $(pr.predictor)")
        key = (pr.predictor, pr.addressee)
        key in seen &&
            _fail(:plan, "duplicate prior for $key")
        push!(seen, key)
        isfinite(pr.location) && isfinite(pr.scale) && pr.scale > 0 ||
            _fail(:plan, "prior for $key must be Normal(finite, positive)")
    end
    for pred in plan.predictors
        # Offset terms carry no coefficient; latent terms carry the per-cell
        # PlateParameter, whose prior lives on the plate parameter itself;
        # gather terms carry a RanefBucket, spline summands a SplineBasis,
        # hsgp summands an HSGPBasis, and monotonic summands (mo1) an
        # increment simplex, whose geometries are self-priored — none needs
        # a coefficient prior. Monotonic (mo) terms DO take a free
        # coefficient, so they stay in the addressee set.
        addressees = Set{Symbol}(t.addressee for t in pred.terms
            if t.kind !== OffsetTerm && t.kind !== LatentTerm &&
               t.kind !== RanefGatherTerm && t.kind !== SplineSummandTerm &&
               t.kind !== HSGPSummandTerm &&
               t.kind !== MonotonicSummandTerm)
        any(t -> t.kind === InterceptTerm, pred.terms) && push!(addressees, :Intercept)
        for a in addressees
            (pred.name, a) in seen ||
                _fail(:plan, "no prior for ($(pred.name), $a)")
        end
    end
    return nothing
end

# Non-leveled responses leave every leveled field at its default (fail
# closed: a stray level field on a Gaussian both means nothing and would
# silently change nothing — reject it).
function _validate_unleveled_fields(r::LikelihoodSpec)
    r.n_levels === nothing ||
        _fail(r.label, "only leveled families take n_levels")
    r.thresholds === nothing ||
        _fail(r.label, "only ordered families take thresholds")
    isempty(r.extra_predictors) ||
        _fail(r.label, "only CategoricalLogit takes extra_predictors")
    isempty(r.count_columns) ||
        _fail(r.label, "only Multinomial takes count_columns")
    r.ordinal_structure === nothing ||
        _fail(r.label, "only Ordinal takes ordinal_structure")
    r.discrimination === nothing ||
        _fail(r.label, "only Ordinal takes discrimination")
    isempty(r.threshold_columns) ||
        _fail(r.label, "only Ordinal takes threshold_columns")
    r.threshold_coefs === nothing ||
        _fail(r.label, "only Ordinal takes threshold_coefs")
    return nothing
end

# Leveled-field rules for predictor-addressing families (CategoricalLogit,
# OrderedLogistic, Ordinal); non-leveled families must leave every leveled
# field at its default. `used_predictors` gains categorical tail predictors.
function _validate_leveled_fields(r::LikelihoodSpec, plan::StructuralPlan,
        pred::PredictorSpec, used_predictors::Set{Symbol})
    _is_leveled_family(r.family) || return _validate_unleveled_fields(r)
    r.family === CategoricalLogitFam &&
        return _validate_categorical_fields(r, plan, used_predictors)
    return _validate_ordered_fields(r, plan, pred, used_predictors)
end

function _validate_categorical_fields(r::LikelihoodSpec, plan::StructuralPlan,
        used_predictors::Set{Symbol})
    for q in r.extra_predictors
        any(p -> p.name === q, plan.predictors) ||
            _fail(r.label, "extra predictor $q is not a plan predictor")
        push!(used_predictors, q)
    end
    all_preds = [r.predictor; r.extra_predictors...]
    length(unique(all_preds)) == length(all_preds) ||
        _fail(r.label, "categorical predictors repeat a predictor " *
            "($(join(all_preds, ", "))) — one linear predictor per non-reference class")
    k_struct = length(all_preds) + 1
    r.n_levels === nothing || r.n_levels == k_struct || _fail(r.label,
        "n_levels $(r.n_levels) disagrees with the $(length(all_preds)) " *
        "categorical predictors (K = predictors + 1 = $k_struct)")
    r.thresholds === nothing ||
        _fail(r.label, "CategoricalLogit takes no thresholds")
    isempty(r.count_columns) ||
        _fail(r.label, "CategoricalLogit takes no count_columns")
    r.ordinal_structure === nothing ||
        _fail(r.label, "CategoricalLogit takes no ordinal_structure")
    r.discrimination === nothing ||
        _fail(r.label, "CategoricalLogit takes no discrimination")
    isempty(r.threshold_columns) ||
        _fail(r.label, "CategoricalLogit takes no threshold_columns")
    r.threshold_coefs === nothing ||
        _fail(r.label, "CategoricalLogit takes no threshold_coefs")
    return nothing
end

function _validate_ordered_fields(r::LikelihoodSpec, plan::StructuralPlan,
        pred::PredictorSpec, used_predictors::Set{Symbol})
    r.thresholds === nothing && _fail(r.label,
        "an ordered response requires its thresholds vector parameter")
    tp = only(p for p in plan.vector_parameters if p.name === r.thresholds)
    want_ordered = r.family === OrderedLogisticFam ||
        r.ordinal_structure === :cumulative
    want_plain = r.family === OrdinalFam && r.ordinal_structure === :stopping
    if r.family === OrdinalFam
        r.ordinal_structure in (:cumulative, :stopping) || _fail(r.label,
            "Ordinal takes ordinal_structure :cumulative or :stopping, got " *
            "$(repr(r.ordinal_structure))")
        any(t -> t.kind === InterceptTerm, pred.terms) && _fail(r.label,
            "an Ordinal eta cannot include a fixed intercept — the estimated " *
            "thresholds already supply the location")
    else
        r.ordinal_structure === nothing ||
            _fail(r.label, "OrderedLogistic takes no ordinal_structure")
    end
    want_family = want_ordered ? :ordered_normal : :vector_normal
    (want_ordered || want_plain) || _fail(r.label,
        "internal: ordinal structure $(r.ordinal_structure) unresolved")
    tp.family === want_family || _fail(r.label,
        "thresholds $(tp.name) is $(tp.family) but this response needs " *
        "$want_family")
    if r.n_levels !== nothing && tp.size !== nothing
        tp.size == r.n_levels - 1 || _fail(r.label,
            "thresholds size $(tp.size) disagrees with n_levels " *
            "$(r.n_levels) (thresholds number K−1)")
    end
    r.n_levels === nothing || r.n_levels >= 1 || _fail(r.label,
        "n_levels must be ≥ 1, got $(r.n_levels)")
    isempty(r.extra_predictors) ||
        _fail(r.label, "an ordered response takes no extra_predictors")
    isempty(r.count_columns) ||
        _fail(r.label, "an ordered response takes no count_columns")
    _validate_ordinal_extras(r, plan, used_predictors)
    return nothing
end

# Ordinal-only extras: discrimination (a positive literal, a data
# column resolved at bind, or a modeled scale naming a LogLink plan
# predictor — positivity is structural via `exp`, the `log(disc)` recipe;
# a predictor under any other link, and any other in-graph name, fail
# closed), per-threshold design columns (StoppingRatio only: cumulative
# category-specific effects can break monotonicity), and their coefficient
# matrix (a `:vector_normal` vector parameter packing (K−1)×p, required
# exactly when design columns are present).
function _validate_ordinal_extras(r::LikelihoodSpec, plan::StructuralPlan,
        used_predictors::Set{Symbol})
    if r.family !== OrdinalFam
        r.discrimination === nothing ||
            _fail(r.label, "only Ordinal takes discrimination")
        isempty(r.threshold_columns) ||
            _fail(r.label, "only Ordinal takes threshold_columns")
        r.threshold_coefs === nothing ||
            _fail(r.label, "only Ordinal takes threshold_coefs")
        return nothing
    end
    d = r.discrimination
    if d isa Real
        (isfinite(d) && d > 0) || _fail(r.label,
            "ordinal discrimination must be finite and strictly positive, " *
            "got $(repr(d))")
    elseif d isa Symbol
        si = findfirst(p -> p.name === d, plan.predictors)
        if si !== nothing
            sp = plan.predictors[si]
            sp.link === LogLink || _fail(r.label,
                "ordinal discrimination predictor $d must carry LogLink " *
                "(a modeled scale is positive by construction via exp, " *
                "got $(sp.link))")
            push!(used_predictors, d)
        end
        # else a data column — resolved at bind.
    elseif d !== nothing
        _fail(r.label, "ordinal discrimination must be a positive literal, " *
            "a data column, or a log-link predictor, got $(repr(d))")
    end
    if !isempty(r.threshold_columns)
        r.ordinal_structure === :stopping || _fail(r.label,
            "per_threshold is supported for StoppingRatio() only; " *
            "unrestricted cumulative category-specific effects can make " *
            "cumulative probabilities non-monotone")
        length(unique(r.threshold_columns)) == length(r.threshold_columns) ||
            _fail(r.label, "threshold columns repeat a column " *
                "($(join(r.threshold_columns, ", ")))")
        r.threshold_coefs === nothing && _fail(r.label,
            "threshold columns require their threshold_coefs vector parameter")
        cp = only(p for p in plan.vector_parameters
            if p.name === r.threshold_coefs)
        cp.family === :vector_normal || _fail(r.label,
            "threshold_coefs $(cp.name) must be :vector_normal, got " *
            "$(cp.family)")
        if r.n_levels !== nothing && cp.size !== nothing
            want = (r.n_levels - 1) * length(r.threshold_columns)
            cp.size == want || _fail(r.label,
                "threshold_coefs size $(cp.size) disagrees with n_levels " *
                "$(r.n_levels) × $(length(r.threshold_columns)) columns " *
                "(packs (K−1)×p = $want)")
        end
    elseif r.threshold_coefs !== nothing
        _fail(r.label, "threshold_coefs without threshold_columns is " *
            "meaningless — drop it or add the design columns")
    end
    return nothing
end

# Simplex-location responses (no linear predictor): Multinomial names its
# count tail + trials; Categorical names an Int response. Both name their
# shared-simplex vector parameter in `predictor`, use IdentityLink (probs
# are used as-is — no link applies), and skip the triple.
function _validate_simplex_response(r::LikelihoodSpec, plan::StructuralPlan)
    r.link === IdentityLink || _fail(r.label,
        "a simplex response uses IdentityLink (probs are used as-is), got " *
        "$(r.link)")
    any(p -> p.name === r.predictor && p.family === :simplex_dirichlet,
        plan.vector_parameters) || _fail(r.label,
        "a simplex response names its :simplex_dirichlet vector parameter " *
        "in `predictor`, got $(r.predictor)")
    r.thresholds === nothing ||
        _fail(r.label, "a simplex response takes no thresholds")
    isempty(r.extra_predictors) ||
        _fail(r.label, "a simplex response takes no extra_predictors")
    r.ordinal_structure === nothing ||
        _fail(r.label, "a simplex response takes no ordinal_structure")
    r.discrimination === nothing ||
        _fail(r.label, "a simplex response takes no discrimination")
    isempty(r.threshold_columns) ||
        _fail(r.label, "a simplex response takes no threshold_columns")
    r.threshold_coefs === nothing ||
        _fail(r.label, "a simplex response takes no threshold_coefs")
    if r.family === MultinomialFam
        all_count = [r.response; r.count_columns...]
        length(unique(all_count)) == length(all_count) ||
            _fail(r.label, "multinomial count columns repeat a column " *
                "($(join(all_count, ", ")))")
        k_struct = length(all_count)
        r.n_levels === nothing || r.n_levels == k_struct || _fail(r.label,
            "n_levels $(r.n_levels) disagrees with the $k_struct count " *
            "columns")
        r.trials === nothing && _fail(r.label,
            "Multinomial response requires trials (Int column or literal)")
    else
        isempty(r.count_columns) ||
            _fail(r.label, "Categorical takes no count_columns")
        r.trials === nothing ||
            _fail(r.label, "Categorical takes no trials")
        r.n_levels === nothing || r.n_levels >= 1 || _fail(r.label,
            "n_levels must be ≥ 1, got $(r.n_levels)")
    end
    if r.range !== nothing
        first(r.range) == 1 || _fail(r.label,
            "response range must start at 1 (got $(r.range)) — " *
            "ranges cover eachindex exactly, no partial windows")
        length(r.range) >= 1 || _fail(r.label,
            "response range $(r.range) is empty")
    end
    return nothing
end

function _validate_responses(plan::StructuralPlan)
    # A kernel plate carries the only likelihood (panel v1): zero
    # top-level responses are admitted iff exactly one kernel plate is
    # present. Responseless GLM plans still fail.
    if isempty(plan.responses)
        # `_validate_kernels` runs before this gate and owns the plate-count
        # diagnosis; here exactly one kernel plate excuses zero responses.
        length(plan.kernel_plates) == 1 ||
            _fail(:plan, "plan has no responses")
    end
    rlabels = [r.label for r in plan.responses]
    length(unique(rlabels)) == length(rlabels) ||
        _fail(:plan, "duplicate response labels")
    for r in plan.responses
        r.label in RESERVED_NODES && _fail(
            r.label,
            "response label collides with a canonical node",
        )
    end
    scan_states = Set{Symbol}(s.state for s in plan.scans)
    used_predictors = Set{Symbol}()
    for r in plan.responses
        # A scan-state latent vector location: the mean is the carried state
        # directly (no linear predictor). Slice 1 admits Gaussian-identity only.
        if r.predictor in scan_states
            (r.family === GaussianFam && r.link === IdentityLink) || _fail(
                r.label,
                "a scan-state response location ($(r.predictor)) is " *
                "Gaussian-identity only in slice 1 (got $(r.family)/$(r.link))",
            )
            _validate_scale(r, plan)
            _validate_evidence_structure(r, plan)
            _validate_unleveled_fields(r)
            continue
        end
        # A simplex-vector location (no linear predictor — the scan-state
        # precedent): Multinomial/Categorical name their shared-simplex
        # vector parameter in `predictor`.
        if _is_simplex_family(r.family)
            _validate_simplex_response(r, plan)
            _validate_scale(r, plan)
            _validate_evidence_structure(r, plan)
            continue
        end
        idx = findfirst(p -> p.name === r.predictor, plan.predictors)
        idx === nothing &&
            _fail(r.label, "response addresses unknown predictor $(r.predictor)")
        push!(used_predictors, r.predictor)
        pred = plan.predictors[idx]
        (r.family, r.link, pred.link) in ADMITTED_TRIPLES || _fail(
            r.label,
            "link triple ($(r.family), $(r.link), $(pred.link)) not admitted " *
            "(admitted: Gaussian/identity, Bernoulli-logit/probit/cloglog, " *
            "Poisson-log, Binomial-logit/probit/cloglog, NB2-log, Gamma-log, " *
            "Beta-logit, Categorical-logit, Ordered-logit, Ordinal spellings)",
        )
        _validate_scale(r, plan)
        _validate_evidence_structure(r, plan)
        _validate_leveled_fields(r, plan, pred, used_predictors)
        if r.range !== nothing
            first(r.range) == 1 || _fail(r.label,
                "response range must start at 1 (got $(r.range)) — " *
                "ranges cover eachindex exactly, no partial windows")
            length(r.range) >= 1 || _fail(r.label,
                "response range $(r.range) is empty")
        end
    end
    for pred in plan.predictors
        pred.name in used_predictors ||
            _fail(pred.label, "predictor $(pred.name) unused by any response")
    end
    return nothing
end

function _validate_response_data(plan::StructuralPlan)
    for r in plan.responses
        _validate_response_column(r, plan)
        _validate_scale_data(r, plan)
        _validate_weights(r, plan)
        _validate_trials(r, plan)
        _validate_evidence_data(r, plan)
        _validate_ordinal_data(r, plan)
    end
    return nothing
end

function _validate_response_column(r::LikelihoodSpec, plan::StructuralPlan)
    _is_derived(plan, r.response) && _fail(r.label,
        "response $(r.response) is a derived column — slice-1 binds " *
        "responses raw (derived responses need shape metadata — planned)")
    haskey(plan.columns, r.response) ||
        _fail(r.label, "response column $(r.response) missing")
    if r.range !== nothing
        last(r.range) == plan.n_obs || _fail(r.label,
            "response range $(r.range) covers $(length(r.range)) cells " *
            "but n_obs is $(plan.n_obs) — ranges cover eachindex exactly")
    end
    col = plan.columns[r.response]
    if _is_bernoulli_family(r.family)
        eltype(col) === Bool && return nothing
        eltype(col) <: Integer && all(x -> x == 0 || x == 1, col) && return nothing
        return _fail(r.label, "Bernoulli response must be Bool or 0/1 integers")
    elseif r.family === PoissonLogFam
        eltype(col) <: Integer && all(>=(0), col) && return nothing
        return _fail(r.label, "Poisson response must be non-negative integers")
    elseif _is_binomial_family(r.family)
        _is_count_column(col) && return nothing
        return _fail(r.label, "Binomial response must be non-negative integers")
    elseif r.family === NegativeBinomial2Fam
        _is_count_column(col) && return nothing
        return _fail(r.label, "NB2 response must be non-negative integers")
    elseif r.family === GaussianFam
        eltype(col) <: Real ||
            _fail(r.label, "Gaussian response must be numeric")
        return nothing
    elseif r.family === GammaLogFam
        # Strictly positive: the gamma kernel guards x > 0, and at exactly
        # 0 it is wrong for shape ≤ 1 (says -Inf; truth is finite/+Inf) —
        # fail closed instead of flowing a wrong value.
        (eltype(col) <: Real && all(>(0), col)) ||
            _fail(r.label, "Gamma response must be strictly positive numerics")
        return nothing
    elseif r.family === BetaLogitFam
        # Strictly inside (0, 1): the beta kernel guards 0 < x < 1, and at
        # exactly 0/1 it is wrong for shapes ≤ 1 (says -Inf; truth is
        # finite/+Inf) — fail closed instead of flowing a wrong value.
        (eltype(col) <: Real && all(x -> 0 < x < 1, col)) ||
            _fail(r.label, "Beta response must be numerics strictly inside (0, 1)")
        return nothing
    elseif r.family === CategoricalLogitFam || _is_ordered_family(r.family) ||
            r.family === CategoricalFam
        # Recoded levels (SB `_brm_response_levels` recodes to contiguous
        # 1..K via sort(unique)): exact contiguity 1..K, K≥1. K=1 is uniform
        # (a zero-information likelihood, SB's one-emission rule) — except
        # CategoricalLogit, whose structural K needs ≥1 non-reference
        # predictor (a single-level categorical is degenerate and stays
        # inexpressible).
        (eltype(col) <: Integer && eltype(col) !== Bool) ||
            return _fail(r.label, "a leveled response must hold integers 1..K")
        K = r.n_levels
        K === nothing && return _fail(r.label,
            "internal: leveled n_levels unresolved at bind")
        all(x -> 1 <= x <= K, col) ||
            return _fail(r.label, "leveled response must hold integers " *
                "1..$K (recoded levels)")
        sort(unique(col)) == collect(1:K) ||
            return _fail(r.label, "leveled response must cover every level " *
                "1..$K exactly (recoded levels have no gaps)")
        return nothing
    elseif r.family === MultinomialFam
        # The count matrix crosses as K raw columns (lead + tail), each a
        # non-negative integer column; row sums meet trials in
        # `_validate_trials`.
        for c in [r.response; r.count_columns...]
            _is_derived(plan, c) && return _fail(r.label,
                "count column $c is derived — slice-2 binds count columns " *
                "raw (derived counts need shape metadata — planned)")
            haskey(plan.columns, c) ||
                return _fail(r.label, "count column $c missing")
            _is_count_column(plan.columns[c]) ||
                return _fail(r.label, "count column $c must hold " *
                    "non-negative integers")
        end
        return nothing
    else
        return _fail(r.label, "response family $(r.family) has no column rule")
    end
end

# Non-Bool integer column, all non-negative (Binomial/NB2 responses;
# Bool would pass `<: Integer` and die downstream — exclude it here).
_is_count_column(col) =
    eltype(col) <: Integer && eltype(col) !== Bool && all(>=(0), col)

# Bernoulli/Binomial span three link variants each (logit + slice-2
# probit/cloglog); response/trials rules are link-independent.
_is_bernoulli_family(f) =
    f === BernoulliLogitFam || f === BernoulliProbitFam || f === BernoulliCloglogFam
_is_binomial_family(f) =
    f === BinomialLogitFam || f === BinomialProbitFam || f === BinomialCloglogFam

function _validate_scale(r::LikelihoodSpec, plan::StructuralPlan)
    need = r.family === GaussianFam ? "Gaussian response requires a scale" :
        r.family === NegativeBinomial2Fam ?
        "NB2 response requires a dispersion phi" :
        r.family === GammaLogFam ? "Gamma response requires a shape alpha" :
        r.family === BetaLogitFam ? "Beta response requires a concentration kappa" : nothing
    if need === nothing
        r.scale === nothing ||
            _fail(r.label, "this response family takes no scale auxiliary")
    else
        r.scale === nothing &&
            _fail(r.label, "$need (parameter or literal)")
    end
    s = r.scale
    s === nothing && return nothing
    if s isa Real
        (isfinite(s) && s > 0) ||
            _fail(r.label, "scale literal must be finite positive")
        return nothing
    end
    # A scalar parameter/assignment scale resolves now; a per-observation scale
    # is a raw data column resolved at bind (see `_validate_scale_data`), so
    # defer an unknown symbol rather than failing structurally (mirrors how
    # per-obs weight/trials columns validate only once data is attached).
    s isa Symbol && s in _union_names(plan) && return nothing
    s isa Symbol && return nothing
    return _fail(r.label, "scale references unknown name $s")
end

# Data-level per-observation scale check: a scalar parameter/assignment name
# resolves structurally; a raw data-column scale (the eight-schools known SE)
# must be finite-positive numerics of length n_obs (a Gaussian/NB2/Gamma scale
# is strictly positive). A derived column scale is rejected — per-obs scales
# bind raw (mirrors the weights/trials raw-only rule).
function _validate_scale_data(r::LikelihoodSpec, plan::StructuralPlan)
    s = r.scale
    (s === nothing || s isa Real) && return nothing
    s isa Symbol || return nothing
    s in _union_names(plan) && return nothing
    _is_derived(plan, s) && _fail(r.label,
        "scale column $s is derived — slice-1 binds per-observation scales " *
        "raw (derived-column scales need shape metadata — planned)")
    haskey(plan.columns, s) ||
        _fail(r.label, "scale references unknown name $s")
    col = plan.columns[s]
    (eltype(col) <: Real && all(isfinite, col) && all(>(0), col)) ||
        _fail(r.label, "per-observation scale $s must be finite positive numerics")
    length(col) == plan.n_obs ||
        _fail(r.label, "scale column $s length $(length(col)) ≠ n_obs $(plan.n_obs)")
    return nothing
end

function _validate_trials(r::LikelihoodSpec, plan::StructuralPlan)
    if r.family === MultinomialFam
        return _validate_multinomial_trials(r, plan)
    end
    if !_is_binomial_family(r.family)
        r.trials === nothing ||
            _fail(r.label, "only Binomial/Multinomial responses take trials")
        return nothing
    end
    r.trials === nothing && _fail(r.label,
        "Binomial response requires trials (Int column or literal)")
    ycol = plan.columns[r.response]
    t = r.trials
    if t isa Int
        t >= 0 || _fail(r.label, "Binomial trials literal must be non-negative")
        all(ycol .<= t) ||
            _fail(r.label, "Binomial response exceeds trials $t")
        return nothing
    end
    _is_derived(plan, t) && _fail(r.label,
        "trials column $t is derived — slice-1 binds trials " *
        "raw (derived trials need shape metadata — planned)")
    haskey(plan.columns, t) ||
        _fail(r.label, "trials column $t missing")
    col = plan.columns[t]
    (eltype(col) <: Integer && eltype(col) !== Bool) ||
        _fail(r.label, "trials column must hold integers")
    all(>=(0), col) ||
        _fail(r.label, "trials column must be non-negative")
    length(col) == plan.n_obs ||
        _fail(r.label, "trials column length $(length(col)) ≠ n_obs $(plan.n_obs)")
    all(ycol .<= col) ||
        _fail(r.label, "Binomial response exceeds trials in some row")
    return nothing
end

# Multinomial trials: an Int literal or raw Int column, and every row's
# count sum meets N exactly (Stan's multinomial errors otherwise — fail
# closed here instead of flowing a wrong value).
function _validate_multinomial_trials(r::LikelihoodSpec, plan::StructuralPlan)
    t = r.trials
    t === nothing && _fail(r.label,
        "Multinomial response requires trials (Int column or literal)")
    counts = [r.response; r.count_columns...]
    rowsums = zeros(Int, plan.n_obs)
    for c in counts
        rowsums .+= plan.columns[c]
    end
    if t isa Int
        t >= 0 || _fail(r.label, "Multinomial trials literal must be non-negative")
        all(rowsums .== t) ||
            _fail(r.label, "multinomial row counts must sum to trials $t " *
                "in every row")
        return nothing
    end
    _is_derived(plan, t) && _fail(r.label,
        "trials column $t is derived — slice-2 binds trials " *
        "raw (derived trials need shape metadata — planned)")
    haskey(plan.columns, t) ||
        _fail(r.label, "trials column $t missing")
    col = plan.columns[t]
    (eltype(col) <: Integer && eltype(col) !== Bool) ||
        _fail(r.label, "trials column must hold integers")
    all(>=(0), col) ||
        _fail(r.label, "trials column must be non-negative")
    all(rowsums .== col) ||
        _fail(r.label, "multinomial row counts must sum to trials in every row")
    return nothing
end

# Ordinal extras at data level: a discrimination column is raw finite
# positive numerics of length n_obs (a literal validated structurally; a
# log-link predictor names a modeled scale — structural positivity, no
# data check; any other in-graph name is rejected), and threshold design
# columns are raw finite numerics of length n_obs.
function _validate_ordinal_data(r::LikelihoodSpec, plan::StructuralPlan)
    r.family === OrdinalFam || return nothing
    d = r.discrimination
    if d isa Symbol && any(p -> p.name === d, plan.predictors) &&
            haskey(plan.columns, d)
        _fail(r.label, "discrimination $d is both a predictor and a data " *
            "column — ambiguous (rename one)")
    end
    if d isa Symbol && !any(p -> p.name === d, plan.predictors)
        _is_derived(plan, d) && _fail(r.label,
            "discrimination column $d is derived — slice-2 binds " *
            "discrimination raw (derived columns need shape metadata — planned)")
        haskey(plan.columns, d) || _fail(r.label,
            "discrimination $d must be a data column or a log-link " *
            "predictor (a modeled scale)")
        col = plan.columns[d]
        (eltype(col) <: Real && all(isfinite, col) && all(>(0), col)) ||
            _fail(r.label, "discrimination column $d must be finite " *
                "positive numerics")
    end
    for c in r.threshold_columns
        _is_derived(plan, c) && _fail(r.label,
            "threshold column $c is derived — slice-2 binds threshold " *
            "design raw (derived columns need shape metadata — planned)")
        haskey(plan.columns, c) ||
            _fail(r.label, "threshold column $c missing")
        col = plan.columns[c]
        (eltype(col) <: Real && all(isfinite, col)) ||
            _fail(r.label, "threshold column $c must be finite numerics")
    end
    return nothing
end

function _validate_weights(r::LikelihoodSpec, plan::StructuralPlan)
    r.weights === nothing && return nothing
    _is_derived(plan, r.weights) && _fail(r.label,
        "weights column $(r.weights) is derived — slice-1 binds weights " *
        "raw (derived weights need shape metadata — planned)")
    haskey(plan.columns, r.weights) ||
        _fail(r.label, "weights column $(r.weights) missing")
    col = plan.columns[r.weights]
    eltype(col) <: Real && all(isfinite, col) && all(>=(0), col) ||
        _fail(r.label, "frequency weights must be finite non-negative numerics")
    return nothing
end

function _validate_evidence_structure(r::LikelihoodSpec, plan::StructuralPlan)
    ev = r.evidence
    ev.kind in (:none, :truncated, :censored, :interval_censored) ||
        _fail(r.label, "evidence kind $(ev.kind) unknown")
    ev.kind === :none && return nothing
    (r.family === GaussianFam || r.family === PoissonLogFam) ||
        _fail(r.label, "evidence wrappers apply to Gaussian/Poisson only (slice 1)")
    if ev.kind === :interval_censored
        ev.lower === nothing ||
            _fail(r.label, "interval evidence takes no lower (the response is the lower endpoint)")
        ev.upper === nothing &&
            _fail(r.label, "interval evidence requires an upper bound")
    end
    return nothing
end

function _validate_evidence_data(r::LikelihoodSpec, plan::StructuralPlan)
    ev = r.evidence
    ev.kind === :none && return nothing
    lo = _bound_values(ev.lower, :lower, r, plan)
    hi = _bound_values(ev.upper, :upper, r, plan)
    if ev.kind === :interval_censored
        resp = plan.columns[r.response]
        all(isfinite, resp) ||
            _fail(r.label, "interval evidence requires finite response values")
        all(resp .< hi) ||
            _fail(r.label, "interval evidence requires response < upper every row")
        return nothing
    end
    (lo === nothing || hi === nothing) && return nothing
    all(lo .< hi) ||
        _fail(r.label, "evidence requires strict lower < upper every row")
    return nothing
end

function _bound_values(bound, side, r::LikelihoodSpec, plan::StructuralPlan)
    bound === nothing && return nothing
    if bound isa Real
        isfinite(bound) || _fail(r.label, "$side bound literal must be finite")
        if r.family === PoissonLogFam
            isinteger(bound) || _fail(r.label,
                "$side bound literal must be integer-valued for Poisson evidence")
        end
        return fill(Float64(bound), plan.n_obs)
    end
    bound isa Symbol || _fail(r.label, "$side bound must be a literal or column")
    _is_derived(plan, bound) && _fail(r.label,
        "$side bound column $bound is derived — slice-1 binds evidence " *
        "bounds raw (derived bounds need shape metadata — planned)")
    haskey(plan.columns, bound) ||
        _fail(r.label, "$side bound column $bound missing")
    col = plan.columns[bound]
    if r.family === PoissonLogFam
        (eltype(col) <: Integer && eltype(col) !== Bool) || _fail(r.label,
            "$side bound column must hold integers for Poisson evidence")
        return col
    end
    eltype(col) <: Real && all(isfinite, col) ||
        _fail(r.label, "$side bound column must be finite numerics")
    return col
end

const _ROLE_RANK = Dict{Symbol,Int}(
    :data => 1, :predictor => 2, :group => 3, :weight => 4, :evidence => 5,
    :trials => 6, :response => 7,
)

_upgrade_role!(roles, col, role) =
    _ROLE_RANK[roles[col]] < _ROLE_RANK[role] && (roles[col] = role)

# Bind-time spline fit (the Stan transformed-data mirror): fit each basis
# from its raw axis columns (host, full LAPACK — eigen/nullspace are
# inexpressible in-graph), assert the fitted widths equal the declared
# static widths, and materialize one bound vector per basis column under
# the contract's <id>_<block>_<j> names. Caller-supplied columns under a
# materialized name are rejected (reserved-name exclusivity); the fit's
# own errors surface as ContractValidationErrors with the [spline] tag.
# Bind-time HSGP fit (the spline host-side transformed-data mirror):
# fills each basis's (mu, L) per axis from the RAW bound columns (SB
# `_brm_fit_hsgp` verbatim: `mu = mean(x)`, `L = c*max|x-mu|`).
# Degenerate axes (L == 0 — constant columns) and non-finite data
# fail closed here (bind owns data errors); the basis itself is
# evaluated in-graph in Stage B from the raw columns + these frozen
# fits (bind-then-build ordering keeps them consistent across
# rebinds — the SB DATA-vs-literal concern).
function _fit_hsgp_bases(plan::StructuralPlan,
        columns::Dict{Symbol,AbstractVector})
    isempty(plan.hsgp_bases) && return HSGPBasis[]
    out = HSGPBasis[]
    for hb in plan.hsgp_bases
        fits = Tuple{Float64,Float64}[]
        for (c, cj) in zip(hb.axes, hb.c)
            haskey(columns, c) ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c is " *
                      "not bound")
            col = columns[c]
            eltype(col) <: Real ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c must " *
                      "be numeric, got $(eltype(col))")
            isempty(col) &&
                _fail(hb.label, "hsgp :$(hb.id): axis column $c is empty")
            mu = sum(col) / length(col)
            L = Float64(cj) * maximum(abs.(col .- mu))
            isfinite(mu) && isfinite(L) ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c is " *
                      "non-finite (mu=$mu, L=$L)")
            L > 0 ||
                _fail(hb.label, "hsgp :$(hb.id): axis $c is degenerate " *
                      "(L == 0 — a constant column has no usable domain)")
            push!(fits, (Float64(mu), L))
        end
        push!(out, HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, fits,
            hb.label))
    end
    return out
end

function _materialize_splines!(plan::StructuralPlan,
        columns::Dict{Symbol,AbstractVector})
    isempty(plan.spline_bases) && return SplineBasis[]
    out = SplineBasis[]
    for sb in plan.spline_bases
        axes = AbstractVector[]
        for c in sb.axes
            haskey(columns, c) ||
                _fail(sb.label, "spline :$(sb.id): axis column $c is " *
                      "not bound")
            eltype(columns[c]) <: Real ||
                _fail(sb.label, "spline :$(sb.id): axis column $c must " *
                      "be numeric, got $(eltype(columns[c]))")
            push!(axes, columns[c])
        end
        wantcols = _spline_basis_columns(sb.id, sb.kind, sb.k)
        for (_, cols) in wantcols, c in cols
            haskey(columns, c) && _fail(sb.label,
                "column $c is reserved for spline :$(sb.id)'s " *
                "materialized basis — rename the caller-supplied column")
        end
        mats = sb.kind === :tps ?
            collect(_rk_apply_spline(_rk_fit_spline(axes[1]; k=sb.k),
                axes[1])) :
            collect(_rk_apply_t2(_rk_fit_t2(axes[1], axes[2]; k=sb.k),
                axes[1], axes[2]))
        blocks = SplineBasisBlock[]
        for (bi, (name, cols)) in enumerate(wantcols)
            size(mats[bi], 2) == length(cols) || _fail(sb.label,
                "spline :$(sb.id): fitted block :$name has width " *
                "$(size(mats[bi], 2)), declared $(length(cols))")
            for (j, c) in enumerate(cols)
                columns[c] = Vector{Float64}(mats[bi][:, j])
            end
            push!(blocks, SplineBasisBlock(name, length(cols), cols))
        end
        push!(out, SplineBasis(sb.id, sb.kind, sb.axes, sb.k, blocks,
            sb.label))
    end
    return out
end

# Kernel-plate bind resolution (the `_RKKernelSpec` layout contract): dims
# keys resolve `subjects`/`timepoints` to positive ints; slice kinds infer
# totally from lengths (`n_sub*T` → :vector, `n_sub` → :scalar — at `T ==
# 1` the lengths coincide and kinds are unobservable, so all-scalar
# stands); scalar slices in vector models materialize flat T-block
# expansions (spline-blocks precedent). Every dims key must be consumed —
# leftovers fail closed (a typo'd key must not silently reshape the
# plate). Returns resolved nodes; the input plan is untouched.
function _resolve_kernels!(plan::StructuralPlan,
        columns::Dict{Symbol,AbstractVector}, dims::AbstractDict{Symbol,<:Integer})
    isempty(plan.kernel_plates) && return KernelPlate[]
    kp = only(plan.kernel_plates)
    for (k, v) in dims
        v > 0 ||
            _fail(kp.label, "dims key `$k` must bind a positive integer, " *
                  "got $v")
    end
    consumed = Set{Symbol}()
    n_sub = if kp.subjects isa Int
        kp.subjects
    else
        haskey(dims, kp.subjects) ||
            _fail(kp.label, "subjects dims key `$(kp.subjects)` is not " *
                  "bound (bind_data `dims` carries it; admitted: an " *
                  "integer literal or a dims-key name)")
        push!(consumed, kp.subjects)
        Int(dims[kp.subjects])
    end
    T = if kp.timepoints isa Int
        kp.timepoints
    elseif kp.timepoints isa Symbol
        haskey(dims, kp.timepoints) ||
            _fail(kp.label, "timepoints dims key `$(kp.timepoints)` is " *
                  "not bound (bind_data `dims` carries it)")
        push!(consumed, kp.timepoints)
        Int(dims[kp.timepoints])
    else
        rest = setdiff(Set{Symbol}(keys(dims)), consumed)
        if isempty(rest)
            nothing
        elseif length(rest) == 1
            tk = only(rest)
            push!(consumed, tk)
            Int(dims[tk])
        else
            _fail(kp.label, "ambiguous timepoints dims keys " *
                  "$(sort!(collect(rest))) (one T key besides subjects)")
        end
    end
    leftovers = setdiff(Set{Symbol}(keys(dims)), consumed)
    isempty(leftovers) ||
        _fail(kp.label, "dims key(s) $(sort!(collect(leftovers))) not " *
              "consumed by kernel plate `$(kp.result)` (typo'd key?)")
    flat = _kernel_flat_length(n_sub, T)
    slices2 = Tuple{Symbol,Symbol,Symbol}[]
    for (col, param, _) in kp.slices
        haskey(columns, col) ||
            _fail(kp.label, "slice column `$col` is not bound")
        L = length(columns[col])
        kind = if T === nothing
            L == n_sub ||
                _fail(kp.label, "slice `$param` column `$col` has length " *
                      "$L ≠ n_sub $n_sub and no T dims key is bound " *
                      "(vector slices need T: bind `kernel_T_<result>`)")
            :scalar
        else
            # Scalar-first: at T == 1 the lengths coincide and kinds are
            # unobservable — recovering scalar is numerically identical
            # (1-element blocks; the inner-1 expansion is identity).
            L == n_sub ? :scalar :
                L == n_sub*T ? :vector :
                _fail(kp.label, "slice `$param` column `$col` has length " *
                      "$L, neither n_sub $n_sub nor n_sub*T $flat")
        end
        push!(slices2, (col, param, kind))
        if kind === :scalar && T !== nothing
            exp = _kexp_name(kp.result, col)
            haskey(columns, exp) &&
                _fail(kp.label, "column `$exp` is reserved for kernel " *
                      "plate `$(kp.result)`'s flat expansion of `$col` — " *
                      "rename the caller-supplied column")
            columns[exp] = repeat(Vector{Float64}(columns[col]); inner = T)
        end
    end
    resolved0 = KernelPlate(kp.result, n_sub, T, slices2,
        kp.assignments, kp.obs, kp.collected, kp.label)
    canon = _canonicalize_kernel_assignments(resolved0)
    return KernelPlate[KernelPlate(kp.result, n_sub, T, slices2,
        canon, kp.obs, kp.collected, kp.label)]
end

"""
    bind_data(plan, columns; roles=Dict(), dims=Dict()) -> StructuralPlan

Attach `columns` to a structure-only plan (or rebind an already-bound one,
replacing columns + roles): infer column roles, merge explicit `roles` over
them, and run data validation. Returns a NEW bound plan; the input is
untouched. Inference precedence: response > trials > evidence > weight >
group > predictor > data; term columns are the only `:predictor` source, so
assignment/extra columns stay `:data`. Bucket grouping columns upgrade to
`:group` (grouping dominates predictor use in the label; both facts stay
visible in terms + buckets). `dims` binds kernel-plate dims keys
(`subject_count`, `timepoint_count`) to positive integers; every key must
be consumed.
"""
function bind_data(plan::StructuralPlan, columns::Dict{Symbol,<:AbstractVector};
        roles::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
        dims::AbstractDict{Symbol,<:Integer} = Dict{Symbol,Int}())
    validate_structure(plan)
    isempty(columns) && throw(ContractValidationError(
        "[bind] bind_data requires non-empty columns"))
    columns = Dict{Symbol,AbstractVector}(columns)
    bases = _materialize_splines!(plan, columns)
    hbases = _fit_hsgp_bases(plan, columns)
    kbases = _resolve_kernels!(plan, columns, dims)
    for (k, v) in roles
        haskey(columns, k) || throw(ContractValidationError(
            "[bind] role for unknown column $k"))
        v in COLUMN_ROLES || throw(ContractValidationError(
            "[bind] unknown role $v for column $k (choose from $COLUMN_ROLES)"))
    end
    inferred = Dict{Symbol,Symbol}(k => :data for k in keys(columns))
    for pred in plan.predictors, t in pred.terms, c in t.columns
        haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
    end
    for b in plan.ranef_buckets
        haskey(inferred, b.group) && _upgrade_role!(inferred, b.group, :group)
    end
    for sb in bases, blk in sb.blocks, c in blk.columns
        haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
    end
    for hb in hbases, c in hb.axes
        haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
    end
    for r in plan.responses
        r.weights !== nothing && haskey(inferred, r.weights) &&
            _upgrade_role!(inferred, r.weights, :weight)
        r.trials isa Symbol && haskey(inferred, r.trials) &&
            _upgrade_role!(inferred, r.trials, :trials)
        for b in (r.evidence.lower, r.evidence.upper)
            b isa Symbol && haskey(inferred, b) &&
                _upgrade_role!(inferred, b, :evidence)
        end
    end
    for r in plan.responses
        haskey(inferred, r.response) && (inferred[r.response] = :response)
        for c in r.count_columns
            haskey(inferred, c) && (inferred[c] = :response)
        end
        for c in r.threshold_columns
            haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
        end
    end
    for kp in kbases
        rcol = only(c for (c, p, _) in kp.slices if p === kp.obs.response)
        haskey(inferred, rcol) && (inferred[rcol] = :response)
    end
    merged = merge(inferred, roles)
    # Kernel plans carry two column lengths by design: n_obs is the flat
    # length (vector models) or n_sub (all-scalar) — never first-column.
    n = isempty(kbases) ? length(first(values(columns))) :
        _kernel_flat_length(only(kbases).subjects, only(kbases).timepoints)
    maps = _eval_levelmaps(plan.levelmaps, columns)
    responses2, vectors2 =
        _infer_leveled_sizes(plan.responses, plan.vector_parameters, columns,
            plan.predictors)
    bound = StructuralPlan(responses2, plan.predictors,
        plan.population_priors, plan.parameters, plan.assignments,
        columns, n; roles = merged, derived = plan.derived,
        levelmaps = maps, plate_parameters = plan.plate_parameters,
        scans = plan.scans, ranef_buckets = plan.ranef_buckets,
        vector_parameters = vectors2, spline_bases = bases,
        spline_vectors = plan.spline_vectors, hsgp_bases = hbases,
        kernel_plates = kbases)
    validate_data(bound)
    return bound
end

# Bind-time level inference (the LevelMap binder-evaluation precedent):
# `n_levels === nothing` fills structurally (CategoricalLogit: 1 +
# predictor count; Multinomial: count-column count) or from the response
# column (max(y) for OrderedLogistic/Ordinal/Categorical); vector-param
# `size === nothing` fills from the linked response (K−1 thresholds, K
# simplex) or, for a monotonic-linked increments simplex, from its frozen
# concentration length (K−1 increments for K levels). Explicit values
# assert against the inference. Returns new (immutable) vectors; unbound
# plans keep `nothing`.
function _infer_leveled_sizes(responses::Vector{LikelihoodSpec},
        vectors::Vector{VectorParameter}, columns::Dict{Symbol,AbstractVector},
        predictors::Vector{PredictorSpec} = PredictorSpec[])
    out_r = LikelihoodSpec[]
    for r in responses
        _is_leveled_family(r.family) || (push!(out_r, r); continue)
        K = _infer_response_levels(r, columns)
        push!(out_r, _with_levels(r, K))
    end
    by_label = Dict{Symbol,LikelihoodSpec}(r.label => r for r in out_r)
    thresh_link = Dict{Symbol,Symbol}()
    simplex_link = Dict{Symbol,Symbol}()
    coefs_link = Dict{Symbol,Symbol}()
    for r in out_r
        r.thresholds !== nothing && (thresh_link[r.thresholds] = r.label)
        r.threshold_coefs !== nothing && (coefs_link[r.threshold_coefs] = r.label)
        if _is_simplex_family(r.family)
            simplex_link[r.predictor] = r.label
        end
    end
    monotonic_link = Dict{Symbol,Symbol}()
    for pred in predictors, t in pred.terms
        (t.kind === MonotonicTerm || t.kind === MonotonicSummandTerm) ||
            continue
        monotonic_link[t.options.increments] = t.label
    end
    out_v = VectorParameter[]
    for p in vectors
        if haskey(thresh_link, p.name)
            K = by_label[thresh_link[p.name]].n_levels
            K === nothing && _fail(p.label, "internal: linked n_levels unresolved")
            want = K - 1
            p.size === nothing || p.size == want || _fail(p.label,
                "thresholds size $(p.size) disagrees with n_levels $K " *
                "(thresholds number K−1)")
            push!(out_v, VectorParameter(p.name, p.family, p.args, want, p.label))
        elseif haskey(coefs_link, p.name)
            lr = by_label[coefs_link[p.name]]
            K = lr.n_levels
            K === nothing && _fail(p.label, "internal: linked n_levels unresolved")
            want = (K - 1) * length(lr.threshold_columns)
            p.size === nothing || p.size == want || _fail(p.label,
                "threshold_coefs size $(p.size) disagrees with n_levels $K " *
                "× $(length(lr.threshold_columns)) columns (packs (K−1)×p = $want)")
            push!(out_v, VectorParameter(p.name, p.family, p.args, want, p.label))
        elseif haskey(simplex_link, p.name)
            K = by_label[simplex_link[p.name]].n_levels
            K === nothing && _fail(p.label, "internal: linked n_levels unresolved")
            p.size === nothing || p.size == K || _fail(p.label,
                "simplex size $(p.size) disagrees with n_levels $K")
            length(p.args.arg1) == K || _fail(p.label,
                "Dirichlet concentration length $(length(p.args.arg1)) " *
                "disagrees with n_levels $K")
            push!(out_v, VectorParameter(p.name, p.family, p.args, K, p.label))
        elseif haskey(monotonic_link, p.name)
            want = length(p.args.arg1)
            want >= 1 || _fail(p.label,
                "monotonic increments need ≥ 1 increment " *
                "(K=1 degenerates emitter-side and never reaches the thin layer)")
            p.size === nothing || p.size == want || _fail(p.label,
                "monotonic increments size $(p.size) disagrees with its " *
                "concentration length $want")
            push!(out_v, VectorParameter(p.name, p.family, p.args, want, p.label))
        else
            _fail(p.label, "internal: vector parameter unlinked at bind")
        end
    end
    return out_r, out_v
end

function _infer_response_levels(r::LikelihoodSpec, columns::Dict{Symbol,AbstractVector})
    if r.family === CategoricalLogitFam
        return 2 + length(r.extra_predictors)
    elseif r.family === MultinomialFam
        return 1 + length(r.count_columns)
    end
    r.n_levels !== nothing && return r.n_levels
    haskey(columns, r.response) ||
        _fail(r.label, "response column $(r.response) missing")
    col = columns[r.response]
    (eltype(col) <: Integer && eltype(col) !== Bool && !isempty(col)) ||
        _fail(r.label, "a leveled response must hold integers 1..K")
    K = maximum(col)
    K >= 1 || _fail(r.label, "a leveled response must hold integers 1..K")
    return K
end

# Rebuild a response with concrete n_levels (explicit values already
# asserted structurally; inference fills `nothing`).
function _with_levels(r::LikelihoodSpec, K::Int)
    r.n_levels === nothing || r.n_levels == K || _fail(r.label,
        "n_levels $(r.n_levels) disagrees with the bound data " *
        "(inferred K = $K)")
    return LikelihoodSpec(r.family, r.link, r.response, r.predictor,
        r.scale, r.weights, r.evidence, r.label, r.trials, r.range;
        n_levels = K, thresholds = r.thresholds,
        extra_predictors = r.extra_predictors, count_columns = r.count_columns,
        ordinal_structure = r.ordinal_structure,
        discrimination = r.discrimination,
        threshold_columns = r.threshold_columns,
        threshold_coefs = r.threshold_coefs)
end

