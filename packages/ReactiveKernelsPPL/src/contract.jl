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

"""Slice-1 likelihood families (D3 narrow slice)."""
@enum LikelihoodFamily::UInt8 begin
    GaussianFam
    BernoulliLogitFam
    PoissonLogFam
    BinomialLogitFam
    NegativeBinomial2Fam
    GammaLogFam
end

"""Link functions. The enum crosses the boundary; the thin layer owns the
inverse-link numerics (R1 counter)."""
@enum LinkFunction::UInt8 begin
    IdentityLink
    LogitLink
    LogLink
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
end
LikelihoodSpec(family, link, response, predictor, scale, weights, evidence,
    label) =
    LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, nothing, nothing)
LikelihoodSpec(family, link, response, predictor, scale, weights, evidence,
    label, range) =
    LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, nothing, range)

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
    SampledParameter(name, family, args, support_override, label)

One non-coefficient latent (scalar, slice 1). `args` use POSITIONAL keys
`(arg1, arg2, …)` in Distributions.jl constructor order with Distributions.jl
semantics (`Exponential(θ)` = scale θ). Values are literals or
[`ParamName`](@ref)s (hierarchical OK, cycles rejected). `support_override`
is `nothing` (infer from family) or `:positive` for half-Normal/half-Cauchy
style truncations of real-support families.
"""
struct SampledParameter
    name::ParamName
    family::Symbol
    args::NamedTuple
    support_override::Union{Nothing,Symbol}
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
    support_override::Union{Nothing,Symbol}
    range::Union{Nothing,UnitRange{Int}}
    label::Symbol
end
"""Provenance/range default to a whole-column (n_obs) plate under the name."""
PlateParameter(name::ParamName, family::Symbol, args::NamedTuple,
    support_override::Union{Nothing,Symbol}) =
    PlateParameter(name, family, args, support_override, nothing, name)
PlateParameter(name::ParamName, family::Symbol, args::NamedTuple,
    support_override::Union{Nothing,Symbol}, range::Union{Nothing,UnitRange{Int}}) =
    PlateParameter(name, family, args, support_override, range, name)

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
latent, and each `scans` state share one name table (duplicates rejected).
N≥1 independent responses; shared predictor Symbols allowed. `levelmaps` sizes
every factor term (binder-evaluated values); `plate_parameters` carries
per-cell latents and `scans` sequential-recurrence latents (both
vector-valued, empty for a plain population-GLM plan).
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
end

# Pre-extension full-positional constructor (9-arg): callers that built a plan
# before `levelmaps`/`scans` existed keep working with both empty.
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
        assignments, derived, columns, n_obs, roles, LevelMap[], ScanSpec[])

"""Column roles: what a bound column IS (CV travel + program transforms read
this). Inferred at bind; explicit roles override. Unbound plans carry none."""
const COLUMN_ROLES = (:response, :predictor, :weight, :evidence, :trials, :data)

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
        scans::Vector{ScanSpec} = ScanSpec[])
    return StructuralPlan(responses, predictors, population_priors,
        parameters, assignments, derived, columns, n_obs, roles, levelmaps,
        plate_parameters, scans)
end

"""Bound ⟺ columns attached. Unbound plans (empty columns) carry structure
only; [`bind_data`](@ref) attaches data (+ roles). Rebinding replaces."""
isbound(plan::StructuralPlan) = !isempty(plan.columns)

"""Admitted (family, likelihood-link, predictor-link) triples (triple pin).
Triples 2 and 3 lower identically; the triple is admission key + lowering
selector, never pairwise link equality."""
const ADMITTED_TRIPLES = (
    (GaussianFam, IdentityLink, IdentityLink),
    (BernoulliLogitFam, LogitLink, IdentityLink),
    (BernoulliLogitFam, LogitLink, LogitLink),
    (PoissonLogFam, LogLink, LogLink),
    (BinomialLogitFam, LogitLink, IdentityLink),
    (NegativeBinomial2Fam, LogLink, LogLink),
    (GammaLogFam, LogLink, LogLink),
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

"""Slice-1 term-name vocabulary (emitter-side admission keys)."""
const TERM_NAMES = Dict{Symbol,TermKind}(
    :intercept => InterceptTerm,
    :continuous => ContinuousTerm,
    :factor => FactorTerm,
    :offset => OffsetTerm,
)

"""Allowlisted assignment functions (slice 1: scalar ops + whole-column
reductions; elementwise math over columns deferred with vector assignments)."""
const ASSIGNMENT_FNS = (
    :+, :-, :*, :/, :^,
    :log, :log10, :log1p, :exp, :expm1, :sqrt, :abs,
    :sum, :mean, :std, :var, :minimum, :maximum, :length,
)

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
    BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam)

"""Term kinds the thin layer can lower (ext handshake predicate)."""
admitted_terms() = (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm)

"""Assignment functions the thin layer can lower (ext handshake predicate)."""
admitted_functions() = ASSIGNMENT_FNS

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
    _validate_topo_order(plan)
    _validate_predictors(plan)
    _validate_levelmaps(plan)
    _validate_priors(plan)
    _validate_responses(plan)
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

function _validate_columns(plan::StructuralPlan)
    plan.n_obs > 0 || _fail(:plan, "n_obs must be positive, got $(plan.n_obs)")
    for (name, col) in plan.columns
        length(col) == plan.n_obs ||
            _fail(name, "column length $(length(col)) ≠ n_obs $(plan.n_obs)")
        !any(ismissing, col) ||
            _fail(name, "column contains missing (slice 1 has no missingness machinery)")
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
    length(unique(params)) == length(params) || _fail(:plan, "duplicate parameter names")
    length(unique(assigns)) == length(assigns) ||
        _fail(:plan, "duplicate assignment names")
    length(unique(deriveds)) == length(deriveds) ||
        _fail(:plan, "duplicate derived-column names")
    length(unique(plates)) == length(plates) ||
        _fail(:plan, "duplicate plate-parameter names")
    length(unique(scanstates)) == length(scanstates) ||
        _fail(:plan, "duplicate scan-state names")
    for (l, r, what) in ((params, assigns, "parameters and assignments"),
        (params, deriveds, "parameters and derived columns"),
        (assigns, deriveds, "assignments and derived columns"),
        (plates, params, "plate parameters and parameters"),
        (plates, assigns, "plate parameters and assignments"),
        (plates, deriveds, "plate parameters and derived columns"),
        (params, scanstates, "parameters and scan states"),
        (assigns, scanstates, "assignments and scan states"),
        (deriveds, scanstates, "derived columns and scan states"),
        (plates, scanstates, "plate parameters and scan states"))
        overlap = intersect(l, r)
        isempty(overlap) ||
            _fail(:plan, "names in both $what: $(join(overlap, ", "))")
    end
    allnames = union(params, assigns, deriveds, plates, scanstates)
    for pn in pnames
        pn in allnames && _fail(
            :plan,
            "predictor $pn collides with a parameter/assignment/derived/plate/scan name",
        )
        block_name(pn) in allnames && _fail(
            :plan,
            "parameter/assignment/derived/plate/scan $(block_name(pn)) collides with predictor $pn block name",
        )
    end
    for n in Iterators.flatten((pnames, params, assigns, deriveds, plates, scanstates))
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
function _validate_support_override(label, family::Symbol,
        ov::Union{Nothing,Symbol}, args::NamedTuple)
    ov === nothing && return nothing
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
        # PlateParameter, whose prior lives on the plate parameter itself, not
        # as a PopulationPrior — neither needs a coefficient prior here.
        addressees = Set{Symbol}(t.addressee for t in pred.terms
            if t.kind !== OffsetTerm && t.kind !== LatentTerm)
        any(t -> t.kind === InterceptTerm, pred.terms) && push!(addressees, :Intercept)
        for a in addressees
            (pred.name, a) in seen ||
                _fail(:plan, "no prior for ($(pred.name), $a)")
        end
    end
    return nothing
end

function _validate_responses(plan::StructuralPlan)
    isempty(plan.responses) && _fail(:plan, "plan has no responses")
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
            "(slice 1: Gaussian/identity, Bernoulli-logit, Poisson-log spellings)",
        )
        _validate_scale(r, plan)
        _validate_evidence_structure(r, plan)
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
    if r.family === BernoulliLogitFam
        eltype(col) === Bool && return nothing
        eltype(col) <: Integer && all(x -> x == 0 || x == 1, col) && return nothing
        return _fail(r.label, "Bernoulli response must be Bool or 0/1 integers")
    elseif r.family === PoissonLogFam
        eltype(col) <: Integer && all(>=(0), col) && return nothing
        return _fail(r.label, "Poisson response must be non-negative integers")
    elseif r.family === BinomialLogitFam
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
    else
        return _fail(r.label, "response family $(r.family) has no column rule")
    end
end

# Non-Bool integer column, all non-negative (Binomial/NB2 responses;
# Bool would pass `<: Integer` and die downstream — exclude it here).
_is_count_column(col) =
    eltype(col) <: Integer && eltype(col) !== Bool && all(>=(0), col)

function _validate_scale(r::LikelihoodSpec, plan::StructuralPlan)
    need = r.family === GaussianFam ? "Gaussian response requires a scale" :
        r.family === NegativeBinomial2Fam ?
        "NB2 response requires a dispersion phi" :
        r.family === GammaLogFam ? "Gamma response requires a shape alpha" : nothing
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
    if r.family !== BinomialLogitFam
        r.trials === nothing ||
            _fail(r.label, "only Binomial responses take trials")
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
    :data => 1, :predictor => 2, :weight => 3, :evidence => 4, :trials => 5,
    :response => 6,
)

_upgrade_role!(roles, col, role) =
    _ROLE_RANK[roles[col]] < _ROLE_RANK[role] && (roles[col] = role)

"""
    bind_data(plan, columns; roles=Dict()) -> StructuralPlan

Attach `columns` to a structure-only plan (or rebind an already-bound one,
replacing columns + roles): infer column roles, merge explicit `roles` over
them, and run data validation. Returns a NEW bound plan; the input is
untouched. Inference precedence: response > trials > evidence > weight >
predictor > data; term columns are the only `:predictor` source, so
assignment/extra columns stay `:data`.
"""
function bind_data(plan::StructuralPlan, columns::Dict{Symbol,<:AbstractVector};
        roles::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}())
    validate_structure(plan)
    isempty(columns) && throw(ContractValidationError(
        "[bind] bind_data requires non-empty columns"))
    columns = Dict{Symbol,AbstractVector}(columns)
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
    end
    merged = merge(inferred, roles)
    n = length(first(values(columns)))
    maps = _eval_levelmaps(plan.levelmaps, columns)
    bound = StructuralPlan(plan.responses, plan.predictors,
        plan.population_priors, plan.parameters, plan.assignments,
        columns, n; roles = merged, derived = plan.derived,
        levelmaps = maps, plate_parameters = plan.plate_parameters,
        scans = plan.scans)
    validate_data(bound)
    return bound
end

