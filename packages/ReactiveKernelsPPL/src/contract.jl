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

"""Bound column values: length-n vectors or n-row matrices (whole-design
data, Stan `matrix[N,K]`). Matrices bind and validate beside vectors;
every per-observation role (response, term, weights, trials, scales,
bounds, grouping, axes, slices) reads vectors only — those readers fetch
through [`_vector_column`](@ref) and fail closed on a matrix."""
const ColumnData = Union{AbstractVector,AbstractMatrix}

# Row count of a bound column: length for vectors, row count for matrices.
_column_nrows(col::AbstractVector) = length(col)
_column_nrows(col::AbstractMatrix) = size(col, 1)

"""Normalize caller-supplied columns to the plan's column table (vectors
and matrices only — anything else fails closed here, not in a converter)."""
function _checked_columns(columns::AbstractDict{Symbol})
    out = Dict{Symbol,ColumnData}()
    for (k, v) in columns
        v isa ColumnData ||
            _fail(:plan, "column $k must be a vector or matrix, got $(summary(v))")
        out[k] = v
    end
    return out
end

"""Fetch a bound column that must be a VECTOR (every per-observation role —
response, term, weights, trials, scales, bounds, grouping, axes, slices —
never reads a matrix; those fail closed here)."""
function _vector_column(columns::AbstractDict{Symbol}, name::Symbol,
        label::Symbol, what::AbstractString)
    col = columns[name]
    col isa AbstractVector ||
        _fail(label, "$what `$name` must be a vector column, got " *
              "$(summary(col)) (matrix columns bind whole-design :data — " *
              "no $what reads one)")
    return col
end

"""Slice-1 likelihood families (D3 narrow slice), plus the leveled slice-2
families (categorical / ordinal / multinomial): reference-coded
multi-logit categorical, cumulative-logit ordinal with ordered cutpoints,
general typed ordinal (2 structures × 3 links), shared-simplex multinomial
over a count matrix, plain categorical over simplex probabilities, and the
joint correlated-outcomes family (per-row MvNormal over a Cholesky factor)."""
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
    MvNormalCholeskyFam
    CensoredAddpropnormalFam
    TgiCategoryFam
    TgiResponseFam
    TgiCensoredFam
    NormalIDGLMFam
    BernoulliLogitGLMFam
    PoissonLogGLMFam
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
random-effects / per-observation-latent location. `ScanSummandTerm` splices a
sequential-recurrence state into the predictor scaled by a sampled scalar
coefficient (`u .* beta`, SB's `ar` latent path with its free `popefs` beta).
`DarSummandTerm` splices a differenced-AR(1) trajectory state into the
predictor UNSCALED (SB's `dar` zero-started integrated path; the formula
intercept is the initial level, so no coefficient is identified)."""
@enum TermKind::UInt8 begin
    InterceptTerm
    ContinuousTerm
    FactorTerm
    OffsetTerm
    LatentTerm
    SplineSummandTerm
    HSGPSummandTerm
    ScanSummandTerm
    MonotonicTerm
    MonotonicSummandTerm
    VaryingEffectTerm
    MatrixTerm
    DarSummandTerm
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
    ScalePredictorRef(predictor, link)

A predictor-fed scale/shape use: the response's auxiliary (Gaussian
sigma, NB2 dispersion phi, Gamma shape alpha) is a whole linear
predictor, varying per observation. `predictor` names the
[`PredictorSpec`](@ref) (planned exactly like a location predictor:
terms, priors, one link); `link` is the scale use-site wrapper —
[`IdentityLink`](@ref) (bare predictor), [`LogLink`](@ref)
(`exp.(predictor)`), or [`LogitLink`](@ref) (`logistic.(predictor)`) —
and must equal the predictor's own link (the one-link-per-predictor
rule). The generator binds the constrained vector once per response
(`_ppl_sc_<label>`) and threads it through the likelihood plate per
cell, so evidence corrections read the per-cell scale.
"""
struct ScalePredictorRef
    predictor::Symbol
    link::LinkFunction
end

"""
    LikelihoodSpec(family, link, response, predictor, scale, weights, evidence, label[, trials[, range]])

One independent response. `scale` is the response's auxiliary —
Gaussian sigma, NB2 dispersion phi, Gamma shape alpha — either scalar
(parameter, assignment, folded literal, or a raw per-observation data
column) or, for Gaussian/NB2/Gamma only, a [`ScalePredictorRef`](@ref)
(predictor-fed per-observation scale); it must be `nothing` otherwise.
(One slot covers every admitted family; a two-auxiliary family such as
Beta needs a new field — noted, not built.) `weights` is a
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
scan-state precedent). A Gaussian-identity response may likewise name a
[`PlateParameter`](@ref) in `predictor`: a latent-mean observation
(`x_obs ~ Normal(x_true, sd)` with scalar constant `sd` — the SB `me`
mirror).


Joint correlated-outcomes responses (`MvNormalCholeskyFam`, SB
`[y1..yK] ~ MvNormalCholesky([mu1..muK], L)`) use the trailing joint
fields, built with keywords (`extra_responses=`, `factor_scales=`,
`factor_corr=`); every other family leaves them at their defaults:

- `extra_responses`: joint outcome columns after `response`
  (outcome order 1..K is `[response; extra_responses...]`, K ≥ 1);
  empty otherwise. Each outcome's mean is an identity-link linear
  predictor: `predictor` for outcome 1, `extra_predictors` (reused
  from the CategoricalLogit shape) for outcomes 2..K.
- `factor_scales`: the joint factor's positive scale vector
  ([`VectorParameter`](@ref), K scales), `nothing` otherwise.
- `factor_corr`: the joint factor's LKJ Cholesky factor
  ([`VectorParameter`](@ref), K×K), `nothing` otherwise.

Joint widths are structural (K = 1 + `length(extra_responses)`):
no `n_levels`, no bind-time size inference — the factor parameters
carry concrete sizes validated against K.

GLM-object responses (`NormalIDGLMFam`, `BernoulliLogitGLMFam`,
`PoissonLogGLMFam`) name their [`DesignMatrix`](@ref) in `predictor`
(no linear predictor — the object owns eta, the Multinomial/Categorical
scan-state precedent) and carry the split coefficients in the trailing
`glm_alpha`/`glm_beta` fields, built with keywords; every other family
leaves them at their defaults:

- `glm_alpha`: the scalar intercept parameter, `nothing` otherwise.
- `glm_beta`: the coefficient-vector parameter (one element per matrix
  column — the matrix is intercept-free, the generator prepends the
  ones column and `alpha`), `nothing` otherwise.
"""
struct LikelihoodSpec
    family::LikelihoodFamily
    link::LinkFunction
    response::ColumnRef
    predictor::Symbol
    scale::Union{Nothing,ParamName,Real,ScalePredictorRef}
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
    extra_responses::Vector{ColumnRef}
    factor_scales::Union{Nothing,ParamName}
    factor_corr::Union{Nothing,ParamName}
    glm_alpha::Union{Nothing,ParamName}
    glm_beta::Union{Nothing,ParamName}
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
        threshold_coefs::Union{Nothing,ParamName} = nothing,
        extra_responses::Vector{ColumnRef} = Symbol[],
        factor_scales::Union{Nothing,ParamName} = nothing,
        factor_corr::Union{Nothing,ParamName} = nothing,
        glm_alpha::Union{Nothing,ParamName} = nothing,
        glm_beta::Union{Nothing,ParamName} = nothing)
    return LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, trials, range, n_levels, thresholds,
        extra_predictors, count_columns, ordinal_structure, discrimination,
        threshold_columns, threshold_coefs, extra_responses, factor_scales,
        factor_corr, glm_alpha, glm_beta)
end

"""
    TermSpec(kind, columns, options, addressee, label)

One additive predictor term: structure only, never materialized designs
(D5a). `addressee` is the prior address (source column or `:Intercept`),
never a per-level label. Terms take no options: factor sizing lives in
the plan's [`LevelMap`](@ref)s (full-rank over exactly the mapped
levels; no contrasts, no reference dropping — that machinery was
BRM-specific and is gone). A `ContinuousTerm` may name a per-cell latent
([`PlateParameter`](@ref)) instead of a data column: the latent vector
enters the design with a free coefficient (the SB `me` mirror).
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
    R2D2Prior(predictor, r2, phi, tau, overrides)

Flat whole-predictor R2D2 variance decomposition (SB mirror, the
`effect(lp,:) ~ r2d2(...)` form — the `sd(...) ~ r2d2(...)` R2D2M2/ICC
grammar belongs to the hierarchical lane, never here). One per
predictor at most; a predictor with an `R2D2Prior` carries NO
[`PopulationPrior`](@ref) rows (coverage moves here).

- `r2` names the scalar `Beta` [`SampledParameter`](@ref).
- `phi` names the `:simplex_dirichlet` [`VectorParameter`](@ref) whose
  length is the share count.
- `tau` is the total scale: a sampled-parameter name (half-Normal) or
  a positive literal (SB's data `tau_bsv`).
- `overrides` maps addressees with an explicit Normal prior to their
  `(location, scale)` — those columns keep their own scale and leave
  the simplex (the SB share_idx/fallback composition). The intercept
  is always share 0 (its override, if stated, supplies loc/scale).

Share assignment follows SB exactly: every non-intercept design
column without an override takes the next share in design-column
order; the emitter derives
`scale[j] = sqrt(phi[share] * R2 * tau^2 / varx[j])`.
"""
struct R2D2Prior
    predictor::Symbol
    r2::Symbol
    phi::Symbol
    tau::Union{Symbol,Real}
    overrides::Dict{Symbol,Tuple{Float64,Float64}}
end

"""
    SupportOverride

A latent's support override: `nothing` (infer from the family), the bare
`Symbol` `:positive` (half-Normal/half-Cauchy truncation of a real-support
family, exact +log(2) at a literal-zero location), the tuple
`(:interval, lo, hi)` (a two-sided finite truncation `truncated(Normal(mu, s),
lo, hi)` — an affine-logistic constrained transform onto `(lo, hi)` with the
renormalized truncated density), or the tuple `(:upper, hi)` (an upper-only
truncation `truncated(Normal(mu, s), -Inf, hi)` — Stan's upper-bound kernel
`x = hi - exp(u)` with the bare-`u` Jacobian and NO truncation renormalizer).
Shared by scalar [`SampledParameter`](@ref)s and per-cell
[`PlateParameter`](@ref)s.
"""
const SupportOverride =
    Union{Nothing,Symbol,Tuple{Symbol,Float64,Float64},Tuple{Symbol,Float64}}

"""
    SampledParameter(name, family, args, support_override, label)

One non-coefficient latent (scalar, slice 1). `args` use POSITIONAL keys
`(arg1, arg2, …)` in Distributions.jl constructor order with Distributions.jl
semantics (`Exponential(θ)` = scale θ). Values are literals or
[`ParamName`](@ref)s (hierarchical OK, cycles rejected). `support_override`
is a [`SupportOverride`](@ref): `nothing` (infer from family), `:positive`
(half-Normal/half-Cauchy), `(:interval, lo, hi)` (a finite truncated
interval), or `(:upper, hi)` (an upper-only truncation with Stan kernel
semantics).
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
    VaryingZRecipe(kind, column, level)

One varying-effect design (Z) column recipe: structure only, never a
materialized vector pre-codegen. `kind` is `:ones` (intercept — `column`
is `:none`, `level` is `nothing`), `:column` (continuous raw column),
or `:dummy` (indicator over `column`: `level` is the level VALUE for
`Int`, exact match for `AbstractString`). The thin layer performs NO
coding inference — treatment/cell-means decisions arrive as explicit
`dummy` recipes from the emitter.
"""
struct VaryingZRecipe
    kind::Symbol
    column::Symbol
    level::Union{Nothing,Int,AbstractString}
end

"""
    VaryingMargin(coefficient, z)

One varying-effect margin in draws order: `coefficient` is the margin
address (`:Intercept`, a column, or a dummy label), `z` is the
[`VaryingZRecipe`](@ref) for its Z column. Margins live on the shared
draws; per-target column ranges live on [`VaryingSlice`](@ref).
"""
struct VaryingMargin
    coefficient::Symbol
    z::VaryingZRecipe
end

"""
    VaryingSdPrior(family, param)

One margin's marginal-scale (`tau`) prior inside a [`VaryingDraws`](@ref)
block — the SB `brm_ranef_sd` per-margin codes in native form:

- `:std_normal` — SB family 0, the unconfigured default (`tau ~ Normal(0, 1)`
  on the positive half; `param` ignored, conventionally `1.0`).
- `:exponential` — SB family 1 (`sd ~ Exponential(scale)` in the caller's
  spelling): `param` is the SCALE `θ` (the `:exponential` /
  `:positive_exponential` convention everywhere else in this contract — an
  SB `rate` inverts once at the boundary, `θ = 1 / rate`).
- `:normal` — SB family 2 (`sd ~ Normal(0, sd)`): `param` is the standard
  deviation `σ`.

An empty `sd_priors` vector (the default) is all-`:std_normal`: SB's
unconfigured prior, emitted exactly as before.
"""
struct VaryingSdPrior
    family::Symbol
    param::Float64
end

VaryingSdPrior(family::Symbol, param::Real) =
    VaryingSdPrior(family, Float64(param))

const _SD_PRIOR_FAMILIES = (:std_normal, :exponential, :normal)

"""
    VaryingDraws(group, kind, margins, lkj_eta, label, suffix[, levels[, sd_priors]])

One shared varying-effect draws block: non-centered geometry over
`K = length(margins)` margins in `G` groups of raw column `group`.
`kind` is `:intercept1` (single-`1` without eta: log-scale/xi
geometry), `:slope1` (single slope without eta: tau/xi geometry), or
`:correlated` (LKJ + tau + z_flat; K >= 2, or K = 1 with eta given —
the vacuous-1x1-LKJ route). `label` is the link identity
(`:draws_<suffix>`); `suffix` is the in-graph naming stem (the group,
or `group_binding` when two draws share a grouping). `levels` is the
grouping's DECLARED levels in numbering order (`nothing` pre-bind, or
when the emitter has no declaration — [`bind_data`](@ref) fills
sort-ordered observed levels). `sd_priors` is the per-margin `tau`
prior ([`VaryingSdPrior`](@ref); empty — the default — is all
`:std_normal`); a non-default entry needs `:correlated` draws (K = 1
takes the vacuous route by passing `eta`). Shared draws are consumed
by value flow: each [`VaryingSlice`](@ref) names this draws' label
plus explicit columns — never by label matching across statements.
"""
struct VaryingDraws
    group::ColumnRef
    kind::Symbol
    margins::Vector{VaryingMargin}
    lkj_eta::Float64
    label::Symbol
    suffix::String
    levels::Union{Nothing,Vector}
    sd_priors::Vector{VaryingSdPrior}
end

# Pre-sd-prior 6/7-arg positional construction keeps working with
# `sd_priors` empty (all-`:std_normal`, the unconfigured default).
VaryingDraws(group::ColumnRef, kind::Symbol, margins::Vector{VaryingMargin},
    lkj_eta::Float64, label::Symbol, suffix::String) =
    VaryingDraws(group, kind, margins, lkj_eta, label, suffix, nothing,
        VaryingSdPrior[])
VaryingDraws(group::ColumnRef, kind::Symbol, margins::Vector{VaryingMargin},
    lkj_eta::Float64, label::Symbol, suffix::String,
    levels::Union{Nothing,Vector}) =
    VaryingDraws(group, kind, margins, lkj_eta, label, suffix, levels,
        VaryingSdPrior[])

"""
    VaryingSlice(draws, columns, target)

One target application of a [`VaryingDraws`](@ref) block: `draws` names
the draws label, `columns` selects its explicit column range, `target`
is the predictor the contribution feeds. One slice per (draws, target);
a draws block's slices partition `1:K` exactly once, in any slice
order (columns are explicit, never positional by body order).
"""
struct VaryingSlice
    draws::Symbol
    columns::UnitRange{Int}
    target::Symbol
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
- `:positive_exponential` — a positive K-vector with an elementwise
  `Exponential(arg1)` prior (the joint factor's scales; `arg1` a
  finite positive literal or a scalar parameter/assignment name —
  sampled scale hyperparameters ride the scalar-prior shape).
- `:cholesky_corr_lkj` — a K×K LKJ Cholesky factor with an
  `lkj_corr_cholesky(arg1)` prior (`arg1` the shape hyperparameter,
  a finite positive literal), packing K(K−1)/2 thetas.

`size` is the constrained length (K−1 for thresholds, K for a simplex;
`nothing` = infer at bind from the linked leveled response). Args are
LITERALS only (hierarchical threshold/Dirichlet concentrations fail
closed — planned), except an exponential scale, which also admits a
scalar parameter/assignment name. K=1 is uniform: zero-length
threshold vectors carry no statements/prior/Jacobian, a 1-simplex is
the constant `[1.0]`, and a 1×1 LKJ factor packs zero thetas
(constraining to `[1.0]` with a `0.0` prior node — Stan's K=1 LKJ).
Joint-factor sizes are STRUCTURAL (`size` = the joint width K,
concrete — never `nothing`).
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
    LinearPKScheduleSpec(name, obs_subj, obs_time, dose_subj, dose_time, dose_amt,
                         ecg = nothing, tgi = nothing)

One linear-PK event-schedule declaration (grouped kernels): `name` the
schedule's model-scope handle; the remaining fields the RAW obs/dose
data columns the bind-time recipe builds the op stream from (surface:
`name = linear_pk_schedule(obs = (subj, time), dose = (subj, time,
amount))`). `ecg`/`tgi` are optional `(subject, time)` column pairs
naming EXTRA READ AXES (surface: `ecg = (subj, time)`, `tgi = (subj,
time)` keywords) whose rows join the stream as read-only points (SB's
joint `vcat(pk, qt, tgi)` union). The schedule materializes op columns
plus `op_ends`, per-axis `(obs_read, obs_map, ecg_read, ecg_map,
tgi_read, tgi_map)` row products, and the `[conc; auc]`-space
`conc_map`/`tgi_auc_map` at bind (see
[`build_linear_pk_schedule`](@ref)); the names here are declaration
only. Axis products materialize only for declared axes (and the
`[conc; auc]`-space maps only with an AUC cell call), so v1 bind
products stay byte-identical without them.
"""
struct LinearPKScheduleSpec
    name::Symbol
    obs_subj::Symbol
    obs_time::Symbol
    dose_subj::Symbol
    dose_time::Symbol
    dose_amt::Symbol
    ecg::Union{Nothing,Tuple{Symbol,Symbol}}
    tgi::Union{Nothing,Tuple{Symbol,Symbol}}
end
"""v1 positional construction (no extra read axes)."""
LinearPKScheduleSpec(name::Symbol, obs_subj::Symbol, obs_time::Symbol,
    dose_subj::Symbol, dose_time::Symbol, dose_amt::Symbol) =
    LinearPKScheduleSpec(name, obs_subj, obs_time, dose_subj, dose_time,
        dose_amt, nothing, nothing)

"""V2 event-LP slope prior scale (SB `effect(log_F, op_log_dose) ~
Normal(0.0, 0.6676)` verbatim — the specific overrides the wildcard
`effect(log_F, :) ~ Normal(0.0, 0.5)` by BRM specificity)."""
const _EVENT_LP_SLOPE_PRIOR_SD = 0.6676

"""V2 event-LP length-scale prior upper bound (SB
`length_scale(:, hsgp(op_log_dose)) ~ Uniform(lower, 2.0)` — the
varyingsource4 upper bound, verbatim)."""
const _EVENT_LP_RHO_PRIOR_HI = 2.0

"""The fixed event-LP provider name (the SB seam name — the
per-subject expansion slices the call arg by this NAME)."""
const EVENT_LP_NAME = :log_F

"""
    LinearPKEventLPSpec(name, schedule, k, c, fit, label)

One linear-PK event-axis bioavailability LP (SB `log_F ~ 0 +
op_log_dose + hsgp(op_log_dose; k = 5)`): `name` the provider's
model-scope flat vector (`:log_F`, fixed — the per-subject expansion
slices by name); `schedule` the op stream it rides; `k`/`c` the 1-D
HSGP modes/boundary factor (V2: 5/1.5); `fit` the bind-time `(mu, L)`
over the `op_log_dose` column (`nothing` pre-bind); `label` the
validation label. The provider evaluates in-graph via
[`linear_pk_event_log_f`](@ref) (data axis + frozen fit, traced
hyperparameters — the Stage-B split); the layout owns
`slope`/`rho`/`sigma`/`beta_raw` with the V2 priors.
"""
struct LinearPKEventLPSpec
    name::Symbol
    schedule::Symbol
    k::Int
    c::Float64
    fit::Union{Nothing,Tuple{Float64,Float64}}
    label::Symbol
end

# Event-LP sampled names, derived purely from the provider name (the
# `_hsgp_names` precedent, SB term-prior vocabulary): the dose slope,
# the HSGP length scale / marginal scale, and the standardized
# coefficients. Single source for name tables, layout, and the
# generator.
function _event_lp_names(el::LinearPKEventLPSpec)
    id = el.name
    return (slope = Symbol("slope_", id), rho = Symbol("rho_", id),
        sigma = Symbol("sigma_", id), beta = Symbol("beta_raw_", id))
end

"""Flat sampled-name list for name tables + claims (slope, rho, sigma,
beta — the `_hsgp_all_names` precedent)."""
function _event_lp_all_names(el::LinearPKEventLPSpec)
    n = _event_lp_names(el)
    return Symbol[n.slope, n.rho, n.sigma, n.beta]
end

"""In-cell observation node: `(response, family, location, scale, params)`
with `response` a cell param (panel and grouped share the node; grouped
carries a LIST). `location`/`scale` are the first/second positional
family args; `params` the remaining args (`()` for two-arg families).
Multi-param families (joint PK/QT/TGI) are grouped-only."""
const KernelObs =
    NamedTuple{(:response, :family, :location, :scale, :params)}

"""
    KernelPlate(result, subjects, timepoints, slices, assignments, obs, collected, label;
                lp_args = [], schedules = [])

One kernel (BRM `_RKKernelPlan`) in one of two forms:

PANEL (panel v1: `lp_args`/`schedules` empty): `result` is the collected
per-subject name (the plate LHS); `subjects` the subject count (integer
literal, or a dims-key `Symbol` resolved at bind); `timepoints` the
per-subject timepoint count (`nothing` for all-scalar models, an integer,
or a dims-key `Symbol` resolved at bind); `slices` the
`(data column, cell param, kind)` triples with `kind ∈ (:vector, :scalar,
:unknown)` (`:unknown` pre-bind — kinds resolve from lengths at bind);
`assignments` the cell-local `name => expr` pairs in cell order (flat
elementwise vocabulary); `obs` the single in-cell observation;
`collected` the trailing collected cell name. Grouping is ABSENT for
panel (implicit 1:n subjects — structural, no sentinel).

GROUPED (joint-kernel form: exactly one schedule in v1): `slices` are
RESPONSE slices (`kind === :response` once bound) over the schedule's
obs axis; `lp_args` the `(subject predictor, cell param)` pairs (LP
values gather per subject in-cell); `schedules` the schedule
declarations the cell calls address; `obs` the in-cell observation
LIST (one per response axis); `timepoints` is always `nothing`
(ragged axes have no rectangular T). The cell vocabulary is calls to
[`CELL_FNS`](@ref), schedule-map gathers, and arithmetic (no flat
dotify — the generator unrolls per subject).
"""
struct KernelPlate
    result::Symbol
    subjects::Union{Int,Symbol}
    timepoints::Union{Nothing,Int,Symbol}
    slices::Vector{Tuple{Symbol,Symbol,Symbol}}
    assignments::Vector{Pair{Symbol,Any}}
    obs::Vector{KernelObs}
    collected::Symbol
    label::Symbol
    lp_args::Vector{Tuple{Symbol,Symbol}}
    schedules::Vector{LinearPKScheduleSpec}
end

"""Panel-v1 positional construction (grouped fields default empty)."""
KernelPlate(result::Symbol, subjects::Union{Int,Symbol},
    timepoints::Union{Nothing,Int,Symbol},
    slices::Vector{Tuple{Symbol,Symbol,Symbol}},
    assignments::Vector{Pair{Symbol,Any}}, obs::KernelObs,
    collected::Symbol, label::Symbol) =
    KernelPlate(result, subjects, timepoints, slices, assignments, [obs],
        collected, label, Tuple{Symbol,Symbol}[], LinearPKScheduleSpec[])
KernelPlate(result::Symbol, subjects::Union{Int,Symbol},
    timepoints::Union{Nothing,Int,Symbol},
    slices::Vector{Tuple{Symbol,Symbol,Symbol}},
    assignments::Vector{Pair{Symbol,Any}}, obs::Vector{<:KernelObs},
    collected::Symbol, label::Symbol) =
    KernelPlate(result, subjects, timepoints, slices, assignments, obs,
        collected, label, Tuple{Symbol,Symbol}[], LinearPKScheduleSpec[])

"""Grouped ⟺ the plate carries schedules (panel plates carry none)."""
_is_grouped_kernel(kp::KernelPlate) = !isempty(kp.schedules)

"""Flat length of a resolved kernel plate: `n_sub * T` (`T = 1` all-scalar)."""
_kernel_flat_length(n_sub::Int, T::Union{Nothing,Int}) =
    T === nothing ? n_sub : n_sub * T

"""Materialized flat-expansion column for a scalar slice (spline-blocks
precedent: deterministic bind product, caller collisions fail closed)."""
_kexp_name(result::Symbol, col::Symbol) = Symbol("$(result)_kexp_$(col)")

"""Columns a kernel plate manages (slice columns + scalar expansions):
exempt from the uniform-`n_obs` rule, validated under kernel rules.
Expansions exist only for resolved vector models (`T isa Int`); gating
on that keeps a stray caller column from hiding behind the exemption.
Grouped plates additionally manage their schedules' raw columns, the
bind-materialized `<sched>_<field>` op columns, and the plain-Symbol
bind columns the cell references (gather indices + nadir ends — the
walker admits unknown Symbols ONLY in those positions, and shapes +
the prep validators prove each one, so the exemption cannot hide a
stray column)."""
function _kernel_managed_columns(kp::KernelPlate)
    out = Set{Symbol}()
    for (col, _, kind) in kp.slices
        push!(out, col)
        kind === :scalar && kp.timepoints isa Int &&
            push!(out, _kexp_name(kp.result, col))
    end
    for s in kp.schedules
        push!(out, s.obs_subj)
        push!(out, s.obs_time)
        push!(out, s.dose_subj)
        push!(out, s.dose_time)
        push!(out, s.dose_amt)
        s.ecg !== nothing && (push!(out, s.ecg[1]); push!(out, s.ecg[2]))
        s.tgi !== nothing && (push!(out, s.tgi[1]); push!(out, s.tgi[2]))
        for f in _SCHED_MATERIALIZED_FIELDS
            push!(out, _sched_col_name(s.name, f))
        end
        for f in _sched_extra_fields(s, kp.assignments)
            push!(out, _sched_col_name(s.name, f))
        end
        # The event-axis product (present only with a declared
        # event-LP): op-length by design, like the op columns.
        push!(out, _sched_col_name(s.name, :op_log_dose))
    end
    union!(out, _cell_bind_columns(kp))
    return out
end

"""Plain-Symbol bind columns a grouped cell references: gather indices
(`v[map]`) + segmented-nadir ends (`nadir(change, ends)`). The walker
admits unknown Symbols only in these positions; shapes prove
bound-ness + lengths and the prep validators prove content."""
function _cell_bind_columns(kp::KernelPlate)
    out = Set{Symbol}()
    for (_, ex) in kp.assignments
        _collect_cell_bind_columns!(out, ex)
    end
    return out
end

function _collect_cell_bind_columns!(out::Set{Symbol}, ex)
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 2 && ex.args[2] isa Symbol
        push!(out, ex.args[2])
    end
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] isa Symbol &&
            ex.args[1] in SEGMENT_CELL_FNS && ex.args[3] isa Symbol
        push!(out, ex.args[3])
    end
    for a in ex.args
        _collect_cell_bind_columns!(out, a)
    end
    return nothing
end

"""Bind-materialized schedule column (`sched` + field → `<sched>_<field>`,
the `_kexp_name` precedent: deterministic bind product, caller
collisions fail closed)."""
_sched_col_name(sched::Symbol, field::Symbol) = Symbol("$(sched)_$(field)")

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
    DarSpec(state, beta, sigma, label)

One differenced-AR(1) trajectory (SB `_sb_dar1`'s `differenced_ar1_path`):
the zero-started integrated path `x[t+1] = x[t] + d[t]` over the AR(1)
increments `d[t] = beta*d[t-1] + sigma*z[t]` (`d[0] = 0`, `x[1] = 0`).
`beta`/`sigma` name the persistence (`Normal` on `(:interval, 0, 1)`) and
innovation-scale (`Normal` on `:positive`) [`SampledParameter`](@ref)s;
the `z` innovations (length `n_obs - 1`) are owned internally under the
reserved `_ppl_dar_z_<state>` name, like a non-centered scan's
`_ppl_scan_z_<state>` slice. The path length is `n_obs` by construction
(the LP direct summand adds elementwise to an `n_obs` predictor), so no
length probe is carried — a `T ≠ n_obs` time axis needs the future
gathering extension (the `rw` free rider), not this node.

A dedicated node rather than a `ScanSpec`: the v1 scan grammar threads
one carried array from SAMPLED seeds with full-length innovations, while
dar starts both carries at zero deterministically, carries (level,
increment) jointly, and innovates `T - 1` times. The generator still
lowers through the shared RK-core `scan(...)` carry-fold tier.
"""
struct DarSpec
    state::Symbol
    beta::Symbol
    sigma::Symbol
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
    DesignMatrix(name, columns, label)

One user-bound design matrix (`X = hcat(1, x1, x2)`): `columns` in hcat
order, `nothing` marking intercept-ones positions. Data/derived columns
only (length-n by bind/construction); latent, scan, parameter, and
nested-matrix names are rejected — the SB `me` mirror stays affine and
nested `hcat` stays a follow-up. The generator emits each matrix once
(`X = Float64.(hcat(...))`); [`MatrixTerm`](@ref)s reference it by name
and splice `X * view(coef, ...)` matvecs. Width is static
(`length(columns)`); the surface sizes coefficient vectors from it.
"""
struct DesignMatrix
    name::Symbol
    columns::Vector{Union{Nothing,Symbol}}
    label::Symbol
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
`scans` sequential-recurrence latents, `dar_paths` differenced-AR(1)
trajectories, `varying_draws`/`varying_slices` the generic
varying-effect draws blocks plus their per-target applications, and
`matrices` user-bound design matrices referenced by [`MatrixTerm`](@ref)s
(empty for a plain population-GLM plan).
"""
struct StructuralPlan
    responses::Vector{LikelihoodSpec}
    predictors::Vector{PredictorSpec}
    population_priors::Vector{PopulationPrior}
    parameters::Vector{SampledParameter}
    assignments::Vector{AssignmentSpec}
    derived::Vector{VectorAssignmentSpec}
    columns::Dict{Symbol,ColumnData}
    n_obs::Int
    roles::Dict{Symbol,Symbol}
    levelmaps::Vector{LevelMap}
    plate_parameters::Vector{PlateParameter}
    scans::Vector{ScanSpec}
    dar_paths::Vector{DarSpec}
    varying_draws::Vector{VaryingDraws}
    varying_slices::Vector{VaryingSlice}
    vector_parameters::Vector{VectorParameter}
    spline_bases::Vector{SplineBasis}
    spline_vectors::Vector{SplineVector}
    hsgp_bases::Vector{HSGPBasis}
    kernel_plates::Vector{KernelPlate}
    r2d2_priors::Vector{R2D2Prior}
    matrices::Vector{DesignMatrix}
    event_lps::Vector{LinearPKEventLPSpec}
end

# Pre-extension full-positional constructor (9-arg): callers that built a plan
# before `levelmaps`/`scans`/`varying_draws`/`vector_parameters`/spline/hsgp
# nodes existed keep working with all empty.
StructuralPlan(
    responses::Vector{LikelihoodSpec},
    predictors::Vector{PredictorSpec},
    population_priors::Vector{PopulationPrior},
    parameters::Vector{SampledParameter},
    assignments::Vector{AssignmentSpec},
    derived::Vector{VectorAssignmentSpec},
    columns::AbstractDict{Symbol},
    n_obs::Int,
    roles::Dict{Symbol,Symbol}) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, _checked_columns(columns), n_obs, roles,
        LevelMap[], ScanSpec[], DarSpec[], VaryingDraws[], VaryingSlice[],
        VectorParameter[],
        SplineBasis[], SplineVector[], HSGPBasis[], KernelPlate[],
        R2D2Prior[], DesignMatrix[], LinearPKEventLPSpec[])

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
        columns::AbstractDict{Symbol},
        n_obs::Int;
        roles::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
        derived::Vector{VectorAssignmentSpec} = VectorAssignmentSpec[],
        levelmaps::Vector{LevelMap} = LevelMap[],
        plate_parameters::Vector{PlateParameter} = PlateParameter[],
        scans::Vector{ScanSpec} = ScanSpec[],
        dar_paths::Vector{DarSpec} = DarSpec[],
        varying_draws::Vector{VaryingDraws} = VaryingDraws[],
        varying_slices::Vector{VaryingSlice} = VaryingSlice[],
        vector_parameters::Vector{VectorParameter} = VectorParameter[],
        spline_bases::Vector{SplineBasis} = SplineBasis[],
        spline_vectors::Vector{SplineVector} = SplineVector[],
        hsgp_bases::Vector{HSGPBasis} = HSGPBasis[],
        kernel_plates::Vector{KernelPlate} = KernelPlate[],
        r2d2_priors::Vector{R2D2Prior} = R2D2Prior[],
        matrices::Vector{DesignMatrix} = DesignMatrix[],
        event_lps::Vector{LinearPKEventLPSpec} = LinearPKEventLPSpec[])
    return StructuralPlan(responses, predictors, population_priors,
        parameters, assignments, derived, _checked_columns(columns), n_obs,
        roles, levelmaps, plate_parameters, scans, dar_paths, varying_draws,
        varying_slices,
        vector_parameters, spline_bases, spline_vectors, hsgp_bases,
        kernel_plates, r2d2_priors, matrices, event_lps)
end

"""`combine_simultaneous` for one schedule's build: a declared
event-LP on the schedule selects the V2 no-pre-sum build (a
nonlinear dose-only effect makes pre-summing raw amounts invalid —
SB builds the joint schedule with `combine_simultaneous=false`);
otherwise the v1 pre-summing build."""
_schedule_combine_simultaneous(plan::StructuralPlan, sched::Symbol) =
    !any(el -> el.schedule === sched, plan.event_lps)

"""Find a design matrix by name, or `nothing`."""
function _find_matrix(plan::StructuralPlan, name::Symbol)
    i = findfirst(m -> m.name === name, plan.matrices)
    return i === nothing ? nothing : plan.matrices[i]
end

"""Per-element prior addressees of a design matrix in column order
(`:Intercept` at intercept positions, the column otherwise)."""
_matrix_element_addressees(m::DesignMatrix) =
    Symbol[c === nothing ? :Intercept : c for c in m.columns]

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
# assignment names. Grouped plates additionally introduce their LP cell
# params and schedule handles — EXCEPT self-aliasing params (`(c, c)`
# slices, `(pname, pname)` LP refs): those are lexical references to an
# outer column/definition (the plate spelling), not introductions.
# Single source for the global name-table gate.
function _kernel_all_names(kp::KernelPlate)
    names = Symbol[kp.result]
    for (c, p, _) in kp.slices
        p == c || push!(names, p)
    end
    for (pname, c) in kp.lp_args
        c == pname || push!(names, c)
    end
    for s in kp.schedules
        push!(names, s.name)
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

"""Slice-1 term-name vocabulary (emitter-side admission keys; `:varying_effect`,
`:spline_summand`, `:monotonic`, `:monotonic_summand`, and `:matrix` joined
with their slices)."""
const TERM_NAMES = Dict{Symbol,TermKind}(
    :intercept => InterceptTerm,
    :continuous => ContinuousTerm,
    :factor => FactorTerm,
    :offset => OffsetTerm,
    :varying_effect => VaryingEffectTerm,
    :spline_summand => SplineSummandTerm,
    :hsgp_summand => HSGPSummandTerm,
    :scan_summand => ScanSummandTerm,
    :monotonic => MonotonicTerm,
    :monotonic_summand => MonotonicSummandTerm,
    :matrix => MatrixTerm,
    :dar_summand => DarSummandTerm,
)

"""Allowlisted assignment functions (slice 1: scalar ops + whole-column
reductions; elementwise math over columns deferred with vector assignments;
the AR(1) slice adds `tanh` for the `phi = tanh(phi_raw)` stationarity map)."""
const ASSIGNMENT_FNS = (
    :+, :-, :*, :/, :^,
    :log, :log10, :log1p, :exp, :expm1, :sqrt, :abs, :tanh,
    :sum, :mean, :std, :var, :minimum, :maximum, :length,
)

"""Vector-returning whole-column functions (exact-GP slice): admitted in
derived columns and predictor locations only; always vector-shaped."""
const VECTOR_FNS = (:gp_exp_quad_cov, :gp_chol_latent)

"""Cell-callable functions (grouped kernels): admitted in grouped-kernel
cell assignments ONLY, always with a declared schedule as the first
argument. The generator emits ONE subject-batched call per assignment
(`<fn>_over_subjects` over the bound op columns + `op_ends`, pkcells.jl);
the function itself is the per-subject cell + host-side oracle path."""
const CELL_FNS = (:linear_pk_read_locs, :linear_pk_read_locs_auc)

"""Arity (argument count) of each [`CELL_FNS`](@ref) entry, schedule first."""
const CELL_FN_ARITY = Dict{Symbol,Int}(:linear_pk_read_locs => 6,
    :linear_pk_read_locs_auc => 7)

"""Op-column fields each [`CELL_FNS`](@ref) entry reads per subject
(positional, after the schedule — the generator passes the bound
`<sched>_<field>` columns to the batched runner, which slices them per
subject at runtime from `op_ends`)."""
const CELL_FN_OP_FIELDS = Dict{Symbol,Vector{Symbol}}(
    :linear_pk_read_locs => [:op_type, :op_dt, :op_amount, :op_interval,
        :op_count, :op_read_idx],
    :linear_pk_read_locs_auc => [:op_type, :op_dt, :op_amount, :op_interval,
        :op_count, :op_read_idx])

"""Call args (by NAME) each [`CELL_FNS`](@ref) entry slices per subject
from a flat op-ordered vector (emitted as `SubjectSlice(name)` — the
batched runner takes `view(name, lo:hi)` over the subject's op range —
for computed event-frame vectors like the W2 `log_F` provider output,
which are generated-code locals, not bind columns)."""
const CELL_FN_SLICED_ARGS = Dict{Symbol,Vector{Symbol}}(
    :linear_pk_read_locs => [:log_F],
    :linear_pk_read_locs_auc => [:log_F])

"""Segmented-scan cell calls: per-subject scans over a row series with
an explicit cumulative-ends vector (no schedule — the nadir runs over
an obs-axis series, not the op stream). Only the nadir wires into the
grouped walker (the rest of `TGI_CELL_FUNCTIONS` stays out — the
joint cell spells latents as elementwise lines)."""
const SEGMENT_CELL_FNS = (:tgi_segmented_nadir,)
"""Arity past the function name: change vector + ends column."""
const SEGMENT_CELL_FN_ARITY =
    Dict{Symbol,Int}(:tgi_segmented_nadir => 2)

"""Schedule-map gathers a grouped cell may index (`reads[sched.obs_map]`,
one static int vector per row — the `reactivekernels-use` §7d
vectorized-gather shape): the per-axis conc-space maps (`obs_map`,
`ecg_map`, `tgi_map`) plus the `[conc; auc]`-space `conc_map` and
`tgi_auc_map` (see [`build_linear_pk_schedule`](@ref)). Each map is
available only when its axis/cell prerequisite holds (see
[`_sched_available_maps`](@ref)) — undeclared-axis gathers fail closed
at structure."""
const SCHEDULE_MAPS = (:obs_map, :ecg_map, :tgi_map, :conc_map, :tgi_auc_map)

"""Bind-materialized columns per schedule (`<sched>_<field>`): the six
op columns plus `op_ends`, per-obs-row `obs_read`, and the flat
`obs_map` gather index (see [`build_linear_pk_schedule`](@ref))."""
const _SCHED_MATERIALIZED_FIELDS =
    (:op_type, :op_dt, :op_amount, :op_interval, :op_count, :op_read_idx,
        :op_ends, :obs_read, :obs_map)

"""Per-axis row products, materialized only for declared extra axes."""
const _SCHED_ECG_FIELDS = (:ecg_read, :ecg_map)
const _SCHED_TGI_FIELDS = (:tgi_read, :tgi_map)

"""`[conc; auc]`-space maps, materialized only with an AUC cell call
(`tgi_auc_map` additionally needs the declared tgi axis)."""
const _SCHED_CONC_FIELDS = (:conc_map,)
const _SCHED_TGI_AUC_FIELDS = (:tgi_auc_map,)
"""Cumulative TGI row ends, materialized only with a nadir cell call
plus the declared tgi axis."""
const _SCHED_SEG_ENDS_FIELDS = (:tgi_seg_ends,)

"""Whether any top-level cell assignment calls the AUC recurrence
(the generator expands top-level calls only, so the scan matches it
exactly — nested calls mis-generate regardless)."""
function _cell_has_auc_call(assignments::Vector{Pair{Symbol,Any}})
    for (_, ex) in assignments
        ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] === :linear_pk_read_locs_auc && return true
    end
    return false
end

"""Whether any top-level cell assignment calls the segmented nadir
(same top-level-only reading as [`_cell_has_auc_call`](@ref))."""
function _cell_has_nadir_call(assignments::Vector{Pair{Symbol,Any}})
    for (_, ex) in assignments
        ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] isa Symbol && ex.args[1] in SEGMENT_CELL_FNS &&
            return true
    end
    return false
end

"""Bind-materialized schedule products beyond [`_SCHED_MATERIALIZED_FIELDS`](@ref):
per-axis products for declared extra axes, `conc_map` with an AUC
cell call, `tgi_auc_map` with an AUC cell call plus the declared tgi
axis, `tgi_seg_ends` with a nadir cell call plus the declared tgi
axis. Single source for bind materialization, exact-rebuild
verification, and managed columns."""
function _sched_extra_fields(sched::LinearPKScheduleSpec,
        assignments::Vector{Pair{Symbol,Any}})
    out = Symbol[]
    sched.ecg !== nothing && append!(out, _SCHED_ECG_FIELDS)
    sched.tgi !== nothing && append!(out, _SCHED_TGI_FIELDS)
    if _cell_has_auc_call(assignments)
        append!(out, _SCHED_CONC_FIELDS)
        sched.tgi !== nothing && append!(out, _SCHED_TGI_AUC_FIELDS)
    end
    if sched.tgi !== nothing && _cell_has_nadir_call(assignments)
        append!(out, _SCHED_SEG_ENDS_FIELDS)
    end
    return out
end

"""Schedule maps available to a cell: `obs_map` always, per-axis maps
for declared extra axes, `conc_map` with an AUC cell call,
`tgi_auc_map` with an AUC cell call plus the declared tgi axis."""
function _sched_available_maps(sched::LinearPKScheduleSpec,
        assignments::Vector{Pair{Symbol,Any}})
    maps = Set{Symbol}([:obs_map])
    sched.ecg !== nothing && push!(maps, :ecg_map)
    sched.tgi !== nothing && push!(maps, :tgi_map)
    if _cell_has_auc_call(assignments)
        push!(maps, :conc_map)
        sched.tgi !== nothing && push!(maps, :tgi_auc_map)
    end
    return maps
end

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
    OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam,
    MvNormalCholeskyFam, NormalIDGLMFam, BernoulliLogitGLMFam,
    PoissonLogGLMFam)

"""Term kinds the thin layer can lower (ext handshake predicate)."""
admitted_terms() = (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm,
    VaryingEffectTerm, SplineSummandTerm, HSGPSummandTerm,
    ScanSummandTerm, MonotonicTerm, MonotonicSummandTerm, MatrixTerm,
    DarSummandTerm)

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
    _validate_dar_paths(plan)
    _validate_assignments_structure(plan)
    _validate_vector_structure(plan)
    _validate_parameters(plan)
    _validate_plate_parameters(plan)
    _validate_vector_parameters(plan)
    _validate_topo_order(plan)
    _validate_matrices(plan)
    _validate_predictors(plan)
    _validate_levelmaps(plan)
    _validate_priors(plan)
    _validate_r2d2(plan)
    _validate_kernels(plan)
    _validate_responses(plan)
    _validate_varying_draws(plan)
    _validate_splines(plan)
    _validate_hsgp(plan)
    _validate_event_lps(plan)
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
    _validate_varying_draws_data(plan)
    _validate_splines_data(plan)
    _validate_hsgp_data(plan)
    _validate_kernels_data(plan)
    _validate_r2d2_data(plan)
    _validate_event_lp_data(plan)
    return nothing
end

_is_ones_margin(m::VaryingMargin) =
    m.z.kind === :ones && m.coefficient === :Intercept

function _validate_margin(m::VaryingMargin, label::Symbol)
    z = m.z
    z.kind in (:ones, :column, :dummy) ||
        _fail(label, "margin $(m.coefficient): Z recipe kind must be " *
              ":ones, :column, or :dummy, got $(repr(z.kind))")
    if z.kind === :ones
        z.column === :none && z.level === nothing ||
            _fail(label, "margin $(m.coefficient): :ones recipe carries " *
                  "no column/level")
        m.coefficient === :Intercept ||
            _fail(label, "margin $(m.coefficient): :ones recipe addresses " *
                  ":Intercept")
    elseif z.kind === :column
        z.level === nothing ||
            _fail(label, "margin $(m.coefficient): :column recipe carries " *
                  "no level")
        m.coefficient === z.column ||
            _fail(label, "margin $(m.coefficient): :column recipe " *
                  "addresses its column $(z.column)")
    else
        z.level !== nothing ||
            _fail(label, "margin $(m.coefficient): :dummy recipe needs " *
                  "a level value")
    end
    return nothing
end

# K=1 sampled names, derived purely from the draws suffix + kind: the
# scalar scale (`log_scale_<s>` / `tau_<s>`) and the G-vector
# (`xi_<s>`). Single source for surface claims, name tables, layout,
# and the generator.
# Correlated sampled names, derived purely from the draws suffix: the
# LKJ Cholesky factor (`L_<s>`, KxK), the marginal-scale vector
# (`tau_<s>`, K), and the standardized draws (`z_flat_<s>`, K*G
# column-major). Single source for surface claims, name tables,
# layout, and the generator. K=1 correlated draws own the same three
# names (`L` packs zero coords).
function _varying_corr_names(d::VaryingDraws)
    d.kind === :correlated ||
        _fail(d.label, "draws kind $(d.kind) owns no correlated " *
              "sampled names (K=1 geometry has its own names)")
    s = d.suffix
    return (Symbol("L_", s), Symbol("tau_", s), Symbol("z_flat_", s))
end

function _varying_k1_names(d::VaryingDraws)
    (d.kind === :intercept1 || d.kind === :slope1) ||
        _fail(d.label, "draws kind $(d.kind) owns no K=1 sampled names " *
              "(correlated draws own L/tau/z names instead)")
    s = d.suffix
    scale = d.kind === :intercept1 ? Symbol("log_scale_", s) :
        Symbol("tau_", s)
    return (scale, Symbol("xi_", s))
end

# Draws labels/suffixes, kinds, margins, slices, and effect-term
# linkage: everything provable without data. A draws block's slice
# ranges partition 1:K exactly once, in any slice order (columns are
# explicit); every slice is consumed by exactly one effect term (a
# dangling slice samples dead parameters).
function _validate_varying_draws(plan::StructuralPlan)
    draws = plan.varying_draws
    slices = plan.varying_slices
    labels = [d.label for d in draws]
    length(unique(labels)) == length(labels) ||
        _fail(:plan, "duplicate varying draws labels (one label per draws block)")
    suffixes = [d.suffix for d in draws]
    length(unique(suffixes)) == length(suffixes) ||
        _fail(:plan, "duplicate varying draws suffixes (in-graph names " *
              "derive from the suffix)")
    prednames = Set{Symbol}(p.name for p in plan.predictors)
    bylabel = Dict{Symbol,VaryingDraws}(d.label => d for d in draws)
    for d in draws
        _validate_draws_shape(d, prednames, slices)
    end
    for s in slices
        haskey(bylabel, s.draws) ||
            _fail(:plan, "varying slice for $(s.target) names unknown " *
                  "draws $(s.draws)")
        s.target in prednames ||
            _fail(bylabel[s.draws].label, "varying slice names unknown " *
                  "predictor $(s.target)")
    end
    # Slice ranges partition 1:K exactly once per draws (sorted: slice
    # order is free, columns are explicit), one slice per
    # (draws, target), and no range is empty.
    for d in draws
        K = length(d.margins)
        own = [s for s in slices if s.draws === d.label]
        targets = [s.target for s in own]
        length(unique(targets)) == length(targets) ||
            _fail(d.label, "draws lists a target twice (one slice per " *
                  "(draws, target))")
        for s in own
            r = s.columns
            first(r) <= last(r) ||
                _fail(d.label, "slice for $(s.target) is empty (each " *
                      "slice carries at least one margin)")
            (first(r) >= 1 && last(r) <= K) ||
                _fail(d.label, "slice for $(s.target) selects $r outside " *
                      "1:$K")
        end
        lo = 1
        for s in sort!(own; by = s -> first(s.columns))
            first(s.columns) == lo ||
                _fail(d.label, "slice for $(s.target) starts at " *
                      "$(first(s.columns)), want $lo (slices partition " *
                      "1:$K exactly once)")
            lo = last(s.columns) + 1
        end
        lo - 1 == K ||
            _fail(d.label, "slices cover $(lo - 1) margins but the draws " *
                  "block carries $K")
    end
    # Effect-term linkage, jointly over predictors + draws + slices.
    for pred in plan.predictors
        for t in pred.terms
            t.kind === VaryingEffectTerm || continue
            o = t.options
            haskey(bylabel, o.draws) ||
                _fail(t.label, "effect term in predictor $(pred.name) " *
                      "references unknown draws $(o.draws)")
            any(s -> s.draws === o.draws && s.target === pred.name,
                slices) ||
                _fail(t.label, "draws $(o.draws) carries no slice for " *
                      "predictor $(pred.name)")
        end
    end
    used = Set{Tuple{Symbol,Symbol}}()
    for pred in plan.predictors
        for t in pred.terms
            t.kind === VaryingEffectTerm || continue
            key = (t.options.draws, pred.name)
            key in used &&
                _fail(t.label, "duplicate effect term for draws " *
                      "$(key[1]) in predictor $(pred.name) (one term per " *
                      "draws per predictor)")
            push!(used, key)
        end
    end
    for s in slices
        (s.draws, s.target) in used ||
            _fail(bylabel[s.draws].label, "draws $(s.draws) slice for " *
                  "predictor $(s.target) is never consumed (dangling " *
                  "slice samples dead parameters — use it or drop it)")
    end
    return nothing
end

function _validate_draws_shape(d::VaryingDraws, prednames::Set{Symbol},
        slices::Vector{VaryingSlice})
    d.kind in (:intercept1, :slope1, :correlated) ||
        _fail(d.label, "draws kind must be :intercept1, :slope1, or " *
              ":correlated, got $(repr(d.kind))")
    K = length(d.margins)
    K >= 1 || _fail(d.label, "draws block has zero margins")
    for m in d.margins
        _validate_margin(m, d.label)
    end
    for s in slices
        s.draws === d.label || continue
        s.target in prednames ||
            _fail(d.label, "draws slice for unknown predictor $(s.target)")
    end
    # Kind dispatch: K=1 without eta is :intercept1 (single-`1`) or
    # :slope1 (single slope); K=1 with eta takes the vacuous-1x1-LKJ
    # correlated route; K>=2 is always :correlated.
    want = if K == 1 && isnan(d.lkj_eta) && _is_ones_margin(first(d.margins))
        :intercept1
    elseif K == 1 && isnan(d.lkj_eta)
        :slope1
    else
        :correlated
    end
    d.kind === want ||
        _fail(d.label, "draws kind $(d.kind) mismatches its margins " *
              "(want $want)")
    if want === :correlated
        isfinite(d.lkj_eta) && d.lkj_eta > 0 ||
            _fail(d.label, "correlated draws need a positive LKJ eta, " *
                  "got $(d.lkj_eta)")
    else
        isnan(d.lkj_eta) ||
            _fail(d.label, "K=1 draws take no LKJ eta (no correlation " *
                  "to parameterize), got $(d.lkj_eta)")
    end
    _validate_sd_priors(d, K)
    _validate_varying_levels_shape(d)
    return nothing
end

# Per-margin `tau` priors: empty (the default) is all-`:std_normal`,
# otherwise one entry per margin in margin order. `:exponential` takes
# a finite positive SCALE, `:normal` a finite positive sd;
# `:std_normal` ignores its param (finite, conventionally 1.0). A
# non-default entry needs `:correlated` draws — SB's sd overrides are
# a correlated-draws feature (no K=1 intercept/slope override path
# exists to mirror), so K = 1 without eta fails closed naming the
# vacuous route.
function _validate_sd_priors(d::VaryingDraws, K::Int)
    sds = d.sd_priors
    (isempty(sds) || length(sds) == K) ||
        _fail(d.label, "draws list $(length(sds)) sd priors for $K " *
              "margins (empty for all-default, else exactly one per margin)")
    for (j, p) in enumerate(sds)
        p.family in _SD_PRIOR_FAMILIES ||
            _fail(d.label, "margin $j sd prior family must be one of " *
                  "$(_SD_PRIOR_FAMILIES), got $(repr(p.family))")
        isfinite(p.param) ||
            _fail(d.label, "margin $j sd prior param is not finite " *
                  "(got $(p.param))")
        if p.family !== :std_normal
            p.param > 0 ||
                _fail(d.label, "margin $j sd prior needs a positive " *
                      "param, got $(p.param)")
            d.kind === :correlated ||
                _fail(d.label, "margin $j carries an explicit sd prior " *
                      "but these draws are :$(d.kind) (sd priors are a " *
                      ":correlated-draws feature — pass `eta` for the " *
                      "vacuous-1x1-LKJ route)")
        end
    end
    return nothing
end

# Declared-levels checks provable without data (coverage needs the bound
# column — `_validate_varying_draws_data`). Emitter-provided levels must
# be non-empty, duplicate-free, and literal-embeddable (the admission
# mirrors `_level_literal` in preprocessing.jl — the generator embeds
# these values into the `_declared_codes` call).
function _validate_varying_levels_shape(d::VaryingDraws)
    d.levels === nothing && return nothing
    !isempty(d.levels) ||
        _fail(d.label, "draws block declares zero grouping levels")
    length(unique(d.levels)) == length(d.levels) ||
        _fail(d.label, "draws block declares duplicate grouping levels " *
              "($(repr(d.levels)))")
    for lv in d.levels
        lv isa Union{Number,String,Bool,Char,Symbol} ||
            _fail(d.label, "declared level $(repr(lv)) is not " *
                  "literal-embeddable (numeric/string/symbol only)")
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

function _validate_margin_data(m::VaryingMargin, label::Symbol,
        plan::StructuralPlan)
    z = m.z
    z.kind === :ones && return nothing
    haskey(plan.columns, z.column) || _is_derived(plan, z.column) ||
        _fail(label, "margin $(m.coefficient): Z column $(z.column) is " *
              "not bound")
    if z.kind === :column
        _is_derived(plan, z.column) && return nothing
        zcol = _vector_column(plan.columns, z.column, label, "Z column")
        eltype(zcol) <: Real ||
            _fail(label, "margin $(m.coefficient): Z column $(z.column) " *
                  "must be numeric (a categorical slope needs explicit " *
                  "`dummy($(z.column), k)` recipes)")
    else
        _is_derived(plan, z.column) &&
            _fail(label, "margin $(m.coefficient): :dummy needs a raw " *
                  "column (level membership needs bound values)")
        zcol = _vector_column(plan.columns, z.column, label, "grouping column")
        z.level in _grouping_levels(zcol) ||
            _fail(label, "margin $(m.coefficient): dummy level " *
                  "$(repr(z.level)) is not a level of $(z.column)")
    end
    return nothing
end

# Grouping columns are raw (level knowledge needs values), Z continuous
# columns mirror the population rule (raw numeric or derived, whose eltype
# is unknown statically), and dummy levels must be members of the column's
# grouping levels (Int value / string exact match).
function _validate_varying_draws_data(plan::StructuralPlan)
    for d in plan.varying_draws
        haskey(plan.columns, d.group) ||
            _fail(d.label, "grouping column $(d.group) is not bound")
        _is_derived(plan, d.group) &&
            _fail(d.label, "grouping column $(d.group) must be raw data " *
                  "(level knowledge needs bound values)")
        d.levels === nothing &&
            _fail(d.label, "draws block has no declared grouping levels " *
                  "(bind_data fills these — hand-built bound plans must too)")
        # Coverage: every observed value needs a declared code (an
        # uncovered value would encode 0 and gather out of bounds).
        levels = d.levels::Vector
        groupcol =
            _vector_column(plan.columns, d.group, d.label, "grouping column")
        for v in groupcol
            v in levels ||
                _fail(d.label, "grouping value $(repr(v)) of $(d.group) " *
                      "is not a declared level (declared: $(repr(levels)))")
        end
        for m in d.margins
            _validate_margin_data(m, d.label, plan)
        end
    end
    # One grouping, one numbering: same-group draws share the per-group
    # `_ppl_gidx_` encoder, so their declared levels must agree exactly
    # (order included — codes are positions).
    for i in eachindex(plan.varying_draws)
        for j in (i + 1):length(plan.varying_draws)
            di, dj = plan.varying_draws[i], plan.varying_draws[j]
            di.group === dj.group || continue
            di.levels == dj.levels ||
                _fail(:plan, "draws $(di.label) and $(dj.label) share " *
                      "grouping $(di.group) but declare different levels " *
                      "($(repr(di.levels)) vs $(repr(dj.levels)))")
        end
    end
    return nothing
end

# Group count for a draws block: the DECLARED level count (unobserved
# declared levels keep prior-only coefficients). Loud defense in
# depth — validate_data proves levels non-nothing on every bound plan.
function _draws_nlevels(d::VaryingDraws)
    d.levels === nothing && throw(ContractValidationError(
        "[layout] draws $(d.label) has no declared grouping levels " *
        "(bind_data fills these — hand-built bound plans must too)"))
    return length(d.levels)
end

# Binder evaluation for draws levels (the LevelMap precedent):
# `nothing` fills sort-ordered observed levels; emitter-provided levels
# pass through (validated by `_validate_varying_levels_shape` +
# `_validate_varying_draws_data`).
function _eval_draws_levels(draws::Vector{VaryingDraws},
        columns::AbstractDict{Symbol})
    out = VaryingDraws[]
    for d in draws
        d.levels !== nothing && (push!(out, d); continue)
        haskey(columns, d.group) ||
            _fail(d.label, "grouping column $(d.group) is not bound")
        groupcol = _vector_column(columns, d.group, d.label, "grouping column")
        levels =
            try
                _grouping_levels(groupcol)
            catch err
                _fail(d.label, "grouping column $(d.group) levels not " *
                             "orderable ($err)")
            end
        push!(out, VaryingDraws(d.group, d.kind, d.margins, d.lkj_eta,
            d.label, d.suffix, collect(levels), d.sd_priors))
    end
    return out
end

"""Predictor levels (grouped kernels): a predictor consumed ONLY as a
kernel LP arg is SUBJECT-level (its design rows are subjects); any
response use makes it obs-level. Mixed use fails closed — split the
predictor instead."""
function _predictor_level(plan::StructuralPlan, pname::Symbol)
    by_kernel = any(kp -> any(((p, _),) -> p === pname, kp.lp_args),
        plan.kernel_plates)
    by_resp = any(r -> _response_uses_predictor(r, pname), plan.responses)
    by_kernel && by_resp &&
        _fail(:plan, "predictor `$pname` feeds both a kernel LP arg and " *
              "a response (mixed-level predictors are not supported — " *
              "split it into a subject-level and an obs-level predictor)")
    by_kernel && return :subject
    return :obs
end

function _response_uses_predictor(r::LikelihoodSpec, pname::Symbol)
    r.predictor === pname && return true
    pname in r.extra_predictors && return true
    r.scale isa ScalePredictorRef && r.scale.predictor === pname &&
        return true
    return false
end

"""Term columns of subject-level predictors (n_sub rows by design):
exempt from the uniform-`n_obs` rule like kernel-managed columns (only
bound columns count — a summand's basis id is not a column and fails
loudly under kernel rules instead). Transitive through derived
assignments: a data column feeding ONLY subject-level consumers inherits
subject level (in-graph `standardize`, etc.). A pulled column that ALSO
feeds obs-level consumers (responses, slices, obs predictors, weights,
evidence, trials) fails loudly — mixed-level lengths are genuinely
ambiguous. (Directly-shared columns across levels predate this rule and
keep their historical behavior; varying-group sharing across levels is
out of scope.)"""
function _subject_predictor_columns(plan::StructuralPlan)
    out = Set{Symbol}()
    seed = Set{Symbol}()
    for pred in plan.predictors
        _predictor_level(plan, pred.name) === :subject || continue
        for t in pred.terms, c in t.columns
            push!(seed, c)
            haskey(plan.columns, c) && push!(out, c)
        end
    end
    if !isempty(seed)
        deps = _assignment_name_deps(plan)
        # Fixpoint: subject names (data or not-yet-materialized deriveds)
        # pull their assignment sources in; bound data sources join `out`.
        queue = collect(seed)
        seen = copy(seed)
        while !isempty(queue)
            for src in get(deps, pop!(queue), ())
                src in seen && continue
                push!(seen, src)
                haskey(plan.columns, src) && push!(out, src)
                push!(queue, src)
            end
        end
        # Mixed-level: pulled data columns feeding obs-level consumers.
        obs = Set{Symbol}()
        for r in plan.responses
            push!(obs, r.response)
            r.weights isa Symbol && push!(obs, r.weights)
            r.trials isa Symbol && push!(obs, r.trials)
            if r.evidence !== nothing && r.evidence.kind !== :none
                r.evidence.lower isa Symbol && push!(obs, r.evidence.lower)
                r.evidence.upper isa Symbol && push!(obs, r.evidence.upper)
            end
        end
        for kp in plan.kernel_plates, (c, _, _) in kp.slices
            push!(obs, c)
        end
        for pred in plan.predictors
            _predictor_level(plan, pred.name) === :obs || continue
            for t in pred.terms, c in t.columns
                push!(obs, c)
            end
        end
        for c in out
            c in obs &&
                _fail(c, "column feeds both subject-level and obs-level " *
                      "consumers (mixed-level lengths are ambiguous — " *
                      "bind explicit per-level copies)")
        end
    end
    return out
end

"""Assignment name dependencies: derived/assignment name → referenced
names (function heads included — callers intersect with bound columns)."""
function _assignment_name_deps(plan::StructuralPlan)
    deps = Dict{Symbol,Set{Symbol}}()
    for a in plan.assignments
        deps[a.name] = _expr_names(a.expr)
    end
    for d in plan.derived
        deps[d.name] = _expr_names(d.expr)
    end
    return deps
end

function _expr_names(ex)::Set{Symbol}
    out = Set{Symbol}()
    _expr_names!(out, ex)
    return out
end

function _expr_names!(out::Set{Symbol}, ex)
    ex isa Symbol && (push!(out, ex); return nothing)
    ex isa Expr || return nothing
    for a in ex.args
        _expr_names!(out, a)
    end
    return nothing
end

function _validate_columns(plan::StructuralPlan)
    plan.n_obs > 0 || _fail(:plan, "n_obs must be positive, got $(plan.n_obs)")
    managed = Set{Symbol}()
    for kp in plan.kernel_plates
        union!(managed, _kernel_managed_columns(kp))
    end
    union!(managed, _subject_predictor_columns(plan))
    for (name, col) in plan.columns
        # Kernel-managed columns (slices + scalar expansions) carry two
        # lengths by design — they validate under kernel rules, not here.
        if col isa AbstractMatrix
            size(col, 1) == plan.n_obs ||
                _fail(name, "matrix column has $(size(col, 1)) rows ≠ " *
                      "n_obs $(plan.n_obs)")
            size(col, 2) >= 1 ||
                _fail(name, "design matrix has 0 columns " *
                      "(bind ≥ 1 predictor column)")
            eltype(col) <: Real ||
                _fail(name, "matrix column must be numeric, " *
                      "got $(eltype(col))")
        else
            name in managed || length(col) == plan.n_obs ||
                _fail(name, "column length $(length(col)) ≠ n_obs $(plan.n_obs)")
        end
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
            axiscol =
                _vector_column(plan.columns, c, sb.label, "spline axis column")
            eltype(axiscol) <: Real ||
                _fail(sb.label, "spline :$(sb.id): axis column $c must " *
                      "be numeric, got $(eltype(axiscol))")
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
            axiscol =
                _vector_column(plan.columns, c, hb.label, "hsgp axis column")
            eltype(axiscol) <: Real ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c must " *
                      "be numeric, got $(eltype(axiscol))")
        end
        length(hb.fits) == length(hb.axes) ||
            _fail(hb.label, "hsgp :$(hb.id): fits not filled at bind " *
                  "(one (mu, L) per axis)")
    end
    return nothing
end

# Event-LP structure (see `_validate_event_lps`): fixed provider
# name, declared schedule (at most one LP per schedule), admitted
# k/c, sane hand-built fits, and call linkage (a declared LP feeds
# at least one 7-arg cell call on its own schedule; a 7-arg call
# needs its schedule's LP — an unbound `log_F` local must never
# reach codegen).
function _validate_event_lps(plan::StructuralPlan)
    els = plan.event_lps
    names = [el.name for el in els]
    length(unique(names)) == length(names) ||
        _fail(:plan, "duplicate event-LP names")
    labels = [el.label for el in els]
    length(unique(labels)) == length(labels) ||
        _fail(:plan, "duplicate event-LP labels")
    scheds = [s.name for kp in plan.kernel_plates for s in kp.schedules]
    for el in els
        el.name === EVENT_LP_NAME ||
            _fail(el.label, "event-LP name must be `$(EVENT_LP_NAME)` " *
                  "(the per-subject expansion slices the call arg by " *
                  "name — one seam, got `$(el.name)`)")
        el.schedule in scheds ||
            _fail(el.label, "event-LP `$(el.name)` schedule " *
                  "`$(el.schedule)` is not declared " *
                  "(`$(el.schedule) = linear_pk_schedule(...)`)")
        el.k isa Int && el.k >= 2 ||
            _fail(el.label, "event-LP `$(el.name)` k must be an integer " *
                  "≥ 2 (the truncation floor needs k²−1 > 0; " *
                  "V2: k = 5, got $(repr(el.k)))")
        el.c isa Real && isfinite(Float64(el.c)) && Float64(el.c) > 1 ||
            _fail(el.label, "event-LP `$(el.name)` c must be finite and " *
                  "exceed 1 (V2: c = 1.5, got $(repr(el.c)))")
        if el.fit !== nothing
            mu, L = el.fit
            isfinite(mu) && isfinite(L) && L > 0 ||
                _fail(el.label, "event-LP `$(el.name)` fit (mu, L) must " *
                      "be finite with L > 0, got ($(mu), $(L))")
        end
    end
    # (At most one LP per schedule needs no separate check: the fixed
    # provider name makes the name-table duplicate rule subsume it —
    # W2 admits a single event-LP per model.)
    # Call linkage over grouped-kernel assignments: the cell walker
    # owns precise 7-arg shape rejection; here every 7-arg call needs
    # its schedule's declared LP, and every declared LP needs a call.
    calls = Tuple{Symbol,Symbol}[]
    for kp in plan.kernel_plates, (_, ex) in kp.assignments
        _collect_event_lp_calls!(calls, ex)
    end
    for (sched, arg) in calls
        any(el -> el.name === arg && el.schedule === sched, els) ||
            _fail(:plan, "cell call threads event-LP `$arg` on " *
                  "schedule `$sched`, which is not declared " *
                  "(`$arg = linear_pk_log_f($sched; k = 5)`)")
    end
    for el in els
        any(((s, a),) -> s === el.schedule && a === el.name, calls) ||
            _fail(el.label, "event-LP `$(el.name)` is never called " *
                  "(declared LPs must feed a 7-arg cell call — " *
                  "unfed LPs would sample dead parameters)")
    end
    return nothing
end

# 7-arg event-LP cell calls a grouped cell makes (schedule, arg2):
# lenient collection — the cell walker owns precise rejection. Eight
# expr args (fn + schedule + log_F + 5 LPs) is the literal event form
# (NOT arity-relative: the AUC sibling shares the shape under its own
# arity, and a relative rule would miss it).
function _collect_event_lp_calls!(calls::Vector{Tuple{Symbol,Symbol}}, ex)
    ex isa Expr || return nothing
    if ex.head === :call && length(ex.args) == 8 &&
            ex.args[1] isa Symbol && ex.args[1] in CELL_FNS &&
            ex.args[2] isa Symbol && ex.args[3] isa Symbol
        push!(calls, (ex.args[2], ex.args[3]))
    end
    for a in ex.args
        _collect_event_lp_calls!(calls, a)
    end
    return nothing
end

# Event-LP data: the op_log_dose bind product verified by exact
# rebuild (never trusted — the schedule precedent), the fit filled
# and rebuild-identical, and the truncation floor below the V2
# prior's upper bound (SB `_linear_pk_hsgp_lower` errors the same
# way).
function _validate_event_lp_data(plan::StructuralPlan)
    for el in plan.event_lps
        col = _sched_col_name(el.schedule, :op_log_dose)
        haskey(plan.columns, col) ||
            _fail(el.label, "event-LP `$(el.name)` product `$col` " *
                  "missing (bind_data materializes the event axis)")
        tcol = _sched_col_name(el.schedule, :op_type)
        acol = _sched_col_name(el.schedule, :op_amount)
        (haskey(plan.columns, tcol) && haskey(plan.columns, acol)) ||
            _fail(el.label, "event-LP `$(el.name)` schedule products " *
                  "missing (bind_data materializes op columns)")
        want = try
            linear_pk_op_log_dose(plan.columns[tcol], plan.columns[acol])
        catch err
            err isa ContractValidationError &&
                _fail(el.label, "event-LP `$(el.name)`: $(err.message)")
            rethrow()
        end
        plan.columns[col] == want ||
            _fail(el.label, "event-LP `$(el.name)` product `$col` is " *
                  "not the event-axis build (bind_data materializes " *
                  "it — a hand-bound plan must carry the identical " *
                  "product)")
        el.fit === nothing &&
            _fail(el.label, "event-LP `$(el.name)` fit not filled at " *
                  "bind (one (mu, L) over `$col`)")
        mu, L = el.fit
        rmu, rL = _hsgp_axis_fit(plan.columns[col], el.c, el.label,
            "event-LP `$(el.name)` fit")
        (mu, L) == (rmu, rL) ||
            _fail(el.label, "event-LP `$(el.name)` fit ($mu, $L) is " *
                  "not the bind fit ($rmu, $rL)")
        floor = only(_hsgp_floors([el.k], [(mu, L)], true))
        floor < _EVENT_LP_RHO_PRIOR_HI ||
            _fail(el.label, "event-LP `$(el.name)` HSGP truncation " *
                  "lower bound $floor is not below " *
                  "$(_EVENT_LP_RHO_PRIOR_HI) (SB errors the same way)")
    end
    return nothing
end

# Event-LP binds: fit (mu, L) over the materialized op_log_dose
# column (the shared `_hsgp_axis_fit` core) + the V2 floor gate.
# Runs AFTER kernel resolution (the axis column materializes there).
function _fit_event_lps(plan::StructuralPlan,
        columns::AbstractDict{Symbol})
    isempty(plan.event_lps) && return LinearPKEventLPSpec[]
    out = LinearPKEventLPSpec[]
    for el in plan.event_lps
        col = _sched_col_name(el.schedule, :op_log_dose)
        haskey(columns, col) ||
            _fail(el.label, "event-LP `$(el.name)`: axis column $col " *
                  "is not bound (bind_data materializes it from the " *
                  "schedule — internal ordering)")
        axiscol = _vector_column(columns, col, el.label,
            "event-LP axis column")
        fit = _hsgp_axis_fit(axiscol, el.c, el.label,
            "event-LP `$(el.name)` axis column $col")
        floor = only(_hsgp_floors([el.k], [fit], true))
        floor < _EVENT_LP_RHO_PRIOR_HI ||
            _fail(el.label, "event-LP `$(el.name)` HSGP truncation " *
                  "lower bound $floor is not below " *
                  "$(_EVENT_LP_RHO_PRIOR_HI) (SB errors the same way)")
        push!(out, LinearPKEventLPSpec(el.name, el.schedule, el.k, el.c,
            fit, el.label))
    end
    return out
end

# Kernel (KernelPlate) structure: everything provable without data.
# v1: at most one kernel per model; a kernel carries the ONLY
# likelihood (no top-level responses alongside — BRM routes kernel models
# away from the GLM flow). Panel plates (no schedules) follow
# `_validate_panel_kernel` (grouping ABSENT — implicit 1:n subjects,
# structural, no sentinel, pinned here + tests); grouped plates follow
# `_validate_grouped_kernel`.
function _validate_kernels(plan::StructuralPlan)
    plates = plan.kernel_plates
    length(plates) <= 1 ||
        _fail(:plan, "v1 admits at most one kernel plate per model " *
              "(got $(length(plates)))")
    isempty(plates) && return nothing
    # Mixed-level predictors (a response and a kernel LP arg sharing one
    # definition) fail with the precise message before the only-likelihood
    # gate below (which would otherwise mask the use-site confusion).
    for kp in plates, (p, _) in kp.lp_args
        _predictor_level(plan, p)
    end
    kp = only(plates)
    isempty(plan.responses) ||
        _fail(kp.label, "a kernel plate carries the only likelihood " *
              "(v1: no top-level responses alongside `$(kp.result)`)")
    # Name hygiene + collisions (result, slice params, cell locals) live in
    # the global `_validate_name_tables` gate via `_kernel_all_names`.
    _check_name_hygiene(kp.label)
    if kp.subjects isa Int
        kp.subjects > 0 ||
            _fail(kp.label, "subject count must be a positive integer, " *
                  "got $(kp.subjects)")
    end
    if _is_grouped_kernel(kp)
        _validate_grouped_kernel(plan, kp)
    else
        _validate_panel_kernel(plan, kp)
    end
    return nothing
end

# Panel-kernel structure (see `_validate_kernels`).
function _validate_panel_kernel(plan::StructuralPlan, kp::KernelPlate)
    length(kp.obs) == 1 ||
        _fail(kp.label, "panel v1 admits exactly one in-cell observation " *
              "(got $(length(kp.obs)))")
    isempty(kp.lp_args) ||
        _fail(kp.label, "panel plates take no LP args (LP args are the " *
              "grouped form — declare a schedule)")
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
    obs = only(kp.obs)
    _validate_kernel_obs_ref(kp, obs, params, known, false)
    kp.collected in union(Set{Symbol}(params), cell_locals) ||
        _fail(kp.label, "collected result `$(kp.collected)` is not a cell " *
              "name (slice param or cell-local assignment)")
    return nothing
end

# One in-cell observation node (panel and grouped share the shape):
# response a slice param; panel Gaussian-only, grouped the joint
# families too; location/scale/params names-or-literals resolving to
# cell/model names (literals finite; Gaussian scale literals positive —
# positional-second args of multi-param families take no positivity).
function _validate_kernel_obs_ref(kp::KernelPlate, obs::KernelObs,
        params::Vector{Symbol}, known::Set{Symbol}, grouped::Bool)
    obs.response in params ||
        _fail(kp.label, "kernel obs response `$(obs.response)` is not a " *
              "slice param (responses enter the cell as slices)")
    if grouped
        obs.family in (GaussianFam, CensoredAddpropnormalFam,
                TgiCategoryFam, TgiResponseFam, TgiCensoredFam) ||
            _fail(kp.label, "grouped kernels admit in-cell observations " *
                  "`Normal.(...)`, `CensoredAddpropnormal.(...)`, " *
                  "`TgiCategory.(...)`, `TgiResponse.(...)`, " *
                  "`TgiCensored.(...)` only, got $(obs.family)")
    else
        obs.family === GaussianFam ||
            _fail(kp.label, "kernel v1 admits a Gaussian in-cell observation " *
                  "only, got $(obs.family)")
    end
    if obs.family === GaussianFam && !isempty(obs.params)
        _fail(kp.label, "Gaussian in-cell observations take no `params` " *
              "(got $(obs.params))")
    end
    for (nm, ref) in ((:location, obs.location), (:scale, obs.scale))
        if ref isa Number && !(ref isa Bool)
            positive = nm === :scale && obs.family === GaussianFam
            (isfinite(ref) && (!positive || ref > 0)) ||
                _fail(kp.label, "kernel obs $nm literal must be finite" *
                      (positive ? " positive" : "") * ", got $ref")
        elseif ref isa Symbol
            ref in known ||
                _fail(kp.label, "kernel obs $nm `$ref` is neither a cell " *
                      "name nor a model-level scalar (cross-cell refs " *
                      "fail closed)")
            grouped && ref in _lp_cell_params(kp) &&
                _fail(kp.label, "kernel obs $nm `$ref` is an LP cell " *
                      "param — gather explicitly (`$ref[subj_map]` " *
                      "with a bound subject column; bare LP cell " *
                      "params do not lower as obs args)")
        else
            _fail(kp.label, "kernel obs $nm must be a cell/model name or " *
                  "a numeric literal, got $(repr(ref))")
        end
    end
    for ref in obs.params
        if ref isa Number && !(ref isa Bool)
            isfinite(ref) ||
                _fail(kp.label, "kernel obs params literal must be " *
                      "finite, got $ref")
        elseif ref isa Symbol
            ref in known ||
                _fail(kp.label, "kernel obs params `$ref` is neither a " *
                      "cell name nor a model-level scalar (cross-cell " *
                      "refs fail closed)")
            grouped && ref in _lp_cell_params(kp) &&
                _fail(kp.label, "kernel obs params `$ref` is an LP cell " *
                      "param — gather explicitly (`$ref[subj_map]` " *
                      "with a bound subject column; bare LP cell " *
                      "params do not lower as obs args)")
        else
            _fail(kp.label, "kernel obs params must be a cell/model name " *
                  "or a numeric literal, got $(repr(ref))")
        end
    end
    return nothing
end

"""Term kinds a subject-level predictor may carry in grouped kernels
(design over subject columns; summands and per-cell latents need
row-alignment work and fail closed). `VaryingEffectTerm` is sound here:
subject rows ARE subjects, so the draws' per-level `r` needs no gather —
`_ppl_gidx_<group>` is positional 1:n_sub (level order must match
subject order; the joint parity harness proves it numerically)."""
const _SUBJECT_TERM_KINDS =
    (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm, VaryingEffectTerm)

# Grouped-kernel structure (see `_validate_kernels`): schedules resolve,
# LP args name subject-exclusive identity predictors with admitted terms,
# the cell follows the grouped vocabulary, obs is a non-empty list.
function _validate_grouped_kernel(plan::StructuralPlan, kp::KernelPlate)
    length(kp.schedules) == 1 ||
        _fail(kp.label, "grouped v1 takes exactly one schedule " *
              "(got $(length(kp.schedules)) — multi-schedule kernels " *
              "are sequenced after the PK slice)")
    kp.timepoints === nothing ||
        _fail(kp.label, "grouped kernels take no timepoints (ragged axes " *
              "have no rectangular T)")
    sched = only(kp.schedules)
    raw = [sched.obs_subj, sched.obs_time, sched.dose_subj, sched.dose_time,
        sched.dose_amt]
    sched.ecg !== nothing && append!(raw, [sched.ecg[1], sched.ecg[2]])
    sched.tgi !== nothing && append!(raw, [sched.tgi[1], sched.tgi[2]])
    length(unique(raw)) == length(raw) ||
        _fail(kp.label, "schedule `$(sched.name)` reuses a raw column " *
              "($(raw)) — obs/dose/extra axes need distinct columns)")
    slices = kp.slices
    isempty(slices) &&
        _fail(kp.label, "kernel plate `$(kp.result)` takes at least one slice")
    cols = [c for (c, _, _) in slices]
    length(unique(cols)) == length(cols) ||
        _fail(kp.label, "duplicate slice columns $(cols)")
    params = [p for (_, p, _) in slices]
    for (_, _, kind) in slices
        kind in (:response, :unknown) ||
            _fail(kp.label, "grouped slice kind must be :response (bound) " *
                  "or :unknown (pre-bind), got $(repr(kind))")
    end
    isempty(kp.lp_args) &&
        _fail(kp.label, "grouped kernels take at least one LP arg " *
              "(`kernel(resp, lp...; subjects)` — LP values gather per " *
              "subject in-cell)")
    pnames = [p for (p, _) in kp.lp_args]
    length(unique(pnames)) == length(pnames) ||
        _fail(kp.label, "duplicate LP-arg predictors $(pnames)")
    cparams = [c for (_, c) in kp.lp_args]
    length(unique(cparams)) == length(cparams) ||
        _fail(kp.label, "duplicate LP-arg cell params $(cparams)")
    for (p, _) in kp.lp_args
        i = findfirst(q -> q.name === p, plan.predictors)
        i === nothing &&
            _fail(kp.label, "kernel LP arg `$p` is not a predictor (LP " *
                  "args name subject-level predictors; model scalars " *
                  "enter the cell as globals)")
        pred = plan.predictors[i]
        pred.link === IdentityLink ||
            _fail(kp.label, "kernel LP arg `$p` needs IdentityLink " *
                  "(got $(pred.link) — subject LPs feed the cell raw)")
        _predictor_level(plan, p) === :subject ||
            _fail(kp.label, "kernel LP arg `$p` is unreachable " *
                  "(internal: mixed-level predictors fail in " *
                  "`_predictor_level`)")
        for t in pred.terms
            t.kind in _SUBJECT_TERM_KINDS ||
                _fail(kp.label, "subject predictor `$p` carries a " *
                      "$(t.kind) term (grouped v1 admits " *
                      "$(_SUBJECT_TERM_KINDS) — summands and per-cell " *
                      "latents are sequenced after the PK slice)")
        end
    end
    # Cell assignments in order: slice params + LP cell params + earlier
    # locals + model-scope scalars (schedule handles are compile-time and
    # enter only as call first-args / gather roots — never as values).
    known = union(Set{Symbol}(params),
        Set{Symbol}(c for (_, c) in kp.lp_args), _union_names(plan))
    schednames = Set{Symbol}(s.name for s in kp.schedules)
    cell_locals = Set{Symbol}()
    for (nm, ex) in kp.assignments
        _collect_grouped_cell_refs!(Symbol[], ex, kp, known, schednames)
        push!(known, nm)
        push!(cell_locals, nm)
    end
    isempty(kp.obs) &&
        _fail(kp.label, "grouped kernels take at least one in-cell " *
              "observation")
    for obs in kp.obs
        _validate_kernel_obs_ref(kp, obs, params, known, true)
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

# Grouped-kernel cell vocabulary: calls to CELL_FNS (schedule first),
# schedule-map gathers (`reads[sched.obs_map]`) and prep-map gathers
# (`v[map]` with a bound integer column), dotted ops/math, undotted
# arithmetic over scalars, `ifelse`, bare names and numeric literals.
# Reductions, whole-column functions, other indexing, loops, branches,
# nested observations, and unknown names fail closed. Shapes
# (scalar/obs-axis/read-space) resolve at bind
# (`_grouped_cell_shapes`); this walker checks names + call/gather
# structure only.
#
# LP cell params are per-subject scalars: bare uses admit ONLY as
# direct cell-call args (`lp_ok`, threaded below) or gather sources
# (checked in `_collect_grouped_cell_gather!`) — every other bare use
# fails closed (flat verbatim emission cannot resolve a per-subject
# name, so the LP value gathers per row explicitly).
"""LP cell params of a grouped plate (the per-subject gatherable names)."""
_lp_cell_params(kp::KernelPlate) = Set{Symbol}(c for (_, c) in kp.lp_args)
"""Response-slice params of a grouped plate (gather leaves, never sources)."""
_slice_params(kp::KernelPlate) = Set{Symbol}(p for (_, p, _) in kp.slices)
function _collect_grouped_cell_refs!(refs, ex, kp::KernelPlate,
        known::Set{Symbol}, schednames::Set{Symbol}, lp_ok::Bool = false)
    label = kp.label
    ex isa Number && return nothing
    ex isa LineNumberNode && return nothing
    if ex isa Symbol
        ex in schednames &&
            _fail(label, "schedule `$ex` is a compile-time handle, not a " *
                  "value (pass it as a cell-call first arg or a gather " *
                  "root: `linear_pk_read_locs($ex, ...)` / " *
                  "`reads[$ex.obs_map]`)")
        ex in known ||
            _fail(label, "cell expression references unknown name `$ex` " *
                  "(slices + LP cell params + earlier cell locals + " *
                  "model-level scalars only)")
        !lp_ok && ex in _lp_cell_params(kp) &&
            _fail(label, "cell expression uses bare LP cell param `$ex` " *
                  "— gather explicitly (`$ex[subj_map]` with a bound " *
                  "subject column) or pass `$ex` to a cell call " *
                  "(per-subject names do not lower in flat verbatim code)")
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
                _collect_grouped_cell_refs!(refs, arg, kp, known, schednames)
            end
            return nothing
        end
        if fn isa Symbol && fn in REDUCTION_FNS
            return _fail(label, "reduction `$fn` does not lower in a cell " *
                                "(aggregate in the obs likelihood, not " *
                                "the cell)")
        end
        if fn isa Symbol && fn in VECTOR_FNS
            return _fail(label, "whole-column `$fn` does not lower in a " *
                                "cell (whole-model constructs only)")
        end
        if fn isa Symbol && fn in CELL_FNS
            args = ex.args[2:end]
            want = CELL_FN_ARITY[fn]
            # The event-LP form threads the provider's flat vector
            # second (SB position, after the schedule): v1 takes it
            # optionally (6 or 7 args), the AUC cell takes it
            # mandatorily (7 args — the runtime has no LP-less form).
            seven = length(args) == want + 1
            admit = fn === :linear_pk_read_locs ?
                (length(args) == want || seven) : length(args) == want
            admit ||
                _fail(label, "cell call `$fn` takes $want arguments " *
                      "(a schedule plus $(want - 1) cell/model names)" *
                      (fn === :linear_pk_read_locs ?
                       " or $(want + 1) with the event-LP " *
                       "`$(EVENT_LP_NAME)` second" :
                       " with the event-LP `$(EVENT_LP_NAME)` second") *
                      ", got $(length(args))")
            s = args[1]
            s isa Symbol && s in schednames ||
                _fail(label, "cell call `$fn` takes a declared schedule " *
                      "first (got $(repr(s)) — admitted schedules: " *
                      "$(sort!(collect(schednames))))")
            rest = args[2:end]
            if seven || fn === :linear_pk_read_locs_auc
                # The event-LP second arg is the provider's flat
                # vector: the fixed seam name only — structure
                # verifies the NAME and skips collection (the
                # provider output is a generated local, not a plan
                # parameter/assignment, so it is not cell-known —
                # the skip is load-bearing, not cosmetic). The
                # per-subject expansion slices the call arg by NAME.
                args[2] === EVENT_LP_NAME ||
                    _fail(label, "cell call `$fn` second argument must " *
                          "be the event-LP `$(EVENT_LP_NAME)` (got " *
                          "$(repr(args[2])) — the 7-arg form is " *
                          "`$fn(sched, log_F, lp...)`)")
                rest = args[3:end]
            end
            # Direct call args: the one verbatim position (besides
            # gather sources) where bare LP cell params admit — the
            # per-subject expansion resolves them. Nested positions
            # keep the default (fail-early on shapes generation
            # cannot resolve).
            for arg in rest
                _collect_grouped_cell_refs!(refs, arg, kp, known,
                    schednames, true)
            end
            return nothing
        end
        if fn isa Symbol && fn in SEGMENT_CELL_FNS
            args = ex.args[2:end]
            want = SEGMENT_CELL_FN_ARITY[fn]
            length(args) == want ||
                _fail(label, "cell call `$fn` takes $want arguments " *
                      "(a change vector + a cumulative-ends integer " *
                      "column: `$fn(change, ends)`), got $(length(args))")
            # The change series is an ordinary cell ref (bare LPs fail
            # — a per-subject scalar is not a row series); the ends
            # are a putative bind column (bind proves bound-ness +
            # the segment contract).
            _collect_grouped_cell_refs!(refs, args[1], kp, known,
                schednames)
            ends = args[2]
            ends isa Symbol ||
                _fail(label, "cell call `$fn` ends argument must be a " *
                      "bound integer column name, got $(repr(ends))")
            ends in schednames &&
                _fail(label, "schedule `$ends` is a compile-time " *
                      "handle, not a value (ends are a bound integer " *
                      "column: `$fn(change, ends)`)")
            ends in known &&
                _collect_grouped_cell_refs!(refs, ends, kp, known,
                    schednames)
            return nothing
        end
        if fn isa Symbol && fn in ASSIGNMENT_FNS
            for arg in ex.args[2:end]
                _collect_grouped_cell_refs!(refs, arg, kp, known, schednames)
            end
            return nothing
        end
        fn isa Symbol && startswith(string(fn), ".") &&
            _fail(label, "dotted operator $fn is not in the grouped-v1 " *
                         "cell vocabulary")
        return _fail(label, "call `$fn` is not in the grouped-v1 cell " *
                            "vocabulary (cell calls: $(CELL_FNS))")
    end
    head === :. &&
        return _collect_grouped_cell_dot!(refs, ex, kp, known, schednames)
    if head === :ref
        return _collect_grouped_cell_gather!(refs, ex, kp, known, schednames)
    end
    head === :(=) && _fail(label, "nested assignment does not lower in a cell")
    head === :kw &&
        _fail(label, "keyword arguments do not lower in a cell")
    return _fail(label, "unsupported expression head $head in a cell " *
                        "(calls + gathers + elementwise only)")
end

function _collect_grouped_cell_dot!(refs, ex, kp::KernelPlate,
        known::Set{Symbol}, schednames::Set{Symbol})
    label = kp.label
    length(ex.args) == 2 && ex.args[1] isa Symbol && ex.args[2] isa Expr &&
        ex.args[2].head === :tuple ||
        return _fail(label, "field access does not lower in a cell " *
                            "(dotted calls take `f.(...)`; schedule maps " *
                            "gather as `reads[sched.obs_map]`)")
    f = ex.args[1]
    args = ex.args[2].args
    if f === :ifelse
        length(args) == 3 ||
            _fail(label, "`ifelse` takes `ifelse.(condition, x, y)`")
        _collect_grouped_cell_condition!(refs, args[1], kp, known, schednames)
        for arg in args[2:end]
            _collect_grouped_cell_refs!(refs, arg, kp, known, schednames)
        end
        return nothing
    end
    f in ELEMENTWISE_FNS ||
        _fail(label, "dotted call `$f.(...)` is not in the grouped-v1 cell " *
                     "vocabulary")
    for arg in args
        _collect_grouped_cell_refs!(refs, arg, kp, known, schednames)
    end
    return nothing
end

function _collect_grouped_cell_condition!(refs, ex, kp::KernelPlate,
        known::Set{Symbol}, schednames::Set{Symbol})
    label = kp.label
    if ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] in ELEMENTWISE_COMPARISONS
        return _collect_grouped_cell_refs!(refs, ex, kp, known, schednames)
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

# Admitted indexing shapes: `reads[sched.map]` (a cell vector
# gathered by a schedule map, a static int vector at codegen) and
# `v[map]` (gathered by a plain-Symbol bound integer column — prep
# maps and subject columns). Sources are cell-call results, LP cell
# params (the generator rewrites them to LP vectors), or cell locals;
# slices and model scalars fail at shape (bind proves spaces).
function _collect_grouped_cell_gather!(refs, ex, kp::KernelPlate,
        known::Set{Symbol}, schednames::Set{Symbol})
    label = kp.label
    length(ex.args) == 2 || _fail(label, "indexing in a cell takes one " *
        "index (`v[sched.map]` or `v[map]`), got $(repr(ex))")
    vec, idx = ex.args[1], ex.args[2]
    vec isa Symbol && vec in known ||
        _fail(label, "gather source must be a cell name, got $(repr(vec))")
    push!(refs, vec)
    if idx isa Symbol
        idx in schednames &&
            _fail(label, "schedule `$idx` is a compile-time handle, not " *
                  "a value (gather by one of its maps: " *
                  "`reads[$idx.obs_map]`)")
        idx in known &&
            _fail(label, "gather index `$idx` names a cell/model value " *
                  "(gather indices are schedule maps like " *
                  "`reads[sched.obs_map]` or bound integer columns — " *
                  "typo'd column?)")
        # A putative bind column: bind proves bound + integer-valued +
        # range + length (shapes + prep validators).
        return nothing
    end
    idx isa Expr && idx.head === :. && length(idx.args) == 2 &&
        idx.args[1] isa Symbol && idx.args[2] isa QuoteNode ||
        _fail(label, "gather index must be a schedule map " *
              "(`reads[sched.obs_map]`) or a bound integer column, " *
              "got $(repr(idx))")
    s, m = idx.args[1], idx.args[2].value
    s in schednames ||
        _fail(label, "gather schedule `$s` is not declared (admitted: " *
              "$(sort!(collect(schednames))))")
    m in SCHEDULE_MAPS ||
        _fail(label, "schedule `$s` has no map `$m` (admitted maps: " *
              "$(SCHEDULE_MAPS))")
    spec = only(sp for sp in kp.schedules if sp.name === s)
    m in _sched_available_maps(spec, kp.assignments) ||
        _fail(label, "schedule `$s` map `$m` is not available " *
              "($(_sched_map_prereq(spec, m, kp.assignments)))")
    return nothing
end

# Prerequisite guidance for an unavailable schedule map (the caller
# proved `m` is a known map on a declared schedule — exactly one
# condition below fires).
function _sched_map_prereq(sched::LinearPKScheduleSpec, m::Symbol,
        assignments::Vector{Pair{Symbol,Any}})
    if m === :ecg_map || m === :tgi_map
        ax = m === :ecg_map ? :ecg : :tgi
        return "axis `$ax` is not declared (`$(sched.name) = " *
               "linear_pk_schedule(..., $ax = (subj, time))`)"
    end
    if !_cell_has_auc_call(assignments)
        return "`$m` needs an AUC cell call " *
               "(`linear_pk_read_locs_auc`) in the cell"
    end
    return "`$m` needs the declared tgi axis (`$(sched.name) = " *
           "linear_pk_schedule(..., tgi = (subj, time))`)"
end

# One declared extra read axis as builder input (`nothing` when
# undeclared): both columns must be bound.
function _grouped_schedule_axis(sched::LinearPKScheduleSpec,
        columns::Dict{Symbol,ColumnData}, axis::Symbol, kp::KernelPlate)
    spec = axis === :ecg ? sched.ecg : sched.tgi
    spec === nothing && return nothing
    for c in spec
        haskey(columns, c) ||
            _fail(kp.label, "schedule `$(sched.name)` $axis column `$c` " *
                  "is not bound")
    end
    return (columns[spec[1]], columns[spec[2]])
end

# Build a grouped schedule from bound columns (shared by bind
# resolution + exact-rebuild verification, so the two can never drift):
# raw columns must be bound, extra axes ride when declared, builder
# errors relabel to the kernel.
function _build_grouped_schedule(plan::StructuralPlan, kp::KernelPlate,
        columns::Dict{Symbol,ColumnData})
    sched = only(kp.schedules)
    for c in (sched.obs_subj, sched.obs_time, sched.dose_subj,
            sched.dose_time, sched.dose_amt)
        haskey(columns, c) ||
            _fail(kp.label, "schedule `$(sched.name)` column `$c` is not bound")
    end
    combine = _schedule_combine_simultaneous(plan, sched.name)
    ecg = _grouped_schedule_axis(sched, columns, :ecg, kp)
    tgi = _grouped_schedule_axis(sched, columns, :tgi, kp)
    return try
        build_linear_pk_schedule(columns[sched.obs_subj],
            columns[sched.obs_time], columns[sched.dose_subj],
            columns[sched.dose_time], columns[sched.dose_amt];
            combine_simultaneous = combine, ecg = ecg, tgi = tgi)
    catch err
        err isa ContractValidationError &&
            _fail(kp.label, "schedule `$(sched.name)`: $(err.message)")
        rethrow()
    end
end

# Grouped-kernel bind checks: resolved subjects, schedule columns
# verified by exact rebuild (hand-bound plans carry verified products,
# never trusted ones — the scalar-expansion precedent), first-obs
# response on the primary axis (foreign-axis responses ride later
# observations), numeric finite slices, subject-predictor columns at
# n_sub, proved cell shapes + per-obs axis agreement, nadir segment
# contracts, gather-map prep contracts, TGI axis order.
function _validate_grouped_kernel_data(plan::StructuralPlan, kp::KernelPlate)
    kp.subjects isa Int ||
        _fail(kp.label, "subjects dims key `$(kp.subjects)` unresolved " *
              "(bind_data with dims first)")
    n_sub = kp.subjects
    length(kp.schedules) == 1 ||
        _fail(kp.label, "grouped v1 takes exactly one schedule " *
              "(got $(length(kp.schedules)))")
    sched = only(kp.schedules)
    built = _build_grouped_schedule(plan, kp, plan.columns)
    built.n_subjects == n_sub ||
        _fail(kp.label, "schedule `$(sched.name)` covers " *
              "$(built.n_subjects) subjects ≠ subjects $n_sub")
    for f in vcat(collect(_SCHED_MATERIALIZED_FIELDS),
            _sched_extra_fields(sched, kp.assignments))
        col = _sched_col_name(sched.name, f)
        haskey(plan.columns, col) ||
            _fail(kp.label, "schedule `$(sched.name)` product `$col` " *
                  "missing (bind_data materializes op columns)")
        plan.columns[col] == getfield(built, f) ||
            _fail(kp.label, "schedule `$(sched.name)` product `$col` is " *
                  "not the schedule build (bind_data materializes it — " *
                  "a hand-bound plan must carry the identical product)")
    end
    n_axis = length(plan.columns[sched.obs_subj])
    plan.n_obs == n_axis ||
        _fail(kp.label, "n_obs $(plan.n_obs) ≠ schedule obs axis $n_axis " *
              "(the first in-cell observation responds on the schedule " *
              "obs axis; foreign-axis responses ride later observations)")
    for (col, param, kind) in kp.slices
        kind === :response ||
            _fail(kp.label, "slice `$param` kind unresolved " *
                  "(bind_data resolves :unknown to :response)")
        haskey(plan.columns, col) ||
            _fail(kp.label, "slice column `$col` is not bound")
        colv = plan.columns[col]
        eltype(colv) <: Real ||
            _fail(kp.label, "slice column `$col` must be numeric, " *
                  "got $(eltype(colv))")
        all(isfinite, colv) ||
            _fail(kp.label, "slice column `$col` must be finite")
        # Any length admits (foreign-axis responses ride their own
        # axis); per-obs axis agreement is proved on shapes below.
    end
    for (p, _) in kp.lp_args
        i = findfirst(q -> q.name === p, plan.predictors)
        i === nothing &&
            _fail(kp.label, "kernel LP arg `$p` is not a predictor")
        for t in plan.predictors[i].terms
            t.kind in _SUBJECT_TERM_KINDS ||
                _fail(kp.label, "subject predictor `$p` carries a " *
                      "$(t.kind) term (grouped v1 admits " *
                      "$(_SUBJECT_TERM_KINDS))")
            for c in t.columns
                # In-graph deriveds compute at eval from bound sources
                # (length-checked at their own level); no bind values exist.
                _is_derived(plan, c) && continue
                haskey(plan.columns, c) ||
                    _fail(kp.label, "subject predictor `$p` column `$c` " *
                          "is not bound")
                colv = plan.columns[c]
                length(colv) == n_sub ||
                    _fail(kp.label, "subject predictor `$p` column `$c` " *
                          "has length $(length(colv)), want n_sub $n_sub")
                if t.kind in (ContinuousTerm, OffsetTerm)
                    eltype(colv) <: Real ||
                        _fail(kp.label, "subject predictor `$p` column " *
                              "`$c` must be numeric, got $(eltype(colv))")
                    all(isfinite, colv) ||
                        _fail(kp.label, "subject predictor `$p` column " *
                              "`$c` must be finite")
                end
            end
        end
    end
    shapes = _prove_grouped_cell_shapes(kp, kp.slices, plan.columns)
    _validate_nadir_ends(kp, shapes, plan.columns)
    _validate_grouped_gather_maps(kp, shapes, plan.columns,
        built.n_reads_total)
    _validate_tgi_axis_order(kp, sched, plan.columns)
    return nothing
end

# Kernel bind checks: resolved dims, total kinds, flat-T-blocked lengths,
# subjects coverage. Runs on bound plans (bind resolves Symbol dims via
# the `dims` map first; hand-bound plans carry Ints directly).
function _validate_kernels_data(plan::StructuralPlan)
    isempty(plan.kernel_plates) && return nothing
    kp = only(plan.kernel_plates)
    _is_grouped_kernel(kp) && return _validate_grouped_kernel_data(plan, kp)
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
        colv = _vector_column(plan.columns, col, kp.label, "slice column")
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
            expv =
                _vector_column(plan.columns, exp, kp.label, "slice expansion")
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
    darstates = [s.state for s in plan.dar_paths]
    vectors = [p.name for p in plan.vector_parameters]
    svec = [v.name for v in plan.spline_vectors]
    vk1 = Symbol[nm for d in plan.varying_draws
        if d.kind === :intercept1 || d.kind === :slope1
        for nm in _varying_k1_names(d)]
    vcorr = Symbol[nm for d in plan.varying_draws
        if d.kind === :correlated
        for nm in _varying_corr_names(d)]
    hsgp = Symbol[nm for hb in plan.hsgp_bases for nm in _hsgp_all_names(hb)]
    kern = Symbol[nm for kp in plan.kernel_plates for nm in _kernel_all_names(kp)]
    mats = Symbol[m.name for m in plan.matrices]
    elps = Symbol[nm for el in plan.event_lps
        for nm in [_event_lp_all_names(el); el.name]]
    length(unique(params)) == length(params) || _fail(:plan, "duplicate parameter names")
    length(unique(assigns)) == length(assigns) ||
        _fail(:plan, "duplicate assignment names")
    length(unique(deriveds)) == length(deriveds) ||
        _fail(:plan, "duplicate derived-column names")
    length(unique(plates)) == length(plates) ||
        _fail(:plan, "duplicate plate-parameter names")
    length(unique(scanstates)) == length(scanstates) ||
        _fail(:plan, "duplicate scan-state names")
    length(unique(darstates)) == length(darstates) ||
        _fail(:plan, "duplicate dar-state names")
    length(unique(vectors)) == length(vectors) ||
        _fail(:plan, "duplicate vector-parameter names")
    length(unique(svec)) == length(svec) ||
        _fail(:plan, "duplicate spline-vector names")
    length(unique(vk1)) == length(vk1) ||
        _fail(:plan, "duplicate K=1 varying names")
    length(unique(vcorr)) == length(vcorr) ||
        _fail(:plan, "duplicate correlated varying names")
    length(unique(hsgp)) == length(hsgp) ||
        _fail(:plan, "duplicate hsgp names")
    length(unique(kern)) == length(kern) ||
        _fail(:plan, "duplicate kernel-plate names")
    length(unique(mats)) == length(mats) ||
        _fail(:plan, "duplicate design-matrix names")
    length(unique(elps)) == length(elps) ||
        _fail(:plan, "duplicate event-LP names")
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
        (vk1, params, "K=1 varying names and parameters"),
        (vk1, assigns, "K=1 varying names and assignments"),
        (vk1, deriveds, "K=1 varying names and derived columns"),
        (vk1, plates, "K=1 varying names and plate parameters"),
        (vk1, scanstates, "K=1 varying names and scan states"),
        (vk1, vectors, "K=1 varying names and vector parameters"),
        (vk1, svec, "K=1 varying names and spline vectors"),
        (vcorr, params, "correlated varying names and parameters"),
        (vcorr, assigns, "correlated varying names and assignments"),
        (vcorr, deriveds, "correlated varying names and derived columns"),
        (vcorr, plates, "correlated varying names and plate parameters"),
        (vcorr, scanstates, "correlated varying names and scan states"),
        (vcorr, vectors, "correlated varying names and vector parameters"),
        (vcorr, svec, "correlated varying names and spline vectors"),
        (vcorr, vk1, "correlated varying names and K=1 varying names"),
        (hsgp, params, "hsgp names and parameters"),
        (hsgp, assigns, "hsgp names and assignments"),
        (hsgp, deriveds, "hsgp names and derived columns"),
        (hsgp, plates, "hsgp names and plate parameters"),
        (hsgp, scanstates, "hsgp names and scan states"),
        (hsgp, vectors, "hsgp names and vector parameters"),
        (hsgp, svec, "hsgp names and spline vectors"),
        (hsgp, vk1, "hsgp names and K=1 varying names"),
        (hsgp, vcorr, "hsgp names and correlated varying names"),
        (kern, params, "kernel-plate names and parameters"),
        (kern, assigns, "kernel-plate names and assignments"),
        (kern, deriveds, "kernel-plate names and derived columns"),
        (kern, plates, "kernel-plate names and plate parameters"),
        (kern, scanstates, "kernel-plate names and scan states"),
        (kern, vectors, "kernel-plate names and vector parameters"),
        (kern, svec, "kernel-plate names and spline vectors"),
        (kern, vk1, "kernel-plate names and K=1 varying names"),
        (kern, vcorr, "kernel-plate names and correlated varying names"),
        (kern, hsgp, "kernel-plate names and hsgp names"),
        (mats, params, "design-matrix names and parameters"),
        (mats, assigns, "design-matrix names and assignments"),
        (mats, deriveds, "design-matrix names and derived columns"),
        (mats, plates, "design-matrix names and plate parameters"),
        (mats, scanstates, "design-matrix names and scan states"),
        (mats, vectors, "design-matrix names and vector parameters"),
        (mats, svec, "design-matrix names and spline vectors"),
        (mats, vk1, "design-matrix names and K=1 varying names"),
        (mats, vcorr, "design-matrix names and correlated varying names"),
        (mats, hsgp, "design-matrix names and hsgp names"),
        (mats, kern, "design-matrix names and kernel-plate names"),
        (darstates, params, "dar states and parameters"),
        (darstates, assigns, "dar states and assignments"),
        (darstates, deriveds, "dar states and derived columns"),
        (darstates, plates, "dar states and plate parameters"),
        (darstates, scanstates, "dar states and scan states"),
        (darstates, vectors, "dar states and vector parameters"),
        (darstates, svec, "dar states and spline vectors"),
        (darstates, vk1, "dar states and K=1 varying names"),
        (darstates, vcorr, "dar states and correlated varying names"),
        (darstates, hsgp, "dar states and hsgp names"),
        (darstates, kern, "dar states and kernel-plate names"),
        (darstates, mats, "dar states and design-matrix names"),
        (elps, params, "event-LP names and parameters"),
        (elps, assigns, "event-LP names and assignments"),
        (elps, deriveds, "event-LP names and derived columns"),
        (elps, plates, "event-LP names and plate parameters"),
        (elps, scanstates, "event-LP names and scan states"),
        (elps, vectors, "event-LP names and vector parameters"),
        (elps, svec, "event-LP names and spline vectors"),
        (elps, vk1, "event-LP names and K=1 varying names"),
        (elps, vcorr, "event-LP names and correlated varying names"),
        (elps, hsgp, "event-LP names and hsgp names"),
        (elps, kern, "event-LP names and kernel-plate names"),
        (elps, mats, "event-LP names and design-matrix names"),
        (elps, darstates, "event-LP names and dar states"))
        overlap = intersect(l, r)
        isempty(overlap) ||
            _fail(:plan, "names in both $what: $(join(overlap, ", "))")
    end
    allnames = union(params, assigns, deriveds, plates, scanstates, darstates,
        vectors, svec, vk1, vcorr, hsgp, kern, mats)
    for pn in pnames
        pn in allnames && _fail(
            :plan,
            "predictor $pn collides with a parameter/assignment/derived/plate/scan/dar/vector/spline/varying/kernel/matrix name",
        )
        block_name(pn) in allnames && _fail(
            :plan,
            "parameter/assignment/derived/plate/scan/dar/vector/spline/varying/kernel/matrix $(block_name(pn)) collides with predictor $pn block name",
        )
    end
    for n in Iterators.flatten((pnames, params, assigns, deriveds, plates, scanstates, darstates, vectors, svec, vk1, vcorr, hsgp, kern, mats))
        _check_name_hygiene(n)
    end
    return nothing
end

# A scan is non-centered when a step writes the carried state deterministically
# (`state[loopvar] = ...`): the layout slice then holds the iid innovations
# and the emitter reconstructs the state via the RK-core `scan(...)`
# carry-fold. Otherwise (every step samples the state) the slice holds the
# state itself (centered form).
_is_noncentered_scan(s::ScanSpec) =
    any(st -> st.kind === :assign && st.indexed, s.step)

# In-graph name of a non-centered scan's innovation slice
# (`_ppl_scan_z_<state>`). Reserved-prefix validation guarantees no user
# name collides with it; the state name itself binds the reconstruction.
_scan_innovation_name(s::ScanSpec) = Symbol(:_ppl_scan_z_, s.state)

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

# In-graph name of a dar trajectory's innovation slice
# (`_ppl_dar_z_<state>`, length `n_obs - 1`). Reserved-prefix validation
# guarantees no user name collides with it; the state name itself binds
# the emitter's `scan(...)` reconstruction.
_dar_innovation_name(s::DarSpec) = Symbol(:_ppl_dar_z_, s.state)

# Structural invariants of each differenced-AR(1) trajectory: the
# persistence names a `Normal` sampled parameter on exactly `(:interval,
# 0, 1)` (SB's `beta ~ normal(0.5, 0.2; lower=0, upper=1)`; overrides
# ride the same spelling with new location/scale) and the scale names a
# `Normal` sampled parameter on `:positive` (SB's `sigma ~
# normal(0, 0.2; lower=0)`). The `n_obs ≥ 2` length gate lives in the
# layout (unbound surface plans carry `n_obs = 0`, like a scan's
# symbolic `hi` — lengths resolve at bind).
function _validate_dar_paths(plan::StructuralPlan)
    for s in plan.dar_paths
        s.beta === s.sigma && _fail(s.label,
            "dar persistence and scale must be distinct sampled parameters " *
            "(SB samples `beta` and `sigma` separately), got :$(s.beta) twice")
        i = findfirst(p -> p.name === s.beta, plan.parameters)
        i === nothing && _fail(s.label,
            "dar persistence :$(s.beta) must name a scalar sampled " *
            "parameter (`$(s.beta) ~ truncated(Normal(0.5, 0.2), 0, 1)`)")
        b = plan.parameters[i]
        (b.family === :normal && b.support_override == (:interval, 0.0, 1.0)) ||
            _fail(s.label,
                "dar persistence :$(s.beta) must be Normal on exactly " *
                "(:interval, 0, 1) (SB's `beta ~ normal(0.5, 0.2; " *
                "lower=0, upper=1)`), got :$(b.family) on " *
                "$(repr(b.support_override))")
        j = findfirst(p -> p.name === s.sigma, plan.parameters)
        j === nothing && _fail(s.label,
            "dar scale :$(s.sigma) must name a scalar sampled parameter " *
            "(`$(s.sigma) ~ HalfNormal(0.2)`)")
        sg = plan.parameters[j]
        (sg.family === :normal && sg.support_override === :positive) ||
            _fail(s.label,
                "dar scale :$(s.sigma) must be Normal on :positive (SB's " *
                "`sigma ~ normal(0, 0.2; lower=0)`), got :$(sg.family) " *
                "on $(repr(sg.support_override))")
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
        condcol =
            _vector_column(plan.columns, ex, label, "ifelse condition column")
        eltype(condcol) === Bool ||
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
# `(:upper, hi)` is an upper-only truncation with a finite hi; the family must
# be real-support (a truncated Normal), and the density is Stan's upper-bound
# kernel (plain normal_lpdf plus the bare-`u` Jacobian — NO truncation
# renormalizer, matching SB which never renormalizes bounds).
function _validate_support_override(label, family::Symbol,
        ov::SupportOverride, args::NamedTuple)
    ov === nothing && return nothing
    if ov isa Tuple
        if ov[1] === :upper
            length(ov) == 2 || _fail(label,
                "tuple support override must be (:upper, hi), got $ov")
            family === :normal || _fail(label,
                "an :upper override is a truncated Normal in slice 1 " *
                "(`truncated(Normal(mu, s), -Inf, hi)`); got $family")
            hi = ov[2]
            isfinite(hi) || _fail(label,
                ":upper bound must be finite; got $hi")
            return nothing
        end
        (ov[1] === :interval && length(ov) == 3) || _fail(label,
            "tuple support override must be (:interval, lo, hi) or " *
            "(:upper, hi); got $ov")
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
_is_glm_family(f) = f === NormalIDGLMFam || f === BernoulliLogitGLMFam ||
    f === PoissonLogGLMFam

"""Vector-parameter families and their positional arg keys."""
const VECTOR_ARITY = Dict{Symbol,Tuple{Vararg{Symbol}}}(
    :ordered_normal => (:arg1, :arg2),
    :vector_normal => (:arg1, :arg2),
    :simplex_dirichlet => (:arg1,),
    :positive_exponential => (:arg1,),
    :cholesky_corr_lkj => (:arg1,),
)

"""Joint-factor vector families (the correlated-outcomes factor pieces)."""
const _JOINT_FACTOR_FAMILIES = (:positive_exponential, :cholesky_corr_lkj)

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
            "vector_normal, simplex_dirichlet, positive_exponential, " *
            "cholesky_corr_lkj)")
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
        elseif p.family === :positive_exponential
            th = p.args.arg1
            if th isa Symbol
                th in _union_names(plan) || _fail(p.label,
                    "exponential scale $th is not a scalar " *
                    "parameter/assignment name")
            else
                (th isa Real && isfinite(th) && th > 0) || _fail(p.label,
                    "exponential scale must be a finite positive literal " *
                    "or a scalar parameter/assignment name, got $(repr(th))")
            end
            p.size !== nothing || _fail(p.label,
                "joint-factor scales need a concrete size (the joint width " *
                "K — sizes are structural, never inferred)")
            p.size >= 1 || _fail(p.label,
                "joint-factor scales size must be ≥ 1, got $(p.size)")
        elseif p.family === :cholesky_corr_lkj
            eta = p.args.arg1
            (eta isa Real && isfinite(eta) && eta > 0) || _fail(p.label,
                "LKJ shape must be a finite positive literal " *
                "(a hyperparameter), got $(repr(eta))")
            p.size !== nothing || _fail(p.label,
                "joint-factor LKJ Cholesky needs a concrete size (the joint " *
                "width K — sizes are structural, never inferred)")
            p.size >= 1 || _fail(p.label,
                "joint-factor LKJ Cholesky size must be ≥ 1, got $(p.size)")
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
    # per-threshold Ordinal, as the simplex `predictor`, or as a joint
    # factor piece), by exactly one monotonic term (as its `increments`
    # simplex), or by exactly one R2D2 prior (as its share `phi`) —
    # never shared.
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
        # A joint response links its factor's two pieces explicitly (existence
        # + family diagnosis belongs to `_validate_joint_response`, which runs
        # later with the full response context — here only count the edges).
        if r.family === MvNormalCholeskyFam
            for f in (r.factor_scales, r.factor_corr)
                f === nothing && continue
                haskey(refs, f) && push!(refs[f], r.label)
            end
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
    for rp in plan.r2d2_priors
        haskey(refs, rp.phi) || _fail(rp.predictor,
            "R2D2 share parameter $(rp.phi) is not a vector parameter")
        p = only(q for q in plan.vector_parameters if q.name === rp.phi)
        p.family === :simplex_dirichlet || _fail(rp.predictor,
            "R2D2 share parameter $(rp.phi) must be a " *
            ":simplex_dirichlet vector parameter, got $(p.family)")
        push!(refs[rp.phi], rp.predictor)
    end
    for p in plan.vector_parameters
        got = refs[p.name]
        isempty(got) && _fail(p.label,
            "vector parameter $(p.name) unused by any response, " *
            "monotonic term, joint-factor link, or R2D2 prior")
        length(got) == 1 || _fail(p.label,
            "vector parameter $(p.name) shared by " *
            "$(join(got, ", ")) — one vector parameter per response, " *
            "monotonic term, joint-factor link, or R2D2 prior")
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

function _validate_matrices(plan::StructuralPlan)
    matnames = Set{Symbol}(m.name for m in plan.matrices)
    for m in plan.matrices
        isempty(m.columns) && _fail(m.label,
            "design matrix $(m.name) has no columns " *
            "(hcat needs at least one)")
        seen_cols = Set{Symbol}()
        n_intercept = 0
        for c in m.columns
            if c === nothing
                n_intercept += 1
                n_intercept > 1 && _fail(m.label,
                    "design matrix $(m.name) has two intercept positions " *
                    "— one coefficient per column")
                continue
            end
            c in seen_cols && _fail(m.label,
                "design matrix $(m.name) repeats column $c " *
                "— one coefficient per column")
            push!(seen_cols, c)
            c in matnames && _fail(m.label,
                "design matrix $(m.name) nests matrix $c — nested hcat " *
                "is not in slice D1 (flatten it)")
            _is_plate_param(plan, c) && _fail(m.label,
                "design matrix $(m.name) over the latent vector $c is not " *
                "in slice D1 (the me mirror stays affine)")
            any(s -> s.state === c, plan.scans) && _fail(m.label,
                "design matrix $(m.name) over scan state $c is not in " *
                "slice D1 (data/derived columns only)")
            any(p -> p.name === c, plan.parameters) && _fail(m.label,
                "design matrix $(m.name) over sampled parameter $c is not " *
                "in slice D1 (data/derived columns only)")
            any(p -> p.name === c, plan.vector_parameters) && _fail(m.label,
                "design matrix $(m.name) over vector parameter $c is not " *
                "in slice D1 (data/derived columns only)")
            any(a -> a.name === c, plan.assignments) && _fail(m.label,
                "design matrix $(m.name) over scalar assignment $c is not " *
                "in slice D1 (data/derived columns only)")
        end
    end
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
    if t.kind === VaryingEffectTerm
        _validate_effect_term(t, pred)
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
    if t.kind === ScanSummandTerm
        _validate_scan_term(t, pred, plan)
        return nothing
    end
    if t.kind === MatrixTerm
        _validate_matrix_term(t, pred, plan)
        return nothing
    end
    if t.kind === DarSummandTerm
        _validate_dar_term(t, pred, plan)
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

# A matrix term names its design matrix in `options` (`(matrix,)` — the
# gather/spline options precedent) and carries exactly the matrix's data
# columns in order (intercept positions excluded — they take no column);
# its addressee is the matrix name. Per-element prior coverage (one
# PopulationPrior row per element addressee) is checked in
# `_validate_priors`, which expands the matrix.
function _validate_matrix_term(t::TermSpec, pred::PredictorSpec, plan::StructuralPlan)
    o = t.options
    Tuple(keys(o)) == (:matrix,) ||
        _fail(t.label, "matrix term options must be exactly " *
              "`(matrix,)`, got $(Tuple(keys(o)))")
    o.matrix isa Symbol ||
        _fail(t.label, "matrix term matrix must be a Symbol, " *
              "got $(repr(o.matrix))")
    m = _find_matrix(plan, o.matrix)
    m === nothing &&
        _fail(t.label, "matrix term addresses unknown design matrix " *
              ":$(o.matrix)")
    data_cols = Symbol[c for c in m.columns if c !== nothing]
    t.columns == data_cols ||
        _fail(t.label, "matrix term columns must be exactly the " *
              "design-matrix data columns in order ($(data_cols)), " *
              "got $(t.columns)")
    t.addressee === o.matrix ||
        _fail(t.label, "matrix term addressee must be its matrix " *
              ":$(o.matrix), got $(t.addressee)")
    return nothing
end

# An effect term names its draws by label in `options` (never a lossy
# suffix parse) and carries exactly the grouping column; its addressee
# is its own label (self-addressed: effect terms take no
# PopulationPrior). Draws linkage (existence, slices, dangling) is
# checked jointly in `_validate_varying_draws`, which sees predictors,
# draws, and slices together.
function _validate_effect_term(t::TermSpec, pred::PredictorSpec)
    o = t.options
    Tuple(keys(o)) == (:draws,) ||
        _fail(t.label, "varying effect options must be exactly " *
              "`(draws,)`, got $(Tuple(keys(o)))")
    o.draws isa Symbol ||
        _fail(t.label, "effect draws must be a Symbol, " *
              "got $(repr(o.draws))")
    length(t.columns) == 1 ||
        _fail(t.label, "varying effect columns must be exactly the " *
              "grouping column, got $(t.columns)")
    t.addressee === t.label ||
        _fail(t.label, "varying effect addressee must be its own label " *
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
# (no population prior, like the spline summands).
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

# A scan summand names its recurrence (`scan_id`) and its sampled scalar
# coefficient (`coef`) in `options` and carries no columns (the state is
# sampled, not data); its addressee is its own label (self-addressed: the
# coefficient's prior lives on the `SampledParameter`, not a population
# prior). v1 admits Normal coefficients only (SB's `ar` beta is a Normal
# `popefs` coefficient); centered and non-centered states both read.
function _validate_scan_term(t::TermSpec, pred::PredictorSpec, plan::StructuralPlan)
    o = t.options
    Tuple(keys(o)) == (:scan_id, :coef) ||
        _fail(t.label, "scan summand options must be exactly " *
              "`(scan_id, coef)`, got $(Tuple(keys(o)))")
    o.scan_id isa Symbol ||
        _fail(t.label, "scan summand scan_id must be a Symbol, " *
              "got $(repr(o.scan_id))")
    o.coef isa Symbol ||
        _fail(t.label, "scan summand coef must be a Symbol, " *
              "got $(repr(o.coef))")
    isempty(t.columns) ||
        _fail(t.label, "scan summand carries no columns (the state is " *
              "sampled, not data), got $(t.columns)")
    t.addressee === t.label ||
        _fail(t.label, "scan summand addressee must be its own label " *
              "(self-addressed, no population prior), got $(t.addressee)")
    any(s -> s.state === o.scan_id, plan.scans) ||
        _fail(t.label, "scan summand addresses unknown scan state " *
              ":$(o.scan_id) (no such `@scan` block)")
    i = findfirst(p -> p.name === o.coef, plan.parameters)
    i === nothing &&
        _fail(t.label, "scan summand coef :$(o.coef) must name a scalar " *
              "sampled parameter (`$(o.coef) ~ Normal(...)`)")
    plan.parameters[i].family === :normal ||
        _fail(t.label, "scan summand coef :$(o.coef) must be Normal in v1 " *
              "(SB's `ar` beta is a Normal population coefficient), got " *
              ":$(plan.parameters[i].family)")
    return nothing
end

# A dar summand names its trajectory (`dar_id`) in `options` and carries
# no columns (the state is sampled, not data) and NO coefficient (the
# zero-started path is beta-free — the formula intercept is the initial
# level, the `mo1` splice shape); its addressee is its own label
# (self-addressed: the persistence/scale priors live on the
# `SampledParameter`s, not a population prior). Beta/sigma linkage is
# checked on the `DarSpec` itself (`_validate_dar_paths`).
function _validate_dar_term(t::TermSpec, pred::PredictorSpec, plan::StructuralPlan)
    o = t.options
    Tuple(keys(o)) == (:dar_id,) ||
        _fail(t.label, "dar summand options must be exactly `(dar_id,)`, " *
              "got $(Tuple(keys(o)))")
    o.dar_id isa Symbol ||
        _fail(t.label, "dar summand dar_id must be a Symbol, " *
              "got $(repr(o.dar_id))")
    isempty(t.columns) ||
        _fail(t.label, "dar summand carries no columns (the state is " *
              "sampled, not data), got $(t.columns)")
    t.addressee === t.label ||
        _fail(t.label, "dar summand addressee must be its own label " *
              "(self-addressed, no population prior), got $(t.addressee)")
    any(s -> s.state === o.dar_id, plan.dar_paths) ||
        _fail(t.label, "dar summand addresses unknown dar state " *
              ":$(o.dar_id) (no such `dar()` trajectory)")
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
    # Scan summands name a recurrence + scalar coefficient in `options`, not
    # columns; structure validation checked both names.
    t.kind === ScanSummandTerm && return nothing
    # Dar summands name a trajectory in `options`, not columns; structure
    # validation checked the name.
    t.kind === DarSummandTerm && return nothing
    for c in t.columns
        # A per-cell latent enters a design only through a ContinuousTerm
        # (a free coefficient scaling the latent vector — the SB `me`
        # mirror); every other term kind over a latent fails closed.
        if _is_plate_param(plan, c) && t.kind !== VaryingEffectTerm
            t.kind === ContinuousTerm || _fail(t.label,
                "term over the latent vector $c must be a ContinuousTerm " *
                "(a free coefficient scaling the latent — got $(t.kind))")
            continue
        end
        haskey(plan.columns, c) || _is_derived(plan, c) ||
            _fail(t.label, "term references missing column $c")
    end
    # Effect terms name the raw grouping column (strings included — the
    # encoder maps levels to codes); presence above is the whole check.
    t.kind === VaryingEffectTerm && return nothing
    # FactorTerm: column presence is checked by the loop above; level
    # coverage is a LevelMap concern (_validate_levelmaps_data).
    if t.kind === ContinuousTerm || t.kind === OffsetTerm
        c = only(t.columns)
        # A latent vector is length-n by construction (like a derived
        # column); its values are parameters, never data eltypes.
        _is_plate_param(plan, c) && return nothing
        # Derived columns are length-n by construction; their eltype is
        # unknown statically (in-graph Julia errors are loud).
        _is_derived(plan, c) && return nothing
        col = _vector_column(plan.columns, c, t.label, "term column")
        eltype(col) <: Real ||
            _fail(t.label, "column $c must be numeric")
    end
    # MatrixTerm: same per-data-column numeric rule over its (intercept-
    # free) columns. Presence rode the loop above; latents fail there
    # (ContinuousTerm-only); derived columns are length-n by
    # construction. Length-n itself rides bind (each bound column is
    # length-checked — the hcat is then safe).
    if t.kind === MatrixTerm
        for c in t.columns
            _is_derived(plan, c) && continue
            col = plan.columns[c]
            eltype(col) <: Real ||
                _fail(t.label, "column $c must be numeric")
        end
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
    col = _vector_column(plan.columns, c, t.label, "monotonic index")
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
        has_intercept = any(t -> t.kind === InterceptTerm, pred.terms)
        if !has_intercept
            for t in pred.terms
                t.kind === MatrixTerm || continue
                m = _find_matrix(plan, t.options.matrix)
                m !== nothing && any(isnothing, m.columns) &&
                    (has_intercept = true; break)
            end
        end
        if has_intercept
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
        columns::AbstractDict{Symbol})
    out = LevelMap[]
    for m in levelmaps
        haskey(columns, m.column) ||
            _fail(:plan, "LevelMap addresses missing column $(m.column)")
        groupcol =
            _vector_column(columns, m.column, :plan, "grouping column")
        levels =
            try
                _grouping_levels(groupcol)
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
    r2d2 = Set{Symbol}(rp.predictor for rp in plan.r2d2_priors)
    glm_labels = Set{Symbol}(r.label for r in plan.responses if _is_glm_family(r.family))
    for pr in plan.population_priors
        any(p -> p.name === pr.predictor, plan.predictors) ||
            pr.predictor in glm_labels ||
            _fail(:plan, "prior addresses unknown predictor $(pr.predictor)")
        pr.predictor in r2d2 && _fail(:plan,
            "predictor $(pr.predictor) carries an R2D2Prior — its prior " *
            "mass lives there, not in a PopulationPrior row (explicit " *
            "Normal columns ride the overrides map)")
        key = (pr.predictor, pr.addressee)
        key in seen &&
            _fail(:plan, "duplicate prior for $key")
        push!(seen, key)
        isfinite(pr.location) && isfinite(pr.scale) && pr.scale > 0 ||
            _fail(:plan, "prior for $key must be Normal(finite, positive)")
    end
    for pred in plan.predictors
        # R2D2 predictors are covered by _validate_r2d2, not here.
        pred.name in r2d2 && continue
        # Offset terms carry no coefficient; latent terms carry the per-cell
        # PlateParameter, whose prior lives on the plate parameter itself;
        # effect terms carry a VaryingDraws, spline summands a SplineBasis,
        # hsgp summands an HSGPBasis, and monotonic summands (mo1) an
        # increment simplex, whose geometries are self-priored — none needs
        # a coefficient prior. Monotonic (mo) terms DO take a free
        # coefficient, so they stay in the addressee set. Matrix terms
        # expand to their per-element addressees (one PopulationPrior row
        # per matrix column).
        addressees = Set{Symbol}()
        for t in pred.terms
            (t.kind === OffsetTerm || t.kind === LatentTerm ||
                t.kind === VaryingEffectTerm ||
                t.kind === SplineSummandTerm ||
                t.kind === HSGPSummandTerm ||
                t.kind === ScanSummandTerm ||
                t.kind === MonotonicSummandTerm ||
                t.kind === DarSummandTerm) && continue
            if t.kind === MatrixTerm
                m = _find_matrix(plan, t.options.matrix)
                m === nothing && _fail(:plan,
                    "internal: matrix term $(t.label) addresses unknown " *
                    "matrix (validate_predictors should have caught this)")
                union!(addressees, _matrix_element_addressees(m))
                continue
            end
            push!(addressees, t.addressee)
        end
        any(t -> t.kind === InterceptTerm, pred.terms) && push!(addressees, :Intercept)
        for a in addressees
            (pred.name, a) in seen ||
                _fail(:plan, "no prior for ($(pred.name), $a)")
        end
    end
    for r in plan.responses
        _is_glm_family(r.family) || continue
        m = _find_matrix(plan, r.predictor)
        m === nothing && continue
        for c in m.columns
            c === nothing && continue
            (r.label, c) in seen ||
                _fail(:plan, "no prior for ($(r.label), $c)")
        end
    end
    return nothing
end

# R2D2 structural checks: predictor linkage (one per predictor),
# parameter families (Beta R2, simplex phi, half-Normal-or-literal tau),
# and override addressees. Share counts need design widths, so they wait
# for `_validate_r2d2_data`.
function _validate_r2d2(plan::StructuralPlan)
    seen = Set{Symbol}()
    for rp in plan.r2d2_priors
        pred = nothing
        for p in plan.predictors
            p.name === rp.predictor && (pred = p)
        end
        pred === nothing && _fail(:plan,
            "R2D2 prior addresses unknown predictor $(rp.predictor)")
        rp.predictor in seen && _fail(:plan,
            "duplicate R2D2 prior for predictor $(rp.predictor) " *
            "(one per predictor)")
        push!(seen, rp.predictor)
        r2 = nothing
        for p in plan.parameters
            p.name === rp.r2 && (r2 = p)
        end
        r2 === nothing && _fail(rp.predictor,
            "R2D2 R2 parameter $(rp.r2) is not a sampled parameter")
        r2.family === :beta || _fail(rp.predictor,
            "R2D2 R2 parameter $(rp.r2) must be Beta, got $(r2.family)")
        # phi linkage + family ride _validate_vector_parameters; tau:
        if rp.tau isa Symbol
            tau = nothing
            for p in plan.parameters
                p.name === rp.tau && (tau = p)
            end
            tau === nothing && _fail(rp.predictor,
                "R2D2 tau $(rp.tau) names neither a sampled parameter " *
                "nor a literal (data tau_bsv inlines as a literal)")
            (tau.family === :normal && tau.support_override === :positive) ||
                _fail(rp.predictor,
                    "sampled R2D2 tau $(rp.tau) must be half-Normal " *
                    "(`HalfNormal(s)`), got $(tau.family) with " *
                    "override $(repr(tau.support_override))")
        else
            isfinite(rp.tau) && rp.tau > 0 || _fail(rp.predictor,
                "literal R2D2 tau must be finite and strictly positive " *
                "(SB `_sb_r2d2_positive`), got $(repr(rp.tau))")
        end
        allowed = Set{Symbol}()
        for t in pred.terms
            if t.kind === MatrixTerm
                m = _find_matrix(plan, t.options.matrix)
                m === nothing && _fail(:plan,
                    "internal: matrix term $(t.label) addresses unknown " *
                    "matrix (validate_predictors should have caught this)")
                union!(allowed, _matrix_element_addressees(m))
                continue
            end
            push!(allowed, t.addressee)
        end
        any(t -> t.kind === InterceptTerm, pred.terms) &&
            push!(allowed, :Intercept)
        for (addr, (loc, sca)) in rp.overrides
            addr in allowed || _fail(rp.predictor,
                "R2D2 override addresses $addr, not a column of " *
                "predictor $(rp.predictor)")
            isfinite(loc) && isfinite(sca) && sca > 0 || _fail(rp.predictor,
                "R2D2 override for $addr must be Normal(finite, " *
                "positive), got ($(repr(loc)), $(repr(sca)))")
        end
    end
    return nothing
end

# R2D2 data checks: the share composition needs design widths + bound
# columns. Runs at bind (after levelmaps bind, after phi size inference).
function _validate_r2d2_data(plan::StructuralPlan)
    isempty(plan.r2d2_priors) && return nothing
    for rp in plan.r2d2_priors
        pred = only(p for p in plan.predictors if p.name === rp.predictor)
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        share, _, _, varx =
            r2d2_column_scales(shape, plan.columns, rp.overrides)
        n_shares = isempty(share) ? 0 : maximum(share)
        n_shares == 0 && _fail(rp.predictor,
            "R2D2 over predictor $(rp.predictor) decomposes nothing " *
            "(intercept-only or every column overridden) — the flat " *
            "slice has no hierarchical-lane rule to no-op for")
        phi = only(p for p in plan.vector_parameters if p.name === rp.phi)
        phi.size == n_shares || _fail(rp.predictor,
            "R2D2 phi $(rp.phi) has $(phi.size) shares but predictor " *
            "$(rp.predictor) decomposes $n_shares columns")
        for j in eachindex(share)
            share[j] == 0 && continue
            isfinite(varx[j]) && varx[j] > 0 || _fail(rp.predictor,
                "R2D2 column $j of predictor $(rp.predictor) has " *
                "non-positive variance $(repr(varx[j])) — a constant " *
                "column cannot join the simplex (drop it or give it " *
                "an explicit Normal prior)")
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
    isempty(r.extra_responses) ||
        _fail(r.label, "only joint MvNormalCholesky takes extra_responses")
    r.factor_scales === nothing ||
        _fail(r.label, "only joint MvNormalCholesky takes factor_scales")
    r.factor_corr === nothing ||
        _fail(r.label, "only joint MvNormalCholesky takes factor_corr")
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
    isempty(r.extra_responses) ||
        _fail(r.label, "CategoricalLogit takes no extra_responses")
    r.factor_scales === nothing ||
        _fail(r.label, "CategoricalLogit takes no factor_scales")
    r.factor_corr === nothing ||
        _fail(r.label, "CategoricalLogit takes no factor_corr")
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
    isempty(r.extra_responses) ||
        _fail(r.label, "an ordered response takes no extra_responses")
    r.factor_scales === nothing ||
        _fail(r.label, "an ordered response takes no factor_scales")
    r.factor_corr === nothing ||
        _fail(r.label, "an ordered response takes no factor_corr")
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
    isempty(r.extra_responses) ||
        _fail(r.label, "a simplex response takes no extra_responses")
    r.factor_scales === nothing ||
        _fail(r.label, "a simplex response takes no factor_scales")
    r.factor_corr === nothing ||
        _fail(r.label, "a simplex response takes no factor_corr")
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

# Joint correlated-outcomes responses (SB
# `[y1..yK] ~ MvNormalCholesky([mu1..muK], L)`): K outcome columns (lead +
# tail) with K identity-link mean predictors (lead + `extra_predictors`,
# reused from the CategoricalLogit shape) and one LKJ factor (scales +
# Cholesky pieces linked explicitly). Widths are structural (K =
# 1 + length(extra_responses)) — no `n_levels`, no bind-time inference.
# Scalar/data-column means route through offset-only predictors
# emitter-side; row weights stay planned (no SB joint semantics to mirror).
function _validate_joint_response(r::LikelihoodSpec, plan::StructuralPlan,
        used_predictors::Set{Symbol})
    r.link === IdentityLink || _fail(r.label,
        "a joint response uses IdentityLink (means enter the MvNormal " *
        "directly — no link applies), got $(r.link)")
    outcomes = [r.response; r.extra_responses...]
    length(unique(outcomes)) == length(outcomes) ||
        _fail(r.label, "joint outcome columns repeat a column " *
            "($(join(outcomes, ", ")))")
    K = length(outcomes)
    preds = [r.predictor; r.extra_predictors...]
    length(preds) == K || _fail(r.label,
        "joint response has $K outcomes but $(length(preds)) mean " *
        "predictors (one identity-link predictor per outcome)")
    length(unique(preds)) == length(preds) ||
        _fail(r.label, "joint mean predictors repeat a predictor " *
            "($(join(preds, ", "))) — one linear predictor per outcome")
    for q in preds
        i = findfirst(p -> p.name === q, plan.predictors)
        i === nothing && _fail(r.label,
            "joint mean predictor $q is not a plan predictor")
        plan.predictors[i].link === IdentityLink || _fail(r.label,
            "joint mean predictor $q must carry IdentityLink (got " *
            "$(plan.predictors[i].link)) — means enter the MvNormal directly")
        push!(used_predictors, q)
    end
    r.factor_scales === nothing && _fail(r.label,
        "a joint response requires its factor_scales vector parameter")
    r.factor_corr === nothing && _fail(r.label,
        "a joint response requires its factor_corr vector parameter")
    si = findfirst(p -> p.name === r.factor_scales, plan.vector_parameters)
    si === nothing && _fail(r.label,
        "factor_scales $(r.factor_scales) is not a vector parameter")
    ci = findfirst(p -> p.name === r.factor_corr, plan.vector_parameters)
    ci === nothing && _fail(r.label,
        "factor_corr $(r.factor_corr) is not a vector parameter")
    sp, cp = plan.vector_parameters[si], plan.vector_parameters[ci]
    sp.family === :positive_exponential || _fail(r.label,
        "factor_scales $(sp.name) must be :positive_exponential, got " *
        "$(sp.family)")
    cp.family === :cholesky_corr_lkj || _fail(r.label,
        "factor_corr $(cp.name) must be :cholesky_corr_lkj, got " *
        "$(cp.family)")
    sp.size == K || _fail(r.label,
        "factor_scales size $(sp.size) disagrees with the $K joint outcomes")
    cp.size == K || _fail(r.label,
        "factor_corr size $(cp.size) disagrees with the $K joint outcomes")
    r.n_levels === nothing ||
        _fail(r.label,
            "a joint response takes no n_levels (widths are structural)")
    r.thresholds === nothing ||
        _fail(r.label, "a joint response takes no thresholds")
    isempty(r.count_columns) ||
        _fail(r.label, "a joint response takes no count_columns")
    r.ordinal_structure === nothing ||
        _fail(r.label, "a joint response takes no ordinal_structure")
    r.discrimination === nothing ||
        _fail(r.label, "a joint response takes no discrimination")
    isempty(r.threshold_columns) ||
        _fail(r.label, "a joint response takes no threshold_columns")
    r.threshold_coefs === nothing ||
        _fail(r.label, "a joint response takes no threshold_coefs")
    r.trials === nothing ||
        _fail(r.label, "a joint response takes no trials")
    r.weights === nothing ||
        _fail(r.label, "joint responses take no weights " *
            "(row weights on a joint density are planned)")
    if r.range !== nothing
        first(r.range) == 1 || _fail(r.label,
            "response range must start at 1 (got $(r.range)) — " *
            "ranges cover eachindex exactly, no partial windows")
        length(r.range) >= 1 || _fail(r.label,
            "response range $(r.range) is empty")
    end
    return nothing
end

function _validate_glm_response(r::LikelihoodSpec, plan::StructuralPlan)
    want_link = r.family === NormalIDGLMFam ? IdentityLink :
        r.family === BernoulliLogitGLMFam ? LogitLink : LogLink
    r.link === want_link || _fail(r.label,
        "a GLM-object response uses its canonical link (got $(r.link))")
    m = _find_matrix(plan, r.predictor)
    m === nothing && _fail(r.label,
        "GLM-object response addresses $(r.predictor), which is not a " *
        "bound design matrix (`X = hcat(...)`)")
    any(c -> c === nothing, m.columns) && _fail(r.label,
        "GLM-object design matrix $(m.name) has an intercept-ones " *
        "position — pass intercept-free X and a separate alpha")
    r.glm_alpha === nothing && _fail(r.label,
        "a GLM-object response requires its scalar intercept parameter")
    ai = findfirst(p -> p.name === r.glm_alpha, plan.parameters)
    ai === nothing && _fail(r.label,
        "GLM intercept :$(r.glm_alpha) must name a scalar sampled " *
        "parameter (`$(r.glm_alpha) ~ Normal(0, 10)`)")
    r.glm_beta === nothing && _fail(r.label,
        "a GLM-object response requires its coefficient-vector parameter")
    length(m.columns) >= 1 || _fail(r.label,
        "GLM-object design matrix $(m.name) has no columns")
    r.n_levels === nothing ||
        _fail(r.label, "a GLM-object response takes no n_levels")
    r.thresholds === nothing ||
        _fail(r.label, "a GLM-object response takes no thresholds")
    isempty(r.extra_predictors) ||
        _fail(r.label, "a GLM-object response takes no extra_predictors")
    isempty(r.count_columns) ||
        _fail(r.label, "a GLM-object response takes no count_columns")
    r.ordinal_structure === nothing ||
        _fail(r.label, "a GLM-object response takes no ordinal_structure")
    r.discrimination === nothing ||
        _fail(r.label, "a GLM-object response takes no discrimination")
    isempty(r.threshold_columns) ||
        _fail(r.label, "a GLM-object response takes no threshold_columns")
    r.threshold_coefs === nothing ||
        _fail(r.label, "a GLM-object response takes no threshold_coefs")
    isempty(r.extra_responses) ||
        _fail(r.label, "a GLM-object response takes no extra_responses")
    r.factor_scales === nothing ||
        _fail(r.label, "a GLM-object response takes no factor_scales")
    r.factor_corr === nothing ||
        _fail(r.label, "a GLM-object response takes no factor_corr")
    r.trials === nothing ||
        _fail(r.label, "a GLM-object response takes no trials")
    r.weights === nothing ||
        _fail(r.label, "GLM-object responses take no weights " *
            "(weighted GLMs stay on the predictor path)")
    r.range === nothing ||
        _fail(r.label, "a GLM-object response takes no range " *
            "(ranged responses stay on the predictor path)")
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
        # A scale predictor feeds a slot exactly like a location predictor,
        # so it counts toward the unused-predictor check below.
        r.scale isa ScalePredictorRef &&
            push!(used_predictors, r.scale.predictor)
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
        # A per-cell latent location (no linear predictor — the scan-state
        # precedent): a latent-mean observation `x_obs ~ Normal(x_true, sd)`
        # with scalar constant `sd` (the SB `me` mirror). Gaussian-identity
        # only; SB's observation likelihood is never weighted, truncated,
        # or ranged, so those fields fail closed.
        if _is_plate_param(plan, r.predictor)
            (r.family === GaussianFam && r.link === IdentityLink) || _fail(
                r.label,
                "a plate-mean response location ($(r.predictor)) is " *
                "Gaussian-identity only (got $(r.family)/$(r.link))",
            )
            r.scale isa Real || _fail(r.label,
                "a plate-mean observation ($(r.predictor)) takes a scalar " *
                "constant scale (SB `me` sd — got " *
                "$(r.scale === nothing ? "nothing" : repr(r.scale)))")
            r.weights === nothing || _fail(r.label,
                "a plate-mean observation ($(r.predictor)) takes no weights")
            r.evidence.kind === :none || _fail(r.label,
                "a plate-mean observation ($(r.predictor)) takes no " *
                "censoring/truncation evidence")
            r.range === nothing || _fail(r.label,
                "a plate-mean observation ($(r.predictor)) takes no range " *
                "(the latent covers the whole column)")
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
        # A joint correlated-outcomes response (K outcomes, K mean
        # predictors, one LKJ factor): validated whole, skipping the
        # single-predictor triple.
        if r.family === MvNormalCholeskyFam
            _validate_joint_response(r, plan, used_predictors)
            _validate_scale(r, plan)
            _validate_evidence_structure(r, plan)
            continue
        end
        # A GLM-object response (whole-data head over a design matrix —
        # the object owns eta, so no PredictorSpec): validated whole,
        # skipping the single-predictor triple.
        if _is_glm_family(r.family)
            _validate_glm_response(r, plan)
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
    for kp in plan.kernel_plates
        for (p, _) in kp.lp_args
            push!(used_predictors, p)
        end
    end
    for pred in plan.predictors
        pred.name in used_predictors ||
            _fail(pred.label, "predictor $(pred.name) unused by any " *
                  "response or kernel LP arg")
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
    col = _vector_column(plan.columns, r.response, r.label, "response")
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
    elseif r.family === NormalIDGLMFam
        eltype(col) <: Real ||
            _fail(r.label, "NormalIDGLM response must be numeric")
        return nothing
    elseif r.family === BernoulliLogitGLMFam
        eltype(col) === Bool && return nothing
        eltype(col) <: Integer && all(x -> x == 0 || x == 1, col) && return nothing
        return _fail(r.label, "BernoulliLogitGLM response must be Bool or 0/1 integers")
    elseif r.family === PoissonLogGLMFam
        eltype(col) <: Integer && all(>=(0), col) && return nothing
        return _fail(r.label, "PoissonLogGLM response must be non-negative integers")
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
            countcol = _vector_column(plan.columns, c, r.label, "count column")
            _is_count_column(countcol) ||
                return _fail(r.label, "count column $c must hold " *
                    "non-negative integers")
        end
        return nothing
    elseif r.family === MvNormalCholeskyFam
        # The joint outcomes cross as K raw numeric columns (lead + tail),
        # row-aligned by the uniform-`n_obs` rule (SB packs complete aligned
        # rows emitter-side; missingness fails closed in `_validate_columns`).
        for c in [r.response; r.extra_responses...]
            _is_derived(plan, c) && return _fail(r.label,
                "joint outcome $c is derived — joint outcomes bind raw " *
                "(derived outcomes need shape metadata — planned)")
            haskey(plan.columns, c) ||
                return _fail(r.label, "joint outcome column $c missing")
            outcol = _vector_column(plan.columns, c, r.label, "joint outcome")
            eltype(outcol) <: Real ||
                return _fail(r.label, "joint outcome $c must be numeric")
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
        r.family === BetaLogitFam ? "Beta response requires a concentration kappa" :
        r.family === NormalIDGLMFam ? "NormalIDGLM response requires a scale sigma" : nothing
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
    s isa ScalePredictorRef && return _validate_scale_predictor(r, plan, s)
    # A scalar parameter/assignment scale resolves now; a per-observation scale
    # is a raw data column resolved at bind (see `_validate_scale_data`), so
    # defer an unknown symbol rather than failing structurally (mirrors how
    # per-obs weight/trials columns validate only once data is attached).
    s isa Symbol && s in _union_names(plan) && return nothing
    s isa Symbol && return nothing
    return _fail(r.label, "scale references unknown name $s")
end

# A predictor-fed scale/shape use (Gaussian sigma, NB2 phi, Gamma alpha):
# the predictor exists, carries the use-site link (the
# one-link-per-predictor rule), and is not the response's own location
# predictor (the two slots take distinct predictors — the BRM-side plan
# rule, mirrored here as defense in depth). Beta-kappa predictors are
# deferred; predictor-fed Binomial trials likewise (trials stay
# column-or-literal by type).
function _validate_scale_predictor(r::LikelihoodSpec, plan::StructuralPlan,
        s::ScalePredictorRef)
    r.family === BetaLogitFam && _fail(r.label,
        "Beta response with a scale predictor: predictor-fed concentration " *
        "(kappa) is deferred — use a scalar kappa (parameter or literal)")
    (r.family === GaussianFam || r.family === NegativeBinomial2Fam ||
        r.family === GammaLogFam) ||
        _fail(r.label, "this response family takes no scale predictor")
    (s.link === IdentityLink || s.link === LogLink ||
        s.link === LogitLink) ||
        _fail(r.label, "scale predictor link must be identity, log, or " *
            "logit (got $(s.link))")
    idx = findfirst(p -> p.name === s.predictor, plan.predictors)
    idx === nothing && _fail(r.label,
        "scale addresses unknown predictor $(s.predictor)")
    pred = plan.predictors[idx]
    pred.link === s.link ||
        _fail(r.label, "scale predictor $(s.predictor) carries link " *
            "$(pred.link), scale use wraps $(s.link) — one link per predictor")
    s.predictor === r.predictor &&
        _fail(r.label, "scale predictor $(s.predictor) is the response's " *
            "own location predictor — location and scale take distinct " *
            "predictors")
    return nothing
end

# Data-level per-observation scale check: a scalar parameter/assignment name
# resolves structurally; a raw data-column scale (the eight-schools known SE)
# must be finite-positive numerics of length n_obs (a Gaussian/NB2/Gamma scale
# is strictly positive). A derived column scale is rejected — per-obs scales
# bind raw (mirrors the weights/trials raw-only rule).
function _validate_scale_data(r::LikelihoodSpec, plan::StructuralPlan)
    s = r.scale
    (s === nothing || s isa Real) && return nothing
    # A predictor-fed scale is an n_obs LP by construction (design over the
    # bound rows); there is no column length to check at bind.
    s isa ScalePredictorRef && return nothing
    s isa Symbol || return nothing
    s in _union_names(plan) && return nothing
    _is_derived(plan, s) && _fail(r.label,
        "scale column $s is derived — slice-1 binds per-observation scales " *
        "raw (derived-column scales need shape metadata — planned)")
    haskey(plan.columns, s) ||
        _fail(r.label, "scale references unknown name $s")
    col = _vector_column(plan.columns, s, r.label, "scale column")
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
    ycol = _vector_column(plan.columns, r.response, r.label, "response")
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
    col = _vector_column(plan.columns, t, r.label, "trials column")
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
    col = _vector_column(plan.columns, t, r.label, "trials column")
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
        col = _vector_column(plan.columns, d, r.label, "discrimination column")
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
        col = _vector_column(plan.columns, c, r.label, "threshold column")
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
    col = _vector_column(plan.columns, r.weights, r.label, "weights column")
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
        resp = _vector_column(plan.columns, r.response, r.label, "response")
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
    col = _vector_column(plan.columns, bound, r.label, "$side bound column")
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
"""One HSGP axis fit (SB `_brm_fit_hsgp` 1-D verbatim): `mu =
mean(col)`, `L = c*max|col-mu|`, numeric/nonempty/finite/`L > 0`
gates. Shared by basis binds and the event-LP bind (dev §4 — one
fit core, two callers)."""
function _hsgp_axis_fit(col::AbstractVector, c::Real, label::Symbol,
        where::String)
    eltype(col) <: Real ||
        _fail(label, "$where must be numeric, got $(eltype(col))")
    isempty(col) &&
        _fail(label, "$where is empty")
    mu = sum(col) / length(col)
    L = Float64(c) * maximum(abs.(col .- mu))
    isfinite(mu) && isfinite(L) ||
        _fail(label, "$where is non-finite (mu=$mu, L=$L)")
    L > 0 ||
        _fail(label, "$where is degenerate " *
              "(L == 0 — a constant column has no usable domain)")
    return (Float64(mu), L)
end

function _fit_hsgp_bases(plan::StructuralPlan,
        columns::AbstractDict{Symbol})
    isempty(plan.hsgp_bases) && return HSGPBasis[]
    out = HSGPBasis[]
    for hb in plan.hsgp_bases
        fits = Tuple{Float64,Float64}[]
        for (c, cj) in zip(hb.axes, hb.c)
            haskey(columns, c) ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c is " *
                      "not bound")
            col = _vector_column(columns, c, hb.label, "hsgp axis column")
            push!(fits, _hsgp_axis_fit(col, cj, hb.label,
                "hsgp :$(hb.id): axis column $c"))
        end
        push!(out, HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, fits,
            hb.label))
    end
    return out
end

function _materialize_splines!(plan::StructuralPlan,
        columns::Dict{Symbol,ColumnData})
    isempty(plan.spline_bases) && return SplineBasis[]
    out = SplineBasis[]
    for sb in plan.spline_bases
        axes = AbstractVector[]
        for c in sb.axes
            haskey(columns, c) ||
                _fail(sb.label, "spline :$(sb.id): axis column $c is " *
                      "not bound")
            axiscol =
                _vector_column(columns, c, sb.label, "spline axis column")
            eltype(axiscol) <: Real ||
                _fail(sb.label, "spline :$(sb.id): axis column $c must " *
                      "be numeric, got $(eltype(axiscol))")
            push!(axes, axiscol)
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

# Result space of each CELL_FNS entry (:reads = flat read-space,
# :obs = obs-space, :scalar). The generator lowers reads per subject
# and vcats; gathers move reads to obs. The AUC cell returns the flat
# per-subject `[conc; auc]` blocks (length 2R) — still read-space
# (gathers move it to an axis first).
const CELL_FN_RESULT_SPACE = Dict{Symbol,Symbol}(
    :linear_pk_read_locs => :reads, :linear_pk_read_locs_auc => :reads)

# Bind-time grouped cell shapes: :scalar (LP cell params, model
# scalars, literals, scalar arithmetic), (:obs, len) (response slices
# at column length, gathers at map length, dotted obs arithmetic at
# the common axis length), :reads (cell-call results). Axis IS length
# (two foreign axes with equal row counts mix silently — the accepted
# hole: even Stan would not catch it; the fixed joint program + parity
# tests guard the deliverable). Read-space arithmetic fails closed
# (gather to the obs axis first — the SB shape); undotted ops over
# obs/read operands fail closed (write the dotted form — the panel
# precedent); dotted ops over mismatched axis lengths fail closed
# naming the axes. Assignments pass through UNCHANGED (no flat dotify
# — the generator unrolls per subject); this pass only proves shapes.
function _grouped_cell_shapes(kp::KernelPlate,
        slices::Vector{Tuple{Symbol,Symbol,Symbol}},
        columns::Dict{Symbol,ColumnData})
    shapes = Dict{Symbol,Any}()
    for (col, p, _) in slices
        shapes[p] = (:obs, length(columns[col]))
    end
    for (_, c) in kp.lp_args
        shapes[c] = :scalar
    end
    for (nm, ex) in kp.assignments
        shapes[nm] = _grouped_cell_shape(ex, kp, shapes, columns)
    end
    return shapes
end

"""Whether a grouped shape is an obs-axis shape (vs :scalar/:reads)."""
_is_obs_shape(s) = s isa Tuple && s[1] === :obs

# One dotted operation's result shape (shared by dotted-operator
# `:call`s and dotted-math `:.`s): read-space fails closed (gather
# first); obs operands must share one axis length (mismatch fails
# closed naming the axes); scalars broadcast.
function _dotted_obs_shape(label::Symbol, ex::Expr, spaces::Vector)
    any(==(:reads), spaces) &&
        _fail(label, "read-space arithmetic does not lower — " *
              "gather to the obs axis first " *
              "(`reads[sched.obs_map]`), then compute")
    lens = sort!(unique!([s[2] for s in spaces if _is_obs_shape(s)]))
    if length(lens) > 1
        _fail(label, "dotted operation mixes obs axes of length " *
              join(lens, " and ") * " (operands must share one " *
              "axis — gather each series to its axis first; got " *
              "$(repr(ex)))")
    end
    return isempty(lens) ? :scalar : (:obs, only(lens))
end

function _grouped_cell_shape(ex, kp::KernelPlate, shapes::Dict{Symbol,Any},
        columns::Dict{Symbol,ColumnData})
    label = kp.label
    ex isa Number && return :scalar
    ex isa LineNumberNode && return :scalar
    # Model scalars (params/assignments) default scalar — structure
    # proved every other Symbol is a cell name in `shapes`.
    ex isa Symbol && return get(shapes, ex, :scalar)
    head = ex.head
    if head === :call
        fn = ex.args[1]
        if fn isa Symbol && fn in CELL_FNS
            trailing = ex.args[3:end]
            if (fn === :linear_pk_read_locs &&
                    length(ex.args) == CELL_FN_ARITY[fn] + 2) ||
                    fn === :linear_pk_read_locs_auc
                # The event-LP second arg is the provider's flat
                # event-axis vector (structure proved the name), not a
                # scalar — the LP scalars start one later.
                trailing = ex.args[4:end]
            end
            for arg in trailing
                aspace = _grouped_cell_shape(arg, kp, shapes, columns)
                aspace === :scalar ||
                    _fail(label, "cell call `$fn` argument `$(arg)` is " *
                          "$aspace-space (call args past the schedule " *
                          "are per-subject LP cell params or model " *
                          "scalars)")
            end
            return CELL_FN_RESULT_SPACE[fn]
        end
        if fn isa Symbol && fn in SEGMENT_CELL_FNS
            # The nadir runs over one obs-axis row series (one entry
            # per row out); the ends column's segment contract is
            # proved by the nadir validator (bound plans).
            changeshape =
                _grouped_cell_shape(ex.args[2], kp, shapes, columns)
            changeshape === :reads &&
                _fail(label, "cell call `$fn` change argument is " *
                      "read-space — gather to the obs axis first " *
                      "(`reads[sched.obs_map]` / `v[map]`)")
            _is_obs_shape(changeshape) ||
                _fail(label, "cell call `$fn` change argument is " *
                      "scalar (the nadir runs over a row series)")
            endscol = ex.args[3]
            haskey(columns, endscol) ||
                _fail(label, "cell call `$fn` ends `$endscol` is not " *
                      "bound (bind_data columns carry it)")
            return (:obs, changeshape[2])
        end
        # Dotted operators (`mu .+ e`): scalars broadcast; obs
        # operands must share one axis length (mismatch fails closed
        # naming the axes); read-space fails closed (gather first).
        if fn isa Symbol && fn in ELEMENTWISE_OPS
            spaces = [_grouped_cell_shape(a, kp, shapes, columns)
                      for a in ex.args[2:end]]
            return _dotted_obs_shape(label, ex, spaces)
        end
        # Undotted arithmetic (operator or math function): scalar-only.
        # Grouped cells emit verbatim (no flat dotify), so an undotted
        # obs operand fails closed (write the dotted form) and a
        # read-space operand fails closed (gather to the obs axis
        # first).
        if fn isa Symbol && fn in ASSIGNMENT_FNS
            spaces = [_grouped_cell_shape(a, kp, shapes, columns)
                      for a in ex.args[2:end]]
            any(==(:reads), spaces) &&
                _fail(label, "read-space arithmetic does not lower — " *
                      "gather to the obs axis first " *
                      "(`reads[sched.obs_map]`), then compute")
            any(_is_obs_shape, spaces) &&
                _fail(label, "undotted `$fn` over obs-space series does " *
                      "not lower — write the dotted form")
            return :scalar
        end
        return _fail(label, "call `$fn` has no grouped shape rule " *
                            "(internal: structure validation admits it " *
                            "but bind does not)")
    end
    if head === :.
        spaces = [_grouped_cell_shape(a, kp, shapes, columns)
                  for a in ex.args[2].args]
        return _dotted_obs_shape(label, ex, spaces)
    end
    if head === :ref
        src, idx = ex.args[1], ex.args[2]
        srcshape = _grouped_cell_shape(src, kp, shapes, columns)
        # Response slices are leaves (structure admits the shape;
        # bind proves the space — the v1 pin).
        src isa Symbol && src in _slice_params(kp) &&
            _fail(label, "gather source `$src` is not read-space " *
                  "(response slices are leaves — gathers read " *
                  "cell-call results, LP cell params, or cell locals: " *
                  "`reads[sched.obs_map]` / `v[map]`)")
        is_lp = src isa Symbol && src in _lp_cell_params(kp)
        (srcshape === :reads || _is_obs_shape(srcshape) || is_lp) ||
            _fail(label, "gather source `$src` is scalar (gathers read " *
                  "cell-call results, LP cell params, or cell locals: " *
                  "`reads[sched.obs_map]` / `v[map]`)")
        mapcol = idx isa Symbol ? idx :
            _sched_col_name(idx.args[1], idx.args[2].value)
        haskey(columns, mapcol) ||
            _fail(label, "gather map `$mapcol` is not bound " *
                  "(bind_data columns carry gather maps)")
        return (:obs, length(columns[mapcol]))
    end
    return _fail(label, "unsupported expression head $head in a grouped " *
                        "cell (internal)")
end

# Per-obs axis agreement: each in-cell observation's response sets the
# axis length; location/scale/params refs that are non-scalar must
# match it exactly (literals and model scalars broadcast). Structure
# proved every Symbol is a cell name or a model scalar; LP cell params
# as obs args already failed there (gather explicitly).
function _validate_grouped_obs_axes(kp::KernelPlate,
        shapes::Dict{Symbol,Any}, columns::Dict{Symbol,ColumnData})
    for obs in kp.obs
        rcol = only(c for (c, p, _) in kp.slices if p === obs.response)
        rlen = length(columns[rcol])
        refs = Any[(:location, obs.location), (:scale, obs.scale)]
        append!(refs, [(:params, pr) for pr in obs.params])
        for (nm, ref) in refs
            ref isa Number && continue
            s = get(shapes, ref, :scalar)
            s === :scalar && continue
            s === :reads &&
                _fail(kp.label, "kernel obs $nm `$ref` is read-space " *
                      "(gather to the response axis first: " *
                      "`reads[sched.obs_map]` / `v[map]`)")
            s[2] == rlen ||
                _fail(kp.label, "kernel obs $nm `$ref` has length " *
                      "$(s[2]) ≠ response `$(obs.response)` length " *
                      "$rlen (one value per response row — gather " *
                      "each arg to the response axis)")
        end
    end
    return nothing
end

# Prove grouped cell shapes + per-obs axis agreement (single site for
# bind resolution + bound-plan validation, so hand-bound plans carry
# proved shapes too — never trusted ones). Returns the shapes.
function _prove_grouped_cell_shapes(kp::KernelPlate,
        slices::Vector{Tuple{Symbol,Symbol,Symbol}},
        columns::Dict{Symbol,ColumnData})
    shapes = _grouped_cell_shapes(kp, slices, columns)
    _validate_grouped_obs_axes(kp, shapes, columns)
    return shapes
end

# Segmented-nadir ends contract (bound plans): each nadir call's ends
# column is integer-valued with one cumulative end per subject,
# nondecreasing from a non-negative start, last end == change length
# (shapes proved the ends bound + the change an obs series, so the
# lengths below are total).
function _validate_nadir_ends(kp::KernelPlate, shapes::Dict{Symbol,Any},
        columns::Dict{Symbol,ColumnData})
    n_sub = kp.subjects
    for (nm, ex) in kp.assignments
        ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] isa Symbol && ex.args[1] in SEGMENT_CELL_FNS ||
            continue
        fn = ex.args[1]
        endscol = ex.args[3]
        ends = columns[endscol]
        eltype(ends) <: Integer ||
            _fail(kp.label, "cell call `$fn` ends `$endscol` must be " *
                  "an integer column, got $(eltype(ends))")
        length(ends) == n_sub ||
            _fail(kp.label, "cell call `$fn` ends `$endscol` has length " *
                  "$(length(ends)) ≠ subjects $n_sub (one cumulative " *
                  "end per subject)")
        issorted(ends) ||
            _fail(kp.label, "cell call `$fn` ends `$endscol` must be " *
                  "nondecreasing (got $ends)")
        ends[1] >= 0 ||
            _fail(kp.label, "cell call `$fn` ends `$endscol` must be " *
                  "non-negative (got $ends)")
        n_rows = shapes[nm][2]
        ends[end] == n_rows ||
            _fail(kp.label, "cell call `$fn` ends `$endscol` last end " *
                  "$(ends[end]) ≠ change length $n_rows")
    end
    return nothing
end

# Gather-map prep contract (bound plans): every gather's map column is
# integer-valued, with values inside the source's index range —
# subject maps (LP-vector sources) within 1..n_sub, read-space maps
# within the flat reads (R for v1 cells, 2R for AUC cells, from the
# built schedule), obs-axis maps within the source axis length. Shapes
# proved bound-ness + source spaces; this proves content.
function _validate_grouped_gather_maps(kp::KernelPlate,
        shapes::Dict{Symbol,Any}, columns::Dict{Symbol,ColumnData},
        n_reads_total::Int)
    n_sub = kp.subjects
    for (_, ex) in kp.assignments
        for (src, mapcol) in _collect_grouped_gather_uses(ex)
            mapv = columns[mapcol]
            eltype(mapv) <: Integer ||
                _fail(kp.label, "gather map `$mapcol` must be an " *
                      "integer column, got $(eltype(mapv))")
            if src in _lp_cell_params(kp)
                all(1 .<= mapv .<= n_sub) ||
                    _fail(kp.label, "subject map `$mapcol` has entries " *
                          "outside 1..$n_sub (gathered `$src` is " *
                          "per-subject — one subject id per row)")
            else
                srcshape = shapes[src]
                bound = srcshape === :reads ?
                    _gather_reads_bound(kp, src, n_reads_total) :
                    srcshape[2]
                all(1 .<= mapv .<= bound) ||
                    _fail(kp.label, "gather map `$mapcol` has entries " *
                          "outside 1..$bound (gathered `$src` has " *
                          "$bound rows)")
            end
        end
    end
    return nothing
end

# Flat reads length behind a read-space gather source: R for v1 cells,
# 2R for AUC cells (per-subject `[conc; auc]` blocks). Shapes proved
# the source is a cell-call result, so the producing call exists.
function _gather_reads_bound(kp::KernelPlate, src::Symbol, n_reads_total::Int)
    for (nm, ex) in kp.assignments
        if nm === src && ex isa Expr && ex.head === :call
            return ex.args[1] === :linear_pk_read_locs_auc ?
                2 * n_reads_total : n_reads_total
        end
    end
    _fail(kp.label, "gather source `$src` has no producing cell call " *
          "(internal)")
end

# Every gather use `(source, map-column)` in a cell RHS (sources are
# cell names, maps schedule maps or plain bind columns — structure
# proved the shapes).
function _collect_grouped_gather_uses(ex)
    out = Tuple{Symbol,Symbol}[]
    _collect_grouped_gather_uses!(out, ex)
    return out
end

function _collect_grouped_gather_uses!(out::Vector{Tuple{Symbol,Symbol}}, ex)
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 2
        src, idx = ex.args[1], ex.args[2]
        mapcol = idx isa Symbol ? idx :
            _sched_col_name(idx.args[1], idx.args[2].value)
        push!(out, (src, mapcol))
    end
    for a in ex.args
        _collect_grouped_gather_uses!(out, a)
    end
    return nothing
end

# TGI axis order (bound plans, tgi declared): rows grouped by subject
# with nondecreasing times within each subject (SB `_joint_tgi_rows`
# order — the bind derives cumulative ends from the subject column
# and cannot reorder caller rows, so unordered frames fail closed).
function _validate_tgi_axis_order(kp::KernelPlate,
        sched::LinearPKScheduleSpec, columns::Dict{Symbol,ColumnData})
    sched.tgi === nothing && return nothing
    tsubj = columns[sched.tgi[1]]
    ttime = columns[sched.tgi[2]]
    issorted(tsubj) ||
        _fail(kp.label, "tgi subject column `$(sched.tgi[1])` must be " *
              "grouped by subject (sort rows by (subject, time) — SB " *
              "`_joint_tgi_rows` order)")
    for s in 1:kp.subjects
        ts = [ttime[i] for i in eachindex(tsubj, ttime) if tsubj[i] == s]
        issorted(ts) ||
            _fail(kp.label, "tgi time column `$(sched.tgi[2])` must be " *
                  "nondecreasing within subject $s (the nadir runs " *
                  "over time-ordered rows)")
    end
    return nothing
end

# Grouped-kernel bind resolution: dims keys resolve `subjects` (no
# timepoints — leftover keys fail closed); the schedule builds from raw
# columns and must cover exactly n_sub subjects; op columns + maps
# materialize (`<sched>_<field>`, caller collisions fail closed);
# slices resolve to :response against the schedule's obs axis; cell
# shapes resolve (scalar/obs/reads). Returns the resolved node; the
# input plan is untouched.
function _resolve_grouped_kernel!(plan::StructuralPlan, kp::KernelPlate,
        columns::Dict{Symbol,ColumnData}, dims::AbstractDict{Symbol,<:Integer})
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
    leftovers = setdiff(Set{Symbol}(keys(dims)), consumed)
    isempty(leftovers) ||
        _fail(kp.label, "dims key(s) $(sort!(collect(leftovers))) not " *
              "consumed by kernel plate `$(kp.result)` (grouped kernels " *
              "take no timepoints dims key — typo'd key?)")
    sched = only(kp.schedules)
    built = _build_grouped_schedule(plan, kp, columns)
    built.n_subjects == n_sub ||
        _fail(kp.label, "schedule `$(sched.name)` covers " *
              "$(built.n_subjects) subjects ≠ subjects $n_sub")
    for f in vcat(collect(_SCHED_MATERIALIZED_FIELDS),
            _sched_extra_fields(sched, kp.assignments))
        col = _sched_col_name(sched.name, f)
        haskey(columns, col) &&
            _fail(kp.label, "column `$col` is reserved for schedule " *
                  "`$(sched.name)`'s bind product — rename the " *
                  "caller-supplied column")
        columns[col] = getfield(built, f)
    end
    # The event-axis product (SB `_linear_pk_dose_event_axis`): only
    # with a declared event-LP on this schedule — v1 bind products
    # stay byte-identical without one.
    for el in plan.event_lps
        el.schedule === sched.name || continue
        ecol = _sched_col_name(sched.name, :op_log_dose)
        haskey(columns, ecol) &&
            _fail(kp.label, "column `$ecol` is reserved for schedule " *
                  "`$(sched.name)`'s event-axis product — rename the " *
                  "caller-supplied column")
        et = _sched_col_name(sched.name, :op_type)
        ea = _sched_col_name(sched.name, :op_amount)
        columns[ecol] = try
            linear_pk_op_log_dose(columns[et], columns[ea])
        catch err
            err isa ContractValidationError &&
                _fail(kp.label, "schedule `$(sched.name)`: $(err.message)")
            rethrow()
        end
    end
    slices2 = Tuple{Symbol,Symbol,Symbol}[]
    for (col, param, _) in kp.slices
        haskey(columns, col) ||
            _fail(kp.label, "slice column `$col` is not bound")
        # Any length admits (foreign-axis responses ride their own
        # axis — the first-obs-on-primary rule is the n_obs check in
        # bound-plan validation); per-obs axis agreement is proved on
        # shapes below.
        push!(slices2, (col, param, :response))
    end
    _prove_grouped_cell_shapes(kp, slices2, columns)
    return KernelPlate(kp.result, n_sub, nothing, slices2, kp.assignments,
        kp.obs, kp.collected, kp.label, kp.lp_args, kp.schedules)
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
        columns::Dict{Symbol,ColumnData}, dims::AbstractDict{Symbol,<:Integer})
    isempty(plan.kernel_plates) && return KernelPlate[]
    kp = only(plan.kernel_plates)
    _is_grouped_kernel(kp) &&
        return KernelPlate[_resolve_grouped_kernel!(plan, kp, columns, dims)]
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
        colv = _vector_column(columns, col, kp.label, "slice column")
        L = length(colv)
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
            columns[exp] = repeat(Vector{Float64}(colv); inner = T)
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
assignment/extra columns stay `:data`. Varying-draws grouping columns
upgrade to `:group` (grouping dominates predictor use in the label; both
facts stay visible in terms + draws). Draws blocks with
`levels === nothing` gain sort-ordered observed levels (SB numbering for
plain vectors — emitter-declared levels pass through). `dims` binds
kernel-plate dims keys
(`subject_count`, `timepoint_count`) to positive integers; every key must
be consumed. Columns are vectors or matrices (whole-design data, Stan
`matrix[N,K]`): a matrix binds with `n_obs` rows, ≥ 1 column, and a
numeric eltype; every per-observation role reads vectors only and fails
closed on a matrix.
"""
function bind_data(plan::StructuralPlan, columns::AbstractDict{Symbol};
        roles::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
        dims::AbstractDict{Symbol,<:Integer} = Dict{Symbol,Int}())
    validate_structure(plan)
    isempty(columns) && throw(ContractValidationError(
        "[bind] bind_data requires non-empty columns"))
    columns = _checked_columns(columns)
    bases = _materialize_splines!(plan, columns)
    hbases = _fit_hsgp_bases(plan, columns)
    kbases = _resolve_kernels!(plan, columns, dims)
    elbases = _fit_event_lps(plan, columns)
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
    for d in plan.varying_draws
        haskey(inferred, d.group) && _upgrade_role!(inferred, d.group, :group)
    end
    for sb in bases, blk in sb.blocks, c in blk.columns
        haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
    end
    for hb in hbases, c in hb.axes
        haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
    end
    for el in elbases
        c = _sched_col_name(el.schedule, :op_log_dose)
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
        for c in r.extra_responses
            haskey(inferred, c) && (inferred[c] = :response)
        end
        for c in r.threshold_columns
            haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
        end
    end
    for kp in kbases
        for obs in kp.obs
            rcol = only(c for (c, p, _) in kp.slices if p === obs.response)
            haskey(inferred, rcol) && (inferred[rcol] = :response)
        end
    end
    merged = merge(inferred, roles)
    # Kernel plans carry two column lengths by design: n_obs is the flat
    # length (vector models) or n_sub (all-scalar) — never first-column.
    # Otherwise n_obs is the first column's ROW count (length for vectors).
    # Grouped plans set n_obs to the primary (first-obs) response length.
    n = if isempty(kbases)
        _column_nrows(first(values(columns)))
    else
        gkp = only(kbases)
        if _is_grouped_kernel(gkp)
            rcol0 = only(c for (c, p, _) in gkp.slices
                if p === first(gkp.obs).response)
            length(columns[rcol0])
        else
            _kernel_flat_length(gkp.subjects, gkp.timepoints)
        end
    end
    maps = _eval_levelmaps(plan.levelmaps, columns)
    draws = _eval_draws_levels(plan.varying_draws, columns)
    responses2, vectors2 =
        _infer_leveled_sizes(plan.responses, plan.vector_parameters, columns,
            plan.predictors, plan.r2d2_priors)
    bound = StructuralPlan(responses2, plan.predictors,
        plan.population_priors, plan.parameters, plan.assignments,
        columns, n; roles = merged, derived = plan.derived,
        levelmaps = maps, plate_parameters = plan.plate_parameters,
        scans = plan.scans, dar_paths = plan.dar_paths,
        varying_draws = draws, varying_slices = plan.varying_slices,
        vector_parameters = vectors2, spline_bases = bases,
        spline_vectors = plan.spline_vectors, hsgp_bases = hbases,
        kernel_plates = kbases, r2d2_priors = plan.r2d2_priors,
        matrices = plan.matrices, event_lps = elbases)
    validate_data(bound)
    return bound
end

# Bind-time level inference (the LevelMap binder-evaluation precedent):
# `n_levels === nothing` fills structurally (CategoricalLogit: 1 +
# predictor count; Multinomial: count-column count) or from the response
# column (max(y) for OrderedLogistic/Ordinal/Categorical); vector-param
# `size === nothing` fills from the linked response (K−1 thresholds, K
# simplex), from its frozen concentration length for a monotonic-linked
# increments simplex (K−1 increments for K levels) or an R2D2-linked
# share simplex (K shares). Explicit values assert against the
# inference. Returns new (immutable) vectors; unbound plans keep
# `nothing`.
function _infer_leveled_sizes(responses::Vector{LikelihoodSpec},
        vectors::Vector{VectorParameter}, columns::AbstractDict{Symbol},
        predictors::Vector{PredictorSpec} = PredictorSpec[],
        r2d2::Vector{R2D2Prior} = R2D2Prior[])
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
    r2d2_link = Dict{Symbol,Symbol}()
    for rp in r2d2
        r2d2_link[rp.phi] = rp.predictor
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
        elseif p.family in _JOINT_FACTOR_FAMILIES
            # Joint-factor sizes are structural (concrete at construction,
            # validated against the joint width) — bind passes them through.
            p.size === nothing && _fail(p.label,
                "internal: joint-factor size unresolved at bind")
            push!(out_v, p)
        elseif haskey(r2d2_link, p.name)
            want = length(p.args.arg1)
            want >= 1 || _fail(p.label,
                "R2D2 shares need ≥ 1 share (an empty concentration " *
                "decomposes nothing)")
            p.size === nothing || p.size == want || _fail(p.label,
                "R2D2 phi size $(p.size) disagrees with its " *
                "concentration length $want")
            push!(out_v, VectorParameter(p.name, p.family, p.args, want, p.label))
        else
            _fail(p.label, "internal: vector parameter unlinked at bind")
        end
    end
    return out_r, out_v
end

function _infer_response_levels(r::LikelihoodSpec, columns::AbstractDict{Symbol})
    if r.family === CategoricalLogitFam
        return 2 + length(r.extra_predictors)
    elseif r.family === MultinomialFam
        return 1 + length(r.count_columns)
    end
    r.n_levels !== nothing && return r.n_levels
    haskey(columns, r.response) ||
        _fail(r.label, "response column $(r.response) missing")
    col = _vector_column(columns, r.response, r.label, "response")
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

