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
    ContractValidationError(message::AbstractString) =
        new(_author_names(message))
end
Base.showerror(io::IO, e::ContractValidationError) =
    print(io, "ContractValidationError: ", e.message)

"""Reference to a raw data column: a key into `StructuralPlan.columns`."""
const ColumnRef = Symbol
"""Reference to a sampled parameter or scalar assignment by name."""
const ParamName = Symbol

"""Bound data values: numbers and arrays of any shape. Elementwise
observations broadcast over their operands' Julia axes; singleton axes
stretch and incompatible dimensions fail. Structured operations such as
design-matrix products, schedule slices and level tables retain their own
shape requirements. Model-level values have no observation axis."""
const ColumnData = Union{AbstractArray,Number}
# What a caller may pass at every entry point (the model call, `@rkppl data`,
# `merge` pins, `bind_data`): anything `ColumnData` holds.
const _SuppliedColumn = ColumnData

# Row count of a bound value read per observation: length for vectors, row
# count for matrices. Numbers and higher-dimensional arrays carry no
# observation axis.
_column_nrows(col::AbstractVector) = length(col)
_column_nrows(col::AbstractMatrix) = size(col, 1)

"""Normalize caller-supplied data values to the plan's value table (numbers
and arrays — anything else fails closed here, not in a converter)."""
function _checked_columns(columns::AbstractDict{Symbol})
    out = Dict{Symbol,ColumnData}()
    for (k, v) in columns
        v isa _SuppliedColumn ||
            _fail(:plan, "data value $k must be a number or an array, got " *
                         "$(summary(v))")
        out[k] = v
    end
    return out
end

# ── Scalar data values ─────────────────────────────────────────────────
# A data name whose value is a number has no observation axis: it lowers
# as the model-level definition `s = _bound_value(_rkppl_value_s)` (see
# `lower_rkppl`), a data-only call that `bind_data` evaluates once, so the
# kernel takes the number as a typed scalar argument — exactly as a
# definition `s = 2.0` would read, except the value binds at the call.
_bound_value(x) = x
_bound_array_value(x::AbstractArray) = x
_bound_value_input(name::Symbol) = Symbol("_rkppl_value_", name)
# Messages name a data value as its author wrote it, never its internal
# input (`ReactiveKernelsPPL._bound_value(_rkppl_value_s)` reads `s`).
_author_names(msg::AbstractString) = replace(String(msg),
    r"(?:ReactiveKernelsPPL\.)?_bound_value\(_rkppl_value_([\p{L}\p{N}_!]+)\)" => s"\1",
    r"_rkppl_value_([\p{L}\p{N}_!]+)" => s"\1")
_is_bound_value_call(ex) = ex isa Expr && ex.head === :call &&
    length(ex.args) == 2 && ex.args[1] in
        (GlobalRef(@__MODULE__, :_bound_value), GlobalRef(@__MODULE__, :_bound_array_value))
_is_bound_array_value_call(ex) = ex isa Expr && ex.head === :call &&
    length(ex.args) == 2 && ex.args[1] == GlobalRef(@__MODULE__, :_bound_array_value)

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

# Elementwise observation operands retain their Julia axes. Structured
# readers (schedule slices, level tables, and design columns) still use
# `_vector_column` where their operation requires a vector.
function _observation_column(columns::AbstractDict{Symbol}, name::Symbol,
        label::Symbol, what::AbstractString)
    col = columns[name]
    col isa AbstractArray ||
        _fail(label, "$what `$name` must be an array, got $(summary(col))")
    return col
end

"""Slice-1 likelihood families (D3 narrow slice), plus the leveled slice-2
families (categorical / ordinal / multinomial): reference-coded
multi-logit categorical, cumulative-logit ordinal with ordered cutpoints,
general typed ordinal (2 structures × 3 links), shared-simplex multinomial
over a count matrix, plain categorical over simplex probabilities, the
joint correlated-outcomes family (per-row MvNormal over a Cholesky factor),
and the finite-mixture family (per-row log-sum-exp over K same-family
univariate components, SB `MixtureModel` mirror)."""
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
    BinomialProbFam
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
    MixtureFam
    StudentTFam
    HurdlePoissonFam
    ZeroInflatedPoissonFam
    InverseGaussianFam
    BetaBinomial2Fam
    VonMisesFam
    NegativeBinomialFam
    ExponentialLogFam
    LogNormalFam
    WeibullFam
    ZeroInflatedBinomialFam
    GammaValueFam
    WeibullValueFam
    CauchyFam
    BetaShapeFam
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
as a linear-predictor value (`lp = theta` or an unscaled summand, identity
design) — the random-effects / per-observation-latent location. `ScanSummandTerm` splices a
sequential-recurrence state into the predictor scaled by a sampled scalar
coefficient (`u .* beta`, SB's `ar` latent path with its free `popefs` beta), or
unscaled when its `coef` is `nothing` (`mu = a .+ x`).
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
    ComposedTerm
end

"""
    ResponseEvidence(kind, lower, upper)

Censoring/truncation evidence wrapper on an admitted family (D3 response
evidence). Bounds are real literals or names of data, sampled values and
ordinary definitions (scalar or per observation). For
`:interval_censored` the response itself is the lower endpoint, so `lower`
must be `nothing` and `upper` is required. The interval is open below:
the cell is `log(CDF(upper) - CDF(response))` for `(response, upper]`.
`:censored` is the clamp law (`Y = clamp(X)`): rows at a bound take the
CDF mass (`yv ≤ lower` / `yv ≥ upper`), never the density.
"""
struct ResponseEvidence
    kind::Symbol # :none | :truncated | :censored | :interval_censored
    lower::Union{Nothing,Real,ColumnRef}
    upper::Union{Nothing,Real,ColumnRef}
end

"""
    ScalePredictorRef(predictor, link)

A per-observation auxiliary use, including scale, shape, probability,
Student degrees of freedom and zero inflation. `predictor` names the
[`PredictorSpec`](@ref), whose emitted value is unchanged. `link` belongs
to this use: identity reads the value directly; log, logit, probit and
cloglog apply `exp`, `logistic`, `normcdf` and `cexpexp`, respectively.
Several positions may share the same predictor under different links.
The generator binds each transformed vector once and threads it through
the likelihood plate. A value outside the distribution parameter's
support contributes `-Inf` through a lazy density branch.
"""
struct ScalePredictorRef
    predictor::Symbol
    link::LinkFunction
end

"""
    MixtureComplementWeights(param, param_first)

2-component mixture weights from one `:unit`-support sampled parameter
(`[s, 1-s]` when `param_first`, `[1-s, s]` otherwise): the collapsed
Beta-weight shape. The complement sums to 1 by construction, so no
simplex validation applies; the generator threads the constrained
scalar and logs each arm directly.
"""
struct MixtureComplementWeights
    param::Symbol
    param_first::Bool
end

"""
    LikelihoodSpec(family, link, response, predictor, scale, weights, evidence, label[, trials[, range]])

One independent response. `scale` is the response's auxiliary —
Gaussian sigma, NB2 dispersion phi, Gamma shape alpha, hurdle p_zero,
InverseGaussian shape lambda, BetaBinomial2 precision phi, VonMises
concentration kappa, NB1 success probability p, LogNormal scale sigma,
Weibull shape k — either scalar (parameter, assignment, folded
literal, or a raw per-observation data column) or a
[`ScalePredictorRef`](@ref). The latter includes LogNormal sigma and
Weibull k. Probability auxiliaries (hurdle p_zero and NB1 p) take values
in [0, 1]; other scale/shape auxiliaries take positive values. The link
need not guarantee support: the density checks the evaluated value.
(One slot covers every admitted family; a two-auxiliary family such as
Beta needs a new field — noted, not built.) `nu` is the StudentT
degrees of freedom, `nothing` otherwise: a sampled parameter/assignment
name, a bound data name, a finite-positive literal, or a
[`ScalePredictorRef`](@ref)
(predictor-fed per-observation nu — the scale-slot mechanism, with its
own `_ppl_sc_<label>_nu` node so scale and nu predictors coexist).
`zi` is the ZIP/ZIB zero-inflation probability and accepts the same value
forms. Fixed `nu` data must be finite positive reals; fixed `zi` data must
be finite reals in [0, 1]. Model-valued inputs use the lazy density guard.
`weights` is a
frequency/power-objective column (D1); analytic/precision weights fail
closed emitter-side. `trials` is the Binomial/BetaBinomial2 trial count
(Int column or Int literal), `nothing` otherwise. `range` carries a literal
`y[1:N]` range, or the indexed response expression for `eachindex`, `axes`
and `:` selections, including positive literal trailing indices.
`nothing` means the whole column. Literal one-dimensional ranges retain
their full-cover check; indexed expressions select the stated response cells.
An explicit indexed observation loop also selects its row operands before
evaluating the retained cell; a top-level slice uses ordinary broadcasting.

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
  finite-positive data column, a legacy predictor name, or a `ScalePredictorRef`
  whose link belongs to this discrimination use.
  Modeled values have finite-positive support checked by a lazy likelihood guard.
- `threshold_columns`: OrdinalFam per-threshold design columns
  (StoppingRatio only), empty otherwise.
- `threshold_coefs`: the (K−1)×p threshold-coefficient matrix packed as a
  `:vector_normal` [`VectorParameter`](@ref) (required exactly when
  `threshold_columns` is non-empty), `nothing` otherwise. Stage-major:
  stage j occupies entries `(j−1)*p+1 .. j*p`.
- `threshold_effects`: an ordinary N×(K−1) matrix value for stopping-ratio
  responses, subtracted from `thresholds[j] - eta[i]` before discrimination.
  It replaces, and cannot accompany, the legacy threshold design fields.

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

Finite-mixture responses (`MixtureFam`, SB `y ~ MixtureModel(comps, w)`
mirror) carry K same-family univariate components in the trailing
mixture fields, built with keywords; every other family leaves them at
their defaults:

- `mixture_family`: the shared component family (v1: Gaussian/identity,
  Bernoulli-logit, Poisson-log, Binomial-logit, NB2-log, Gamma-log,
  Beta-logit), `nothing` otherwise.
- `mixture_locs`: per-component location uses, length K (K ≥ 1): each a
  predictor name (link-space, inverted per the component link), a sampled
  scalar parameter name, or a numeric literal (both constrained-scale, no
  inversion). Empty otherwise.
- `mixture_scales`: per-component scale uses, length K: each the
  component's `scale`-slot use (`nothing`, parameter/assignment name,
  literal, per-observation column, or [`ScalePredictorRef`](@ref)) —
  required exactly when the component family takes a scale auxiliary.
  Empty otherwise.
- `mixture_weights`: the length-K mixing weights: a literal
  `Vector{Float64}` (finite, nonnegative, sums to 1), a
  shared bound numeric vector, a `:simplex_dirichlet` [`VectorParameter`](@ref) name, or a
  [`MixtureComplementWeights`](@ref) pair (K == 2), `nothing`
  otherwise.
- `mixture_trials`: one trial-count column or literal per Binomial component
  when their arguments differ. Empty when `trials` supplies the shared argument.

`predictor` is an anchor only (first location predictor, else first
scale predictor, else the weights simplex name, else the first
location/scale parameter name, else the response name); `scale` is `nothing`.
Mixture widths are structural
(K = `length(mixture_locs)`): no `n_levels`, no bind-time size
inference. Binomial components may have distinct trial counts.
Case-A `mi()` missingness (SB parity, log density only) names its observed-row
index column in the trailing `mi_jobs` field, built with keywords (`mi_jobs=`);
every other response leaves it at `nothing`:

- `mi_jobs`: the `Jobs` column (sorted-ascending unique `Int` row indices,
  a nonempty selection on the response's full observation axis). The
  response, every Multinomial count column and every joint outcome cross
  PACKED (aligned with `Jobs`); they ride the managed-columns exemption.
  Vector predictors and trials gather by full-row `Jobs`; shared probabilities
  and covariance factors stay whole. GLM objects select full design rows
  before constructing eta. Only observed rows contribute, with no missing
  response latent (SB keeps `y_mis` in generated quantities here). Scalar
  observation families compose with weights, evidence, ranges and trials.
  A range filters Jobs on the full axis and the same packed outcome positions;
  Case-B downstream merged-response uses fail closed emitter-side.

VonMises responses (`VonMisesFam`, SB `brm_von_mises` mirror) carry the
principal-interval endpoints in the trailing `interval` field, built
with keywords (`interval=`); every other family leaves it at `nothing`:

- `interval`: `nothing` for exact `VonMises` (moving inclusive support
  `[mu - pi, mu + pi]`), or the `(lo, hi)` literal pair for
  `CircularVonMises` (fixed half-open support `[lo, hi)`, finite with
  `lo < hi` and width exactly `2pi` — the BRM `_brm_circular_interval`
  rule). The generator wraps mu into `[lo, hi)` per cell and guards
  `y` to the support, exactly the SB branch structure.
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
    range::Union{Nothing,UnitRange{Int},Expr}
    n_levels::Union{Nothing,Int}
    thresholds::Union{Nothing,ParamName}
    extra_predictors::Vector{Symbol}
    count_columns::Vector{ColumnRef}
    ordinal_structure::Union{Nothing,Symbol}
    discrimination::Union{Nothing,Real,ColumnRef,ScalePredictorRef}
    threshold_columns::Vector{ColumnRef}
    threshold_coefs::Union{Nothing,ParamName}
    extra_responses::Vector{ColumnRef}
    factor_scales::Union{Nothing,ParamName}
    factor_corr::Union{Nothing,ParamName}
    glm_alpha::Union{Nothing,ParamName}
    glm_beta::Union{Nothing,ParamName}
    mixture_family::Union{Nothing,LikelihoodFamily}
    mixture_locs::Vector{Union{Symbol,Real}}
    mixture_scales::Vector{Union{Nothing,Symbol,Real,ScalePredictorRef}}
    mixture_weights::Union{Nothing,Symbol,Vector{Float64},MixtureComplementWeights}
    mixture_trials::Vector{Union{ColumnRef,Int}}
    nu::Union{Nothing,ParamName,Real,ScalePredictorRef}
    zi::Union{Nothing,ParamName,Real,ScalePredictorRef}
    mi_jobs::Union{Nothing,ColumnRef}
    interval::Union{Nothing,Tuple{Union{Real,Symbol},Union{Real,Symbol}}}
    threshold_effects::Union{Nothing,Symbol}
end
# Preserve the all-fields constructor predating per-component trials.
LikelihoodSpec(family, link, response, predictor, scale, weights, evidence,
    label, trials, range, n_levels, thresholds, extra_predictors, count_columns,
    ordinal_structure, discrimination, threshold_columns, threshold_coefs,
    extra_responses, factor_scales, factor_corr, glm_alpha, glm_beta,
    mixture_family, mixture_locs, mixture_scales, mixture_weights,
    nu, zi, mi_jobs, interval, threshold_effects) =
    LikelihoodSpec(family, link, response, predictor, scale, weights, evidence,
        label, trials, range, n_levels, thresholds, extra_predictors, count_columns,
        ordinal_structure, discrimination, threshold_columns, threshold_coefs,
        extra_responses, factor_scales, factor_corr, glm_alpha, glm_beta,
        mixture_family, mixture_locs, mixture_scales, mixture_weights,
        Union{ColumnRef,Int}[], nu, zi, mi_jobs, interval, threshold_effects)

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
        discrimination::Union{Nothing,Real,ColumnRef,ScalePredictorRef} = nothing,
        threshold_columns::Vector{ColumnRef} = Symbol[],
        threshold_coefs::Union{Nothing,ParamName} = nothing,
        extra_responses::Vector{ColumnRef} = Symbol[],
        factor_scales::Union{Nothing,ParamName} = nothing,
        factor_corr::Union{Nothing,ParamName} = nothing,
        glm_alpha::Union{Nothing,ParamName} = nothing,
        glm_beta::Union{Nothing,ParamName} = nothing,
        mixture_family::Union{Nothing,LikelihoodFamily} = nothing,
        mixture_locs::Vector{Union{Symbol,Real}} = Union{Symbol,Real}[],
        mixture_scales::Vector{Union{Nothing,Symbol,Real,ScalePredictorRef}} =
            Union{Nothing,Symbol,Real,ScalePredictorRef}[],
        mixture_weights::Union{Nothing,Symbol,Vector{Float64},MixtureComplementWeights} = nothing,
        mixture_trials::Vector{Union{ColumnRef,Int}} = Union{ColumnRef,Int}[],
        nu::Union{Nothing,ParamName,Real,ScalePredictorRef} = nothing,
        zi::Union{Nothing,ParamName,Real,ScalePredictorRef} = nothing,
        mi_jobs::Union{Nothing,ColumnRef} = nothing,
        interval::Union{Nothing,Tuple{Union{Real,Symbol},Union{Real,Symbol}}} = nothing,
        threshold_effects::Union{Nothing,Symbol} = nothing)
    return LikelihoodSpec(family, link, response, predictor, scale, weights,
        evidence, label, trials, range, n_levels, thresholds,
        extra_predictors, count_columns, ordinal_structure, discrimination,
        threshold_columns, threshold_coefs, extra_responses, factor_scales,
        factor_corr, glm_alpha, glm_beta, mixture_family, mixture_locs,
        mixture_scales, mixture_weights, mixture_trials, nu, zi, mi_jobs, interval,
        threshold_effects)
end

"""
    TermSpec(kind, columns, options, addressee, label)

One additive predictor term: structure only, never materialized designs
(D5a). `addressee` is the prior address (source column or `:Intercept`),
never a per-level label. An optimized ordinary parameter read carries
`parameter` (its declaration name) and `sign` in `options`; its declaration
owns the prior and coordinates. Factor sizing lives in
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

# An affine term reads an ordinary parameter; it never owns that
# parameter's prior, transform, coordinates, or other readers.
_parameter_term(t::TermSpec) = hasproperty(t.options, :parameter)
_term_structure_options(t::TermSpec) = _parameter_term(t) ?
    Base.structdiff(t.options, (parameter=nothing, sign=nothing)) : t.options
_parameter_terms(p::PredictorSpec) = any(_parameter_term, p.terms)
_legacy_predictor(p::PredictorSpec) = PredictorSpec(p.name, p.link,
    TermSpec[t for t in p.terms if !_parameter_term(t)], p.label)
_affine_block_name(p::PredictorSpec) = _parameter_terms(p) ?
    Symbol(:_ppl_affine_coef_, p.name) : block_name(p.name)

"""
    PopulationPrior(predictor, addressee, location, scale)
    PopulationPrior(predictor, addressee, family, location, scale[, nu])

Per-addressee population prior addressed by `(predictor,
column|:Intercept)`. `family` is one of [`POPULATION_FAMILIES`](@ref)
(`:flat` contributes 0.0 — StanBlocks flat-token parity — and ignores
location/scale/nu); `nu` is the StudentT degrees of freedom (`NaN`
otherwise, always a literal). A factor source column address applies
one shared prior across its full-rank level block (one coefficient per
mapped level); a matrix broadcast applies one shared family with
per-element locations/scales. `location`/`scale` are literals or
sampled-hyperparameter names (the centered-hierarchical shape:
validation pins a scalar sampled parameter or scalar assignment for a
location, a positive-support sampled parameter for a scale). The 4-arg
form is Normal. The emitter fills `Normal(0,1)` defaults so coverage is
complete by construction — except factor coefficients, whose broadcast
prior also sizes the block and is therefore required, never defaulted.
"""
struct PopulationPrior
    predictor::Symbol
    addressee::Symbol
    family::Symbol
    location::Union{Real,Symbol}
    scale::Union{Real,Symbol}
    nu::Real
end
PopulationPrior(predictor::Symbol, addressee::Symbol,
    location::Union{Real,Symbol}, scale::Union{Real,Symbol}) =
    PopulationPrior(predictor, addressee, :normal, location, scale, NaN)
PopulationPrior(predictor::Symbol, addressee::Symbol, family::Symbol,
    location::Union{Real,Symbol}, scale::Union{Real,Symbol}) =
    PopulationPrior(predictor, addressee, family, location, scale, NaN)

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
    HorseshoePrior(predictor, addressee, local_scale, global_scale, sign)

One per-coefficient horseshoe prior: standard-normal raw coefficient,
normalized HalfCauchy(local_scale) lambda, and HalfCauchy(global_scale) tau.
Each scalar built-in call owns its own tau. For one shared global tau use
`horseshoe_coefs(X)`. `addressee` is a predictor's intercept or continuous
column and `sign` is its use polarity. The derived coordinate is
`sign * raw * lambda * tau`; the latent priors use SampledParameter.
Scales are finite strictly positive literals.
"""
struct HorseshoePrior
    predictor::Symbol
    addressee::Symbol
    local_scale::Float64
    global_scale::Float64
    sign::Int
end

"""Synthesized triple/scalar names for one horseshoe addressee (surface,
validator, and generator share these — the names are the contract)."""
horseshoe_raw_name(pred::Symbol, addr::Symbol) =
    Symbol(:horseshoe_, pred, :_, addr, :_raw)
horseshoe_lambda_name(pred::Symbol, addr::Symbol) =
    Symbol(:horseshoe_, pred, :_, addr, :_lambda)
horseshoe_tau_name(pred::Symbol, addr::Symbol) =
    Symbol(:horseshoe_, pred, :_, addr, :_tau)
horseshoe_normal_name(pred::Symbol, addr::Symbol) =
    Symbol(:horseshoe_, pred, :_, addr, :_normal)

"""
    SupportOverride

A latent's support override is `nothing` (infer natural support), `:positive`
(a normalized symmetric half at literal-zero location), or
`(:truncated, lo, hi)` (normalized base density on the intersection with its
natural support). `(:restricted, lo, hi)` intersects the support without
renormalizing the original density; `(:restricted_half, lo, hi)` retains a wrapped
HalfNormal/HalfCauchy's original log(2) normalization. Bounds may be literals,
data or sampled expressions.
The older normalized `:interval`, `:upper` and `:lower` representations
remain supported for hand-built plans. Shared by scalar and per-cell priors.
"""
const SupportOverride =
    Union{Nothing,Symbol,Tuple{Symbol,Float64,Float64},Tuple{Symbol,Float64},
        Tuple{Symbol,Symbol},
        Tuple{Symbol,Union{Float64,Symbol,Expr},Union{Float64,Symbol,Expr}}}

_support_args(ov) = ov isa Tuple ? ov[2:end] : ()
_has_interval_bounds(ov) = ov isa Tuple &&
    ov[1] in (:truncated, :restricted, :restricted_half)

"""
    SampledParameter(name, family, args, support_override, label)

One scalar latent with positional `(arg1, arg2, …)` arguments in
Distributions constructor order (Exponential uses scale). Arguments may read
data, sampled parameters or ordinary definitions. Explicit support follows
SupportOverride; truncation normalizes and restriction preserves the density.
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
Distributions.jl semantics). Arguments may be shared scalars or per-cell
data/derived columns. `range` is the plate's index set: the `Symbol` of the column `v`
of an `eachindex(v)` / `axes(v, 1)` plate (one cell per entry of bound or
defined `v`), a literal `UnitRange{Int}` defining its own cells (the `1:N`
case), or `nothing` (the consuming response establishes the
axis in hand-built plans). A
real-support prior (`normal`/`cauchy`/half-versions) lays out identity (an
unconstrained block); positive/unit support constrains per element.
"""
struct PlateParameter
    name::ParamName
    family::Symbol
    args::NamedTuple
    support_override::SupportOverride
    range::Union{Nothing,UnitRange{Int},Symbol,Expr}
    label::Symbol
end
"""Provenance/range default to the consuming response's axis under the name."""
PlateParameter(name::ParamName, family::Symbol, args::NamedTuple,
    support_override::SupportOverride) =
    PlateParameter(name, family, args, support_override, nothing, name)
PlateParameter(name::ParamName, family::Symbol, args::NamedTuple,
    support_override::SupportOverride,
    range::Union{Nothing,UnitRange{Int},Symbol,Expr}) =
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

One margin's normalized positive-scale prior. `:std_normal` is HalfNormal(1),
`:normal` is HalfNormal(param), `:cauchy` is HalfCauchy(param), and
`:exponential` is Exponential(param) in the Distributions scale convention.
An empty sd_priors vector gives the HalfNormal(1) default for every margin.
"""
struct VaryingSdPrior
    family::Symbol
    param::Float64
end

VaryingSdPrior(family::Symbol, param::Real) =
    VaryingSdPrior(family, Float64(param))

const _SD_PRIOR_FAMILIES = (:std_normal, :exponential, :normal, :cauchy)

"""
    VaryingMultiMembership(groups, weights, normalize)

Multi-membership grouping metadata for a [`VaryingDraws`](@ref) block
(SB `mm(...)` mirror): `groups` is the ≥2 membership columns (raw
data — one shared draws block over their UNION levels), `weights` is
`nothing` (equal `1/M` weights) or one raw numeric column per group,
`normalize` row-normalizes supplied weights to sum to one (SB
`_brm_prepare_mm`). `nothing` on the draws = plain single-column
grouping.
"""
struct VaryingMultiMembership
    groups::Vector{Symbol}
    weights::Union{Nothing,Vector{Symbol}}
    normalize::Bool
end

"""
    VaryingStrata(by, levels)

Stratified-grouping metadata for a [`VaryingDraws`](@ref) block (SB
`gr(g, by=b)` mirror): `by` is the raw stratum column, `levels` the
strata levels in numbering order (`nothing` pre-bind —
[`bind_data`](@ref) fills sort-ordered observed levels). `nothing` on
the draws = unstratified.
"""
struct VaryingStrata
    by::Symbol
    levels::Union{Nothing,Vector}
end

"""
    VaryingDraws(group, kind, margins, lkj_eta, label, suffix[, levels[, sd_priors[, mm[, strata]]]])

One shared varying-effect draws block over
`K = length(margins)` margins in `G` groups of raw column `group`.
`kind` is `:correlated` (non-centered LKJ + tau + z_flat).
There is one geometry for every K and every margin: a single
margin, intercept or slope, is the 1x1 case, whose LKJ factor is the
fixed `[1]` (zero coordinates, a `0.0` prior node), so its sd is the
same half-normal `tau` (`tau ~ Normal(0, 1)` on `tau > 0` plus the
`exp`-layout Jacobian) as every margin of a larger block. K = 1 draws
carry `lkj_eta == 1.0`: no other eta parameterizes anything there.
(StanBlocks-BRMI samples a K=1 intercept's sd in log space — a
LogNormal(0, 1) sd. That asymmetry depended on whether `eta` was
written, so it is not mirrored.) `label` is the link identity
(`:draws_<suffix>`); `suffix` is the in-graph naming stem (the group,
or `group_binding` when two draws share a grouping). `levels` is the
grouping's DECLARED levels in numbering order (`nothing` pre-bind, or
when the emitter has no declaration — [`bind_data`](@ref) fills
sort-ordered observed levels). `sd_priors` is the per-margin `tau`
prior ([`VaryingSdPrior`](@ref); empty — the default — is all
`:std_normal`), at any K. `mm` is
[`VaryingMultiMembership`](@ref) metadata (`nothing` = plain
grouping; `group` is then the mm naming symbol, not a data column).
`strata` is [`VaryingStrata`](@ref) metadata (`nothing` =
unstratified). Shared draws are consumed by value flow: each
[`VaryingSlice`](@ref) names this draws' label plus explicit columns
— never by label matching across statements.
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
    mm::Union{Nothing,VaryingMultiMembership}
    strata::Union{Nothing,VaryingStrata}
end

# Pre-mm/strata 6/7/8-arg positional construction keeps working with
# `mm`/`strata` unset (plain single-column grouping).
VaryingDraws(group::ColumnRef, kind::Symbol, margins::Vector{VaryingMargin},
    lkj_eta::Float64, label::Symbol, suffix::String) =
    VaryingDraws(group, kind, margins, lkj_eta, label, suffix, nothing,
        VaryingSdPrior[], nothing, nothing)
VaryingDraws(group::ColumnRef, kind::Symbol, margins::Vector{VaryingMargin},
    lkj_eta::Float64, label::Symbol, suffix::String,
    levels::Union{Nothing,Vector}) =
    VaryingDraws(group, kind, margins, lkj_eta, label, suffix, levels,
        VaryingSdPrior[], nothing, nothing)
VaryingDraws(group::ColumnRef, kind::Symbol, margins::Vector{VaryingMargin},
    lkj_eta::Float64, label::Symbol, suffix::String,
    levels::Union{Nothing,Vector}, sd_priors::Vector{VaryingSdPrior}) =
    VaryingDraws(group, kind, margins, lkj_eta, label, suffix, levels,
        sd_priors, nothing, nothing)

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
  factorial normalizer) + the ordered-transform Jacobian. The surface
  `c ~ Ordered(Normal(m, s), n)`; unlinked (a free-standing ordered value)
  it carries a literal extent or a data-only expression resolved at bind.
- `:vector_normal` — a plain (unconstrained, identity-transform) vector
  with an elementwise `Normal(arg1, arg2)` prior (stopping-ratio stage
  thresholds). Cauchy, Laplace, Logistic and StudentT have corresponding
  `:ordered_*` and `:vector_*` element-density families.
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

`size` is the constrained length (K−1 for thresholds, K for a simplex),
or a data-only expression resolved at bind. `nothing` fills from a linked
response or concentration. Element prior arguments may be scalar literals,
data or declared values. K=1 is uniform: zero-length
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
    size::Union{Nothing,Int,Expr}
    label::Symbol
    # Retain whole-value sizing dependencies after `size` resolves at bind.
    extent_expr::Union{Nothing,Expr}
end
VectorParameter(name::ParamName, family::Symbol, args::NamedTuple,
    size::Union{Nothing,Int,Expr}, label::Symbol) =
    VectorParameter(name, family, args, size, label, size isa Expr ? size : nothing)
"""Provenance defaults to the parameter's own name."""
VectorParameter(name::ParamName, family::Symbol, args::NamedTuple,
    size::Union{Nothing,Int,Expr}) =
    VectorParameter(name, family, args, size, name)

"""
    ArrayParameter(name, family, args, dims, support_override, label)

One declared array-valued parameter: a plain value the model reads by
name (`z`, `z[g]`, `z[1]`, `B * w`, `L[2, 1]`), never a hidden
coefficient block. Two kinds of family:

- An elementwise family (any [`SAMPLED_ARITY`](@ref) key): the
  surface `z[1:K] .~ Normal.(mu, s)` — independent draws per element,
  with Distributions.jl `Normal.(…)` broadcast semantics. `args` use
  positional keys `(arg1, …)`; each is a Real literal, a Symbol (a scalar
  parameter/assignment name, an array parameter, a data column, a
  derived column, or a vector-valued assignment — read per element), a
  literal vector (`Expr(:vect, …)`, per element), or a value expression
  (per element). `support_override` follows [`SampledParameter`](@ref)
  (`HalfNormal.(s)` is `:normal` + `:positive`).
- `:lkj_cholesky` — `L ~ LKJCholesky(K, eta)`: the lower-triangular
  Cholesky factor of a K×K correlation matrix (Distributions.jl
  `LKJCholesky(K, eta)` density, `uplo = 'L'`). `args = (arg1 = eta,)`, a
  finite positive literal; `dims` is `[K, K]`.
- A multivariate slice family `<stem>_<slices>` (`mv_slices.jl`):
  `eachrow(B[a, b]) .~ D` (`_rows`), `eachcol(B[a, b]) .~ D` (`_cols`) or
  `b[ax] ~ D` (`_vector`) — every slice of the array one draw of the
  multivariate `D`. Stems: `mvnormal_cholesky` (`args = (arg1 = mean,
  arg2 = factor)`), `mvnormal` (`(arg1 = mean, arg2 = covariance)`),
  `dirichlet` (`(arg1 = concentration,)`, rows/cols only) and
  `ordered_normal` (`(arg1 = m, arg2 = s, arg3 = K)` for
  `Ordered(Normal(m, s), K)`, rows/cols only). A vector argument is a
  shared value or a per-slice `:(eachrow(M))` / `:(eachcol(M))`
  expression; matrices and scalars are shared. Multivariate normal slices
  are centered (the entries are the coordinates); simplex and ordered
  slices constrain along each slice.

A `phi ~ Dirichlet(alpha)` simplex is not an array parameter: it stays a
[`VectorParameter`](@ref), which definitions already read as a model-level
value (`phi[1]`, `cumsum(phi)`, a gather `cum[c]`).

`dims` holds one entry per array axis, each either a literal `Int` (from
`1:K`) or the surface sizing expression, resolved against bound data:
`:(levels(g))` (the sorted distinct values of grouping column `g`; reading
`z[g]` then looks each observation's value up on that axis),
`:(length(levels(g)) - k)` (from `1:length(levels(g)) - k`: a positional
axis of that many elements, `k` a literal ≥ 0, bare `length(levels(g))`
for 0), `:(axes(M, d))` or `:(size(M, d))` (axis `d` of a bound matrix or a
design matrix `M`). Arrays have one or two axes.
"""
struct ArrayParameter
    name::ParamName
    family::Symbol
    args::NamedTuple
    dims::Vector{Any}
    support_override::SupportOverride
    label::Symbol
end
"""Provenance defaults to the parameter's own name."""
ArrayParameter(name::ParamName, family::Symbol, args::NamedTuple,
    dims::Vector{Any}, support_override::SupportOverride) =
    ArrayParameter(name, family, args, dims, support_override, name)

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

One free spline coefficient block: flat fixed coefficients, standard-normal
raw coefficients, and normalized positive smoothing scales. `width` derives
from structural `k`; `basis` is the owning basis id. Priors retain one plate.
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
    HyperPrior(family, args[, support_override])

A literal-argument positive hyperparameter prior with the same family,
arguments and explicit normalized support as a sampled parameter.
"""
struct HyperPrior
    family::Symbol
    args::NamedTuple
    support_override::SupportOverride
end

HyperPrior(family::Symbol, args::NamedTuple) = HyperPrior(family, args, nothing)

# Admitted base families and their Distributions arities. Real-support
# families require an explicit half or truncation.
const _HYPER_PRIOR_FAMILIES = Dict{Symbol,Tuple{Vararg{Int}}}(
    :lognormal => (2,), :inverse_gamma => (2,), :gamma => (2,),
    :exponential => (1,), :normal => (2,), :cauchy => (2,),
    :student_t => (3,), :uniform => (2,))

# Explicit support travels with a hyper prior, just as for a sampled prior.
_hyper_support_override(hp::HyperPrior) = hp.support_override
function _hyper_prior_bounds(hp::HyperPrior)
    hp.support_override === nothing && hp.family !== :uniform && return nothing
    transform, lo, hi = _entry_transform(hp.family, hp.support_override, hp.args)
    transform === :interval && return (lo, hi)
    transform === :floored && return (lo, Inf)
    transform === :exp && return nothing
    throw(ContractValidationError("[hyper prior] use an explicit positive distribution, got $(hp.family) with support $(hp.support_override)"))
end
_hyper_prior_bounds(::Nothing) = nothing

function _validate_hyper_prior(hp::HyperPrior, label::Symbol, what)
    haskey(_HYPER_PRIOR_FAMILIES, hp.family) || _fail(label,
        "$what prior family $(hp.family) is not admitted (admitted: " *
        "$(join(sort!(collect(keys(_HYPER_PRIOR_FAMILIES))), ", ")))")
    length(hp.args) in _HYPER_PRIOR_FAMILIES[hp.family] || _fail(label,
        "$what prior $(hp.family) takes " *
        "$(_HYPER_PRIOR_FAMILIES[hp.family]) arguments, got " *
        "$(length(hp.args))")
    keys(hp.args) == ntuple(i -> Symbol(:arg, i), length(hp.args)) ||
        _fail(label, "$what prior args must be `(arg1, ...)`, got " *
              "$(keys(hp.args))")
    all(v -> v isa Real && !(v isa Bool) && isfinite(v), values(hp.args)) ||
        _fail(label, "$what prior args must be finite numeric " *
              "literals, got $(hp.args)")
    _validate_support_override(label, hp.family, hp.support_override, hp.args)
    _validate_uniform_args(label, hp.family, hp.args)
    SAMPLED_SUPPORT[hp.family] === :real && hp.support_override === nothing &&
        _fail(label, "$what needs an explicit positive prior: HalfNormal, HalfCauchy or truncated(D, 0, Inf)")
    bounds = _hyper_prior_bounds(hp)
    bounds === nothing || (0 <= bounds[1] < bounds[2]) ||
        _fail(label, "$what prior must have positive support, got $bounds")
    return nothing
end

"""
    HSGPHyperLP(intercept, group)

A grouped HSGP hyperparameter's log-linear hyper-predictor (SB
`log(length_scale(hsgp(x))) ~ 1 + (1 | g)` / `~ (1 | g)`): per group
`log h_g = beta0 + sd * z_g` (`intercept` false drops `beta0`), with the
BRM defaults `beta0 ~ Normal(0, 1)`, `sd ~ HalfNormal(1)`, non-centered `z ~ Normal(0, 1)`. `group` must be
the basis's `by` column (one hyper level per term group). A length scale
is floored per group at the validity floor (SB `brm_hsgp_by_hyper_S`
`fmax(rho_g, rho_lower)`).
"""
struct HSGPHyperLP
    intercept::Bool
    group::Symbol
end

# A hyper-predictor carries no stated bounds (its floor is applied
# in-graph per group).
_hyper_prior_bounds(::HSGPHyperLP) = nothing

"""
    HSGPGrouping(column, levels)

A per-group HSGP (SB `hsgp(x; by = g)`): the basis weights vary by the
levels of data column `column` (`levels` sort-ordered observed levels,
`nothing` pre-bind — [`bind_data`](@ref) fills them).
"""
struct HSGPGrouping
    column::Symbol
    levels::Union{Nothing,Vector{Any}}
end

"""
    HSGPBasis(id, axes, K, c, iso, fits, label, cov, period[, rho_prior,
              sigma_prior[, domain[, by]]])

One Hilbert-space GP basis (SB `_sb_hsgp` / `_sb_hsgp_periodic`):
`axes` raw data columns, `K` modes per axis, `c` boundary factors per
axis (`L = c*max|x-mu|`, `c > 1`), `iso` length-scale sharing. `fits`
holds the bind-time `(mu, L)` per axis (empty pre-bind); the basis
itself is evaluated in-graph from the raw columns + frozen fits (trig
is elementwise-expressible — unlike spline eigen, no bind-time
materialization). `M = prod(K)` basis functions; the term owns
`beta_raw_<id>` (M-vector), `rho_<id>` (iso scalar) or
`rho_<id>_1..d` (aniso scalars), `sigma_<id>` (scalar).

`cov` selects the kernel (`:exp_quad` or `:periodic`, SB
``hsgp(...; cov=...)``); `period` is the periodic kernel's formula
constant (finite and positive iff periodic, `NaN` otherwise). A
periodic basis takes exactly one isotropic axis, carries no fits and
no domain (`c` is validated but ignored, the SB mirror), and owns `M
= 2k` basis functions (cosines then sines over `k` harmonics).

`rho_prior` / `sigma_prior` override the default `lognormal(0, 1)`
priors ([`HyperPrior`](@ref), SB `length_scale(:, hsgp(x)) ~ ...` /
`sd(:, hsgp(x)) ~ ...`). An explicit length-scale prior replaces the
whole default declaration INCLUDING the approximation-validity floor
(BRM `_brm_hsgp_declared_rho_lower`): the length scales then ride a
plain `exp` transform; a `Uniform(lo, hi)` prior instead BOUNDS the
hyperparameter (`(:interval, lo, hi)`, SB prior-bound intersection).
`nothing` keeps the default.

`domain` (SB `hsgp(...; domain=...)`) fixes the eigenfunction domain
per axis as `(lower, upper)` pairs: bind uses `(mu, L) = ((lo+hi)/2,
(hi-lo)/2)` instead of the data-fitted `L = c*max|x-mu|` (so `c` does
not apply), and every bound axis value must lie inside its pair.

`by` ([`HSGPGrouping`](@ref), SB `hsgp(x; by = g)`) makes the basis
weights per group (`G*M` standardized weights) over one shared basis;
the hyperparameters stay shared unless `rho_prior` / `sigma_prior` carry
an [`HSGPHyperLP`](@ref) (per-group log-linear hyper-predictors). v1:
one isotropic exp-quad axis.
"""
struct HSGPBasis
    id::Symbol
    axes::Vector{Symbol}
    K::Vector{Int}
    c::Vector{Float64}
    iso::Bool
    fits::Vector{Tuple{Float64,Float64}}
    label::Symbol
    cov::Symbol
    period::Float64
    rho_prior::Union{Nothing,HyperPrior,HSGPHyperLP}
    sigma_prior::Union{Nothing,HyperPrior,HSGPHyperLP}
    domain::Union{Nothing,Vector{Tuple{Float64,Float64}}}
    by::Union{Nothing,HSGPGrouping}
end

"""Default-prior construction (SB `_sb_hsgp`'s `lognormal(0, 1)`
length scales on the validity floor + `lognormal(0, 1)` marginal
scale)."""
HSGPBasis(id::Symbol, axes::Vector{Symbol}, K::Vector{Int},
    c::Vector{Float64}, iso::Bool,
    fits::Vector{Tuple{Float64,Float64}}, label::Symbol, cov::Symbol,
    period::Float64) =
    HSGPBasis(id, axes, K, c, iso, fits, label, cov, period, nothing,
        nothing, nothing, nothing)

"""Stated-hyper-prior construction without a fixed domain."""
HSGPBasis(id::Symbol, axes::Vector{Symbol}, K::Vector{Int},
    c::Vector{Float64}, iso::Bool,
    fits::Vector{Tuple{Float64,Float64}}, label::Symbol, cov::Symbol,
    period::Float64, rho_prior, sigma_prior) =
    HSGPBasis(id, axes, K, c, iso, fits, label, cov, period, rho_prior,
        sigma_prior, nothing, nothing)

"""Fixed-domain construction without grouping."""
HSGPBasis(id::Symbol, axes::Vector{Symbol}, K::Vector{Int},
    c::Vector{Float64}, iso::Bool,
    fits::Vector{Tuple{Float64,Float64}}, label::Symbol, cov::Symbol,
    period::Float64, rho_prior, sigma_prior, domain) =
    HSGPBasis(id, axes, K, c, iso, fits, label, cov, period, rho_prior,
        sigma_prior, domain, nothing)

"""Exp-quad v1 positional construction (periodic defaults: `cov =
:exp_quad`, `period = NaN`)."""
HSGPBasis(id::Symbol, axes::Vector{Symbol}, K::Vector{Int},
    c::Vector{Float64}, iso::Bool,
    fits::Vector{Tuple{Float64,Float64}}, label::Symbol) =
    HSGPBasis(id, axes, K, c, iso, fits, label, :exp_quad, NaN)

abstract type PKScheduleSpec end

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
struct LinearPKScheduleSpec <: PKScheduleSpec
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

function _sched_raw_columns(s::LinearPKScheduleSpec)
    raw = [s.obs_subj, s.obs_time, s.dose_subj, s.dose_time, s.dose_amt]
    s.ecg !== nothing && append!(raw, s.ecg)
    s.tgi !== nothing && append!(raw, s.tgi)
    return raw
end
_sched_ends_field(::LinearPKScheduleSpec) = :op_ends

"""The conventional event-LP name used by legacy serialized plans.
Ordinary cell calls derive the event axis from argument position."""
const EVENT_LP_NAME = :log_F

"""
    LinearPKEventLPSpec(name, schedule, k, c, fit, label)

Retired implicit-provider representation, retained for compatibility with
serialized plan structure. A nonempty `StructuralPlan.event_lps` is rejected:
use the ordinary [`linear_pk_log_f`](@ref) library submodel instead. Its
parameters are authored statements rather than layout-generated blocks.
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
    Union{NamedTuple{(:response, :family, :location, :scale, :params)},
        NamedTuple{(:response, :family, :location, :scale, :params, :link)}}
_kernel_obs_link(obs::KernelObs) = hasproperty(obs, :link) ? obs.link : IdentityLink

"""Scalar response-space in-cell obs families (v2 panel set; grouped
admits these plus the joint families). Cell args are constrained-scale
values (bare args skip link inversion — literals prove domains)."""
const _KERNEL_SCALAR_FAMS = (GaussianFam, BernoulliLogitFam,
    PoissonLogFam, NegativeBinomial2Fam, GammaLogFam, BetaLogitFam,
    StudentTFam, BinomialProbFam, CauchyFam)

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
(ragged axes have no rectangular T). `subjects === nothing` (the
top-level schedule-chain form, see `_extract_kernel_cells`) takes the
subject count from named data at bind — the schedule's subject column —
and consumes no dims key. The cell vocabulary is calls to
[`CELL_FNS`](@ref), schedule-map gathers, and arithmetic (no flat
dotify — each PK call emits an RK subject plate containing event scans).
"""
struct KernelPlate
    result::Symbol
    subjects::Union{Nothing,Int,Symbol}
    timepoints::Union{Nothing,Int,Symbol}
    slices::Vector{Tuple{Symbol,Symbol,Symbol}}
    assignments::Vector{Pair{Symbol,Any}}
    obs::Vector{KernelObs}
    collected::Symbol
    label::Symbol
    lp_args::Vector{Tuple{Symbol,Symbol}}
    schedules::Vector{PKScheduleSpec}
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

"""Bernoulli in-cell Bool twin: `<result>_kbool_<response>` — the exact
Bool lanes a non-Bool Bernoulli flat reads (the `!=` comparison
misdifferentiates under native Enzyme — snag
`bernoulli-int-la-78487520`). Named by response PARAM (one twin per
obs, stable across raw/expansion flats)."""
_kbool_name(result::Symbol, response::Symbol) =
    Symbol("$(result)_kbool_$(response)")

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
        union!(out, _sched_raw_columns(s))
        for f in _sched_materialized_fields(s)
            push!(out, _sched_col_name(s.name, f))
        end
        for f in _sched_extra_fields(s, kp.assignments)
            push!(out, _sched_col_name(s.name, f))
        end
        # The event-axis product (present only with a declared
        # event-LP): op-length by design, like the op columns.
        s isa LinearPKScheduleSpec &&
            push!(out, _sched_col_name(s.name, :op_log_dose))
    end
    for obs in kp.obs
        # Bernoulli Bool twins (present only for non-Bool flats —
        # the absent twin is an inert managed name).
        obs.family === BernoulliLogitFam &&
            push!(out, _kbool_name(kp.result, obs.response))
    end
    union!(out, _cell_bind_columns(kp))
    return out
end

# Bool-twin materialization (the kexp precedent: deterministic bind
# product, caller-supplied collisions reserved-fail): Bernoulli plates
# read Bool lanes — the `!=` comparison misdifferentiates under native
# Enzyme (snag `bernoulli-int-la-78487520`), so non-Bool flats (integer
# columns, Float64 expansions) gain an exact Bool twin. Dense
# `Vector{Bool}` — broadcast comparison yields a BitVector and
# BitArrays are overlay-hostile (reactant ladder-1b). Returns the twin
# name, or `nothing` when the flat is already Bool.
function _materialize_kernel_bool_twin!(kp::KernelPlate, obs::KernelObs,
        flatv::AbstractVector, columns::Dict{Symbol,ColumnData})
    eltype(flatv) === Bool && return nothing
    twin = _kbool_name(kp.result, obs.response)
    haskey(columns, twin) &&
        _fail(kp.label, "column `$twin` is reserved for kernel plate " *
              "`$(kp.result)`'s Bool twin of `$(obs.response)` — " *
              "rename the caller-supplied column")
    columns[twin] = Vector{Bool}(flatv .!= 0)
    return twin
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

"""
    ScanSetup(target, index, kind, family, args, expr)

One literal-index seed fill of a `@scan` carried array `target`, mirroring
[`ScanStep`](@ref):

- `kind === :sample` — `target[index] ~ Dist(args…)`: a sampled seed.
  `family` is an internal family symbol (see the surface's
  `_PARAM_FAMILIES`), `args` the positional distribution-argument
  expressions; `expr` is `nothing`.
- `kind === :assign` — `target[index] = expr`: a deterministic seed (a
  literal, or an expression over scalar parameters, definitions and earlier
  seeds). `family`/`args` are `nothing`.
"""
struct ScanSetup
    target::Symbol
    index::Int
    kind::Symbol
    family::Union{Symbol,Nothing}
    args::Union{Vector{Any},Nothing}
    expr::Any
end

"""
    ScanStep(kind, target, indexed, family, args, expr)

One `@scan` recurrence-body statement.

- `kind === :sample` — a `~` statement: a carried array at the loop index
  (`indexed = true`, `target` one of the scan's `states`, centered form) or a
  fresh per-step local innovation (`indexed = false`, non-centered form).
  `family`/`args` describe the distribution; `expr` is `nothing`.
- `kind === :assign` — a `=` statement: a carried array's deterministic write
  (`indexed = true`) or a per-step deterministic local (`indexed = false`).
  `expr` is the RHS AST; `family`/`args` are `nothing`.

Statements run in order, as in a Julia loop body: a step reads a carried
array's backward lags `a[t - k]`, its current value `a[t]` once an earlier
step wrote it, and locals defined by earlier steps.
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
    ScanSpec(states, loopvar, lo, hi, setup, step, maxlag, label)

One sequential recurrence (`@scan begin <setup>; for loopvar in lo:hi … end end`):
the carried arrays `states` (in first-seed order; one or more — a tuple
carry), the loop variable and range `lo:hi` (`hi` a literal `Int` or a data
length `Symbol`), the ordered `setup` seed fills (every carried array seeded at
`1:lo-1`), the ordered recurrence `step`s, and the maximum backward lag read.
Each carried array is a plan-level latent the recurrence produces (the value
visible after the block); the per-step locals and `loopvar` are scoped to the
recurrence and never enter the plan name table.
"""
struct ScanSpec
    states::Vector{Symbol}
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
`beta`/`sigma` name the persistence (`Normal` truncated to `[0, 1]`)
and innovation-scale (half-Normal, including `truncated(Normal(0, s), 0, Inf)`)
[`SampledParameter`](@ref)s, each with Distributions semantics (the
truncation normalizers stay);
the `z` innovations (one fewer than the path's rows) are owned internally
under the reserved `_ppl_dar_z_<state>` name, like a non-centered scan's
`_ppl_scan_z_<state>` slice. The path length follows its consuming response's
axis, because the LP direct summand adds elementwise to that predictor.

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
only). `subset` selects from `DataAPI.levels(column)`: `:` (full cover),
a literal `UnitRange{Int}`, a literal `Vector{Int}` of positions, or
`(lo, :end)`. Keyed by `(predictor, column)` — one map per factor term.
"""
struct LevelMap
    predictor::Symbol
    column::ColumnRef
    values::Vector
    source::Symbol
    subset::Union{Colon,UnitRange{Int},Vector{Int},Tuple{Int,Symbol},Tuple{Int,Int,Symbol}}
end

"""
    DesignMatrix(name, columns, label)

One user-bound design matrix (`X = hcat(ones(length(x1)), x1, x2)`): `columns` in hcat
order, `nothing` marking intercept-ones positions. Data/derived columns
only (length-n by bind/construction); latent, scan, parameter, and
nested-matrix names are rejected — the SB `me` mirror stays affine and
nested `hcat` stays a follow-up. The generator emits each matrix once
(`X = Float64.(hcat(...))`); [`MatrixTerm`](@ref)s reference it by name
and splice `X * view(coef, ...)` matvecs. Width is static
(`length(columns)`); the surface sizes coefficient vectors from it.

A matrix no matrix term and no GLM response reads is a VALUE matrix (the
program reads `X` as a matrix: `var.(eachcol(X))`, `X * v` over an array):
[`bind_data`](@ref) builds it from its data columns and binds it under its
name, so every reader sees a bound data matrix
(`_value_design_matrix_names`).
"""
struct DesignMatrix
    name::Symbol
    columns::Vector{Union{Nothing,Symbol}}
    label::Symbol
end

"""A lexical submodel call: its authored path, return-value binding, and
local bindings. Identifiers are private plan names; `path` and local keys
are author names. A per-cell call owns arrays of its scalar local values."""
struct SubmodelScope
    path::Tuple{Vararg{Symbol}}
    binding::Symbol
    locals::Dict{Symbol,Symbol}
    per_cell::Bool
end

function _scope_name_paths(scopes::Vector{SubmodelScope})
    paths = Dict{Symbol,Tuple{Vararg{Symbol}}}()
    for scope in scopes, (local_name, identifier) in scope.locals
        paths[identifier] = (scope.path..., local_name)
    end
    return paths
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
(empty for a plain population-GLM plan). `array_parameters` are the
declared array-valued parameters ([`ArrayParameter`](@ref)) the model reads
by name. `submodel_scopes` records lexical author paths separately from
the private identifiers used by the mathematical plan.
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
    horseshoe_priors::Vector{HorseshoePrior}
    matrices::Vector{DesignMatrix}
    event_lps::Vector{LinearPKEventLPSpec}
    array_parameters::Vector{ArrayParameter}
    submodel_scopes::Vector{SubmodelScope}
    conditioned::Set{Symbol}
    indexed_observations::Set{Symbol}
    external_observations::Vector{SampledParameter}
end

# The former full constructor has no external observations.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, dar_paths, varying_draws, varying_slices, vector_parameters,
    spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
    horseshoe_priors, matrices, event_lps, array_parameters, submodel_scopes,
    conditioned, indexed_observations) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, dar_paths, varying_draws, varying_slices, vector_parameters,
        spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
        horseshoe_priors, matrices, event_lps, array_parameters, submodel_scopes,
        conditioned, indexed_observations, SampledParameter[])

StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, dar_paths, varying_draws, varying_slices, vector_parameters,
    spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
    horseshoe_priors, matrices, event_lps, array_parameters, submodel_scopes,
    conditioned) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, dar_paths, varying_draws, varying_slices, vector_parameters,
        spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
        horseshoe_priors, matrices, event_lps, array_parameters, submodel_scopes,
        conditioned, Set{Symbol}())

StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, dar_paths, varying_draws, varying_slices, vector_parameters,
    spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
    horseshoe_priors, matrices, event_lps, array_parameters, submodel_scopes) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, dar_paths, varying_draws, varying_slices, vector_parameters,
        spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
        horseshoe_priors, matrices, event_lps, array_parameters, submodel_scopes,
        Set{Symbol}())

# Existing full-positional plans have no lexical submodel metadata.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, dar_paths, varying_draws, varying_slices, vector_parameters,
    spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
    horseshoe_priors, matrices, event_lps, array_parameters) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps,
        plate_parameters, scans, dar_paths, varying_draws, varying_slices,
        vector_parameters, spline_bases, spline_vectors, hsgp_bases,
        kernel_plates, r2d2_priors, horseshoe_priors, matrices, event_lps,
        array_parameters, SubmodelScope[])

# Pre-array full-positional constructor (24-arg): plans built before
# `array_parameters` existed keep working with none.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, dar_paths, varying_draws, varying_slices, vector_parameters,
    spline_bases, spline_vectors, hsgp_bases, kernel_plates, r2d2_priors,
    horseshoe_priors, matrices, event_lps) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps,
        plate_parameters, scans, dar_paths, varying_draws, varying_slices,
        vector_parameters, spline_bases, spline_vectors, hsgp_bases,
        kernel_plates, r2d2_priors, horseshoe_priors, matrices, event_lps,
        ArrayParameter[])

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
        R2D2Prior[], HorseshoePrior[], DesignMatrix[],
        LinearPKEventLPSpec[])

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
        horseshoe_priors::Vector{HorseshoePrior} = HorseshoePrior[],
        matrices::Vector{DesignMatrix} = DesignMatrix[],
        event_lps::Vector{LinearPKEventLPSpec} = LinearPKEventLPSpec[],
        array_parameters::Vector{ArrayParameter} = ArrayParameter[],
        submodel_scopes::Vector{SubmodelScope} = SubmodelScope[],
        conditioned::Set{Symbol} = Set{Symbol}(),
        indexed_observations::Set{Symbol} = Set{Symbol}(),
        external_observations::Vector{SampledParameter} = SampledParameter[])
    return StructuralPlan(responses, predictors, population_priors,
        parameters, assignments, derived, _checked_columns(columns), n_obs,
        roles, levelmaps, plate_parameters, scans, dar_paths, varying_draws,
        varying_slices,
        vector_parameters, spline_bases, spline_vectors, hsgp_bases,
        kernel_plates, r2d2_priors, horseshoe_priors, matrices, event_lps,
        array_parameters, submodel_scopes, conditioned, indexed_observations, external_observations)
end

"""The horseshoe entries covering `pred` (empty when the predictor keeps
its sampled coefficient block)."""
_horseshoe_for(plan::StructuralPlan, pred::Symbol) =
    [h for h in plan.horseshoe_priors if h.predictor === pred]

"""Preserve separate simultaneous doses when a cell consumes an event-axis
bioavailability vector. Adding amounts before a nonlinear dose effect would
change the model; schedules without that vector may combine amounts."""
_schedule_combine_simultaneous(plan::StructuralPlan, sched::Symbol) =
    !any(kp -> any(call -> first(call) === sched,
        _event_lp_calls(kp)), plan.kernel_plates)

"""Find a design matrix by name, or `nothing`."""
function _find_matrix(plan::StructuralPlan, name::Symbol)
    i = findfirst(m -> m.name === name, plan.matrices)
    return i === nothing ? nothing : plan.matrices[i]
end

"""Value matrices of `plan`: design matrices no [`MatrixTerm`](@ref) and no
GLM response reads. The program reads each as a matrix value, so
[`bind_data`](@ref) builds it from its columns and binds it as data."""
function _value_design_matrix_names(plan::StructuralPlan)
    used = Set{Symbol}()
    for pred in plan.predictors, t in pred.terms
        # A malformed hand-built term (`_validate_matrix_term` names it)
        # references no matrix here.
        t.kind === MatrixTerm || continue
        X = get(t.options, :matrix, nothing)
        X isa Symbol && push!(used, X)
    end
    for r in plan.responses
        _is_glm_family(r.family) && push!(used, r.predictor)
    end
    return Set{Symbol}(m.name for m in plan.matrices if m.name ∉ used)
end

"""Per-element prior addressees of a design matrix in column order
(`:Intercept` at intercept positions, the column otherwise)."""
_matrix_element_addressees(m::DesignMatrix) =
    Symbol[c === nothing ? :Intercept : c for c in m.columns]

"""Bound plans carry columns or a resolved observation-free execution size.
[`bind_data`](@ref) also binds prior-only programs with no input columns."""
isbound(plan::StructuralPlan) = !isempty(plan.columns) ||
    (plan.n_obs > 0 && !_has_observation_axis(plan))

"""Canonical likelihood links, retained for family vocabulary and stable routes.
Predictor metadata does not transform a value or restrict another use.
Multinomial/Categorical name a
simplex vector parameter instead of a linear predictor, so they skip the
triple (the scan-state precedent) and validate on the simplex path."""
const _CANONICAL_TRIPLES = (
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
    (BetaShapeFam, IdentityLink, IdentityLink),
    (CategoricalLogitFam, LogitLink, IdentityLink),
    (OrderedLogisticFam, LogitLink, IdentityLink),
    (OrdinalFam, LogitLink, IdentityLink),
    (OrdinalFam, ProbitLink, IdentityLink),
    (OrdinalFam, CloglogLink, IdentityLink),
    (StudentTFam, IdentityLink, IdentityLink),
    (HurdlePoissonFam, LogLink, LogLink),
    (ZeroInflatedPoissonFam, LogLink, LogLink),
    (InverseGaussianFam, LogLink, LogLink),
    (BetaBinomial2Fam, LogitLink, IdentityLink),
    (VonMisesFam, IdentityLink, IdentityLink),
    (NegativeBinomialFam, LogLink, LogLink),
    (ExponentialLogFam, LogLink, LogLink),
    (LogNormalFam, IdentityLink, IdentityLink),
    (WeibullFam, LogLink, LogLink),
    (GammaValueFam, IdentityLink, IdentityLink),
    (WeibullValueFam, IdentityLink, IdentityLink),
)

# Predictor metadata never applies a transform. Each scalar likelihood
# parameter owns its use-site inverse; categorical/ordinal links retain
# their distribution-specific meaning.
const _VALUE_LINK_FAMILIES = (GaussianFam, BernoulliLogitFam, PoissonLogFam,
    BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam, BetaLogitFam,
    StudentTFam, HurdlePoissonFam, ZeroInflatedPoissonFam, InverseGaussianFam,
    BetaBinomial2Fam, VonMisesFam, NegativeBinomialFam, ExponentialLogFam,
    LogNormalFam, WeibullFam, ZeroInflatedBinomialFam)
const ADMITTED_TRIPLES = Tuple((family, link, predictor_link)
    for (family, canonical, _) in (_CANONICAL_TRIPLES...,
        (ZeroInflatedBinomialFam, IdentityLink, IdentityLink))
    for link in (family in _VALUE_LINK_FAMILIES ? instances(LinkFunction) : (canonical,))
    for predictor_link in instances(LinkFunction))

"""Positional arity per sampled family (Distributions.jl order; `:student_t`
is `(nu, mu, sigma)` in Stan order, matching the response spelling)."""
const SAMPLED_ARITY = Dict{Symbol,Int}(
    :normal => 2,
    :cauchy => 2,
    :exponential => 1,
    :gamma => 2,
    :lognormal => 2,
    :beta => 2,
    :inverse_gamma => 2,
    :student_t => 3,
    :laplace => 2,
    :logistic => 2,
    :uniform => 2,
    :weibull => 2,
    :flat => 0,
)

"""Inferred unconstrained support per sampled family (`:flat` = real;
`:uniform` bounds come from its literal args)."""
const SAMPLED_SUPPORT = Dict{Symbol,Symbol}(
    :normal => :real,
    :cauchy => :real,
    :flat => :real,
    :student_t => :real,
    :laplace => :real,
    :logistic => :real,
    :exponential => :positive,
    :gamma => :positive,
    :lognormal => :positive,
    :inverse_gamma => :positive,
    :beta => :unit,
    :uniform => :interval,
    :weibull => :positive,
)

"""Admitted per-addressee population-prior families (prior-vocab slice)."""
const POPULATION_FAMILIES =
    (:normal, :student_t, :cauchy, :laplace, :logistic, :flat, :uniform)

"""Real-support families symmetric about their location: a `:positive`
half at a literal-zero location renormalizes by exactly `+log(2)`."""
const SYMMETRIC_SAMPLED_FAMILIES =
    (:normal, :cauchy, :student_t, :laplace, :logistic)

# Spline block/width/vector/name rules, derived purely from (kind, k):
# the single source of truth shared by surface lowering (which builds the
# nodes) and contract validation (which re-derives and compares). Widths
# are static — the fit can only confirm them at bind, never change them.
# Block order is SB's data order (:fixed first, then pen/rr/rn/nr); the t2
# sd index follows the pen-block position (rr→1, rn→2, nr→3). Neither
# kind carries a constant fixed column: the author's intercept owns it
# (tps is one column narrower than SB's, `_rk_apply_spline`).
function _spline_blocks(kind::Symbol, k::Union{Int,Tuple{Int,Int}})
    if kind === :tps
        k isa Int ||
            _fail(:plan, "tps spline k must be an Int, got $(repr(k))")
        return [(:fixed, 1), (:pen, k - 2)]
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

# `sd_prior` is the stated smoothing-sd hyper prior (`spline_basis(...;
# sd=...)`), `nothing` for the default `Normal(0, 1)`; its family fixes the
# sd vector's support (`_hyper_support_override`).
function _spline_vector_specs(id::Symbol, kind::Symbol, k,
        sd_prior::Union{Nothing,HyperPrior} = nothing)
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
    sdfam, sdargs = sd_prior === nothing ? (:normal, (arg1=0, arg2=1)) :
        (sd_prior.family, sd_prior.args)
    push!(specs, (sd, sdfam, sdargs, sd_prior === nothing ? :positive : _hyper_support_override(sd_prior), nsd))
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
    hyper(tag, spec) = spec isa HSGPHyperLP ?
        (beta0 = Symbol("beta0_", tag, :_, id), sd = Symbol("sd_", tag, :_,
            id), z = Symbol("z_", tag, :_, id), intercept = spec.intercept) :
        nothing
    return (beta = beta, rhos = rhos, sigma = sigma,
        rho_hyper = hyper(:rho, hb.rho_prior),
        sigma_hyper = hyper(:sigma, hb.sigma_prior))
end

"""Group count of an [`HSGPBasis`](@ref) (1 ungrouped; the bound
`by` level count grouped — `nothing` levels pre-bind count as 0)."""
_hsgp_n_groups(hb::HSGPBasis) = hb.by === nothing ? 1 :
    hb.by.levels === nothing ? 0 : length(hb.by.levels)

"""Basis-function count for an [`HSGPBasis`](@ref): `M = prod(K)`
exp-quad, `M = 2k` periodic (cosines then sines over `k`
harmonics — SB `2 * only(K)`)."""
_hsgp_n_basis(hb::HSGPBasis) =
    hb.cov === :periodic ? 2 * only(hb.K) : prod(hb.K)

# Flat sampled-name list for name tables + claims (beta, rhos, sigma).
function _hsgp_all_names(hb::HSGPBasis)
    n = _hsgp_names(hb)
    out = Symbol[n.beta, n.sigma, n.rhos...]
    for h in (n.rho_hyper, n.sigma_hyper)
        h === nothing && continue
        h.intercept && push!(out, h.beta0)
        push!(out, h.sd, h.z)
    end
    return out
end

# Every model-scope name a kernel plate introduces (cell names become flat
# model-scope locals at codegen): the result, slice params, and cell-local
# assignment names. Grouped plates additionally introduce their LP cell
# params and schedule handles — EXCEPT self-aliasing params (`(c, c)`
# slices, `(pname, pname)` LP refs): those are lexical references to an
# outer column/definition (the plate spelling), not introductions.
# Single source for the global name-table gate.
function _kernel_all_names(kp::KernelPlate)
    # A top-level schedule chain names its kernel by the cell value its
    # first observation reads (`result === collected`, an assignment —
    # counted once, below).
    names = any(p -> p.first === kp.result, kp.assignments) ? Symbol[] :
        Symbol[kp.result]
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
    per = Float64[_hsgp_axis_floor(k, L) for (k, (_, L)) in zip(K, fits)]
    return iso ? [maximum(per)] : per
end

"""One axis's length-scale validity floor for `K` basis functions on the
half-width-`L` domain (SB `_brm_hsgp_rho_lower`): `(4L/pi) *
sqrt(log(100)/(K^2-1))`, `0.0` (unbounded) at `K == 1`. Shared by the
built-in layout floors and [`hsgp_rho_floors`](@ref)."""
_hsgp_axis_floor(K::Integer, L::Real) =
    K == 1 ? 0.0 : (4 * L / pi) * sqrt(log(100.0) / (K * K - 1))

"""Exp-quad HSGP eigenvalue of basis function `k` on the half-width-`L`
domain (SB `lambda`): `(k*pi/(2L))^2`. Shared by the in-graph built-in
basis and the data-side [`hsgp_basis`](@ref)."""
_hsgp_lambda(k::Integer, L::Real) = (k * pi / (2.0 * L))^2

"""
    _hsgp_periodic_rho_lower(K) -> Float64

Periodic length-scale validity floor for `K` harmonics (SB
`_brm_hsgp_periodic_rho_lower` verbatim): the `rho` at which the
`K`-th harmonic's spectral amplitude has fallen to `1/100` of the
first's (`I_K(a)/I_1(a) = 100^-2` with `a = 1/rho^2`), solved by
bisection on the exponentially scaled Bessel functions. Depends on
`K` alone — no data-derived domain. `K == 1` stays unbounded
(`0.0`, the exp-quad degenerate-basis rule).
"""
function _hsgp_periodic_rho_lower(K::Integer)
    K > 1 || return 0.0
    target = 100.0^-2
    ratio(loga) = let a = exp(loga)
        SpecialFunctions.besselix(K, a) /
            SpecialFunctions.besselix(1, a) - target
    end
    lo, hi = log(1e-12), log(1e7)
    ratio(lo) < 0 < ratio(hi) || error(
        "hsgp: internal periodic validity-floor bracket failed for k=$K")
    for _ in 1:200
        mid = (lo + hi) / 2
        ratio(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return 1 / sqrt(exp((lo + hi) / 2))
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
    :composed => ComposedTerm,
)

"""Allowlisted assignment functions (slice 1: scalar ops + whole-column
reductions; elementwise math over columns deferred with vector assignments;
the AR(1) slice adds `tanh` for the `phi = tanh(phi_raw)` stationarity map)."""
const ASSIGNMENT_FNS = (
    :+, :-, :*, :/, :^,
    :log, :log10, :log1p, :exp, :expm1, :sqrt, :abs, :tanh, :logaddexp,
    :sum, :mean, :std, :var, :minimum, :maximum, :length,
)

"""Cell-callable functions (grouped kernels): admitted in grouped-kernel
cell assignments ONLY, always with a declared schedule as the first
argument. Each call exposes an RK subject plate containing retained event
scans. Standalone and grouped entry points use the same authored step graph."""
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

_sched_materialized_fields(::LinearPKScheduleSpec) = _SCHED_MATERIALIZED_FIELDS
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

"""Whether any top-level cell assignment calls a PK recurrence
(same top-level-only reading as [`_cell_has_auc_call`](@ref);
segmented-nadir calls are data-driven and do not count)."""
function _cell_has_pk_call(assignments::Vector{Pair{Symbol,Any}})
    for (_, ex) in assignments
        ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] isa Symbol && ex.args[1] in CELL_FNS &&
            return true
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
`Expr(:., f, ...)`; single-argument, mirroring the scalar math subset —
plus two-argument `logaddexp` for occupancy marginalization)."""
const ELEMENTWISE_FNS =
    (:log, :log10, :log1p, :exp, :expm1, :sqrt, :abs, :logaddexp, :logistic)

const _KERNEL_ELEMENTWISE_FNS = (ELEMENTWISE_FNS..., :logistic)

"""Operand count of a built-in elementwise map (`ifelse.(c, x, y)`,
`logaddexp.(a, b)`; every other one takes one)."""
const ELEMENTWISE_FN_ARITY = Dict{Symbol,Int}(:ifelse => 3, :logaddexp => 2)
_elementwise_arity(f::Symbol) = get(ELEMENTWISE_FN_ARITY, f, 1)
_operands_phrase(n::Int) = ("one", "two", "three")[n] *
    (n == 1 ? " operand" : " operands")

"""Families the thin layer can lower (ext handshake predicate)."""
admitted_families() = (GaussianFam, BernoulliLogitFam, PoissonLogFam,
    BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam,
    BernoulliProbitFam, BernoulliCloglogFam, BinomialProbitFam,
    BinomialCloglogFam, BinomialProbFam, BetaLogitFam, CategoricalLogitFam,
    OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam,
    MvNormalCholeskyFam, NormalIDGLMFam, BernoulliLogitGLMFam,
    PoissonLogGLMFam, MixtureFam, StudentTFam, HurdlePoissonFam,
    ZeroInflatedPoissonFam, InverseGaussianFam, BetaBinomial2Fam, VonMisesFam,
    NegativeBinomialFam, ExponentialLogFam, LogNormalFam, WeibullFam,
    ZeroInflatedBinomialFam, GammaValueFam, WeibullValueFam, BetaShapeFam)

"""Term kinds the thin layer can lower (ext handshake predicate)."""
admitted_terms() = (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm,
    VaryingEffectTerm, SplineSummandTerm, HSGPSummandTerm,
    ScanSummandTerm, MonotonicTerm, MonotonicSummandTerm, MatrixTerm,
    DarSummandTerm, ComposedTerm)

"""Assignment functions the thin layer can lower (ext handshake predicate):
scalar/reduction vocabulary plus vector-returning whole-column functions —
the BUILT-IN vocabulary. Beyond it, an `=` definition may call any function
visible in the model's module (functions as values): a data-only call is
evaluated once by [`bind_data`](@ref), any other runs in the generated
kernel."""
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
    _validate_array_parameters(plan)
    _validate_scans(plan)
    _validate_dar_paths(plan)
    _validate_assignments_structure(plan)
    _validate_vector_structure(plan)
    _validate_parameters(plan)
    _validate_plate_parameters(plan)
    for observation in plan.external_observations
        _validate_external_parameter(plan, observation; observation=true)
    end
    _validate_vector_parameters(plan)
    _validate_topo_order(plan)
    _validate_matrices(plan)
    _validate_predictors(plan)
    _validate_levelmaps(plan)
    _validate_priors(plan)
    _validate_r2d2(plan)
    _validate_horseshoe(plan)
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
    _validate_array_parameters_data(plan)
    _validate_assignments_data(plan)
    _validate_vector_data(plan)
    _validate_predictor_columns(plan)
    _validate_levelmaps_data(plan)
    _validate_response_data(plan)
    _validate_plate_parameters_data(plan)
    for s in plan.scans
        _scan_length(plan, s)
    end
    for s in plan.dar_paths
        _value_rows(plan, s.state)
    end
    _validate_varying_draws_data(plan)
    _validate_splines_data(plan)
    _validate_hsgp_data(plan)
    _validate_kernels_data(plan)
    _validate_r2d2_data(plan)
    return nothing
end

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

# Sampled names, derived purely from the draws suffix: the LKJ
# Cholesky factor (`L_<s>`, KxK), the marginal-scale vector
# (`tau_<s>`, K), and the standardized draws (`z_flat_<s>`, K*G
# column-major). Single source for surface
# claims, name tables, layout, and the generator. K=1 draws own the
# same three names (`L` packs zero coords).
function _varying_corr_names(d::VaryingDraws)
    s = d.suffix
    return (Symbol("L_", s), Symbol("tau_", s),
        Symbol("z_flat_", s))
end

_is_correlated_kind(kind) = kind === :correlated

# Stratified sampled names, derived from the draws suffix + stratum
# position: per-stratum LKJ factor (`L_<s>_s<k>`, KxK) and
# marginal-scale vector (`tau_<s>_s<k>`, K). The standardized draws
# (`z_flat_<s>`, K*G column-major) are shared — the third of
# `_varying_corr_names(d)`. Single source for name tables, layout,
# and the generator.
function _varying_strata_names(d::VaryingDraws, k::Int)
    d.strata !== nothing ||
        _fail(d.label, "draws block is not stratified (per-stratum " *
              "names need a `gr(g, by=b)` grouping)")
    s = d.suffix
    return (Symbol("L_", s, "_s", k), Symbol("tau_", s, "_s", k))
end

# Sampled names one `:correlated` draws block contributes to the name
# tables: the shared triple, or — stratified with known strata levels
# (bound plans) — the per-stratum frames plus the shared `z_flat`.
# Unbound stratified draws contribute only the shared `z_flat` (the
# per-stratum names need S, unknown pre-bind — the bound pass checks
# the real names, so no placeholder can falsely collide).
function _varying_corr_table_names(d::VaryingDraws)
    st = d.strata
    st === nothing && return collect(_varying_corr_names(d))
    if st.levels === nothing
        return [_varying_corr_names(d)[3]]
    end
    z = _varying_corr_names(d)[3]
    out = Symbol[z]
    for k in 1:length(st.levels)
        append!(out, _varying_strata_names(d, k))
    end
    return out
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
    # (draws, target), and no range is empty. mm draws feed exactly
    # one target (SB rejects `mm(...)` with a shared `|ID|` — defense
    # in depth against emitter bugs; stratified multi-slice stays
    # legal for `|ID|+gr`, which SB supports).
    for d in draws
        K = length(d.margins)
        own = [s for s in slices if s.draws === d.label]
        if d.mm !== nothing
            length(own) <= 1 ||
                _fail(d.label, "multi-membership draws feed " *
                      "$(length(own)) targets (SB rejects `mm(...)` with " *
                      "a shared `|ID|` — one slice per mm draws block)")
        end
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
    _is_correlated_kind(d.kind) ||
        _fail(d.label, "draws kind must be :correlated " *
              "(one geometry for every K — a single " *
              "margin is the 1x1 case), got $(repr(d.kind))")
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
    isfinite(d.lkj_eta) && d.lkj_eta > 0 ||
        _fail(d.label, "draws need a positive LKJ eta, got $(d.lkj_eta)")
    K == 1 && d.lkj_eta != 1.0 &&
        _fail(d.label, "K=1 draws carry LKJ eta 1.0 (the 1x1 factor is " *
              "the fixed `[1]`, so no other eta parameterizes anything), " *
              "got $(d.lkj_eta)")
    _validate_sd_priors(d, K)
    _validate_draws_grouping(d, K)
    _validate_varying_levels_shape(d)
    return nothing
end

# Multi-membership / stratified grouping shape (provable without
# data). Both are `:correlated` with eta exactly 1.0 (SB hardcodes
# `lkj_corr_cholesky(1.)` on both paths). Neither path has an SB
# generic-prior sibling, so both reject non-default `sd_priors`; SB
# has no mm × stratified shape.
function _validate_draws_grouping(d::VaryingDraws, K::Int)
    mm = d.mm
    st = d.strata
    mm === nothing && st === nothing && return nothing
    mm !== nothing && st !== nothing &&
        _fail(d.label, "draws block is both multi-membership and " *
              "stratified (SB has no `mm(...)` × `gr(g, by=b)` shape — " *
              "pick one)")
    if mm !== nothing
        M = length(mm.groups)
        M >= 2 ||
            _fail(d.label, "multi-membership draws need at least two " *
                  "grouping columns, got $M")
        # Repeated groups/weights are degenerate but well-defined
        # (slots are positional) and SB accepts them — no check.
        if mm.weights !== nothing
            W = length(mm.weights)
            W == M ||
                _fail(d.label, "multi-membership draws list $W weight " *
                      "columns for $M groups (one per group, or omit all)")
        end
        d.lkj_eta == 1.0 ||
            _fail(d.label, "multi-membership draws need eta 1.0 (SB " *
                  "hardcodes `lkj_corr_cholesky(1.)`), got $(d.lkj_eta)")
        isempty(d.sd_priors) ||
            _fail(d.label, "multi-membership draws take no sd priors " *
                  "(SB has no generic-prior mm sibling)")
    end
    if st !== nothing
        d.lkj_eta == 1.0 ||
            _fail(d.label, "stratified draws need eta 1.0 (SB hardcodes " *
                  "`lkj_corr_cholesky(1.)`), got $(d.lkj_eta)")
        isempty(d.sd_priors) ||
            _fail(d.label, "stratified draws take no sd priors (SB " *
                  "supports no configured priors on `gr(g, by=b)` blocks)")
        st.by !== d.group ||
            _fail(d.label, "stratified draws need distinct group and " *
                  "stratum columns, got `gr($(d.group), by=$(d.group))`")
    end
    return nothing
end

# Per-margin `tau` priors: empty (the default) is all-`:std_normal`,
# otherwise one entry per margin in margin order. `:exponential` takes
# a finite positive SCALE, `:normal` a finite positive sd;
# `:std_normal` ignores its param (finite, conventionally 1.0). Every
# K takes them (one geometry for every K).
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
        p.family === :std_normal || p.param > 0 ||
            _fail(d.label, "margin $j sd prior needs a positive " *
                  "param, got $(p.param)")
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

# A latent plate owns its declared domain; aligned response uses must cover
# their authored indices or obey Julia's broadcast dimensions.
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
        if p.range isa Symbol
            # One cell per entry of the iterated column (Julia's
            # `eachindex(v)`), with the column's own domain.
            (haskey(plan.columns, p.range) || _is_derived(plan, p.range) ||
                any(a -> a.name === p.range, plan.assignments)) || _fail(p.label,
                "plate over `eachindex($(p.range))`: `$(p.range)` is not bound or defined")
        end
        # The declaration owns its domain. A reduction or gather may use the
        # whole latent array beside responses with unrelated lengths. Only
        # aligned uses impose observation dimensions, checked below.
        n = _plate_rows(plan, p)
        names = Set{Symbol}([p.name])
        for r in plan.responses
            p.name in _response_reads(plan, r, names; whole_values=true) || continue
            if r.response in plan.indexed_observations
                indices = _response_range_indices(plan, r)
                checkindex(Bool, Base.OneTo(n), indices) || _fail(p.label,
                    "latent vector $(p.name) does not cover the authored " *
                    "response range $(repr(r.range))")
                continue
            end
            m = _response_rows(plan, r)
            (n == m || n == 1 || m == 1) || _fail(p.label,
                "latent vector $(p.name) has $n cells, which cannot broadcast " *
                "with the $m rows of $(r.response)")
        end
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
        if d.mm !== nothing
            _validate_mm_draws_data(d, plan)
        else
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
            groupcol = _vector_column(plan.columns, d.group, d.label,
                "grouping column")
            for v in groupcol
                v in levels ||
                    _fail(d.label, "grouping value $(repr(v)) of $(d.group) " *
                          "is not a declared level (declared: $(repr(levels)))")
            end
        end
        if d.strata !== nothing
            _validate_strata_draws_data(d, plan)
        end
        for m in d.margins
            _validate_margin_data(m, d.label, plan)
        end
    end
    # One grouping, one numbering: same-group draws share the per-group
    # `_ppl_gidx_` encoder, so their declared levels must agree exactly
    # (order included — codes are positions). mm draws carry per-suffix
    # encoders (never shared), so they sit out the agreement.
    for i in eachindex(plan.varying_draws)
        for j in (i + 1):length(plan.varying_draws)
            di, dj = plan.varying_draws[i], plan.varying_draws[j]
            di.group === dj.group || continue
            di.mm === nothing && dj.mm === nothing || continue
            di.levels == dj.levels ||
                _fail(:plan, "draws $(di.label) and $(dj.label) share " *
                      "grouping $(di.group) but declare different levels " *
                      "($(repr(di.levels)) vs $(repr(dj.levels)))")
        end
    end
    return nothing
end

# Multi-membership data checks (SB `_brm_prepare_mm` mirror): every
# membership column bound, raw, n_obs-long, and covered by the UNION
# levels; every weight column bound, vector, real non-Bool, finite,
# nonnegative, n_obs-long, with a positive finite row total (SB
# validates totals whenever weights are supplied, normalized or not).
function _validate_mm_draws_data(d::VaryingDraws, plan::StructuralPlan)
    mm = d.mm::VaryingMultiMembership
    M = length(mm.groups)
    d.levels === nothing &&
        _fail(d.label, "draws block has no declared grouping levels " *
              "(bind_data fills the union — hand-built bound plans must too)")
    levels = d.levels::Vector
    n_obs = _value_rows(plan, first(mm.groups))
    for (mi, gcol) in enumerate(mm.groups)
        haskey(plan.columns, gcol) ||
            _fail(d.label, "membership column $gcol (slot $mi of $M) " *
                  "is not bound")
        _is_derived(plan, gcol) &&
            _fail(d.label, "membership column $gcol must be raw data " *
                  "(level knowledge needs bound values)")
        col = _vector_column(plan.columns, gcol, d.label, "membership column")
        length(col) == n_obs ||
            _fail(d.label, "membership column $gcol has $(length(col)) " *
                  "rows; expected $n_obs")
        for v in col
            v in levels ||
                _fail(d.label, "grouping value $(repr(v)) of $gcol " *
                      "is not a declared union level (declared: " *
                      "$(repr(levels)))")
        end
    end
    mm.weights === nothing && return nothing
    for (mi, wcol) in enumerate(mm.weights)
        haskey(plan.columns, wcol) ||
            _fail(d.label, "weight column $wcol (slot $mi of $M) " *
                  "is not bound")
        w = _vector_column(plan.columns, wcol, d.label, "weight column")
        eltype(w) <: Real && !(eltype(w) <: Bool) ||
            _fail(d.label, "weight column $wcol must be real-valued, got " *
                  "eltype $(eltype(w))")
        length(w) == n_obs ||
            _fail(d.label, "weight column $wcol has $(length(w)) rows; " *
                  "expected $n_obs")
        for (i, v) in enumerate(w)
            isfinite(v) ||
                _fail(d.label, "weight column $wcol at row $i must be " *
                      "finite, got $(repr(v))")
            v >= 0 ||
                _fail(d.label, "weight column $wcol at row $i must be " *
                      "nonnegative, got $(repr(v))")
        end
    end
    wcols = [plan.columns[w] for w in mm.weights]
    for i in 1:n_obs
        total = sum(Float64(w[i]) for w in wcols)
        isfinite(total) ||
            _fail(d.label, "weights have a non-finite total at row $i")
        total > 0 ||
            _fail(d.label, "weights must have a positive total at row $i")
    end
    return nothing
end

# Stratified data checks: the `by` column bound, raw, n_obs-long, and
# covered by the strata levels; every group level sits in exactly one
# stratum (SB `_brm_group_strata` — a straddling group is loud).
function _validate_strata_draws_data(d::VaryingDraws, plan::StructuralPlan)
    st = d.strata::VaryingStrata
    haskey(plan.columns, st.by) ||
        _fail(d.label, "stratum column $(st.by) is not bound")
    _is_derived(plan, st.by) &&
        _fail(d.label, "stratum column $(st.by) must be raw data " *
              "(level knowledge needs bound values)")
    st.levels === nothing &&
        _fail(d.label, "draws block has no declared strata levels " *
              "(bind_data fills these — hand-built bound plans must too)")
    slevels = st.levels::Vector
    bycol = _vector_column(plan.columns, st.by, d.label, "stratum column")
    n = _value_rows(plan, d.group)
    length(bycol) == n ||
        _fail(d.label, "stratum column $(st.by) has $(length(bycol)) " *
              "rows; expected $n")
    for v in bycol
        v in slevels ||
            _fail(d.label, "stratum value $(repr(v)) of $(st.by) " *
                  "is not a declared stratum (declared: $(repr(slevels)))")
    end
    gcol = _vector_column(plan.columns, d.group, d.label, "grouping column")
    smap = Dict{Any,Any}()
    for (g, b) in zip(gcol, bycol)
        if haskey(smap, g)
            smap[g] == b ||
                _fail(d.label, "gr($(d.group), by=$(st.by)): group level " *
                      "$(repr(g)) straddles multiple strata " *
                      "($(repr(smap[g])) vs $(repr(b)))")
        else
            smap[g] = b
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

# Stratum count for a stratified draws block: the DECLARED strata
# level count. Loud defense in depth — validate_data proves strata
# levels non-nothing on every bound stratified plan.
function _strata_nlevels(d::VaryingDraws)
    st = d.strata
    st === nothing && throw(ContractValidationError(
        "[layout] draws $(d.label) is not stratified (stratum count " *
        "needs a `gr(g, by=b)` grouping)"))
    st.levels === nothing && throw(ContractValidationError(
        "[layout] draws $(d.label) has no declared strata levels " *
        "(bind_data fills these — hand-built bound plans must too)"))
    return length(st.levels)
end

# Binder evaluation for draws levels (the LevelMap precedent):
# `nothing` fills sort-ordered observed levels; emitter-provided levels
# pass through (validated by `_validate_varying_levels_shape` +
# `_validate_varying_draws_data`). mm draws fit the UNION across
# membership columns (SB `_brm_mm_fit_levels`); stratified draws also
# fill strata levels from the `by` column.
function _eval_draws_levels(draws::Vector{VaryingDraws},
        columns::AbstractDict{Symbol})
    out = VaryingDraws[]
    for d in draws
        if d.mm !== nothing
            push!(out, _bind_mm_draws(d, columns))
            continue
        end
        if d.levels !== nothing
            push!(out, _maybe_fill_strata(d, columns))
            continue
        end
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
        d2 = VaryingDraws(d.group, d.kind, d.margins, d.lkj_eta,
            d.label, d.suffix, collect(levels), d.sd_priors, d.mm, d.strata)
        push!(out, _maybe_fill_strata(d2, columns))
    end
    return out
end

# Union-level fit for one mm draws block (SB `_brm_mm_fit_levels`:
# pool per-column levels, dedup, sort). Emitter-provided union levels
# (the categorical escape hatch — same policy as plain groupings)
# pass through untouched.
function _bind_mm_draws(d::VaryingDraws, columns::AbstractDict{Symbol})
    mm = d.mm::VaryingMultiMembership
    d.levels !== nothing && return _maybe_fill_strata(d, columns)
    pooled = Any[]
    for g in mm.groups
        haskey(columns, g) ||
            _fail(d.label, "membership column $g is not bound")
        col = _vector_column(columns, g, d.label, "membership column")
        append!(pooled, _grouping_levels(col))
    end
    unique!(pooled)
    levels =
        try
            sort!(pooled)
        catch err
            _fail(d.label, "membership columns have levels that are not " *
                         "mutually orderable ($err)")
        end
    d2 = VaryingDraws(d.group, d.kind, d.margins, d.lkj_eta,
        d.label, d.suffix, collect(levels), d.sd_priors, d.mm, d.strata)
    return _maybe_fill_strata(d2, columns)
end

# Strata-level fill for one stratified draws block (sort-ordered
# observed levels of the `by` column — the group-level precedent).
# Anything unstratified (or already-filled) passes through untouched.
function _maybe_fill_strata(d::VaryingDraws, columns::AbstractDict{Symbol})
    st = d.strata
    st === nothing && return d
    st.levels !== nothing && return d
    haskey(columns, st.by) ||
        _fail(d.label, "stratum column $(st.by) is not bound")
    bycol = _vector_column(columns, st.by, d.label, "stratum column")
    slevels =
        try
            _grouping_levels(bycol)
        catch err
            _fail(d.label, "stratum column $(st.by) levels not " *
                         "orderable ($err)")
        end
    return VaryingDraws(d.group, d.kind, d.margins, d.lkj_eta,
        d.label, d.suffix, d.levels, d.sd_priors, d.mm,
        VaryingStrata(st.by, collect(slevels)))
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
    r.nu isa ScalePredictorRef && r.nu.predictor === pname &&
        return true
    r.zi isa ScalePredictorRef && r.zi.predictor === pname &&
        return true
    r.discrimination isa ScalePredictorRef &&
        r.discrimination.predictor === pname && return true
    r.discrimination === pname && return true
    # Mixture slots ride dedicated fields (the anchor may name a
    # parameter, and non-anchor component predictors live in the slots).
    pname in r.mixture_locs && return true
    for s in r.mixture_scales
        s isa ScalePredictorRef && s.predictor === pname && return true
    end
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

"""Bound columns without an observation axis of their own: `(modelvals,
managed)` — model-level values (bind-time data definitions and raw inputs
read only whole) carry no axis; kernel-, subject- and mi-managed columns
carry two lengths by design and validate under their own rules."""
function _axis_exempt_columns(plan::StructuralPlan)
    managed = Set{Symbol}()
    for kp in plan.kernel_plates
        union!(managed, _kernel_managed_columns(kp))
    end
    union!(managed, _subject_predictor_columns(plan))
    union!(managed, _mi_managed_columns(plan))
    # Model-level data values (functions as values: a data-only
    # assignment bound at bind, or a data-only definition only used
    # whole) and the raw inputs read only whole carry no observation
    # axis.
    computed = _bound_module_data_names(plan)
    inputs, wholedefs = _bound_model_level_inputs(plan)
    modelvals = union(
        intersect(computed, Set{Symbol}(a.name for a in plan.assignments)),
        intersect(computed, wholedefs), inputs)
    union!(modelvals, (n for n in computed if plan.columns[n] isa Number))
    return modelvals, managed
end

# ── Observation axes ─────────────────────────────────────────────────
# A statement broadcasts over the columns it reads (standard Julia), so
# the rows one observation statement reads need not be the rows another
# reads: `y1 .~ Normal.(b0 .* x1, s)` over 4 rows beside `y2 .~
# Poisson.(exp.(b0 .* x2))` over 3. Responses that read a common
# per-observation column share an axis. When every per-observation column
# has the same rows there is one axis, `n_obs`, and nothing below runs
# (plans bind exactly as before); otherwise each column has the rows of
# the one axis that reads it, and `n_obs` is the total observed rows (the
# kernel-plate precedent: total likelihood lanes).

"""Plan slots whose dimensions resolve from their authored inputs or uses.
New slots must establish the same property before joining this list."""
const _MULTI_AXIS_SLOTS = (:responses, :predictors, :population_priors,
    :parameters, :assignments, :derived, :columns, :n_obs, :roles,
    :levelmaps, :vector_parameters, :submodel_scopes, :conditioned,
    :plate_parameters, :scans, :dar_paths, :varying_draws, :varying_slices,
    :spline_bases, :spline_vectors, :hsgp_bases, :matrices,
    :r2d2_priors, :horseshoe_priors, :array_parameters, :kernel_plates,
    :event_lps, :indexed_observations, :external_observations)

# Observation-shaped values and their data dependencies. Parameters sized
# by levels or coefficient width are shared values, so their priors do not
# join observation axes. A draws application, basis or matrix does carry
# rows and must expose its inputs through the same dependency walk.
function _observation_nodes(plan::StructuralPlan)
    nodes = Dict{Symbol,Any}()
    for p in plan.predictors
        draws = [d for sl in plan.varying_slices if sl.target === p.name
            for d in plan.varying_draws if d.label === sl.draws]
        nodes[p.name] = (p, draws)
    end
    for d in plan.derived
        nodes[d.name] = d.expr
    end
    for a in plan.assignments
        nodes[a.name] = a.expr
    end
    for m in plan.matrices
        nodes[m.name] = m.columns
    end
    for sb in plan.spline_bases
        nodes[sb.id] = (sb.axes, [b.columns for b in sb.blocks])
    end
    for hb in plan.hsgp_bases
        nodes[hb.id] = (hb.axes, hb.by)
    end
    for p in plan.plate_parameters
        # The iterator determines the latent array's extent. Prior inputs
        # affect its density, but do not join a consuming response's axis.
        nodes[p.name] = p.range
    end
    for s in plan.scans, state in s.states
        nodes[state] = s.hi
    end
    for p in plan.array_parameters
        # An axes(X, 1) vector has X's rows. Width- and level-sized
        # coefficient arrays stay shared values rather than joining axes.
        length(p.dims) == 1 || continue
        d = only(p.dims)
        _is_axis_dim(d) && d.args[3] == 1 || continue
        nodes[p.name] = (d.args[2], p.args)
    end
    return nodes
end

"""Per-observation columns (of `perobs`) a response reads: the names its
own fields hold, then — transitively — the names every predictor, derived
column and definition among them holds. `whole_values=true` follows composed
expressions and treats reductions as whole-array reads for latent-domain
validation, without changing the default observation-axis classification."""
function _response_reads(plan::StructuralPlan, r,
        perobs::Set{Symbol}; whole_values::Bool=false)
    nodes = _observation_nodes(plan)
    cands = union(perobs, keys(nodes))
    function record_reads(x)
        free = copy(cands)
        _drop_held_names!(free, x)
        return setdiff(cands, free)
    end
    function held(x)
        if x isa Expr
            whole, aligned = Set{Symbol}(), Set{Symbol}()
            # A gather consumes its source whole. Module-call arguments
            # also carry their own axes rather than the result's axis.
            _classify_reads!(whole, aligned, x, cands, false;
                reductions_whole=whole_values)
            return aligned
        end
        return record_reads(x)
    end
    # A composed term's columns are dependency metadata, not aligned reads.
    # Follow its mathematical expression so reductions and gathers retain
    # their own domains, even when nested under predictor records.
    held(x::TermSpec) = !whole_values ? record_reads(x) :
        x.kind === ComposedTerm ? held(x.options.tree) :
            union(held(x.columns), held(x.options))
    held(x::PredictorSpec) = whole_values ?
        reduce(union, (held(t) for t in x.terms); init=Set{Symbol}()) : record_reads(x)
    held(x::Tuple) = whole_values ?
        reduce(union, (held(v) for v in x); init=Set{Symbol}()) : record_reads(x)
    reads = Set{Symbol}()
    seen = Set{Symbol}()
    queue = collect(held(r))
    while !isempty(queue)
        s = pop!(queue)
        s in seen && continue
        push!(seen, s)
        s in perobs && push!(reads, s)
        haskey(nodes, s) && append!(queue, held(nodes[s]))
    end
    return reads
end

# Structured constructors keep their established row-domain contract. Ordinary
# elementwise responses use Julia broadcast axes, including singleton dimensions.
function _uses_structured_observation_axes(plan::StructuralPlan)
    any(r -> r.mi_jobs !== nothing, plan.responses) && return true
    !isempty(plan.plate_parameters) && !any(r -> r.range isa Expr, plan.responses) && return true
    return any(f -> !isempty(getfield(plan, f)),
        (:scans, :dar_paths, :varying_draws, :varying_slices,
         :spline_bases, :spline_vectors, :hsgp_bases, :matrices,
         :r2d2_priors, :horseshoe_priors, :kernel_plates, :event_lps))
end

"""Observation axes of a plan whose `columns` are bound: `nothing` when
every response column has the same rows (one axis — `n_obs`; any other
column validates against it), otherwise `(; rows, total)` with `rows`
mapping each per-observation column to the rows of the axis that reads it
and `total` the observed rows summed over axes. Fails when the plan fills
a slot outside [`_MULTI_AXIS_SLOTS`](@ref), when the columns one axis
reads differ in rows, or when no observation statement reads a column."""
function _structured_observation_axes(plan::StructuralPlan)
    isempty(plan.responses) && return nothing
    modelvals, managed = _axis_exempt_columns(plan)
    mi_managed = _mi_managed_columns(plan)
    perobs = Set{Symbol}(k for (k, v) in plan.columns
        if k ∉ modelvals && k ∉ mi_managed &&
            v isa Union{AbstractVector,AbstractMatrix})
    resps = [r.response for r in plan.responses]
    all(r -> haskey(plan.columns, r.response), plan.responses) || return nothing
    rows = Dict{Symbol,Int}(c => _column_nrows(plan.columns[c]) for c in perobs)
    response_rows = [_structured_response_rows(plan, r) for r in plan.responses]
    length(unique(response_rows)) <= 1 && isempty(plan.kernel_plates) && return nothing
    lens = join(sort!(unique(response_rows)), ", ")
    for f in fieldnames(StructuralPlan)
        f in _MULTI_AXIS_SLOTS && continue
        isempty(getfield(plan, f)) || _fail(:plan, "the responses " *
            "observe $lens rows (several observation axes), and `$f` " *
            "reads or sizes by a single observation axis — `$f` beside " *
            "several observation axes is not built yet")
    end
    reads = [_response_reads(plan, r, perobs) for r in plan.responses]
    # A kernel-managed column used by an ordinary response also has to
    # agree with that response's rows. Other kernel columns keep their
    # own subject/time or schedule validation.
    filter!(c -> c ∉ managed || any(cs -> c in cs, reads), perobs)
    # Union-find: responses reading a common column share an axis.
    link = collect(eachindex(reads))
    root(i) = link[i] == i ? i : (link[i] = root(link[i]))
    owner = Dict{Symbol,Int}()
    for (i, rd) in enumerate(reads), c in rd
        link[root(i)] = root(get!(owner, c, i))
    end
    axisrows = Dict{Symbol,Int}()
    total = 0
    for i in eachindex(reads)
        root(i) == i || continue
        members = [j for j in eachindex(reads) if root(j) == i]
        resp = resps[first(members)]
        n = response_rows[first(members)]
        total += n
        for j in members
            response_rows[j] == n || _fail(resps[j], "column length " *
                "$(response_rows[j]) ≠ the $n rows of $resp, which an " *
                "observation statement reads beside it (one observation axis)")
        end
        for j in members, c in sort!(collect(reads[j]))
            rows[c] == n || _fail(c, "column length $(rows[c]) ≠ the $n " *
                "rows of $resp, which an observation statement reads " *
                "beside it (one observation axis)")
            axisrows[c] = n
        end
    end
    for c in sort!(collect(perobs))
        haskey(axisrows, c) || _fail(c, "column $c has $(rows[c]) rows, " *
            "but the responses observe $lens rows (several observation " *
            "axes) and no observation statement reads $c, so it has no axis")
    end
    return (; rows = axisrows, total)
end

function _validate_response_range_expr(r::LikelihoodSpec)
    ex = r.range
    ex isa Expr || return nothing
    Meta.isexpr(ex, :ref) && length(ex.args) >= 2 && ex.args[1] === r.response ||
        _fail(r.label, "response range must retain its indexed response expression")
    index = ex.args[2]
    valid = index === :(:) ||
        (Meta.isexpr(index, :call, 3) && index.args[1] === :(:) &&
            index.args[2] === 1 && index.args[3] isa Int && index.args[3] >= 1) ||
        (Meta.isexpr(index, :call, 2) && index.args[1] === :eachindex && index.args[2] isa Symbol) ||
        (Meta.isexpr(index, :call, 3) && index.args[1] === :axes && index.args[2] isa Symbol &&
            index.args[3] isa Integer && !(index.args[3] isa Bool) && index.args[3] >= 1)
    valid || _fail(r.label, "response range uses eachindex(v), axes(v, d), or :")
    all(i -> i isa Integer && !(i isa Bool) && i >= 1, ex.args[3:end]) ||
        _fail(r.label, "response trailing indices must be positive literal integers")
    r.mi_jobs === nothing || _fail(r.label, "dynamic index ranges beside packed mi rows are not built yet")
    return nothing
end

_response_range_source(r::LikelihoodSpec) = r.range.args[2] === :(:) ?
    r.response : r.range.args[2].args[1] === :(:) ? r.response : r.range.args[2].args[2]

function _response_range_indices(plan::StructuralPlan, r::LikelihoodSpec)
    _validate_response_range_expr(r)
    source = _response_range_source(r)
    haskey(plan.columns, source) || _fail(r.label, "response index source $source is not bound")
    value = plan.columns[source]
    value isa AbstractArray || _fail(r.label, "response index source $source must be an array")
    index = r.range.args[2]
    index isa Expr && index.args[1] === :(:) && return 1:index.args[3]
    return index === :(:) || index.args[1] === :eachindex ? eachindex(value) :
        axes(value, index.args[3])
end

function _selected_response_column(plan::StructuralPlan, r::LikelihoodSpec)
    indices = _response_range_indices(plan, r)
    value = _observation_column(plan.columns, r.response, r.label, "response")
    # An empty authored loop performs no indexed read, including a
    # trailing dimension that contains no element.
    isempty(indices) && r.response in plan.indexed_observations && return eltype(value)[]
    trailing = r.range.args[3:end]
    checkbounds(Bool, value, indices, trailing...) ||
        _fail(r.label, "response slice $(repr(r.range)) indexes outside $(size(value))")
    return getindex(value, indices, trailing...)
end

function _indexed_operand_axes(plan::StructuralPlan, r::LikelihoodSpec, col, design)
    indices = _response_range_indices(plan, r)
    value = plan.columns[col]
    isempty(indices) && return (Base.OneTo(0),)
    valid = design ? checkbounds(Bool, value, indices, Colon()) : checkbounds(Bool, value, indices)
    valid || _fail(r.label, "indexed operand $col does not cover the authored response range")
    return (Base.OneTo(length(indices)),)
end

function _response_slot_column(plan, r, name, what)
    value = _observation_column(plan.columns, name, r.label, what)
    r.range isa Expr || return value
    name === r.response && return _selected_response_column(plan, r)
    r.response in plan.indexed_observations || return value
    indices = _response_range_indices(plan, r)
    checkbounds(Bool, value, indices) || _fail(r.label,
        "$what $name does not cover the authored response range")
    value[indices]
end

"""Broadcast domains of bound observation statements: `(; rows, total,
domains)`, where `domains` maps response labels to Julia broadcast axes.
Singleton operands do not join independent domains. Structured constructors
retain their existing row-domain validation.
Several domains beside slots outside `_MULTI_AXIS_SLOTS` are not built yet."""
function _observation_axes(plan::StructuralPlan)
    _uses_structured_observation_axes(plan) && return _structured_observation_axes(plan)
    isempty(plan.kernel_plates) || return nothing
    modelvals, managed = _axis_exempt_columns(plan)
    perobs = Set{Symbol}(k for (k, v) in plan.columns
        if k ∉ modelvals && k ∉ managed &&
            v isa AbstractArray)
    resps = [r.response for r in plan.responses]
    all(in(perobs), resps) || return nothing
    isempty(resps) && return nothing
    reads = [_response_reads(plan, r, perobs) for r in plan.responses]
    designs = _observation_design_columns(plan)
    colaxes = Dict(c => c in designs ? (axes(plan.columns[c], 1),) :
        axes(plan.columns[c]) for c in perobs)
    domains = Dict{Symbol,Tuple}()
    for (r, rd) in zip(plan.responses, reads)
        responseaxes = r.range isa Expr ? axes(_selected_response_column(plan, r)) : colaxes[r.response]
        operandaxes(c) = r.range isa Expr && r.response in plan.indexed_observations ?
            _indexed_operand_axes(plan, r, c, c in designs) : colaxes[c]
        valuesread = r.range isa Expr ? _response_reads(plan, _with(r; range=nothing), perobs) : rd
        if r.response in plan.indexed_observations
            domain = responseaxes
            n = prod(length, domain; init = 1)
            if n == 0
                # An empty authored loop never indexes its other operands.
                domains[r.label] = domain
                continue
            end
            for c in rd
                c === r.response && continue
                operandaxes(c) == domain && continue
                m = prod(length, operandaxes(c); init = 1)
                m < n && _fail(c, "indexed column length $m ≠ the $n " *
                    "rows of $(r.response); explicit `@plate` indexing " *
                    "does not stretch singleton operands")
                _fail(c, "explicit `@plate` indexing with different " *
                    "operand axes is not built yet: $c has $(colaxes[c]), " *
                    "$(r.response) has $domain")
            end
        end
        domain = responseaxes
        for c in sort!(collect(setdiff(rd, (r.response,))))
            c in valuesread || continue # a range source contributes indices, not broadcast values
            domain = try
                Base.Broadcast.broadcast_shape(domain, operandaxes(c))
            catch e
                e isa DimensionMismatch || rethrow()
                if length(colaxes[c]) == length(domain) == 1
                    _fail(c, "column length $(length(plan.columns[c])) ≠ " *
                        "the $(length(only(domain))) rows of $(r.response), " *
                        "which an observation statement reads beside it " *
                        "(one observation axis): " * sprint(showerror, e))
                end
                _fail(c, "column shape $(size(plan.columns[c])) is not " *
                    "broadcast-compatible with response $(r.response): " *
                    sprint(showerror, e))
            end
        end
        domains[r.label] = domain
    end
    several = length(unique(values(domains))) > 1
    lens = join(sort!(unique(prod(length, a; init = 1) for a in values(domains))), ", ")
    if several
        for f in fieldnames(StructuralPlan)
            f in _MULTI_AXIS_SLOTS && continue
            isempty(getfield(plan, f)) || _fail(:plan, "the responses " *
                "observe $lens rows (several observation axes), and `$f` " *
                "reads or sizes by a single observation axis — `$f` beside " *
                "several observation axes is not built yet")
        end
        for r in plan.responses
            r.mi_jobs === nothing || _fail(r.label, "mi() missingness on " *
                "$(r.response) beside several observation axes is not built yet")
        end
    elseif any(r -> r.mi_jobs !== nothing, plan.responses)
        return nothing
    end
    # Shared singleton operands impose no domain. Shared non-singleton
    # operands join equal domains; partially shared dimensions can also
    # broadcast over distinct response domains.
    link = collect(eachindex(reads))
    root(i) = link[i] == i ? i : (link[i] = root(link[i]))
    owner = Dict{Tuple{Symbol,Tuple},Int}()
    for (i, rd) in enumerate(reads), c in rd
        all(a -> length(a) == 1, colaxes[c]) && continue
        key = (c, domains[plan.responses[i].label])
        link[root(i)] = root(get!(owner, key, i))
    end
    axisrows = Dict{Symbol,Int}()
    total = 0
    for i in eachindex(reads)
        root(i) == i || continue
        members = [j for j in eachindex(reads) if root(j) == i]
        n = prod(length, domains[plan.responses[i].label]; init = 1)
        total += n
        for j in members, c in sort!(collect(reads[j]))
            axisrows[c] = prod(length, colaxes[c]; init = 1)
        end
    end
    for c in sort!(collect(perobs))
        haskey(axisrows, c) && continue
        several && _fail(c, "column $c has $(size(plan.columns[c])) shape, " *
            "but no observation statement reads it, so it has no axis")
    end
    several || (total = prod(length, first(values(domains)); init = 1))
    return (; rows = axisrows, total, domains)
end

# A matrix used in a matrix-vector product contributes its row axis.
# Its columns remain the contracted dimension, rather than observations.
function _observation_design_columns(plan::StructuralPlan)
    out = Set{Symbol}()
    function visit(ex)
        ex isa Expr || return
        if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :* &&
                ex.args[2] isa Symbol &&
                get(plan.columns, ex.args[2], nothing) isa AbstractMatrix
            _observation_array_operand(plan, ex.args[3]) && push!(out, ex.args[2])
        end
        foreach(visit, ex.args)
    end
    for d in (plan.assignments..., plan.derived...)
        visit(d.expr)
    end
    for p in plan.predictors, t in p.terms
        t.kind === ComposedTerm && visit(t.options.tree)
    end
    return out
end

# An opaque helper can obscure sizes without making a broadcast with a
# known array scalar-valued. Follow definitions and elementwise operands;
# an undotted opaque call alone proves no array shape.
function _observation_array_operand(plan, ex, seen = Set{Symbol}())
    a = _value_axes(plan, ex; data_axes = true)
    a === nothing || return !isempty(a)
    if ex isa Symbol
        ex in seen && return false
        push!(seen, ex)
        defs = (plan.assignments..., plan.derived...)
        i = findfirst(d -> d.name === ex, defs)
        result = i !== nothing && _observation_array_operand(plan, defs[i].expr, seen)
        delete!(seen, ex)
        return result
    end
    ex isa Expr || return false
    if _is_dotted_call(ex)
        return any(x -> _observation_array_operand(plan, x, seen), ex.args[2].args)
    elseif ex.head === :call && ex.args[1] in ELEMENTWISE_OPS
        return any(x -> _observation_array_operand(plan, x, seen), ex.args[2:end])
    end
    return false
end

"""Observation count of a response's full domain. Ordinary elementwise
responses use broadcast cells; structured responses retain their row contract."""
function _response_rows(plan::StructuralPlan, r::LikelihoodSpec)
    if !_uses_structured_observation_axes(plan) && haskey(plan.columns, r.response)
        obs = _observation_axes(plan)
        obs === nothing || return prod(length, obs.domains[r.label]; init = 1)
    end
    return _structured_response_rows(plan, r)
end

"""Rows of a response's full observation axis; mi() packs only its observed
entries, so its location's data establishes the full axis."""
function _structured_response_rows(plan::StructuralPlan, r::LikelihoodSpec)
    r.range isa Expr && return length(_selected_response_column(plan, r))
    r.mi_jobs === nothing && haskey(plan.columns, r.response) &&
        return _column_nrows(plan.columns[r.response])
    modelvals, _ = _axis_exempt_columns(plan)
    managed = _mi_managed_columns(plan)
    perobs = Set{Symbol}(k for (k, v) in plan.columns
        if k ∉ modelvals && k ∉ managed && v isa Union{AbstractVector,AbstractMatrix})
    reads = _response_reads(plan, r, perobs)
    ns = unique!([_column_nrows(plan.columns[c]) for c in reads])
    length(ns) == 1 && return only(ns)
    isempty(ns) && return plan.n_obs
    _fail(r.label, "mi() location reads columns with different rows $ns")
end

"""Rows of an observation-shaped value, resolved from its bound inputs.
Values with no data anchor (an intercept or dar path) use their response
consumers. No dimension is inferred from the total of unrelated axes."""
function _value_rows(plan::StructuralPlan, name::Symbol)
    haskey(plan.columns, name) && return _column_nrows(plan.columns[name])
    modelvals, _ = _axis_exempt_columns(plan)
    managed = _mi_managed_columns(plan)
    perobs = Set{Symbol}(k for (k, v) in plan.columns
        if v isa Union{AbstractVector,AbstractMatrix} &&
            k ∉ modelvals && k ∉ managed)
    reads = _response_reads(plan, name, perobs)
    ns = unique!([_column_nrows(plan.columns[c]) for c in reads])
    if isempty(ns)
        names = Set{Symbol}([name])
        ns = unique!([_response_rows(plan, r) for r in plan.responses
            if name in _response_reads(plan, r, names)])
    end
    length(ns) == 1 && return only(ns)
    if isempty(ns)
        axes = unique!(vcat([_response_rows(plan, r) for r in plan.responses],
            [_kernel_plate_nlanes(kp, plan.columns) for kp in plan.kernel_plates]))
        length(axes) == 1 && return only(axes)
        isempty(axes) && return plan.n_obs
        _fail(name, "value $name has no data range or response to establish its rows")
    end
    _fail(name, "value $name reads or feeds different row counts $ns")
end

function _plate_rows(plan::StructuralPlan, p::PlateParameter)
    p.range isa UnitRange && return length(p.range)
    if p.range isa Expr
        iterator = p.range
        source = iterator.args[2]
        if haskey(plan.columns, source)
            value = plan.columns[source]
            value isa AbstractArray || _fail(p.label, "plate index source $source must be an array")
            return iterator.args[1] === :eachindex ? length(eachindex(value)) :
                length(axes(value, iterator.args[3]))
        end
        iterator.args[1] === :eachindex && return _value_rows(plan, source)
        iterator.args[3] === 1 && return _value_rows(plan, source)
        _fail(p.label, "plate axis $(repr(iterator)) needs its bound source array")
    end
    return _value_rows(plan, p.range isa Symbol ? p.range : p.name)
end

# Missing is a valid label, but remains invalid numeric evidence. Only
# exempt a column when every read is a declared level axis or its gather.
# A second, numeric read of that same column removes the exemption.
function _label_only_columns(plan::StructuralPlan)
    labels, numeric = Set{Symbol}(), Set{Symbol}()
    raw = Set{Symbol}(keys(plan.columns))
    function visit(x)
        if x isa Symbol
            x in raw && push!(numeric, x)
        elseif x isa ArrayParameter
            for f in fieldnames(ArrayParameter)
                if f === :dims
                    for d in x.dims
                        _is_levels_dim(d) ? push!(labels, d.args[2]) : visit(d)
                    end
                else
                    visit(getfield(x, f))
                end
            end
        elseif x isa LevelMap
            push!(labels, x.column)
        elseif x isa TermSpec && x.kind === FactorTerm
            union!(labels, x.columns)
            visit(x.options)
        elseif x isa Expr
            if x.head === :call && !isempty(x.args)
                fn = x.args[1]
                if length(x.args) == 2 && x.args[2] isa Symbol &&
                        (fn in (:levels, :unique, :_ppl_axis_values) ||
                         fn isa GlobalRef && isdefined(fn.mod, fn.name) &&
                            getfield(fn.mod, fn.name) in (Base.unique, DataAPI.levels))
                    push!(labels, x.args[2])
                    return nothing
                elseif fn in (:_ppl_codes, :_ppl_axis_codes)
                    push!(labels, x.args[2])
                    fn === :_ppl_codes && push!(labels, x.args[3])
                    return nothing
                end
            elseif _is_gather_ref(plan, x)
                d = _gather_axis(plan, x)
                if _is_levels_dim(d)
                    axis = _gather_index_axis(plan, x)
                    push!(labels, x.args[axis + 1])
                    for (i, a) in enumerate(x.args)
                        i == axis + 1 || visit(a)
                    end
                    return nothing
                end
            end
            foreach(visit, x.args)
        elseif x isa Union{AbstractArray,Tuple,NamedTuple,AbstractSet}
            foreach(visit, x)
        elseif isstructtype(typeof(x)) && parentmodule(typeof(x)) === @__MODULE__
            for f in fieldnames(typeof(x))
                isdefined(x, f) && visit(getfield(x, f))
            end
        end
        return nothing
    end
    for f in fieldnames(StructuralPlan)
        f in (:columns, :roles, :n_obs, :submodel_scopes) && continue
        visit(getfield(plan, f))
    end
    return setdiff!(labels, numeric)
end

function _validate_columns(plan::StructuralPlan)
    plan.n_obs >= 0 || _fail(:plan, "n_obs must be nonnegative, got $(plan.n_obs)")
    modelvals, managed = _axis_exempt_columns(plan)
    labelcols = _label_only_columns(plan)
    axes = _observation_axes(plan)
    axes === nothing || !hasproperty(axes, :domains) || axes.total == plan.n_obs || _fail(:plan,
        "n_obs $(plan.n_obs) disagrees with the $(axes.total) broadcast observations")
    for (name, col) in plan.columns
        name in modelvals && continue
        _uses_structured_observation_axes(plan) && col isa AbstractArray &&
            ndims(col) > 2 && _fail(name, "structured per-observation data " *
                "$name must retain its vector or matrix row contract")
        # Kernel-managed columns (slices + scalar expansions) carry two
        # lengths by design — they validate under kernel rules, not here.
        # On several observation axes a column has its axis's rows.
        n = axes === nothing ? plan.n_obs : get(axes.rows, name, plan.n_obs)
        if axes !== nothing && hasproperty(axes, :domains) && haskey(axes.rows, name)
            # The response's broadcast_shape check validated every axis.
        elseif col isa AbstractMatrix
            size(col, 1) == n ||
                _fail(name, "matrix column has $(size(col, 1)) rows ≠ " *
                      "n_obs $n")
            size(col, 2) >= 1 ||
                _fail(name, "design matrix has 0 columns " *
                      "(bind ≥ 1 predictor column)")
            eltype(col) <: Real ||
                _fail(name, "matrix column must be numeric, " *
                      "got $(eltype(col))")
        else
            name in managed || length(col) == n || length(col) == 1 ||
                _fail(name, "column length $(length(col)) ≠ n_obs $n")
        end
        name in labelcols || !any(ismissing, col) ||
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
        _, sdname = _spline_block_roles(sb.id, sb.kind, sb.k)
        for (vname, vfamily, vargs, vsupport, vwidth) in wantvec
            v = byname[vname]
            # The smoothing-sd vector may carry a stated hyper prior
            # (SB `sd(mu, s(x)) ~ ...`) — same width, family/args from
            # the admitted set, support derived from the family.
            if vname === sdname && !(v.family === vfamily && v.args == vargs && v.support_override == vsupport)
                hp = HyperPrior(v.family, v.args, v.support_override)
                _validate_hyper_prior(hp, :plan, "spline :$(sb.id) sd")
                _, vfamily, vargs, vsupport, _ = only(s for s in
                    _spline_vector_specs(sb.id, sb.kind, sb.k, hp)
                    if first(s) === sdname)
            end
            (v.family === vfamily && v.args == vargs &&
             v.support_override == vsupport && v.width == vwidth) ||
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
        hb.rho_prior isa HyperPrior && _validate_hyper_prior(hb.rho_prior,
            :plan, "hsgp :$(hb.id) length-scale")
        hb.sigma_prior isa HyperPrior && _validate_hyper_prior(
            hb.sigma_prior, :plan, "hsgp :$(hb.id) sd")
        for (spec, what) in ((hb.rho_prior, "length-scale"),
                             (hb.sigma_prior, "sd"))
            spec isa HSGPHyperLP || continue
            hb.by === nothing && _fail(:plan, "hsgp :$(hb.id): a $what " *
                "hyper-predictor needs a grouped basis (`by = ...`)")
            spec.group === hb.by.column || _fail(:plan, "hsgp :$(hb.id): " *
                "the $what hyper-predictor groups by $(spec.group) but " *
                "the basis groups by $(hb.by.column) — one hyper level " *
                "per term group")
        end
        if hb.by !== nothing
            (hb.cov === :exp_quad && hb.iso && length(hb.axes) == 1) ||
                _fail(:plan, "hsgp :$(hb.id): `by` grouping takes one " *
                    "isotropic exp-quad axis in v1 (aniso / periodic " *
                    "grouped bases are planned)")
            lv = hb.by.levels
            lv === nothing || (!isempty(lv) && allunique(lv)) ||
                _fail(:plan, "hsgp :$(hb.id): `by` levels must be " *
                    "non-empty and distinct, got $(repr(lv))")
        end
        if hb.domain !== nothing
            hb.cov === :periodic && _fail(:plan, "hsgp :$(hb.id): a " *
                "periodic basis has no domain (drop `domain=`)")
            length(hb.domain) == length(hb.axes) || _fail(:plan,
                "hsgp :$(hb.id): domain has $(length(hb.domain)) pairs " *
                "for $(length(hb.axes)) axes (one `(lower, upper)` per axis)")
            all(p -> isfinite(p[1]) && isfinite(p[2]) && p[1] < p[2],
                hb.domain) || _fail(:plan, "hsgp :$(hb.id): domain pairs " *
                "must be finite with lower < upper, got $(hb.domain)")
        end
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
        hb.cov in (:exp_quad, :periodic) ||
            _fail(:plan, "hsgp :$(hb.id): cov must be :exp_quad or " *
                  ":periodic, got $(repr(hb.cov))")
        if hb.cov === :periodic
            # SB "periodic hsgp requires one isotropic axis" + the
            # `_brm_gp_period` contract (required iff periodic); `c` is
            # validated above but ignored (the SB mirror — no domain).
            d == 1 ||
                _fail(:plan, "hsgp :$(hb.id): periodic takes exactly " *
                      "one axis column, got $d")
            hb.iso ||
                _fail(:plan, "hsgp :$(hb.id): periodic requires " *
                      "iso=true (one isotropic axis)")
            hb.period isa Real && isfinite(hb.period) && hb.period > 0 ||
                _fail(:plan, "hsgp :$(hb.id): periodic requires a " *
                      "finite positive period, got $(repr(hb.period))")
            isempty(hb.fits) ||
                _fail(:plan, "hsgp :$(hb.id): periodic carries no " *
                      "fits (no domain to fit)")
        else
            isnan(hb.period) ||
                _fail(:plan, "hsgp :$(hb.id): exp_quad carries no " *
                      "period (got $(repr(hb.period)) — period is " *
                      "meaningful only with cov=:periodic)")
        end
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
            if hb.cov === :periodic
                # SB `_brm_gp_axes`: finite values (no degeneracy gate —
                # a constant axis is a usable periodic domain).
                all(isfinite, axiscol) ||
                    _fail(hb.label, "hsgp :$(hb.id): axis column $c " *
                          "must be finite")
            end
        end
        if hb.cov === :periodic
            isempty(hb.fits) ||
                _fail(hb.label, "hsgp :$(hb.id): periodic carries no " *
                      "fits (no domain to fit)")
        else
            length(hb.fits) == length(hb.axes) ||
                _fail(hb.label, "hsgp :$(hb.id): fits not filled at bind " *
                      "(one (mu, L) per axis)")
        end
    end
    return nothing
end

# Compatibility representation only: old hand-built event-LP plans would
# otherwise mint parameters without authored statements (decision 10ldrvz).
function _validate_event_lps(plan::StructuralPlan)
    isempty(plan.event_lps) || _fail(:plan,
        "implicit event-LP parameters are retired; use the linear_pk_log_f library submodel")
    return nothing
end

# 7-arg event-LP cell calls a grouped cell makes (schedule, arg2):
# lenient collection — the cell walker owns precise rejection. Eight
# expr args (fn + schedule + log_F + 5 LPs) is the literal event form
# (NOT arity-relative: the AUC sibling shares the shape under its own
# arity, and a relative rule would miss it).
function _event_lp_calls(kp::KernelPlate)
    calls = Tuple{Symbol,Symbol}[]
    for (_, ex) in kp.assignments
        _collect_event_lp_calls!(calls, ex)
    end
    return calls
end

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

# Kernel (KernelPlate) structure: everything provable without data.
# v2: N kernel plates per model (panel plates compose freely; at most
# one grouped plate — multi-schedule grouped models are a sequenced
# follow-up). Top-level responses retain their own axes beside the
# kernel-managed likelihood lanes. Panel plates (no schedules) follow
# `_validate_panel_kernel` (grouping ABSENT — implicit 1:n subjects,
# structural, no sentinel, pinned here + tests); grouped plates follow
# `_validate_grouped_kernel`.
function _validate_kernels(plan::StructuralPlan)
    plates = plan.kernel_plates
    isempty(plates) && return nothing
    # A predictor shared by an observation and a subject-level kernel
    # argument still has to satisfy the predictor-level contract.
    for kp in plates, (p, _) in kp.lp_args
        _predictor_level(plan, p)
    end
    ngrouped = count(_is_grouped_kernel, plates)
    ngrouped <= 1 ||
        _fail(:plan, "v2 admits at most one grouped kernel plate per " *
              "model (got $ngrouped — sequenced follow-up)")
    # Name hygiene + collisions (result, slice params, cell locals) live in
    # the global `_validate_name_tables` gate via `_kernel_all_names`.
    for kp in plates
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
    end
    return nothing
end

# Panel-kernel structure (see `_validate_kernels`).
function _validate_panel_kernel(plan::StructuralPlan, kp::KernelPlate)
    kp.subjects === nothing &&
        _fail(kp.label, "panel plates take a subject count (an integer " *
              "or a dims-key name); only schedule-fed kernels derive it " *
              "from data")
    length(kp.obs) <= 1 ||
        _fail(kp.label, "panel admits at most one in-cell observation " *
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
    for obs in kp.obs
        _validate_kernel_obs_ref(kp, obs, params, known, false)
    end
    kp.collected in union(Set{Symbol}(params), cell_locals) ||
        _fail(kp.label, "collected result `$(kp.collected)` is not a cell " *
              "name (slice param or cell-local assignment)")
    return nothing
end

# One in-cell observation node (panel and grouped share the shape):
# response a slice param; panel the scalar response-space set, grouped
# the joint families too; location/scale/params names-or-literals
# resolving to cell/model names (literals finite; constrained-scale
# domains per family below — bare cell args skip link inversion, so
# the value itself must be valid. Positional-second args of the
# multi-param JOINT families take no positivity).
function _validate_kernel_obs_ref(kp::KernelPlate, obs::KernelObs,
        params::Vector{Symbol}, known::Set{Symbol}, grouped::Bool)
    obs.response in params ||
        _fail(kp.label, "kernel obs response `$(obs.response)` is not a " *
              "slice param (responses enter the cell as slices)")
    if grouped
        obs.family in (_KERNEL_SCALAR_FAMS..., CensoredAddpropnormalFam,
                TgiCategoryFam, TgiResponseFam, TgiCensoredFam) ||
            _fail(kp.label, "grouped kernels admit in-cell observations " *
                  "`Normal.(...)`, `Bernoulli.(...)`, `Poisson.(...)`, " *
                  "`NegativeBinomial2.(...)`, `Gamma.(...)`, `Beta.(...)`, " *
                  "`StudentT.(...)`, `CensoredAddpropnormal.(...)`, " *
                  "`TgiCategory.(...)`, `TgiResponse.(...)`, " *
                  "`TgiCensored.(...)` only, got $(obs.family)")
    else
        obs.family in _KERNEL_SCALAR_FAMS ||
            _fail(kp.label, "panel kernels admit scalar in-cell " *
                  "observations `Normal.(...)`, `Bernoulli.(...)`, " *
                  "`Poisson.(...)`, `NegativeBinomial2.(...)`, " *
                  "`Gamma.(...)`, `Beta.(...)`, `StudentT.(...)` only, " *
                  "got $(obs.family)")
    end
    obs.family in _KERNEL_SCALAR_FAMS &&
        return _validate_kernel_scalar_obs(kp, obs, known, grouped)
    # Joint families below (Gaussian rides the scalar path above).
    for (nm, ref) in ((:location, obs.location), (:scale, obs.scale))
        if ref isa Number && !(ref isa Bool)
            # Joint families take no positivity (Gaussian rides the
            # scalar path above).
            isfinite(ref) ||
                _fail(kp.label, "kernel obs $nm literal must be finite, got $ref")
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

# One scalar response-space in-cell observation (v2): shape rules per
# family (1-arg Bernoulli/Poisson leave `scale === nothing`; StudentT
# carries sigma in `params`), then per-slot validation. Symbol refs
# resolve to cell/model names (LP cell params gather explicitly, the
# grouped precedent); literals prove constrained-scale domains (the
# mixture-literal precedent — bare cell args skip link inversion, so
# the value itself must be valid).
function _validate_kernel_scalar_obs(kp::KernelPlate, obs::KernelObs,
        known::Set{Symbol}, grouped::Bool)
    fam = obs.family
    link = _kernel_obs_link(obs)
    (link === IdentityLink ||
        (fam === BernoulliLogitFam && link === LogitLink) ||
        (fam === PoissonLogFam && link === LogLink)) ||
        _fail(kp.label, "in-cell $fam does not take link $link")
    one_arg = fam === BernoulliLogitFam || fam === PoissonLogFam
    if one_arg
        obs.scale === nothing ||
            _fail(kp.label, "$fam in-cell observations take their " *
                  "location only (got a scale slot)")
    else
        obs.scale === nothing &&
            _fail(kp.label, "$fam in-cell observations take location + " *
                  "scale (got location only)")
    end
    want_params = fam === StudentTFam ? 1 : 0
    length(obs.params) == want_params ||
        _fail(kp.label, "$fam in-cell observations take " *
              (want_params == 0 ? "no `params`" :
               "exactly one `params` entry (sigma)") *
              " (got $(obs.params))")
    _validate_kernel_obs_arg(kp, obs, :location, obs.location, known, grouped)
    obs.scale === nothing ||
        _validate_kernel_obs_arg(kp, obs, :scale, obs.scale, known, grouped)
    for ref in obs.params
        _validate_kernel_obs_arg(kp, obs, :params, ref, known, grouped)
    end
    return nothing
end

function _validate_kernel_obs_arg(kp::KernelPlate, obs::KernelObs,
        slot::Symbol, ref, known::Set{Symbol}, grouped::Bool)
    if ref isa Number && !(ref isa Bool)
        _kernel_obs_literal_domain(kp, obs, slot, ref)
        return nothing
    elseif ref isa Symbol
        ref in known ||
            _fail(kp.label, "kernel obs $slot `$ref` is neither a cell " *
                  "name nor a model-level scalar (cross-cell refs " *
                  "fail closed)")
        grouped && ref in _lp_cell_params(kp) &&
            _fail(kp.label, "kernel obs $slot `$ref` is an LP cell " *
                  "param — gather explicitly (`$ref[subj_map]` " *
                  "with a bound subject column; bare LP cell " *
                  "params do not lower as obs args)")
        return nothing
    else
        _fail(kp.label, "kernel obs $slot must be a cell/model name or " *
              "a numeric literal, got $(repr(ref))")
    end
end

# Constrained-scale domain of one scalar-obs literal arg, per family
# (slot roles: Bernoulli location = p; Poisson location = mu; NB2 =
# (mu, phi); Gamma = Distributions (shape, scale); Beta = (a, b)
# shapes; StudentT = (nu, mu, sigma) in (location, scale, params)).
function _kernel_obs_literal_domain(kp::KernelPlate, obs::KernelObs,
        slot::Symbol, v::Real)
    fam = obs.family
    if fam === GaussianFam || fam === CauchyFam
        # v1 message, byte-preserved.
        positive = slot === :scale
        (isfinite(v) && (!positive || v > 0)) ||
            _fail(kp.label, "kernel obs $slot literal must be finite" *
                  (positive ? " positive" : "") * ", got $v")
        return nothing
    end
    isfinite(v) ||
        _fail(kp.label, "kernel obs $slot literal must be finite, got $v")
    slot === :location && _kernel_obs_link(obs) !== IdentityLink && return nothing
    ok = if fam === BinomialProbFam
        slot === :location ? (v isa Integer && v >= 0) : 0 <= v <= 1
    elseif fam === BernoulliLogitFam
        0 <= v <= 1
    elseif fam === PoissonLogFam
        v >= 0
    elseif fam === NegativeBinomial2Fam
        slot === :scale ? v > 0 : v >= 0
    elseif fam === GammaLogFam || fam === BetaLogitFam
        v > 0
    elseif fam === StudentTFam
        slot === :scale ? true : v > 0
    else
        true
    end
    ok && return nothing
    domain, role = if fam === BinomialProbFam
        slot === :location ? ("a nonnegative integer", "n") : ("a probability in [0, 1]", "p")
    elseif fam === BernoulliLogitFam
        "a probability in [0, 1]", "p"
    elseif fam === PoissonLogFam
        "a nonnegative mean", "mu"
    elseif fam === NegativeBinomial2Fam
        slot === :location ? ("a nonnegative mean", "mu") :
            ("positive", "phi")
    elseif fam === GammaLogFam
        slot === :location ? ("positive", "alpha") : ("positive", "scale")
    elseif fam === BetaLogitFam
        slot === :location ? ("positive", "a") : ("positive", "b")
    elseif fam === StudentTFam
        slot === :location ? ("positive degrees of freedom", "nu") :
            ("positive", "sigma")
    else
        "positive", string(slot)
    end
    _fail(kp.label, "kernel obs $slot literal $v is not $domain " *
          "(response-space $role)")
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
    raw = _sched_raw_columns(sched)
    length(unique(raw)) == length(raw) ||
        _fail(kp.label, "schedule `$(sched.name)` reuses a raw column " *
              "($(raw)) — obs/dose/extra axes need distinct columns)")
    slices = kp.slices
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
    # locals + model-scope values (schedule handles are compile-time and
    # enter only as call first-args / gather roots — never as values).
    known = union(Set{Symbol}(params),
        Set{Symbol}(c for (_, c) in kp.lp_args), _all_names(plan))
    schednames = Set{Symbol}(s.name for s in kp.schedules)
    cell_locals = Set{Symbol}()
    scalar_names = Set{Symbol}(p.name for p in plan.parameters)
    union!(scalar_names,Set{Symbol}(a.name for a in plan.assignments))
    for (nm, ex) in kp.assignments
        _collect_grouped_cell_refs!(Symbol[], ex, kp, known, schednames)
        for (source,_) in _collect_grouped_gather_uses(ex)
            source isa Expr || continue
            all(a -> a isa Number || a isa Symbol && a in scalar_names,source.args) ||
                _fail(kp.label,"literal gather entries must be numeric literals or model scalar names")
        end
        push!(known, nm)
        push!(cell_locals, nm)
    end
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
            length(ex.args) == 2 || _fail(label, "cell reduction `$fn` takes one value")
            _collect_kernel_cell_refs!(refs, ex.args[2], kp, known)
            return nothing
        end
        if fn isa Symbol && (fn in ASSIGNMENT_FNS || fn === :logistic)
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
    f in _KERNEL_ELEMENTWISE_FNS ||
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
        if fn isa Symbol && fn in _KERNEL_ELEMENTWISE_FNS
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
            length(ex.args) == 2 || _fail(label, "cell reduction `$fn` takes one value")
            arg, _ = _canonicalize_kernel_cell(ex.args[2], kp, shapes)
            return (Expr(:call, fn, arg), :scalar)
        end
        return _fail(label, "call `$fn` is not in the panel-v1 cell vocabulary")
    end
    if head === :.
        length(ex.args) == 2 && ex.args[1] isa Symbol && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple ||
            _fail(label, "field access does not lower in a cell " *
                         "(dotted calls take `f.(...)`)")
        f = ex.args[1]
        (f === :ifelse || f in _KERNEL_ELEMENTWISE_FNS) ||
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
                       " or $(want + 1) with an event vector second" :
                       fn === :linear_pk_read_locs_auc ?
                       " with an event vector second" : "") *
                      ", got $(length(args))")
            s = args[1]
            s isa Symbol && s in schednames ||
                _fail(label, "cell call `$fn` takes a declared schedule " *
                      "first (got $(repr(s)) — admitted schedules: " *
                      "$(sort!(collect(schednames))))")
            rest = args[2:end]
            if (fn === :linear_pk_read_locs && seven) || fn === :linear_pk_read_locs_auc
                # An ordinary declared value supplies the event vector.
                # Its axis follows this position, independent of its name.
                args[2] isa Symbol && args[2] in known ||
                    _fail(label, "cell call `$fn` second argument must " *
                          "name a declared event vector (got $(repr(args[2])))")
                push!(refs, args[2])
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
    if vec isa Expr && vec.head === :vect
        isempty(vec.args) && _fail(label,"literal gather source is empty")
        for a in vec.args
            _collect_grouped_cell_refs!(refs,a,kp,known,schednames)
        end
    else
        vec isa Symbol && vec in known ||
            _fail(label, "gather source must be a cell name or scalar vector literal, got $(repr(vec))")
        push!(refs, vec)
    end
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
    for c in _sched_raw_columns(sched)
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
    for f in vcat(collect(_sched_materialized_fields(sched)),
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
    # First-obs-on-primary (per plate): the first in-cell observation
    # responds on the schedule obs axis; foreign-axis responses ride
    # later observations.
    n_axis = length(plan.columns[sched.obs_subj])
    _kernel_plate_nlanes(kp, plan.columns) == n_axis ||
        _fail(kp.label, "primary response length " *
              "$(_kernel_plate_nlanes(kp, plan.columns)) ≠ schedule obs " *
              "axis $n_axis (the first in-cell observation responds on " *
              "the schedule obs axis; foreign-axis responses ride later " *
              "observations)")
    for obs in kp.obs
        # Scalar obs validate their response column per family (the
        # panel mirror); joint responses prove on shapes below.
        obs.family in _KERNEL_SCALAR_FAMS || continue
        rcol = only(c for (c, p, _) in kp.slices if p === obs.response)
        colv = _vector_column(plan.columns, rcol, kp.label, "obs response")
        _validate_kernel_obs_column(kp, obs, rcol, colv)
        _validate_kernel_bool_twin(kp, obs, rcol, plan)
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
    _validate_kernel_binomial_trials(kp, plan)
    return nothing
end

# Kernel bind checks: resolved dims, total kinds, flat-T-blocked lengths,
# subjects coverage. Runs on bound plans (bind resolves Symbol dims via
# the `dims` map first; hand-bound plans carry Ints directly).
# Likelihood lanes of one resolved plate (panel: flat length; grouped:
# primary-response length): the bind_data `n` summand and the hand-bound
# n_obs check share it. Per-plate bodies run first, so subjects are
# resolved and slice columns bound whenever this is called.
function _kernel_plate_nlanes(kp::KernelPlate, columns::AbstractDict{Symbol})
    _is_grouped_kernel(kp) ||
        return _kernel_flat_length(kp.subjects, kp.timepoints)
    isempty(kp.obs) && return length(columns[only(kp.schedules).obs_subj])
    rcol0 = only(c for (c, p, _) in kp.slices if p === first(kp.obs).response)
    return length(columns[rcol0])
end

function _validate_kernels_data(plan::StructuralPlan)
    isempty(plan.kernel_plates) && return nothing
    for kp in plan.kernel_plates
        if _is_grouped_kernel(kp)
            _validate_grouped_kernel_data(plan, kp)
        else
            _validate_panel_kernel_data(plan, kp)
        end
    end
    # n_obs totals kernel lanes and ordinary observation axes (bind_data
    # sets the sum; a hand-bound plan must carry it).
    lanes =
        [_kernel_plate_nlanes(kp, plan.columns) for kp in plan.kernel_plates]
    axes = _observation_axes(plan)
    total = sum(lanes) + (axes === nothing ? 0 : axes.total)
    plan.n_obs == total ||
        _fail(:plan, "n_obs $(plan.n_obs) ≠ total likelihood lanes $total " *
              "($(join(["$(kp.result)=$n"
                           for (kp, n) in zip(plan.kernel_plates, lanes)], ", ")))")
    return nothing
end

function _validate_panel_kernel_data(plan::StructuralPlan, kp::KernelPlate)
    kp.subjects isa Int ||
        _fail(kp.label, "subjects dims key `$(kp.subjects)` unresolved " *
              "(bind_data with dims first)")
    n_sub = kp.subjects
    T = kp.timepoints
    T isa Symbol &&
        _fail(kp.label, "timepoints dims key `$T` unresolved " *
              "(bind_data with dims first)")
    flat = _kernel_flat_length(n_sub, T)
    if T === nothing
        all(s -> s[3] === :scalar, kp.slices) ||
            _fail(kp.label, "a vector slice needs T (bind the " *
                  "`kernel_T_$(kp.result)` dims key)")
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
            expv == repeat(colv; inner = T) ||
                _fail(kp.label, "expansion `$exp` is not the flat " *
                      "T-block repeat of `$col`")
        end
    end
    for obs in kp.obs
        obs.family in _KERNEL_SCALAR_FAMS || continue
        si = findfirst(s -> s[2] === obs.response, kp.slices)
        kind = kp.slices[si][3]
        rcol = kp.slices[si][1]
        colv = _vector_column(plan.columns, rcol, kp.label, "obs response")
        _validate_kernel_obs_column(kp, obs, rcol, colv)
        flat = (kind === :scalar && T !== nothing) ?
            _kexp_name(kp.result, rcol) : rcol
        _validate_kernel_bool_twin(kp, obs, flat, plan)
    end
    _validate_kernel_binomial_trials(kp, plan)
    return nothing
end

function _validate_kernel_binomial_trials(kp::KernelPlate, plan::StructuralPlan)
    any(o -> o.family === BinomialProbFam, kp.obs) || return nothing
    integers = Set{Symbol}()
    for (column, param, _) in kp.slices
        eltype(plan.columns[column]) <: Integer && push!(integers, param)
    end
    function integer_value(ex)
        ex isa Integer && return true
        ex isa Symbol && return ex in integers
        ex isa Expr || return false
        ex.head === :call && !isempty(ex.args) || return false
        fn = ex.args[1]
        fn === :length && return true
        fn in (:+, :-, :*, :%, :.+, :.-, :.*, :.% , :sum, :minimum, :maximum) &&
            return all(integer_value, ex.args[2:end])
        return false
    end
    for (name, ex) in kp.assignments
        integer_value(ex) && push!(integers, name)
    end
    for obs in kp.obs
        obs.family === BinomialProbFam || continue
        integer_value(obs.location) || _fail(kp.label,
            "Binomial trials must be integer data or an integer-valued cell expression")
        if obs.location isa Symbol
            index = findfirst(s -> s[2] === obs.location, kp.slices)
            index === nothing && continue
            raw = plan.columns[kp.slices[index][1]]
            all(>=(0), raw) || _fail(kp.label, "Binomial trials must be nonnegative")
        end
    end
    return nothing
end

# Bernoulli in-cell Bool twin verification (bind products are verified,
# not trusted — the kexp precedent): non-Bool flats read their twin.
function _validate_kernel_bool_twin(kp::KernelPlate, obs::KernelObs,
        flat::Symbol, plan::StructuralPlan)
    obs.family === BernoulliLogitFam || return nothing
    flatv = _vector_column(plan.columns, flat, kp.label, "obs flat")
    eltype(flatv) === Bool && return nothing
    twin = _kbool_name(kp.result, obs.response)
    haskey(plan.columns, twin) ||
        _fail(kp.label, "Bernoulli in-cell Bool twin `$twin` missing " *
              "(bind_data materializes it for non-Bool responses)")
    twinv = _vector_column(plan.columns, twin, kp.label, "Bool twin")
    eltype(twinv) === Bool ||
        _fail(kp.label, "Bool twin `$twin` must be Bool, got " *
              "$(eltype(twinv))")
    twinv == (flatv .!= 0) ||
        _fail(kp.label, "Bool twin `$twin` is not `(flat .!= 0)`")
    return nothing
end

# One scalar in-cell observation's RESPONSE column, per family (the
# `_validate_response_column` mirror — same domains, kernel-attributed
# messages). Validates the RAW slice column: scalar-slice expansions
# are Float64 by construction (exact for 0/1 + counts); Bernoulli
# non-Bool flats read their bind-materialized Bool twin.
function _validate_kernel_obs_column(kp::KernelPlate, obs::KernelObs,
        col::Symbol, colv::AbstractVector)
    fam = obs.family
    if fam === BernoulliLogitFam
        (eltype(colv) === Bool ||
            (eltype(colv) <: Integer && all(x -> x == 0 || x == 1, colv))) ||
            _fail(kp.label, "Bernoulli in-cell response `$col` must be " *
                  "Bool or 0/1 integers")
    elseif fam === BinomialProbFam
        _is_count_column(colv) ||
            _fail(kp.label, "Binomial in-cell response `$col` must be non-negative integers")
        obs.location isa Number && any(>(obs.location), colv) &&
            _fail(kp.label, "Binomial in-cell response `$col` exceeds its trials")
    elseif fam === PoissonLogFam || fam === NegativeBinomial2Fam
        _is_count_column(colv) ||
            _fail(kp.label, (fam === PoissonLogFam ? "Poisson" : "NB2") *
                  " in-cell response `$col` must be non-negative integers")
    elseif fam === GammaLogFam
        (eltype(colv) <: Real && all(>(0), colv)) ||
            _fail(kp.label, "Gamma in-cell response `$col` must be " *
                  "strictly positive numerics")
    elseif fam === BetaLogitFam
        (eltype(colv) <: Real && all(x -> 0 < x < 1, colv)) ||
            _fail(kp.label, "Beta in-cell response `$col` must be " *
                  "numerics strictly inside (0, 1)")
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
    scanstates = Symbol[st for s in plan.scans for st in s.states]
    darstates = [s.state for s in plan.dar_paths]
    vectors = [p.name for p in plan.vector_parameters]
    svec = [v.name for v in plan.spline_vectors]
    vcorr = Symbol[nm for d in plan.varying_draws
        for nm in _varying_corr_table_names(d)]
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
        (vcorr, params, "correlated varying names and parameters"),
        (vcorr, assigns, "correlated varying names and assignments"),
        (vcorr, deriveds, "correlated varying names and derived columns"),
        (vcorr, plates, "correlated varying names and plate parameters"),
        (vcorr, scanstates, "correlated varying names and scan states"),
        (vcorr, vectors, "correlated varying names and vector parameters"),
        (vcorr, svec, "correlated varying names and spline vectors"),
        (hsgp, params, "hsgp names and parameters"),
        (hsgp, assigns, "hsgp names and assignments"),
        (hsgp, deriveds, "hsgp names and derived columns"),
        (hsgp, plates, "hsgp names and plate parameters"),
        (hsgp, scanstates, "hsgp names and scan states"),
        (hsgp, vectors, "hsgp names and vector parameters"),
        (hsgp, svec, "hsgp names and spline vectors"),
        (hsgp, vcorr, "hsgp names and correlated varying names"),
        (kern, params, "kernel-plate names and parameters"),
        (kern, assigns, "kernel-plate names and assignments"),
        (kern, deriveds, "kernel-plate names and derived columns"),
        (kern, plates, "kernel-plate names and plate parameters"),
        (kern, scanstates, "kernel-plate names and scan states"),
        (kern, vectors, "kernel-plate names and vector parameters"),
        (kern, svec, "kernel-plate names and spline vectors"),
        (kern, vcorr, "kernel-plate names and correlated varying names"),
        (kern, hsgp, "kernel-plate names and hsgp names"),
        (mats, params, "design-matrix names and parameters"),
        (mats, assigns, "design-matrix names and assignments"),
        (mats, deriveds, "design-matrix names and derived columns"),
        (mats, plates, "design-matrix names and plate parameters"),
        (mats, scanstates, "design-matrix names and scan states"),
        (mats, vectors, "design-matrix names and vector parameters"),
        (mats, svec, "design-matrix names and spline vectors"),
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
        vectors, svec, vcorr, hsgp, kern, mats)
    arrays = _array_names(plan)
    length(unique(arrays)) == length(arrays) ||
        _fail(:plan, "duplicate array-parameter names")
    overlap = intersect(arrays, union(allnames, elps))
    isempty(overlap) || _fail(:plan, "names in both array parameters and " *
        "other parameters/assignments/derived/plate/scan/dar/vector/spline/" *
        "varying/hsgp/kernel/matrix/event-LP names: $(join(overlap, ", "))")
    allnames = union(allnames, arrays)
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
    for n in Iterators.flatten((pnames, params, assigns, deriveds, plates, scanstates, darstates, vectors, svec, vcorr, hsgp, kern, mats))
        _check_name_hygiene(n)
    end
    return nothing
end

# A scan is non-centered when a step writes a carried array
# deterministically (`state[loopvar] = ...`): the layout slice then holds the
# sampled seeds and the per-step innovations, and the emitter reconstructs
# every carried array via the RK-core `scan(...)` carry-fold. Otherwise
# (every carried write samples) the slice holds the state itself (centered
# form). Mixed statements and tuple carries use the ordered reconstruction.
_is_noncentered_scan(s::ScanSpec) =
    !(length(s.states) == 1 && all(f -> f.kind === :sample, s.setup) &&
        length(s.step) == 1 && only(s.step).kind === :sample && only(s.step).indexed)

# A scan's trajectory length: a literal, a supplied integer value, or the
# consuming response's rows for the historical unsupplied length-name form.
_scan_length(plan::StructuralPlan, s::ScanSpec) =
    s.hi isa Int ? s.hi : haskey(plan.columns, s.hi) ?
        _scan_bound_length(plan.columns[s.hi], s) : _value_rows(plan, first(s.states))

function _scan_bound_length(value, s)
    (value isa Integer && !(value isa Bool) && value >= s.lo - 1) ||
        _fail(s.label, "scan bound $(s.hi) must bind an integer at least $(s.lo - 1), got $(repr(value))")
    return Int(value)
end

# A non-centered scan's latent slice length: one coordinate per sampled seed,
# plus `T - m` per innovation local (one per loop iteration).
_scan_latent_size(s::ScanSpec, T::Int) =
    count(f -> f.kind === :sample, s.setup) +
    count(st -> st.kind === :sample, s.step) * (T - (s.lo - 1))

# In-graph name of a non-centered scan's latent slice
# (`_ppl_scan_z_<first state>`). Reserved-prefix validation guarantees no
# user name collides with it; the state names themselves bind the
# reconstruction.
_scan_innovation_name(s::ScanSpec) = Symbol(:_ppl_scan_z_, first(s.states))

# Structural invariants of each sequential recurrence. The surface parser
# (`parse_scan_block`) already enforces these; this is defense-in-depth for a
# hand-built plan and the invariants the layout/emitter will rely on.
function _validate_scans(plan::StructuralPlan)
    for s in plan.scans
        isempty(s.states) && _fail(s.label, "a scan carries no array")
        length(unique(s.states)) == length(s.states) || _fail(s.label,
            "scan carried arrays $(s.states) repeat a name")
        m = s.lo - 1
        m >= 1 || _fail(s.label,
            "scan loop start $(s.lo) leaves no seed fill (start at 2 or later)")
        for a in s.states
            idx = [f.index for f in s.setup if f.target === a]
            idx == collect(1:m) || _fail(s.label,
                "scan carried array $a must be seeded at 1..$m in order " *
                "(the loop starts at $(s.lo)), got seed indices $idx")
        end
        for f in s.setup
            f.target in s.states || _fail(s.label,
                "scan seed fill targets $(f.target), which is not a carried " *
                "array of the scan $(s.states)")
            f.kind in (:sample, :assign) || _fail(s.label,
                "scan seed fill kind must be :sample or :assign, got " *
                "$(repr(f.kind))")
        end
        s.maxlag >= 0 || _fail(s.label, "scan maxlag must be nonnegative")
        isbound(plan) && _resolve_scan(plan, s)
        m >= s.maxlag || _fail(s.label,
            "scan maxlag $(s.maxlag) exceeds the $m seeded value(s) per array")
        for a in s.states
            n = count(st -> st.indexed && st.target === a, s.step)
            n == 1 || _fail(s.label,
                "scan recurrence must write its carried array $a exactly " *
                "once per step, got $n write(s)")
        end
        for st in s.step
            st.indexed && !(st.target in s.states) && _fail(s.label,
                "scan step writes $(st.target)[$(s.loopvar)], which is not a " *
                "carried array of the scan $(s.states)")
        end
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
# persistence names a `Normal` sampled parameter truncated to exactly
# `[0, 1]` (`beta ~ truncated(Normal(0.5, 0.2), 0, 1)`; other
# location/scale ride the same spelling) and the scale names a half-Normal
# (`sigma ~ HalfNormal(0.2)` or `truncated(Normal(0, 0.2), 0, Inf)`).
# Legacy normalized support overrides and general `:truncated` agree. Both keep
# Distributions semantics — the truncation normalizers stay (user decision
# `0m1j3iz`, prong `dar-kernel`). The `n_obs ≥ 2` length gate lives in the
# layout (unbound surface plans carry `n_obs = 0`, like a scan's symbolic
# `hi` — lengths resolve at bind).
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
        (b.family === :normal && b.support_override in
            ((:interval, 0.0, 1.0), (:truncated, 0.0, 1.0))) ||
            _fail(s.label,
                "dar persistence :$(s.beta) must be a Normal truncated to " *
                "exactly [0, 1] (`truncated(Normal(0.5, 0.2), 0, " *
                "1)`), got :$(b.family) on $(repr(b.support_override))")
        j = findfirst(p -> p.name === s.sigma, plan.parameters)
        j === nothing && _fail(s.label,
            "dar scale :$(s.sigma) must name a scalar sampled parameter " *
            "(`$(s.sigma) ~ HalfNormal(0.2)`)")
        sg = plan.parameters[j]
        (sg.family === :normal && (sg.support_override === :positive ||
            (sg.support_override == (:truncated, 0.0, Inf) &&
                get(sg.args, :arg1, nothing) == 0))) ||
            _fail(s.label,
                "dar scale :$(s.sigma) must be a half-Normal " *
                "(`HalfNormal(0.2)` or `truncated(Normal(0, 0.2), 0, Inf)`), " *
                "got :$(sg.family) on " *
                "$(repr(sg.support_override))")
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
    # Derived responses bind-materialize into the columns (their bound
    # values ARE the response data), so response-named derived columns
    # are exempt from the overlap rule; every other derived name still
    # collides (a caller column the model derives is a shadowing bug).
    resps = Set{Symbol}(r.response for r in plan.responses)
    # Module data definitions bind-materialize under their own names too.
    computed = _bound_module_data_names(plan)
    col_overlap = filter(
        n -> haskey(plan.columns, n) && n ∉ computed,
        union([p.name for p in plan.parameters],
            [a.name for a in plan.assignments],
            [d.name for d in plan.derived if d.name ∉ resps],
            [p.name for p in plan.plate_parameters],
            _array_names(plan)),
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
        # A data-only module value read per observation is checked at
        # bind (its length or row count), not by its expression's shape.
        _is_vector_valued(d.expr, plan) || _is_bind_data_derived(plan, d.name) ||
            _fail(d.label, "derived column is scalar-valued — write it as " *
                "a scalar assignment instead")
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
        # A data-only module value read per observation is checked at
        # bind (its length or row count), not by its expression's shape.
        _is_vector_valued(d.expr, plan) || _is_bind_data_derived(plan, d.name) ||
            _fail(d.label, "derived column is scalar-valued — write it as " *
                "a scalar assignment instead")
    end
    return nothing
end

# A bare-Symbol derived expression aliases a column (raw or derived) —
# never a scalar name (length mismatch).
function _validate_vector_alias(d::VectorAssignmentSpec, plan, bound::Bool)
    d.expr isa Symbol || return nothing
    target = d.expr
    any(s -> target in s.states, plan.scans) && return nothing
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
        [p.name for p in plan.plate_parameters], _vector_value_names(plan),
        _array_names(plan), [a for s in plan.scans for a in s.states])

"""Vector-parameter names (simplexes, cutpoints, …): model-level array values
that definitions may read whole (`cumsum(vcat(0.0, zeta))`), never scalars."""
_vector_value_names(plan::StructuralPlan) =
    [v.name for v in plan.vector_parameters]

# Every name the expression reads is a model-level value (parameter,
# assignment, vector parameter) — no column, derived column or unknown name.
function _model_level_expr(ex, plan::StructuralPlan)
    known = union(Set{Symbol}(_union_names(plan)),
        Set{Symbol}(_vector_value_names(plan)))
    syms = _expr_value_symbols(ex)
    return !isempty(syms) && all(s -> s in known, syms)
end

_is_derived(plan::StructuralPlan, name::Symbol) =
    any(d -> d.name === name, plan.derived)

_is_plate_param(plan::StructuralPlan, name::Symbol) =
    any(p -> p.name === name, plan.plate_parameters)

# With bound=false (structure), bare Symbols are opaque refs and column
# checks are skipped — classification needs columns. With bound=true, bare
# columns fail (row-varying outside a reduction) and reduction args must be
# bound columns or derived names. Derived names are known in both states,
# so derived-outside-a-reduction fails at structure already.
# `(name = value,)` stores a NamedTuple key, not a model assignment/read.
_tuple_field_value(ex) = Meta.isexpr(ex, :(=), 2) && ex.args[1] isa Symbol ?
    ex.args[2] : ex

function _collect_assignment_refs!(refs, ex, plan, label, bound::Bool)
    ex isa Number && return nothing
    ex isa LineNumberNode && return nothing
    _is_plate_column_expr(ex) &&
        return _collect_plate_column_refs!(refs, ex, plan, label, bound)
    _is_matrix_math(ex, plan) &&
        return _collect_opaque_refs!(refs, ex, plan, label, bound)
    # Expressions over declared array parameters (`phi[1]`, `sd .* z`,
    # `sum(z)`) follow the array-value vocabulary (`arrays.jl`).
    _mentions_array(ex, plan) &&
        return _collect_array_value_refs!(refs, ex, plan, label, bound)
    if ex isa Symbol
        _is_derived(plan, ex) && _fail(
            label,
            "$ex is a derived column: reference it inside reductions " *
            "(`mean($ex)`) or elementwise in a derived definition " *
            "(`log.($ex)`-style); scalar positions take scalars",
        )
        # (A bound model-level data value — an assignment evaluated at
        # bind — is a column entry but no observation column.)
        # A raw whole-value input is a bound operand, not a graph node.
        bound && haskey(plan.columns, ex) &&
            ex in first(_bound_model_level_inputs(plan)) && return nothing
        bound && haskey(plan.columns, ex) &&
            !any(a -> a.name === ex, plan.assignments) && _fail(
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
        if fn isa GlobalRef
            # A module function takes whole values (functions as values).
            _collect_opaque_refs!(refs, ex, plan, label, bound)
            return nothing
        end
        if fn isa Symbol && fn in ELEMENTWISE_OPS
            # Broadcasting over model-level operands stays model-level; a
            # bare column operand still fails as row-varying below.
            for arg in ex.args[2:end]
                _collect_assignment_refs!(refs, arg, plan, label, bound)
            end
            return nothing
        end
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
            # A reduction of a module call's whole value is plain Julia
            # (`maximum(hsgp_rho_floors(lambda))`), like the call itself.
            if _is_module_value_call(arg)
                _collect_opaque_refs!(refs, arg, plan, label, bound)
                return nothing
            end
            # A reduction of a model-level value is plain Julia
            # (`sum(zeta)`, `sum(abs2.(w))`); over a column it stays a bare
            # name (nested column transforms stage as their own definition).
            if _model_level_expr(arg, plan)
                _collect_assignment_refs!(refs, arg, plan, label, bound)
                return nothing
            end
            if arg isa Expr
                _collect_vector_refs!(refs, arg, plan, label, bound)
                return nothing
            end
            return _collect_vector_reduction!(refs, ex, plan, label, bound)
        end
        for arg in ex.args[2:end]
            _collect_assignment_refs!(refs, arg, plan, label, bound)
        end
        return nothing
    end
    if head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        # Model-level broadcast (`tanh.(w)`, `f.(zeta, s)`): operands are
        # model-level values; a bare column operand fails as row-varying.
        for arg in ex.args[2].args
            _collect_assignment_refs!(refs, arg, plan, label, bound)
        end
        return nothing
    end
    head === :. && _fail(
        label,
        "broadcast expressions are row-varying — stage them as their own " *
        "derived `name = ...` first",
    )
    if head === :ref
        return _collect_model_value_ref!(refs, ex, plan, label, bound)
    end
    if head === :vect || head === :tuple
        for a in ex.args
            head === :tuple && (a = _tuple_field_value(a))
            _collect_assignment_refs!(refs, a, plan, label, bound)
        end
        return nothing
    end
    return _fail(label, "unsupported expression head $head (pure calls only)")
end

# Indexing a model-level value, or one element of a column, has the same
# meaning inside a scalar assignment and an array expression. Declared
# arrays keep their axis validation; other values keep ordinary Julia reads.
function _collect_model_value_ref!(refs, ex::Expr, plan, label, bound::Bool)
    # Opaque result shapes cannot bypass the existing data-only index check.
    length(ex.args) == 2 && _check_gather_index(ex, plan, label, false)
    _collect_array_ref!(refs, ex, plan, label, bound;
        allow_gather=false) && return nothing
    obj = ex.args[1]
    if !(bound && obj isa Symbol && haskey(plan.columns, obj))
        _collect_assignment_refs!(refs, obj, plan, label, bound)
    end
    for i in ex.args[2:end]
        i === :(:) && continue
        _collect_assignment_refs!(refs, i, plan, label, bound)
    end
    return nothing
end

# Arguments of a module function call (functions as values) are whole
# Julia values: columns, derived columns, parameters, assignments and
# vector parameters alike, in any expression. Collects the plan names read
# (bound columns are inputs, not nodes); keyword names and function values
# are not reads.
function _collect_opaque_refs!(refs, ex, plan, label, bound::Bool)
    ex isa Union{Number,LineNumberNode,GlobalRef,QuoteNode,String} &&
        return nothing
    if ex isa Symbol
        # `:` in a positional read (`Z[:, 1]`) is a whole axis, not a name.
        ex === :(:) && return nothing
        bound && haskey(plan.columns, ex) && return nothing
        push!(refs, ex)
        return nothing
    end
    ex isa Expr || _fail(label, "unsupported literal $(repr(ex)) in a " *
                                "module function call")
    if ex.head === :kw && length(ex.args) == 2
        return _collect_opaque_refs!(refs, ex.args[2], plan, label, bound)
    end
    args = ex.head === :call ? ex.args[2:end] : ex.args
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        args = ex.args[2].args
    end
    for a in args
        ex.head === :tuple && (a = _tuple_field_value(a))
        _collect_opaque_refs!(refs, a, plan, label, bound)
    end
    return nothing
end

# A gather (`cum[c]`) indexes by row-varying integer DATA. A parameter, a
# per-cell latent, a vector parameter, or a definition reading one is a
# value, never an index (`c[x_true]` would index by a real number at run
# time). Bound, a raw index column must hold integers.
function _check_gather_index(ex::Expr, plan::StructuralPlan, label,
        bound::Bool)
    idx = ex.args[2]
    names = Set{Symbol}(_all_names(plan))
    for s in _expr_value_symbols(idx)
        _reads_data_only(plan, s, names, Set{Symbol}()) || _fail(label,
            "gather `$(repr(ex))` indexes by $s, which is not data — a " *
            "gather index is an integer data column (`v[c]`); a " *
            "parameter or latent is a value, never an index")
    end
    if bound && idx isa Symbol && haskey(plan.columns, idx) &&
            !(idx in names)
        col = _vector_column(plan.columns, idx, label, "gather index")
        (eltype(col) <: Integer && eltype(col) !== Bool) || _fail(label,
            "gather index $idx must hold integer positions, got eltype " *
            "$(eltype(col))")
    end
    return nothing
end

# `s` names data alone: a raw column (any name the plan does not define),
# or a definition whose expression reads only such names.
function _reads_data_only(plan::StructuralPlan, s::Symbol, names,
        active::Set{Symbol})
    s in names || return true
    s in active && return false
    i = findfirst(d -> d.name === s, plan.derived)
    j = findfirst(a -> a.name === s, plan.assignments)
    defn = i !== nothing ? plan.derived[i].expr :
        j !== nothing ? plan.assignments[j].expr : nothing
    defn === nothing && return false  # a parameter or latent
    push!(active, s)
    ok = all(t -> _reads_data_only(plan, t, names, active),
        _expr_value_symbols(defn))
    pop!(active, s)
    return ok
end

# Elementwise walker for derived columns (contract v3). Vector mode admits
# dotted operators, dotted math, `ifelse`, reductions over bare names, and
# bare names/literals; scalar subterms (undotted allowlist calls) delegate
# to the scalar collector. With bound=true, direct column references in
# math positions must be numeric and bare-Symbol `ifelse` conditions must be
# Bool columns.
# An array-cell plate column (a derived column's right-hand side
# `plate(lanes..., Ref(shared)...) do args...; cell; end`, one value per
# index — `_plate_column_expr` in the surface): the inputs are graph
# values; the cell body is RK's to plan. Lanes are data columns or the
# level codes `_ppl_codes(g, h)` of column `g` on `levels(h)`.
_is_plate_column_expr(ex) = ex isa Expr && ex.head === :do &&
    length(ex.args) == 2 && ex.args[1] isa Expr && ex.args[1].head === :call &&
    !isempty(ex.args[1].args) && ex.args[1].args[1] === :plate

# Level plate columns are array values with a declared level axis. Keep
# that provenance in the first input until the generator binds the axis.
function _level_plate_axis(ex)
    _is_plate_column_expr(ex) || return nothing
    inp = ex.args[1].args[2]
    return inp isa Expr && inp.head === :call && length(inp.args) == 2 &&
        inp.args[1] === :_ppl_level_indices ? inp.args[2] : nothing
end

function _collect_plate_column_refs!(refs, ex, plan, label, bound::Bool)
    known = union(_union_names(plan), _vector_value_names(plan),
        Set{Symbol}(d.name for d in plan.derived),
        Set{Symbol}(p.name for p in plan.array_parameters),
        Set{Symbol}(p.name for p in plan.plate_parameters))
    for inp in ex.args[1].args[2:end]
        if Meta.isexpr(inp, :ref, 2) && inp.args[1] isa Expr &&
                inp.args[1].head === :call &&
                inp.args[1].args[1] in (:_ppl_codes, :_ppl_axis_codes)
            _collect_opaque_refs!(refs, inp.args[2], plan, label, bound)
            inp = inp.args[1]
        end
        if inp isa Symbol
            bound && !haskey(plan.columns, inp) && !(inp in known) &&
                _fail(label, "array plate column reads unknown name $inp")
            bound && haskey(plan.columns, inp) || push!(refs, inp)
        elseif inp isa Expr && inp.head === :call && inp.args[1] isa GlobalRef
            _collect_opaque_refs!(refs, inp, plan, label, bound)
        elseif inp isa Expr && inp.head === :call && length(inp.args) == 4 &&
                inp.args[1] === :_ppl_axis_codes
            g, name, axis = inp.args[2:end]
            g isa Symbol && name isa Symbol && axis isa Int || _fail(label,
                "array plate axis codes take a column, array name and axis")
            axes = _gather_axes(plan, name)
            axes !== nothing && 1 <= axis <= length(axes) || _fail(label,
                "array plate codes address an unknown axis of $name")
            bound && _validate_gather_axis(plan, name, label, axes[axis], g)
        elseif inp isa Expr && inp.head === :call && length(inp.args) == 4 &&
                inp.args[1] === :_ppl_level_gather
            nm, g, ld = inp.args[2:end]
            axs = _gather_axes(plan, nm)
            axs !== nothing && ld isa Int && 1 <= ld <= length(axs) &&
                _is_levels_dim(axs[ld]) || _fail(label,
                    "array plate column cannot align $(repr(inp))")
            push!(refs, nm)
            # levels(g) includes unused categorical pool members too;
            # validate every cell label rather than only observed g rows.
            bound && _validate_level_gather(plan, nm, label, axs[ld], g,
                _array_axis_levels(plan, nm, label, g))
        elseif inp isa Expr && inp.head === :call &&
                ((length(inp.args) == 3 && inp.args[1] === :_ppl_codes) ||
                 (length(inp.args) == 2 && inp.args[1] in
                    (:_ppl_level_indices, :_ppl_level_values)))
            for c in inp.args[2:end]
                c isa Symbol || _fail(label, "array plate column codes " *
                    "take data columns, got $(repr(c))")
                bound && !haskey(plan.columns, c) && _fail(label,
                    "array plate column codes read unknown column $c")
            end
        elseif inp isa Expr && inp.head === :call && length(inp.args) == 2 &&
                inp.args[1] === :Ref && inp.args[2] isa Symbol
            nm = inp.args[2]
            bound && !haskey(plan.columns, nm) && !(nm in known) &&
                _fail(label, "array plate column reads unknown name $nm")
        elseif Meta.isexpr(inp, :call, 2) && inp.args[1] === :Ref &&
                Meta.isexpr(inp.args[2], :call, 2) &&
                inp.args[2].args[1] === GlobalRef(Base, :vec)
            _collect_opaque_refs!(refs, inp.args[2], plan, label, bound)
        else
            _fail(label, "array plate column input $(repr(inp)) is not a " *
                "column, aligned level values, level codes or `Ref(name)`")
        end
    end
    return nothing
end

function _collect_vector_refs!(refs, ex, plan, label, bound::Bool)
    ex isa Number && return nothing
    ex isa LineNumberNode && return nothing
    if ex isa Symbol
        # Per-cell latent (plate) parameters are vectors, so a derived column
        # may transform one (`theta = mu .+ tau .* z`) — the non-centered shape.
        if _is_derived(plan, ex) || _is_plate_param(plan, ex) ||
                any(s -> ex in s.states, plan.scans) ||
                ex in _union_names(plan) || ex in _vector_value_names(plan) ||
                _is_array_param(plan, ex)
            push!(refs, ex)
            return nothing
        end
        bound || return nothing
        haskey(plan.columns, ex) && return nothing
        return _fail(label, "derived column references unknown name $ex")
    end
    ex isa Expr || _fail(label, "unsupported literal $(repr(ex)) (numeric literals only)")
    head = ex.head
    if _is_plate_column_expr(ex)
        return _collect_plate_column_refs!(refs, ex, plan, label, bound)
    end
    if head === :call
        fn = ex.args[1]
        _is_matrix_math(ex, plan) &&
            return _collect_opaque_refs!(refs, ex, plan, label, bound)
        _is_data_matvec(ex, plan) &&
            return _collect_data_matvec!(refs, ex, plan, label, bound)
        if fn isa GlobalRef
            # An undotted module call inside a column expression is a
            # model-level subterm over whole values.
            _collect_opaque_refs!(refs, ex, plan, label, bound)
            return nothing
        end
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
    _is_row_gather(plan, ex) && _fail(label, "`$(repr(ex))` is one row " *
        "per observation (a matrix) — multiply it by a vector " *
        "(`$(repr(ex)) * v`) or read one column (`$(ex.args[1])[$(ex.args[2]), 1]`)")
    head === :ref && _collect_array_ref!(refs, ex, plan, label, bound;
        allow_gather = true) && return nothing
    if head === :ref && length(ex.args) == 2
        # Gather (`cum[c]`): a whole model-level value indexed by a
        # row-varying integer column keeps n_obs.
        _check_gather_index(ex, plan, label, bound)
        _collect_opaque_refs!(refs, ex.args[1], plan, label, bound)
        _collect_vector_refs!(refs, ex.args[2], plan, label, bound)
        return nothing
    end
    head === :ref && return _fail(label, "indexing changes length — " *
                                          "derived columns keep n_obs " *
                                          "(gathers take one row-varying " *
                                          "index: `v[c]`)")
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
    if arg isa Expr
        _collect_vector_refs!(refs, arg, plan, label, bound)
        return nothing
    end
    arg isa Symbol || _fail(
        label,
        "reduction $fn argument must be a bare column or derived name " *
        "(stage nested transforms as their own `name = ...` first)",
    )
    # Reductions read a graph value whole, including latent plate vectors,
    # declared arrays and scan states. Keep its dependency edge just as for
    # a derived value; its observation axis is irrelevant to the reduction.
    if arg in _all_names(plan)
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
    length(ex.args) == 2 && ex.args[1] isa Union{Symbol,GlobalRef} &&
        ex.args[2] isa Expr && ex.args[2].head === :tuple ||
        return _fail(label, "field access does not lower in vector " *
                            "expressions (dotted calls take `f.(...)`)")
    f = ex.args[1]
    args = ex.args[2].args
    if f isa GlobalRef
        # Module function broadcast (functions as values): elementwise over
        # its operands, any arity.
        for arg in args
            _collect_vector_refs!(refs, arg, plan, label, bound)
        end
        return nothing
    end
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
    length(args) == _elementwise_arity(f) || _fail(label, "`$f.` takes " *
        "exactly $(_elementwise_arity(f) == 2 ? "two arguments" :
            "one argument")")
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
    _is_plate_column_expr(ex) && return true
    ex isa Symbol && return !(ex in _union_names(plan) ||
        ex in _vector_value_names(plan)) && !_is_array_param(plan, ex)
    ex isa Expr && ex.head === :ref && ex.args[1] isa Symbol &&
        (_is_array_param(plan, ex.args[1]) ||
            _is_array_assignment(plan, ex.args[1])) &&
        return _array_index_kind(plan, ex) === :gather &&
            all(i -> i isa Int || _is_row_index(plan, i), ex.args[2:end])
    ex isa Number && return false
    ex isa LineNumberNode && return false
    ex isa Expr || return false
    head = ex.head
    head === :ref && length(ex.args) == 2 &&
        return _literal_row_range(ex.args[2]) || _is_vector_valued(ex.args[2], plan)
    if head === :call
        isempty(ex.args) && return false
        _is_matrix_math(ex, plan) && return true
        fn = ex.args[1]
        fn === GlobalRef(Base, :hcat) && return length(ex.args) > 1
        fn isa GlobalRef && return false  # undotted module call: model-level
        if fn === :* && length(ex.args) == 3 &&
                _observation_matrix_gather(ex.args[2], plan) &&
                _model_vector_value(ex.args[3], plan)
            # A gathered N×K array times a model-level K-vector is an
            # observation vector, although neither operand is a vector
            # column on its own.
            return true
        end
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

# Matrix values retain Julia's whole-value arithmetic even when a matrix
# definition is stored in the observation-column table. This shape test
# never evaluates live values or changes their dimensions.
function _is_matrix_value(ex, plan::StructuralPlan, active = Set{Symbol}())
    if ex isa Symbol
        haskey(plan.columns, ex) && return plan.columns[ex] isa AbstractMatrix
        ex in active && return false
        i = findfirst(d -> d.name === ex, plan.derived)
        j = findfirst(a -> a.name === ex, plan.assignments)
        rhs = i !== nothing ? plan.derived[i].expr :
            j !== nothing ? plan.assignments[j].expr : nothing
        rhs === nothing && return false
        push!(active, ex)
        result = _is_matrix_value(rhs, plan, active)
        delete!(active, ex)
        return result
    end
    ex isa Expr || return false
    ex.head === :call && !isempty(ex.args) || return false
    fn = ex.args[1]
    fn === GlobalRef(Base, :hcat) && return length(ex.args) > 1
    fn in (:+, :-, :*, :/, ELEMENTWISE_OPS...) || return false
    return any(a -> _is_matrix_value(a, plan, active), ex.args[2:end])
end

_is_matrix_math(ex, plan) = ex isa Expr && ex.head === :call &&
    !isempty(ex.args) && ex.args[1] in (:+, :-, :*, :/) &&
    any(a -> _is_matrix_value(a, plan), ex.args[2:end])

function _observation_matrix_gather(ex, plan)
    return _is_row_gather(plan, ex)
end

function _model_vector_value(ex, plan)
    if ex isa Symbol
        _is_array_param(plan, ex) && return length(_array_param(plan, ex).dims) == 1
        return ex in _vector_value_names(plan)
    end
    ex isa Expr && ex.head === :ref || return false
    (_is_array_param(plan, ex.args[1]) ||
        _is_array_assignment(plan, ex.args[1])) || return false
    return count(isequal(:(:)), ex.args[2:end]) == 1 &&
        all(_is_position, ex.args[2:end])
end

function _validate_parameters(plan::StructuralPlan)
    names = _union_names(plan)
    for p in plan.parameters
        if p.family === :external
            _validate_external_parameter(plan, p)
            continue
        end
        observed_binomial = p.family === :binomial && p.name in plan.conditioned
        (observed_binomial || haskey(SAMPLED_ARITY, p.family)) || _fail(
            p.label,
            "sampled family $(p.family) not in the slice-1 set " *
            "($(join(sort!(collect(keys(SAMPLED_ARITY))), ", ")))",
        )
        arity = observed_binomial ? 2 : SAMPLED_ARITY[p.family]
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
            _is_matrix_value(v, plan) && _fail(p.label,
                "arg $k references matrix value $v; $(p.family) takes scalar arguments")
            (v in names || (observed_binomial && (!isbound(plan) || haskey(plan.columns,v)))) ||
                _fail(p.label, "arg $k references unknown name $v")
        end
        _validate_uniform_args(p.label, p.family, p.args)
        _validate_support_override(p.label, p.family, p.support_override, p.args)
    end
    return nothing
end

# Uniform priors carry their own interval support: bounds are finite
# literals in order. Sampled or per-cell bounds are rejected — static
# layout cannot transform parameter-dependent support (SB allows sampled
# Uniform endpoints; the thin layer fails closed on them).
function _validate_uniform_args(label, family::Symbol, args::NamedTuple)
    family === :uniform || return nothing
    lo, hi = args.arg1, args.arg2
    all(x -> x isa Symbol || (x isa Real && isfinite(x)), (lo, hi)) ||
        _fail(label, "uniform bounds must be finite values or declared names")
    if lo isa Real && hi isa Real
        lo < hi || _fail(label, "uniform needs lower < upper, got ($lo, $hi)")
    end
    return nothing
end

# Shared support-override rule for scalar and per-cell latent parameters:
# `:positive` half-truncates a symmetric real-support family, and the
# +log(2) renormalization is exact only for a literal zero location.
# `(:interval, lo, hi)` is a two-sided finite truncation with finite lo < hi;
# the family must be real-support (a truncated Normal), and the density carries
# the exact -log(cdf(hi) - cdf(lo)) renormalization at any location.
# `(:upper, hi)` is an upper-only truncation with a finite hi; the family must
# be real-support (a truncated Normal), and the density is Stan's upper-bound
# transform with its normalized upper-tail density.
function _validate_support_override(label, family::Symbol,
        ov::SupportOverride, args::NamedTuple)
    ov === nothing && return nothing
    if _has_interval_bounds(ov)
        ov[1] === :truncated && family === :flat &&
            _fail(label, "truncation requires a proper univariate distribution")
        if ov[1] === :restricted_half
            family in (:normal, :cauchy) && args.arg1 == 0 ||
                _fail(label, "restricted half support requires a zero-location Normal or Cauchy")
        end
        length(ov) == 3 || _fail(label, "truncated support takes (lo, hi)")
        for x in ov[2:end]
            x isa Symbol || x isa Expr || (x isa Real && !isnan(x)) ||
                _fail(label, "truncation bounds must be values or declared names")
        end
        if ov[2] isa Real && ov[3] isa Real
            ov[2] < ov[3] || _fail(label, "truncation needs lower < upper")
        end
        return nothing
    end
    family === :uniform && _fail(label,
        "a uniform prior carries its own interval support — no support " *
        "override applies, got $ov")
    if ov isa Tuple
        if ov[1] === :lower
            length(ov) == 2 || _fail(label,
                "tuple support override must be (:lower, lo), got $ov")
            family === :lognormal || _fail(label,
                "a :lower override is a lower-truncated LogNormal " *
                "(`truncated(LogNormal(m, s), lo, Inf)`); got $family")
            lo = ov[2]
            lo isa Symbol || (isfinite(lo) && lo >= 0) || _fail(label,
                ":lower bound must be a finite non-negative literal or a " *
                "data name; got $lo")
            return nothing
        end
        if ov[1] === :upper
            length(ov) == 2 || _fail(label,
                "tuple support override must be (:upper, hi), got $ov")
            (family === :normal || family === :flat) || _fail(label,
                "an :upper override is a truncated Normal or an " *
                "upper-bounded flat in slice 1 " *
                "(`truncated(Normal(mu, s), -Inf, hi)` / " *
                "`Flat()` with a hand-built upper support); got $family")
            hi = ov[2]
            isfinite(hi) || _fail(label,
                ":upper bound must be finite; got $hi")
            return nothing
        end
        head = ov[1]
        (head === :interval &&
            length(ov) == 3) || _fail(label,
            "tuple support override must be (:interval, lo, hi), " *
            "or (:upper, hi); got $ov")
        (family === :normal || family === :flat) || _fail(label,
            "an $head override is a truncated Normal or a " *
            "flat-on-an-interval in slice 1 " *
            "(`truncated(Normal(mu, s), lo, hi)` / " *
            "`Uniform(lo, hi)`); got $family")
        lo, hi = ov[2], ov[3]
        (isfinite(lo) && isfinite(hi)) || _fail(label,
            "$head bounds must be finite (a one-sided or half truncation " *
            "uses :positive); got ($lo, $hi)")
        lo < hi || _fail(label,
            "$head lower bound must be < upper bound; got ($lo, $hi)")
        return nothing
    end
    ov === :positive ||
        _fail(label, "support override must be :positive; use HalfNormal(s), HalfCauchy(s) or truncated(D, lo, hi), got $ov")
    family === :flat && _fail(label,
        "Flat() is an improper real prior; use Exponential(s) or Uniform(lo, hi) for explicit support")
    family in SYMMETRIC_SAMPLED_FAMILIES || _fail(label,
        "$ov override only applies to symmetric real-support families " *
        "($(join(SYMMETRIC_SAMPLED_FAMILIES, ", "))); got $family")
    loc = family === :student_t ? values(args)[2] : first(values(args))
    loc isa Real && loc == 0 || _fail(label,
        "$ov override requires literal zero location " *
        "(the half shape truncates at 0); got $(repr(loc))")
    return nothing
end

# Per-cell latent (plate) parameters: same family/arity/support grammar as
# scalar SampledParameters. Prior args are either SHARED across cells (a
# literal or a scalar parameter/assignment name) or PER-CELL (a derived column,
# giving a varying prior mean/scale — the varying-intercept shape); never
# another latent vector. The range declares its own domain; `nothing`
# infers a consuming response's axis at binding.
function _validate_plate_parameters(plan::StructuralPlan)
    for p in plan.plate_parameters
        if p.family === :external
            _validate_external_parameter(plan, p)
            if p.name ∉ plan.conditioned
                p.args.geometry.shape == () || _fail(p.label, "plate geometry must be scalar")
                p.args.geometry.unconstrained == 1 || _fail(p.label,
                    "a scalar plate cell needs one packed coordinate; declare an array for joint geometry")
            end
            continue
        end
        haskey(SAMPLED_ARITY, p.family) || _fail(p.label,
            "plate family $(p.family) not in the slice-1 set " *
            "($(join(sort!(collect(keys(SAMPLED_ARITY))), ", ")))")
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
        _validate_uniform_args(p.label, p.family, p.args)
        _validate_support_override(p.label, p.family, p.support_override, p.args)
        if p.range isa UnitRange
            r = p.range
            first(r) == 1 || _fail(p.label,
                "plate range must start at 1 (`1:N`), got $(first(r)):$(last(r))")
        elseif p.range isa Expr
            r = p.range
            valid = (Meta.isexpr(r, :call, 2) && r.args[1] === :eachindex && r.args[2] isa Symbol) ||
                (Meta.isexpr(r, :call, 3) && r.args[1] === :axes && r.args[2] isa Symbol &&
                    r.args[3] isa Int && r.args[3] >= 1)
            valid || _fail(p.label, "plate iterator must be eachindex(v) or axes(v, d)")
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

# Ordered vectors use the same scalar element densities as ordinary
# priors, with the ordering transform independent of the chosen family.
const _VECTOR_ELEMENT_FAMILIES = Dict{Symbol,Symbol}(
    :ordered_normal => :normal, :vector_normal => :normal,
    :ordered_cauchy => :cauchy, :ordered_laplace => :laplace,
    :ordered_logistic => :logistic, :ordered_student_t => :student_t,
    :vector_cauchy => :cauchy, :vector_laplace => :laplace,
    :vector_logistic => :logistic, :vector_student_t => :student_t,
)
_is_ordered_parameter(f) = haskey(_VECTOR_ELEMENT_FAMILIES, f) &&
    startswith(String(f), "ordered_")

"""Vector-parameter families and their positional arg keys."""
const VECTOR_ARITY = Dict{Symbol,Tuple{Vararg{Symbol}}}(
    :ordered_normal => (:arg1, :arg2),
    :ordered_cauchy => (:arg1, :arg2),
    :ordered_laplace => (:arg1, :arg2),
    :ordered_logistic => (:arg1, :arg2),
    :ordered_student_t => (:arg1, :arg2, :arg3),
    :vector_normal => (:arg1, :arg2),
    :vector_cauchy => (:arg1, :arg2),
    :vector_laplace => (:arg1, :arg2),
    :vector_logistic => (:arg1, :arg2),
    :vector_student_t => (:arg1, :arg2, :arg3),
    :simplex_dirichlet => (:arg1,),
    :positive_exponential => (:arg1,),
    :cholesky_corr_lkj => (:arg1,),
)

"""Joint-factor vector families (the correlated-outcomes factor pieces)."""
const _JOINT_FACTOR_FAMILIES = (:positive_exponential, :cholesky_corr_lkj)

# Constrained vector (cutpoint/threshold/simplex) parameters: family/arity
# plus ordinary value arguments. Sizes resolve at bind (`nothing` = infer from the linked leveled
# response, or from the concentration length for a monotonic-linked
# simplex); an explicit size is bounds-checked here and linked-checked in
# `_validate_responses`. Vector declarations can have several readers.
function _validate_vector_parameters(plan::StructuralPlan)
    for p in plan.vector_parameters
        p.size isa Expr && !haskey(_VECTOR_ELEMENT_FAMILIES, p.family) &&
            _fail(p.label, "family $(p.family) needs a concrete integer size")
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
            if alpha isa AbstractVector
                isempty(alpha) && _fail(p.label, "Dirichlet concentration vector must be nonempty")
                all(x -> x isa Real && isfinite(x) && x > 0, alpha) || _fail(
                    p.label, "Dirichlet concentrations must be finite and strictly positive")
                p.size === nothing || p.size == length(alpha) || _fail(p.label,
                    "simplex size $(p.size) disagrees with its concentration length $(length(alpha))")
            elseif alpha isa Symbol || alpha isa Expr
                for ref in _value_symbols(alpha)
                    (!isbound(plan) || ref in _all_names(plan) || haskey(plan.columns, ref)) ||
                        _fail(p.label, "Dirichlet concentration references unknown name $ref")
                end
            else
                _fail(p.label, "Dirichlet concentration must be a vector value")
            end
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
            for (key, arg) in pairs(p.args)
                if arg isa Symbol
                    (!isbound(plan) || arg in _all_names(plan) ||
                        haskey(plan.columns, arg)) || _fail(p.label,
                            "element prior $key references unknown name $arg")
                else
                    (arg isa Real && isfinite(arg)) || _fail(p.label,
                        "element prior $key must be finite, got $(repr(arg))")
                end
            end
            scale = _VECTOR_ELEMENT_FAMILIES[p.family] === :student_t ? p.args.arg3 : p.args.arg2
            scale isa Real && scale <= 0 && _fail(p.label,
                "element prior scale must be positive")
            if _VECTOR_ELEMENT_FAMILIES[p.family] === :student_t
                nu = p.args.arg1
                nu isa Real && nu <= 0 && _fail(p.label,
                    "element prior degrees of freedom must be positive")
            end
            p.size === nothing || p.size isa Expr || p.size >= 0 || _fail(p.label,
                "threshold size must be ≥ 0, got $(p.size)")
        end
    end
    # Linkage: a vector parameter may be referenced by several responses
    # (as `thresholds` for ordered families, as `threshold_coefs` for
    # per-threshold Ordinal, as the simplex `predictor`, or as a joint
    # factor piece), by monotonic terms (as their `increments`
    # simplex), or by an R2D2 prior (as its share `phi`).
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
        # A mixture response links its weights simplex the same way (a
        # literal-weights mixture links nothing).
        if r.family === MixtureFam && r.mixture_weights isa Symbol
            haskey(refs, r.mixture_weights) &&
                push!(refs[r.mixture_weights], r.label)
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
    # Every declaration contributes its prior, including an otherwise
    # unused latent. Its extent must resolve from the declaration at bind.
    return nothing
end

"""Canonical in-graph node names: reserved across every plan namespace."""
const RESERVED_NODES = (:prior, :likelihood, :log_jacobian, :posterior, :pointwise, :unconstrained)

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
    for s in plan.scans
        refs = Set{Symbol}()
        locals = Set(st.target for st in s.step if !st.indexed)
        for st in (s.setup..., s.step...)
            for expr in (st.kind === :sample ? st.args : (st.expr,))
                union!(refs, _expr_value_symbols(expr))
            end
        end
        filter!(r -> r in allnames && r ∉ s.states && r ∉ locals, refs)
        for state in s.states
            deps[state] = copy(refs)
        end
    end
    for p in plan.parameters
        refs = Set{Symbol}()
        for v in (values(p.args)..., _support_args(p.support_override)...)
            for ref in _value_symbols(v)
                (ref in names || p.family === :external || (p.family === :binomial && p.name in plan.conditioned &&
                    (!isbound(plan) || haskey(plan.columns,ref)))) ||
                    _fail(p.label, "bound or arg references unknown name $ref")
                ref in names && push!(refs, ref)
            end
        end
        deps[p.name] = refs
    end
    # Vector parameters are constrained in the layout transforms, ahead of
    # every definition that reads them (functions as values).
    for v in _vector_value_names(plan)
        haskey(deps, v) || (deps[v] = Set{Symbol}())
    end
    for p in plan.vector_parameters
        p.family === :simplex_dirichlet || continue
        deps[p.name] = Set(ref for ref in _value_symbols(p.args.arg1)
            if ref in allnames)
    end
    # Per-cell latent (plate) parameters: prior args are shared scalars,
    # per-cell derived columns, or raw data columns (never another latent).
    # They constrain in the layout transforms like scalar params; a
    # param/assignment/derived arg must be computed before the prior, so it is
    # a real dependency edge. A raw data-column arg is always available (not a
    # node) and is validated at bind, so it adds no edge here.
    for p in plan.plate_parameters
        refs = Set{Symbol}()
        for v in (values(p.args)..., _support_args(p.support_override)...)
            v isa Symbol || continue
            v in allnames && push!(refs, v)
        end
        deps[p.name] = refs
    end
    # Declared array parameters constrain in the layout transforms; a
    # name their prior arguments read is a dependency (expressions
    # included).
    for p in plan.array_parameters
        refs = Set{Symbol}()
        for v in (values(p.args)..., _support_args(p.support_override)...)
            (v isa Symbol || v isa Expr) || continue
            for r in _symbols_in(v)
                r in allnames && push!(refs, r)
            end
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
            d.expr in allnames && push!(refs, d.expr)
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
    # One coefficient per column binds term matrices only: a value matrix
    # is plain data (`hcat(x, x)` is a valid matrix).
    values = _value_design_matrix_names(plan)
    for m in plan.matrices
        isempty(m.columns) && _fail(m.label,
            "design matrix $(m.name) has no columns " *
            "(hcat needs at least one)")
        seen_cols = Set{Symbol}()
        n_intercept = 0
        for c in m.columns
            if c === nothing
                n_intercept += 1
                n_intercept > 1 && m.name ∉ values && _fail(m.label,
                    "design matrix $(m.name) has two intercept positions " *
                    "— one coefficient per column")
                continue
            end
            c in seen_cols && m.name ∉ values && _fail(m.label,
                "design matrix $(m.name) repeats column $c " *
                "— one coefficient per column")
            push!(seen_cols, c)
            c in matnames && _fail(m.label,
                "design matrix $(m.name) nests matrix $c — nested hcat " *
                "is not in slice D1 (flatten it)")
            _is_plate_param(plan, c) && _fail(m.label,
                "design matrix $(m.name) over the latent vector $c is not " *
                "in slice D1 (the me mirror stays affine)")
            any(s -> c in s.states, plan.scans) && _fail(m.label,
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
    if _parameter_term(t)
        _validate_parameter_term(t, plan)
        t = TermSpec(t.kind, t.columns,
            _term_structure_options(t),
            t.addressee, t.label)
    end
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
    if t.kind === ComposedTerm
        _validate_composed_term(t, pred, plan)
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

function _validate_parameter_term(t::TermSpec, plan::StructuralPlan)
    name = t.options.parameter
    name isa Symbol || _fail(t.label, "affine parameter must be a name")
    (hasproperty(t.options, :sign) && t.options.sign in (-1, 1)) || _fail(t.label,
        "affine parameter sign must be -1 or 1")
    if t.kind in (FactorTerm, MatrixTerm)
        any(p -> p.name === name, plan.array_parameters) || _fail(t.label,
            "affine term reads unknown array parameter $name")
    else
        any(p -> p.name === name, plan.parameters) || _fail(t.label,
            "affine term reads unknown scalar parameter $name")
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
    o = _term_structure_options(t)
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

# A scan summand names its carried array (`scan_id`) and its coefficient
# (`coef`) in `options` and carries no columns (the state is sampled or
# reconstructed, not data); its addressee is its own label (self-addressed:
# the coefficient's prior lives on the `SampledParameter`, not a population
# prior). `coef` is a sampled scalar (SB's `ar` latent path with its free
# beta; v1 admits Normal coefficients only) or `nothing` — the state spliced
# directly, beta-free (`mu = a .+ x`, the dar/`mo1` shape). Centered and
# non-centered states both read.
function _validate_scan_term(t::TermSpec, pred::PredictorSpec, plan::StructuralPlan)
    o = t.options
    Tuple(keys(o)) == (:scan_id, :coef) ||
        _fail(t.label, "scan summand options must be exactly " *
              "`(scan_id, coef)`, got $(Tuple(keys(o)))")
    o.scan_id isa Symbol ||
        _fail(t.label, "scan summand scan_id must be a Symbol, " *
              "got $(repr(o.scan_id))")
    (o.coef === nothing || o.coef isa Symbol) ||
        _fail(t.label, "scan summand coef must be a Symbol or nothing " *
              "(a beta-free splice), got $(repr(o.coef))")
    isempty(t.columns) ||
        _fail(t.label, "scan summand carries no columns (the state is " *
              "sampled, not data), got $(t.columns)")
    t.addressee === t.label ||
        _fail(t.label, "scan summand addressee must be its own label " *
              "(self-addressed, no population prior), got $(t.addressee)")
    any(s -> o.scan_id in s.states, plan.scans) ||
        _fail(t.label, "scan summand addresses unknown scan state " *
              ":$(o.scan_id) (no such `@scan` carried array)")
    o.coef === nothing && return nothing
    o.coef in _union_names(plan) ||
        _fail(t.label, "scan summand coef :$(o.coef) must name a scalar " *
              "parameter or assignment")
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

const _COMPOSED_OPS = (:.*, :.+, :.-)
# Elementwise unary maps admitted over a composed subtree (`exp.(la)` —
# the IRT discrimination `a = exp(log_a)`; `logistic.(xi)` — sigmoid
# transient/saturating curves), spelled as Julia dotted calls. At a
# location, one of these over ONE bare sub-predictor is a link spelling.
const _COMPOSED_UNARY = (:exp, :logistic, :normcdf, :cexpexp)
# The other dotted operators (`mu ./ s`, `mu .^ 2`, comparisons for
# `ifelse.`): plain broadcast math over the LP nodes.
const _COMPOSED_MORE_OPS = Tuple(op for op in ELEMENTWISE_OPS
    if op ∉ _COMPOSED_OPS)
# Every elementwise map a composed tree admits: the link-shaped unary
# maps, the dotted built-in math functions, `ifelse.`, and any dotted
# function visible in the model module (a `GlobalRef` head after
# resolution — functions as values: `hypot.(s1, mu .* s2)`).
_composed_map_fn(f) = f isa GlobalRef || f === :ifelse ||
    f in _COMPOSED_UNARY || f in ELEMENTWISE_FNS
const _COMPOSED_AFFINE_KINDS =
    (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm)
# Sub-predictors are affine plus varying-effect summands (per-level
# random effects: the IRT person ability `theta ~ 0 + (1 | person)`).
const _COMPOSED_SUB_KINDS = (_COMPOSED_AFFINE_KINDS..., VaryingEffectTerm)

"""Recurse a composed tree: leaves must be declared subs/scalars/data
columns (a literal arrives as a named scalar leaf), nodes dotted
operators of matching arity or admitted elementwise maps
(`_composed_map_fn`). Returns the leaf set."""
function _validate_composed_tree(tree, subs::Vector{Symbol},
        scalars::Vector{Symbol}, label::Symbol,
        datas::Vector{Symbol} = Symbol[])
    allowed = union(subs, scalars, datas)
    leaves = Symbol[]
    if _is_plate_column_expr(tree)
        # The lambda owns its cell locals; only the plate inputs reference
        # model values. Its body remains the ordinary retained RK cell.
        for input in tree.args[1].args[2:end]
            for name in _expr_value_symbols(input)
                name in allowed || _fail(label,
                    "composed plate input $name is not a declared value")
                push!(leaves, name)
            end
        end
        return leaves
    end
    function walk(node)
        if node isa Symbol
            node in allowed || _fail(label,
                "composed tree leaf $node is neither a declared " *
                "sub-predictor $subs, a scalar $scalars, nor a data " *
                "column $datas")
            push!(leaves, node)
            return nothing
        end
        if node isa Expr && node.head === :. && length(node.args) == 2
            f, tup = node.args
            _composed_map_fn(f) || _fail(label,
                "composed tree map $(repr(f)). is not admitted (" *
                "$(join(string.(_COMPOSED_UNARY, "."), ", ")), the dotted " *
                "built-in math functions, `ifelse.`, or a module function)")
            Meta.isexpr(tup, :tuple) && !isempty(tup.args) || _fail(label,
                "composed tree map $f. takes operands, got $(repr(node))")
            f isa Symbol && length(tup.args) != _elementwise_arity(f) &&
                _fail(label, "composed tree map $f. takes " *
                    "$(_operands_phrase(_elementwise_arity(f))), got " *
                    "$(repr(node))")
            foreach(walk, tup.args)
            return nothing
        end
        node isa Expr && node.head === :call && !isempty(node.args) &&
            node.args[1] isa Symbol || _fail(label,
                "composed tree node $(repr(node)) is not a dotted call " *
                "(dotted operators and maps over sub-predictors and scalars)")
        op = node.args[1]
        op in _COMPOSED_OPS || op in _COMPOSED_MORE_OPS || _fail(label,
            "composed tree op $op is not admitted (dotted operators only)")
        args = node.args[2:end]
        if op === :.-
            length(args) == 1 || length(args) == 2 ||
                _fail(label, "composed `.−` takes one or two operands, " *
                      "got $(length(args))")
        else
            length(args) == 2 ||
                _fail(label, "composed `$op` takes two operands, " *
                      "got $(length(args))")
        end
        for a in args
            walk(a)
        end
        return nothing
    end
    walk(tree)
    return leaves
end

function _validate_composed_term(t::TermSpec, pred::PredictorSpec,
        plan::StructuralPlan)
    o = t.options
    Tuple(keys(o)) == (:tree, :subs, :scalars) ||
        _fail(t.label, "composed options must be exactly " *
              "`(tree, subs, scalars)`, got $(Tuple(keys(o)))")
    o.subs isa Vector{Symbol} && o.scalars isa Vector{Symbol} ||
        _fail(t.label, "composed subs/scalars must be `Vector{Symbol}`")
    # Columns are exactly the tree's data leaves (bound data read
    # elementwise in-graph, e.g. `(log_time .- loc) .* exp.(ls)`).
    datas = Symbol[c for c in t.columns]
    for c in datas
        c in o.subs || c in o.scalars || continue
        _fail(t.label, "composed column $c collides with a sub-predictor " *
              "or scalar name")
    end
    isempty(o.subs) || length(pred.terms) == 1 ||
        _fail(t.label, "composed term is the whole linear predictor " *
              "(no sibling design terms in v1)")
    !isempty(o.subs) && any(pp -> pp.predictor === pred.name, plan.population_priors) &&
        _fail(t.label, "composed predictor $(pred.name) takes no " *
              "population priors (coefficients live in the " *
              "sub-predictors)")
    myidx = findfirst(p -> p.name === pred.name, plan.predictors)
    for s in o.subs
        sidx = findfirst(p -> p.name === s, plan.predictors)
        sidx === nothing &&
            _fail(t.label, "composed sub-predictor $s names no predictor")
        sidx < myidx ||
            _fail(t.label, "composed sub-predictor $s must precede " *
                  "$(pred.name) (LP nodes emit in plan order)")
        sub = plan.predictors[sidx]
        all(u -> u.kind in _COMPOSED_SUB_KINDS, sub.terms) ||
            _fail(t.label, "composed sub-predictor $s must be affine " *
                  "plus varying effects (intercept/continuous/factor/" *
                  "offset/varying-effect terms only — no nested " *
                  "compositions, latents, or other summands)")
    end
    known = _union_names(plan)
    for c in o.scalars
        c in known ||
            _fail(t.label, "composed scalar $c is neither a sampled " *
                  "parameter nor a scalar assignment")
    end
    leaves = _validate_composed_tree(o.tree, o.subs, o.scalars, t.label,
        datas)
    for c in datas
        c in leaves || _fail(t.label, "composed column $c is not a leaf " *
            "of the tree (columns are exactly the tree's data leaves)")
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

# Scalar offset values have no observation axis until preprocessing
# broadcasts them. They are assignments, never design coefficients.
_is_scalar_offset(t::TermSpec, plan::StructuralPlan) =
    t.kind === OffsetTerm &&
    _value_axes(plan, only(t.columns); data_axes = true) == Any[]

function _validate_term_columns(t::TermSpec, pred::PredictorSpec, plan::StructuralPlan)
    _is_scalar_offset(t, plan) && return nothing
    # Latent terms name a per-cell latent VECTOR (a PlateParameter), not a
    # raw/derived data column; structure validation checked its presence.
    t.kind === LatentTerm && return nothing
    # Scan summands name a recurrence + scalar coefficient in `options`, not
    # columns; structure validation checked both names.
    t.kind === ScanSummandTerm && return nothing
    # Dar summands name a trajectory in `options`, not columns; structure
    # validation checked the name.
    t.kind === DarSummandTerm && return nothing
    # mm effect terms name the mm naming symbol (not a data column);
    # membership binding is proven by `_validate_mm_draws_data`.
    if t.kind === VaryingEffectTerm
        i = findfirst(d -> d.label === t.options.draws, plan.varying_draws)
        i !== nothing && plan.varying_draws[i].mm !== nothing && return nothing
    end
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
        # Compositions read retained assignments as whole values. An
        # ordinary module call may return an array or a scalar without a
        # statically known shape; it is still declared in the value graph.
        t.kind === ComposedTerm &&
            any(a -> a.name === c, plan.assignments) && continue
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
        col = _observation_column(plan.columns, c, t.label, "term column")
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
    K = plan.vector_parameters[i].size + 1
    all(v -> 1 <= v <= K, col) ||
        _fail(t.label, "monotonic index $c holds codes outside 1..$K " *
              "(K − 1 = $(K - 1) is the linked increments simplex size)")
    return nothing
end

"""Julia grouping levels, including a categorical column's pool order and
unobserved levels. Copy the result so binder-owned metadata never aliases
caller-owned pool storage. DataAPI's plain-vector fallback sorts uniques."""
_grouping_levels(col::AbstractVector) = Vector(DataAPI.levels(col))
_grouping_levels(col::AbstractArray) = _grouping_levels(vec(col))

# One map per factor term, keyed (predictor, column), so a factor lookup
# resolves to exactly one map.
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
        (s isa Tuple && ((length(s) == 2 && s[1] isa Int && s[2] === :end) ||
            (length(s) == 3 && s[1] isa Int && s[2] isa Int && s[2] != 0 && s[3] === :end))) ||
        return _fail(:plan, "LevelMap subset must be `:`, a UnitRange, " *
                            "a Vector{Int}, (lo, :end), or (lo, step, :end) — got $(repr(s))")
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

# Binder evaluation: Julia levels, then the subset selection
# (bounds-checked against the pool count).
function _eval_levelmaps(levelmaps::Vector{LevelMap},
        columns::AbstractDict{Symbol})
    out = LevelMap[]
    for m in levelmaps
        haskey(columns, m.column) ||
            _fail(:plan, "LevelMap addresses missing column $(m.column)")
        groupcol = columns[m.column]
        groupcol isa AbstractArray ||
            _fail(:plan, "grouping column $(m.column) must be an array")
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
    elseif length(s) == 3
        indices = s[1]:s[2]:K
        all(i -> 1 <= i <= K, indices) || _fail(:plan,
            "LevelMap stepped range exceeds $K levels of $(m.column)")
        levels[indices]
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
    # zero-variance column. An empty evaluated pool is valid; an unfilled
    # map over a nonempty pool still fails.
    for m in plan.levelmaps
        isempty(m.values) || continue
        isempty(only(_eval_levelmaps(LevelMap[m], plan.columns)).values) || _fail(:plan,
            "LevelMap for ($(m.predictor), $(m.column)) has no evaluated " *
            "values (bind_data fills these — hand-built bound plans must too)")
    end
    return nothing
end

# Whether a sampled parameter is provably positive for the scale-hyper
# role: `:positive` support, or an `:interval` whose lower bound is a
# non-negative literal (override tuple or the uniform family's own
# args — non-literal bounds fail closed).
function _hyper_scale_positive(p::Union{SampledParameter,ArrayParameter})
    sup = support_of(p.family, p.support_override)
    sup === :positive && return true
    sup === :interval || return false
    if p.support_override isa Tuple && p.support_override[1] === :interval
        lo = p.support_override[2]
        return lo isa Real && !(lo isa Bool) && lo >= 0
    end
    p.family === :uniform || return false
    hasproperty(p.args, :arg1) || return false
    lo = p.args.arg1
    return lo isa Real && !(lo isa Bool) && lo >= 0
end

function _validate_priors(plan::StructuralPlan)
    seen = Set{Tuple{Symbol,Symbol}}()
    rows = Dict{Tuple{Symbol,Symbol},PopulationPrior}()
    r2d2 = Set{Symbol}(rp.predictor for rp in plan.r2d2_priors)
    hs = Set{Tuple{Symbol,Symbol}}(
        (h.predictor, h.addressee) for h in plan.horseshoe_priors)
    hs_preds = Set{Symbol}(h.predictor for h in plan.horseshoe_priors)
    param_names = Set{Symbol}(p.name for p in plan.parameters)
    by_param = Dict{Symbol,SampledParameter}(p.name => p for p in plan.parameters)
    assign_names = Set{Symbol}(a.name for a in plan.assignments)
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
        key in hs && _fail(:plan,
            "prior for $key duplicates a HorseshoePrior — a horseshoe " *
            "addressee carries its triple, not a PopulationPrior row")
        key in seen &&
            _fail(:plan, "duplicate prior for $key")
        push!(seen, key)
        rows[key] = pr
        pr.family in POPULATION_FAMILIES || _fail(:plan,
            "prior for $key has family $(pr.family) (admitted: " *
            "$(join(POPULATION_FAMILIES, ", ")))")
        # `:flat` contributes 0.0 and ignores location/scale/nu.
        pr.family === :flat && continue
        # `:uniform` carries finite literal bounds (location, scale) =
        # (lo, hi) with lo < hi — no hyperparameter bounds in slice 1.
        if pr.family === :uniform
            (pr.location isa Real && pr.scale isa Real &&
                isfinite(pr.location) && isfinite(pr.scale) &&
                pr.location < pr.scale) ||
                _fail(:plan, "prior for $key must be uniform with " *
                      "finite literal bounds lo < hi, got " *
                      "($(repr(pr.location)), $(repr(pr.scale)))")
            continue
        end
        # Location/scale are literals or hyperparameter names (the
        # centered-hierarchical shape). A location hyperparameter names
        # a scalar sampled parameter or scalar assignment; a scale
        # hyperparameter names a positive-support sampled parameter or a
        # scalar assignment (`b ~ Normal(0, sqrt(phi1 * R2) * tau)` —
        # its positivity is the author's, as for `Normal(0, s)` in
        # Julia). Coefficient priors are DAG sinks (nothing references
        # them, and parameters cannot reference coefficients), so the
        # new edge cannot cycle — no topological check is owed.
        _loc = pr.location
        if _loc isa Symbol
            _loc in param_names || _loc in assign_names ||
                _fail(:plan, "prior for $key has location " *
                      "hyperparameter $_loc — a location hyperparameter " *
                      "names a scalar sampled parameter or scalar " *
                      "assignment (not a data column, coefficient, or " *
                      "vector)")
        else
            isfinite(_loc) ||
                _fail(:plan, "prior for $key must be $(pr.family) " *
                      "with finite location and positive scale")
        end
        _sc = pr.scale
        if _sc isa Symbol && _sc in assign_names
            # A scalar assignment scale (in-graph value).
        elseif _sc isa Symbol
            haskey(by_param, _sc) ||
                _fail(:plan, "prior for $key has scale hyperparameter " *
                      "$_sc — a scale hyperparameter names a " *
                      "positive-support sampled parameter or a scalar " *
                      "assignment (not a data column, coefficient, or " *
                      "vector)")
            # Interval hypers count when the whole interval is
            # non-negative (the open-interval transform keeps reads
            # strictly inside, so the scale stays positive), as with
            # `Uniform(0, hi)`.
            _hyper_scale_positive(by_param[_sc]) ||
                _fail(:plan, "prior for $key has scale hyperparameter " *
                      "$_sc — a scale hyperparameter names a " *
                      "positive-support sampled parameter " *
                      "(`Exponential`, `Gamma`, `HalfNormal`, " *
                      "`HalfCauchy`, `truncated(Normal(0, s), 0, Inf)`, " *
                      "`Uniform(0, hi)`, ...)")
        else
            (isfinite(_sc) && _sc > 0) ||
                _fail(:plan, "prior for $key must be $(pr.family) " *
                      "with finite location and positive scale")
        end
        pr.family === :student_t &&
            (!(pr.nu isa Real) || !isfinite(pr.nu) || !(pr.nu > 0)) &&
            _fail(:plan, "prior for $key must be student_t with " *
                  "finite positive nu, got $(repr(pr.nu))")
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
        # coefficient, so they stay in the addressee set. Composed terms
        # carry no coefficient either (theirs live in the sub-predictors).
        # Matrix terms
        # expand to their per-element addressees (one PopulationPrior row
        # per matrix column).
        addressees = Set{Symbol}()
        for t in pred.terms
            _parameter_term(t) && continue
            (t.kind === OffsetTerm || t.kind === LatentTerm ||
                t.kind === VaryingEffectTerm ||
                t.kind === SplineSummandTerm ||
                t.kind === HSGPSummandTerm ||
                t.kind === ScanSummandTerm ||
                t.kind === MonotonicSummandTerm ||
                t.kind === DarSummandTerm ||
                t.kind === ComposedTerm) && continue
            if t.kind === MatrixTerm
                m = _find_matrix(plan, t.options.matrix)
                m === nothing && _fail(:plan,
                    "internal: matrix term $(t.label) addresses unknown " *
                    "matrix (validate_predictors should have caught this)")
                elems = _matrix_element_addressees(m)
                union!(addressees, elems)
                # One family per matrix block (data-derived widths stay in
                # one plate; only present rows are compared — missing rows
                # fail coverage below).
                present = [rows[(pred.name, e)] for e in elems
                    if haskey(rows, (pred.name, e))]
                isempty(present) || all(p -> p.family === present[1].family,
                    present) || _fail(:plan,
                    "matrix block $(t.options.matrix) of predictor " *
                    "$(pred.name) mixes prior families " *
                    "($(join(unique!(map(p -> p.family, copy(present))), ", "))) — " *
                    "one family per matrix block")
                continue
            end
            push!(addressees, t.addressee)
        end
        any(t -> t.kind === InterceptTerm && !_parameter_term(t),
            pred.terms) && push!(addressees, :Intercept)
        for a in addressees
            # A horseshoe predictor covers an addressee by its entry or by
            # a synthesized Normal scalar (family checked in
            # _validate_horseshoe); every other predictor by a
            # PopulationPrior row.
            (pred.name, a) in seen || (pred.name, a) in hs ||
                (pred.name in hs_preds &&
                    horseshoe_normal_name(pred.name, a) in param_names) ||
                _fail(:plan, "no prior for ($(pred.name), $a)")
        end
    end
    for r in plan.responses
        _is_glm_family(r.family) || continue
        _is_array_param(plan, r.glm_beta) && continue
        m = _find_matrix(plan, r.predictor)
        m === nothing && continue
        for c in m.columns
            c === nothing && continue
            (r.label, c) in seen ||
                _fail(:plan, "no prior for ($(r.label), $c)")
            # GLM-object beta vectors are Normal-only; non-Normal
            # betas use the decomposed predictor form.
            rows[(r.label, c)].family === :normal || _fail(:plan,
                "GLM-object beta prior for ($(r.label), $c) is " *
                "Normal-only (got $(rows[(r.label, c)].family)) — write " *
                "the decomposed predictor form for other families")
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

# Horseshoe structural checks: predictor linkage, one structured prior
# per predictor, scalar-only addressees on scalar-only predictors, use
# polarity, finite positive scales, and triple linkage (each entry's
# (raw, lambda, tau) sampled parameters exist with the SB geometry:
# standard-Normal raw and normalized half-Cauchy scales agreeing with the entry).
function _validate_horseshoe(plan::StructuralPlan)
    r2d2 = Set{Symbol}(rp.predictor for rp in plan.r2d2_priors)
    seen = Set{Tuple{Symbol,Symbol}}()
    by_name = Dict{Symbol,SampledParameter}(
        p.name => p for p in plan.parameters)
    for h in plan.horseshoe_priors
        pred = nothing
        for p in plan.predictors
            p.name === h.predictor && (pred = p)
        end
        pred === nothing && _fail(:plan,
            "horseshoe prior addresses unknown predictor $(h.predictor)")
        h.predictor in r2d2 && _fail(:plan,
            "predictor $(h.predictor) carries both an R2D2Prior and a " *
            "HorseshoePrior — one structured prior per predictor")
        key = (h.predictor, h.addressee)
        key in seen && _fail(:plan, "duplicate horseshoe prior for $key")
        push!(seen, key)
        for t in pred.terms
            (t.kind === InterceptTerm || t.kind === ContinuousTerm ||
                t.kind === OffsetTerm) || _fail(h.predictor,
                "horseshoe over predictor $(h.predictor) meets a " *
                "$(t.kind) term — the flat slice covers " *
                "intercept/continuous coefficients only")
        end
        addrs = Set{Symbol}()
        for t in pred.terms
            t.kind === InterceptTerm && push!(addrs, :Intercept)
            t.kind === ContinuousTerm && push!(addrs, only(t.columns))
        end
        h.addressee in addrs || _fail(h.predictor,
            "horseshoe prior addresses $(h.addressee), not an " *
            "intercept/continuous column of predictor $(h.predictor)")
        (h.sign == 1 || h.sign == -1) || _fail(h.predictor,
            "horseshoe prior for $key has sign $(h.sign) (use polarity " *
            "is +1/-1)")
        isfinite(h.local_scale) && h.local_scale > 0 || _fail(h.predictor,
            "horseshoe prior for $key has local_scale " *
            "$(repr(h.local_scale)) (finite strictly positive)")
        isfinite(h.global_scale) && h.global_scale > 0 || _fail(h.predictor,
            "horseshoe prior for $key has global_scale " *
            "$(repr(h.global_scale)) (finite strictly positive)")
        raw = get(by_name, horseshoe_raw_name(h.predictor, h.addressee),
            nothing)
        raw === nothing && _fail(h.predictor,
            "horseshoe prior for $key names no raw parameter " *
            "($(horseshoe_raw_name(h.predictor, h.addressee)))")
        (raw.family === :normal && raw.args == (arg1 = 0, arg2 = 1) &&
            raw.support_override === nothing) || _fail(h.predictor,
            "horseshoe raw $(raw.name) must be standard-Normal " *
            "(identity support), got $(raw.family)$(raw.args) with " *
            "override $(repr(raw.support_override))")
        for (nm, sc, role) in (
                (horseshoe_lambda_name(h.predictor, h.addressee),
                    h.local_scale, "lambda"),
                (horseshoe_tau_name(h.predictor, h.addressee),
                    h.global_scale, "tau"))
            q = get(by_name, nm, nothing)
            q === nothing && _fail(h.predictor,
                "horseshoe prior for $key names no $role parameter ($nm)")
            (q.family === :cauchy && q.support_override === :positive &&
                length(q.args) == 2 && q.args[1] == 0 &&
                q.args[2] == sc) || _fail(h.predictor,
                "horseshoe $role $nm must be normalized HalfCauchy($sc), got $(q.family)$(q.args) " *
                "with override $(repr(q.support_override))")
        end
    end
    # Non-horseshoe addressees of a horseshoe predictor ride Normal
    # scalars (the mixed-predictor coordinate).
    for pname in Set{Symbol}(h.predictor for h in plan.horseshoe_priors)
        pred = only(p for p in plan.predictors if p.name === pname)
        hs_addrs = Set{Symbol}(h.addressee
            for h in plan.horseshoe_priors if h.predictor === pname)
        for t in pred.terms
            addr = t.kind === InterceptTerm ? :Intercept :
                t.kind === ContinuousTerm ? only(t.columns) : nothing
            addr === nothing && continue
            addr in hs_addrs && continue
            nm = horseshoe_normal_name(pname, addr)
            q = get(by_name, nm, nothing)
            q === nothing && _fail(pname,
                "horseshoe predictor $pname addressee $addr carries " *
                "neither an entry nor its Normal scalar ($nm)")
            (q.family === :normal && q.support_override === nothing &&
                length(q.args) == 2 && isfinite(q.args[1]) &&
                isfinite(q.args[2]) && q.args[2] > 0) || _fail(pname,
                "horseshoe Normal scalar $nm must be " *
                "Normal(finite, positive) (identity support), got " *
                "$(q.family)$(q.args) with override " *
                "$(repr(q.support_override))")
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
    r.mixture_family === nothing ||
        _fail(r.label, "only Mixture takes mixture_family")
    isempty(r.mixture_locs) ||
        _fail(r.label, "only Mixture takes mixture_locs")
    isempty(r.mixture_scales) ||
        _fail(r.label, "only Mixture takes mixture_scales")
    r.mixture_weights === nothing ||
        _fail(r.label, "only Mixture takes mixture_weights")
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
    return _validate_ordered_fields(r, plan, used_predictors)
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
    r.mixture_family === nothing ||
        _fail(r.label, "CategoricalLogit takes no mixture_family")
    isempty(r.mixture_locs) ||
        _fail(r.label, "CategoricalLogit takes no mixture_locs")
    isempty(r.mixture_scales) ||
        _fail(r.label, "CategoricalLogit takes no mixture_scales")
    r.mixture_weights === nothing ||
        _fail(r.label, "CategoricalLogit takes no mixture_weights")
    return nothing
end

function _validate_ordered_fields(r::LikelihoodSpec, plan::StructuralPlan,
        used_predictors::Set{Symbol})
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
    else
        r.ordinal_structure === nothing ||
            _fail(r.label, "OrderedLogistic takes no ordinal_structure")
    end
    (want_ordered || want_plain) || _fail(r.label,
        "internal: ordinal structure $(r.ordinal_structure) unresolved")
    haskey(_VECTOR_ELEMENT_FAMILIES, tp.family) || _fail(r.label,
        "thresholds $(tp.name) is $(tp.family) but this response needs " *
        "a real-support vector element prior")
    if r.n_levels !== nothing && tp.size isa Int
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
    r.mixture_family === nothing ||
        _fail(r.label, "an ordered response takes no mixture_family")
    isempty(r.mixture_locs) ||
        _fail(r.label, "an ordered response takes no mixture_locs")
    isempty(r.mixture_scales) ||
        _fail(r.label, "an ordered response takes no mixture_scales")
    r.mixture_weights === nothing ||
        _fail(r.label, "an ordered response takes no mixture_weights")
    _validate_ordinal_extras(r, plan, used_predictors)
    return nothing
end

# Ordinal-only extras: discrimination (a positive literal, a data
# column resolved at bind, or a modeled scale with support checked at
# execution), per-threshold design columns (StoppingRatio only: cumulative
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
    if d isa ScalePredictorRef
        any(p -> p.name === d.predictor, plan.predictors) || _fail(r.label,
            "ordinal discrimination predictor $(d.predictor) is missing")
        push!(used_predictors, d.predictor)
    elseif d isa Real
        (isfinite(d) && d > 0) || _fail(r.label,
            "ordinal discrimination must be finite and strictly positive, " *
            "got $(repr(d))")
    elseif d isa Symbol
        si = findfirst(p -> p.name === d, plan.predictors)
        if si !== nothing
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
    if r.threshold_effects !== nothing
        r.ordinal_structure === :stopping || _fail(r.label,
            "matrix threshold effects for cumulative responses are not " *
            "built yet (row cutpoints must remain ordered)")
        isempty(r.threshold_columns) && r.threshold_coefs === nothing ||
            _fail(r.label, "use either a threshold-effect matrix or the " *
                "legacy threshold design, never both")
        name = r.threshold_effects
        plan.n_obs == 0 || haskey(plan.columns, name) || _is_derived(plan, name) ||
            any(a -> a.name === name, plan.assignments) ||
            any(a -> a.name === name, plan.array_parameters) ||
            _fail(r.label, "threshold-effect matrix $name is undeclared")
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
    r.mixture_family === nothing ||
        _fail(r.label, "a simplex response takes no mixture_family")
    isempty(r.mixture_locs) ||
        _fail(r.label, "a simplex response takes no mixture_locs")
    isempty(r.mixture_scales) ||
        _fail(r.label, "a simplex response takes no mixture_scales")
    r.mixture_weights === nothing ||
        _fail(r.label, "a simplex response takes no mixture_weights")
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
    if r.range isa UnitRange
        (r.mi_jobs === nothing ? first(r.range) == 1 : first(r.range) >= 1) || _fail(r.label,
            "response range must start at 1 (got $(r.range)) — " *
            "ranges cover eachindex exactly, no partial windows")
        length(r.range) >= 1 || _fail(r.label,
            "response range $(r.range) is empty")
    end
    return nothing
end

# Prob-space Binomial responses (SB `binomial(n, theta)` with a Beta prior,
# the Rate family): no linear predictor — the location names the
# scalar probability in `predictor`, or an identity-link value predictor.
# A fixed probability contributes likelihood without a coordinate.
function _validate_binomial_prob_response(r::LikelihoodSpec, plan::StructuralPlan)
    r.link === IdentityLink || _fail(r.label,
        "a prob-space Binomial response uses IdentityLink (probs are used " *
        "as-is), got $(r.link)")
    pred = findfirst(p -> p.name === r.predictor, plan.predictors)
    pred !== nothing && plan.predictors[pred].link === IdentityLink && return nothing
    i = findfirst(p -> p.name === r.predictor, plan.parameters)
    i === nothing && _fail(r.label,
        "a prob-space Binomial response names its Beta-sampled probability " *
        "in `predictor` (`theta ~ Beta(...)` in the model), got " *
        "$(r.predictor)")
    plan.parameters[i].family === :beta || _fail(r.label,
        "a prob-space Binomial probability is Beta-sampled, got " *
        "$(r.predictor) ~ $(plan.parameters[i].family)")
    return nothing
end

# Zero-inflated prob-space Binomial (the BinomialProb precedent plus the
# ZIP zi slot): the location names a Beta-sampled scalar probability in
# `predictor`, trials ride the shared Binomial rule, and zi (structural-zero
# probability, parameter or literal) is checked by `_validate_zi`.
function _validate_zib_response(r::LikelihoodSpec, plan::StructuralPlan)
    any(p -> p.name === r.predictor, plan.predictors) && return nothing
    any(p -> p.name === r.predictor, plan.parameters) ||
        _fail(r.label, "zero-inflated Binomial probability is undeclared")
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
function _joint_factor_arrays(r::LikelihoodSpec, plan::StructuralPlan)
    si = findfirst(p -> p.name === r.factor_scales, plan.array_parameters)
    ci = findfirst(p -> p.name === r.factor_corr, plan.array_parameters)
    (si === nothing || ci === nothing) && return nothing
    return (plan.array_parameters[si], plan.array_parameters[ci])
end

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
        push!(used_predictors, q)
    end
    r.factor_scales === nothing && _fail(r.label,
        "a joint response requires its factor_scales vector parameter")
    r.factor_corr === nothing && _fail(r.label,
        "a joint response requires its factor_corr vector parameter")
    ap = _joint_factor_arrays(r, plan)
    if ap !== nothing
        sp, cp = ap
        sp.dims == Any[K] || _fail(r.label,
            "joint scale vector $(sp.name) must have the $K outcome entries")
        _hyper_scale_positive(sp) || _fail(r.label,
            "joint scale vector $(sp.name) must have positive support")
        cp.family === :lkj_cholesky && cp.dims == Any[K, K] ||
            _fail(r.label, "joint correlation $(cp.name) must be a $K×$K LKJCholesky factor")
        _lkj_uplo(cp) === 'L' || _fail(r.label,
            "joint covariance uses a lower LKJCholesky factor")
    else
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
    end
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
    r.mixture_family === nothing ||
        _fail(r.label, "a joint response takes no mixture_family")
    isempty(r.mixture_locs) ||
        _fail(r.label, "a joint response takes no mixture_locs")
    isempty(r.mixture_scales) ||
        _fail(r.label, "a joint response takes no mixture_scales")
    r.mixture_weights === nothing ||
        _fail(r.label, "a joint response takes no mixture_weights")
    r.trials === nothing ||
        _fail(r.label, "a joint response takes no trials")
    r.weights === nothing ||
        _fail(r.label, "joint responses take no weights " *
            "(row weights on a joint density are planned)")
    if r.range isa UnitRange
        (r.mi_jobs === nothing ? first(r.range) == 1 : first(r.range) >= 1) || _fail(r.label,
            "response range must start at 1 (got $(r.range)) — " *
            "ranges cover eachindex exactly, no partial windows")
        length(r.range) >= 1 || _fail(r.label,
            "response range $(r.range) is empty")
    end
    return nothing
end

# GLM-object fields belong to the GLM families only: on any other response
# a stray glm_alpha/glm_beta means nothing and would silently change nothing
# (the `_validate_unleveled_fields` rule) — reject it. Runs on every response.
function _validate_glm_fields(r::LikelihoodSpec)
    _is_glm_family(r.family) && return nothing
    r.glm_alpha === nothing ||
        _fail(r.label, "only GLM-object responses take glm_alpha")
    r.glm_beta === nothing ||
        _fail(r.label, "only GLM-object responses take glm_beta")
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
    r.mixture_family === nothing ||
        _fail(r.label, "a GLM-object response takes no mixture_family")
    isempty(r.mixture_locs) ||
        _fail(r.label, "a GLM-object response takes no mixture_locs")
    isempty(r.mixture_scales) ||
        _fail(r.label, "a GLM-object response takes no mixture_scales")
    r.mixture_weights === nothing ||
        _fail(r.label, "a GLM-object response takes no mixture_weights")
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

# Finite-mixture component families admitted in v1 (the D3/slice-2 scalar
# families under their primary link; probit/cloglog are follow-ups).
const _MIXTURE_V1_FAMILIES = (GaussianFam, BernoulliLogitFam, PoissonLogFam,
    BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam, BetaLogitFam)

_mixture_canon_link(f::LikelihoodFamily) =
    f === GaussianFam ? IdentityLink :
    f === BernoulliLogitFam ? LogitLink :
    f === PoissonLogFam ? LogLink :
    f === BinomialLogitFam ? LogitLink :
    f === NegativeBinomial2Fam ? LogLink :
    f === GammaLogFam ? LogLink :
    f === BetaLogitFam ? LogitLink :
    throw(ContractValidationError("internal: mixture link for $f unresolved"))

function _validate_mixture_weights(r::LikelihoodSpec, w, K::Int)
    w isa AbstractVector{<:Real} ||
        _fail(r.label, "mixture weights must be a numeric probability vector")
    length(w) == K || _fail(r.label, "mixture has $K components but $(length(w)) weights")
    all(isfinite, w) || _fail(r.label, "mixture weights must be finite")
    all(>=(0), w) || _fail(r.label, "mixture weights must be nonnegative")
    isapprox(sum(w), 1.0; atol = 1e-8) ||
        _fail(r.label, "mixture weights must sum to 1 (got $(sum(w)))")
    return nothing
end

# A literal component location: finite, and inside the family's
# constrained domain (bare params/literals skip link inversion, so the
# value itself must be valid — the SB runtime-domain mirror).
function _validate_mixture_literal_loc(r::LikelihoodSpec, f::LikelihoodFamily,
        k::Int, v::Real)
    v isa Bool && _fail(r.label,
        "mixture component $k location is Boolean — locations are numeric")
    isfinite(v) || _fail(r.label,
        "mixture component $k location must be finite")
    if f === BernoulliLogitFam || f === BinomialLogitFam
        (0 <= v <= 1) || _fail(r.label,
            "mixture component $k location $v is not a probability " *
            "(bare Bernoulli/Binomial locations are constrained-scale)")
    elseif f === PoissonLogFam || f === NegativeBinomial2Fam
        v >= 0 || _fail(r.label,
            "mixture component $k location $v is negative " *
            "(bare Poisson/NB2 locations are constrained-scale means)")
    elseif f === GammaLogFam
        v > 0 || _fail(r.label,
            "mixture component $k location $v is not positive " *
            "(bare Gamma locations are constrained-scale means)")
    elseif f === BetaLogitFam
        (0 < v < 1) || _fail(r.label,
            "mixture component $k location $v is not inside (0, 1) " *
            "(bare Beta locations are constrained-scale means)")
    end
    return nothing
end

# A finite-mixture response (SB `y ~ MixtureModel(comps, w)` mirror):
# K same-family components over dedicated slots, validated whole. The
# anchor in `predictor` is never resolved here (it may name a
# parameter); component predictors join `used_predictors` like any
# location/scale predictor. Sharing one predictor across components is
# unambiguous and admitted; fully-fixed mixtures fail closed (a
# constant density has no plan).
function _validate_mixture_response(r::LikelihoodSpec, plan::StructuralPlan,
        used_predictors::Set{Symbol})
    f = r.mixture_family
    f === nothing && _fail(r.label,
        "a mixture response requires its mixture_family")
    f in _MIXTURE_V1_FAMILIES || _fail(r.label,
        "mixture components over $f are not admitted in v1 (admitted: " *
        "Gaussian/identity, Bernoulli-logit, Poisson-log, Binomial-logit, " *
        "NB2-log, Gamma-log, Beta-logit)")
    K = length(r.mixture_locs)
    K >= 1 || _fail(r.label, "a mixture response needs ≥ 1 component")
    length(r.mixture_scales) == K || _fail(r.label,
        "mixture has $K locations but $(length(r.mixture_scales)) scales " *
        "(one scale slot per component)")
    loc_preds = Symbol[]
    for (k, loc) in enumerate(r.mixture_locs)
        if loc isa Real
            _validate_mixture_literal_loc(r, f, k, loc)
        elseif loc isa Symbol
            i = findfirst(p -> p.name === loc, plan.predictors)
            if i !== nothing
                pred = plan.predictors[i]
                (f, r.link, pred.link) in ADMITTED_TRIPLES || _fail(r.label,
                    "mixture component $k link triple ($f, $(r.link), " *
                    "$(pred.link)) not admitted")
                push!(used_predictors, loc)
                push!(loc_preds, loc)
            elseif any(p -> p.name === loc, plan.parameters)
                nothing # A sampled scalar parameter: constrained-scale, prior-owned.
            elseif any(a -> a.name === loc, plan.assignments)
                _fail(r.label, "mixture component $k location $loc is a " *
                    "scalar assignment — v1 locations are predictors, " *
                    "sampled parameters, or literals")
            else
                _fail(r.label, "mixture component $k location $loc is not " *
                    "a predictor, sampled parameter, or literal")
            end
        else
            _fail(r.label, "mixture component $k location has no lowering " *
                "(predictor, sampled parameter, or literal)")
        end
    end
    need = _scale_need(f)
    for (k, s) in enumerate(r.mixture_scales)
        if need === nothing
            s === nothing || _fail(r.label,
                "mixture component $k takes no scale auxiliary ($f " *
                "components carry location only)")
        else
            s === nothing &&
                _fail(r.label, "$need (mixture component $k)")
        end
        s isa Bool && _fail(r.label,
            "mixture component $k scale is Boolean — scales are numeric")
        _validate_scale_use(r, plan, s, "mixture component $k scale", f,
            loc_preds)
        s isa ScalePredictorRef && push!(used_predictors, s.predictor)
    end
    w = r.mixture_weights
    w === nothing && _fail(r.label,
        "a mixture response requires its mixture_weights (a literal " *
        "length-K vector, a simplex parameter, or a complement pair)")
    if w isa Symbol
        vi = findfirst(p -> p.name === w, plan.vector_parameters)
        if vi === nothing
            # A shared input vector is checked when bound, just as a literal.
            if isbound(plan)
                haskey(plan.columns, w) || _fail(r.label, "mixture weights $w are not bound")
                _validate_mixture_weights(r, plan.columns[w], K)
            end
        else
            vp = plan.vector_parameters[vi]
            vp.family === :simplex_dirichlet || _fail(r.label,
                "mixture weights $w must be :simplex_dirichlet, got $(vp.family)")
            count = vp.args.arg1 isa AbstractVector ? length(vp.args.arg1) : vp.size
            count === nothing || count == K || _fail(r.label,
                "mixture weights concentration length $count disagrees with the $K components")
        end
    elseif w isa MixtureComplementWeights
        K == 2 || _fail(r.label,
            "complement-pair mixture weights take exactly 2 components " *
            "(got $K)")
        pi = findfirst(p -> p.name === w.param, plan.parameters)
        pi === nothing && _fail(r.label,
            "complement-pair mixture weight $(w.param) is not a sampled " *
            "parameter")
        get(SAMPLED_SUPPORT, plan.parameters[pi].family, :unknown) === :unit ||
            _fail(r.label,
            "complement-pair mixture weight $(w.param) must be " *
            ":unit-support (a Beta/uniform/interval parameter), got " *
            "$(plan.parameters[pi].family)")
    else
        _validate_mixture_weights(r, w, K)
    end
    r.scale === nothing ||
        _fail(r.label, "a mixture response carries no top-level scale " *
            "(scales ride the per-component mixture_scales)")
    _validate_evidence_structure(r, plan)
    r.range === nothing ||
        _fail(r.label, "mixture responses take no range (v1)")
    r.n_levels === nothing ||
        _fail(r.label,
            "a mixture response takes no n_levels (widths are structural)")
    r.thresholds === nothing ||
        _fail(r.label, "a mixture response takes no thresholds")
    isempty(r.extra_predictors) ||
        _fail(r.label, "a mixture response takes no extra_predictors " *
            "(locations ride mixture_locs)")
    isempty(r.count_columns) ||
        _fail(r.label, "a mixture response takes no count_columns")
    r.ordinal_structure === nothing ||
        _fail(r.label, "a mixture response takes no ordinal_structure")
    r.discrimination === nothing ||
        _fail(r.label, "a mixture response takes no discrimination")
    isempty(r.threshold_columns) ||
        _fail(r.label, "a mixture response takes no threshold_columns")
    r.threshold_coefs === nothing ||
        _fail(r.label, "a mixture response takes no threshold_coefs")
    isempty(r.extra_responses) ||
        _fail(r.label, "a mixture response takes no extra_responses")
    r.factor_scales === nothing ||
        _fail(r.label, "a mixture response takes no factor_scales")
    r.factor_corr === nothing ||
        _fail(r.label, "a mixture response takes no factor_corr")
    r.glm_alpha === nothing ||
        _fail(r.label, "a mixture response takes no glm_alpha")
    r.glm_beta === nothing ||
        _fail(r.label, "a mixture response takes no glm_beta")
    if f === BinomialLogitFam
        (r.trials !== nothing || length(r.mixture_trials) == K) ||
            _fail(r.label, "mixture over Binomial requires shared trials or one trials use per component")
        isempty(r.mixture_trials) || (r.trials === nothing && length(r.mixture_trials) == K) ||
            _fail(r.label, "Binomial mixture trials must use either a shared argument or exactly $K component arguments")
    else
        (r.trials === nothing && isempty(r.mixture_trials)) ||
            _fail(r.label, "only Binomial mixtures take trials")
    end
    return nothing
end

# Case-A `mi()` responses a linear predictor observes through packed obs
# slices: these columns ride the managed exemption (the kernel-managed
# precedent), validated under mi rules instead of the uniform-`n_obs`
# rule. The exemption names exactly the columns the `mi_jobs` field
# points at, so a stray caller column cannot hide behind it.
_mi_packed_columns(r::LikelihoodSpec) = [r.response; r.count_columns; r.extra_responses]

function _mi_managed_columns(plan::StructuralPlan)
    out = Set{Symbol}()
    for r in plan.responses
        r.mi_jobs === nothing && continue
        union!(out, _mi_packed_columns(r))
        push!(out, r.mi_jobs)
    end
    return out
end

# Runs first in the per-response loop: every special predictor shape
# (scan/plate/simplex/joint/GLM) `continue`s past the standard checks,
# so the mi gate must precede them all.
function _validate_mi_structure(r::LikelihoodSpec, plan::StructuralPlan)
    r.mi_jobs === nothing && return nothing
    r.mi_jobs ∉ _mi_packed_columns(r) ||
        _fail(r.label, "mi() Jobs column $(r.mi_jobs) must differ from " *
              "every packed response column")
    return nothing
end

function _validate_mi_data(r::LikelihoodSpec, plan::StructuralPlan)
    r.mi_jobs === nothing && return nothing
    haskey(plan.columns, r.mi_jobs) ||
        _fail(r.label, "mi() Jobs column $(r.mi_jobs) missing")
    jobs = plan.columns[r.mi_jobs]
    jobs isa AbstractVector ||
        _fail(r.label, "mi() Jobs column $(r.mi_jobs) must be a vector, " *
              "got $(summary(jobs))")
    eltype(jobs) <: Integer && eltype(jobs) !== Bool ||
        _fail(r.label, "mi() Jobs column $(r.mi_jobs) must hold integers, " *
              "got $(eltype(jobs))")
    o = length(jobs)
    o >= 1 || _fail(r.label,
        "mi() Jobs column $(r.mi_jobs) is empty (at least one observed " *
        "row is required)")
    n = _response_rows(plan, r)
    all(j -> 1 <= j <= n, jobs) ||
        _fail(r.label, "mi() Jobs column $(r.mi_jobs) must index " *
              "1:n_obs ($n)")
    length(unique(jobs)) == o ||
        _fail(r.label, "mi() Jobs column $(r.mi_jobs) must not repeat " *
              "rows (a repeated row would double-count its likelihood)")
    issorted(jobs) ||
        _fail(r.label, "mi() Jobs column $(r.mi_jobs) must ascend " *
              "(the emitter crosses findall order)")
    for name in _mi_packed_columns(r)
        yobs = _vector_column(plan.columns, name, r.label, "mi() response")
        length(yobs) == o ||
            _fail(r.label, "mi() response $name has $(length(yobs)) " *
                  "rows but Jobs selects $o (every outcome must align with Jobs exactly)")
    end
    return nothing
end

function _validate_responses(plan::StructuralPlan)
    # A program may contain only priors, or have every density removed by pins.
    rlabels = [r.label for r in plan.responses]
    length(unique(rlabels)) == length(rlabels) ||
        _fail(:plan, "duplicate response labels")
    for r in plan.responses
        _validate_response_range_expr(r)
        r.label in RESERVED_NODES && _fail(
            r.label,
            "response label collides with a canonical node",
        )
    end
    scan_states = Set{Symbol}(st for s in plan.scans for st in s.states)
    used_predictors = Set{Symbol}()
    for r in plan.responses
        r.threshold_effects === nothing || r.family === OrdinalFam ||
            _fail(r.label, "only Ordinal takes threshold_effects")
        _validate_mi_structure(r, plan)
        _validate_glm_fields(r)
        # A mixture response validates whole (dedicated component slots;
        # the anchor is polymorphic, so this branch leads the scan-state
        # and plate-param checks below).
        if r.family === MixtureFam
            _validate_mixture_response(r, plan, used_predictors)
            _validate_evidence_structure(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
            continue
        end
        # A scale predictor feeds a slot exactly like a location predictor,
        # so it counts toward the unused-predictor check below. A
        # predictor-fed nu or ZIP zi counts the same way.
        r.scale isa ScalePredictorRef &&
            push!(used_predictors, r.scale.predictor)
        r.nu isa ScalePredictorRef &&
            push!(used_predictors, r.nu.predictor)
        r.zi isa ScalePredictorRef &&
            push!(used_predictors, r.zi.predictor)
        # A scan-state latent vector location: the mean is the carried state
        # directly (no linear predictor), with the response's ordinary family/link.
        if r.predictor in scan_states
            _validate_scale(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
            _validate_evidence_structure(r, plan)
            _validate_unleveled_fields(r)
            continue
        end
        # A per-cell latent feeds the ordinary scalar-family location path.
        if _is_plate_param(plan, r.predictor)
            any(t -> t[1] === r.family && t[2] === r.link, ADMITTED_TRIPLES) ||
                _fail(r.label, "plate location family/link $(r.family)/$(r.link) has no scalar observation emitter")
            _validate_scale(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
            _validate_evidence_structure(r, plan)
            _validate_unleveled_fields(r)
            continue
        end
        # A bare sampled-parameter location (constrained-scale, no link
        # inversion — the mixture bare-mean slots, single-family form):
        # Bernoulli/Binomial-logit and Poisson-log over a scalar
        # parameter. Evidence stays fail-closed (the cdf arms are
        # link-space); weights/range ride the generic machinery.
        # Prob-space families skip this gate — they carry their own
        # location validation in the dedicated arms below.
        if any(p -> p.name === r.predictor, plan.parameters) &&
                r.family !== BinomialProbFam &&
                r.family !== ZeroInflatedBinomialFam
            ((r.family === BernoulliLogitFam ||
                r.family === BinomialLogitFam) &&
                r.link === LogitLink) ||
                (r.family === PoissonLogFam && r.link === LogLink) ||
                _fail(r.label,
                "a sampled-parameter response location ($(r.predictor)) is " *
                "Bernoulli/Binomial-logit or Poisson-log only in v1 " *
                "(got $(r.family)/$(r.link))")
            _validate_scale(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
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
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
            _validate_evidence_structure(r, plan)
            continue
        end
        # A prob-space Binomial location (no linear predictor — the
        # simplex precedent): the location names a Beta-sampled scalar
        # parameter in `predictor`.
        if r.family === BinomialProbFam
            _validate_binomial_prob_response(r, plan)
            any(p -> p.name === r.predictor, plan.predictors) &&
                push!(used_predictors, r.predictor)
            _validate_scale(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
            _validate_evidence_structure(r, plan)
            _validate_unleveled_fields(r)
            continue
        end
        # A zero-inflated prob-space Binomial location (the BinomialProb
        # precedent plus zi): the location names a Beta-sampled scalar
        # parameter in `predictor`.
        if r.family === ZeroInflatedBinomialFam &&
                any(p -> p.name === r.predictor, plan.parameters)
            _validate_zib_response(r, plan)
            any(p -> p.name === r.predictor, plan.predictors) &&
                push!(used_predictors, r.predictor)
            _validate_scale(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
            _validate_evidence_structure(r, plan)
            _validate_unleveled_fields(r)
            continue
        end
        # A joint correlated-outcomes response (K outcomes, K mean
        # predictors, one LKJ factor): validated whole, skipping the
        # single-predictor triple.
        if r.family === MvNormalCholeskyFam
            _validate_joint_response(r, plan, used_predictors)
            _validate_scale(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
            _validate_evidence_structure(r, plan)
            continue
        end
        # A GLM-object response (whole-data head over a design matrix —
        # the object owns eta, so no PredictorSpec): validated whole,
        # skipping the single-predictor triple.
        if _is_glm_family(r.family)
            _validate_glm_response(r, plan)
            _validate_scale(r, plan)
            _validate_nu(r, plan)
            _validate_zi(r, plan)
            _validate_interval(r, plan)
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
            "Beta-logit, Categorical-logit, Ordered-logit, Ordinal spellings, " *
            "Student-identity, Hurdle-Poisson-log, ZIP-log, " *
            "InverseGaussian-log, BetaBinomial2-logit, VonMises-identity, " *
            "Exponential-log, LogNormal-identity)",
        )
        _validate_scale(r, plan)
        _validate_nu(r, plan)
        _validate_zi(r, plan)
        _validate_interval(r, plan)
        _validate_evidence_structure(r, plan)
        _validate_leveled_fields(r, plan, pred, used_predictors)
        if r.range isa UnitRange
            (r.mi_jobs === nothing ? first(r.range) == 1 : first(r.range) >= 1) ||
                _fail(r.label, "response range must use valid one-based indices")
            length(r.range) >= 1 || _fail(r.label,
                "response range $(r.range) is empty")
        end
    end
    for kp in plan.kernel_plates
        for (p, _) in kp.lp_args
            push!(used_predictors, p)
        end
    end
    # Composed sub-predictors are used by their composed predictor (the
    # tree references them, not any response slot).
    for pred in plan.predictors
        for t in pred.terms
            t.kind === ComposedTerm || continue
            union!(used_predictors, t.options.subs)
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
        if r.family === MixtureFam && r.mixture_weights isa Symbol &&
                !any(p -> p.name === r.mixture_weights, plan.vector_parameters)
            w = r.mixture_weights
            haskey(plan.columns, w) || _fail(r.label, "mixture weights $w are not bound")
            _validate_mixture_weights(r, plan.columns[w], length(r.mixture_locs))
        end
        _validate_mi_data(r, plan)
        _validate_response_column(r, plan)
        _validate_scale_data(r, plan)
        _validate_auxiliary_data(r, plan, r.nu, :nu)
        _validate_auxiliary_data(r, plan, r.zi, :zi)
        _validate_weights(r, plan)
        _validate_trials(r, plan)
        _validate_evidence_data(r, plan)
        _validate_ordinal_data(r, plan)
    end
    return nothing
end

function _validate_response_column(r::LikelihoodSpec, plan::StructuralPlan)
    if _is_derived(plan, r.response) && !haskey(plan.columns, r.response)
        _fail(r.label, "response $(r.response) is a derived column with " *
            "no bound values — bind_data materializes derived responses, " *
            "so a hand-built plan must include the column")
    end
    haskey(plan.columns, r.response) ||
        _fail(r.label, "response column $(r.response) missing")
    if r.range isa UnitRange
        n = _response_rows(plan, r)
        (r.mi_jobs === nothing ? last(r.range) == n : last(r.range) <= n) ||
            _fail(r.label, "response range $(r.range) is incompatible with $n rows")
    end
    col = r.range isa Expr ? _selected_response_column(plan, r) :
        _observation_column(plan.columns, r.response, r.label, "response")
    if r.evidence.kind in (:interval_censored, :censored) ||
            (r.evidence.kind === :truncated && (_evidence_discrete(r.family) ||
            r.family in (BernoulliLogitGLMFam, PoissonLogGLMFam)))
        # Interval endpoints need not lie in the underlying support. A clamp
        # atom can likewise lie outside that support, or between integers.
        eltype(col) <: Real && all(isfinite, col) ||
            _fail(r.label, "evidence observations must be finite numerics")
        return nothing
    end
    if _is_bernoulli_family(r.family)
        eltype(col) === Bool && return nothing
        eltype(col) <: Integer && all(x -> x == 0 || x == 1, col) && return nothing
        return _fail(r.label, "Bernoulli response must be Bool or 0/1 integers")
    elseif r.family === PoissonLogFam
        eltype(col) <: Integer && all(>=(0), col) && return nothing
        return _fail(r.label, "Poisson response must be non-negative integers")
    elseif _is_binomial_family(r.family)
        _is_count_column(col) && return nothing
        what = r.family === BetaBinomial2Fam ? "BetaBinomial2" :
            r.family === ZeroInflatedBinomialFam ? "ZeroInflatedBinomial" : "Binomial"
        return _fail(r.label, "$what response must be non-negative integers")
    elseif r.family === NegativeBinomial2Fam
        _is_count_column(col) && return nothing
        return _fail(r.label, "NB2 response must be non-negative integers")
    elseif r.family === NegativeBinomialFam
        _is_count_column(col) && return nothing
        return _fail(r.label, "NB1 response must be non-negative integers")
    elseif r.family === HurdlePoissonFam
        _is_count_column(col) && return nothing
        return _fail(r.label, "Hurdle response must be non-negative integers")
    elseif r.family === ZeroInflatedPoissonFam
        _is_count_column(col) && return nothing
        return _fail(r.label, "ZIP response must be non-negative integers")
    elseif r.family === InverseGaussianFam
        # Strictly positive: the Wald kernel guards y > 0 (SB
        # `brm_inverse_gaussian_lpdf` returns -inf at y ≤ 0) — fail closed
        # instead of flowing a wrong value.
        (eltype(col) <: Real && all(>(0), col)) ||
            _fail(r.label, "InverseGaussian response must be strictly positive numerics")
        return nothing
    elseif r.family === ExponentialLogFam
        # Non-negative (0 is valid: Exponential(μ) logpdf at 0 is finite):
        # the exponential kernel guards y >= 0 (SB `exponential_lpdf`
        # returns -inf at y < 0) — fail closed instead of flowing a
        # wrong value. Bool values have their ordinary numeric meaning.
        (eltype(col) <: Real && all(>=(0), col)) ||
            _fail(r.label, "Exponential response must be non-negative numerics")
        return nothing
    elseif r.family === WeibullFam || r.family === WeibullValueFam
        # Strictly positive: the Weibull kernel guards y > 0 (Stan
        # `weibull_lpdf` rejects y < 0 and returns -inf at y = 0 for
        # k < 1) — fail closed instead of flowing a wrong value.
        (eltype(col) <: Real && all(>(0), col)) ||
            _fail(r.label, "Weibull response must be strictly positive numerics")
        return nothing
    elseif r.family === VonMisesFam
        # Finite numeric angles; a circular
        # response additionally honors the half-open principal interval
        # (SB `brm_von_mises_lpdf` returns -inf at y < lo / y >= hi) —
        # fail closed instead of flowing a wrong value. The exact
        # moving support [mu - pi, mu + pi] reads live mu, so only the
        # kernel guards it.
        (eltype(col) <: Real && all(isfinite, col)) ||
            _fail(r.label, "VonMises response must be finite numerics")
        if r.interval !== nothing && all(a -> a isa Real, r.interval)
            lo, hi = r.interval
            all(y -> lo <= y < hi, col) ||
                _fail(r.label, "CircularVonMises response must lie in " *
                    "[$(lo), $(hi))")
        end
        return nothing
    elseif r.family === LogNormalFam
        # Strictly positive:
        # the lognormal kernel guards y > 0 (Stan `lognormal_lpdf`
        # returns -inf at y ≤ 0) — fail closed instead of flowing a
        # wrong value.
        (eltype(col) <: Real && all(>(0), col)) ||
            _fail(r.label, "LogNormal response must be strictly positive numerics")
        return nothing
    elseif r.family === GaussianFam
        eltype(col) <: Real ||
            _fail(r.label, "Gaussian response must be numeric")
        return nothing
    elseif r.family === StudentTFam
        eltype(col) <: Real ||
            _fail(r.label, "Student response must be numeric")
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
    elseif r.family === GammaLogFam || r.family === GammaValueFam
        # Strictly positive: the gamma kernel guards x > 0, and at exactly
        # 0 it is wrong for shape ≤ 1 (says -Inf; truth is finite/+Inf) —
        # fail closed instead of flowing a wrong value.
        (eltype(col) <: Real && all(>(0), col)) ||
            _fail(r.label, "Gamma response must be strictly positive numerics")
        return nothing
    elseif r.family === BetaLogitFam || r.family === BetaShapeFam
        # Strictly inside (0, 1): the beta kernel guards 0 < x < 1, and at
        # exactly 0/1 it is wrong for shapes ≤ 1 (says -Inf; truth is
        # finite/+Inf) — fail closed instead of flowing a wrong value.
        (eltype(col) <: Real && all(x -> 0 < x < 1, col)) ||
            _fail(r.label, "Beta response must be numerics strictly inside (0, 1)")
        return nothing
    elseif r.family === CategoricalLogitFam || _is_ordered_family(r.family) ||
            r.family === CategoricalFam
        # Ordinal support is stated by its cutpoints; observed categories
        # need not exhaust that support. Other recoded responses retain
        # exact contiguity 1..K. K=1 is uniform
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
        (_is_ordered_family(r.family) || r.evidence.kind !== :none ||
            sort(unique(col)) == collect(1:K)) ||
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
    elseif r.family === MixtureFam
        return _validate_mixture_response_column(r, plan, col)
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

# Mixture response column: the single-family column rule of the shared
# component family (same-family mixtures share one support — SB's rule).
function _validate_mixture_response_column(r::LikelihoodSpec,
        plan::StructuralPlan, col::AbstractVector)
    f = r.mixture_family
    if f === BernoulliLogitFam
        eltype(col) === Bool && return nothing
        eltype(col) <: Integer && all(x -> x == 0 || x == 1, col) && return nothing
        return _fail(r.label,
            "mixture response must be Bool or 0/1 integers (Bernoulli components)")
    elseif f === PoissonLogFam
        eltype(col) <: Integer && all(>=(0), col) && return nothing
        return _fail(r.label,
            "mixture response must be non-negative integers (Poisson components)")
    elseif f === BinomialLogitFam
        _is_count_column(col) && return nothing
        return _fail(r.label,
            "mixture response must be non-negative integers (Binomial components)")
    elseif f === NegativeBinomial2Fam
        _is_count_column(col) && return nothing
        return _fail(r.label,
            "mixture response must be non-negative integers (NB2 components)")
    elseif f === GaussianFam
        eltype(col) <: Real ||
            _fail(r.label,
                "mixture response must be numeric (Gaussian components)")
        return nothing
    elseif f === GammaLogFam
        (eltype(col) <: Real && all(>(0), col)) ||
            _fail(r.label,
                "mixture response must be strictly positive numerics (Gamma components)")
        return nothing
    elseif f === BetaLogitFam
        (eltype(col) <: Real && all(x -> 0 < x < 1, col)) ||
            _fail(r.label,
                "mixture response must be numerics strictly inside (0, 1) (Beta components)")
        return nothing
    end
    return _fail(r.label, "mixture over $f has no column rule")
end

# Integer column, all non-negative. Bool is an Integer with values zero/one.
_is_count_column(col) =
    eltype(col) <: Integer && all(>=(0), col)

# Bernoulli spans three link variants (logit + slice-2 probit/cloglog);
# Binomial spans those three plus prob-space (BinomialProbFam, a
# Beta-sampled probability, no link) plus zero-inflated prob-space
# (ZeroInflatedBinomialFam, a Beta-sampled probability + zi, no link);
# response/trials rules are link-independent. BetaBinomial2 shares the
# Binomial trials rule (trials required, y ≤ n).
_is_bernoulli_family(f) =
    f === BernoulliLogitFam || f === BernoulliProbitFam || f === BernoulliCloglogFam
_is_binomial_family(f) =
    f === BinomialLogitFam || f === BinomialProbitFam || f === BinomialCloglogFam ||
    f === BinomialProbFam || f === BetaBinomial2Fam ||
    f === ZeroInflatedBinomialFam

# Scale-auxiliary requirement per family: the need string, or `nothing`
# when the family takes no scale (`_validate_scale` for single responses,
# `_validate_mixture_response` per component).
_scale_need(fam::LikelihoodFamily) =
    fam === GaussianFam ? "Gaussian response requires a scale" :
    fam === NegativeBinomial2Fam ? "NB2 response requires a dispersion phi" :
    fam in (GammaLogFam, GammaValueFam) ? "Gamma response requires a shape alpha" :
    fam === BetaLogitFam ? "Beta response requires a concentration kappa" :
    fam === BetaShapeFam ? "Beta response requires a second shape beta" :
    fam === NormalIDGLMFam ? "NormalIDGLM response requires a scale sigma" :
    fam === StudentTFam ? "Student response requires a scale sigma" :
    fam === HurdlePoissonFam ? "Hurdle response requires a hurdle probability p_zero" :
    fam === InverseGaussianFam ? "InverseGaussian response requires a shape lambda" :
    fam === BetaBinomial2Fam ? "BetaBinomial2 response requires a precision phi" :
    fam === VonMisesFam ? "VonMises response requires a concentration kappa" :
    fam === NegativeBinomialFam ? "NB1 response requires a success probability p" :
    fam === LogNormalFam ? "LogNormal response requires a scale sigma" :
    fam in (WeibullFam, WeibullValueFam) ? "Weibull response requires a shape k" : nothing

function _validate_scale(r::LikelihoodSpec, plan::StructuralPlan)
    need = _scale_need(r.family)
    if need === nothing
        r.scale === nothing ||
            _fail(r.label, "this response family takes no scale auxiliary")
    else
        r.scale === nothing &&
            _fail(r.label, "$need (parameter or literal)")
    end
    return _validate_scale_use(r, plan, r.scale, "scale")
end

# One scale-slot use (a single response's `scale` or one mixture
# component's scale): literals are finite-positive, predictor refs route
# to the predictor rules (against `fam`, forbidding `forbidden`), and
# symbols defer to bind (parameter/assignment now, raw per-observation
# column at `_validate_scale_data`).
function _validate_scale_use(r::LikelihoodSpec, plan::StructuralPlan, s,
        what::AbstractString, fam::LikelihoodFamily = r.family,
        forbidden::Vector{Symbol} = [r.predictor])
    s === nothing && return nothing
    if s isa Real
        if fam === HurdlePoissonFam
            (isfinite(s) && 0 <= s <= 1) ||
                _fail(r.label, "$what literal must lie in [0, 1] (a hurdle probability)")
            return nothing
        elseif fam === NegativeBinomialFam
            (isfinite(s) && 0 <= s <= 1) ||
                _fail(r.label, "$what literal must lie in [0, 1] (an NB1 success probability)")
            return nothing
        end
        (isfinite(s) && s > 0) ||
            _fail(r.label, "$what literal must be finite positive")
        return nothing
    end
    s isa ScalePredictorRef &&
        return _validate_scale_predictor_use(r, plan, s, fam, forbidden)
    # A scalar parameter/assignment scale resolves now; a per-observation scale
    # is a raw data column resolved at bind (see `_validate_scale_data`), so
    # defer an unknown symbol rather than failing structurally (mirrors how
    # per-obs weight/trials columns validate only once data is attached).
    s isa Symbol && s in _union_names(plan) && return nothing
    s isa Symbol && return nothing
    return _fail(r.label, "$what references unknown name $s")
end

# An auxiliary use reads the raw predictor under its own link. Several
# slots and responses may read the same predictor with different links;
# the predictor's metadata does not transform its emitted value.
function _validate_scale_predictor(r::LikelihoodSpec, plan::StructuralPlan,
        s::ScalePredictorRef)
    return _validate_scale_predictor_use(r, plan, s, r.family, [r.predictor])
end

function _validate_scale_predictor_use(r::LikelihoodSpec, plan::StructuralPlan,
        s::ScalePredictorRef, fam::LikelihoodFamily,
        forbidden::Vector{Symbol}, slot::String = "scale")
    _scale_need(fam) !== nothing || fam === StudentTFam ||
        _fail(r.label, "this response family takes no $slot predictor")
    any(p -> p.name === s.predictor, plan.predictors) || _fail(r.label,
        "$slot addresses unknown predictor $(s.predictor)")
    return nothing
end

# Bound scale data must satisfy their distribution's domain and row count.
# Derived and sampled arguments are checked by the value graph and density
# endpoint because their values depend on the query.
function _validate_scale_data(r::LikelihoodSpec, plan::StructuralPlan)
    if r.family === MixtureFam
        for s in r.mixture_scales
            _validate_scale_data_use(r, plan, s)
        end
        return nothing
    end
    return _validate_scale_data_use(r, plan, r.scale)
end

function _validate_scale_data_use(r::LikelihoodSpec, plan::StructuralPlan, s)
    (s === nothing || s isa Real) && return nothing
    # A predictor-fed scale is an n_obs LP by construction (design over the
    # bound rows); there is no column length to check at bind.
    s isa ScalePredictorRef && return nothing
    s isa Symbol || return nothing
    # A data-only scalar definition used as a scale is materialized at bind.
    # Validate that value just like a caller-supplied scale; live definitions
    # and sampled parameters have no bound value here.
    (s in _union_names(plan) || s in _all_names(plan)) &&
        !haskey(plan.columns, s) && return nothing
    haskey(plan.columns, s) ||
        _fail(r.label, "scale references unknown name $s")
    value = plan.columns[s]
    col = value isa Number ? value : _response_slot_column(plan, r, s, "scale column")
    if r.family === HurdlePoissonFam
        (eltype(col) <: Real && all(isfinite, col) &&
            all(x -> 0 <= x <= 1, col)) ||
            _fail(r.label, "per-observation p_zero $s must be finite " *
                "numerics in [0, 1]")
    elseif r.family === NegativeBinomialFam
        (eltype(col) <: Real && all(isfinite, col) &&
            all(x -> 0 <= x <= 1, col)) ||
            _fail(r.label, "per-observation NB1 p $s must be finite " *
                "numerics in [0, 1]")
    else
        (eltype(col) <: Real && all(isfinite, col) && all(>(0), col)) ||
            _fail(r.label, "per-observation scale $s must be finite positive numerics")
    end
    # The observation domain validates every broadcast dimension.
    return nothing
end

# Student degrees of freedom: required (a sampled parameter/assignment
# name, a finite-positive literal, or a predictor-fed per-observation nu),
# scalar or per observation. Declared value nodes resolve structurally;
# their values are guarded by the density endpoint.
function _validate_nu(r::LikelihoodSpec, plan::StructuralPlan)
    if r.family !== StudentTFam
        r.nu === nothing ||
            _fail(r.label, "only Student responses take nu (degrees of freedom)")
        return nothing
    end
    n = r.nu
    n === nothing &&
        _fail(r.label, "Student response requires nu (degrees of freedom, " *
            "parameter, literal, or predictor)")
    if n isa ScalePredictorRef
        return _validate_scale_predictor_use(r, plan, n, r.family,
            [r.predictor], "nu")
    end
    n isa Real && ((isfinite(n) && n > 0) ||
        _fail(r.label, "nu literal must be finite positive"))
    n isa Symbol && (n in _all_names(plan) || haskey(plan.columns, n) ||
        !isbound(plan) || _fail(r.label, "nu references unknown name $n"))
    return nothing
end

# Zero-inflation probability: required (a sampled parameter/assignment
# name, a literal in [0, 1], or a predictor-fed zi submodel), scalar or
# per-observation value. Declared value nodes resolve structurally;
# their values are guarded by the density endpoint.
function _validate_zi(r::LikelihoodSpec, plan::StructuralPlan)
    if r.family !== ZeroInflatedPoissonFam &&
            r.family !== ZeroInflatedBinomialFam
        r.zi === nothing ||
            _fail(r.label, "only ZIP/ZIB responses take zi (zero-inflation probability)")
        return nothing
    end
    z = r.zi
    z === nothing &&
        _fail(r.label, "ZIP/ZIB response requires zi (zero-inflation " *
            "probability, parameter or literal)")
    z isa ScalePredictorRef && return _validate_zi_predictor(r, plan, z)
    z isa Real && ((isfinite(z) && 0 <= z <= 1) ||
        _fail(r.label, "zi literal must lie in [0, 1]"))
    z isa Symbol && (z in _all_names(plan) || haskey(plan.columns, z) ||
        !isbound(plan) || _fail(r.label, "zi references unknown name $z"))
    return nothing
end

# Raw data arguments resolve at binding, like scales. Live definitions and
# sampled values keep the ordinary endpoint's lazy support guard.
function _validate_auxiliary_data(r::LikelihoodSpec, plan::StructuralPlan, value,
        slot::Symbol)
    value isa Symbol || return nothing
    value in _all_names(plan) && !haskey(plan.columns, value) && return nothing
    haskey(plan.columns, value) ||
        _fail(r.label, "$slot references unknown name $value")
    data = plan.columns[value]
    selected = data isa Number ? data :
        _response_slot_column(plan, r, value, "$slot data")
    (eltype(selected) <: Real && all(isfinite, selected)) ||
        _fail(r.label, "$slot data $value must be finite real numerics")
    if slot === :nu
        all(>(0), selected) ||
            _fail(r.label, "nu data $value must be finite positive numerics")
    else
        all(x -> 0 <= x <= 1, selected) ||
            _fail(r.label, "zi data $value must lie in [0, 1]")
    end
    return nothing
end

# The zero-inflation slot uses the same per-use link semantics as scale.
function _validate_zi_predictor(r::LikelihoodSpec, plan::StructuralPlan,
        z::ScalePredictorRef)
    any(p -> p.name === z.predictor, plan.predictors) || _fail(r.label,
        "zi addresses unknown predictor $(z.predictor)")
    return nothing
end

# VonMises principal interval: `nothing` for exact `VonMises` (moving
# support), or the `(lo, hi)` literal pair for `CircularVonMises`
# (half-open support) — the BRM `_brm_circular_interval` rule:
# finite endpoints, `lo < hi`, width exactly `2pi` (within `8eps`,
# the shared tolerance). Only VonMises responses take it.
function _validate_interval(r::LikelihoodSpec, plan::StructuralPlan)
    if r.family !== VonMisesFam
        r.interval === nothing ||
            _fail(r.label, "only VonMises responses take interval " *
                "(principal-interval endpoints)")
        return nothing
    end
    r.interval === nothing && return nothing
    for a in r.interval
        a isa Real || a in _all_names(plan) ||
            (!isbound(plan) || haskey(plan.columns, a)) || _fail(r.label,
                "interval references unknown name $a")
    end
    # Live endpoints are checked in the generated cell, before modulo or
    # density evaluation. Literal endpoints can be checked immediately.
    if all(a -> a isa Real, r.interval)
        lo, hi = r.interval
        (isfinite(lo) && isfinite(hi) && lo < hi) || _fail(r.label,
            "VonMises interval must be finite with lo < hi")
        isapprox(hi - lo, 2 * Float64(pi); rtol = 8eps(Float64),
            atol = 8eps(Float64)) || _fail(r.label,
                "VonMises interval must have length 2pi")
    end
    return nothing
end

function _validate_trials(r::LikelihoodSpec, plan::StructuralPlan)
    if r.family === MultinomialFam
        return _validate_multinomial_trials(r, plan)
    end
    if r.family === MixtureFam
        return _validate_mixture_trials(r, plan)
    end
    if !_is_binomial_family(r.family)
        r.trials === nothing ||
            _fail(r.label, "only Binomial/BetaBinomial2/Multinomial responses take trials")
        return nothing
    end
    what = r.family === BetaBinomial2Fam ? "BetaBinomial2 response" :
        r.family === ZeroInflatedBinomialFam ? "ZeroInflatedBinomial response" :
        "Binomial response"
    r.trials === nothing && _fail(r.label,
        "$what requires trials (Int column or literal)")
    return _validate_trials_values(r, plan, what)
end

# Binomial-component mixtures share one trials use (SB's
# identical-expression rule holds structurally — one field); every
# other component family takes none.
function _validate_mixture_trials(r::LikelihoodSpec, plan::StructuralPlan)
    if r.mixture_family === BinomialLogitFam
        (r.trials !== nothing || !isempty(r.mixture_trials)) ||
            _fail(r.label, "mixture over Binomial requires trials")
        if isempty(r.mixture_trials)
            return _validate_trials_values(r, plan, "mixture response")
        end
        for t in r.mixture_trials
            # An observation can be outside one component's support; that
            # component contributes zero probability to the mixture.
            _validate_trials_values(_with(r; trials = t), plan,
                "mixture component"; check_support = false)
        end
        return nothing
    end
    r.trials === nothing ||
        _fail(r.label, "only Binomial mixtures take trials")
    return nothing
end

function _validate_trials_values(r::LikelihoodSpec, plan::StructuralPlan,
        what::AbstractString; check_support::Bool = true)
    ycol = _response_slot_column(plan, r, r.response, "response")
    t = r.trials
    if t isa Int
        t >= 0 || _fail(r.label, "trials literal must be non-negative")
        (!check_support || r.evidence.kind in (:censored, :interval_censored) ||
            all(ycol .<= t)) ||
            _fail(r.label, "$what exceeds trials $t")
        return nothing
    end
    _is_derived(plan, t) && _fail(r.label,
        "trials column $t is derived — slice-1 binds trials " *
        "raw (derived trials need shape metadata — planned)")
    haskey(plan.columns, t) ||
        _fail(r.label, "trials column $t missing")
    col = _response_slot_column(plan, r, t, "trials column")
    (eltype(col) <: Integer && eltype(col) !== Bool) ||
        _fail(r.label, "trials column must hold integers")
    all(>=(0), col) ||
        _fail(r.label, "trials column must be non-negative")
    # The observation domain validates every broadcast dimension.
    if r.mi_jobs !== nothing
        n = _response_rows(plan, r)
        length(col) == n ||
            _fail(r.label, "trials column length $(length(col)) ≠ n_obs $n")
    end
    observed_trials = r.mi_jobs === nothing ? col : col[plan.columns[r.mi_jobs]]
    (!check_support || r.evidence.kind in (:censored, :interval_censored) ||
        all(ycol .<= observed_trials)) ||
        _fail(r.label, "$what exceeds trials in some row")
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
    n = r.mi_jobs === nothing ? _response_rows(plan, r) : length(plan.columns[r.mi_jobs])
    rowsums = zeros(Int, n)
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
    observed_trials = r.mi_jobs === nothing ? col : col[plan.columns[r.mi_jobs]]
    all(rowsums .== observed_trials) ||
        _fail(r.label, "multinomial row counts must sum to trials in every row")
    return nothing
end

# Ordinal extras at data level: a discrimination column is raw finite
# positive numerics of length n_obs (a literal validated structurally; a
# predictor names a modeled scale whose support is checked at execution),
# and threshold design columns are raw finite numerics of length n_obs.
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
            "discrimination $d must be a data column or a " *
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
    if r.threshold_effects !== nothing &&
            haskey(plan.columns, r.threshold_effects)
        _ordinal_effects_matrix(plan.columns[r.threshold_effects],
            _response_rows(plan, r), r.n_levels)
    end
    return nothing
end

function _validate_weights(r::LikelihoodSpec, plan::StructuralPlan)
    r.weights === nothing && return nothing
    haskey(plan.columns, r.weights) ||
        _fail(r.label, "weights column $(r.weights) missing")
    col = _response_slot_column(plan, r, r.weights, "weights column")
    eltype(col) <: Real && all(isfinite, col) && all(>=(0), col) ||
        _fail(r.label, "frequency weights must be finite non-negative numerics")
    return nothing
end

function _validate_evidence_structure(r::LikelihoodSpec, plan::StructuralPlan)
    ev = r.evidence
    ev.kind in (:none, :truncated, :censored, :interval_censored) ||
        _fail(r.label, "evidence kind $(ev.kind) unknown")
    ev.kind === :none && return nothing
    r.family in (MvNormalCholeskyFam, MultinomialFam) &&
        _fail(r.label, "scalar evidence bounds require a univariate distribution")
    for b in (ev.lower, ev.upper)
        isbound(plan) && b isa Symbol && b ∉ _all_names(plan) && !haskey(plan.columns, b) &&
            _fail(r.label, "evidence bound $b is not a declared value")
    end
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
    if r.mi_jobs !== nothing
        jobs = plan.columns[r.mi_jobs]
        lo = lo isa AbstractArray ? lo[jobs] : lo
        hi = hi isa AbstractArray ? hi[jobs] : hi
    end
    if ev.kind === :interval_censored
        resp = _response_slot_column(plan, r, r.response, "response")
        all(isfinite, resp) ||
            _fail(r.label, "interval evidence requires finite response values")
        (hi === nothing || all(resp .< hi)) ||
            _fail(r.label, "interval evidence requires response < upper every row")
        return nothing
    end
    (lo === nothing || hi === nothing || all(lo .<= hi)) ||
        _fail(r.label, "evidence requires lower ≤ upper every row")
    resp = _response_slot_column(plan, r, r.response, "response")
    below = lo === nothing ? false : resp .< lo
    above = hi === nothing ? false : resp .> hi
    bad = lo === nothing && hi === nothing ? Int[] : findall(below .| above)
    if r.mi_jobs !== nothing && r.range !== nothing
        selected = findall(j -> j in r.range, plan.columns[r.mi_jobs])
        filter!(i -> i in selected, bad)
    end
    isempty(bad) || _fail(r.label,
        "response $(r.response) is outside its $(ev.kind) bounds " *
        "at rows $(join(bad, ", ")) (requires lower ≤ response ≤ upper)")
    return nothing
end

function _bound_values(bound, side, r::LikelihoodSpec, plan::StructuralPlan)
    bound === nothing && return nothing
    value = bound isa Real ? bound : get(plan.columns, bound, nothing)
    # Live bounds are validated by the generated cell; bound data retain
    # their Julia broadcast axes and are gathered once for packed outcomes.
    value === nothing && return nothing
    if value isa Real
        isfinite(value) || _fail(r.label, "$side bound must be finite")
        return Float64(value)
    end
    value = _response_slot_column(plan, r, bound, "$side bound")
    eltype(value) <: Real && all(isfinite, value) ||
        _fail(r.label, "$side bound must contain finite numerics")
    return value
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
gates. Shared by basis binds."""
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
        if hb.cov === :periodic
            # No domain to fit (SB `_brm_hsgp_basis_state` periodic
            # branch): the axis still binds as a numeric finite vector
            # (SB `_brm_gp_axes` — no degeneracy gate, a constant axis
            # is a usable periodic domain).
            c = only(hb.axes)
            haskey(columns, c) ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c is " *
                      "not bound")
            col = _vector_column(columns, c, hb.label, "hsgp axis column")
            eltype(col) <: Real ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c must " *
                      "be numeric, got $(eltype(col))")
            all(isfinite, col) ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c " *
                      "must be finite")
            push!(out, HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso,
                Tuple{Float64,Float64}[], hb.label, hb.cov, hb.period,
                hb.rho_prior, hb.sigma_prior, hb.domain))
            continue
        end
        fits = Tuple{Float64,Float64}[]
        for (j, (c, cj)) in enumerate(zip(hb.axes, hb.c))
            haskey(columns, c) ||
                _fail(hb.label, "hsgp :$(hb.id): axis column $c is " *
                      "not bound")
            col = _vector_column(columns, c, hb.label, "hsgp axis column")
            if hb.domain === nothing
                push!(fits, _hsgp_axis_fit(col, cj, hb.label,
                    "hsgp :$(hb.id): axis column $c"))
            else
                # Fixed domain (SB `_brm_hsgp_domain_fits` /
                # `_brm_check_hsgp_domain`): the data must lie inside.
                lo, hi = hb.domain[j]
                eltype(col) <: Real && all(isfinite, col) ||
                    _fail(hb.label, "hsgp :$(hb.id): axis column $c " *
                          "must be finite numeric")
                all(v -> lo <= v <= hi, col) || _fail(hb.label,
                    "hsgp :$(hb.id): axis column $c has values outside " *
                    "its fixed domain ($lo, $hi)")
                push!(fits, ((lo + hi) / 2, (hi - lo) / 2))
            end
        end
        by = hb.by
        if by !== nothing && by.levels === nothing
            haskey(columns, by.column) || _fail(hb.label, "hsgp " *
                ":$(hb.id): grouping column $(by.column) is not bound")
            gcol = _vector_column(columns, by.column, hb.label,
                "hsgp grouping column")
            levels = try
                _grouping_levels(gcol)
            catch err
                _fail(hb.label, "hsgp :$(hb.id): grouping column " *
                    "$(by.column) levels not orderable ($err)")
            end
            by = HSGPGrouping(by.column, collect(Any, levels))
        end
        push!(out, HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, fits,
            hb.label, hb.cov, hb.period, hb.rho_prior, hb.sigma_prior,
            hb.domain, by))
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
# — each PK call emits a subject plate containing scans); this pass only proves
# shapes.
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
                # event-axis vector (validated above), not a
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
        srcshape = if src isa Expr && src.head === :vect
            all(a -> _grouped_cell_shape(a,kp,shapes,columns) === :scalar,src.args) ||
                _fail(label,"literal gather entries must be scalar")
            (:obs,length(src.args))
        else
            _grouped_cell_shape(src,kp,shapes,columns)
        end
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
            if src isa Expr
                bound = length(src.args)
                all(1 .<= mapv .<= bound) ||
                    _fail(kp.label,"gather map `$mapcol` has entries outside 1..$bound (scalar vector literal)")
            elseif src in _lp_cell_params(kp)
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
    out = Tuple{Union{Symbol,Expr},Symbol}[]
    _collect_grouped_gather_uses!(out, ex)
    return out
end

function _collect_grouped_gather_uses!(out::Vector{Tuple{Union{Symbol,Expr},Symbol}}, ex)
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
        columns::Dict{Symbol,ColumnData}, dims::AbstractDict{Symbol,<:Integer},
        consumed::Set{Symbol})
    for (k, v) in dims
        v > 0 ||
            _fail(kp.label, "dims key `$k` must bind a positive integer, " *
                  "got $v")
    end
    # Leftovers fail once, globally, in `_resolve_kernels!` (a key for a
    # sibling plate is not this plate's typo).
    sched = only(kp.schedules)
    built = _build_grouped_schedule(plan, kp, columns)
    n_sub = if kp.subjects isa Int
        kp.subjects
    elseif kp.subjects === nothing
        # Shape from named data: the schedule's subject column (subjects
        # are 1:n, proved by the build) — no dims key.
        built.n_subjects
    else
        haskey(dims, kp.subjects) ||
            _fail(kp.label, "subjects dims key `$(kp.subjects)` is not " *
                  "bound (bind_data `dims` carries it; admitted: an " *
                  "integer literal or a dims-key name)")
        push!(consumed, kp.subjects)
        Int(dims[kp.subjects])
    end
    # Dose/PK coherence (v2 axis 1): dose rows must feed the model and
    # PK calls need dose rows. Dose-free subjects alongside dosed ones
    # stay admitted — this gates only the global emptiness.
    ndose = length(columns[sched.dose_subj])
    if _cell_has_pk_call(kp.assignments)
        ndose > 0 ||
            _fail(kp.label, "cell calls a PK recurrence but schedule " *
                  "`$(sched.name)` binds no dose rows (bind dose data " *
                  "or drop the call)")
    elseif ndose > 0
        _fail(kp.label, "schedule `$(sched.name)` binds $ndose dose " *
              "rows but the cell makes no PK call (missing read_locs " *
              "call? — or bind empty dose columns for a dose-free plate)")
    end
    built.n_subjects == n_sub ||
        _fail(kp.label, "schedule `$(sched.name)` covers " *
              "$(built.n_subjects) subjects ≠ subjects $n_sub")
    for f in vcat(collect(_sched_materialized_fields(sched)),
            _sched_extra_fields(sched, kp.assignments))
        col = _sched_col_name(sched.name, f)
        haskey(columns, col) &&
            _fail(kp.label, "column `$col` is reserved for schedule " *
                  "`$(sched.name)`'s bind product — rename the " *
                  "caller-supplied column")
        columns[col] = getfield(built, f)
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
    for obs in kp.obs
        obs.family === BernoulliLogitFam || continue
        rcol = only(c for (c, p, _) in slices2 if p === obs.response)
        flatv = _vector_column(columns, rcol, kp.label, "slice column")
        _materialize_kernel_bool_twin!(kp, obs, flatv, columns)
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
# expansions (spline-blocks precedent). Each plate consumes its own keys
# through one shared set (subjects named per plate, or derived from a
# schedule's subject column; timepoints only via the `kernel_T_<result>`
# key); leftovers fail once, globally (a key for plate B is not plate
# A's typo). Returns resolved nodes; the input plan is untouched.
function _resolve_kernels!(plan::StructuralPlan,
        columns::Dict{Symbol,ColumnData}, dims::AbstractDict{Symbol,<:Integer})
    if isempty(plan.kernel_plates)
        # Nothing consumes a dims key: a key here is a typo or a shape
        # the model never asks for (plates take their shapes from data).
        isempty(dims) ||
            _fail(:plan, "dims key(s) $(sort!(collect(keys(dims)))) not " *
                  "consumed (the model has no kernel plate — `@plate` " *
                  "ranges and schedules take their shapes from data)")
        return KernelPlate[]
    end
    # One shared consumed set: each plate consumes its own dims keys;
    # leftovers fail once, globally (a key for plate B is not plate
    # A's typo).
    consumed = Set{Symbol}()
    out = KernelPlate[]
    for kp in plan.kernel_plates
        if _is_grouped_kernel(kp)
            push!(out,
                _resolve_grouped_kernel!(plan, kp, columns, dims, consumed))
        else
            push!(out,
                _resolve_panel_kernel!(kp, columns, dims, consumed))
        end
    end
    leftovers = setdiff(Set{Symbol}(keys(dims)), consumed)
    isempty(leftovers) ||
        _fail(:plan, "dims key(s) $(sort!(collect(leftovers))) not " *
              "consumed by any kernel plate (typo'd key? — grouped " *
              "kernels take no timepoints dims key; multi-plate " *
              "timepoints keys spell `kernel_T_<result>`)")
    return out
end

function _resolve_panel_kernel!(kp::KernelPlate,
        columns::Dict{Symbol,ColumnData}, dims::AbstractDict{Symbol,<:Integer},
        consumed::Set{Symbol})
    for (k, v) in dims
        v > 0 ||
            _fail(kp.label, "dims key `$k` must bind a positive integer, " *
                  "got $v")
    end
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
        # Unnamed timepoints (surface always leaves `nothing`): only the
        # `kernel_T_<result>` key names the plate's T (the slice-length
        # errors prescribe this spelling). No other key is ever taken as
        # T — a typo'd or stray key stays unconsumed and fails in the
        # global leftovers gate, and a T-needing plate fails at its slice
        # lengths naming the convention.
        conv = Symbol("kernel_T_$(kp.result)")
        if haskey(dims, conv)
            push!(consumed, conv)
            Int(dims[conv])
        else
            nothing
        end
    end
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
                      "(vector slices need T: bind `kernel_T_$(kp.result)`)")
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
            # Numeric gate ahead of the T-block conversion (data
            # validation's twin check runs after this resolve — without
            # this, a String/Symbol column dies as a raw MethodError).
            eltype(colv) <: Real ||
                _fail(kp.label, "slice column `$col` must be numeric, " *
                      "got $(eltype(colv))")
            columns[exp] = repeat(colv; inner = T)
        end
    end
    for obs in kp.obs
        obs.family === BernoulliLogitFam || continue
        si = findfirst(s -> s[2] === obs.response, slices2)
        col, kind = slices2[si][1], slices2[si][3]
        flatv = (kind === :scalar && T !== nothing) ?
            columns[_kexp_name(kp.result, col)] :
            _vector_column(columns, col, kp.label, "slice column")
        _materialize_kernel_bool_twin!(kp, obs, flatv, columns)
    end
    resolved0 = KernelPlate(kp.result, n_sub, T, slices2,
        kp.assignments, kp.obs, kp.collected, kp.label)
    canon = _canonicalize_kernel_assignments(resolved0)
    return KernelPlate(kp.result, n_sub, T, slices2,
        canon, kp.obs, kp.collected, kp.label)
end

# n_obs derivation skips mi-managed columns (packed y_obs/Jobs), bound
# module data values and model-level data inputs: every other column
# crosses at length n, so the first non-managed column pins n_obs
# order-independently. A bound hand-authored mi plan may supply its full
# extent explicitly; an unbound plan without an anchor cannot derive it.
function _bind_nrows(columns::AbstractDict{Symbol}, managed::Set{Symbol})
    for (k, v) in columns
        k in managed && continue
        # A number or a higher-dimensional array has no observation axis.
        (v isa Number || ndims(v) > 2) && continue
        return _column_nrows(v)
    end
    throw(ContractValidationError(
        "[bind] cannot derive n_obs: every crossed column is mi-managed " *
        "(an mi-only column set carries no length-n anchor)"))
end

# Bind-time materialization of derived responses: a response over a
# derived column (`ly = log.(earn)` then `ly .~ ...`) evaluates its
# (structure-proven) expression over the bound columns and joins them,
# so every downstream consumer — roles, n_obs, validation, generation —
# reads it exactly like a raw column. The in-graph local (`ly =
# log.(earn)` in `_assignment_statements`) recomputes the identical value
# (same expression, same inputs); `bound=` folds the data-only duplicate
# on compiled paths.
# ── Functions as values: bind-time data definitions ───────────────────
# A definition whose expression calls a module function and reads only
# data (raw columns or other data-only definitions) is data: `bind_data`
# evaluates bind-time definitions once, as plain Julia, and binds each value
# under the definition's name. Every consumer reads it exactly like a bound
# column — the generated kernel takes it as a data argument (bound by
# `prepare_query`), never recomputing it. A derived column (observation
# aligned) must come out a length-n_obs vector and validates as a column;
# a model-level assignment may be any number, vector or matrix. Whole-value
# definitions used only by parameter-dependent calls fold at preparation.

# Plan names an expression reads (call heads, keyword names and function
# values are not reads).
function _expr_value_symbols(ex, out::Set{Symbol} = Set{Symbol}())
    if ex isa Symbol
        push!(out, ex)
    elseif ex isa Expr
        if ex.head === :kw && length(ex.args) == 2
            _expr_value_symbols(ex.args[2], out)
        elseif ex.head === :call
            for a in ex.args[2:end]
                _expr_value_symbols(a, out)
            end
        elseif ex.head === :. && length(ex.args) == 2 &&
                ex.args[2] isa Expr && ex.args[2].head === :tuple
            for a in ex.args[2].args
                _expr_value_symbols(a, out)
            end
        else
            for a in ex.args
                ex.head === :tuple && (a = _tuple_field_value(a))
                _expr_value_symbols(a, out)
            end
        end
    end
    return out
end

"""Names of the data-only definitions of `plan`, given the raw
(caller-supplied) column names: assignments and derived columns that call a
module function or fill a bind-time slot, and read only raw data or other
data-only definitions.
Derived responses keep their own materialization. `bind_only` excludes
definitions used only during preparation, while preserving their names
for caller-supplied column collision checks."""
function _module_data_names(plan::StructuralPlan, raw::AbstractSet{Symbol};
        bind_only = false, unbound::Bool=false)
    required = Set{Symbol}(r.weights for r in plan.responses if r.weights !== nothing)
    for r in plan.responses, scale in (r.scale, r.mixture_scales...)
        scale isa Symbol && push!(required, scale)
    end
    for p in plan.array_parameters, d in p.dims
        _is_levels_dim(d) && push!(required, d.args[2])
    end
    for p in plan.vector_parameters
        p.extent_expr === nothing || union!(required, _expr_value_symbols(p.extent_expr))
    end
    nodes = Dict{Symbol,Any}()
    for a in plan.assignments
        # Literal definitions can be dependencies of a module call, such
        # as the library's named reference dose. They are data-only too.
        nodes[a.name] = a.expr
    end
    for d in plan.derived
        nodes[d.name] = d.expr
    end
    function indices(ex)
        ex isa Expr || return nothing
        if ex.head === :ref && length(ex.args) >= 2 && ex.args[2] isa Symbol
            push!(required, ex.args[2])
        end
        foreach(indices, ex.args)
        return nothing
    end
    foreach(indices, values(nodes))
    resps = Set{Symbol}(r.response for r in plan.responses)
    known = _all_names(plan)
    memo = Dict{Symbol,Bool}()
    function dataonly(nm, active)
        haskey(memo, nm) && return memo[nm]
        nm in active && return false
        push!(active, nm)
        ok = all(s -> s in raw || (unbound && s ∉ known) ||
                (haskey(nodes, s) && dataonly(s, active)),
            _expr_value_symbols(nodes[nm]))
        delete!(active, nm)
        memo[nm] = ok
        return ok
    end
    axes = Set{Symbol}()
    function gather_indices(ex)
        ex isa Expr || return
        if ex.head === :ref && length(ex.args) >= 2 &&
                ex.args[1] isa Symbol &&
                (_is_array_param(plan, ex.args[1]) ||
                    _is_array_assignment(plan, ex.args[1]))
            for idx in ex.args[2:end]
                idx isa Symbol && push!(axes, idx)
            end
        end
        foreach(gather_indices, ex.args)
    end
    foreach(gather_indices, values(nodes))
    for p in plan.array_parameters, dim in p.dims
        dim isa Expr && dim.head === :call && dim.args[1] === :levels &&
            dim.args[2] isa Symbol && push!(axes, dim.args[2])
    end
    union!(required, axes)
    names = Set{Symbol}(nm for (nm, ex) in nodes
        if nm ∉ resps && (_contains_module_call(ex) || nm in required) &&
            dataonly(nm, Set{Symbol}()))
    bind_only || return names
    onlydata = union(raw, Set{Symbol}(nm for nm in keys(nodes)
        if dataonly(nm, Set{Symbol}())))
    return setdiff(names, _preparation_data_names(plan, nodes, raw, onlydata, names))
end

# Whole-value data definitions consumed by a parameter-dependent module
# call belong to preparation, whether lowering inlined them or retained
# them as assignments (notably beside declared arrays). They may return
# arbitrary Julia values; the generated data-only statements fold once.
# Keep every dependency needed by bind-time consumers at bind instead.
function _preparation_data_names(plan, nodes, raw, onlydata, names)
    isempty(names) && return Set{Symbol}()
    _, wholedefs = _model_level_inputs(plan, raw)
    function dependencies!(found, nm; data_only = false)
        nm in found && return
        haskey(nodes, nm) || return
        data_only && nm ∉ onlydata && return
        push!(found, nm)
        for dep in _expr_value_symbols(nodes[nm])
            dependencies!(found, dep; data_only)
        end
    end
    runtime = Set{Symbol}()
    for (nm, ex) in nodes
        nm ∈ onlydata && continue
        _contains_module_call(ex) || continue
        dependencies!(runtime, nm)
    end
    prepared = intersect(names, wholedefs, runtime)
    isempty(prepared) && return prepared

    # Any other structural slot can require a value during binding (prior
    # arguments, array dimensions, responses, matrices, ...). Follow its
    # dependencies as well as those of all remaining bind-time calls, so
    # a shared helper is never evaluated both at bind and at preparation.
    free = Set{Symbol}(keys(nodes))
    for f in fieldnames(StructuralPlan)
        f in (:assignments, :derived, :columns, :submodel_scopes) && continue
        _drop_held_names!(free, getfield(plan, f))
    end
    # Whole-value classification exempts `axes(M, d)` from observation
    # alignment, but resolving an array's shape still needs M at bind.
    for p in plan.array_parameters, d in p.dims
        _drop_held_names!(free, d)
    end
    needed = Set{Symbol}()
    for nm in union(setdiff(Set{Symbol}(keys(nodes)), free), setdiff(names, prepared))
        dependencies!(needed, nm; data_only = true)
    end
    return setdiff!(prepared, needed)
end

function _module_data_names(plan::StructuralPlan; unbound::Bool=false)
    nodes = union(Set{Symbol}(a.name for a in plan.assignments),
        Set{Symbol}(d.name for d in plan.derived))
    raw = Set{Symbol}(k for k in keys(plan.columns) if k ∉ nodes)
    return _module_data_names(plan, raw; unbound)
end

"""In a bound plan: the module data definitions bound as columns."""
function _bound_module_data_names(plan::StructuralPlan)
    return Set{Symbol}(n for n in _module_data_names(plan)
        if haskey(plan.columns, n))
end

"""Model-level data inputs of `plan`: raw columns of `raw` read only as
whole values. A read is whole inside an argument of an undotted module
call (`f(gx)`, at any depth), as the gathered value of a gather
(`gx[g]`, `(b .* gx)[g]`), or anywhere in a definition whose every use is
whole (`scale = a .+ b .* gx` passed only to calls) — so naming a
subexpression never changes the verdict. Such a column has no observation
axis — any length or shape, like the model-level values computed from
it — and it neither anchors nor crosses `n_obs`. Any other read keeps a
column observation-aligned: a definition read outside those positions
(`gx .+ 1` in a predictor, `f.(gx)`, `mean(gx)`, a gather index), or any
name an observation-bearing structural slot holds (new slots remain
fail-closed). Prior-only reads keep their declared latent domain.
Returns `(inputs, defs)`: the columns, and the
whole-context definitions (whose values may have any length too)."""
function _model_level_inputs(plan::StructuralPlan, raw::AbstractSet{Symbol})
    # A column with no value consumer has no observation axis. Infer this
    # from the finished plan, where optimized matrix/predictor reads are
    # represented too, rather than from an intermediate definition table.
    unused = Set{Symbol}(raw)
    for f in fieldnames(StructuralPlan)
        f in (:columns, :n_obs, :roles, :submodel_scopes) && continue
        _drop_held_names!(unused, getfield(plan, f))
    end
    defs = Pair{Symbol,Any}[a.name => a.expr for a in plan.assignments]
    append!(defs, Pair{Symbol,Any}[d.name => d.expr for d in plan.derived])
    # A value matrix is a data-only definition, even when its hcat recipe
    # lives in the matrix table. Its consumers determine its observation
    # axis; construction reads the source columns whole. Keep this edge in
    # the same dependency analysis as ordinary data-only definitions.
    value_matrices = _value_design_matrix_names(plan)
    for m in plan.matrices
        m.name in value_matrices || continue
        push!(defs, m.name => Expr(:call, GlobalRef(Base, :hcat),
            (c for c in m.columns if c !== nothing)...))
    end
    # Scalar, declared-array and latent-plate prior arguments are model values.
    # Propagate that context through their definitions just as for a
    # module-call argument; their raw data have no observation axis.
    for ps in (plan.parameters, plan.array_parameters, plan.plate_parameters), p in ps
        push!(defs, Symbol(:_ppl_prior_input_, p.name) =>
            Expr(:tuple, values(p.args)..., _support_args(p.support_override)...))
        p.family === :external && p.name in plan.conditioned && push!(defs,
            Symbol(:_ppl_external_value_, p.name) =>
                Expr(:call, GlobalRef(Base, :identity), p.name))
    end
    # Open densities receive ordinary whole values. Their explicit broadcast
    # has its own Julia axes, independent of a legacy response's row axis.
    for p in plan.external_observations
        push!(defs, Symbol(:_ppl_external_value_, p.name) =>
            Expr(:tuple, (Expr(:call, GlobalRef(Base, :identity), value)
                for value in (p.name, p.args.rhs))...))
    end
    # An array read only for a declared axis is a whole preparation input.
    # Its row count does not become an observation count. In contrast, a
    # grouping axis retains its existing alignment and level-code contract.
    for p in plan.array_parameters
        axes = filter(_is_axis_dim, p.dims)
        isempty(axes) || push!(defs, Symbol(:_ppl_axis_input_, p.name) =>
            Expr(:tuple, axes...))
    end
    # A latent plate's authored iterator likewise consumes its source as a
    # whole array. It sizes that latent block, independently of responses.
    for p in plan.plate_parameters
        p.range isa Union{Symbol,Expr} || continue
        push!(defs, Symbol(:_ppl_axis_input_, p.name) => p.range)
    end
    for p in plan.array_parameters, (i, d) in enumerate(p.dims)
        _is_levels_dim(d) || continue
        push!(defs, Symbol(:_ppl_axis_input_, p.name, :_, i) =>
            Expr(:call, GlobalRef(Base, :identity), d.args[2]))
    end
    for p in plan.vector_parameters
        push!(defs, Symbol(:_ppl_prior_input_, p.name) =>
            Expr(:tuple, values(p.args)...))
        p.extent_expr === nothing || push!(defs, Symbol(:_ppl_extent_input_, p.name) =>
            p.extent_expr)
    end
    for s in plan.scans
        exprs = Any[s.hi]
        for st in (s.setup..., s.step...)
            append!(exprs, st.kind === :sample ? st.args : (st.expr,))
        end
        push!(defs, Symbol(:_ppl_scan_input_, s.label) => Expr(:tuple, exprs...))
    end
    for r in plan.responses
        r.family === MixtureFam && r.mixture_weights isa Symbol || continue
        push!(defs, Symbol(:_ppl_weights_input_, r.label) =>
            Expr(:call, GlobalRef(Base, :identity), r.mixture_weights))
    end
    _has_observation_axis(plan) || return Set{Symbol}(raw), Set{Symbol}(first.(defs))

    # Whole-context definitions can exist without raw data dependencies
    # (for example collected literal indices selecting a sampled vector).
    # Their actual consumers still determine the context below.
    free = union(Set{Symbol}(raw), Set{Symbol}(first(d) for d in defs))
    named = copy(free)
    for f in fieldnames(StructuralPlan)
        f in (:assignments, :derived, :columns, :n_obs, :roles,
            :submodel_scopes) && continue
        if f === :parameters
            for p in plan.parameters
                p.family === :external && p.name in plan.conditioned && continue
                delete!(free, p.name)
            end
        elseif f === :external_observations
            continue
        elseif f === :vector_parameters
            for p in plan.vector_parameters
                delete!(free, p.name)
            end
        elseif f === :scans
            # Raw scan reads are model values indexed inside the retained loop.
            # Other uses (response columns, affine columns) still pin their axis.
            continue
        elseif f === :plate_parameters
            for p in plan.plate_parameters
                # The iterator and prior inputs belong to this latent domain.
                # Other uses, including likelihood reads, still pin their
                # usual observation alignment in the same dependency analysis.
                delete!(free, p.name)
            end
        elseif f === :array_parameters
            for p in plan.array_parameters
                delete!(free, p.name)
                _drop_held_names!(free, filter(d -> !_is_axis_dim(d) && !_is_levels_dim(d), p.dims))
            end
        elseif f === :responses
            for r in plan.responses
                _drop_held_names!(free, r.family === MixtureFam ?
                    _with(r; mixture_weights = nothing) : r)
            end
        elseif f === :matrices
            # Affine/GLM matrices retain their row contract. Value-matrix
            # recipes are definitions above, not observation consumers.
            for m in plan.matrices
                m.name in value_matrices || _drop_held_names!(free, m)
            end
        else
            _drop_held_names!(free, getfield(plan, f))
        end
    end
    held = setdiff!(named, free)
    inputs, ctx = _whole_value_reads(defs, raw, held)
    return union!(setdiff!(inputs, held), unused), ctx
end

"""Whole-value reads over `defs` (`name => expr`): the columns of `raw`
read only as whole values, and the whole-context definitions — every use
a whole read or inside another whole-context definition (the greatest
fixpoint), none of them in `pinned`."""
function _whole_value_reads(defs, raw::AbstractSet{Symbol},
        pinned::AbstractSet{Symbol})
    names = Set{Symbol}(first(d) for d in defs)
    known = union(Set{Symbol}(raw), names)
    reads = Dict{Symbol,Tuple{Set{Symbol},Set{Symbol}}}()
    for (nm, ex) in defs
        w, p = Set{Symbol}(), Set{Symbol}()
        _classify_reads!(w, p, ex, known, false)
        reads[nm] = (w, p)
    end
    ctx = setdiff(names, pinned)
    changed = true
    while changed
        changed = false
        for (nm, (_, p)) in reads
            nm in ctx && continue
            for d in p
                d in ctx || continue
                delete!(ctx, d)
                changed = true
            end
        end
    end
    inputs = Set{Symbol}()
    aligned = Set{Symbol}()
    for (nm, (w, p)) in reads
        union!(inputs, intersect(w, raw))
        union!(nm in ctx ? inputs : aligned, intersect(p, raw))
    end
    return setdiff!(inputs, aligned), ctx
end

"""In a bound plan: `(inputs, defs)` of [`_model_level_inputs`](@ref)."""
function _bound_model_level_inputs(plan::StructuralPlan)
    nodes = union(Set{Symbol}(a.name for a in plan.assignments),
        Set{Symbol}(d.name for d in plan.derived))
    inputs, definitions = _model_level_inputs(plan,
        Set{Symbol}(k for k in keys(plan.columns) if k ∉ nodes))
    return union(inputs, Set(_conditioned_input(n) for n in plan.conditioned)), definitions
end

_conditioned_input(name::Symbol) = Symbol("_rkppl_conditioned_", name)
_has_observation_axis(plan) = !isempty(plan.responses) || !isempty(plan.kernel_plates) ||
    any(p -> p.range === nothing, plan.plate_parameters) ||
    !isempty(plan.scans) || !isempty(plan.dar_paths)

# Reads of `raw` names in `ex` (the reads `_expr_value_symbols` sees):
# inside an argument of an undotted module call into `whole`, anywhere
# else into `per_obs`.
function _classify_reads!(whole::Set{Symbol}, per_obs::Set{Symbol}, ex,
        raw::AbstractSet{Symbol}, inside::Bool; reductions_whole::Bool=false)
    visit(a, context) = _classify_reads!(whole, per_obs, a, raw, context;
        reductions_whole)
    if ex isa Symbol
        ex in raw && push!(inside ? whole : per_obs, ex)
    elseif ex isa Expr
        if _is_plate_column_call(ex) && _plate_column_axis(ex) !== nothing
            axis = _plate_column_axis(ex)
            for a in ex.args[3:end]
                visit(a, a !== axis)
            end
        elseif ex.head === :call && length(ex.args) == 3 &&
                ex.args[1] === :_ppl_codes
            visit(ex.args[2], inside)
            visit(ex.args[3], true)
        elseif _is_plate_column_expr(ex)
            for a in ex.args[1].args[2:end]
                shared = _is_ref_call(a)
                visit(shared ? a.args[2] : a, inside || shared)
            end
        elseif ex.head === :kw && length(ex.args) == 2
            visit(ex.args[2], inside)
        elseif ex.head === :call
            inner = inside || (!isempty(ex.args) && (ex.args[1] isa GlobalRef ||
                reductions_whole && ex.args[1] in REDUCTION_FNS))
            for a in ex.args[2:end]
                visit(a, inner)
            end
        elseif _is_dotted_call(ex)
            for a in ex.args[2].args
                visit(a, inside)
            end
        elseif ex.head === :ref && length(ex.args) == 2
            # A gather `v[c]` takes the gathered value whole; its index
            # keeps the surrounding context.
            visit(ex.args[1], true)
            visit(ex.args[2], inside)
        else
            for a in ex.args
                visit(a, inside)
            end
        end
    end
    return nothing
end

# Drop from `out` every Symbol `x` holds, at any depth.
_drop_held_names!(out::Set{Symbol}, x::Symbol) = (delete!(out, x); nothing)
# A `GlobalRef` names a module function (a composed tree's dotted map),
# never a model name; its binding is cyclic, so the struct walk below
# must not enter it.
_drop_held_names!(::Set{Symbol},
    ::Union{Number,AbstractString,Function,Module,Type,GlobalRef,
        AbstractArray{<:Number}}) =
    nothing
function _drop_held_names!(out::Set{Symbol},
        x::Union{AbstractArray,Tuple,NamedTuple,AbstractSet})
    for v in x
        isempty(out) && return nothing
        _drop_held_names!(out, v)
    end
    return nothing
end
function _drop_held_names!(out::Set{Symbol}, x::AbstractDict)
    for (k, v) in x
        isempty(out) && return nothing
        _drop_held_names!(out, k)
        _drop_held_names!(out, v)
    end
    return nothing
end
function _drop_held_names!(out::Set{Symbol}, x::Expr)
    # A code vector aligns the first column to a whole label pool. The
    # pool's length is a parameter axis, never an observation row count.
    if x.head === :call && length(x.args) == 3 && x.args[1] === :_ppl_codes
        _drop_held_names!(out, x.args[2])
        return nothing
    end
    for a in x.args
        x.head === :tuple && (a = _tuple_field_value(a))
        _drop_held_names!(out, a)
    end
    return nothing
end
_drop_held_names!(::Set{Symbol}, ::SubmodelScope) = nothing
# Sizing an innovation by a matrix axis reads its shape, not one value
# per observation. Other array fields still pin their ordinary readers.
function _drop_held_names!(out::Set{Symbol}, p::ArrayParameter)
    for f in fieldnames(ArrayParameter)
        if f === :dims
            for d in p.dims
                (_is_axis_dim(d) || _is_levels_dim(d)) || _drop_held_names!(out, d)
            end
        else
            _drop_held_names!(out, getfield(p, f))
        end
    end
    return nothing
end
function _drop_held_names!(out::Set{Symbol}, x::T) where {T}
    isstructtype(T) || return nothing
    for f in fieldnames(T)
        isempty(out) && return nothing
        isdefined(x, f) && _drop_held_names!(out, getfield(x, f))
    end
    return nothing
end

# The `_bound_value` inputs a plan reads, wherever lowering moved their
# definitions (a definition used only in a predictor is inlined there).
function _bound_value_inputs(plan::StructuralPlan)
    found = Set{Symbol}()
    seen = Base.IdSet{Any}()
    for f in fieldnames(StructuralPlan)
        f in (:columns, :submodel_scopes) && continue
        _collect_bound_value_inputs!(found, getfield(plan, f), seen)
    end
    return found
end
function _collect_bound_value_inputs!(found, x::Expr, seen)
    if _is_bound_value_call(x) && x.args[2] isa Symbol
        push!(found, x.args[2])
    end
    for a in x.args
        _collect_bound_value_inputs!(found, a, seen)
    end
    return nothing
end
function _collect_bound_value_inputs!(found,
        x::Union{AbstractArray,Tuple,NamedTuple,AbstractSet,AbstractDict},
        seen)
    eltype(x) <: Union{Number,Symbol,AbstractString} && return nothing
    x in seen && return nothing
    push!(seen, x)
    for v in (x isa AbstractDict ? values(x) : x)
        _collect_bound_value_inputs!(found, v, seen)
    end
    return nothing
end
# Plan records (specs, terms, priors): walk their fields.
const _PLAN_RECORD_MODULE = @__MODULE__
function _collect_bound_value_inputs!(found, x::T, seen) where {T}
    (isstructtype(T) && parentmodule(T) === _PLAN_RECORD_MODULE) ||
        return nothing
    x in seen && return nothing
    push!(seen, x)
    for f in fieldnames(T)
        isdefined(x, f) &&
            _collect_bound_value_inputs!(found, getfield(x, f), seen)
    end
    return nothing
end

# Route each caller value lowered as a scalar definition to its input name
# (the caller passes `s`; the plan reads `_rkppl_value_s`).
function _route_bound_values!(plan::StructuralPlan,
        columns::Dict{Symbol,ColumnData})
    for input in _bound_value_inputs(plan)
        haskey(columns, input) && continue
        name = Symbol(chopprefix(String(input), "_rkppl_value_"))
        haskey(columns, name) || throw(ContractValidationError(
            "[bind] data value $name is missing (the model reads it as a " *
            "model-level value)"))
        columns[input] = pop!(columns, name)
    end
    return columns
end

# A response bound to a number is one observation (`y .~ Normal.(mu, s)`
# with a scalar `y`, as in Julia): it binds as a one-entry vector.
function _scalar_responses_as_observations!(plan::StructuralPlan,
        columns::Dict{Symbol,ColumnData})
    for r in plan.responses, nm in (r.response, r.count_columns...,
            r.extra_responses...)
        v = get(columns, nm, nothing)
        v isa Number && (columns[nm] = [v])
    end
    return columns
end

function _materialize_module_data!(plan::StructuralPlan,
        columns::Dict{Symbol,ColumnData}; already = Set{Symbol}())
    names = setdiff(_module_data_names(plan, Set{Symbol}(keys(columns))), already)
    isempty(names) && return names
    for nm in sort!(collect(names))
        haskey(columns, nm) && throw(ContractValidationError(
            "[bind] column $nm is computed by the model (`$nm = ...` calls " *
            "a module function on data) — drop it from bind_data"))
    end
    names = setdiff(_module_data_names(plan, Set{Symbol}(keys(columns));
        bind_only = true), already)
    isempty(names) && return names
    exprs = Dict{Symbol,Any}(a.name => a.expr for a in plan.assignments)
    for d in plan.derived
        exprs[d.name] = d.expr
    end
    vector_defs = Set{Symbol}(d.name for d in plan.derived)
    memo = Dict{Symbol,Any}()
    # Identical module calls share one evaluation (`(Xf, Zp) = f(x)` reads
    # `f(x)` twice): within one bind every name has one value.
    calls = Dict{Any,Any}()
    function lookup(nm::Symbol)
        haskey(memo, nm) && return memo[nm]
        haskey(columns, nm) && return columns[nm]
        haskey(exprs, nm) || throw(ContractValidationError(
            "[bind] data definition reads $nm, which is not bound data"))
        memo[nm] = _eval_value_expr(exprs[nm], lookup, nm; calls)
        return memo[nm]
    end
    for nm in sort!(collect(names))
        v = try
            lookup(nm)
        catch e
            e isa ContractValidationError && rethrow()
            throw(ContractValidationError(
                "[bind] data definition $nm failed to evaluate: " *
                sprint(showerror, e)))
        end
        if nm in vector_defs
            v isa Number || v isa AbstractArray ||
                throw(ContractValidationError(
                "[bind] data definition $nm is read per observation (an " *
                "operand with Julia broadcast axes) but evaluated to $(summary(v))"))
        else
            # Other Julia values stay graph assignments, where preparation
            # can fold them without imposing a column type.
            v isa ColumnData || continue
        end
        # Dense storage retains every observation operand's rank and axes.
        columns[nm] = v isa AbstractVector ? collect(v) :
            v isa AbstractArray ? Array(v) : v
    end
    return names
end

# Value matrices (`_value_design_matrix_names`): each binds under
# its name as `Float64.(hcat(...))` of its bound columns, a ones column at
# the intercept `ones(length(x))` (the design-matrix meaning).
function _materialize_value_matrices!(plan::StructuralPlan,
        columns::Dict{Symbol,ColumnData})
    names = _value_design_matrix_names(plan)
    isempty(names) && return names
    for m in plan.matrices
        m.name in names || continue
        haskey(columns, m.name) && throw(ContractValidationError(
            "[bind] column $(m.name) is computed by the model " *
            "(`$(m.name) = hcat(...)`) — drop it from bind_data"))
        datacols = Symbol[c for c in m.columns if c !== nothing]
        isempty(datacols) && throw(ContractValidationError(
            "[bind] matrix $(m.name) has no data column to size its " *
            "intercept"))
        for c in datacols
            haskey(columns, c) || throw(ContractValidationError(
                "[bind] matrix $(m.name) reads column $c, which is not " *
                "bound"))
            columns[c] isa AbstractVector || throw(ContractValidationError(
                "[bind] matrix $(m.name): column $c must be a vector, got " *
                "$(summary(columns[c]))"))
        end
        n = length(columns[first(datacols)])
        parts = [c === nothing ? ones(n) : columns[c] for c in m.columns]
        columns[m.name] = try
            Float64.(hcat(parts...))
        catch e
            throw(ContractValidationError("[bind] matrix $(m.name) = " *
                "hcat(...) failed: " * sprint(showerror, e)))
        end
    end
    return names
end

# Plain-Julia evaluation of a resolved definition expression (functions as
# values): module calls through their `GlobalRef`s, built-in vocabulary
# heads through the generated-model scope — the bindings the kernel uses.
function _eval_value_expr(ex, lookup, label; calls = nothing)
    ex isa Union{Number,String} && return ex
    ex isa QuoteNode && return ex.value
    ex isa GlobalRef && return getglobal(ex.mod, ex.name)
    ex isa Symbol && return lookup(ex)
    ex isa Expr || throw(ContractValidationError(
        "[bind] $label: unsupported literal $(repr(ex))"))
    ev(a) = _eval_value_expr(a, lookup, label; calls)
    h = ex.head
    cached = h === :call && calls !== nothing && ex.args[1] isa GlobalRef
    cached && haskey(calls, ex) && return calls[ex]
    if h === :call
        f = _eval_callee(ex.args[1], label)
        pos = Any[]
        kws = Pair{Symbol,Any}[]
        for a in ex.args[2:end]
            if a isa Expr && a.head === :parameters
                for p in a.args
                    if p isa Expr && p.head === :kw
                        push!(kws, p.args[1] => ev(p.args[2]))
                    elseif p isa Symbol
                        push!(kws, p => lookup(p))
                    else
                        throw(ContractValidationError(
                            "[bind] $label: unsupported keyword form $(repr(p))"))
                    end
                end
            elseif a isa Expr && a.head === :kw
                push!(kws, a.args[1] => ev(a.args[2]))
            else
                push!(pos, ev(a))
            end
        end
        # `invokelatest`: the model module's functions may postdate the
        # caller's world (bind_data is callable from any world).
        v = Base.invokelatest(f, pos...; kws...)
        cached && (calls[ex] = v)
        return v
    elseif h === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        f = _eval_callee(ex.args[1], label)
        return Base.invokelatest(broadcast, f, map(ev, ex.args[2].args)...)
    elseif h === :ref
        return getindex(ev(ex.args[1]), map(ev, ex.args[2:end])...)
    elseif h === :vect
        return Base.vect(map(ev, ex.args)...)
    elseif h === :tuple
        return Tuple(map(ev, ex.args))
    end
    throw(ContractValidationError(
        "[bind] $label: unsupported expression head $h in a data definition"))
end

function _eval_callee(fn, label)
    fn isa GlobalRef && return getglobal(fn.mod, fn.name)
    fn isa Symbol || throw(ContractValidationError(
        "[bind] $label: unsupported call head $(repr(fn))"))
    s = string(fn)
    if startswith(s, ".") && length(s) > 1 && fn !== :.
        op = Symbol(s[2:end])
        isdefined(PPLGeneratedModels, op) || throw(ContractValidationError(
            "[bind] $label: unknown operator $fn"))
        g = getglobal(PPLGeneratedModels, op)
        return (args...) -> broadcast(g, args...)
    end
    # The generated scope binds `logistic` to the distribution kernel; the
    # value vocabulary's `logistic` is the inverse-logit math function.
    fn === :logistic && return PPLGeneratedModels._ppl_logistic
    isdefined(PPLGeneratedModels, fn) || throw(ContractValidationError(
        "[bind] $label: unknown function $fn"))
    return getglobal(PPLGeneratedModels, fn)
end

function _materialize_derived_responses!(plan::StructuralPlan,
        columns::Dict{Symbol,ColumnData})
    derived =
        [r.response for r in plan.responses if _is_derived(plan, r.response)]
    isempty(derived) && return columns
    det_exprs = Dict{Symbol,Any}(a.name => a.expr for a in plan.assignments)
    for d in plan.derived
        det_exprs[d.name] = d.expr
    end
    for name in derived
        haskey(columns, name) && throw(ContractValidationError(
            "[bind] column $name is derived in the model — drop it from " *
            "bind_data (derived responses materialize from bound data)"))
        try
            value = _eval_derived_name(plan, name, det_exprs, columns,
                Dict{Symbol,Any}(), name)
            # A gather from a partly-missing array retains its union
            # eltype even when every selected observation is present.
            # Narrow that representation without dropping any value;
            # selected missing values still fail response validation.
            if value isa AbstractArray && Missing <: eltype(value) &&
                    Base.nonmissingtype(eltype(value)) <: Real &&
                    !any(ismissing, value)
                value = Base.nonmissingtype(eltype(value)).(value)
            end
            columns[name] = value
        catch e
            e isa ContractValidationError && rethrow()
            throw(ContractValidationError(
                "[bind] derived response $name failed to evaluate: " *
                sprint(showerror, e)))
        end
    end
    return columns
end

# Host evaluator for derived-response expressions: bound columns and
# (transitive) deterministic definitions resolve; sampled, latent,
# coefficient, and unknown names fail naming the response. Forms mirror
# the contract walkers (`_collect_vector_refs!` /
# `_collect_assignment_refs!`), which prove them before bind — anything
# else fails closed here too.
const _DERIVED_DOTTED_OPS = Dict{Symbol,Function}(
    :.+ => +, :.- => -, :.* => *, :./ => /, :.^ => ^, :.% => %,
    :.== => ==, :.!= => !=, :.< => <, :.> => >, :.<= => <=, :.>= => >=)
const _DERIVED_MATH_FNS = Dict{Symbol,Function}(
    :log => log, :log10 => log10, :log1p => log1p, :exp => exp,
    :expm1 => expm1, :sqrt => sqrt, :abs => abs)
const _DERIVED_SCALAR_FNS = Dict{Symbol,Function}(
    :+ => +, :- => -, :* => *, :/ => /, :^ => ^,
    :log => log, :log10 => log10, :log1p => log1p, :exp => exp,
    :expm1 => expm1, :sqrt => sqrt, :abs => abs, :tanh => tanh,
    :sum => sum, :mean => mean, :std => std, :var => var,
    :minimum => minimum, :maximum => maximum, :length => length)

function _eval_derived_name(plan::StructuralPlan, name::Symbol, det_exprs,
        columns, memo::Dict{Symbol,Any}, root::Symbol)
    haskey(memo, name) && return memo[name]
    haskey(columns, name) && return columns[name]
    haskey(det_exprs, name) || throw(ContractValidationError(
        "[bind] derived response $root reads $name which is not bound " *
        "data (responses derive from bound columns only)"))
    memo[name] = _eval_derived_node(
        det_exprs[name], plan, det_exprs, columns, memo, root)
    return memo[name]
end

function _eval_derived_node(ex, plan::StructuralPlan, det_exprs, columns,
        memo::Dict{Symbol,Any}, root::Symbol)
    ex isa Number && return ex
    ex isa Symbol &&
        return _eval_derived_name(plan, ex, det_exprs, columns, memo, root)
    ex isa Expr || throw(ContractValidationError(
        "[bind] derived response $root: unsupported literal $(repr(ex)) " *
        "(numeric literals only)"))
    if _contains_module_call(ex)
        # Functions as values: a module call (anywhere below) evaluates as
        # plain Julia over the resolved inputs.
        return _eval_value_expr(ex, nm -> _eval_derived_name(plan, nm,
            det_exprs, columns, memo, root), root)
    end
    if ex.head === :ref
        vals = [_eval_derived_node(a, plan, det_exprs, columns, memo, root)
            for a in ex.args]
        return getindex(vals...)
    end
    if ex.head === :call
        fn = ex.args[1]
        if fn isa Symbol && haskey(_DERIVED_DOTTED_OPS, fn)
            f = _DERIVED_DOTTED_OPS[fn]
            vals = [_eval_derived_node(a, plan, det_exprs, columns, memo, root)
                for a in ex.args[2:end]]
            return broadcast(f, vals...)
        end
        if fn isa Symbol && haskey(_DERIVED_SCALAR_FNS, fn)
            f = _DERIVED_SCALAR_FNS[fn]
            vals = [_eval_derived_node(a, plan, det_exprs, columns, memo, root)
                for a in ex.args[2:end]]
            return f(vals...)
        end
        return throw(ContractValidationError(
            "[bind] derived response $root: `$fn` is not admitted in a " *
            "derived response (elementwise operators, elementwise math, " *
            "`ifelse`, and scalar/reduction calls only)"))
    end
    if ex.head === :.
        # `f.(x)` / `ifelse.(c, x, y)` — the contract walker proves the
        # `f.(tuple)` shape before bind.
        f = ex.args[1]
        args = ex.args[2].args
        if f === :ifelse
            vals = [_eval_derived_node(a, plan, det_exprs, columns, memo, root)
                for a in args]
            return ifelse.(vals...)
        end
        f isa Symbol && haskey(_DERIVED_MATH_FNS, f) && length(args) == 1 ||
            throw(ContractValidationError(
                "[bind] derived response $root: `$(repr(ex))` is not " *
                "admitted in a derived response (single-argument " *
                "elementwise math only)"))
        return _DERIVED_MATH_FNS[f].(
            _eval_derived_node(args[1], plan, det_exprs, columns, memo, root))
    end
    return throw(ContractValidationError(
        "[bind] derived response $root: unsupported expression head " *
        "$(ex.head) in a derived response"))
end

"""
    bind_data(plan, columns; roles=Dict(), dims=Dict(), conditioned=plan.conditioned) -> StructuralPlan

`columns` accepts a symbol-keyed `AbstractDict` or a `NamedTuple` of data
values, as does value-based `lower_rkppl`. Named tuples use the same binding
and validation path as dictionaries; neither container nor its values is mutated.
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
be consumed. Data values are numbers or arrays of any shape; observation
values follow their authored broadcast axes and structured operations retain
their own shape requirements. Data-only definitions that call a module
function (functions as values) are evaluated here, once, and bound under
their own names — never supplied by the caller; a model-level one may be a
number or an array of any shape and carries no `n_obs` requirement. So does a model-level
data input: a column every definition reads only inside an argument of an
undotted module call (`gx_m = f(gx)`, or inlined `(b .* f(gx))[g]`) and no
response, predictor or other slot names — it may have any length or shape
and never sets `n_obs` (see `_model_level_inputs`).

Binding resolves module values and calls in a package-owned latest-world
scope, including definitions just evaluated in a builder.
"""
bind_data(plan::StructuralPlan, columns::NamedTuple; kwargs...) =
    bind_data(plan, Dict{Symbol,Any}(pairs(columns)); kwargs...)

function bind_data(plan::StructuralPlan, columns::AbstractDict{Symbol};
        roles::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
        dims::AbstractDict{Symbol,<:Integer} = Dict{Symbol,Int}(),
        conditioned = plan.conditioned)
    # Resolve the callee binding in the same scope as its evaluation: on
    # Julia 1.12, a call-only invokelatest cannot see a just-created binding
    # while evaluating its function argument in the caller's older world.
    return Base.invokelatest(_bind_data_latest, plan, columns; roles, dims, conditioned)
end

function _bind_data_latest(plan::StructuralPlan, columns::AbstractDict{Symbol};
        roles::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
        dims::AbstractDict{Symbol,<:Integer} = Dict{Symbol,Int}(),
        conditioned = plan.conditioned)
    sampled = Set(p.name for ps in (plan.parameters, plan.array_parameters,
        plan.vector_parameters, plan.plate_parameters) for p in ps)
    observing = intersect(sampled, Set{Symbol}(_conditioned_names(conditioned)))
    plan = _with(plan; conditioned = union(plan.conditioned, observing))
    validate_structure(plan)
    isempty(columns) && _has_observation_axis(plan) && throw(ContractValidationError(
        "[bind] bind_data requires non-empty columns"))
    columns = _checked_columns(columns)
    _route_conditioned_values!(plan, columns)
    _route_bound_values!(plan, columns)
    _scalar_responses_as_observations!(plan, columns)
    # Value matrices bind next, as data: every later step reads them
    # exactly like a caller-supplied matrix.
    _materialize_value_matrices!(plan, columns)
    raw = Set{Symbol}(keys(columns))
    computed = _materialize_module_data!(plan, columns)
    # Raw inputs read only as whole values (module-call arguments,
    # gathered values): no n_obs anchor.
    inputs, _ = _model_level_inputs(plan, raw)
    union!(inputs, (_conditioned_input(n) for n in plan.conditioned))
    _materialize_derived_responses!(plan, columns)
    bases = _materialize_splines!(plan, columns)
    hbases = _fit_hsgp_bases(plan, columns)
    kbases = _resolve_kernels!(plan, columns, dims)
    # Schedule products now exist: fold definitions that consume them by
    # the same data-only evaluator, once, before filling parameter shapes.
    union!(computed, _materialize_module_data!(plan, columns; already = computed))
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
        if d.mm !== nothing
            for g in d.mm.groups
                haskey(inferred, g) && _upgrade_role!(inferred, g, :group)
            end
        end
        if d.strata !== nothing
            by = d.strata.by
            haskey(inferred, by) && _upgrade_role!(inferred, by, :group)
        end
    end
    for sb in bases, blk in sb.blocks, c in blk.columns
        haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
    end
    for hb in hbases, c in hb.axes
        haskey(inferred, c) && _upgrade_role!(inferred, c, :predictor)
    end
    for hb in hbases
        hb.by === nothing && continue
        haskey(inferred, hb.by.column) &&
            _upgrade_role!(inferred, hb.by.column, :group)
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
    # Kernel plans carry two column lengths by design: n_obs is the
    # TOTAL likelihood lanes across plates (panel: flat length — vector
    # models — or n_sub — all-scalar; grouped: primary-response
    # length) — never first-column. Otherwise n_obs is the first
    # NON-managed column's ROW count (length for vectors): mi packs
    # y_obs/Jobs short by design, and Dict order is not a crossing
    # contract (deriving from a packed column fails every full-length
    # column by order luck). Observation columns of several lengths are
    # several observation axes: n_obs is their total rows.
    axis_plan = _with(plan; columns = columns, spline_bases = bases,
        hsgp_bases = hbases, kernel_plates = kbases)
    n = if !_has_observation_axis(plan)
        1 # There is no observation axis; conditioned declarations have their own shapes.
    elseif isempty(kbases)
        axes = _observation_axes(axis_plan)
        axes === nothing ? _bind_nrows(columns,
            union(_mi_managed_columns(plan), computed, inputs)) : axes.total
    else
        axes = _observation_axes(axis_plan)
        sum(kp -> _kernel_plate_nlanes(kp, columns), kbases) +
            (axes === nothing ? 0 : axes.total)
    end
    maps = _eval_levelmaps(plan.levelmaps, columns)
    draws = _eval_draws_levels(plan.varying_draws, columns)
    responses2, vectors2 =
        _infer_leveled_sizes(plan.responses, plan.vector_parameters, columns,
            plan.predictors, plan.r2d2_priors, _with(plan; columns = columns, n_obs = n))
    bound = _with(plan; responses = responses2, columns = columns,
        n_obs = n, roles = merged, levelmaps = maps, varying_draws = draws,
        vector_parameters = vectors2, spline_bases = bases,
        hsgp_bases = hbases, kernel_plates = kbases)
    _validate_conditioned_values(bound)
    validate_data(bound)
    return bound
end

function _route_conditioned_values!(plan, columns)
    for name in plan.conditioned
        input = _conditioned_input(name)
        if haskey(columns, name)
            columns[input] = pop!(columns, name)
        end
        haskey(columns, input) || _fail(:condition, "missing observed value $name")
    end
    return columns
end

function _validate_conditioned_values(plan)
    known = Set{Symbol}()
    for p in plan.parameters
        p.name in plan.conditioned || continue
        push!(known, p.name)
        p.family === :external && continue
        plan.columns[_conditioned_input(p.name)] isa Number ||
            _fail(:condition, "$(p.name) is scalar; broadcast an observation vector with `.~`")
        if p.family === :binomial
            n = _layout_bound(plan,p.args.arg1)
            n isa Real && isfinite(n) && isinteger(n) && n >= 0 || _fail(p.label,
                "Binomial trials must be a nonnegative integer, got $(repr(n))")
            probability = p.args.arg2
            if probability isa Number || haskey(plan.columns,probability)
                value = probability isa Number ? probability : plan.columns[probability]
                value isa Real && 0 <= value <= 1 || _fail(p.label,
                    "Binomial probability must be a scalar in [0,1], got $(repr(value))")
            end
        end
    end
    for p in plan.array_parameters
        p.name in plan.conditioned || continue
        push!(known, p.name)
        value = plan.columns[_conditioned_input(p.name)]
        value isa AbstractArray && size(value) == Tuple(_array_dims(plan, p)) ||
            _fail(:condition, "$(p.name) needs shape $(Tuple(_array_dims(plan, p)))")
    end
    for p in plan.vector_parameters
        p.name in plan.conditioned || continue
        push!(known, p.name)
        value = plan.columns[_conditioned_input(p.name)]
        value isa AbstractVector && length(value) == p.size ||
            _fail(:condition, "$(p.name) needs a vector of length $(p.size)")
    end
    for p in plan.plate_parameters
        p.name in plan.conditioned || continue
        push!(known, p.name)
        value = plan.columns[_conditioned_input(p.name)]
        n = _plate_rows(plan, p)
        value isa AbstractVector && length(value) == n ||
            _fail(:condition, "$(p.name) needs a vector of length $n")
    end
    isempty(setdiff(plan.conditioned, known)) || _fail(:condition,
        "conditioned names must have sampling declarations")
    return nothing
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
function _dirichlet_size(plan, alpha, label)
    alpha isa AbstractVector && return length(alpha)
    alpha isa Expr && alpha.head === :vect && return length(alpha.args)
    _is_fill_call(alpha) && return alpha.args[3]
    plan === nothing && _fail(label, "Dirichlet concentration shape requires a bound plan")
    if alpha isa Symbol
        if haskey(plan.columns, alpha)
            value = plan.columns[alpha]
            value isa AbstractVector || _fail(label, "Dirichlet concentration $alpha must be a vector")
            isempty(value) && _fail(label, "Dirichlet concentration vector $alpha must be nonempty")
            all(x -> x isa Real && isfinite(x) && x > 0, value) || _fail(label,
                "Dirichlet concentrations in $alpha must be finite and strictly positive")
            return length(value)
        end
        definitions = (plan.assignments..., plan.derived...)
        i = findfirst(a -> a.name === alpha, definitions)
        i === nothing || return _dirichlet_size(plan, definitions[i].expr, label)
    end
    axes = _value_axes(plan, alpha; data_axes = true)
    axes !== nothing && length(axes) == 1 || _fail(label,
        "Dirichlet concentration must have a known vector shape; got $(repr(alpha))")
    return _array_dim_size(plan, Symbol(label), label, only(axes))
end

function _resolve_vector_extent(p::VectorParameter, plan, columns, responses)
    extent = p.size
    if extent === nothing && haskey(_VECTOR_ELEMENT_FAMILIES, p.family)
        reader = findfirst(r -> r.thresholds === p.name, responses)
        reader === nothing && return p
        name = responses[reader].response
        haskey(columns,name) || _fail(p.label,"vector $(p.name) length needs bound data $name")
        extent = length(DataAPI.levels(columns[name])) - 1
    elseif extent isa Expr
        definitions = Dict(a.name => a.expr for a in
            (plan === nothing ? () : (plan.assignments...,plan.derived...)))
        active = Set{Symbol}()
        function lookup(name)
            haskey(columns,name) && return columns[name]
            haskey(definitions,name) && name ∉ active || _fail(p.label,
                "vector $(p.name) length needs data-only value $name")
            push!(active,name)
            value = _eval_value_expr(resolve_levels(definitions[name]),lookup,p.label)
            delete!(active,name)
            return value
        end
        resolve_levels(ex) = ex isa Expr ? Expr(ex.head,
            (i == 1 && ex.head === :call && a === :levels ? GlobalRef(DataAPI,:levels) :
                resolve_levels(a) for (i,a) in enumerate(ex.args))...) : ex
        extent = try
            _eval_value_expr(resolve_levels(extent),lookup,p.label)
        catch e
            e isa ContractValidationError && rethrow()
            _fail(p.label,"vector $(p.name) length failed in Julia: " * sprint(showerror,e))
        end
    else
        return p
    end
    extent isa Integer && !(extent isa Bool) && extent >= 0 || _fail(p.label,
        "vector $(p.name) length must be a nonnegative integer, got $(repr(extent))")
    return _with(p;size=Int(extent))
end

function _infer_leveled_sizes(responses::Vector{LikelihoodSpec},
        vectors::Vector{VectorParameter}, columns::AbstractDict{Symbol},
        predictors::Vector{PredictorSpec} = PredictorSpec[],
        r2d2::Vector{R2D2Prior} = R2D2Prior[], plan = nothing)
    # Explicit extents are ordinary data-only Julia expressions. Resolve
    # them before responses read the vector's declared support.
    vectors = VectorParameter[_resolve_vector_extent(p, plan, columns, responses) for p in vectors]
    out_r = LikelihoodSpec[]
    for r in responses
        _is_leveled_family(r.family) || (push!(out_r, r); continue)
        # A stated threshold vector owns its extent, including categories
        # absent from these observations. An inferred extent still comes
        # from contiguous observed level codes.
        tp = r.thresholds === nothing ? nothing :
            findfirst(p -> p.name === r.thresholds, vectors)
        declared = _is_ordered_family(r.family) && tp !== nothing ?
            vectors[tp].size : nothing
        K = declared !== nothing ? declared + 1 : if r.evidence.kind !== :none && r.n_levels === nothing &&
                r.family !== CategoricalLogitFam
            # Clamp endpoints need not be category integers. Infer the law's
            # support from its explicit probability/threshold vector first.
            ref = r.family === CategoricalFam ? r.predictor : r.thresholds
            vi = findfirst(v -> v.name === ref, vectors)
            width = if vi !== nothing
                v = vectors[vi]
                v.family === :simplex_dirichlet ? _dirichlet_size(plan, v.args.arg1, v.label) : v.size
            elseif haskey(columns, ref)
                length(columns[ref])
            else
                nothing
            end
            width === nothing ? _infer_response_levels(r, columns) :
                width + (r.family === CategoricalFam ? 0 : 1)
        else
            _infer_response_levels(r, columns)
        end
        push!(out_r, _with_levels(r, K))
    end
    by_label = Dict{Symbol,LikelihoodSpec}(r.label => r for r in out_r)
    thresh_link = Dict{Symbol,Symbol}()
    simplex_link = Dict{Symbol,Symbol}()
    coefs_link = Dict{Symbol,Symbol}()
    mixture_link = Dict{Symbol,Symbol}()
    for r in out_r
        r.thresholds !== nothing && (thresh_link[r.thresholds] = r.label)
        r.threshold_coefs !== nothing && (coefs_link[r.threshold_coefs] = r.label)
        if _is_simplex_family(r.family)
            simplex_link[r.predictor] = r.label
        end
        if r.family === MixtureFam && r.mixture_weights isa Symbol
            mixture_link[r.mixture_weights] = r.label
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
        concentration_size = p.family === :simplex_dirichlet ?
            _dirichlet_size(plan, p.args.arg1, p.label) : nothing
        if haskey(thresh_link, p.name)
            K = by_label[thresh_link[p.name]].n_levels
            K === nothing && _fail(p.label, "internal: linked n_levels unresolved")
            want = K - 1
            p.size === nothing || p.size == want || _fail(p.label,
                "thresholds size $(p.size) disagrees with n_levels $K " *
                "(thresholds number K−1)")
            push!(out_v, _with(p; size = want))
        elseif haskey(coefs_link, p.name)
            lr = by_label[coefs_link[p.name]]
            K = lr.n_levels
            K === nothing && _fail(p.label, "internal: linked n_levels unresolved")
            want = (K - 1) * length(lr.threshold_columns)
            p.size === nothing || p.size == want || _fail(p.label,
                "threshold_coefs size $(p.size) disagrees with n_levels $K " *
                "× $(length(lr.threshold_columns)) columns (packs (K−1)×p = $want)")
            push!(out_v, _with(p; size = want))
        elseif haskey(simplex_link, p.name)
            K = by_label[simplex_link[p.name]].n_levels
            K === nothing && _fail(p.label, "internal: linked n_levels unresolved")
            p.size === nothing || p.size == K || _fail(p.label,
                "simplex size $(p.size) disagrees with n_levels $K")
            concentration_size == K || _fail(p.label,
                "Dirichlet concentration length $(concentration_size) " *
                "disagrees with n_levels $K")
            push!(out_v, _with(p; size = K))
        elseif haskey(mixture_link, p.name)
            K = length(by_label[mixture_link[p.name]].mixture_locs)
            K >= 1 || _fail(p.label, "internal: linked mixture unresolved")
            p.size === nothing || p.size == K || _fail(p.label,
                "mixture weights size $(p.size) disagrees with the " *
                "$K components")
            concentration_size == K || _fail(p.label,
                "Dirichlet concentration length $(concentration_size) " *
                "disagrees with the $K mixture components")
            push!(out_v, _with(p; size = K))
        elseif haskey(monotonic_link, p.name)
            want = concentration_size
            want >= 1 || _fail(p.label,
                "monotonic increments need ≥ 1 increment " *
                "(K=1 degenerates emitter-side and never reaches the thin layer)")
            p.size === nothing || p.size == want || _fail(p.label,
                "monotonic increments size $(p.size) disagrees with its " *
                "concentration length $want")
            push!(out_v, _with(p; size = want))
        elseif p.family in _JOINT_FACTOR_FAMILIES
            # Joint-factor sizes are structural (concrete at construction,
            # validated against the joint width) — bind passes them through.
            p.size === nothing && _fail(p.label,
                "internal: joint-factor size unresolved at bind")
            push!(out_v, p)
        elseif haskey(r2d2_link, p.name)
            want = concentration_size
            want >= 1 || _fail(p.label,
                "R2D2 shares need ≥ 1 share (an empty concentration " *
                "decomposes nothing)")
            p.size === nothing || p.size == want || _fail(p.label,
                "R2D2 phi size $(p.size) disagrees with its " *
                "concentration length $want")
            push!(out_v, _with(p; size = want))
        elseif haskey(_VECTOR_ELEMENT_FAMILIES, p.family) && p.size !== nothing
            # A plain vector with a concrete structural size (read whole
            # by definitions or unused) retains its declared extent.
            push!(out_v, p)
        elseif _is_ordered_parameter(p.family) && p.size !== nothing
            # A free-standing ordered vector (`c ~ Ordered(Normal(0, 1), 3)`
            # read as a value) has its literal structural size.
            push!(out_v, p)
        elseif p.family === :simplex_dirichlet
            # A free-standing simplex (functions as values: definitions
            # compute with it, `cumsum(vcat(0.0, zeta))`) sizes from its
            # literal concentration.
            want = concentration_size
            p.size === nothing || p.size == want || _fail(p.label,
                "simplex size $(p.size) disagrees with its concentration " *
                "length $want")
            push!(out_v, VectorParameter(p.name, p.family, p.args, want, p.label))
        else
            _fail(p.label, "vector parameter $(p.name) needs a declared extent " *
                "or a linked response from which to infer it")
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
    sort(unique(col)) == collect(1:K) || _fail(r.label,
        "an inferred leveled response must cover every level 1..$K " *
        "(declare ordinal cutpoints to state unobserved categories)")
    return K
end

# Rebuild a response with concrete n_levels (explicit values already
# asserted structurally; inference fills `nothing`).
function _with_levels(r::LikelihoodSpec, K::Int)
    r.n_levels === nothing || r.n_levels == K || _fail(r.label,
        "n_levels $(r.n_levels) disagrees with the bound data " *
        "(inferred K = $K)")
    return _with(r; n_levels = K)
end

# Field-preserving copy: rebuild `x` through its all-fields positional
# constructor, replacing only the named fields. A hand-written field list
# silently resets every field added after it was written; unknown override
# names fail loudly.
function _with(x::T; overrides...) where {T}
    names = fieldnames(T)
    for k in keys(overrides)
        k in names || throw(ArgumentError("$T has no field $k"))
    end
    return T((haskey(overrides, f) ? overrides[f] : getfield(x, f)
        for f in names)...)
end
