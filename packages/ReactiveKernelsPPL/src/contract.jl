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
# A latent plate iterating a value (not bound data) selects its prior
# inputs at `1:n`, where `n` is the latent's own cell count. Binding
# supplies `n` under this internal input (`_bind_plate_extents!`).
_plate_extent_input(name::Symbol) = Symbol("_rkppl_cells_", name)
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
"""
@enum TermKind::UInt8 begin
    InterceptTerm
    ContinuousTerm
    FactorTerm
    OffsetTerm
    LatentTerm
    ScanSummandTerm
    MatrixTerm
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


# Explicit support travels with a hyper prior, just as for a sampled prior.
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
`scans` sequential-recurrence latents, and
`matrices` user-bound design matrices referenced by [`MatrixTerm`](@ref)s
(empty for a plain population-GLM plan). `array_parameters` are the
declared array-valued parameters ([`ArrayParameter`](@ref)) the model reads
by name. `submodel_scopes` records lexical author paths separately from
the private identifiers used by the mathematical plan. `cell_broadcasts`
maps each response observed by a dotted `@plate` cell (`y[i] .~ D.(…)`) to
the names that cell reads per index; when the bound response holds one
array per index, each cell broadcasts over its own entries.
`named_values` keeps each authored alias the lowering absorbed addressable
by its name: `name => node` binds `name` to the graph value `node` (a
predictor, which carries its own name, or another value).
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
    vector_parameters::Vector{VectorParameter}
    matrices::Vector{DesignMatrix}
    array_parameters::Vector{ArrayParameter}
    submodel_scopes::Vector{SubmodelScope}
    conditioned::Set{Symbol}
    indexed_observations::Set{Symbol}
    external_observations::Vector{SampledParameter}
    cell_broadcasts::Dict{Symbol,Vector{Symbol}}
    named_values::Vector{Pair{Symbol,Symbol}}
end

# The former full constructor has no optimized authored names.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, vector_parameters,
    matrices,
    array_parameters, submodel_scopes,
    conditioned, indexed_observations, external_observations, cell_broadcasts) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, vector_parameters,
        matrices,
        array_parameters, submodel_scopes,
        conditioned, indexed_observations, external_observations,
        cell_broadcasts, Pair{Symbol,Symbol}[])

# The former full constructor has no dotted-cell observations.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, vector_parameters,
    matrices,
    array_parameters, submodel_scopes,
    conditioned, indexed_observations, external_observations) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, vector_parameters,
        matrices,
        array_parameters, submodel_scopes,
        conditioned, indexed_observations, external_observations,
        Dict{Symbol,Vector{Symbol}}())

# The former full constructor has no external observations.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, vector_parameters,
    matrices,
    array_parameters, submodel_scopes,
    conditioned, indexed_observations) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, vector_parameters,
        matrices,
        array_parameters, submodel_scopes,
        conditioned, indexed_observations, SampledParameter[])

StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, vector_parameters,
    matrices,
    array_parameters, submodel_scopes,
    conditioned) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, vector_parameters,
        matrices,
        array_parameters, submodel_scopes,
        conditioned, Set{Symbol}())

StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, vector_parameters,
    matrices,
    array_parameters, submodel_scopes) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
        scans, vector_parameters,
        matrices,
        array_parameters, submodel_scopes,
        Set{Symbol}())

# Existing full-positional plans have no lexical submodel metadata.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, vector_parameters,
    matrices, array_parameters) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps,
        plate_parameters, scans,
        vector_parameters, matrices,
        array_parameters, SubmodelScope[])

# Pre-array full-positional constructor: plans built before
# `array_parameters` existed keep working with none.
StructuralPlan(responses, predictors, population_priors, parameters,
    assignments, derived, columns, n_obs, roles, levelmaps, plate_parameters,
    scans, vector_parameters,
    matrices) =
    StructuralPlan(responses, predictors, population_priors, parameters,
        assignments, derived, columns, n_obs, roles, levelmaps,
        plate_parameters, scans,
        vector_parameters, matrices,
        ArrayParameter[])

# Pre-extension full-positional constructor (9-arg): callers that built a plan
# before `levelmaps`/`scans`/`vector_parameters`
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
        LevelMap[], PlateParameter[], ScanSpec[],
        VectorParameter[],
        DesignMatrix[])

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
        vector_parameters::Vector{VectorParameter} = VectorParameter[],
        matrices::Vector{DesignMatrix} = DesignMatrix[],
        array_parameters::Vector{ArrayParameter} = ArrayParameter[],
        submodel_scopes::Vector{SubmodelScope} = SubmodelScope[],
        conditioned::Set{Symbol} = Set{Symbol}(),
        indexed_observations::Set{Symbol} = Set{Symbol}(),
        external_observations::Vector{SampledParameter} = SampledParameter[],
        cell_broadcasts::Dict{Symbol,Vector{Symbol}} = Dict{Symbol,Vector{Symbol}}(),
        named_values::Vector{Pair{Symbol,Symbol}} = Pair{Symbol,Symbol}[])
    return StructuralPlan(responses, predictors, population_priors,
        parameters, assignments, derived, _checked_columns(columns), n_obs,
        roles, levelmaps, plate_parameters, scans,
        vector_parameters, matrices,
        array_parameters, submodel_scopes, conditioned, indexed_observations,
        external_observations, cell_broadcasts, named_values)
end

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

"""Admitted emitter-side term names."""
const TERM_NAMES = Dict{Symbol,TermKind}(
    :intercept => InterceptTerm,
    :continuous => ContinuousTerm,
    :factor => FactorTerm,
    :offset => OffsetTerm,
    :scan_summand => ScanSummandTerm,
    :matrix => MatrixTerm,
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
    ScanSummandTerm, MatrixTerm,
    ComposedTerm)

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

function _axis_exempt_columns(plan::StructuralPlan)
    managed = Set{Symbol}()
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
    # Binding-owned presence masks encode observations, not model dimensions.
    union!(modelvals, (_observed_mask_name(r.response) for r in plan.responses
        if haskey(plan.columns, _observed_mask_name(r.response))))
    union!(modelvals, (_observed_mask_name(s.array) for s in _observed_selections(plan, plan.columns)
        if haskey(plan.columns, _observed_mask_name(s.array))))
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
New slots must establish the same property before joining this list.
`named_values` only names existing nodes, so it has their dimensions."""
const _MULTI_AXIS_SLOTS = (:responses, :predictors, :population_priors,
    :parameters, :assignments, :derived, :columns, :n_obs, :roles,
    :levelmaps, :vector_parameters, :submodel_scopes, :conditioned,
    :plate_parameters, :scans,
    :matrices,
    :array_parameters, :indexed_observations, :external_observations,
    :cell_broadcasts, :named_values)

# Observation-shaped values and their data dependencies. Parameters sized
# by levels or coefficient width are shared values, so their priors do not
# join observation axes. A basis or matrix does carry rows and must expose
# its inputs through the same dependency walk.
function _observation_nodes(plan::StructuralPlan)
    nodes = Dict{Symbol,Any}()
    for p in plan.predictors
        nodes[p.name] = p
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
        perobs::Set{Symbol}; whole_values::Bool=false,
        stop::Set{Symbol}=Set{Symbol}())
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
        haskey(nodes, s) && !(s in stop) && append!(queue, held(nodes[s]))
    end
    return reads
end

# Structured constructors keep their established row-domain contract. Ordinary
# elementwise responses use Julia broadcast axes, including singleton dimensions.
function _uses_structured_observation_axes(plan::StructuralPlan)
    any(r -> r.mi_jobs !== nothing, plan.responses) && return true
    !isempty(plan.plate_parameters) && !any(r -> r.range isa Expr, plan.responses) && return true
    return any(f -> !isempty(getfield(plan, f)),
        (:scans, :matrices))
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
    length(unique(response_rows)) <= 1 && return nothing
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
            index.args[2] isa Int && index.args[3] isa Int) ||
        (Meta.isexpr(index, :call, 2) && index.args[1] === :eachindex && index.args[2] isa Symbol) ||
        (Meta.isexpr(index, :call, 3) && index.args[1] === :axes && index.args[2] isa Symbol &&
            index.args[3] isa Integer && !(index.args[3] isa Bool) && index.args[3] >= 1)
    valid || _fail(r.label, "response range uses a literal a:b, eachindex(v), axes(v, d), or :")
    all(i -> i isa Integer && !(i isa Bool) && i >= 1, ex.args[3:end]) ||
        _fail(r.label, "response trailing indices must be positive literal integers")
    r.mi_jobs === nothing || _fail(r.label, "dynamic index ranges beside packed mi rows are not built yet")
    return nothing
end

_response_range_source(r::LikelihoodSpec) = r.range.args[2] === :(:) ?
    r.response : r.range.args[2].args[1] === :(:) ? r.response : r.range.args[2].args[2]

_response_range_indices(plan::StructuralPlan, r::LikelihoodSpec) =
    _response_range_indices(plan.columns, r)

function _response_range_indices(columns::AbstractDict, r::LikelihoodSpec)
    _validate_response_range_expr(r)
    source = _response_range_source(r)
    haskey(columns, source) || _fail(r.label, "response index source $source is not bound")
    value = columns[source]
    value isa AbstractArray || _fail(r.label, "response index source $source must be an array")
    index = r.range.args[2]
    index isa Expr && index.args[1] === :(:) && return index.args[2]:index.args[3]
    return _response_column_axis(value, index)
end

# Bound columns have arbitrary array types. Inferring this query once for any
# value keeps their generic `eachindex`/`axes` methods out of the lowering
# code's compiled dependencies, which other packages' array methods would
# otherwise invalidate.
Base.@nospecializeinfer _response_column_axis(@nospecialize(value), @nospecialize(index)) =
    index === :(:) || index.args[1] === :eachindex ? eachindex(value) :
        axes(value, index.args[3])

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
    if r.range isa Expr
        if name === r.response
            value = _selected_response_column(plan, r)
        elseif r.response in plan.indexed_observations
            indices = _response_range_indices(plan, r)
            checkbounds(Bool, value, indices) || _fail(r.label,
                "$what $name does not cover the authored response range")
            value = value[indices]
        end
    end
    mask = _response_observed_mask(plan, r)
    mask === nothing && return value
    # Validate only observation lanes. Singleton operands retain ordinary
    # broadcasting; dimensions and parameter extents still use the full data.
    lanes = broadcast((v, present) -> (v, present), value, mask)
    return first.(lanes)[last.(lanes)]
end

# ── Observed selections ──────────────────────────────────────────────
# Observation statements cover their whole response. Missing entries are
# skipped automatically (PROVISIONAL user decision `1uhcm3b`, 2026-10-05),
# never selected manually by shortening the authored observation domain.
# A selection is a response range (a plate's `y[2:6]` or `y[axes(y, 2)]`,
# `y[i, 1]`, `y[eachindex(x)]`) or a statement's data-indexed left-hand
# side (`y[rows] .~ …`, which `_indexed_observation_definitions` rewrites
# into an observed gather). A definition the author wrote, such as a lag,
# is ordinary data use, not a selection. Each observation covers its array.
const _OBSERVED_GATHER_PREFIX = "_rkppl_observed_"
_observed_mask_name(name::Symbol) = Symbol(:_rkppl_present_, name)

function _response_observed_mask(plan, r)
    mask = get(plan.columns, _observed_mask_name(r.response), nothing)
    mask === nothing && return nothing
    r.range isa Expr || return mask
    indices = _response_range_indices(plan, r)
    return mask[indices, r.range.args[3:end]...]
end

_has_observed_response(plan, r) =
    !haskey(plan.columns, _observed_mask_name(r.response)) ||
        any(plan.columns[_observed_mask_name(r.response)])

function _observed_selections(plan::StructuralPlan, columns::AbstractDict)
    gathers = Dict{Symbol,Any}(d.name => d.expr for d in plan.derived)
    out = NamedTuple{(:array, :label, :index, :kernel),Tuple{Symbol,Symbol,Tuple,Bool}}[]
    for r in plan.responses
        r.mi_jobs === nothing || continue
        if r.range isa Expr
            get(columns, r.response, nothing) isa AbstractArray || continue
            index = (_response_range_indices(columns, r), r.range.args[3:end]...)
            push!(out, (; array = r.response, label = r.label, index, kernel = true))
        elseif startswith(string(r.response), _OBSERVED_GATHER_PREFIX)
            ex = get(gathers, r.response, nothing)
            Meta.isexpr(ex, :ref, 2) && get(columns, ex.args[1], nothing) isa AbstractArray ||
                continue
            i = ex.args[2]
            index = i isa Symbol ? get(columns, i, nothing) :
                _literal_row_range(i) ? (i.args[2]:i.args[3]) : nothing
            index isa AbstractArray || continue
            push!(out, (; array = ex.args[1], label = r.label, index = (index,), kernel = false))
        end
    end
    return out
end

_entry_name(name, I::CartesianIndex) = "$name[$(join(Tuple(I), ", "))]"
_entry_names(name, entries) = join((_entry_name(name, I) for I in first(entries, 3)), ", ") *
    (length(entries) > 3 ? " and $(length(entries) - 3) more" : "")

const _SHAPE_READS = (:axes, :eachindex, :size, :length)
_is_shape_read(ex) = Meta.isexpr(ex, :call) && !isempty(ex.args) &&
    (ex.args[1] in _SHAPE_READS || (ex.args[1] isa GlobalRef &&
        ex.args[1].mod === Base && ex.args[1].name in _SHAPE_READS))

# A column read only through its own selections: no other plan value
# names it. Shape reads (`axes(y, 1)` iterators, index sources) contribute
# axes, not values.
function _read_only_by_selection(plan::StructuralPlan, name::Symbol)
    found = false
    function visit(x)
        found && return nothing
        if x === name
            found = true
        elseif x isa VectorAssignmentSpec &&
                startswith(string(x.name), _OBSERVED_GATHER_PREFIX)
            # The observation gather has already been checked for full
            # coverage. It carries the same owned values and presence mask.
            return nothing
        elseif _is_shape_read(x)
            return nothing
        elseif x isa Expr
            foreach(visit, x.args)
        elseif x isa QuoteNode
            visit(x.value)
        elseif x isa Union{AbstractArray,Tuple,NamedTuple,AbstractSet,Pair}
            foreach(visit, x)
        elseif x isa AbstractDict
            foreach(p -> (visit(first(p)); visit(last(p))), x)
        elseif isstructtype(typeof(x)) && parentmodule(typeof(x)) === @__MODULE__
            for f in fieldnames(typeof(x))
                isdefined(x, f) && visit(getfield(x, f))
            end
        end
        return nothing
    end
    for f in fieldnames(StructuralPlan)
        f in (:columns, :roles, :n_obs, :submodel_scopes, :responses,
            :indexed_observations, :cell_broadcasts) && continue
        visit(getfield(plan, f))
    end
    for r in plan.responses, f in fieldnames(LikelihoodSpec)
        f === :range && continue
        r.response === name && f in (:response, :label) && continue
        visit(getfield(r, f))
    end
    return !found
end

# Full coverage applies even to missing entries. Skipping is the binder's
# responsibility, rather than an authored subset or a union of partial statements.
function _validate_observed_selections!(plan::StructuralPlan, columns::AbstractDict)
    for s in _observed_selections(plan, columns)
        value = columns[s.array]
        # Out-of-range selections fail with their statement's own message.
        checkbounds(Bool, value, s.index...) || continue
        mask = falses(size(value))
        mask[s.index...] .= true
        left = [I for I in CartesianIndices(value) if !mask[I]]
        isempty(left) || _fail(s.label, "partial observation of $(s.array) leaves " *
            "$(_entry_names(s.array, left)) outside the statement — observe the " *
            "whole response; missing observations are skipped automatically")
    end
    return columns
end

# Missing is a host-side representation. RK receives an owned concrete array
# and a Bool mask, retaining every axis. The neutral stored value is never an
# observation: the generated plate guards density evaluation with the mask.
function _prepare_missing_responses!(plan::StructuralPlan, columns::AbstractDict)
    names = Dict(r.response => r for r in plan.responses if r.mi_jobs === nothing)
    for s in _observed_selections(plan, columns)
        names[s.array] = only(r for r in plan.responses if r.label === s.label)
    end
    for (name, r) in names
        value = get(columns, name, nothing)
        value isa AbstractArray || continue
        Missing <: eltype(value) || continue
        present = .!ismissing.(value)
        if !all(present)
            isempty(r.count_columns) && isempty(r.extra_responses) || _fail(r.label,
                "missing components of a joint response require marginalization, which is not built yet")
            _read_only_by_selection(plan, name) || _fail(r.label,
                "missing response $name is also read as a model value; automatic " *
                "likelihood skipping does not define that value")
        end
        T = Base.nonmissingtype(eltype(value))
        if T === Union{}
            # An all-Missing array carries no numeric type. Counts/categories
            # require an integer port; continuous responses use Float64.
            T = _is_bernoulli_family(r.family) || _is_binomial_family(r.family) ||
                _is_ordered_family(r.family) || r.family in
                (PoissonLogFam, NegativeBinomial2Fam, NegativeBinomialFam,
                 HurdlePoissonFam, ZeroInflatedPoissonFam, CategoricalFam,
                 CategoricalLogitFam, BernoulliLogitGLMFam, PoissonLogGLMFam) ? Int : Float64
        end
        T <: Real && isconcretetype(T) || _fail(r.label,
            "response $name needs a concrete numeric type beside Missing, got $(eltype(value))")
        owned = Array{T}(undef, size(value))
        map!(x -> ismissing(x) ? one(T) : convert(T, x), owned, value)
        columns[name] = owned
        if !all(present)
            maskname = _observed_mask_name(name)
            haskey(columns, maskname) && _fail(r.label, "data name $maskname is reserved for observation presence")
            columns[maskname] = Array(present)
        end
    end
    return columns
end

# Selected observation arguments have private gensym producers, one per
# argument in _desugar_selected_plate. Their cell locals belong to that
# observation. Guard those producers at binding and generation; public value
# definitions and latent prior arguments retain their ordinary evaluation.
function _guard_missing_observation_argument(name, ex, plan, columns)
    indices = _selected_plate_indices(ex)
    indices === nothing && return ex
    private(n) = Base.isgensym(n) && occursin("_rkppl_selected", String(n))
    # Location extraction can retain the private argument as an OffsetTerm
    # reading a synthetic column, instead of keeping its plate in the term.
    private(name) || any(p -> private(p.name) &&
        any(t -> name in t.columns, p.terms), plan.predictors) || return ex
    readers = [r for r in plan.responses if name in
        _response_reads(plan, r, Set{Symbol}([name]))]
    isempty(readers) && return ex
    present = nothing
    for r in readers
        r.range isa Expr || return ex
        rows = _response_range_indices(columns, r)
        # Resolved module calls spell Base.eachindex, while the response
        # range retains eachindex. Compare their ordinary bound indices.
        collect(_eval_value_expr(indices, n -> columns[n], r.label)) == rows || return ex
        value = get(columns, r.response, nothing)
        value isa AbstractArray || return ex
        mask = get(columns, _observed_mask_name(r.response), nothing)
        mask === nothing && !(Missing <: eltype(value)) && return ex
        mask === nothing && (mask = .!ismissing.(value))
        all(mask) && return ex
        selected = mask[rows, r.range.args[3:end]...]
        present = present === nothing ? Array(selected) : present .| selected
    end
    guarded = deepcopy(ex)
    inputs, lambda = guarded.args
    lane = gensym(:_rkppl_argument_present)
    push!(inputs.args, QuoteNode(present))
    push!(lambda.args[1].args, lane)
    lambda.args[2] = Expr(:block, LineNumberNode(0, :rkppl_plate),
        Expr(:if, lane, lambda.args[2], 0))
    return guarded
end

_guarded_argument_is_bound(ex, columns) =
    all(inp -> _expr_value_symbols(inp) ⊆ Set{Symbol}(keys(columns)), ex.args[1].args[2:end])

# A guarded data-only cell can return a numeric value or its integer neutral
# arm. Its ordinary preparation-time result needs promoted concrete storage
# before becoming an RK bound operand. Traced numeric arrays already have a
# concrete element type; no value-dependent host work is performed on them.
function _concrete_guarded_numeric_storage(value)
    if value isa AbstractArray{<:Real} && !isconcretetype(eltype(value)) && !isempty(value)
        T = foldl((t, v) -> promote_type(t, typeof(v)), value; init=Union{})
        isconcretetype(T) && return T.(value)
    end
    return value
end

_concrete_guarded_argument(ex) = Expr(:call,
    GlobalRef(@__MODULE__, :_concrete_guarded_numeric_storage), ex)

"""Broadcast domains of bound observation statements: `(; rows, total,
domains)`, where `domains` maps response labels to Julia broadcast axes.
Singleton operands do not join independent domains. Structured constructors
retain their existing row-domain validation.
Several domains beside slots outside `_MULTI_AXIS_SLOTS` are not built yet."""
function _observation_axes(plan::StructuralPlan)
    _uses_structured_observation_axes(plan) && return _structured_observation_axes(plan)
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
        # `eachrow(E)` supplies one threshold-effect vector per response
        # cell. Its stage axis belongs to the ordinal law, not the broadcast
        # domain. Keep that role local to this reader of the ordinary matrix.
        axesread(c) = c === r.threshold_effects ? (axes(plan.columns[c], 1),) : colaxes[c]
        operandaxes(c) = r.range isa Expr && r.response in plan.indexed_observations ?
            _indexed_operand_axes(plan, r, c, c in designs || c === r.threshold_effects) : axesread(c)
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
    reads = _response_reads(plan, r, _row_columns(plan))
    ns = unique!([_column_nrows(plan.columns[c]) for c in reads])
    length(ns) == 1 && return only(ns)
    isempty(ns) && return plan.n_obs
    _fail(r.label, "mi() location reads columns with different rows $ns")
end

"""Bound vector and matrix columns that carry observation rows: every
column except model-level values and mi()-managed columns."""
function _row_columns(plan::StructuralPlan)
    modelvals, _ = _axis_exempt_columns(plan)
    managed = _mi_managed_columns(plan)
    return Set{Symbol}(k for (k, v) in plan.columns
        if k ∉ modelvals && k ∉ managed && v isa Union{AbstractVector,AbstractMatrix})
end

"""Rows of an observation-shaped value, resolved from its bound inputs.
Values with no data anchor (an intercept or dar path) use their response
consumers. No dimension is inferred from the total of unrelated axes."""
function _value_rows(plan::StructuralPlan, name::Symbol)
    haskey(plan.columns, name) && return _column_nrows(plan.columns[name])
    reads = _response_reads(plan, name, _row_columns(plan))
    ns = unique!([_column_nrows(plan.columns[c]) for c in reads])
    if isempty(ns)
        names = Set{Symbol}([name])
        ns = unique!([_response_rows(plan, r) for r in plan.responses
            if name in _response_reads(plan, r, names)])
    end
    length(ns) == 1 && return only(ns)
    if isempty(ns)
        axes = unique!([_response_rows(plan, r) for r in plan.responses])
        length(axes) == 1 && return only(axes)
        isempty(axes) && return plan.n_obs
        _fail(name, "value $name has no data range or response to establish its rows")
    end
    _fail(name, "value $name reads or feeds different row counts $ns")
end

function _plate_rows(plan::StructuralPlan, p::PlateParameter;
        active = Set{Symbol}())
    p.range isa UnitRange && return length(p.range)
    if p.range isa Expr
        return _value_iterator_length(plan, p, p.range, active)
    end
    p.range isa Symbol && return _value_iterator_length(plan, p,
        Expr(:call, :eachindex, p.range), active)
    return _value_rows(plan, p.name)
end

function _value_iterator_length(plan::StructuralPlan, p::PlateParameter,
        iterator, active = Set{Symbol}())
    source = iterator.args[2]
    shape = _value_axes(plan, source, active; data_axes = true)
    # A response cannot determine an unrelated value's extent. Resolve
    # data and declared axes without running sampled values; an opaque
    # live result whose shape is unavailable remains a capability gap.
    shape === nothing && _fail(p.label, "latent plate iterator " *
        "$(repr(iterator)) has no extent established by bound data; " *
        "the shape of $source cannot be inferred without evaluating " *
        "sampled values")
    # Julia numbers and zero-dimensional arrays have one index too.
    sizes = Int[d isa Integer ? d : _array_dim_size(plan, source, p.label, d)
        for d in shape]
    iterator.args[1] === :eachindex && return prod(sizes)
    k = iterator.args[3]
    return k <= length(sizes) ? sizes[k] : 1
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
        f in (:columns, :roles, :n_obs, :submodel_scopes, :cell_broadcasts) && continue
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

function _expr_names!(out::Set{Symbol}, ex)
    ex isa Symbol && (push!(out, ex); return nothing)
    ex isa Expr || return nothing
    for a in ex.args
        _expr_names!(out, a)
    end
    return nothing
end

function _expr_names(ex)::Set{Symbol}
    out = Set{Symbol}()
    _expr_names!(out, ex)
    return out
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
    vectors = [p.name for p in plan.vector_parameters]
    mats = Symbol[m.name for m in plan.matrices]
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
    length(unique(mats)) == length(mats) ||
        _fail(:plan, "duplicate design-matrix names")
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
        (mats, params, "design-matrix names and parameters"),
        (mats, assigns, "design-matrix names and assignments"),
        (mats, deriveds, "design-matrix names and derived columns"),
        (mats, plates, "design-matrix names and plate parameters"),
        (mats, scanstates, "design-matrix names and scan states"),
        (mats, vectors, "design-matrix names and vector parameters"))
        overlap = intersect(l, r)
        isempty(overlap) ||
            _fail(:plan, "names in both $what: $(join(overlap, ", "))")
    end
    allnames = union(params, assigns, deriveds, plates, scanstates,
        vectors, mats)
    arrays = _array_names(plan)
    length(unique(arrays)) == length(arrays) ||
        _fail(:plan, "duplicate array-parameter names")
    overlap = intersect(arrays, allnames)
    isempty(overlap) || _fail(:plan, "names in both array parameters and " *
        "other parameters/assignments/derived/plate/scan/vector/" *
        "matrix names: $(join(overlap, ", "))")
    allnames = union(allnames, arrays)
    for pn in pnames
        pn in allnames && _fail(
            :plan,
            "predictor $pn collides with a parameter/assignment/derived/plate/scan/vector/matrix name",
        )
        block_name(pn) in allnames && _fail(
            :plan,
            "parameter/assignment/derived/plate/scan/vector/matrix $(block_name(pn)) collides with predictor $pn block name",
        )
    end
    for n in Iterators.flatten((pnames, params, assigns, deriveds, plates, scanstates, vectors, mats))
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
    end
    return nothing
end

# In-graph name of a dar trajectory's innovation slice
# (`_ppl_dar_z_<state>`, length `n_obs - 1`). Reserved-prefix validation
# guarantees no user name collides with it; the state name itself binds
# the emitter's `scan(...)` reconstruction.
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

# A derived column whose value `bind_data` checks rather than its
# expression's shape: a data-only module value read per observation (its
# length or row count), or a derived response, which binding evaluates
# and validates as the response (`_materialize_derived_responses!`).
_bind_checked_derived(plan::StructuralPlan, name::Symbol) =
    _is_bind_data_derived(plan, name) ||
        any(r -> r.response === name, plan.responses)

function _validate_vector_structure(plan::StructuralPlan)
    for d in plan.derived
        _validate_vector_alias(d, plan, false)
        d.expr isa Symbol && continue
        refs = Symbol[]
        _collect_vector_refs!(refs, d.expr, plan, d.label, false)
        _is_vector_valued(d.expr, plan) || _bind_checked_derived(plan, d.name) ||
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
        _is_vector_valued(d.expr, plan) || _bind_checked_derived(plan, d.name) ||
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
        (i === :(:) || _is_endpoint_position(i)) && continue
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
        # `:` in a positional read (`Z[:, 1]`) is a whole axis, and `end`
        # / `begin` a position (`Z[end, 1]`), not a name.
        (ex === :(:) || _is_endpoint(ex)) && return nothing
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

# A selected observation plate (`_desugar_selected_plate`) iterates
# `Base.collect(indices)`: its column holds one value per authored index, in
# loop order. Keep that provenance in the first input, like a level axis.
function _selected_plate_indices(ex)
    _is_plate_column_expr(ex) || return nothing
    inp = ex.args[1].args[2]
    return Meta.isexpr(inp, :call, 2) && inp.args[1] == GlobalRef(Base, :collect) ?
        inp.args[2] : nothing
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

# A column's elements are real numbers. Type tests rather than an `eltype`
# call keep this check concretely inferred.
_is_real_column(col) = col isa Real || col isa AbstractArray{<:Real}

# Direct column references in math positions must be numeric (derived and
# nested references are validated where they resolve).
function _check_numeric_position!(args, plan, label, bound::Bool)
    bound || return nothing
    for arg in args
        arg isa Symbol && haskey(plan.columns, arg) &&
            !_is_real_column(plan.columns[arg]) &&
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
            all(i -> i isa Int || _is_endpoint_position(i) ||
                _is_row_index(plan, i), ex.args[2:end])
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
# response or an explicit value shape); an explicit size is bounds-checked here and linked-checked in
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
    # factor piece).
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
    # Every declaration contributes its prior, including an otherwise
    # unused latent. Its extent must resolve from the declaration at bind.
    return nothing
end

"""Canonical in-graph node names: reserved across every plan namespace."""
const RESERVED_NODES = (:prior, :likelihood, :log_jacobian, :posterior, :pointwise, :unconstrained)

"""
    topological_order(plan) -> Vector{Symbol}

Evaluation order over parameters ∪ assignments ∪ derived columns (Kahn's
algorithm). Among names whose dependencies are met, the earliest declared
comes first, so the same plan always gives the same order. Loud on cycles
(and, on bound plans, unknown references — unbound plans defer name
resolution to bind). Shared by validation and the generator.
"""
function topological_order(plan::StructuralPlan)
    names = _union_names(plan)
    allnames = _all_names(plan)
    deps = Dict{Symbol,Set{Symbol}}()
    # Names in declaration order: the order of the plan's tables, each name
    # at its first entry.
    declared = Symbol[]
    function depend!(name, refs)
        haskey(deps, name) || push!(declared, name)
        deps[name] = refs
    end
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
            depend!(state, copy(refs))
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
        depend!(p.name, refs)
    end
    # Vector parameters are constrained in the layout transforms, ahead of
    # every definition that reads them (functions as values).
    for v in _vector_value_names(plan)
        haskey(deps, v) || depend!(v, Set{Symbol}())
    end
    for p in plan.vector_parameters
        p.family === :simplex_dirichlet || continue
        depend!(p.name, Set(ref for ref in _value_symbols(p.args.arg1)
            if ref in allnames))
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
        depend!(p.name, refs)
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
        depend!(p.name, refs)
    end
    for a in plan.assignments
        refs = Symbol[]
        _collect_assignment_refs!(refs, a.expr, plan, a.label, isbound(plan))
        if isbound(plan)
            for r in refs
                r in allnames || _fail(a.label, "assignment references unknown name $r")
            end
        end
        depend!(a.name, Set{Symbol}(r for r in refs if r in allnames))
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
        depend!(d.name, Set{Symbol}(r for r in refs if r in allnames))
    end
    # Take ready names by declaration rank, never by Dict/Set iteration:
    # that orders independent names by their hashes, and a gensym name (a
    # private producer hoisted out of a cell, such as a selected observation
    # argument) hashes differently in every lowering, so one source would
    # generate programs whose statements differ in order.
    rank = Dict{Symbol,Int}(name => i for (i, name) in enumerate(declared))
    remaining = Dict{Symbol,Int}(name => length(d) for (name, d) in deps)
    dependents = Dict{Symbol,Vector{Symbol}}(name => Symbol[] for name in declared)
    for name in declared, d in deps[name]
        push!(dependents[d], name)
    end
    order = Symbol[]
    # Ready ranks, latest first, so `pop!` takes the earliest declared.
    ready = Int[rank[name] for name in Iterators.reverse(declared)
        if remaining[name] == 0]
    while !isempty(ready)
        name = declared[pop!(ready)]
        push!(order, name)
        for m in dependents[name]
            remaining[m] -= 1
            remaining[m] == 0 || continue
            r = rank[m]
            insert!(ready, searchsortedfirst(ready, r; rev = true), r)
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
    if t.kind === ScanSummandTerm
        _validate_scan_term(t, pred, plan)
        return nothing
    end
    if t.kind === MatrixTerm
        _validate_matrix_term(t, pred, plan)
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
# gather options precedent) and carries exactly the matrix's data
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
# A data design-matrix product is affine in its coefficient vector, just
# as its equivalent sum of intercept and continuous terms is.
const _COMPOSED_AFFINE_KINDS =
    (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm, MatrixTerm)
# Sub-predictors are affine.
const _COMPOSED_SUB_KINDS = _COMPOSED_AFFINE_KINDS

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
                  "plus varying effects (intercept/continuous/factor/matrix/" *
                  "offset/varying-effect terms only — no nested " *
                  "compositions, latents, or other summands)")
    end
    # Value leaves are scalars or whole model-level arrays (`z .+ w .* z`
    # with `z[1:K] .~ …`): their values combine with Julia broadcasting.
    known = union(_union_names(plan), _array_names(plan),
        _vector_value_names(plan))
    for c in o.scalars
        c in known ||
            _fail(t.label, "composed value leaf $c is neither a sampled " *
                  "parameter, a declared array nor an assignment")
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
    for c in t.columns
        # A per-cell latent enters a design only through a ContinuousTerm
        # (a free coefficient scaling the latent vector — the SB `me`
        # mirror); every other term kind over a latent fails closed.
        if _is_plate_param(plan, c)
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
        if _cell_broadcast_operand(plan, c)
            col = _array_entries(col)
        elseif _holds_arrays(col)
            _fail(t.label, "column $c holds one array per entry, which an " *
                "observation reads as Julia broadcasting does: elementwise " *
                "over those arrays; read `$c[i]` in a dotted `@plate` cell " *
                "(`y[i] .~ D.(…)`) to broadcast over each entry's values")
        end
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
    param_names = Set{Symbol}(p.name for p in plan.parameters)
    by_param = Dict{Symbol,SampledParameter}(p.name => p for p in plan.parameters)
    assign_names = Set{Symbol}(a.name for a in plan.assignments)
    glm_labels = Set{Symbol}(r.label for r in plan.responses if _is_glm_family(r.family))
    for pr in plan.population_priors
        any(p -> p.name === pr.predictor, plan.predictors) ||
            pr.predictor in glm_labels ||
            _fail(:plan, "prior addresses unknown predictor $(pr.predictor)")
        key = (pr.predictor, pr.addressee)
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
        # Offset, latent and scan terms carry their
        # own priors. Composed terms read priors from their sub-predictors;
        # matrix terms expand to one addressee per matrix column.
        addressees = Set{Symbol}()
        for t in pred.terms
            _parameter_term(t) && continue
            (t.kind === OffsetTerm || t.kind === LatentTerm ||
                t.kind === ScanSummandTerm ||
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
            (pred.name, a) in seen ||
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

# Leveled-field rules follow the response family, independently of whether
# its location is a predictor or a latent value. Non-leveled families leave every leveled
# field at its default. `used_predictors` gains categorical tail predictors.
function _validate_leveled_fields(r::LikelihoodSpec, plan::StructuralPlan,
        used_predictors::Set{Symbol})
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
            _validate_leveled_fields(r, plan, used_predictors)
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
            _validate_leveled_fields(r, plan, used_predictors)
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
        _validate_leveled_fields(r, plan, used_predictors)
        if r.range isa UnitRange
            (r.mi_jobs === nothing ? first(r.range) == 1 : first(r.range) >= 1) ||
                _fail(r.label, "response range must use valid one-based indices")
            length(r.range) >= 1 || _fail(r.label,
                "response range $(r.range) is empty")
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
                  "response")
    end
    return nothing
end

function _validate_response_data(plan::StructuralPlan)
    for r in plan.responses
        _validate_response_data(r, _cell_broadcast_response(plan, r) ?
            _cell_broadcast_entries(plan, r) : plan)
    end
    return nothing
end

function _validate_response_data(r::LikelihoodSpec, plan::StructuralPlan)
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
    return nothing
end

# A response value holding one array per index (`y = [[0.7, 0.3], [1.4]]`).
_holds_arrays(col) = col isa AbstractArray && !(eltype(col) <: Number) &&
    (isempty(col) ? eltype(col) <: AbstractArray : all(x -> x isa AbstractArray, col))

# Data a cell-broadcast response reads per index hold that index's array.
_cell_broadcast_operand(plan::StructuralPlan, c::Symbol) =
    _holds_arrays(get(plan.columns, c, nothing)) &&
        any(((y, reads),) -> c in reads && _holds_arrays(get(plan.columns, y, nothing)),
            plan.cell_broadcasts)
_array_entries(col) = isempty(col) || all(isempty, col) ?
    eltype(eltype(col))[] : identity.(collect(Iterators.flatten(col)))

"""A dotted `@plate` cell (`y[i] .~ D.(…)`) whose bound response holds one
array per index: each cell broadcasts over its own index's entries."""
_cell_broadcast_response(plan::StructuralPlan, r::LikelihoodSpec) =
    _cell_broadcast_observation(plan, r.response)
_cell_broadcast_observation(plan::StructuralPlan, name::Symbol) =
    haskey(plan.cell_broadcasts, name) && _holds_arrays(get(plan.columns, name, nothing))

# Observing one array per index otherwise broadcasts a univariate
# distribution over arrays, which Julia refuses: it takes no array argument.
function _refuse_array_response(r::LikelihoodSpec, plan::StructuralPlan)
    y = r.response
    y in plan.indexed_observations && _fail(r.label, "each `$y[i]` holds an " *
        "array, which a univariate distribution does not observe; broadcast " *
        "the cell over its entries, `$y[i] .~ D.(…)`, as Julia does")
    _fail(r.label, "`$y` holds one array per entry, and `$y .~ D.(…)` " *
        "broadcasts the distribution over those arrays as Julia does (a " *
        "univariate distribution takes no array argument); observe each " *
        "entry's values in a dotted `@plate` cell: `@plate for i in " *
        "eachindex($y); $y[i] .~ D.(…); end`")
end

# Data validators check observation entries. A cell-broadcast response is
# checked on the entries its cells observe: each data array the response
# reads is broadcast against each index's response array, as the cell does
# (an operand read per index supplies that index's value; any other is
# shared by every index), and the results are concatenated.
function _cell_broadcast_entries(plan::StructuralPlan, r::LikelihoodSpec)
    r.mi_jobs === nothing || _fail(r.label,
        "mi() missingness over a response holding arrays is not built yet")
    y = plan.columns[r.response]
    perindex = plan.cell_broadcasts[r.response]
    arrays = Set{Symbol}(k for (k, v) in plan.columns if v isa AbstractArray)
    columns = copy(plan.columns)
    # A definition the cell reads per index (`sc[i]` with `sc = dose .* s`)
    # supplies that index's value; the data it reads are whole inputs of
    # that value, not observation operands.
    for c in _response_reads(plan, r, arrays; stop = Set{Symbol}(perindex))
        c === r.response && continue
        v = plan.columns[c]
        cells = try
            broadcast(y, c in perindex ? v : Ref(v)) do yi, vi
                broadcast((_, x) -> x, yi, vi)
            end
        catch e
            e isa DimensionMismatch || rethrow()
            _fail(r.label, "`$c` does not broadcast with the arrays of " *
                "`$(r.response)`, as each `$(r.response)[i] .~ …` cell requires: " *
                sprint(showerror, e))
        end
        columns[c] = _array_entries(cells)
    end
    entries = _array_entries(y)
    any(ismissing, entries) && _fail(r.label, "missing entries inside the " *
        "arrays of `$(r.response)` are not built yet; drop them from their array")
    columns[r.response] = entries
    return _with(plan; columns = Dict{Symbol,ColumnData}(columns))
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
    mask = _response_observed_mask(plan, r)
    mask === nothing || (col = col[mask])
    _holds_arrays(col) && _refuse_array_response(r, plan)
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
    for p in plan.plate_parameters
        p.range isa Symbol && push!(required, p.range)
        p.range isa Expr && push!(required, p.range.args[2])
    end
    for s in plan.scans
        s.hi isa Symbol && push!(required, s.hi)
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
    return setdiff(names, _preparation_data_names(plan, nodes, raw, onlydata,
        names; bind_roots = axes))
end

# Whole-value data definitions consumed by a parameter-dependent module
# call belong to preparation, whether lowering inlined them or retained
# them as assignments (notably beside declared arrays). They may return
# arbitrary Julia values; the generated data-only statements fold once.
# Keep every dependency needed by bind-time consumers at bind instead.
function _preparation_data_names(plan, nodes, raw, onlydata, names;
        bind_roots = Set{Symbol}())
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
        f in (:assignments, :derived, :columns, :submodel_scopes,
            :cell_broadcasts) && continue
        _drop_held_names!(free, getfield(plan, f))
    end
    # Whole-value classification exempts `axes(M, d)` from observation
    # alignment, but resolving an array's shape still needs M at bind.
    for p in plan.array_parameters, d in p.dims
        _drop_held_names!(free, d)
    end
    # Binding validates every gather index against the array it reads.
    setdiff!(free, bind_roots)
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
        f in (:columns, :n_obs, :roles, :submodel_scopes, :cell_broadcasts) && continue
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
        elseif f in (:external_observations, :cell_broadcasts)
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
_has_observation_axis(plan) = !isempty(plan.responses) ||
    any(p -> p.range === nothing, plan.plate_parameters) ||
    !isempty(plan.scans)

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
        exprs[d.name] = _guard_missing_observation_argument(d.name, d.expr, plan, columns)
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
    # An array is a value lowering already folded (a literal concentration
    # in `length([1.0, 1.0])`); like any Julia constant it evaluates to itself.
    ex isa Union{Number,String,AbstractArray} && return ex
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
            value isa Number || value isa AbstractArray ||
                throw(ContractValidationError("[bind] derived response " *
                    "$name evaluated to $(summary(value)); a response " *
                    "observes a number or an array"))
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
        vals = Any[_eval_derived_node(a, plan, det_exprs, columns, memo, root)
            for a in ex.args]
        return getindex(vals...)
    end
    if ex.head === :call
        fn = ex.args[1]
        if fn isa Symbol && haskey(_DERIVED_DOTTED_OPS, fn)
            f = _DERIVED_DOTTED_OPS[fn]
            vals = Any[_eval_derived_node(a, plan, det_exprs, columns, memo, root)
                for a in ex.args[2:end]]
            return broadcast(f, vals...)
        end
        if fn isa Symbol && haskey(_DERIVED_SCALAR_FNS, fn)
            f = _DERIVED_SCALAR_FNS[fn]
            vals = Any[_eval_derived_node(a, plan, det_exprs, columns, memo, root)
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
            vals = Any[_eval_derived_node(a, plan, det_exprs, columns, memo, root)
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
assignment/extra columns stay `:data`; `:group` is an explicit role. The retired `dims`
argument accepts only an empty dictionary; array dimensions and plate ranges
come from authored expressions. Data values are numbers or arrays of any shape; observation
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
    isempty(dims) || _fail(:bind, "dims keys are not consumed: use explicit array dimensions and authored @plate ranges")
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
    # A derived response that evaluates to a number is one observation,
    # exactly like the same number bound as data.
    _scalar_responses_as_observations!(plan, columns)
    _validate_observed_selections!(plan, columns)
    _prepare_missing_responses!(plan, columns)
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
    merged = merge(inferred, roles)
    # Observation statements determine their own axes; unrelated axes add
    # their likelihood rows without sizing one value from another's extent.
    axis_plan = _with(plan; columns = columns)
    n = if !_has_observation_axis(plan)
        1
    else
        axes = _observation_axes(axis_plan)
        axes === nothing ? _bind_nrows(columns,
            union(_mi_managed_columns(plan), computed, inputs)) : axes.total
    end
    # Latent plates over values now know their cell counts; the data-only
    # indices selecting their prior inputs evaluate like any other.
    isempty(_bind_plate_extents!(_with(axis_plan; n_obs = n), columns)) ||
        _materialize_module_data!(plan, columns; already = computed)
    maps = _eval_levelmaps(plan.levelmaps, columns)
    responses2, vectors2 =
        _infer_leveled_sizes(plan.responses, plan.vector_parameters, columns,
            plan.predictors, _with(plan; columns = columns, n_obs = n))
    bound = _with(plan; responses = responses2, columns = columns,
        n_obs = n, roles = merged, levelmaps = maps,
        vector_parameters = vectors2)
    _validate_conditioned_values(bound)
    validate_data(bound)
    return bound
end

# Supply `_plate_extent_input` for each latent plate whose prior inputs
# select through it: the same cell count that sizes the latent itself.
function _bind_plate_extents!(plan::StructuralPlan, columns)
    reads = Set{Symbol}()
    for d in (plan.assignments..., plan.derived...)
        _expr_value_symbols(d.expr, reads)
    end
    inputs = Set{Symbol}()
    for p in plan.plate_parameters
        input = _plate_extent_input(p.name)
        input in reads || continue
        haskey(columns, input) && _fail(p.label, "internal input $input " *
            "is the latent's cell count — drop it from bind_data")
        columns[input] = _plate_rows(plan, p)
        push!(inputs, input)
    end
    return inputs
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
# simplex), or a declared value shape. Explicit values assert against the
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
    # A bound plan retains the authored expression for rebinding. Its
    # previous concrete size belongs to the previous data.
    extent = p.extent_expr === nothing ? p.size : p.extent_expr
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
        plan = nothing)
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
    out_v = VectorParameter[]
    for p in vectors
        if p.family === :simplex_dirichlet && p.size === nothing && p.extent_expr === nothing &&
                (plan === nothing || all(s -> _reads_data_only(plan, s,
                    Set{Symbol}(_all_names(plan)), Set{Symbol}()),
                    _expr_value_symbols(p.args.arg1)))
            # Preserve inferred concentration sizing just like an authored
            # data expression, without changing the linked-width checks.
            # A live concentration (`Dirichlet(3, a)`, `a .+ 0.5`) is not
            # data; its structural length comes from `_dirichlet_size`.
            p = _with(p; extent_expr = Expr(:call, :length, p.args.arg1))
        end
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
        elseif p.family in _JOINT_FACTOR_FAMILIES
            # Joint-factor sizes are structural (concrete at construction,
            # validated against the joint width) — bind passes them through.
            p.size === nothing && _fail(p.label,
                "internal: joint-factor size unresolved at bind")
            push!(out_v, p)
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
            push!(out_v, _with(p; size = want))
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
