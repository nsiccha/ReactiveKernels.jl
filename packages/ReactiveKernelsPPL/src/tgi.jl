# Tumor-growth-inhibition (TGI) block cell vocabulary + observation likelihoods.
#
# SB-mirror of the Bruno `joint_pk_qt_tgi_brm1` tumor block
# (`web-pkpd/src/brm_joint_tgi.jl` @ bruno mirror ref `kb-impl/Bruno-arv393-tgi`
# 3af2284686857a37974c83cffa1e555610239dd9; design doc
# `dev-docs/arv393-tgi-model.md`). Aggregate mode only (no per-lesion
# `ragged`); logdensity only (no `_rng`, no held_out/GQ, no replay).
#
# Ownership split (parent `ReactiveKernels:brm:tgi` wave-1): the PK recurrence
# cell (`linear_pk_read_locs_auc_cell`, the AUC identity), the event-schedule
# recipe, the exposure reference recipe, grouped-kernel bind/emission, and the
# BRM-side emitter are sibling slices. This file assumes per-subject
# AUC/concentration columns as HAVE ports and provides:
#
#   * `tgi_options` — validated tumor-block options (SB `_joint_tgi_options`
#     + `_joint_tgi_centered_subjects`, minus formula strings: fixed-effect
#     structure arrives as plan predictors, not BRM-body text);
#   * threshold/measure math (SB `_joint_tgi_body` threshold block);
#   * latent structures `log_linear` / `resistant_fraction` (SB cell lines +
#     `tgi_log_survival` + `tgi_running_nadir`);
#   * observation likelihoods `tgi_censored` / `tgi_category` / `tgi_response`
#     (SB `@deffun` lpmf/lpdfs, minus `_rng`);
#   * `TGI_CELL_FUNCTIONS` — the admitted cell-call spellings the
#     grouped-kernel foundation wires into its checker;
#   * `tgi_nadir_scan_expr` — the kernel-side nadir formulation (`scan`;
#     the host loop cannot trace on param-dependent inputs).
#
# Julianic deltas from the SB counterpart (all value-exact on in-contract
# inputs; each is covered by a dedicated test):
#
#   * scalar core + broadcast/dispatch instead of vector-only Stan functions
#     (`tgi_category_lpmf` has scalar and vector-reduced methods;
#     `tgi_category_lpmfs` is the pointwise SB-mirror);
#   * branchless `ifelse` instead of Stan `if`/`?:`, so every helper traces
#     through Reactant. `tgi_interval_logprob` feeds DUMMY distinct inputs
#     to its inner log-difference when the interval is empty: `ifelse`
#     evaluates both sides, and an exact-tie `log_diff_exp` has an infinite
#     pullback that a zero cotangent turns into NaN (0 * Inf). Empty
#     intervals are REAL inputs here (a deep nadir empties the SD/PR
#     interval exactly via `min`), and Stan's real branch yields gradient 0
#     there — the dummy feed reproduces that exactly;
#   * `tgi_report_logprob` is one branchless `logaddexp` (`log(eps/k)` +
#     `log1p(-eps) + lp`): exact on finite and `-Inf` inputs alike, eps = 0
#     included — no `lp > -inf` branch;
#   * `min` instead of Stan `fmin` in z-score clamping and the nadir: identical
#     on finite inputs; on NaN (out of contract — assessments are finite)
#     Julia propagates where `fmin` drops, i.e. fail-loud, not silent;
#   * structure/threshold/observation selection is compile-time (separate
#     functions + host-side option switch at emission), never a runtime
#     string inside kernel code.
#
# Reactant-traceability contract: every exported math function is pure (no
# input mutation, no exceptions, no strings/dicts), branchless, annotated
# `Number` (Reactant's `TracedRNumber` is a `Number`, not a `Real`), and
# built from Base + `erfc` + `logaddexp` — the `_ordinal_logF`/`_log_diff_exp`
# precedent (`generator.jl`), all three traced-ready via Reactant's own
# `SpecialFunctions`/`LogExpFunctions` extensions (whose `logaddexp`
# mirrors the host `x == y` guard, so `eps = 0` + empty interval stays
# `-Inf` under tracing too). Option validation lives OUTSIDE traced code
# (host-side, at build time). The host nadir loop traces only over
# host/bound data; param-dependent nadirs use the `scan` formulation.
#
# Recommended grouped-kernel emission shapes (proved in `test_tgi.jl`):
# elementwise latent lines as flat recipes; nadir via `tgi_nadir_scan_expr`;
# each observation likelihood as a whole-vector `tgi_*_lpmf/lpdf` recipe over
# bound responses + traced latents (the ordinal-plate `ifelse(y == j, …)`
# precedent for integer responses).

# --- options ---------------------------------------------------------------

"""Tumor observation models the thin layer can lower (SB `JOINT_TGI_OBSERVATIONS`)."""
const TGI_OBSERVATIONS = ("continuous", "ordinal", "binary", "none")

"""Tumor latent structures the thin layer can lower (SB `JOINT_TGI_STRUCTURES`)."""
const TGI_STRUCTURES = ("log_linear", "resistant_fraction")

"""Tumor threshold modes the thin layer can lower (SB `JOINT_TGI_THRESHOLDS`)."""
const TGI_THRESHOLDS = ("lugano_ct", "recist", "estimated")

"""Continuous tumor size measures (SB `JOINT_TGI_MEASURES`)."""
const TGI_MEASURES = ("spd", "sld")

"""
    TGIOptions

Validated tumor-block options (SB `_joint_tgi_options` +
`_joint_tgi_centered_subjects`, minus formula strings — fixed-effect
structure arrives as plan predictors, not BRM-body text). Construct via
[`tgi_options`](@ref), which fails closed naming the admitted spellings.
`lloq` is the left-censoring limit in size units (`nothing` ⇒ no censoring);
`centered_subjects` selects the centered subject parameterization
(default `false`, the non-centered reports'/models' parameterization per
user directive 2026-09-18 — the actual lowering rides the existing
varying-effect machinery at integration).
"""
struct TGIOptions
    observation::String
    structure::String
    thresholds::String
    measure::String
    misclassification::Float64
    lloq::Union{Nothing,Float64}
    centered_subjects::Bool
end

_tgi_fail(msg) = throw(ContractValidationError("[tgi] $msg"))

"""
    tgi_options(; observation = "ordinal", structure = "log_linear",
        thresholds = "lugano_ct", measure = "spd", misclassification = 0.01,
        lloq = nothing, centered_subjects = nothing) -> TGIOptions

Validate tumor-block options (SB `_joint_tgi_options`, error texts mirrored).
`misclassification` is the fixed probability of a report outside the size
rule; `lloq` the left-censoring limit (`nothing` ⇒ positive sizes only);
`centered_subjects` (`nothing` ⇒ `false`) overrides the default subject
parameterization. Every rejection names the admitted spellings.
"""
function tgi_options(; observation::AbstractString = "ordinal",
        structure::AbstractString = "log_linear",
        thresholds::AbstractString = "lugano_ct",
        measure::AbstractString = "spd", misclassification::Real = 0.01,
        lloq::Union{Nothing,Real} = nothing,
        centered_subjects::Union{Nothing,Bool} = nothing)
    observation in TGI_OBSERVATIONS || _tgi_fail(
        "unknown tumor observation model $(repr(observation)); expected one " *
        "of $(join(TGI_OBSERVATIONS, ", "))")
    structure in TGI_STRUCTURES || _tgi_fail(
        "unknown tumor structure $(repr(structure)); expected one of " *
        "$(join(TGI_STRUCTURES, ", "))")
    thresholds in TGI_THRESHOLDS || _tgi_fail(
        "unknown tumor threshold mode $(repr(thresholds)); expected one of " *
        "$(join(TGI_THRESHOLDS, ", "))")
    measure in TGI_MEASURES || _tgi_fail(
        "unknown tumor size measure $(repr(measure)); expected one of " *
        "$(join(TGI_MEASURES, ", "))")
    isnothing(lloq) || (isfinite(lloq) && lloq > 0) || _tgi_fail(
        "tumor lloq (left-censoring limit) must be positive and finite")
    isfinite(misclassification) && 0 <= misclassification < 0.5 || _tgi_fail(
        "misclassification must lie in [0, 0.5)")
    return TGIOptions(String(observation), String(structure),
        String(thresholds), String(measure), Float64(misclassification),
        isnothing(lloq) ? nothing : Float64(lloq),
        isnothing(centered_subjects) ? false : centered_subjects)
end

# --- constants (SB-mirror; `log` of the same literal is bit-identical) ------

"""The tumor clock: one unit is eight weeks, the first protocol restaging."""
const TGI_TIME_SCALE_H = 1344.0

"""CT-based Lugano 2014 PR boundary on the log size-ratio scale (SPD-native)."""
const TGI_LOG_PR = log(0.5)

"""CT-based Lugano 2014 size-progression boundary (SPD-native)."""
const TGI_LOG_PD = log(1.5)

"""RECIST 1.1 PR boundary on the log size-ratio scale (SLD-native)."""
const TGI_RECIST_LOG_PR = log(0.7)

"""RECIST 1.1 progression boundary (SLD-native; the +5 mm floor is not modeled)."""
const TGI_RECIST_LOG_PD = log(1.2)

# --- thresholds (SB `_joint_tgi_body` threshold block) ---------------------

"""
    tgi_measure_dim(measure) -> Int

Dimensionality of a size measure: SPD is bidimensional, SLD
unidimensional (SB `_joint_tgi_measure_dim`). Fails closed naming the
admitted measures.
"""
function tgi_measure_dim(measure::AbstractString)
    measure in TGI_MEASURES || _tgi_fail(
        "unknown tumor size measure $(repr(measure)); expected one of " *
        "$(join(TGI_MEASURES, ", "))")
    return measure == "sld" ? 1 : 2
end

"""
    tgi_threshold_scale(options) -> Float64

Rescale of a fixed size rule onto the modeled measure: the ratio of the
modeled dimensionality to the rule's native one (Lugano/CT native SPD,
RECIST native SLD). The default `lugano_ct` × `spd` scales by exactly 1.0.
"""
function tgi_threshold_scale(options::TGIOptions)
    native_dim = options.thresholds == "recist" ? 1 : 2
    return tgi_measure_dim(options.measure) / native_dim
end

"""
    tgi_fixed_cutpoints(options) -> (c_pr, c_pd)

Fixed (PR, size-progression) log-ratio cutpoints for `lugano_ct`/`recist`
(SB `log_pr`/`log_pd`). Fails closed on `estimated` (free cutpoints —
see [`tgi_estimated_cutpoints`](@ref)).
"""
function tgi_fixed_cutpoints(options::TGIOptions)
    options.thresholds == "estimated" && _tgi_fail(
        "threshold mode `estimated` has no fixed cutpoints (boundaries " *
        "are free parameters — see `tgi_estimated_cutpoints`)")
    s = tgi_threshold_scale(options)
    if options.thresholds == "recist"
        return (s * TGI_RECIST_LOG_PR, s * TGI_RECIST_LOG_PD)
    end
    return (s * TGI_LOG_PR, s * TGI_LOG_PD)
end

"""
    tgi_estimated_cutpoints(c_cr, d_pr, d_pd) -> (c_cr, c_pr, c_pd)

Free cutpoints for threshold mode `estimated` (SB `c_cr`/`c_pr`/`c_pd`):
`c_pr = c_cr + d_pr`, `c_pd = c_cr + d_pr + d_pd`.
"""
tgi_estimated_cutpoints(c_cr::Number, d_pr::Number, d_pd::Number) =
    (c_cr, c_cr + d_pr, c_cr + d_pr + d_pd)

"""
    tgi_uses_nadir(options) -> Bool

Whether progression is measured from the running nadir (`lugano_ct` /
`recist`) rather than from baseline (`estimated` ⇒ reference 0).
"""
tgi_uses_nadir(options::TGIOptions) = options.thresholds != "estimated"

# --- latent structures (SB cell lines + `tgi_log_survival`) -----------------

"""
    tgi_ratio_loglinear(t, exposure, g, k)

Log size ratio to the size at first dose, `log_linear` structure (SB
`tgi_r = tgi_g * tgi_t - tgi_k * tgi_exposure`): `g * t - k * exposure`.
Scalar core; broadcast for vectors. `t` is on the tumor clock
(`TGI_TIME_SCALE_H`); `exposure` is the AUC in reference units.
"""
tgi_ratio_loglinear(t::Number, exposure::Number, g::Number, k::Number) =
    g * t - k * exposure
tgi_ratio_loglinear(t::AbstractVector, exposure::AbstractVector, g::Number,
        k::Number) = tgi_ratio_loglinear.(t, exposure, g, k)

"""
    tgi_log_survival(kill, phi)

Log surviving size fraction when a fraction `phi` of the tumor is not
killed (SB `tgi_log_survival`): `logaddexp(log1p(-phi) - kill, log(phi))`.
Scalar core; broadcast for vectors.
"""
tgi_log_survival(kill::Number, phi::Number) =
    logaddexp(log1p(-phi) - kill, log(phi))
tgi_log_survival(kill::AbstractVector, phi::Number) =
    tgi_log_survival.(kill, phi)

"""
    tgi_ratio_resistant(t, exposure, g, k, phi)

Log size ratio, `resistant_fraction` structure (SB `tgi_r` with
`tgi_log_survival(tgi_k * tgi_exposure, inv_logit(tgi_logit_phi))`):
`g * t + tgi_log_survival(k * exposure, phi)`. Pass `phi` (the caller
applies [`tgi_inv_logit`](@ref) to the logit parameter). Scalar core;
broadcast for vectors.
"""
tgi_ratio_resistant(t::Number, exposure::Number, g::Number, k::Number, phi::Number) =
    g * t + tgi_log_survival(k * exposure, phi)
tgi_ratio_resistant(t::AbstractVector, exposure::AbstractVector, g::Number,
        k::Number, phi::Number) = tgi_ratio_resistant.(t, exposure, g, k, phi)

"""Inverse logit (Stan `inv_logit`): `1 / (1 + exp(-x))` — no overflow to NaN."""
tgi_inv_logit(x::Number) = 1 / (1 + exp(-x))

"""
    tgi_running_nadir(r) -> Vector

Reference each assessment's size progression is measured from: the smallest
model-predicted log size change among the PREVIOUS assessments and the
baseline scan (change 0). Assessments must be in time order (SB
`tgi_running_nadir`). Plain loop, eltype-generic: it is what generated
code runs (through [`tgi_segmented_nadir`](@ref)) natively, under Enzyme,
and under Reactant, where a traced change vector is read element-wise
through the traced-gather hook and the loop unrolls at trace time.
[`tgi_nadir_scan_expr`](@ref) is the equivalent `scan` spelling for
hand-authored kernels. `min` matches Stan `fmin` on finite inputs.
"""
function tgi_running_nadir(r::AbstractVector)
    # Eltype-generic: a traced change vector (Reactant) yields traced
    # scalars — reads go through the traced-gather hook (`_traced_op_read`,
    # pkcells.jl), the running minimum stays scalar arithmetic.
    T = promote_type(eltype(r), Float64)
    out = Vector{T}(undef, length(r))
    current = zero(T)
    for i in eachindex(r)
        out[i] = current
        current = min(current, _traced_op_read(r, i))
    end
    return out
end

"""
    tgi_nadir_scan_expr(change, ref) -> Expr

Kernel-side nadir formulation as a `scan` statement
(`ref = scan(change; init = 0.0) do carry, x (min(carry, x), carry) end`):
output-before-update reproduces [`tgi_running_nadir`](@ref) exactly. The
sequence must be non-empty (`scan` contract); subjects without assessments
need their empty case handled at emission. `change` is the subject's row
sequence: a plain vector name or a range-copy over the subject's
segment (the segmented unroll slices one subject's rows; copies, not
views — `scan` over a view scalar-indexes under Reactant).
"""
function tgi_nadir_scan_expr(change::Union{Symbol,Expr}, ref::Symbol)
    ex = :($ref = scan($change; init = 0.0) do carry, x
        (min(carry, x), carry)
    end)
    return Base.remove_linenums!(ex)
end

"""
    tgi_segmented_nadir(change, ends)

Per-subject running nadir over a concatenated row series: `change` the
flat change-from-baseline rows (subjects blocked), `ends` the
cumulative per-subject row ends (the `op_ends` precedent — segment `s`
is rows `prev+1:ends[s]`, empty when `ends[s] == prev`). Each segment
runs [`tgi_running_nadir`](@ref); the result vcats the segments (one
entry per row, in order).

The grouped cell form `tgi_ref = tgi_segmented_nadir(tgi_change,
tgi_seg_ends)` emits as exactly this call (one statement, whatever the
subject count — the ends are a bound column), so this is both the
host-side oracle and the generated-code path: native, Enzyme, and
Reactant (traced `change`, element reads through the traced-gather hook,
loop unrolled at trace time). Validates the segment contract defensively
(`ends` nondecreasing from a non-negative start, last end == row count).
"""
function tgi_segmented_nadir(change::AbstractVector,
        ends::AbstractVector{<:Integer})
    all(i -> ends[i] >= ends[i - 1], 2:length(ends)) ||
        throw(ArgumentError("tgi_segmented_nadir ends must be " *
                            "nondecreasing (got $ends)"))
    (isempty(ends) || ends[1] >= 0) ||
        throw(ArgumentError("tgi_segmented_nadir ends must be " *
                            "non-negative (got $ends)"))
    (isempty(ends) ? 0 : ends[end]) == length(change) ||
        throw(ArgumentError("tgi_segmented_nadir last end " *
                            "$(isempty(ends) ? 0 : ends[end]) ≠ row count " *
                            "$(length(change))"))
    T = promote_type(eltype(change), Float64)
    out = Vector{T}[]
    prev = 0
    for hi in ends
        push!(out, tgi_running_nadir(view(change, (prev + 1):hi)))
        prev = hi
    end
    isempty(out) && return zeros(T, 0)
    return reduce(vcat, out)
end

# --- likelihood primitives (SB `@deffun` math) ------------------------------

"""
    tgi_normal_lcdf(x)

Stan `normal_lcdf(x, 0, 1)` via `log(0.5 * erfc(-x / sqrt(2)))` — the
`_ordinal_logF` probit precedent (accurate in moderate ranges; extreme
tails round, the accepted slice-1 probit caveat).
"""
tgi_normal_lcdf(x::Number) = log(0.5 * erfc(-x / sqrt(2)))

"""
    tgi_log_diff_exp(a, b)

Stable log-difference of log-probs, `a ≥ b` (Stan `log_diff_exp`):
`a + log1p(-exp(b - a))` — the `_log_diff_exp` precedent.
"""
tgi_log_diff_exp(a::Number, b::Number) = a + log1p(-exp(b - a))

"""
    tgi_interval_logprob(lo, hi)

Log-probability that a standard normal falls in `(lo, hi]`, taken on the
numerically favourable tail (SB `tgi_interval_logprob`). Arguments clamp
to ±30; an empty interval is `-Inf`. Branchless (`ifelse`): when the
interval is empty the inner log-difference runs on DUMMY distinct inputs
(`sa = -1, sb = 1`) instead of the tied real ones, so no infinite
pullback meets the zero cotangent (see this file's header).
"""
function tgi_interval_logprob(lo::Number, hi::Number)
    a = min(max(lo, -30.0), 30.0)
    b = min(max(hi, -30.0), 30.0)
    nonempty = b > a
    sa = ifelse(nonempty, a, -1.0)
    sb = ifelse(nonempty, b, 1.0)
    upper = tgi_log_diff_exp(tgi_normal_lcdf(-sa), tgi_normal_lcdf(-sb))
    lower = tgi_log_diff_exp(tgi_normal_lcdf(sb), tgi_normal_lcdf(sa))
    return ifelse(nonempty, ifelse(sa > 0, upper, lower), -Inf)
end

"""
    tgi_report_logprob(lp, eps, k)

Mix a size-rule log-probability with a uniform report over `k` categories
(SB `tgi_report_logprob`): one branchless `logaddexp`, exact for finite
and `-Inf` `lp` alike (`eps = 0` included).
"""
tgi_report_logprob(lp::Number, eps::Number, k::Number) =
    logaddexp(log(eps) - log(k), log1p(-eps) + lp)

# --- observation likelihoods (SB `tgi_category` / `tgi_response` / `tgi_censored`) ---

"""
    tgi_category_lpmf(y, r, ref, c_cr, c_pr, c_pd, sigma, eps)

Log-probability of a response category (`1=CR, 2=PR, 3=SD, 4=PD`) as a
coarsening of the noisy log size change `rho ~ Normal(r, sigma)`: PD when
`rho >= c_pd + ref`, otherwise CR below `c_cr`, PR below `c_pr`, SD above
(SB `tgi_category_lpmfs`, progression takes precedence so a deep nadir can
leave the SD/PR interval empty). `eps` mixes a uniform report over the
four categories. Scalar method + vector-reduced method (dispatch);
[`tgi_category_lpmfs`](@ref) is the pointwise SB-mirror. `y` takes
`Number` (bound codes trace as constants); code validity (`1..4`) is a
bind-side contract (SB row validation).
"""
function tgi_category_lpmf(y::Number, r::Number, ref::Number, c_cr::Number,
        c_pr::Number, c_pd::Number, sigma::Number, eps::Number)
    pd = (c_pd + ref - r) / sigma
    pr = min((c_pr - r) / sigma, pd)
    cr = min((c_cr - r) / sigma, pr)
    lp = ifelse(y == 1, tgi_interval_logprob(-30.0, cr),
        ifelse(y == 2, tgi_interval_logprob(cr, pr),
            ifelse(y == 3, tgi_interval_logprob(pr, pd),
                tgi_interval_logprob(pd, 30.0))))
    return tgi_report_logprob(lp, eps, 4.0)
end

"""Pointwise category log-probabilities (SB `tgi_category_lpmfs`)."""
tgi_category_lpmfs(y::AbstractVector, r::AbstractVector, ref::AbstractVector,
        c_cr::Number, c_pr::Number, c_pd::Number, sigma::Number, eps::Number) =
    tgi_category_lpmf.(y, r, ref, c_cr, c_pr, c_pd, sigma, eps)

"""Reduced category log-probability (SB `tgi_category_lpmf`)."""
tgi_category_lpmf(y::AbstractVector, r::AbstractVector, ref::AbstractVector,
        c_cr::Number, c_pr::Number, c_pd::Number, sigma::Number, eps::Number) =
    sum(tgi_category_lpmfs(y, r, ref, c_cr, c_pr, c_pd, sigma, eps))

"""
    tgi_response_lpmf(y, r, ref, c_pr, c_pd, sigma, eps)

Log-probability of a responder (CR or PR) indicator (SB
`tgi_response_lpmfs`): `y == 1` takes `(-30, pr]`, otherwise `(pr, 30]`,
with `pr = min(c_pr - r, c_pd + ref - r) / sigma`; `eps` mixes a uniform
report over the two outcomes. Scalar method + vector-reduced method
(dispatch); [`tgi_response_lpmfs`](@ref) is the pointwise SB-mirror. `y`
takes `Number` (bound codes trace as constants); code validity (`0..1`)
is a bind-side contract (SB row validation).
"""
function tgi_response_lpmf(y::Number, r::Number, ref::Number, c_pr::Number,
        c_pd::Number, sigma::Number, eps::Number)
    pr = min(c_pr - r, c_pd + ref - r) / sigma
    lp = ifelse(y == 1, tgi_interval_logprob(-30.0, pr),
        tgi_interval_logprob(pr, 30.0))
    return tgi_report_logprob(lp, eps, 2.0)
end

"""Pointwise responder log-probabilities (SB `tgi_response_lpmfs`)."""
tgi_response_lpmfs(y::AbstractVector, r::AbstractVector, ref::AbstractVector,
        c_pr::Number, c_pd::Number, sigma::Number, eps::Number) =
    tgi_response_lpmf.(y, r, ref, c_pr, c_pd, sigma, eps)

"""Reduced responder log-probability (SB `tgi_response_lpmf`)."""
tgi_response_lpmf(y::AbstractVector, r::AbstractVector, ref::AbstractVector,
        c_pr::Number, c_pd::Number, sigma::Number, eps::Number) =
    sum(tgi_response_lpmfs(y, r, ref, c_pr, c_pd, sigma, eps))

"""
    tgi_censored_lpdf(y, mu, sigma, lloq_log)

Left-censored (M3/BLQ) normal log-density (SB `tgi_censored_lpdfs`): a value
at or below the log-limit contributes the Normal CDF at the bound, an
interior value the log-density. Values arrive clamped to the bound by data
preparation (SB `_joint_tgi_columns`); a below-bound value contributes the
CDF all the same. Scalar method + vector-reduced method (dispatch);
[`tgi_censored_lpdfs`](@ref) is the pointwise SB-mirror.
"""
function tgi_censored_lpdf(y::Number, mu::Number, sigma::Number, lloq_log::Number)
    z = (lloq_log - mu) / sigma
    lpdf = -0.5 * log(2pi) - log(sigma) - 0.5 * ((y - mu) / sigma)^2
    return ifelse(y <= lloq_log, tgi_normal_lcdf(z), lpdf)
end

"""Pointwise left-censored log-densities (SB `tgi_censored_lpdfs`)."""
tgi_censored_lpdfs(y::AbstractVector, mu::AbstractVector, sigma::Number,
        lloq_log::Number) = tgi_censored_lpdf.(y, mu, sigma, lloq_log)

"""Reduced left-censored log-density (SB `tgi_censored_lpdf`)."""
tgi_censored_lpdf(y::AbstractVector, mu::AbstractVector, sigma::Number,
        lloq_log::Number) = sum(tgi_censored_lpdfs(y, mu, sigma, lloq_log))

# --- cell vocabulary admission ---------------------------------------------

"""
Admitted TGI cell-call spellings (grouped-kernel cells may call these by
name; the foundation wires this tuple into its checker). Scalar cores take
`Number` (`Real`/`Integer` would reject Reactant's `TracedRNumber`, which
is a `Number` but neither — bound codes trace as constants); vector forms
take `AbstractVector` (broadcast + reduce — traceable, no scalar
indexing); `tgi_running_nadir` is the host loop (kernel-side nadirs use
[`tgi_nadir_scan_expr`](@ref)).
"""
const TGI_CELL_FUNCTIONS = (
    :tgi_ratio_loglinear, :tgi_ratio_resistant, :tgi_log_survival,
    :tgi_running_nadir, :tgi_interval_logprob, :tgi_report_logprob,
    :tgi_category_lpmf, :tgi_category_lpmfs,
    :tgi_response_lpmf, :tgi_response_lpmfs,
    :tgi_censored_lpdf, :tgi_censored_lpdfs,
    :tgi_normal_lcdf, :tgi_log_diff_exp, :tgi_inv_logit,
)

# --- per-family lowering builders (grouped emitter) ---------------------------
#
# One statements builder per TGI observation family, mirroring the QT
# slice's `_qt_joint_qt/pk_likelihood_stmts` over `_plate_sum_stmts`
# (pointwise plate + scalar sum node): each returns `(stmts, term)`,
# `term` joining the kernel's likelihood sum. The plate cell per
# element is the scalar `tgi_*` lpmf/lpdf over threaded dovars. Value
# refs take Symbol-or-Real (the `_thread_ref!` precedent: symbols
# thread as plate inputs, reals inline); `response` is always the
# bound response column; `label` scopes the pw/node names.

"""
    tgi_category_stmts(; response, r, ref, c_cr, c_pr, c_pd, sigma, eps, label)

RECIST-category likelihood plate: `TgiCategory.(r, ref, c_cr, c_pr,
c_pd, sigma, eps)` over the bound category codes (one scalar
[`tgi_category_lpmf`](@ref) per element). Cutpoints ride as per-row
vectors (Symbol) or fixed literals (Real, inlined).
"""
function tgi_category_stmts(; response::Symbol, r, ref, c_cr, c_pr, c_pd,
        sigma, eps, label::Symbol)
    klabel = Symbol(:tgi_category_, label)
    pw = _pw_name(klabel)
    node = _lik_name(klabel)
    inputs = Any[response]
    rv = _thread_ref!(inputs, r)
    refv = _thread_ref!(inputs, ref)
    ccrv = _thread_ref!(inputs, c_cr)
    cprv = _thread_ref!(inputs, c_pr)
    cpdv = _thread_ref!(inputs, c_pd)
    sv = _thread_ref!(inputs, sigma)
    epsv = _thread_ref!(inputs, eps)
    yv = _dovar(1)
    cell = :(tgi_category_lpmf($yv, $rv, $refv, $ccrv, $cprv, $cpdv, $sv,
        $epsv))
    return Expr[_plate_sum_stmts(pw, node, inputs, cell)...], node
end

"""
    tgi_response_stmts(; response, r, ref, c_pr, c_pd, sigma, eps, label)

Response-indicator likelihood plate: `TgiResponse.(r, ref, c_pr, c_pd,
sigma, eps)` over the bound response codes (one scalar
[`tgi_response_lpmf`](@ref) per element).
"""
function tgi_response_stmts(; response::Symbol, r, ref, c_pr, c_pd, sigma,
        eps, label::Symbol)
    klabel = Symbol(:tgi_response_, label)
    pw = _pw_name(klabel)
    node = _lik_name(klabel)
    inputs = Any[response]
    rv = _thread_ref!(inputs, r)
    refv = _thread_ref!(inputs, ref)
    cprv = _thread_ref!(inputs, c_pr)
    cpdv = _thread_ref!(inputs, c_pd)
    sv = _thread_ref!(inputs, sigma)
    epsv = _thread_ref!(inputs, eps)
    yv = _dovar(1)
    cell = :(tgi_response_lpmf($yv, $rv, $refv, $cprv, $cpdv, $sv, $epsv))
    return Expr[_plate_sum_stmts(pw, node, inputs, cell)...], node
end

"""
    tgi_censored_stmts(; response, mu, sigma, lloq, label)

Censored-Gaussian likelihood plate: `TgiCensored.(mu, sigma, lloq)`
over the bound log measurements (one scalar
[`tgi_censored_lpdf`](@ref) per element).
"""
function tgi_censored_stmts(; response::Symbol, mu, sigma, lloq,
        label::Symbol)
    klabel = Symbol(:tgi_censored_, label)
    pw = _pw_name(klabel)
    node = _lik_name(klabel)
    inputs = Any[response]
    muv = _thread_ref!(inputs, mu)
    sv = _thread_ref!(inputs, sigma)
    llqv = _thread_ref!(inputs, lloq)
    yv = _dovar(1)
    cell = :(tgi_censored_lpdf($yv, $muv, $sv, $llqv))
    return Expr[_plate_sum_stmts(pw, node, inputs, cell)...], node
end
