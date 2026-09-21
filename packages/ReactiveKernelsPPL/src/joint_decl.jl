# Joint PK+QT+TGI declaration block (W3a) — SB-mirror of the default-config
# `joint_pk_qt_tgi_brm1` subject-frame declarations through EXISTING thin-layer
# varying/preprocessing machinery (no new IR).
#
# SB source of truth (bruno `kb-impl/Bruno-arv393-tgi` @ `3af22846`):
# - PK/QT declarations: `web-pkpd/src/brm_integration.jl` joint body
#   (~lines 4149-4175) + `JOINT_PK_QT_FORMULAS_V2` (:207-229), audited end to
#   end by triple `web-pkpd/test/stan_audit_triples/joint_brm2_default.md`
#   (§1 BRM text, §2 SLIC, §3 Stan).
# - TGI declarations: `web-pkpd/src/brm_joint_tgi.jl` `_joint_tgi_body`
#   (:374-388, aggregate branch; no audit triple covers the tumor block, so
#   TGI terms cite SB source lines + the brm2-triple lowering pattern).
#
# COVERED (default config only — aggregate, direct_linear, gaussian QT obs,
# log_linear, lugano_ct, spd; everything else fails closed, sequenced
# separately): the 7 subject-frame LPs (`log_Vc log_k10 log_k12 log_k21
# log_ka qt_base qt_slope`) with fixed-effect covariates per
# `JOINT_PK_QT_FORMULAS_V2`, the shared 7-dim `|p|` block (`LKJCholesky(7,
# 2.0)`), the TGI `|tg|` 2-dim block (`LKJCholesky(2, 2.0)`), the continuous
# `|tb|` block (`tgi_ly0`), every `effect()` prior, and the scalar params
# (`sigma_add sigma_prop qt_scale tgi_sigma tgi_c_cr`).
#
# TGI observation is admitted as `:continuous` ONLY: the fixed HAVE seam
# carries `tgi_ly0`, which SB emits only under `observation == "continuous"`
# (`brm_joint_tgi.jl:342,374-378`); SB's own default (`"ordinal"`) has no
# `tgi_ly0` LP at all. Other observations change the declaration SET and are
# sequenced separately. `resistant_fraction` (`tgi_logit_phi`), `estimated`
# thresholds (`tgi_d_pr`/`tgi_d_pd`), per-lesion blocks (`|tgl|`/`|tbl|`),
# and regimes are likewise out (regimes touch likelihoods only — W3b).
# `log_F` + HSGP are the W2 event-LP lane; the grouped kernel + likelihoods
# are W3b assembly. `misclassification` is a fixed cell literal in SB, not a
# declaration — validated here, emitted nowhere.
#
# SB→thin-layer mapping (each proven term-for-term in `test_joint_decl.jl`):
# - `standardize(x)` → in-graph derived column `(x .- mean(x)) ./ std(x)`
#   (D5a: preprocessing computes in-graph from raw columns). Base reductions
#   are mathematically identical to BRM's fitted zscale
#   (`preparation_numeric.jl`, sample SD, n−1) but sum in a different order
#   (~ulp delta, documented in the W3a report — the in-graph vocabulary has
#   no loops, so BRM's iterative mean / scaled-sumsq is inexpressible).
# - `indication` (categorical, SB reference coding
#   `append_row(0, β)[idx]`, triple §3) → `FactorTerm` + `LevelMap`
#   `(2, :end)` subset: EXACT given SB's level order. SB numbers CA.levels
#   order (`["AITL", "BLCL"]`, `brm_integration.jl:2835,2837-2847`); the thin
#   layer sorts observed levels — identical for this level set, pinned by the
#   prep contract (both levels required, unknown levels rejected, as in SB).
# - `(1|ID|subject)` blocks → `VaryingDraws` `:correlated` + per-LP
#   single-column `VaryingSlice`s (SB `_sb_emit_id_bucket_sampling!` /
#   `_sb_emit_id_ranef_block!`, intercept fast path). `|tb|` (K=1) takes the
#   vacuous-1x1-LKJ `:correlated` route with SB's default eta 1.0 — SB routes
#   EVERY `|ID|` bucket through `ranef_correlated_draws_effect`, including
#   K=1 (`sbimpl.jl` `_sb_emit_id_buckets!`), so the tau-sampled geometry is
#   the mirror and `:intercept1` (log-scale/xi) would be wrong. The K=1 LKJ
#   term is exactly 0.0 on both sides.
# - `effect()` → `PopulationPrior` rows; scalars → `SampledParameter`s
#   (Distributions.jl scale semantics = SB's BRM spelling; SB emits Stan
#   rates as 1/scale, triple §3).
#
# CONSUMED FIXES (both snagged, both landed, both verified term-for-term in
# `test_joint_decl.jl` — no interims remain):
# - `sd()` marginal-scale priors (`thin-layer-varyi-c90f059f`, fix `6228160`,
#   canonical main `72a3aafc`): per-margin `VaryingSdPrior(:exponential, θ)`
#   with SB's scales (`JOINT_DECL_SD_SCALES`).
# - `tgi_c_cr` upper truncation (`thin-layer-upper-c3c06483`, fix `5c38658`,
#   canonical main `1aecc92`): `(:upper, hi)` with Stan kernel semantics.
#
# HAVE seam (parent W3b consumes exactly this): per-subject values for the 10
# LP symbols below. The `_ppl_lp_<name>` nodes hold LP values (logs for the 7
# log-LPs); constrained values are `exp.(LP)` for log-LPs, identity for
# `qt_base qt_slope tgi_ly0` (the constrained map is stated in the W3a
# report; predictor names follow SB LPs, so TGI log-LPs are `log_tgi_kg` /
# `log_tgi_kd`).

"""Default-config PK/QT subject-frame formula spellings the thin layer lowers
(SB `JOINT_PK_QT_FORMULAS_V2`, `brm_integration.jl:207-215` — byte-compared at
admission; any other spelling fails closed)."""
const JOINT_DECL_PK_FORMULAS_V2 = (
    log_vc = "1 + male + standardize(age_yr) + standardize(weight_kg) + indication",
    log_k10 = "1 + male + standardize(age_yr) + standardize(weight_kg) + indication",
    log_k12 = "1 + indication",
    log_k21 = "1 + indication",
    log_ka = "1 + indication",
    qt_base = "1 + male + standardize(age_yr) + indication + qt_prolonging_drug_ongoing",
    qt_slope = "1 + indication",
)

"""Default-config TGI LP formula spellings (SB `JOINT_TGI_FORMULAS_V1`,
`brm_joint_tgi.jl:68` — all intercept-only)."""
const JOINT_DECL_TGI_FORMULAS_V1 = (tgi_kg = "1", tgi_kd = "1", tgi_ly0 = "1")

"""Joint declaration LP symbols in SB bucket order (`|p|` 1..7, then `|tg|`,
then `|tb|`). TGI log-LPs are named by SB LP (`log_tgi_kg`), not by SB
constrained (`tgi_kg`) — the constrained map is `exp` for the 7 log-LPs,
identity for `qt_base qt_slope tgi_ly0`."""
const JOINT_DECL_LPS = (:log_Vc, :log_k10, :log_k12, :log_k21, :log_ka,
    :qt_base, :qt_slope, :log_tgi_kg, :log_tgi_kd, :tgi_ly0)

# SB population-intercept priors (loc, scale) per LP — baked literals from the
# joint body (`brm_integration.jl:4157-4161`: `log(vc_prior_median=10.0)`,
# `JOINT_PK_K10_PRIOR_LOG_MEDIAN`, k12/k21/k12/ka medians with
# `k12_k21_prior_sd=2.0`) and the TGI body (`brm_joint_tgi.jl:382-383`).
# `qt_prior_scale` and `tgi_baseline_log_size` are data-derived prep inputs,
# not constants (see `joint_decl_fragments`).
const JOINT_DECL_PK_INTERCEPT_PRIORS = Dict{Symbol,Tuple{Float64,Float64}}(
    :log_Vc => (2.302585092994046, 0.8),
    :log_k10 => (-1.405170185988091, 0.8),
    :log_k12 => (-0.34657359027997265, 2.0),
    :log_k21 => (-2.649158683274018, 2.0),
    :log_ka => (-2.0794415416798357, 0.8),
)
const JOINT_DECL_TGI_INTERCEPT_SCALES = Dict{Symbol,Tuple{Float64,Float64}}(
    :log_tgi_kg => (0.0, 1.0),
    :log_tgi_kd => (0.0, 1.5),
)

"""SB fixed population-covariate prior scales (PK block): male / standardized
age+weight 0.1 (`brm_integration.jl:3883-3892`), indication 1.0 (:3904-3908)."""
const JOINT_DECL_PK_COV_SCALES = (male = 0.1, standardize_age_yr = 0.1,
    standardize_weight_kg = 0.1, indication = 1.0)

"""SB `sd()` marginal-scale priors per bucket, Distributions.jl SCALE
(`sd(:, p) ~ Exponential(0.3333)` constant, `brm_integration.jl:3693`;
`sd(:, tg) ~ Exponential(0.5)` + `sd(:, tb) ~ Exponential(1.0)`,
`brm_joint_tgi.jl:384,377`). SB emits Stan rates 1/scale (3.0/2.0/1.0,
triple §2-§3). Carried as `VaryingSdPrior(:exponential, θ)` (fix
`6228160` — the IR takes the scale directly, no inversion)."""
const JOINT_DECL_SD_SCALES = (p = 0.3333333333333333, tg = 0.5, tb = 1.0)

"""SB `cor()` LKJ shapes per bucket (`cor(:, p) ~ LKJCholesky(7, 2.0)`,
triple §1; `cor(:, tg) ~ LKJCholesky(2, 2.0)`, `brm_joint_tgi.jl:385`;
`|tb|` K=1 takes SB's default eta 1.0 — vacuous, term exactly 0.0)."""
const JOINT_DECL_LKJ = (p = (7, 2.0), tg = (2, 2.0), tb = (1, 1.0))

"""SB residual-scale prior scale, V2 `raw_axes` (`pk_residual_scale = 0.25`,
`brm_integration.jl:3977`; `sigma_add/sigma_prop ~ Exponential(0.25)`)."""
const JOINT_DECL_PK_RESIDUAL_SCALE = 0.25

"""SB TGI scalar priors: `tgi_sigma ~ LogNormal(log(0.13), 0.5)`
(`brm_joint_tgi.jl:386`); `tgi_c_cr ~ Normal(-2.3, 1.0; upper=log(0.5))`
under default lugano_ct+spd (:363-369 — carried as `(:upper, hi)`, fix
`5c38658`)."""
const JOINT_DECL_TGI_SIGMA = (log(0.13), 0.5)
const JOINT_DECL_TGI_C_CR = (loc = -2.3, scale = 1.0, upper = log(0.5))

"""SB-declared indication levels (`_ARV393_INDICATION_LEVELS`,
`brm_integration.jl:2835`): CA.levels order = sort order — the (2,:end)
subset mirror rests on this equality (prep-pinned)."""
const JOINT_DECL_INDICATION_LEVELS = ("AITL", "BLCL")

"""TGI layouts the declarations lower (aggregate only; per-lesion blocks
`|tgl|`/`|tbl|` + ragged regrouping are sequenced separately)."""
const JOINT_DECL_TGI_LAYOUTS = (:aggregate,)
"""TGI observations the declarations lower (`tgi_ly0` exists only under
continuous — the HAVE seam fixes it, so nothing else is admitted)."""
const JOINT_DECL_TGI_OBSERVATIONS = (:continuous,)
"""TGI structures the declarations lower (`resistant_fraction` owns
`tgi_logit_phi`, sequenced separately)."""
const JOINT_DECL_TGI_STRUCTURES = (:log_linear,)
"""TGI thresholds the declarations lower (`recist` rescales, `estimated` owns
`tgi_d_pr`/`tgi_d_pd` — both sequenced separately)."""
const JOINT_DECL_TGI_THRESHOLDS = (:lugano_ct,)
"""TGI measures the declarations lower (`sld` halves log thresholds and
reinterprets rates — sequenced separately)."""
const JOINT_DECL_TGI_MEASURES = (:spd,)

"""Admit one joint-declaration option, fail-closed naming the admitted
spellings (the `admit_qt_spine` precedent)."""
function _admit_joint_decl(what::AbstractString, got::Symbol,
        admitted::Tuple{Vararg{Symbol}})
    got in admitted ||
        throw(ContractValidationError("[joint_decl] $what `$got` is not " *
              "admitted (admitted: " * join(map(repr, admitted), ", ") *
              "; everything else is sequenced separately)"))
    return got
end

admit_joint_decl_tgi_layout(v::Symbol) =
    _admit_joint_decl("TGI layout", v, JOINT_DECL_TGI_LAYOUTS)
admit_joint_decl_tgi_observation(v::Symbol) =
    _admit_joint_decl("TGI observation", v, JOINT_DECL_TGI_OBSERVATIONS)
admit_joint_decl_tgi_structure(v::Symbol) =
    _admit_joint_decl("TGI structure", v, JOINT_DECL_TGI_STRUCTURES)
admit_joint_decl_tgi_thresholds(v::Symbol) =
    _admit_joint_decl("TGI thresholds", v, JOINT_DECL_TGI_THRESHOLDS)
admit_joint_decl_tgi_measure(v::Symbol) =
    _admit_joint_decl("TGI measure", v, JOINT_DECL_TGI_MEASURES)

"""Admit the PK/QT subject-frame formula spellings: byte-equal to
`JOINT_DECL_PK_FORMULAS_V2` (SB `JOINT_PK_QT_FORMULAS_V2`) or fail closed
naming the offending margin. The builder does not parse formulas — the
admitted spelling selects the fixed term structure below."""
function admit_joint_decl_pk_formulas(formulas::NamedTuple)
    want = JOINT_DECL_PK_FORMULAS_V2
    keys(formulas) == keys(want) ||
        throw(ContractValidationError("[joint_decl] PK/QT formulas carry " *
              "margins $(keys(formulas)), want $(keys(want)) (default " *
              "config only; anything else is sequenced separately)"))
    for k in keys(want)
        string(formulas[k]) == string(want[k]) ||
            throw(ContractValidationError("[joint_decl] PK/QT formula " *
                  "`$k` is $(repr(string(formulas[k]))), want " *
                  "$(repr(string(want[k]))) (default config only)"))
    end
    return formulas
end

"""Admit the TGI LP formula spellings: byte-equal to
`JOINT_DECL_TGI_FORMULAS_V1` (all `"1"`) or fail closed."""
function admit_joint_decl_tgi_formulas(formulas::NamedTuple)
    want = JOINT_DECL_TGI_FORMULAS_V1
    keys(formulas) == keys(want) ||
        throw(ContractValidationError("[joint_decl] TGI formulas carry " *
              "margins $(keys(formulas)), want $(keys(want)) (default " *
              "config only)"))
    for k in keys(want)
        string(formulas[k]) == string(want[k]) ||
            throw(ContractValidationError("[joint_decl] TGI formula `$k` " *
                  "is $(repr(string(formulas[k]))), want \"1\" (covariate " *
                  "TGI margins are sequenced separately)"))
    end
    return formulas
end

"""
    validate_joint_decl_prep(; subject, male, age_yr, weight_kg, indication,
        qt_prolonging_drug_ongoing, qt_prior_scale, tgi_baseline_log_size,
        misclassification = 0.01) -> nothing

Bind-time prep contract for the joint declarations (SB prep mirror): every
fitted subject carries a finite 0/1 `male`, finite `age_yr`/`weight_kg`
with ≥2 values and nonzero sample variance (BRM zscale gates,
`preparation_numeric.jl`), a finite 0/1 `qt_prolonging_drug_ongoing`
(`brm_integration.jl:3138-3145`), and an `indication` in exactly the
SB-declared levels with BOTH levels present (SB errors on unknown levels,
`brm_integration.jl:2837-2847`; the (2,:end) subset mirror needs both —
degenerate single-level data is outside the default config). `subject`
names are unique (one row per fitted subject). `qt_prior_scale`
(SB `qt_effect_prior_width_ms / maximum(y_scale_ms)`, `brm_integration.jl:
3382,3400`) is finite positive; `tgi_baseline_log_size` (SB mean of
per-subject first log tumor sizes, `brm_joint_tgi.jl:517-523`) is finite;
`misclassification` keeps SB's own `[0, 0.5)` finite validation
(`brm_joint_tgi.jl:132-133`, default 0.01 — a W3b cell literal, emitted
nowhere here).
"""
function validate_joint_decl_prep(;
        subject::AbstractVector,
        male::AbstractVector,
        age_yr::AbstractVector{<:Real},
        weight_kg::AbstractVector{<:Real},
        indication::AbstractVector,
        qt_prolonging_drug_ongoing::AbstractVector,
        qt_prior_scale::Real,
        tgi_baseline_log_size::Real,
        misclassification::Real = 0.01)
    n = length(subject)
    n >= 1 ||
        throw(ContractValidationError("[joint_decl] subject frame is empty"))
    for (nm, col) in (("male", male), ("age_yr", age_yr),
            ("weight_kg", weight_kg), ("indication", indication),
            ("qt_prolonging_drug_ongoing", qt_prolonging_drug_ongoing))
        length(col) == n ||
            throw(ContractValidationError("[joint_decl] `$nm` has " *
                  "$(length(col)) rows, want $n (one row per fitted subject)"))
    end
    length(unique(subject)) == n ||
        throw(ContractValidationError("[joint_decl] `subject` names repeat " *
              "(one row per fitted subject)"))
    for (nm, col) in (("male", male),
            ("qt_prolonging_drug_ongoing", qt_prolonging_drug_ongoing))
        all(x -> x isa Real && isfinite(Float64(x)) &&
            Float64(x) in (0.0, 1.0), col) ||
            throw(ContractValidationError("[joint_decl] `$nm` must be " *
                  "finite 0/1 for every fitted subject (SB container " *
                  "formatter gate)"))
    end
    for (nm, col) in (("age_yr", age_yr), ("weight_kg", weight_kg))
        all(isfinite, col) ||
            throw(ContractValidationError("[joint_decl] `$nm` must be " *
                  "finite for every fitted subject"))
        n >= 2 ||
            throw(ContractValidationError("[joint_decl] `standardize($nm)` " *
                  "needs ≥2 fitted subjects for the sample SD (BRM zscale " *
                  "gate)"))
        var(col; corrected = true) > 0 ||
            throw(ContractValidationError("[joint_decl] `standardize($nm)` " *
                  "needs nonzero sample variance (BRM zscale gate)"))
    end
    got_levels = sort!(unique!(string.(indication)))
    want_levels = sort!(collect(String, JOINT_DECL_INDICATION_LEVELS))
    got_levels == want_levels ||
        throw(ContractValidationError("[joint_decl] `indication` levels " *
              "$(got_levels) ≠ SB-declared $(want_levels) (SB errors on " *
              "unknown levels; the (2,:end) reference mirror needs both " *
              "levels present)"))
    s = Float64(qt_prior_scale)
    isfinite(s) && s > 0 ||
        throw(ContractValidationError("[joint_decl] `qt_prior_scale` must " *
              "be finite and positive (SB qt_effect_prior_width_ms / " *
              "maximum(y_scale_ms)), got $(repr(qt_prior_scale))"))
    b = Float64(tgi_baseline_log_size)
    isfinite(b) ||
        throw(ContractValidationError("[joint_decl] " *
              "`tgi_baseline_log_size` must be finite (SB mean of " *
              "per-subject first log tumor sizes), got " *
              "$(repr(tgi_baseline_log_size))"))
    m = Float64(misclassification)
    isfinite(m) && 0 <= m < 0.5 ||
        throw(ContractValidationError("[joint_decl] `misclassification` " *
              "must lie in [0, 0.5) (SB's own validation; default 0.01), " *
              "got $(repr(misclassification))"))
    return nothing
end

# --- IR builders (emitter-shaped, SB order) ---

"""In-graph `standardize()` derived columns (D5a): SB `standardize_age_yr` /
`standardize_weight_kg`, shared by every LP that names them (SB materializes
one column per standardized covariate)."""
function joint_decl_derived()
    return VectorAssignmentSpec[
        VectorAssignmentSpec(:standardize_age_yr,
            :((age_yr .- mean(age_yr)) ./ std(age_yr)), :standardize_age_yr),
        VectorAssignmentSpec(:standardize_weight_kg,
            :((weight_kg .- mean(weight_kg)) ./ std(weight_kg)),
            :standardize_weight_kg),
    ]
end

# One population term (addressee = label, the surface convention).
_joint_decl_intercept() =
    TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept, :Intercept)
_joint_decl_continuous(col::Symbol) =
    TermSpec(ContinuousTerm, [col], NamedTuple(), col, col)
_joint_decl_factor(col::Symbol) =
    TermSpec(FactorTerm, [col], NamedTuple(), col, col)
_joint_decl_varying(target::Symbol, draws::Symbol, suffix::String) =
    TermSpec(VaryingEffectTerm, [:subject], (draws = draws,),
        Symbol("r_", target, :_, suffix), Symbol("r_", target, :_, suffix))

"""The 10 subject-frame LPs in SB order (design order = SB X order: triple
§2 `hcat` sequence, indication factor last among fixed effects; the varying
term rides last — design-neutral)."""
function joint_decl_predictors()
    I = _joint_decl_intercept
    C = _joint_decl_continuous
    F = _joint_decl_factor
    pk_full = [
        I(), C(:male), C(:standardize_age_yr), C(:standardize_weight_kg),
        F(:indication),
    ]
    preds = PredictorSpec[
        PredictorSpec(:log_Vc, IdentityLink,
            [pk_full..., _joint_decl_varying(:log_Vc, :draws_p_subject,
                "p_subject")], :log_Vc),
        PredictorSpec(:log_k10, IdentityLink,
            [pk_full..., _joint_decl_varying(:log_k10, :draws_p_subject,
                "p_subject")], :log_k10),
        PredictorSpec(:log_k12, IdentityLink,
            [I(), F(:indication),
                _joint_decl_varying(:log_k12, :draws_p_subject,
                    "p_subject")], :log_k12),
        PredictorSpec(:log_k21, IdentityLink,
            [I(), F(:indication),
                _joint_decl_varying(:log_k21, :draws_p_subject,
                    "p_subject")], :log_k21),
        PredictorSpec(:log_ka, IdentityLink,
            [I(), F(:indication),
                _joint_decl_varying(:log_ka, :draws_p_subject,
                    "p_subject")], :log_ka),
        PredictorSpec(:qt_base, IdentityLink,
            [I(), C(:male), C(:standardize_age_yr),
                C(:qt_prolonging_drug_ongoing), F(:indication),
                _joint_decl_varying(:qt_base, :draws_p_subject,
                    "p_subject")], :qt_base),
        PredictorSpec(:qt_slope, IdentityLink,
            [I(), F(:indication),
                _joint_decl_varying(:qt_slope, :draws_p_subject,
                    "p_subject")], :qt_slope),
        PredictorSpec(:log_tgi_kg, IdentityLink,
            [I(), _joint_decl_varying(:log_tgi_kg, :draws_tg_subject,
                "tg_subject")], :log_tgi_kg),
        PredictorSpec(:log_tgi_kd, IdentityLink,
            [I(), _joint_decl_varying(:log_tgi_kd, :draws_tg_subject,
                "tg_subject")], :log_tgi_kd),
        PredictorSpec(:tgi_ly0, IdentityLink,
            [I(), _joint_decl_varying(:tgi_ly0, :draws_tb_subject,
                "tb_subject")], :tgi_ly0),
    ]
    return preds
end

"""Every `effect()` prior (triple §1 + `brm_joint_tgi.jl:382-383,376`).
`qt_prior_scale` / `tgi_baseline_log_size` are the data-derived prep inputs;
everything else is an SB-constant baked literal (cited in
`JOINT_DECL_PK_INTERCEPT_PRIORS` / `JOINT_DECL_PK_COV_SCALES` /
`JOINT_DECL_TGI_INTERCEPT_SCALES`)."""
function joint_decl_population_priors(;
        qt_prior_scale::Real, tgi_baseline_log_size::Real)
    qs = Float64(qt_prior_scale)
    tb = Float64(tgi_baseline_log_size)
    priors = PopulationPrior[]
    for lp in (:log_Vc, :log_k10)
        loc, sc = JOINT_DECL_PK_INTERCEPT_PRIORS[lp]
        push!(priors, PopulationPrior(lp, :Intercept, loc, sc))
        push!(priors, PopulationPrior(lp, :male, 0.0,
            JOINT_DECL_PK_COV_SCALES.male))
        push!(priors, PopulationPrior(lp, :standardize_age_yr, 0.0,
            JOINT_DECL_PK_COV_SCALES.standardize_age_yr))
        push!(priors, PopulationPrior(lp, :standardize_weight_kg, 0.0,
            JOINT_DECL_PK_COV_SCALES.standardize_weight_kg))
        push!(priors, PopulationPrior(lp, :indication, 0.0,
            JOINT_DECL_PK_COV_SCALES.indication))
    end
    for lp in (:log_k12, :log_k21, :log_ka)
        loc, sc = JOINT_DECL_PK_INTERCEPT_PRIORS[lp]
        push!(priors, PopulationPrior(lp, :Intercept, loc, sc))
        push!(priors, PopulationPrior(lp, :indication, 0.0,
            JOINT_DECL_PK_COV_SCALES.indication))
    end
    push!(priors, PopulationPrior(:qt_base, :Intercept, 0.0, qs))
    push!(priors, PopulationPrior(:qt_base, :male, 0.0, qs))
    push!(priors, PopulationPrior(:qt_base, :standardize_age_yr, 0.0, qs))
    push!(priors, PopulationPrior(:qt_base, :qt_prolonging_drug_ongoing, 0.0,
        qs))
    push!(priors, PopulationPrior(:qt_base, :indication, 0.0, qs))
    push!(priors, PopulationPrior(:qt_slope, :Intercept, 0.0, qs))
    push!(priors, PopulationPrior(:qt_slope, :indication, 0.0, qs))
    for lp in (:log_tgi_kg, :log_tgi_kd)
        loc, sc = JOINT_DECL_TGI_INTERCEPT_SCALES[lp]
        push!(priors, PopulationPrior(lp, :Intercept, loc, sc))
    end
    push!(priors, PopulationPrior(:tgi_ly0, :Intercept, tb, 1.5))
    return priors
end

"""Indication `LevelMap`s: `(2, :end)` subset per PK/QT LP — the exact SB
reference-coding mirror (`append_row(0, β)[idx]`, triple §3) given the
prep-pinned level order. TGI LPs are intercept-only (no factor)."""
function joint_decl_levelmaps()
    return LevelMap[
        LevelMap(lp, :indication, [], :levels, (2, :end))
        for lp in (:log_Vc, :log_k10, :log_k12, :log_k21, :log_ka,
            :qt_base, :qt_slope)
    ]
end

"""The three shared draws blocks + their per-LP single-column slices (SB
bucket order; all group on the subject frame). Suffixes mirror SB bucket
names (`b_p_subject` → `p_subject`: `L_p_subject` / `tau_p_subject` /
`z_flat_p_subject`). `|tb|` takes the vacuous-1x1-LKJ `:correlated` route
with SB's default eta 1.0 (term exactly 0.0 — the `:intercept1`
log-scale/xi geometry would NOT mirror SB's tau-sampled K=1 bucket).
Each block carries its explicit `sd()` marginal-scale priors
(`VaryingSdPrior(:exponential, θ)` with SB's scales — fix `6228160`,
snag `thin-layer-varyi-c90f059f`); each block is uniform, so the generator
emits one tau plate per block."""
function joint_decl_varying()
    ones_margin() =
        VaryingMargin(:Intercept, VaryingZRecipe(:ones, :none, nothing))
    exp_prior(s) = VaryingSdPrior(:exponential, s)
    draws = VaryingDraws[
        VaryingDraws(:subject, :correlated, [ones_margin() for _ in 1:7],
            JOINT_DECL_LKJ.p[2], :draws_p_subject, "p_subject", nothing,
            [exp_prior(JOINT_DECL_SD_SCALES.p) for _ in 1:7]),
        VaryingDraws(:subject, :correlated, [ones_margin() for _ in 1:2],
            JOINT_DECL_LKJ.tg[2], :draws_tg_subject, "tg_subject", nothing,
            [exp_prior(JOINT_DECL_SD_SCALES.tg) for _ in 1:2]),
        VaryingDraws(:subject, :correlated, [ones_margin()],
            JOINT_DECL_LKJ.tb[2], :draws_tb_subject, "tb_subject", nothing,
            [exp_prior(JOINT_DECL_SD_SCALES.tb)]),
    ]
    slices = VaryingSlice[
        VaryingSlice(:draws_p_subject, j:j, lp)
        for (j, lp) in enumerate((:log_Vc, :log_k10, :log_k12, :log_k21,
            :log_ka, :qt_base, :qt_slope))
    ]
    push!(slices, VaryingSlice(:draws_tg_subject, 1:1, :log_tgi_kg))
    push!(slices, VaryingSlice(:draws_tg_subject, 2:2, :log_tgi_kd))
    push!(slices, VaryingSlice(:draws_tb_subject, 1:1, :tgi_ly0))
    return (draws = draws, slices = slices)
end

"""The scalar params (triple §1 + `brm_joint_tgi.jl:386,368`). `tgi_c_cr`
carries SB's `upper=log(0.5)` as `(:upper, hi)` (Stan kernel semantics —
fix `5c38658`, snag `thin-layer-upper-c3c06483`); the bound is folded to
the literal (PPL bounds are literal-only)."""
function joint_decl_scalars()
    return SampledParameter[
        SampledParameter(:sigma_add, :exponential,
            (arg1 = JOINT_DECL_PK_RESIDUAL_SCALE,), nothing, :sigma_add),
        SampledParameter(:sigma_prop, :exponential,
            (arg1 = JOINT_DECL_PK_RESIDUAL_SCALE,), nothing, :sigma_prop),
        SampledParameter(:qt_scale, :lognormal, (arg1 = 0.0, arg2 = 1.0),
            nothing, :qt_scale),
        SampledParameter(:tgi_sigma, :lognormal,
            (arg1 = JOINT_DECL_TGI_SIGMA[1], arg2 = JOINT_DECL_TGI_SIGMA[2]),
            nothing, :tgi_sigma),
        SampledParameter(:tgi_c_cr, :normal,
            (arg1 = JOINT_DECL_TGI_C_CR.loc,
                arg2 = JOINT_DECL_TGI_C_CR.scale),
            (:upper, -0.6931471805599453), # = log(0.5), folded literal
            :tgi_c_cr),
    ]
end

"""
    joint_decl_fragments(; pk_formulas, tgi_formulas, qt_spine, qt_obs_family,
        tgi_layout, tgi_observation, tgi_structure, tgi_thresholds, tgi_measure,
        qt_prior_scale, tgi_baseline_log_size, misclassification,
        vc_prior_median, k12_k21_prior_sd, qt_amplitude_prior_scale)
        -> (; predictors, population_priors, derived, levelmaps,
            varying_draws, varying_slices, parameters, prep)

The FULL default-config declaration block as emitter-shaped IR fragments (no
responses — W3b assembly adds the grouped kernel + likelihoods; the test
wraps a verification plan with dummy Gaussian responses around these).

Admission (fail-closed, default config only): formula spellings byte-compared
to SB's; QT spine/family via the `qt_joint.jl` admitters; TGI options via the
`admit_joint_decl_*` admitters above; SB scalar knobs pinned to their defaults
(`vc_prior_median=10.0`, `k12_k21_prior_sd=2.0`,
`qt_amplitude_prior_scale=1.0` — the baked intercept priors assume them).
`qt_prior_scale` / `tgi_baseline_log_size` are data-derived prep values
(finite/positive validated, NOT pinned — W4 passes SB's prepared numbers);
`misclassification` keeps SB's `[0, 0.5)` validation and rides `prep` to W3b's
cell literal (it is not a declaration).
"""
function joint_decl_fragments(;
        pk_formulas::NamedTuple = JOINT_DECL_PK_FORMULAS_V2,
        tgi_formulas::NamedTuple = JOINT_DECL_TGI_FORMULAS_V1,
        qt_spine::Symbol = :direct_linear,
        qt_obs_family::Symbol = :gaussian,
        tgi_layout::Symbol = :aggregate,
        tgi_observation::Symbol = :continuous,
        tgi_structure::Symbol = :log_linear,
        tgi_thresholds::Symbol = :lugano_ct,
        tgi_measure::Symbol = :spd,
        qt_prior_scale::Real,
        tgi_baseline_log_size::Real,
        misclassification::Real = 0.01,
        vc_prior_median::Real = 10.0,
        k12_k21_prior_sd::Real = 2.0,
        qt_amplitude_prior_scale::Real = 1.0)
    admit_joint_decl_pk_formulas(pk_formulas)
    admit_joint_decl_tgi_formulas(tgi_formulas)
    admit_qt_spine(qt_spine)
    admit_qt_obs_family(qt_obs_family)
    admit_joint_decl_tgi_layout(tgi_layout)
    admit_joint_decl_tgi_observation(tgi_observation)
    admit_joint_decl_tgi_structure(tgi_structure)
    admit_joint_decl_tgi_thresholds(tgi_thresholds)
    admit_joint_decl_tgi_measure(tgi_measure)
    vc_prior_median == 10.0 ||
        throw(ContractValidationError("[joint_decl] `vc_prior_median` " *
              "admitted only at the SB default 10.0 (the baked log_Vc " *
              "intercept prior assumes it), got " *
              repr(vc_prior_median)))
    k12_k21_prior_sd == 2.0 ||
        throw(ContractValidationError("[joint_decl] `k12_k21_prior_sd` " *
              "admitted only at the SB default 2.0, got " *
              repr(k12_k21_prior_sd)))
    qt_amplitude_prior_scale == 1.0 ||
        throw(ContractValidationError("[joint_decl] " *
              "`qt_amplitude_prior_scale` admitted only at the SB default " *
              "1.0, got $(repr(qt_amplitude_prior_scale))"))
    qs = Float64(qt_prior_scale)
    isfinite(qs) && qs > 0 ||
        throw(ContractValidationError("[joint_decl] `qt_prior_scale` must " *
              "be finite and positive (SB qt_effect_prior_width_ms / " *
              "maximum(y_scale_ms)), got $(repr(qt_prior_scale))"))
    tb = Float64(tgi_baseline_log_size)
    isfinite(tb) ||
        throw(ContractValidationError("[joint_decl] " *
              "`tgi_baseline_log_size` must be finite (SB mean of " *
              "per-subject first log tumor sizes), got " *
              "$(repr(tgi_baseline_log_size))"))
    m = Float64(misclassification)
    isfinite(m) && 0 <= m < 0.5 ||
        throw(ContractValidationError("[joint_decl] `misclassification` " *
              "must lie in [0, 0.5) (SB's own validation; default 0.01), " *
              "got $(repr(misclassification))"))
    varying = joint_decl_varying()
    return (;
        predictors = joint_decl_predictors(),
        population_priors = joint_decl_population_priors(;
            qt_prior_scale = qs, tgi_baseline_log_size = tb),
        derived = joint_decl_derived(),
        levelmaps = joint_decl_levelmaps(),
        varying_draws = varying.draws,
        varying_slices = varying.slices,
        parameters = joint_decl_scalars(),
        prep = (; qt_prior_scale = qs, tgi_baseline_log_size = tb,
            misclassification = m),
    )
end
