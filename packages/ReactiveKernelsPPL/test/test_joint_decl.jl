# Joint PK+QT+TGI declarations (W3a): SB-mirror term-for-term proofs for the
# default-config `joint_pk_qt_tgi_brm1` subject-frame declaration block,
# plus the memo-config (`net_kill` + recist/sld) shape proofs.
#
# SB truth: bruno `kb-impl/Bruno-arv393-tgi` @ `3af22846` — PK/QT terms cite
# the audit triple `web-pkpd/test/stan_audit_triples/joint_brm2_default.md`
# (§1 BRM text / §2 SLIC / §3 Stan); TGI terms cite `brm_joint_tgi.jl` source
# lines + the brm2-triple lowering pattern (no triple covers the tumor
# block). Every prior term below names its SB anchor in a comment.
#
# The verification plan wraps the IR fragments (`joint_decl_fragments`) with
# 10 dummy Gaussian responses (one per LP, distinct literal scales) over a
# 4-subject frame, so `_query(:prior)` proves every prior term and
# `_query(:likelihood)` proves all 10 LP value vectors end to end against
# independent hand oracles (per-row Distributions.jl loops + explicit
# per-margin/per-group ranef loops, never the fused forms). Gradients go
# through Enzyme-vs-findiff (`_check_gradient`, needs `test_generator.jl`
# included first — see `runtests.jl` order).
#
# CONSUMED FIXES (no interims remain — every prior term below is SB-explicit):
# - `thin-layer-varyi-c90f059f` (fix `6228160`, canonical main `72a3aafc`):
#   tau priors are the joint's explicit `sd() ~ Exponential` via per-margin
#   `VaryingSdPrior(:exponential, θ)` (SB scales pass through directly).
# - `thin-layer-upper-c3c06483` (fix `5c38658`, canonical main `1aecc92`):
#   `tgi_c_cr` carries `(:upper, hi)` with Stan kernel semantics (plain
#   normal_lpdf + bare-u Jacobian, support enforced).

using Distributions: Normal, Exponential, LogNormal, logpdf
using SpecialFunctions: loggamma

# --- fixture: 4-subject frame + 10 dummy response columns ---

_jd_subject() = ["S01", "S02", "S03", "S04"]
_jd_male() = [0.0, 1.0, 0.0, 1.0]
_jd_age() = [45.0, 62.0, 38.0, 55.0]
_jd_weight() = [70.0, 88.0, 62.0, 95.0]
_jd_indication() = ["AITL", "BLCL", "AITL", "BLCL"]
_jd_ongoing() = [0.0, 0.0, 1.0, 0.0]

# Triple §1 value (10.0/12.0) + a representative TGI baseline (the corpus
# program pins the same literals — see the twin test below).
_jd_qt_scale() = 10.0 / 12.0
_jd_tgi_baseline() = 3.5

# Dummy response columns + distinct literal scales (corpus 49 pins the same
# scales; distinctness makes LP↔response wiring mixups move the likelihood).
const _JD_DUMMY = (
    (lp = :log_Vc, resp = :ydVc, scale = 1.5),
    (lp = :log_k10, resp = :ydK10, scale = 1.6),
    (lp = :log_k12, resp = :ydK12, scale = 1.7),
    (lp = :log_k21, resp = :ydK21, scale = 1.8),
    (lp = :log_ka, resp = :ydKa, scale = 1.9),
    (lp = :qt_base, resp = :ydQb, scale = 2.0),
    (lp = :qt_slope, resp = :ydQs, scale = 2.1),
    (lp = :log_tgi_kg, resp = :ydTg, scale = 2.2),
    (lp = :log_tgi_kd, resp = :ydKd, scale = 2.3),
    (lp = :tgi_ly0, resp = :ydLy0, scale = 2.4),
)

function _jd_columns()
    cols = Dict{Symbol,AbstractVector}(
        :subject => _jd_subject(),
        :male => _jd_male(),
        :age_yr => _jd_age(),
        :weight_kg => _jd_weight(),
        :indication => _jd_indication(),
        :qt_prolonging_drug_ongoing => _jd_ongoing(),
    )
    for (j, d) in enumerate(_JD_DUMMY)
        cols[d.resp] = Float64[j + i / 10 for i in 1:4]
    end
    return cols
end

function _jd_responses()
    return LikelihoodSpec[
        LikelihoodSpec(GaussianFam, IdentityLink, d.resp, d.lp, d.scale,
            nothing, ResponseEvidence(:none, nothing, nothing),
            Symbol(d.resp, :_resp))
        for d in _JD_DUMMY
    ]
end

function _jd_plan()
    f = joint_decl_fragments(qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    return StructuralPlan(_jd_responses(), f.predictors, f.population_priors,
        f.parameters, AssignmentSpec[], Dict{Symbol,AbstractVector}(), 4;
        derived = f.derived, levelmaps = f.levelmaps,
        varying_draws = f.varying_draws, varying_slices = f.varying_slices)
end

@testset "joint decl admission" begin
    # Happy path admits.
    f = joint_decl_fragments(qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test length(f.predictors) == 10
    # Formula spellings: byte-compared to SB's.
    badpk = merge(JOINT_DECL_PK_FORMULAS_V2, (; log_ka = "1 + male"))
    @test_throws ContractValidationError joint_decl_fragments(;
        pk_formulas = badpk, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    badtgi = merge(JOINT_DECL_TGI_FORMULAS_V1, (; tgi_kg = "1 + male"))
    @test_throws ContractValidationError joint_decl_fragments(;
        tgi_formulas = badtgi, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    # QT spine/family ride the qt_joint admitters.
    @test_throws ContractValidationError joint_decl_fragments(;
        qt_spine = :direct_emax, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        qt_obs_family = :student_t, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    # TGI options: default only.
    @test_throws ContractValidationError joint_decl_fragments(;
        tgi_layout = :per_lesion, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        tgi_observation = :ordinal, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        tgi_structure = :resistant_fraction, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        tgi_thresholds = :estimated, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        tgi_measure = :sld, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    # SB scalar knobs: pinned to defaults (the baked priors assume them).
    @test_throws ContractValidationError joint_decl_fragments(;
        vc_prior_median = 12.0, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        k12_k21_prior_sd = 1.0, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        qt_amplitude_prior_scale = 2.0, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    # Prep values: validated, not pinned.
    @test_throws ContractValidationError joint_decl_fragments(;
        qt_prior_scale = -1.0, tgi_baseline_log_size = _jd_tgi_baseline())
    @test_throws ContractValidationError joint_decl_fragments(;
        qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = Inf)
    @test_throws ContractValidationError joint_decl_fragments(;
        qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline(), misclassification = 0.5)
    prep = f.prep
    @test (prep.qt_prior_scale, prep.tgi_baseline_log_size,
        prep.misclassification) == (_jd_qt_scale(), _jd_tgi_baseline(), 0.01)
end

@testset "joint decl prep validation" begin
    good = (; subject = _jd_subject(), male = _jd_male(),
        age_yr = _jd_age(), weight_kg = _jd_weight(),
        indication = _jd_indication(),
        qt_prolonging_drug_ongoing = _jd_ongoing(),
        qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    @test validate_joint_decl_prep(; good...) === nothing
    # Ragged frame / repeated subjects.
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., male = [0.0, 1.0])
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., subject = ["S01", "S01", "S03", "S04"])
    # Non-binary / non-finite covariates.
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., male = [0.0, 1.0, 0.5, 1.0])
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., age_yr = [45.0, 62.0, NaN, 55.0])
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., qt_prolonging_drug_ongoing = [0.0, 0.0, 2.0, 0.0])
    # BRM zscale gates: ≥2 subjects + nonzero variance.
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., weight_kg = [70.0, 70.0, 70.0, 70.0])
    # Indication: SB-declared levels, both present (SB errors on unknown;
    # the (2,:end) mirror needs the reference level observed).
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., indication = ["AITL", "BLCL", "AITL", "DLBCL"])
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., indication = ["BLCL", "BLCL", "BLCL", "BLCL"])
    # Prep values.
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., qt_prior_scale = 0.0)
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., tgi_baseline_log_size = NaN)
    @test_throws ContractValidationError validate_joint_decl_prep(;
        good..., misclassification = -0.1)
end

@testset "joint decl fragments shape" begin
    f = joint_decl_fragments(qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    # Predictor roster in SB order (triple §1 LP order, TGI bodies after).
    @test [p.name for p in f.predictors] == collect(JOINT_DECL_LPS)
    @test all(p -> p.link === IdentityLink, f.predictors)
    byname = Dict(p.name => p for p in f.predictors)
    # Design order = SB X order (triple §2 hcat sequence; indication last
    # among fixed effects; the varying term rides last, design-neutral).
    @test [(t.kind, t.columns) for t in byname[:log_Vc].terms] ==
        [(InterceptTerm, Symbol[]), (ContinuousTerm, [:male]),
            (ContinuousTerm, [:standardize_age_yr]),
            (ContinuousTerm, [:standardize_weight_kg]),
            (FactorTerm, [:indication]),
            (VaryingEffectTerm, [:subject])]
    @test [(t.kind, t.columns) for t in byname[:qt_base].terms] ==
        [(InterceptTerm, Symbol[]), (ContinuousTerm, [:male]),
            (ContinuousTerm, [:standardize_age_yr]),
            (ContinuousTerm, [:qt_prolonging_drug_ongoing]),
            (FactorTerm, [:indication]),
            (VaryingEffectTerm, [:subject])]
    @test [(t.kind, t.columns) for t in byname[:qt_slope].terms] ==
        [(InterceptTerm, Symbol[]), (FactorTerm, [:indication]),
            (VaryingEffectTerm, [:subject])]
    @test [(t.kind, t.columns) for t in byname[:log_tgi_kg].terms] ==
        [(InterceptTerm, Symbol[]), (VaryingEffectTerm, [:subject])]
    @test byname[:log_Vc].terms[end].options.draws === :draws_p_subject
    @test byname[:log_tgi_kg].terms[end].options.draws ===
        :draws_tg_subject
    @test byname[:tgi_ly0].terms[end].options.draws === :draws_tb_subject
    # Derived standardize columns (SB `standardize_*`, one column each).
    @test [d.name for d in f.derived] ==
        [:standardize_age_yr, :standardize_weight_kg]
    # Indication maps: (2,:end) subset per PK/QT LP (TGI LPs have none).
    @test [(m.predictor, m.column, m.subset) for m in f.levelmaps] ==
        [(lp, :indication, (2, :end))
            for lp in (:log_Vc, :log_k10, :log_k12, :log_k21, :log_ka,
                :qt_base, :qt_slope)]
    # Draws: |p| K=7 eta 2.0, |tg| K=2 eta 2.0, |tb| K=1 eta 1.0 vacuous.
    @test [(d.label, d.kind, d.lkj_eta, length(d.margins))
        for d in f.varying_draws] ==
        [(:draws_p_subject, :correlated, 2.0, 7),
            (:draws_tg_subject, :correlated, 2.0, 2),
            (:draws_tb_subject, :correlated, 1.0, 1)]
    @test all(d -> d.group === :subject, f.varying_draws)
    @test all(d -> all(m -> m.coefficient === :Intercept &&
            m.z.kind === :ones, d.margins), f.varying_draws)
    # Explicit sd() scales (SB `sd(:, p/tg/tb)`, fix 6228160).
    @test [(p.family, p.param) for p in f.varying_draws[1].sd_priors] ==
        fill((:exponential, 0.3333333333333333), 7)
    @test [(p.family, p.param) for p in f.varying_draws[2].sd_priors] ==
        fill((:exponential, 0.5), 2)
    @test [(p.family, p.param) for p in f.varying_draws[3].sd_priors] ==
        [(:exponential, 1.0)]
    # Slices partition each block once, in SB bucket order.
    @test [(s.draws, s.columns, s.target) for s in f.varying_slices] ==
        vcat(
            [(:draws_p_subject, j:j, lp)
                for (j, lp) in enumerate((:log_Vc, :log_k10, :log_k12,
                    :log_k21, :log_ka, :qt_base, :qt_slope))],
            [(:draws_tg_subject, 1:1, :log_tgi_kg),
                (:draws_tg_subject, 2:2, :log_tgi_kd),
                (:draws_tb_subject, 1:1, :tgi_ly0)],
        )
    # effect() inventory: 26 rows (triple §1 + brm_joint_tgi.jl:376-383).
    pr = Dict((p.predictor, p.addressee) => (p.location, p.scale)
        for p in f.population_priors)
    @test length(f.population_priors) == 26
    @test pr[(:log_Vc, :Intercept)] == (2.302585092994046, 0.8)
    @test pr[(:log_k10, :Intercept)] == (-1.405170185988091, 0.8)
    @test pr[(:log_k12, :Intercept)] == (-0.34657359027997265, 2.0)
    @test pr[(:log_k21, :Intercept)] == (-2.649158683274018, 2.0)
    @test pr[(:log_ka, :Intercept)] == (-2.0794415416798357, 0.8)
    @test pr[(:log_Vc, :male)] == (0.0, 0.1)
    @test pr[(:log_k10, :standardize_weight_kg)] == (0.0, 0.1)
    @test pr[(:log_ka, :indication)] == (0.0, 1.0)
    qs = _jd_qt_scale()
    @test pr[(:qt_base, :Intercept)] == (0.0, qs)
    @test pr[(:qt_base, :male)] == (0.0, qs)
    @test pr[(:qt_base, :standardize_age_yr)] == (0.0, qs)
    @test pr[(:qt_base, :qt_prolonging_drug_ongoing)] == (0.0, qs)
    @test pr[(:qt_base, :indication)] == (0.0, qs)
    @test pr[(:qt_slope, :Intercept)] == (0.0, qs)
    @test pr[(:qt_slope, :indication)] == (0.0, qs)
    @test pr[(:log_tgi_kg, :Intercept)] == (0.0, 1.0)
    @test pr[(:log_tgi_kd, :Intercept)] == (0.0, 1.5)
    @test pr[(:tgi_ly0, :Intercept)] == (_jd_tgi_baseline(), 1.5)
    # Scalars (triple §1 + brm_joint_tgi.jl:386,368; tgi_c_cr upper via
    # fix 5c38658, folded literal = log(0.5)).
    sc = Dict(p.name => (p.family, p.args, p.support_override)
        for p in f.parameters)
    @test sc[:sigma_add] == (:exponential, (arg1 = 0.25,), nothing)
    @test sc[:sigma_prop] == (:exponential, (arg1 = 0.25,), nothing)
    @test sc[:qt_scale] == (:lognormal, (arg1 = 0.0, arg2 = 1.0), nothing)
    @test sc[:tgi_sigma] ==
        (:lognormal, (arg1 = -2.0402208285265546, arg2 = 0.5), nothing)
    @test sc[:tgi_c_cr] ==
        (:normal, (arg1 = -2.3, arg2 = 1.0), (:upper, -0.6931471805599453))
end

# --- independent oracles (fresh code, never the fused forms) ---

# BRM zscale, verbatim port of `_brm_fit_zscale_numeric` + `_brm_apply_zscale`
# (`BayesianRegressionModels.jl/src/backend_plan.jl`, via
# `preparation_numeric.jl`): iterative mean + scaled-sumsq sample SD (n−1).
# Bounds the honest ~ulp delta between SB's fitted loop and the in-graph
# Base reductions (loops are inexpressible in-graph).
function _jd_brm_zscale(xs::AbstractVector{<:Real})
    m = float(first(xs))
    for (off, v) in enumerate(Iterators.drop(xs, 1))
        c = off + 1
        m += float(v) / c - m / c
    end
    ms, ss = 0.0, 0.0
    for v in xs
        mag = abs(float(v) - m)
        iszero(mag) && continue
        if ms < mag
            r = ms / mag
            ms, ss = mag, 1.0 + ss * r * r
        else
            r = mag / ms
            ss += r * r
        end
    end
    s = ms * sqrt(ss / (length(xs) - 1))
    return (xs .- m) ./ s
end

# Stan `lkj_corr_cholesky_lpdf`, transcribed fresh from the published math
# (LKJ09 theorem 5 as Stan implements it: general-eta constant via lgamma
# sums + per-diagonal `[(K-i) + 2(eta-1)] log L[ii]`, rows 2..K; K=1 is
# exactly 0.0). Cross-checks the thin-layer node on the joint's exact inputs
# (K=7/2, eta 2.0); the LKJ math itself is proven in `test_varying.jl`
# (K=2 closed form) + `test_lkj_jacobian.jl` (volume element).
function _jd_stan_lkj(L::AbstractMatrix, eta::Real)
    K = size(L, 1)
    K == 1 && return 0.0
    e = Float64(eta)
    Km1 = K - 1
    c = Km1 * loggamma(e + 0.5 * Km1)
    for k in 1:Km1
        c -= 0.5 * k * log(pi) + loggamma(e + 0.5 * (Km1 - k))
    end
    for i in 2:K
        c += ((K - i) + 2 * e - 2) * log(Float64(L[i, i]))
    end
    return c
end

# One correlated block's per-row contributions, explicit per-margin/per-group
# loops over the SB `(diag(tau)*L*z)'` shape (column-major `z_flat`,
# `s + (g-1)*K`; ones-Z needs no multiply — the SB intercept fast path).
function _jd_ref_r(idx::Vector{Int}, L::AbstractMatrix, tau::AbstractVector,
        zflat::AbstractVector, js::UnitRange{Int})
    K = length(tau)
    r = zeros(Float64, length(idx))
    for m in eachindex(idx)
        g = idx[m]
        for j in js
            acc = 0.0
            for s in 1:j
                acc += tau[j] * L[j, s] * zflat[s + (g - 1) * K]
            end
            r[m] += acc
        end
    end
    return r
end

# Full LP oracle: SB transformed-parameters math per LP (pop design + BLCL
# reference contrast + ranef slice) from constrained values.
function _jd_oracle_lps(nt, cols)
    male = cols[:male]
    sage = (cols[:age_yr] .- sum(cols[:age_yr]) / 4) ./
        sqrt(sum((x - sum(cols[:age_yr]) / 4)^2 for x in cols[:age_yr]) / 3)
    swt = (cols[:weight_kg] .- sum(cols[:weight_kg]) / 4) ./
        sqrt(sum((x - sum(cols[:weight_kg]) / 4)^2
            for x in cols[:weight_kg]) / 3)
    blcl = Float64.(cols[:indication] .== "BLCL")
    one = ones(4)
    idx = [findfirst(==(v), ["S01", "S02", "S03", "S04"])
        for v in cols[:subject]]
    Lp = Matrix(nt.L_p_subject)
    Ltg = Matrix(nt.L_tg_subject)
    Ltb = Matrix(nt.L_tb_subject)
    @assert Ltb == [1.0;;] # vacuous 1x1
    # Per-LP: design columns in SB X order + ranef margin slice.
    coef = Dict(
        :log_Vc => Vector(nt.log_Vc), :log_k10 => Vector(nt.log_k10),
        :log_k12 => Vector(nt.log_k12), :log_k21 => Vector(nt.log_k21),
        :log_ka => Vector(nt.log_ka), :qt_base => Vector(nt.qt_base),
        :qt_slope => Vector(nt.qt_slope),
        :log_tgi_kg => Vector(nt.log_tgi_kg),
        :log_tgi_kd => Vector(nt.log_tgi_kd),
        :tgi_ly0 => Vector(nt.tgi_ly0),
    )
    pop = Dict{Symbol,Vector{Float64}}(
        :log_Vc => coef[:log_Vc][1] .* one .+ coef[:log_Vc][2] .* male .+
            coef[:log_Vc][3] .* sage .+ coef[:log_Vc][4] .* swt .+
            coef[:log_Vc][5] .* blcl,
        :log_k10 => coef[:log_k10][1] .* one .+ coef[:log_k10][2] .* male .+
            coef[:log_k10][3] .* sage .+ coef[:log_k10][4] .* swt .+
            coef[:log_k10][5] .* blcl,
        :log_k12 => coef[:log_k12][1] .* one .+ coef[:log_k12][2] .* blcl,
        :log_k21 => coef[:log_k21][1] .* one .+ coef[:log_k21][2] .* blcl,
        :log_ka => coef[:log_ka][1] .* one .+ coef[:log_ka][2] .* blcl,
        :qt_base => coef[:qt_base][1] .* one .+ coef[:qt_base][2] .* male .+
            coef[:qt_base][3] .* sage .+
            coef[:qt_base][4] .* cols[:qt_prolonging_drug_ongoing] .+
            coef[:qt_base][5] .* blcl,
        :qt_slope => coef[:qt_slope][1] .* one .+
            coef[:qt_slope][2] .* blcl,
        :log_tgi_kg => coef[:log_tgi_kg][1] .* one,
        :log_tgi_kd => coef[:log_tgi_kd][1] .* one,
        :tgi_ly0 => coef[:tgi_ly0][1] .* one,
    )
    r = Dict{Symbol,Vector{Float64}}()
    for (j, lp) in enumerate(
            (:log_Vc, :log_k10, :log_k12, :log_k21, :log_ka, :qt_base,
                :qt_slope))
        r[lp] = _jd_ref_r(idx, Lp, Vector(nt.tau_p_subject),
            Vector(nt.z_flat_p_subject), j:j)
    end
    r[:log_tgi_kg] = _jd_ref_r(idx, Ltg, Vector(nt.tau_tg_subject),
        Vector(nt.z_flat_tg_subject), 1:1)
    r[:log_tgi_kd] = _jd_ref_r(idx, Ltg, Vector(nt.tau_tg_subject),
        Vector(nt.z_flat_tg_subject), 2:2)
    tau_tb = only(Vector(nt.tau_tb_subject))
    z_tb = Vector(nt.z_flat_tb_subject)
    r[:tgi_ly0] = [tau_tb * z_tb[g] for g in idx]
    return Dict(lp => pop[lp] + r[lp] for lp in JOINT_DECL_LPS)
end

# Per-coefficient (locations, scales) in SB design order — the triple §1
# `effect()` inventory expanded (indication fans out 1-wide: 2 levels under
# the (2,:end) reference subset). The e2e prior oracle reads ONLY this table
# (never the fragments), so it independently re-pins every effect term.
function _jd_coef_priors()
    qs = _jd_qt_scale()
    tb = _jd_tgi_baseline()
    return Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}(
        :log_Vc => ([2.302585092994046, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.1, 0.1, 0.1, 1.0]),
        :log_k10 => ([-1.405170185988091, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.1, 0.1, 0.1, 1.0]),
        :log_k12 => ([-0.34657359027997265, 0.0], [2.0, 1.0]),
        :log_k21 => ([-2.649158683274018, 0.0], [2.0, 1.0]),
        :log_ka => ([-2.0794415416798357, 0.0], [0.8, 1.0]),
        :qt_base => ([0.0, 0.0, 0.0, 0.0, 0.0], fill(qs, 5)),
        :qt_slope => ([0.0, 0.0], [qs, qs]),
        :log_tgi_kg => ([0.0], [1.0]),
        :log_tgi_kd => ([0.0], [1.5]),
        :tgi_ly0 => ([tb], [1.5]),
    )
end

# Full prior oracle from constrained values: coef Normals (table above) +
# LKJ pair (fresh Stan transcription) + explicit-`sd()` tau plates
# (`Exponential(scale)` per margin, SB `sd(:, p/tg/tb)`) + std-normal z
# plates + scalar priors (triple §1 + brm_joint_tgi.jl:386,368).
function _jd_oracle_prior(nt)
    pr = 0.0
    cp = _jd_coef_priors()
    for lp in JOINT_DECL_LPS
        loc, sc = cp[lp]
        c = Vector(getproperty(nt, lp))
        for k in eachindex(c)
            pr += logpdf(Normal(loc[k], sc[k]), c[k])
        end
    end
    pr += _jd_stan_lkj(Matrix(nt.L_p_subject), 2.0)
    pr += _jd_stan_lkj(Matrix(nt.L_tg_subject), 2.0)
    # |tb| K=1 LKJ term is exactly 0.0 (both sides — asserted in the e2e).
    for (tau, s) in ((Vector(nt.tau_p_subject), JOINT_DECL_SD_SCALES.p),
            (Vector(nt.tau_tg_subject), JOINT_DECL_SD_SCALES.tg),
            (Vector(nt.tau_tb_subject), JOINT_DECL_SD_SCALES.tb))
        pr += sum(logpdf(Exponential(s), t) for t in tau)
    end
    for v in (Vector(nt.z_flat_p_subject), Vector(nt.z_flat_tg_subject),
            Vector(nt.z_flat_tb_subject))
        pr += sum(logpdf(Normal(0, 1), x) for x in v)
    end
    pr += logpdf(Exponential(0.25), nt.sigma_add)
    pr += logpdf(Exponential(0.25), nt.sigma_prop)
    pr += logpdf(LogNormal(0.0, 1.0), nt.qt_scale)
    pr += logpdf(LogNormal(-2.0402208285265546, 0.5), nt.tgi_sigma)
    # Stan upper-bound kernel: plain normal_lpdf, NO renormalizer.
    pr += logpdf(Normal(-2.3, 1.0), nt.tgi_c_cr)
    return pr
end

# Explicit-vs-default tau-prior delta: per margin, SB-explicit
# `Exponential(scale)` density MINUS the SB-default Normal(0,1) kernel at
# the same tau values (the fix keeps layout/geometry identical — only the
# prior function differs). A regression discriminator — nonzero at the
# probe, so dropping the sd_priors flips the e2e red.
function _jd_explicit_sd_delta(nt)
    d = 0.0
    for (tau, s) in ((Vector(nt.tau_p_subject), JOINT_DECL_SD_SCALES.p),
            (Vector(nt.tau_tg_subject), JOINT_DECL_SD_SCALES.tg),
            (Vector(nt.tau_tb_subject), JOINT_DECL_SD_SCALES.tb))
        for t in tau
            d += logpdf(Exponential(s), t) - logpdf(Normal(0, 1), t)
        end
    end
    return d
end

@testset "joint decl zscale bound" begin
    # In-graph Base reductions vs BRM's fitted loop: mathematically identical
    # (mean + sample SD, n−1), ~ulp apart in summation order (loops are
    # inexpressible in-graph — the documented julianic delta).
    for col in (_jd_age(), _jd_weight())
        base = (col .- sum(col) / 4) ./
            sqrt(sum((x - sum(col) / 4)^2 for x in col) / 3)
        @test maximum(abs.(base .- _jd_brm_zscale(col))) < 1e-12
    end
end

@testset "joint decl bind" begin
    bound = bind_data(_jd_plan(), _jd_columns())
    @test isbound(bound)
    # Indication maps evaluate to the non-reference level only (SB's
    # `append_row(0, β)[idx]`: AITL rides the intercept = 0 contrast).
    for m in bound.levelmaps
        @test m.values == ["BLCL"]
    end
    # Draws levels: sort-ordered subjects = SB `subject_idx` numbering.
    for d in bound.varying_draws
        @test d.levels == ["S01", "S02", "S03", "S04"]
    end
    @test bound.roles[:subject] === :group
end

@testset "joint decl layout" begin
    bound = bind_data(_jd_plan(), _jd_columns())
    layout = assign_layout(bound)
    # 26 coefs + 5 scalars + |p| (21+7+28) + |tg| (1+2+8) + |tb| (0+1+4).
    @test layout.total == 103
    @test length(coordinate_names(layout)) == 103
    byname = Dict(e.name => e for e in layout.entries)
    @test (byname[:L_p_subject].size, byname[:L_p_subject].transform) ==
        (21, :lkj)
    @test (byname[:tau_p_subject].size, byname[:tau_p_subject].transform) ==
        (7, :exp)
    @test byname[:z_flat_p_subject].size == 28
    @test (byname[:L_tg_subject].size, byname[:L_tb_subject].size) == (1, 0)
    @test byname[:z_flat_tb_subject].size == 4
    # tgi_c_cr upper truncation (fix 5c38658): Stan kernel semantics —
    # x = hi - exp(u), bare-u Jacobian, plain normal_lpdf, no renormalizer.
    @test byname[:tgi_c_cr].transform === :upper
    @test byname[:tgi_c_cr].hi == -0.6931471805599453
    @test byname[:sigma_add].transform === :exp
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    @test size(nt.L_p_subject) == (7, 7)
    @test size(nt.L_tg_subject) == (2, 2)
    @test nt.L_tb_subject == [1.0;;]
    @test size(nt.b_p_subject) == (4, 7) # derived draws, SB shape
    @test nt.tgi_c_cr < -0.6931471805599453 # upper support enforced
    @test unconstrain(layout, nt) ≈ u
end

@testset "joint decl LKJ cross-check" begin
    bound = bind_data(_jd_plan(), _jd_columns())
    layout = assign_layout(bound)
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    # Fresh Stan transcription == thin-layer node on the joint's exact
    # inputs (op order differs by construction — tight ≈, not bit).
    @test _jd_stan_lkj(Matrix(nt.L_p_subject), 2.0) ≈
        lkj_corr_cholesky_logpdf(Matrix(nt.L_p_subject), 2.0)
    @test _jd_stan_lkj(Matrix(nt.L_tg_subject), 2.0) ≈
        lkj_corr_cholesky_logpdf(Matrix(nt.L_tg_subject), 2.0)
    @test lkj_corr_cholesky_logpdf(Matrix(nt.L_tb_subject), 1.0) == 0.0
end

# Gradient check on an ALREADY-PREPARED :posterior kernel — the exact
# `_check_gradient` body (same backend, findiff, tolerances) minus its
# internal `prepare`, so the e2e pays one joint-scale prepare per want
# instead of twice for `:posterior`. On canonical `05116b82` that prepare is
# no longer the historical 9-minute outlier: a fresh-process repro measured
# plan 0.52 s and cold prepare 26.3 s, with warm follow-up preparations in
# the 5–15 s range. Keep this helper as the cheaper gradient-proof path and
# cite a fresh measurement before changing the verification strategy again.
function _jd_check_gradient(kern, u)
    prep = prepare_ad(kern, _GEN_BACKEND, u; active = :unconstrained)
    g = ReactiveKernels.ad_value_and_gradient!(prep, similar(u), u)[2]
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(kern, u); rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "joint decl e2e values and gradient" begin
    cols = _jd_columns()
    bound = bind_data(_jd_plan(), cols)
    built = build_kernel(bound)
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    lps = _jd_oracle_lps(nt, cols)
    ll = sum(
        sum(logpdf(Normal(lps[d.lp][i], d.scale), cols[d.resp][i])
            for i in 1:4)
        for d in _JD_DUMMY)
    pr = _jd_oracle_prior(nt)
    jac = logjac(built.layout, u)
    # Three prepares, one per want (ll/pr/posterior) — the `:log_jacobian`
    # query path is covered in `test_query.jl` and adds another joint-scale
    # prepare here for no new proof (snag `prepare-9min-on-034fa1d6`).
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    ks = sort!(collect(keys(bound.columns)))
    bnt = NamedTuple{Tuple(ks)}(Tuple(bound.columns[k] for k in ks))
    kern = prepare(built.spec; have = (:unconstrained, ks...),
        want = :posterior, bound = bnt)
    @test kern(u) ≈ ll + pr + jac
    _jd_check_gradient(kern, u)
end

@testset "joint decl sd explicit" begin
    # Fix 6228160: the emitted prior carries the joint's EXPLICIT sd()
    # (proven term-for-term in the e2e above). This testset pins the
    # carrier: bind preserves the per-margin sd_priors, the generator emits
    # one uniform Exponential tau plate per block, and the explicit-vs-
    # default delta is REAL (nonzero at the probe — dropping the sd_priors
    # flips the e2e red).
    bound = bind_data(_jd_plan(), _jd_columns())
    @test [length(d.sd_priors) for d in bound.varying_draws] == [7, 2, 1]
    @test all(d -> all(p -> p.family === :exponential,
        d.sd_priors), bound.varying_draws)
    built = build_kernel(bound)
    src = string(kernel_expr(bound, built.layout))
    for stem in ("tau_p_subject", "tau_tg_subject", "tau_tb_subject")
        @test occursin("_ppl_prior_$stem", src)
    end
    @test occursin("exponential", src)
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    d = _jd_explicit_sd_delta(nt)
    @test isfinite(d) && abs(d) > 1e-6
end

# Corpus 49 (surface declaration program) as a plan: parse the canonical file
# (same `# data:` + `begin` shape `test_corpus.jl` consumes).
function _jd_corpus_plan()
    path = joinpath(@__DIR__, "corpus", "51_joint_declarations.jl")
    lines = split(read(path, String), '\n')
    datanames =
        Tuple(Symbol(s) for s in split(strip(lines[1][8:end])))
    return lower_rkppl(Meta.parse(join(lines[2:end], '\n')), datanames)
end

@testset "joint decl surface twin" begin
    # Corpus 49 lowers to the same plan as the IR builders, modulo the
    # draws-naming stem (surface derives `subject`/`subject_d_tg`/
    # `subject_d_tb` from binding order; IR uses SB-mirroring
    # `p_subject`/`tg_subject`/`tb_subject` — names never move numbers).
    su = _jd_corpus_plan()
    ir = _jd_plan()
    @test length(su.predictors) == length(ir.predictors) == 10
    @test length(su.responses) == length(ir.responses) == 10
    # Draws correspondence by (kind, K, eta) — deterministic, order-free.
    sig(d) = (d.kind, length(d.margins), d.lkj_eta)
    sumap = Dict(sig(d) => d.label for d in su.varying_draws)
    irmap = Dict(sig(d) => d.label for d in ir.varying_draws)
    @test keys(sumap) == keys(irmap)
    ren = Dict(sumap[k] => irmap[k] for k in keys(sumap))
    @test [ren[d.label] for d in su.varying_draws] ==
        [d.label for d in ir.varying_draws]
    for (a, b) in zip(su.varying_draws, ir.varying_draws)
        @test (a.group, a.kind, a.lkj_eta) ==
            (b.group, b.kind, b.lkj_eta)
        @test [(m.coefficient, m.z.kind, m.z.column) for m in a.margins] ==
            [(m.coefficient, m.z.kind, m.z.column) for m in b.margins]
        # The surface cannot spell sd priors (IR-only carrier): the twin
        # differs by exactly the tau-prior delta (accounted below).
        @test isempty(a.sd_priors)
        @test length(b.sd_priors) == length(b.margins)
        @test all(p -> p.family === :exponential, b.sd_priors)
    end
    # Slices match under the renaming.
    @test [(ren[s.draws], s.columns, s.target) for s in su.varying_slices] ==
        [(s.draws, s.columns, s.target) for s in ir.varying_slices]
    # Predictors match (terms: kinds + columns + renamed draws refs; the
    # varying-term addressee carries the draws-naming stem, so each side must
    # match ITS OWN `r_<target>_<suffix>` convention, not each other).
    _stem(label) = String(label)[7:end] # strip "draws_"
    for (a, b) in zip(su.predictors, ir.predictors)
        @test (a.name, a.link) == (b.name, b.link)
        @test length(a.terms) == length(b.terms)
        for (ta, tb) in zip(a.terms, b.terms)
            @test (ta.kind, ta.columns) == (tb.kind, tb.columns)
            if ta.kind === VaryingEffectTerm
                @test ren[ta.options.draws] == tb.options.draws
                @test String(ta.addressee) ==
                    "r_$(b.name)_$(_stem(ta.options.draws))"
                @test String(tb.addressee) ==
                    "r_$(b.name)_$(_stem(tb.options.draws))"
            else
                @test (ta.addressee, ta.options) ==
                    (tb.addressee, tb.options)
            end
        end
    end
    # Priors, derived, levelmaps, scalars: field-wise equality (these structs
    # carry no `==` — it falls back to `===`, which distinct-but-identical
    # vectors/exprs fail; the `_maps_equal` precedent).
    @test [(p.predictor, p.addressee, p.location, p.scale)
        for p in su.population_priors] ==
        [(p.predictor, p.addressee, p.location, p.scale)
            for p in ir.population_priors]
    _stripln(x) = x
    _stripln(e::Expr) = Expr(e.head,
        filter(a -> !isa(a, LineNumberNode), _stripln.(e.args))...)
    @test [(d.name, string(_stripln(d.expr)), d.label)
        for d in su.derived] ==
        [(d.name, string(_stripln(d.expr)), d.label)
            for d in ir.derived]
    @test [(m.predictor, m.column, m.values, m.source, m.subset)
        for m in su.levelmaps] ==
        [(m.predictor, m.column, m.values, m.source, m.subset)
            for m in ir.levelmaps]
    @test [(p.name, p.family, p.args, p.support_override, p.label)
        for p in su.parameters] ==
        [(p.name, p.family, p.args, p.support_override, p.label)
            for p in ir.parameters]
    for (a, b) in zip(su.responses, ir.responses)
        @test (a.family, a.link, a.response, a.predictor, a.scale,
            a.evidence) ==
            (b.family, b.link, b.response, b.predictor, b.scale,
                b.evidence)
    end
    # Bind-level agreement (levelmaps evaluate identically on both sides).
    # There is deliberately NO posterior-numbers comparison here: additional
    # joint-scale preparations remain materially expensive (snag
    # `prepare-9min-on-034fa1d6`; no longer 9 minutes, but cold preparations
    # still cost tens of seconds on the 2026-09-25 canonical measurement).
    # They would re-prove what the structural comparison above (same plan
    # modulo the naming stem) plus deterministic lowering plus the e2e
    # already imply. (Proven once at 9320e1a — twin 150/150 with delta
    # accounting — then cut as pure waste.)
    cols = _jd_columns()
    bsu, bir = bind_data(su, cols), bind_data(ir, cols)
    @test [m.values for m in bsu.levelmaps] ==
        [m.values for m in bir.levelmaps] == fill(["BLCL"], 7)
    @test assign_layout(bsu).total == assign_layout(bir).total == 103
end

@testset "joint decl emission shape" begin
    bound = bind_data(_jd_plan(), _jd_columns())
    built = build_kernel(bound)
    src = string(kernel_expr(bound, built.layout))
    # All 10 LP nodes on the HAVE seam.
    for lp in JOINT_DECL_LPS
        @test occursin("_ppl_lp_$lp", src)
    end
    # In-graph standardize, LKJ pair, grouped index encoder.
    @test occursin("mean(age_yr)", src)
    @test occursin("std(weight_kg)", src)
    @test occursin("_ppl_prior_L_p_subject", src)
    @test occursin("_ppl_prior_L_tg_subject", src)
    @test occursin("_ppl_gidx_subject", src)
end

@testset "joint decl memo config" begin
    # Full memo spelling admits through fragments (SB memo joint:
    # memo TGI formulas + net_kill + recist/sld pair).
    f = joint_decl_fragments(; tgi_formulas = JOINT_DECL_TGI_FORMULAS_MEMO,
        tgi_parametrization = :net_kill, tgi_thresholds = :recist,
        tgi_measure = :sld, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    # Roster swaps the growth LP to :tgi_net (SB memo bucket order).
    @test [p.name for p in f.predictors] ==
        [:log_Vc, :log_k10, :log_k12, :log_k21, :log_ka, :qt_base,
            :qt_slope, :tgi_net, :log_tgi_kd, :tgi_ly0]
    byname = Dict(p.name => p for p in f.predictors)
    # Memo TGI fixed effects: Intercept + indication on growth/kd.
    @test [(t.kind, t.columns) for t in byname[:tgi_net].terms] ==
        [(InterceptTerm, Symbol[]), (FactorTerm, [:indication]),
            (VaryingEffectTerm, [:subject])]
    @test [(t.kind, t.columns) for t in byname[:log_tgi_kd].terms] ==
        [(InterceptTerm, Symbol[]), (FactorTerm, [:indication]),
            (VaryingEffectTerm, [:subject])]
    # Memo ly0: full margins + covariates (memo formulas).
    @test [(t.kind, t.columns) for t in byname[:tgi_ly0].terms] ==
        [(InterceptTerm, Symbol[]), (ContinuousTerm, [:male]),
            (ContinuousTerm, [:standardize_age_yr]),
            (ContinuousTerm, [:standardize_weight_kg]),
            (FactorTerm, [:indication]),
            (VaryingEffectTerm, [:subject])]
    # Shared 9-block: growth/kd varying terms ride draws_p_subject.
    @test byname[:tgi_net].terms[end].options.draws === :draws_p_subject
    @test byname[:log_tgi_kd].terms[end].options.draws === :draws_p_subject
    # Draws: |p| K=9 eta 1.0, |tb| K=1 eta 1.0 vacuous; no |tg|.
    @test [(d.label, d.kind, d.lkj_eta, length(d.margins))
        for d in f.varying_draws] ==
        [(:draws_p_subject, :correlated, 1.0, 9),
            (:draws_tb_subject, :correlated, 1.0, 1)]
    # Mixed sd() scales: 7x p-scale + 2x tg-scale in the shared block.
    @test [(p.family, p.param) for p in f.varying_draws[1].sd_priors] ==
        vcat(fill((:exponential, 0.3333333333333333), 7),
            fill((:exponential, 0.5), 2))
    @test [(p.family, p.param) for p in f.varying_draws[2].sd_priors] ==
        [(:exponential, 1.0)]
    # Slices: TGI margins at columns 8/9 of the shared block.
    @test [(s.draws, s.columns, s.target) for s in f.varying_slices] ==
        vcat(
            [(:draws_p_subject, j:j, lp)
                for (j, lp) in enumerate((:log_Vc, :log_k10, :log_k12,
                    :log_k21, :log_ka, :qt_base, :qt_slope))],
            [(:draws_p_subject, 8:8, :tgi_net),
                (:draws_p_subject, 9:9, :log_tgi_kd),
                (:draws_tb_subject, 1:1, :tgi_ly0)],
        )
    # Levelmaps gain the memo TGI LPs (growth + kd + ly0).
    @test [(m.predictor, m.column, m.subset) for m in f.levelmaps] ==
        vcat([(lp, :indication, (2, :end))
                for lp in (:log_Vc, :log_k10, :log_k12, :log_k21, :log_ka,
                    :qt_base, :qt_slope)],
            [(:tgi_net, :indication, (2, :end)),
                (:log_tgi_kd, :indication, (2, :end)),
                (:tgi_ly0, :indication, (2, :end))])
    # effect() inventory: 26 + 2 growth/kd indication + 4 ly0 covariates.
    pr = Dict((p.predictor, p.addressee) => (p.location, p.scale)
        for p in f.population_priors)
    @test length(f.population_priors) == 32
    @test pr[(:tgi_net, :Intercept)] == (0.0, 1.5)
    @test pr[(:tgi_net, :indication)] == (0.0, 1.0)
    @test pr[(:tgi_ly0, :male)] == (0.0, 0.1)
    # recist+sld folds the PR-boundary upper to log(0.7).
    byparam = Dict(p.name => p for p in f.parameters)
    @test byparam[:tgi_c_cr].support_override == (:upper, log(0.7))
    # Pair gate: recist+spd fails closed (lugano_ct+sld covered above).
    @test_throws ContractValidationError joint_decl_fragments(;
        tgi_thresholds = :recist, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
end

# --- memo e2e twin (same 4-subject frame, memo fragments) ---

const _JD_MEMO_DUMMY = (
    (lp = :log_Vc, resp = :ydVc, scale = 1.5),
    (lp = :log_k10, resp = :ydK10, scale = 1.6),
    (lp = :log_k12, resp = :ydK12, scale = 1.7),
    (lp = :log_k21, resp = :ydK21, scale = 1.8),
    (lp = :log_ka, resp = :ydKa, scale = 1.9),
    (lp = :qt_base, resp = :ydQb, scale = 2.0),
    (lp = :qt_slope, resp = :ydQs, scale = 2.1),
    (lp = :tgi_net, resp = :ydTg, scale = 2.2),
    (lp = :log_tgi_kd, resp = :ydKd, scale = 2.3),
    (lp = :tgi_ly0, resp = :ydLy0, scale = 2.4),
)

const _JD_MEMO_LPS = [d.lp for d in _JD_MEMO_DUMMY]

function _jd_memo_responses()
    return LikelihoodSpec[
        LikelihoodSpec(GaussianFam, IdentityLink, d.resp, d.lp, d.scale,
            nothing, ResponseEvidence(:none, nothing, nothing),
            Symbol(d.resp, :_resp))
        for d in _JD_MEMO_DUMMY
    ]
end

function _jd_memo_plan()
    f = joint_decl_fragments(; tgi_formulas = JOINT_DECL_TGI_FORMULAS_MEMO,
        tgi_parametrization = :net_kill, tgi_thresholds = :recist,
        tgi_measure = :sld, qt_prior_scale = _jd_qt_scale(),
        tgi_baseline_log_size = _jd_tgi_baseline())
    return StructuralPlan(_jd_memo_responses(), f.predictors,
        f.population_priors, f.parameters, AssignmentSpec[],
        Dict{Symbol,AbstractVector}(), 4; derived = f.derived,
        levelmaps = f.levelmaps, varying_draws = f.varying_draws,
        varying_slices = f.varying_slices)
end

function _jd_memo_oracle_lps(nt, cols)
    male = cols[:male]
    sage = (cols[:age_yr] .- sum(cols[:age_yr]) / 4) ./
        sqrt(sum((x - sum(cols[:age_yr]) / 4)^2 for x in cols[:age_yr]) / 3)
    swt = (cols[:weight_kg] .- sum(cols[:weight_kg]) / 4) ./
        sqrt(sum((x - sum(cols[:weight_kg]) / 4)^2
            for x in cols[:weight_kg]) / 3)
    blcl = Float64.(cols[:indication] .== "BLCL")
    one = ones(4)
    idx = [findfirst(==(v), ["S01", "S02", "S03", "S04"])
        for v in cols[:subject]]
    Lp = Matrix(nt.L_p_subject)
    Ltb = Matrix(nt.L_tb_subject)
    @assert Ltb == [1.0;;] # vacuous 1x1
    coef = Dict(
        :log_Vc => Vector(nt.log_Vc), :log_k10 => Vector(nt.log_k10),
        :log_k12 => Vector(nt.log_k12), :log_k21 => Vector(nt.log_k21),
        :log_ka => Vector(nt.log_ka), :qt_base => Vector(nt.qt_base),
        :qt_slope => Vector(nt.qt_slope),
        :tgi_net => Vector(nt.tgi_net),
        :log_tgi_kd => Vector(nt.log_tgi_kd),
        :tgi_ly0 => Vector(nt.tgi_ly0),
    )
    pop = Dict{Symbol,Vector{Float64}}(
        :log_Vc => coef[:log_Vc][1] .* one .+ coef[:log_Vc][2] .* male .+
            coef[:log_Vc][3] .* sage .+ coef[:log_Vc][4] .* swt .+
            coef[:log_Vc][5] .* blcl,
        :log_k10 => coef[:log_k10][1] .* one .+ coef[:log_k10][2] .* male .+
            coef[:log_k10][3] .* sage .+ coef[:log_k10][4] .* swt .+
            coef[:log_k10][5] .* blcl,
        :log_k12 => coef[:log_k12][1] .* one .+ coef[:log_k12][2] .* blcl,
        :log_k21 => coef[:log_k21][1] .* one .+ coef[:log_k21][2] .* blcl,
        :log_ka => coef[:log_ka][1] .* one .+ coef[:log_ka][2] .* blcl,
        :qt_base => coef[:qt_base][1] .* one .+ coef[:qt_base][2] .* male .+
            coef[:qt_base][3] .* sage .+
            coef[:qt_base][4] .* cols[:qt_prolonging_drug_ongoing] .+
            coef[:qt_base][5] .* blcl,
        :qt_slope => coef[:qt_slope][1] .* one .+
            coef[:qt_slope][2] .* blcl,
        :tgi_net => coef[:tgi_net][1] .* one .+
            coef[:tgi_net][2] .* blcl,
        :log_tgi_kd => coef[:log_tgi_kd][1] .* one .+
            coef[:log_tgi_kd][2] .* blcl,
        :tgi_ly0 => coef[:tgi_ly0][1] .* one .+
            coef[:tgi_ly0][2] .* male .+ coef[:tgi_ly0][3] .* sage .+
            coef[:tgi_ly0][4] .* swt .+ coef[:tgi_ly0][5] .* blcl,
    )
    r = Dict{Symbol,Vector{Float64}}()
    for (j, lp) in enumerate(
            (:log_Vc, :log_k10, :log_k12, :log_k21, :log_ka, :qt_base,
                :qt_slope, :tgi_net, :log_tgi_kd))
        r[lp] = _jd_ref_r(idx, Lp, Vector(nt.tau_p_subject),
            Vector(nt.z_flat_p_subject), j:j)
    end
    tau_tb = only(Vector(nt.tau_tb_subject))
    z_tb = Vector(nt.z_flat_tb_subject)
    r[:tgi_ly0] = [tau_tb * z_tb[g] for g in idx]
    return Dict(lp => pop[lp] + r[lp] for lp in _JD_MEMO_LPS)
end

function _jd_memo_coef_priors()
    qs = _jd_qt_scale()
    tb = _jd_tgi_baseline()
    return Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}(
        :log_Vc => ([2.302585092994046, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.1, 0.1, 0.1, 1.0]),
        :log_k10 => ([-1.405170185988091, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.1, 0.1, 0.1, 1.0]),
        :log_k12 => ([-0.34657359027997265, 0.0], [2.0, 1.0]),
        :log_k21 => ([-2.649158683274018, 0.0], [2.0, 1.0]),
        :log_ka => ([-2.0794415416798357, 0.0], [0.8, 1.0]),
        :qt_base => ([0.0, 0.0, 0.0, 0.0, 0.0], fill(qs, 5)),
        :qt_slope => ([0.0, 0.0], [qs, qs]),
        :tgi_net => ([0.0, 0.0], [1.5, 1.0]),
        :log_tgi_kd => ([0.0, 0.0], [1.5, 1.0]),
        :tgi_ly0 => ([tb, 0.0, 0.0, 0.0, 0.0], [1.5, 0.1, 0.1, 0.1, 1.0]),
    )
end

function _jd_memo_oracle_prior(nt)
    pr = 0.0
    cp = _jd_memo_coef_priors()
    for lp in _JD_MEMO_LPS
        loc, sc = cp[lp]
        c = Vector(getproperty(nt, lp))
        for k in eachindex(c)
            pr += logpdf(Normal(loc[k], sc[k]), c[k])
        end
    end
    pr += _jd_stan_lkj(Matrix(nt.L_p_subject), 1.0)
    # |tb| K=1 LKJ term is exactly 0.0 (both sides — asserted in the e2e).
    tau_p = Vector(nt.tau_p_subject)
    pr += sum(logpdf(Exponential(JOINT_DECL_SD_SCALES.p), t)
        for t in tau_p[1:7])
    pr += sum(logpdf(Exponential(JOINT_DECL_SD_SCALES.tg), t)
        for t in tau_p[8:9])
    pr += sum(logpdf(Exponential(JOINT_DECL_SD_SCALES.tb), t)
        for t in Vector(nt.tau_tb_subject))
    for v in (Vector(nt.z_flat_p_subject), Vector(nt.z_flat_tb_subject))
        pr += sum(logpdf(Normal(0, 1), x) for x in v)
    end
    pr += logpdf(Exponential(0.25), nt.sigma_add)
    pr += logpdf(Exponential(0.25), nt.sigma_prop)
    pr += logpdf(LogNormal(0.0, 1.0), nt.qt_scale)
    pr += logpdf(LogNormal(-2.0402208285265546, 0.5), nt.tgi_sigma)
    # Stan upper-bound kernel: plain normal_lpdf, NO renormalizer.
    pr += logpdf(Normal(-2.3, 1.0), nt.tgi_c_cr)
    return pr
end

@testset "joint decl memo bind+layout" begin
    bound = bind_data(_jd_memo_plan(), _jd_columns())
    @test isbound(bound)
    for m in bound.levelmaps
        @test m.values == ["BLCL"]
    end
    for d in bound.varying_draws
        @test d.levels == ["S01", "S02", "S03", "S04"]
    end
    # Mixed sd() carrier: 9 margins in |p| (7x p-scale + 2x tg-scale).
    @test [length(d.sd_priors) for d in bound.varying_draws] == [9, 1]
    @test [p.param for p in bound.varying_draws[1].sd_priors] ==
        vcat(fill(0.3333333333333333, 7), fill(0.5, 2))
    layout = assign_layout(bound)
    # 32 coefs + 5 scalars + |p| (36+9+36) + |tb| (0+1+4).
    @test layout.total == 123
    @test length(coordinate_names(layout)) == 123
    byname = Dict(e.name => e for e in layout.entries)
    @test (byname[:L_p_subject].size, byname[:L_p_subject].transform) ==
        (36, :lkj)
    @test (byname[:tau_p_subject].size, byname[:tau_p_subject].transform) ==
        (9, :exp)
    @test byname[:z_flat_p_subject].size == 36
    @test !haskey(byname, :L_tg_subject)
    # recist+sld upper truncation: hi = log(0.7).
    @test byname[:tgi_c_cr].transform === :upper
    @test byname[:tgi_c_cr].hi == -0.3566749439387324
    u = collect(range(-0.4, 0.4; length = layout.total))
    nt = constrain(layout, u)
    @test size(nt.L_p_subject) == (9, 9)
    @test size(nt.b_p_subject) == (4, 9)
    @test nt.tgi_c_cr < -0.3566749439387324
    @test unconstrain(layout, nt) ≈ u
end

@testset "joint decl memo e2e values and gradient" begin
    cols = _jd_columns()
    bound = bind_data(_jd_memo_plan(), cols)
    built = build_kernel(bound)
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    lps = _jd_memo_oracle_lps(nt, cols)
    ll = sum(
        sum(logpdf(Normal(lps[d.lp][i], d.scale), cols[d.resp][i])
            for i in 1:4)
        for d in _JD_MEMO_DUMMY)
    pr = _jd_memo_oracle_prior(nt)
    jac = logjac(built.layout, u)
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    ks = sort!(collect(keys(bound.columns)))
    bnt = NamedTuple{Tuple(ks)}(Tuple(bound.columns[k] for k in ks))
    kern = prepare(built.spec; have = (:unconstrained, ks...),
        want = :posterior, bound = bnt)
    @test kern(u) ≈ ll + pr + jac
    _jd_check_gradient(kern, u)
end
