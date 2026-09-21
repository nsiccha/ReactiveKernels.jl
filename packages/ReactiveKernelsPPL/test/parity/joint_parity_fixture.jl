# Joint SB-parity fixture (durable W4 artifact, 2026-09-21).
# Self-contained: fixture columns (SB V2 fixture values, prep recipes
# asserted against stan_data at authoring time), the final joint surface
# per config (declaration + varying + schedule + proven joint cell), the
# SB-theta mapper, and the pinned SB numbers. Vendored theta:
# parity/<config>_p1_constrained.txt (SB point1 constrained values).
# Provenance: brm-tgi W3d/W4 harness (final_ast/final_columns/scaffold +
# oracle-reader transcriptions); port verified byte-identical against the
# harness builders before commit (see test_joint_parity.jl).

const _PARITY_DIR = @__DIR__

"""Vendored SB point1 constrained theta per config."""
function _parity_theta(config::AbstractString)
    @assert config in ("continuous", "ordinal", "binary")
    path = joinpath(_PARITY_DIR, config * "_p1_constrained.txt")
    out = Dict{String,Float64}()
    for line in readlines(path)
        startswith(line, "seed=") && continue
        k, v = split(line, "="; limit = 2)
        out[k] = parse(Float64, v)
    end
    return out
end

# --- fixture raw constants (replicated from
# joint-oracle/provenance/joint_pk_qt_fixture.jl; cross-checked against the
# decoded stan_data in the @asserts below, never trusted) ---
const _FX_SUBJECT_NAME = ["S1", "S2", "S3"]
const _FX_MALE = [1.0, 0.0, 1.0]
const _FX_AGE_YR = [50.0, 60.0, 45.0]
const _FX_WEIGHT_KG = [75.0, 65.0, 80.0]
const _FX_INDICATION = ["AITL", "BLCL", "AITL"]
const _FX_COMED = [1.0, 0.0, 1.0]
const _FX_OBS_SUBJ = [1, 1, 1, 1, 2, 2, 2]
const _FX_OBS_TIME = [73.0, 150.0, 150.0, 180.0, 0.0, 24.0, 60.0]
const _FX_OBS_VALUE = [0.8, 0.45, 0.47, 0.3, 0.0, 0.6, 0.25]
const _FX_OBS_LLOQ = [0.1, 0.1, NaN, 0.1, 0.05, 0.05, NaN]
const _FX_DOSE_SUBJ = [1, 1, 1, 1, 1, 1, 2, 2, 2]
const _FX_DOSE_TIME = [0.0, 24.0, 48.0, 96.0, 108.0, 120.0, 0.0, 24.0, 48.0]
const _FX_DOSE_AMT = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 80.0, 80.0, 80.0]
const _FX_QT_SUBJ = [2, 1, 3, 1, 2, 3, 1]
const _FX_QT_TIME = [24.0, -1.0, 1.0, 25.0, 0.0, 25.0, 72.0]
const _FX_QT_Y = [0.15, -0.20, 0.05, 0.25, -0.10, 0.12, 0.08]
const _FX_QT_INV_SQRT_K =
    [1.0, 1.0, inv(sqrt(2.0)), 1.0, 1.0, inv(sqrt(3.0)), 1.0]
const _FX_TGI_SUBJ = [1, 2, 1, 2, 1, 2]
const _FX_TGI_TIME = [0.0, -24.0, 96.0, 48.0, 170.0, 100.0]
const _FX_TGI_BASELINE = [0.0, -48.0, 0.0, -48.0, 0.0, -48.0]
const _FX_TGI_VALUE_CONT = [2100.0, 900.0, 1500.0, 950.0, 800.0, 1400.0]
const _FX_TGI_VALUE_ORD = [3, 3, 3, 3, 2, 4]
const _FX_TGI_VALUE_BIN = [0, 0, 0, 0, 1, 0]

# --- fixture bind columns (SB V2 fixture; sorted axes per the D8 rule;
# BLQ clamp, filled lloq, tgi clock /1344) ---
function _parity_sched_columns()
    pk_conc = max.(_FX_OBS_VALUE, [0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05])
    qperm = sortperm(collect(zip(_FX_QT_SUBJ, _FX_QT_TIME)))
    tperm = sortperm(collect(zip(_FX_TGI_SUBJ, _FX_TGI_TIME)))
    return Dict{Symbol,AbstractVector}(
        :subj => _FX_OBS_SUBJ, :time => _FX_OBS_TIME,
        :dsubj => _FX_DOSE_SUBJ, :dtime => _FX_DOSE_TIME,
        :damt => _FX_DOSE_AMT,
        :pk_conc => pk_conc,
        :pk_lloq => [0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05],
        :esubj => _FX_QT_SUBJ[qperm], :etime => _FX_QT_TIME[qperm],
        :ecg_y => _FX_QT_Y[qperm], :qt_w => _FX_QT_INV_SQRT_K[qperm],
        :tsubj => _FX_TGI_SUBJ[tperm], :ttime_h => _FX_TGI_TIME[tperm],
        :tgi_t => _FX_TGI_TIME[tperm] ./ 1344.0,
        :tgi_y => log.(_FX_TGI_VALUE_CONT[tperm]),
        :tgi_cc => _FX_TGI_VALUE_ORD[tperm],
        :tgi_bb => _FX_TGI_VALUE_BIN[tperm],
        :T0_data => [0.0, -48.0 / 1344.0, 0.0],
    )
end

"""Full bind columns per config (drops the config-absent tgi response)."""
function _parity_columns(config::AbstractString = "continuous")
    @assert config in ("continuous", "ordinal", "binary")
    cols = _parity_sched_columns()
    drop = config == "continuous" ? (:tgi_cc, :tgi_bb) :
        config == "ordinal" ? (:tgi_y, :tgi_bb) : (:tgi_y, :tgi_cc)
    foreach(k -> delete!(cols, k), drop)
    return merge(Dict{Symbol,AbstractVector}(
            :subject => ["S1", "S2", "S3"],
            :male => [1.0, 0.0, 1.0],
            :age_yr => [50.0, 60.0, 45.0],
            :weight_kg => [75.0, 65.0, 80.0],
            :indication => ["AITL", "BLCL", "AITL"],
            :qt_prolonging_drug_ongoing => [1.0, 0.0, 1.0],
        ), cols)
end

const _FN_DECL = """
    standardize_age_yr = (age_yr .- mean(age_yr)) ./ std(age_yr)
    standardize_weight_kg = (weight_kg .- mean(weight_kg)) ./ std(weight_kg)
    a_Vc ~ Normal(2.302585092994046, 0.8)
    b_Vc_male ~ Normal(0.0, 0.1)
    b_Vc_age ~ Normal(0.0, 0.1)
    b_Vc_wt ~ Normal(0.0, 0.1)
    c_Vc[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_k10 ~ Normal(-1.405170185988091, 0.8)
    b_k10_male ~ Normal(0.0, 0.1)
    b_k10_age ~ Normal(0.0, 0.1)
    b_k10_wt ~ Normal(0.0, 0.1)
    c_k10[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_k12 ~ Normal(-0.34657359027997265, 2.0)
    c_k12[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_k21 ~ Normal(-2.649158683274018, 2.0)
    c_k21[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_ka ~ Normal(-2.0794415416798357, 0.8)
    c_ka[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_qb ~ Normal(0.0, 0.8333333333333334)
    b_qb_male ~ Normal(0.0, 0.8333333333333334)
    b_qb_age ~ Normal(0.0, 0.8333333333333334)
    b_qb_ongoing ~ Normal(0.0, 0.8333333333333334)
    c_qb[levels(indication)[2:end]] .~ Normal.(0.0, 0.8333333333333334)
    a_qs ~ Normal(0.0, 0.8333333333333334)
    c_qs[levels(indication)[2:end]] .~ Normal.(0.0, 0.8333333333333334)
    a_tg ~ Normal(0.0, 1.0)
    a_tkd ~ Normal(0.0, 1.5)"""

const _FN_DECL_TB = """
    a_tb ~ Normal(7.226043693517912, 1.5)"""

const _FN_SCALES = """
    sigma_add ~ Exponential(0.25)
    sigma_prop ~ Exponential(0.25)
    qt_scale ~ LogNormal(0.0, 1.0)
    tgi_sigma ~ LogNormal(-2.0402208285265546, 0.5)"""

const _FN_CCR =
    "tgi_c_cr ~ truncated(Normal(-2.3, 1.0), -Inf, -0.6931471805599453)"

const _FN_VARYING = """
    d_p ~ varying_draws(subject, [1, 1, 1, 1, 1, 1, 1]; eta = 2.0)
    r_Vc ~ varying_slice(d_p, 1)
    r_k10 ~ varying_slice(d_p, 2)
    r_k12 ~ varying_slice(d_p, 3)
    r_k21 ~ varying_slice(d_p, 4)
    r_ka ~ varying_slice(d_p, 5)
    r_qb ~ varying_slice(d_p, 6)
    r_qs ~ varying_slice(d_p, 7)
    d_tg ~ varying_draws(subject, [1, 1]; eta = 2.0)
    r_tg ~ varying_slice(d_tg, 1)
    r_tkd ~ varying_slice(d_tg, 2)"""

const _FN_VARYING_TB = """
    d_tb ~ varying_draws(subject, [1]; eta = 1.0)
    r_tb ~ varying_slice(d_tb, 1)"""

const _FN_LPS = """
    log_Vc = a_Vc .+ b_Vc_male .* male .+ b_Vc_age .* standardize_age_yr .+ b_Vc_wt .* standardize_weight_kg .+ c_Vc[indication] .+ r_Vc
    log_k10 = a_k10 .+ b_k10_male .* male .+ b_k10_age .* standardize_age_yr .+ b_k10_wt .* standardize_weight_kg .+ c_k10[indication] .+ r_k10
    log_k12 = a_k12 .+ c_k12[indication] .+ r_k12
    log_k21 = a_k21 .+ c_k21[indication] .+ r_k21
    log_ka = a_ka .+ c_ka[indication] .+ r_ka
    qt_base = a_qb .+ b_qb_male .* male .+ b_qb_age .* standardize_age_yr .+ b_qb_ongoing .* qt_prolonging_drug_ongoing .+ c_qb[indication] .+ r_qb
    qt_slope = a_qs .+ c_qs[indication] .+ r_qs
    log_tgi_kg = a_tg .+ r_tg
    log_tgi_kd = a_tkd .+ r_tkd"""

const _FN_LPS_TB = """
    tgi_ly0 = a_tb .+ r_tb"""

const _FN_SCHED = """
    tgi_baseline_time = T0_data
    pk_sched = linear_pk_schedule(obs = (:subj, :time),
        dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime),
        tgi = (:tsubj, :ttime_h))
    log_F = linear_pk_log_f(pk_sched; k = 5)"""

const _FN_CELL_HEAD = """
        pk_reads = linear_pk_read_locs_auc(pk_sched, log_F, subject_Vc,
            subject_k10, subject_k12, subject_k21, subject_ka)
        conc = pk_reads[pk_sched.conc_map]
        mu = conc[pk_sched.obs_map]
        conc_ecg = conc[pk_sched.ecg_map]
        qbase_rows = qbase[esubj]
        qslope_rows = qslope[esubj]
        qt_loc = qbase_rows .+ qslope_rows .* (conc_ecg ./ 0.8)
        qt_sd = qt_scale .* qt_weight
        tgi_exposure = pk_reads[pk_sched.tgi_auc_map] ./ 140.62960372536338
        tgi_kg_rows = exp.(tgi_g[tsubj])
        tgi_kd_rows = exp.(tgi_k[tsubj])
        tgi_r = tgi_kg_rows .* tgi_t .- tgi_kd_rows .* tgi_exposure
        tgi_t0_rows = tgi_t0[tsubj]
        tgi_change = tgi_r .- tgi_kg_rows .* tgi_t0_rows
        tgi_ref = tgi_segmented_nadir(tgi_change, pk_sched_tgi_seg_ends)"""

const _FN_CELL_MU = """
        tgi_b0_rows = tgi_b0[tsubj]
        tgi_mu = tgi_b0_rows .+ tgi_r"""

function final_ast(config::AbstractString = "continuous")
    @assert config in ("continuous", "ordinal", "binary")
    cont = config == "continuous"
    decl = _FN_DECL * (cont ? "\n" * _FN_DECL_TB : "") * "\n" * _FN_SCALES *
        "\n" * _FN_CCR
    varying = _FN_VARYING * (cont ? "\n" * _FN_VARYING_TB : "")
    lps = _FN_LPS * (cont ? "\n" * _FN_LPS_TB : "")
    lplist = "log_Vc, log_k10, log_k12, log_k21, log_ka, qt_base, " *
        "qt_slope, log_tgi_kg, log_tgi_kd" * (cont ? ", tgi_ly0" : "") *
        ", tgi_baseline_time"
    doparams = "subject_Vc, subject_k10, subject_k12, subject_k21, " *
        "subject_ka, qbase, qslope, tgi_g, tgi_k" *
        (cont ? ", tgi_b0" : "") * ", tgi_t0"
    (resp, doparam, sdline, obs) = config == "continuous" ?
        ("tgi_y", "tt", "",
            "tt .~ Normal.(tgi_mu, tgi_sigma)") :
        config == "ordinal" ?
        ("tgi_cc", "ccv",
            "tgi_sd = 1.4142135623730951 * tgi_sigma",
            "ccv .~ TgiCategory.(tgi_change, tgi_ref, tgi_c_cr, " *
            "-0.6931471805599453, 0.4054651081081644, tgi_sd, 0.01)") :
        ("tgi_bb", "bbv",
            "tgi_sd = 1.4142135623730951 * tgi_sigma",
            "bbv .~ TgiResponse.(tgi_change, tgi_ref, " *
            "-0.6931471805599453, 0.4054651081081644, tgi_sd, 0.01)")
    cell = _FN_CELL_HEAD * (cont ? "\n" * _FN_CELL_MU : "") *
        (isempty(sdline) ? "" : "\n            $sdline")
    return Meta.parse("begin\n$decl\n$varying\n$lps\n$_FN_SCHED\n" * """
        pk_loc ~ kernel(pk_conc, ecg_y, $resp, pk_lloq, qt_w, tgi_t,
                $lplist;
                subjects = kernel_nsub_pk_loc) do yy, qq, $doparam, pk_lloq,
                qt_weight, tgi_t, $doparams
""" * cell * "\n" * """
            yy .~ CensoredAddpropnormal.(mu, sigma_add, sigma_prop, pk_lloq)
            qq .~ Normal.(qt_loc, qt_sd)
            $obs
            mu
        end
    end""")
end

"""Lowered plan with SB per-margin varying sd priors attached (SB scales
p=1/3, tg=1/2, tb=1; the surface has no sd-prior syntax — contract-level
`VaryingSdPrior`, same as `joint_decl_varying`)."""
function final_plan(config::AbstractString = "continuous")
    plan = lower_rkppl(final_ast(config), final_data(config))
    scales = Dict("subject" => (0.3333333333333333, 7),
        "subject_d_tg" => (0.5, 2), "subject_d_tb" => (1.0, 1))
    for (i, d) in enumerate(plan.varying_draws)
        @assert haskey(scales, d.suffix) "unknown draws $(d.suffix)"
        (s, k) = scales[d.suffix]
        @assert length(d.margins) == k "margin drift $(d.suffix)"
        plan.varying_draws[i] = VaryingDraws(d.group, d.kind, d.margins,
            d.lkj_eta, d.label, d.suffix, d.levels,
            [VaryingSdPrior(:exponential, s) for _ in 1:k])
    end
    return plan
end

function final_data(config::AbstractString = "continuous")
    base = Set([:subject, :male, :age_yr, :weight_kg, :indication,
        :qt_prolonging_drug_ongoing, :subj, :time, :dsubj, :dtime,
        :damt, :pk_conc, :pk_lloq, :esubj, :etime, :ecg_y, :qt_w,
        :tsubj, :ttime_h, :tgi_t, :T0_data])
    push!(base, config == "continuous" ? :tgi_y :
        config == "ordinal" ? :tgi_cc : :tgi_bb)
    return base
end

_pop(lp::AbstractString) = "pop_$(lp)_beta_pop"
_cat(lp::AbstractString) = "cat_$(lp)_indication_beta"

"""Design-order fixed-coef map: predictor => [(SB stem, idx)...]."""
const _FIXED_MAP = Dict{Symbol,Vector{Tuple{String,Int}}}(
    :log_Vc => [(_pop("log_Vc"), 1), (_pop("log_Vc"), 2),
        (_pop("log_Vc"), 3), (_pop("log_Vc"), 4),
        (_cat("log_Vc"), 1)],
    :log_k10 => [(_pop("log_k10"), 1), (_pop("log_k10"), 2),
        (_pop("log_k10"), 3), (_pop("log_k10"), 4),
        (_cat("log_k10"), 1)],
    :log_k12 => [(_pop("log_k12"), 1), (_cat("log_k12"), 1)],
    :log_k21 => [(_pop("log_k21"), 1), (_cat("log_k21"), 1)],
    :log_ka => [(_pop("log_ka"), 1), (_cat("log_ka"), 1)],
    :qt_base => [(_pop("qt_base"), 1), (_pop("qt_base"), 2),
        (_pop("qt_base"), 3), (_pop("qt_base"), 4),
        (_cat("qt_base"), 1)],
    :qt_slope => [(_pop("qt_slope"), 1), (_cat("qt_slope"), 1)],
    :log_tgi_kg => [(_pop("log_tgi_kg"), 1)],
    :log_tgi_kd => [(_pop("log_tgi_kd"), 1)],
    :tgi_ly0 => [(_pop("tgi_ly0"), 1)],
)

"""RKPPL constrained-space NamedTuple at SB point1 θ (config)."""
function final_theta_nt(config::AbstractString)
    c = _parity_theta(config)
    L7 = [c["b_p_subject_L.$i.$j"] for i in 1:7, j in 1:7]
    Ltg = [c["b_tg_subject_L.$i.$j"] for i in 1:2, j in 1:2]
    base = Pair{Symbol,Any}[
        :sigma_add => c["sigma_add"], :sigma_prop => c["sigma_prop"],
        :qt_scale => c["qt_scale"], :tgi_sigma => c["tgi_sigma"],
        :L_subject => L7,
        :tau_subject => [c["b_p_subject_tau.$j"] for j in 1:7],
        :z_flat_subject => [c["b_p_subject_z_flat.$k"] for k in 1:21],
        :L_subject_d_tg => Ltg,
        :tau_subject_d_tg => [c["b_tg_subject_tau.$j"] for j in 1:2],
        :z_flat_subject_d_tg => [c["b_tg_subject_z_flat.$k"]
                                 for k in 1:6],
        :slope_log_F => c["pop_log_F_beta_pop.1"],
        :rho_log_F => c["hsgp_op_log_dose_rho_iso"],
        :sigma_log_F => c["hsgp_op_log_dose_sigma"],
        :beta_raw_log_F =>
            [c["hsgp_op_log_dose_beta_raw.$j"] for j in 1:5],
    ]
    # Zero-width data-passthrough LP (no coords, still keyed).
    push!(base, :tgi_baseline_time => Float64[])
    for (pred, terms) in _FIXED_MAP
        (config != "continuous" && pred === :tgi_ly0) && continue
        push!(base, pred => [c["$stem.$idx"] for (stem, idx) in terms])
    end
    push!(base, :tgi_c_cr => c["tgi_c_cr"])
    if config == "continuous"
        push!(base, :L_subject_d_tb => ones(1, 1))
        push!(base, :tau_subject_d_tb => [c["b_tb_subject_tau.1"]])
        push!(base, :z_flat_subject_d_tb =>
            [c["b_tb_subject_z_flat.$k"] for k in 1:3])
    end
    return NamedTuple{Tuple(first.(base))}(Tuple(last.(base)))
end

"""SB pinned constrained-space densities (lp − jac), point1 per config."""
const _PARITY_SB = Dict{String,Float64}(
    "continuous" => -349.3879680789393,
    "ordinal" => -96.44172759999246,
    "binary" => -92.72352892476464,
)

"""Documented global consts (RKPPL − SB); tighten to absolute 1e-9 once
the delegated const hunt resolves (user direction 2026-09-21)."""
const _PARITY_GAP = Dict{String,Float64}(
    "continuous" => -44.370536171807,
    "ordinal" => -34.7756857316521,
    "binary" => -34.7756857316521,
)
