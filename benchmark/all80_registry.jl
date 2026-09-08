# RK-binding registry for the all-implemented-model posteriordb benchmark.
#
# One entry per implemented idiomatic-RK posteriordb module. Each entry is a
# NamedTuple:
#   mod        :: Symbol   — ReactiveKernelsPPLExamples.<mod> (the Example module)
#   build      :: Symbol   — build_<name>_graph function in that module
#   have       :: Tuple    — prepare have-ports (first is :unconstrained)
#   bind       :: d::AbstractDict -> NamedTuple  — PosteriorDB data -> bound ports (FULL data)
#   off_rk     :: Float64  — EXPECTED (Stan - RK) log-density offset (0 = faithful same density)
#   off_tu     :: Float64  — EXPECTED (Stan - upstream-Turing) offset
#   off_reason :: String
#   boundary   :: nothing | q::Vector -> q_boundary  — probe OUTSIDE a restricted prior support
#   stan_perm  :: nothing | Vector{Int} — RK-order q -> artifact-Stan-order q. The reference
#     .stan declares its own parameter order, which need NOT match the RK unconstrained
#     packing (eight_schools_centered declares theta[8],mu,tau). Without this the Stan
#     oracle is evaluated at a scrambled point and every gate fails (harness bug, not RK).
#
# The graph is reached with getproperty(getproperty(ReactiveKernelsPPLExamples, mod), build)().
# Data field names are the upstream make_model's data["..."] keys (same PosteriorDB Dict).
# Reference Stan + data both come from PosteriorDB.jl; nothing is copied inline.
#
# `validate_registry` (below) is a MECHANICAL structural check — keys == the implemented
# module inventory, each key resolves to an upstream make_model posterior + a loadable
# PosteriorDB dataset. It does NOT check density parity; that is the per-model hard gate
# in the driver body (value offset + gradient + support), run at measurement time.

module All80Registry

using ReactiveKernelsPPLExamples

F(d, k) = Float64.(d[k]); Iv(d, k) = Int.(d[k]); Bv(d, k) = Bool.(d[k]); Is(d, k) = Int(d[k])

# arK lag design (mirrors ARKExample._ark_lag): ylag[i,k]=y[K+i-k], yt[i]=y[K+i].
function _ark_lag(y, K)
    T = length(y); yl = zeros(T - K, K)
    for i in 1:(T - K), k in 1:K; yl[i, k] = y[K + i - k]; end
    yl
end

const REGISTRY = Dict{String,NamedTuple}()
# All 82 posteriors have an upstream make_model (verified by DISPATCH APPLICABILITY in
# all80_validate.jl — not by text grep, which misses multiline signatures). Every row's
# Turing column is measured. `off_tu` is the EXACT source-derived, parameter-independent
# normalization offset (Stan − Turing): 0 unless the upstream Turing model adds explicit
# prior normalizers the reference Stan (implicit uniform on bounded params) omits, e.g.
# GLM_Poisson: off_tu = log(40)+3·log(20). The driver GATES measured == declared.
reg!(key; kw...) = (REGISTRY[key] = (; off_rk = 0.0, off_tu = 0.0,
    off_reason = "faithful same density; propto=false jacobian=true; no dropped constants",
    boundary = nothing, stan_perm = nothing, probe_q = nothing, kw...); nothing)

# ---------------- Poisson / binomial GLM ----------------
reg!("GLM_Poisson_Data-GLM_Poisson_model"; mod = :GLMPoissonExample, build = :build_glm_poisson_graph,
    have = (:unconstrained, :year, :counts), bind = d -> (year = F(d, "year"), counts = Iv(d, "C")),
    off_tu = log(40) + 3 * log(20),
    off_reason = "upstream Turing writes explicit alpha~U(-20,20), beta1:3~U(-10,10); reference .stan uses IMPLICIT uniform on the same bounds (constant 0). Stan−Turing = log(40)+3·log(20) ≈ 12.676; HMC-invariant, gradient/support identical.")
reg!("GLM_Binomial_data-GLM_Binomial_model"; mod = :GLMBinomialExample, build = :build_glm_binomial_graph,
    have = (:unconstrained, :year, :counts, :totals), bind = d -> (year = F(d, "year"), counts = Iv(d, "C"), totals = Iv(d, "N")))
# GLMM_Poisson: beta2 declared <-10,20> but prior uniform(-10,10) -> support boundary at beta2>10.
reg!("GLMM_Poisson_data-GLMM_Poisson_model"; mod = :GLMMPoissonExample, build = :build_glmm_poisson_graph,
    have = (:unconstrained, :year, :counts), bind = d -> (year = F(d, "year"), counts = Iv(d, "C")),
    off_reason = "faithful; beta2 support restriction (uniform(-10,10) inside <-10,20> transform)",
    boundary = q -> (qb = copy(q); qb[3] = 5.0; qb))   # u_beta2=5 -> beta2 ~ 19.8 > 10

# ---------------- Gaussian linear regression ----------------
reg!("sblri-blr"; mod = :BLRExample, build = :build_blr_graph, have = (:unconstrained, :predictors, :responses),
    bind = d -> (predictors = _mat(d["X"]), responses = F(d, "y")))
reg!("kilpisjarvi_mod-kilpisjarvi"; mod = :KilpisjarviExample, build = :build_kilpisjarvi_graph,
    have = (:unconstrained, :x, :y, :xpred, :pmualpha, :psalpha, :pmubeta, :psbeta),
    bind = d -> (x = F(d, "x"), y = F(d, "y"), xpred = haskey(d, "xpred") ? F(d, "xpred") : F(d, "x"),
                 pmualpha = Float64(d["pmualpha"]), psalpha = Float64(d["psalpha"]),
                 pmubeta = Float64(d["pmubeta"]), psbeta = Float64(d["psbeta"])))

_mat(x) = x isa AbstractMatrix ? Float64.(x) : reduce(vcat, [permutedims(Float64.(r)) for r in x])

# kidscore (kidiq) — ARM Ch.3/4 Gaussian regressions
reg!("kidiq-kidscore_momhs"; mod = :KidscoreMomhsExample, build = :build_kidscore_momhs_graph,
    have = (:unconstrained, :kid_score, :mom_hs), bind = d -> (kid_score = F(d, "kid_score"), mom_hs = F(d, "mom_hs")))
reg!("kidiq-kidscore_momiq"; mod = :KidscoreMomiqExample, build = :build_kidscore_momiq_graph,
    have = (:unconstrained, :kid_score, :mom_iq), bind = d -> (kid_score = F(d, "kid_score"), mom_iq = F(d, "mom_iq")))
reg!("kidiq-kidscore_momhsiq"; mod = :KidscoreMomhsiqExample, build = :build_kidscore_momhsiq_graph,
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq), bind = d -> (kid_score = F(d, "kid_score"), mom_hs = F(d, "mom_hs"), mom_iq = F(d, "mom_iq")))
reg!("kidiq-kidscore_interaction"; mod = :KidscoreInteractionExample, build = :build_kidscore_interaction_graph,
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq), bind = d -> (kid_score = F(d, "kid_score"), mom_hs = F(d, "mom_hs"), mom_iq = F(d, "mom_iq")))
reg!("kidiq_with_mom_work-kidscore_interaction_c"; mod = :KidscoreInteractionCExample, build = :build_kidscore_interaction_c_graph,
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq), bind = d -> (kid_score = F(d, "kid_score"), mom_hs = F(d, "mom_hs"), mom_iq = F(d, "mom_iq")))
reg!("kidiq_with_mom_work-kidscore_interaction_c2"; mod = :KidscoreInteractionC2Example, build = :build_kidscore_interaction_c2_graph,
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq), bind = d -> (kid_score = F(d, "kid_score"), mom_hs = F(d, "mom_hs"), mom_iq = F(d, "mom_iq")))
reg!("kidiq_with_mom_work-kidscore_interaction_z"; mod = :KidscoreInteractionZExample, build = :build_kidscore_interaction_z_graph,
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq), bind = d -> (kid_score = F(d, "kid_score"), mom_hs = F(d, "mom_hs"), mom_iq = F(d, "mom_iq")))
# work2/3/4 are indicator contrasts of the 4-level factor mom_work (ref = level 1),
# matching upstream X = hcat(ones, mom_work.==2, mom_work.==3, mom_work.==4).
reg!("kidiq_with_mom_work-kidscore_mom_work"; mod = :KidscoreMomWorkExample, build = :build_kidscore_mom_work_graph,
    have = (:unconstrained, :kid_score, :work2, :work3, :work4),
    bind = d -> (let mw = Iv(d, "mom_work"); (kid_score = F(d, "kid_score"),
        work2 = Float64.(mw .== 2), work3 = Float64.(mw .== 3), work4 = Float64.(mw .== 4)) end))

# earnings
reg!("earnings-earn_height"; mod = :EarnHeightExample, build = :build_earn_height_graph, have = (:unconstrained, :height, :earn),
    bind = d -> (height = F(d, "height"), earn = F(d, "earn")))
reg!("earnings-log10earn_height"; mod = :Log10earnHeightExample, build = :build_log10earn_height_graph, have = (:unconstrained, :height, :earn),
    bind = d -> (height = F(d, "height"), earn = F(d, "earn")))
reg!("earnings-logearn_height"; mod = :LogearnHeightExample, build = :build_logearn_height_graph, have = (:unconstrained, :height, :earn),
    bind = d -> (height = F(d, "height"), earn = F(d, "earn")))
reg!("earnings-logearn_height_male"; mod = :LogearnHeightMaleExample, build = :build_logearn_height_male_graph, have = (:unconstrained, :height, :male, :earn),
    bind = d -> (height = F(d, "height"), male = F(d, "male"), earn = F(d, "earn")))
reg!("earnings-logearn_interaction"; mod = :LogearnInteractionExample, build = :build_logearn_interaction_graph, have = (:unconstrained, :height, :male, :earn),
    bind = d -> (height = F(d, "height"), male = F(d, "male"), earn = F(d, "earn")))
reg!("earnings-logearn_interaction_z"; mod = :LogearnInteractionZExample, build = :build_logearn_interaction_z_graph, have = (:unconstrained, :height, :male, :earn),
    bind = d -> (height = F(d, "height"), male = F(d, "male"), earn = F(d, "earn")))
reg!("earnings-logearn_logheight_male"; mod = :LogearnLogheightMaleExample, build = :build_logearn_logheight_male_graph, have = (:unconstrained, :height, :male, :earn),
    bind = d -> (height = F(d, "height"), male = F(d, "male"), earn = F(d, "earn")))

# mesquite
reg!("mesquite-mesquite"; mod = :MesquiteExample, build = :build_mesquite_graph,
    have = (:unconstrained, :weight, :diam1, :diam2, :canopy_height, :total_height, :density, :group),
    bind = d -> (weight = F(d, "weight"), diam1 = F(d, "diam1"), diam2 = F(d, "diam2"), canopy_height = F(d, "canopy_height"),
                 total_height = F(d, "total_height"), density = F(d, "density"), group = F(d, "group")))
reg!("mesquite-logmesquite"; mod = :LogmesquiteExample, build = :build_logmesquite_graph,
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height, :total_height, :density, :group),
    bind = d -> (log_weight = log.(F(d, "weight")), diam1 = F(d, "diam1"), diam2 = F(d, "diam2"), canopy_height = F(d, "canopy_height"),
                 total_height = F(d, "total_height"), density = F(d, "density"), group = F(d, "group")))
reg!("mesquite-logmesquite_logvolume"; mod = :LogmesquiteLogvolumeExample, build = :build_logmesquite_logvolume_graph,
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height),
    bind = d -> (log_weight = log.(F(d, "weight")), diam1 = F(d, "diam1"), diam2 = F(d, "diam2"), canopy_height = F(d, "canopy_height")))
reg!("mesquite-logmesquite_logva"; mod = :LogmesquiteLogvaExample, build = :build_logmesquite_logva_graph,
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height, :group),
    bind = d -> (log_weight = log.(F(d, "weight")), diam1 = F(d, "diam1"), diam2 = F(d, "diam2"), canopy_height = F(d, "canopy_height"), group = F(d, "group")))
reg!("mesquite-logmesquite_logvas"; mod = :LogmesquiteLogvasExample, build = :build_logmesquite_logvas_graph,
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height, :total_height, :density, :group),
    bind = d -> (log_weight = log.(F(d, "weight")), diam1 = F(d, "diam1"), diam2 = F(d, "diam2"), canopy_height = F(d, "canopy_height"),
                 total_height = F(d, "total_height"), density = F(d, "density"), group = F(d, "group")))
reg!("mesquite-logmesquite_logvash"; mod = :LogmesquiteLogvashExample, build = :build_logmesquite_logvash_graph,
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height, :total_height, :group),
    bind = d -> (log_weight = log.(F(d, "weight")), diam1 = F(d, "diam1"), diam2 = F(d, "diam2"), canopy_height = F(d, "canopy_height"),
                 total_height = F(d, "total_height"), group = F(d, "group")))

# nes (one module serves several nes* posteriors; use the representative nes1972-nes)
# nes N×9 design matrix in Stan column order [1, real_ideo, race_adj, age30_44,
# age45_64, age65up, educ1, gender, income]; age bands are indicators of age_discrete
# (levels 2/3/4; level 1 is the reference). Matches upstream _make_nes + nes.stan.
reg!("nes1972-nes"; mod = :NESExample, build = :build_nes_graph, have = (:unconstrained, :predictors, :responses),
    bind = d -> (let ad = Iv(d, "age_discrete");
        (predictors = hcat(ones(length(ad)), F(d, "real_ideo"), F(d, "race_adj"),
             Float64.(ad .== 2), Float64.(ad .== 3), Float64.(ad .== 4),
             F(d, "educ1"), F(d, "gender"), F(d, "income")),
         responses = F(d, "partyid7")) end))

# eight_schools
reg!("eight_schools-eight_schools_centered"; mod = :EightSchoolsExample, build = :build_eight_schools_graph,
    have = (:unconstrained, :observations, :observation_scales), bind = d -> (observations = F(d, "y"), observation_scales = F(d, "sigma")),
    stan_perm = [3, 4, 5, 6, 7, 8, 9, 10, 1, 2],
    off_rk = -log(2.0),
    off_tu = -log(2.0),
    off_reason = "artifact .stan declares (theta[8],mu,tau), RK packs (mu,logtau,theta[8]) — stan_perm reorders (verified by probe). The centered RK graph uses the PROPER HalfCauchy(0,5) (+log2 normalizer) while the reference .stan drops the truncation normalizer: Stan−RK = −log2, stable, grads 1e-16 (same class as arK/arma11; upstream Turing is also proper, Stan−Turing = −log2). NOTE: the noncentered RK module matches Stan with off 0 — the two RK modules differ by log2 in the tau prior; flagged to the posteriordb lane, each side gated against its own .stan here.")
reg!("eight_schools-eight_schools_noncentered"; mod = :EightSchoolsNoncenteredExample, build = :build_eight_schools_noncentered_graph,
    have = (:unconstrained, :observations, :observation_scales), bind = d -> (observations = F(d, "y"), observation_scales = F(d, "sigma")),
    off_tu = -log(2.0),
    off_reason = "faithful same density; upstream Turing's truncated Cauchy on tau includes the +log2 half-line normalizer the reference .stan drops: Stan−Turing = −log2 (measured −0.6931471805599401, grads 1e-16).")

# time series
reg!("arK-arK"; mod = :ARKExample, build = :build_ark_graph, have = (:unconstrained, :ylag, :yt),
    bind = d -> (ylag = _ark_lag(F(d, "y"), Is(d, "K")), yt = F(d, "y")[(Is(d, "K") + 1):end]),
    off_tu = -log(2.0),
    off_reason = "upstream Turing's truncated/half-Cauchy on σ includes the +log2 half-line normalizer that reference .stan (implicit lower=0 + cauchy_lpdf) drops. Stan−Turing = −log2 ≈ −0.693 (source-derived, observed-confirmed).")
reg!("arma-arma11"; mod = :ARMA11Example, build = :build_arma11_graph, have = (:unconstrained, :series),
    bind = d -> (series = F(d, "y"),),
    off_rk = -log(2.0),
    off_tu = -log(2.0),
    off_reason = "the RK graph AND upstream Turing both use the PROPER truncated/half-Cauchy on σ (the +log2 half-line normalizer) that reference .stan (implicit lower=0 + cauchy_lpdf) drops: Stan−RK = Stan−Turing = −log2 ≈ −0.693 (source-derived; RK offset measured exactly −log(2) — ReactiveKernels:performance arma11 replay). Same class as eight_schools_centered / arK; the off_rk=0 default here was a metadata omission (off_tu was declared, off_rk was not), independent of the authored-scan Enzyme-activity performance snag.")

# growth
reg!("dugongs_data-dugongs_model"; mod = :DugongsGrowthExample, build = :build_dugongs_graph, have = (:unconstrained, :ages, :lengths),
    bind = d -> (ages = F(d, "x"), lengths = F(d, "Y")))

# rate models
reg!("Rate_1_data-Rate_1_model"; mod = :Rate1Example, build = :build_rate_1_graph, have = (:unconstrained, :n, :k),
    bind = d -> (n = Is(d, "n"), k = Is(d, "k")))
reg!("Rate_2_data-Rate_2_model"; mod = :Rate2Example, build = :build_rate_2_graph, have = (:unconstrained, :n1, :n2, :k1, :k2),
    bind = d -> (n1 = Is(d, "n1"), n2 = Is(d, "n2"), k1 = Is(d, "k1"), k2 = Is(d, "k2")))
reg!("Rate_3_data-Rate_3_model"; mod = :Rate3Example, build = :build_rate_3_graph, have = (:unconstrained, :n1, :n2, :k1, :k2),
    bind = d -> (n1 = Is(d, "n1"), n2 = Is(d, "n2"), k1 = Is(d, "k1"), k2 = Is(d, "k2")))
reg!("Rate_4_data-Rate_4_model"; mod = :Rate4Example, build = :build_rate_4_graph, have = (:unconstrained, :n, :k),
    bind = d -> (n = Is(d, "n"), k = Is(d, "k")))
reg!("Rate_5_data-Rate_5_model"; mod = :Rate5Example, build = :build_rate_5_graph, have = (:unconstrained, :n1, :n2, :k1, :k2),
    bind = d -> (n1 = Is(d, "n1"), n2 = Is(d, "n2"), k1 = Is(d, "k1"), k2 = Is(d, "k2")))

# ---------------- Bernoulli-logit GLM (wells / dogs / nes_logit / sesame) ----------------
reg!("wells_data-wells_dist"; mod = :WellsDistExample, build = :build_wells_dist_graph, have = (:unconstrained, :dist, :switched),
    bind = d -> (dist = F(d, "dist"), switched = Bv(d, "switched")))
reg!("wells_data-wells_dist100_model"; mod = :WellsDist100Example, build = :build_wells_dist100_graph, have = (:unconstrained, :dist, :switched),
    bind = d -> (dist = F(d, "dist"), switched = Bv(d, "switched")))
reg!("wells_data-wells_dist100ars_model"; mod = :WellsDist100arsExample, build = :build_wells_dist100ars_graph, have = (:unconstrained, :dist, :arsenic, :switched),
    bind = d -> (dist = F(d, "dist"), arsenic = F(d, "arsenic"), switched = Bv(d, "switched")))
reg!("wells_data-wells_interaction_model"; mod = :WellsInteractionExample, build = :build_wells_interaction_graph, have = (:unconstrained, :dist, :arsenic, :switched),
    bind = d -> (dist = F(d, "dist"), arsenic = F(d, "arsenic"), switched = Bv(d, "switched")))
reg!("wells_data-wells_interaction_c_model"; mod = :WellsInteractionCExample, build = :build_wells_interaction_c_graph, have = (:unconstrained, :dist, :arsenic, :switched),
    bind = d -> (dist = F(d, "dist"), arsenic = F(d, "arsenic"), switched = Bv(d, "switched")))
reg!("wells_data-wells_dae_model"; mod = :WellsDaeExample, build = :build_wells_dae_graph, have = (:unconstrained, :dist, :arsenic, :educ, :switched),
    bind = d -> (dist = F(d, "dist"), arsenic = F(d, "arsenic"), educ = F(d, "educ"), switched = Bv(d, "switched")))
reg!("wells_data-wells_dae_c_model"; mod = :WellsDaeCExample, build = :build_wells_dae_c_graph, have = (:unconstrained, :dist, :arsenic, :educ, :switched),
    bind = d -> (dist = F(d, "dist"), arsenic = F(d, "arsenic"), educ = F(d, "educ"), switched = Bv(d, "switched")))
reg!("wells_data-wells_dae_inter_model"; mod = :WellsDaeInterExample, build = :build_wells_dae_inter_graph, have = (:unconstrained, :dist, :arsenic, :educ, :switched),
    bind = d -> (dist = F(d, "dist"), arsenic = F(d, "arsenic"), educ = F(d, "educ"), switched = Bv(d, "switched")))
reg!("wells_data-wells_daae_c_model"; mod = :WellsDaaeCExample, build = :build_wells_daae_c_graph, have = (:unconstrained, :dist, :arsenic, :assoc, :educ, :switched),
    bind = d -> (dist = F(d, "dist"), arsenic = F(d, "arsenic"), assoc = F(d, "assoc"), educ = F(d, "educ"), switched = Bv(d, "switched")))
reg!("dogs-dogs"; mod = :DogsExample, build = :build_dogs_graph, have = (:unconstrained, :n_avoid, :n_shock, :y),
    bind = d -> (let (na, ns, yf) = _dogs_ports(d); (n_avoid = na, n_shock = ns, y = yf) end))
reg!("dogs-dogs_log"; mod = :DogsLogExample, build = :build_dogs_log_graph, have = (:unconstrained, :n_avoid, :n_shock, :y),
    bind = d -> (let (na, ns, yf) = _dogs_ports(d); (n_avoid = na, n_shock = ns, y = yf) end),
    probe_q = [-0.2, 0.1])  # zeros(2) sits on both Uniform box boundaries (beta1∈[-100,0], beta2∈[0,100])
reg!("dogs-dogs_hierarchical"; mod = :DogsHierarchicalExample, build = :build_dogs_hierarchical_graph, have = (:unconstrained, :prev_avoid, :prev_shock, :y),
    bind = d -> (let (na, ns, yf) = _dogs_ports(d); (prev_avoid = na, prev_shock = ns, y = yf) end))
reg!("nes_logit_data-nes_logit_model"; mod = :NesLogitExample, build = :build_nes_logit_graph, have = (:unconstrained, :income, :vote),
    bind = d -> (income = F(d, "income"), vote = Bv(d, "vote")))
reg!("sesame_data-sesame_one_pred_a"; mod = :SesameOnePredAExample, build = :build_sesame_one_pred_a_graph, have = (:unconstrained, :encouraged, :watched),
    bind = d -> (encouraged = F(d, "encouraged"), watched = F(d, "watched")))

# ---------------- binomial-logit GLMM / hierarchical ----------------
reg!("seeds_data-seeds_model"; mod = :SeedsExample, build = :build_seeds_graph, have = (:unconstrained, :counts, :totals, :x1, :x2),
    bind = d -> (counts = Iv(d, "n"), totals = Iv(d, "N"), x1 = F(d, "x1"), x2 = F(d, "x2")))
reg!("seeds_data-seeds_centered_model"; mod = :SeedsCenteredExample, build = :build_seeds_centered_model_graph, have = (:unconstrained, :counts, :totals, :x1, :x2),
    bind = d -> (counts = Iv(d, "n"), totals = Iv(d, "N"), x1 = F(d, "x1"), x2 = F(d, "x2")))
reg!("seeds_data-seeds_stanified_model"; mod = :SeedsStanifiedExample, build = :build_seeds_stanified_model_graph, have = (:unconstrained, :counts, :totals, :x1, :x2),
    bind = d -> (counts = Iv(d, "n"), totals = Iv(d, "N"), x1 = F(d, "x1"), x2 = F(d, "x2")))
reg!("surgical_data-surgical_model"; mod = :SurgicalExample, build = :build_surgical_graph, have = (:unconstrained, :successes, :totals),
    bind = d -> (successes = Iv(d, "r"), totals = Iv(d, "n")))
reg!("lsat_data-lsat_model"; mod = :LsatExample, build = :build_lsat_graph, have = (:unconstrained, :student_idx, :question_idx, :response),
    bind = d -> _lsat_bind(d))
reg!("pilots-pilots"; mod = :PilotsExample, build = :build_pilots_graph, have = (:unconstrained, :group_id, :scenario_id, :y),
    bind = d -> (group_id = Iv(d, "group_id"), scenario_id = Iv(d, "scenario_id"), y = F(d, "y")))
reg!("rats_data-rats_model"; mod = :RatsModelExample, build = :build_rats_model_graph, have = (:unconstrained, :rat, :x, :y, :xbar),
    bind = d -> _rats_bind(d))
reg!("election88-election88_full"; mod = :Election88FullExample, build = :build_election88_full_graph,
    have = (:unconstrained, :age, :edu, :age_edu, :state, :region_full, :black, :female, :v_prev_full, :y,
            :n_age, :n_edu, :n_age_edu, :n_state, :n_region_full),
    bind = d -> _election88_bind(d))

# ---------------- Gaussian / hierarchical: radon (radon_mn data variant) ----------------
for (k, mod, bf, hv) in [
    ("radon_mn-radon_pooled", :RadonPooledExample, :build_radon_pooled_graph, (:unconstrained, :floor_measure, :log_radon)),
    ("radon_mn-radon_county_intercept", :RadonCountyInterceptExample, :build_radon_county_intercept_graph, (:unconstrained, :county_idx, :floor_measure, :log_radon)),
    ("radon_mn-radon_partially_pooled_centered", :RadonPartiallyPooledCenteredExample, :build_radon_partially_pooled_centered_graph, (:unconstrained, :county_idx, :log_radon)),
    ("radon_mn-radon_partially_pooled_noncentered", :RadonPartiallyPooledNoncenteredExample, :build_radon_partially_pooled_noncentered_graph, (:unconstrained, :county_idx, :log_radon)),
    ("radon_mn-radon_variable_intercept_centered", :RadonVariableInterceptCenteredExample, :build_radon_variable_intercept_centered_graph, (:unconstrained, :county_idx, :floor_measure, :log_radon)),
    ("radon_mn-radon_variable_intercept_noncentered", :RadonVariableInterceptNoncenteredExample, :build_radon_variable_intercept_noncentered_graph, (:unconstrained, :county_idx, :floor_measure, :log_radon)),
    ("radon_mn-radon_variable_slope_centered", :RadonVariableSlopeCenteredExample, :build_radon_variable_slope_centered_graph, (:unconstrained, :county_idx, :floor_measure, :log_radon)),
    ("radon_mn-radon_variable_slope_noncentered", :RadonVariableSlopeNoncenteredExample, :build_radon_variable_slope_noncentered_graph, (:unconstrained, :county_idx, :floor_measure, :log_radon)),
    ("radon_mn-radon_variable_intercept_slope_centered", :RadonVariableInterceptSlopeCenteredExample, :build_radon_variable_intercept_slope_centered_graph, (:unconstrained, :county_idx, :floor_measure, :log_radon)),
    ("radon_mn-radon_variable_intercept_slope_noncentered", :RadonVariableInterceptSlopeNoncenteredExample, :build_radon_variable_intercept_slope_noncentered_graph, (:unconstrained, :county_idx, :floor_measure, :log_radon)),
    ("radon_mn-radon_hierarchical_intercept_centered", :RadonHierarchicalInterceptCenteredExample, :build_radon_hierarchical_intercept_centered_graph, (:unconstrained, :county_idx, :log_uppm, :floor_measure, :log_radon)),
    ("radon_mn-radon_hierarchical_intercept_noncentered", :RadonHierarchicalInterceptNoncenteredExample, :build_radon_hierarchical_intercept_noncentered_graph, (:unconstrained, :county_idx, :log_uppm, :floor_measure, :log_radon)),
    ("radon_mod-radon_county", :RadonCountyExample, :build_radon_county_graph, (:unconstrained, :county_idx, :log_radon)),
]
    reg!(k; mod = mod, build = bf, have = hv, bind = d -> _radon_bind(d, hv))
end

# ---------------- mixtures (marginalized) ----------------
reg!("normal_2-normal_mixture"; mod = :NormalMixtureExample, build = :build_normal_mixture_graph, have = (:unconstrained, :y), bind = d -> (y = F(d, "y"),),
    probe_q = [0.0, -10.0, 10.0])  # separated components; zeros→mu1==mu2 logsumexp tie (AD subgradient ambiguous)
reg!("low_dim_gauss_mix-low_dim_gauss_mix"; mod = :LowDimGaussMixExample, build = :build_low_dim_gauss_mix_graph, have = (:unconstrained, :y), bind = d -> (y = F(d, "y"),))
reg!("low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse"; mod = :LowDimGaussMixCollapseExample, build = :build_low_dim_gauss_mix_collapse_graph, have = (:unconstrained, :y), bind = d -> (y = F(d, "y"),),
    probe_q = [-1.0, 1.0, 0.0, 0.0, 0.0])  # separated components; zeros→equal scales/weights logsumexp tie

# ---------------- capture-recapture (data augmentation; derived binds — run-validated) ----------------
reg!("M0_data-M0_model"; mod = :M0Example, build = :build_m0_graph, have = (:unconstrained, :s, :lchoose, :T, :M), bind = d -> _cr_bind_m0(d))
reg!("Mb_data-Mb_model"; mod = :MbExample, build = :build_mb_graph, have = (:unconstrained, :a, :b, :e, :f, :s, :T, :M), bind = d -> _cr_bind_mb(d))
reg!("Mt_data-Mt_model"; mod = :MtExample, build = :build_mt_graph, have = (:unconstrained, :Y, :s, :T, :M), bind = d -> _cr_bind_mt(d))
reg!("Mh_data-Mh_model"; mod = :MhExample, build = :build_mh_graph, have = (:unconstrained, :y, :lchoose, :T, :M), bind = d -> _cr_bind_mh(d),
    off_tu = log(5.0),
    off_reason = "upstream Turing writes sigma~Uniform(0,5) (omega,mean_p~U(0,1) contribute 0); reference .stan uses implicit uniform on the same bounds. Stan−Turing = log(5) ≈ 1.609 (source-derived, observed-confirmed).")
reg!("Mth_data-Mth_model"; mod = :MthModelExample, build = :build_mth_model_graph, have = (:unconstrained, :Y, :s, :T, :M), bind = d -> _cr_bind_mt(d))
reg!("Mtbh_data-Mtbh_model"; mod = :MtbhModelExample, build = :build_mtbh_model_graph, have = (:unconstrained, :Y, :Yprev, :s, :T, :M), bind = d -> _cr_bind_mtbh(d))

# ---------------- discrete marginalization ----------------
reg!("Survey_data-Survey_model"; mod = :SurveyModelExample, build = :build_survey_model_graph, have = (:unconstrained, :ns, :lc, :sk, :m, :log_1_nmax), bind = d -> _survey_bind(d))

# --- bind helpers (full real posteriordb data -> RK have-ports; reuse module transforms) ---
# dogs: reuse the module's own running-count design (n_avoid/n_shock BEFORE each trial,
# 0 at t=1; y flattened dog-major) over the FULL n_dogs×n_trials shock matrix data["y"].
function _dogs_ymat(d)
    y = d["y"]
    y isa AbstractMatrix ? Bool.(round.(Int, y)) :
        Bool.(reduce(vcat, [permutedims(round.(Int, collect(r))) for r in y]))
end
_dogs_ports(d) = ReactiveKernelsPPLExamples.DogsExample._dogs_design(_dogs_ymat(d))

# radon: all 12 radon_mn variants share dataset radon_mn (N=919, J=85; keys county_idx,
# floor_measure, log_radon, log_uppm[per-obs length N]). radon_mod-radon_county is the
# ONLY renamer (keys county/y, no floor/uppm); haskey discriminates. county_idx is
# already 1-based (int<lower=1,upper=J>). Only the ports in `hv` are read.
function _radon_bind(d, hv)
    port = Dict{Symbol,Any}(
        :county_idx    => () -> haskey(d, "county_idx") ? Iv(d, "county_idx") : Iv(d, "county"),
        :floor_measure => () -> F(d, "floor_measure"),
        :log_uppm      => () -> F(d, "log_uppm"),
        :log_radon     => () -> haskey(d, "log_radon") ? F(d, "log_radon") : F(d, "y"),
    )
    ks = Tuple(k for k in hv if k != :unconstrained)
    return NamedTuple{ks}(map(k -> port[k](), ks))
end
# lsat: expand culm/response (R patterns × T questions) into the T×N r[k,j] table
# (mirrors lsat_model.stan transformed data), flattened question-outer/student-inner.
function _lsat_bind(d)
    N = Is(d, "N"); R = Is(d, "R"); T = Is(d, "T")
    culm = Iv(d, "culm"); resp = d["response"]
    rkj(i, k) = Int(resp isa AbstractMatrix ? resp[i, k] : resp[i][k])
    r = zeros(Int, T, N)
    for j in 1:culm[1], k in 1:T; r[k, j] = rkj(1, k); end
    for i in 2:R, j in (culm[i-1] + 1):culm[i], k in 1:T; r[k, j] = rkj(i, k); end
    (; student_idx = Int[j for k in 1:T for j in 1:N],
       question_idx = Int[k for k in 1:T for j in 1:N],
       response = Int[r[k, j] for k in 1:T for j in 1:N])
end

# rats: direct passthrough (N/Npts are structural, inferred from lengths).
_rats_bind(d) = (; rat = Iv(d, "rat"), x = F(d, "x"), y = F(d, "y"), xbar = Float64(d["xbar"]))

# election88_full: 14 data ports; the five n_* are DIRECT data keys (group sizes), not maxima.
_election88_bind(d) = (;
    age = Iv(d, "age"), edu = Iv(d, "edu"), age_edu = Iv(d, "age_edu"),
    state = Iv(d, "state"), region_full = Iv(d, "region_full"),
    black = F(d, "black"), female = F(d, "female"), v_prev_full = F(d, "v_prev_full"),
    y = Bool.(round.(Int, d["y"])),
    n_age = Is(d, "n_age"), n_edu = Is(d, "n_edu"), n_age_edu = Is(d, "n_age_edu"),
    n_state = Is(d, "n_state"), n_region_full = Is(d, "n_region_full"))

# capture-recapture (data augmentation). PosteriorDB returns y as Matrix{Int} M×T
# (M0/Mb/Mt/Mth/Mtbh) or Vector{Int} length M (Mh); the modules compute their derived
# ports at const-time, so these re-implement the SAME transforms (matched byte-for-byte
# vs the module const AND the upstream pdb_ model).
function _cr_bind_m0(d)
    T = Is(d, "T"); M = Is(d, "M"); s = Int.(vec(sum(d["y"]; dims = 2)))
    (; s, lchoose = [log(float(binomial(T, si))) for si in s], T, M)
end
function _cr_bind_mb(d)
    y = Int.(d["y"]); M = Is(d, "M"); T = Is(d, "T")
    a = zeros(M); b = zeros(M); e = zeros(M); f = zeros(M)
    for i in 1:M
        a[i] += y[i, 1]; b[i] += 1 - y[i, 1]
        for t in 2:T
            if y[i, t - 1] == 0
                a[i] += y[i, t]; b[i] += 1 - y[i, t]
            else
                e[i] += y[i, t]; f[i] += 1 - y[i, t]
            end
        end
    end
    (; a, b, e, f, s = Int.(vec(sum(y; dims = 2))), T, M)
end
function _cr_bind_mt(d)   # Mt AND Mth (identical :Y,:s,:T,:M derivation)
    Y = Float64.(d["y"]); T = Is(d, "T"); M = Is(d, "M")
    (; Y, s = Int.(vec(sum(Y; dims = 2))), T, M)
end
function _cr_bind_mh(d)
    T = Is(d, "T"); M = Is(d, "M"); y = Iv(d, "y")
    (; y, lchoose = [log(float(binomial(T, yi))) for yi in y], T, M)
end
function _cr_bind_mtbh(d)
    Y = Float64.(d["y"]); T = Is(d, "T"); M = Is(d, "M")
    (; Y, Yprev = hcat(zeros(M), Y[:, 1:T - 1]), s = Int.(vec(sum(Y; dims = 2))), T, M)
end

# Survey: sufficient-statistic marginalization terms; reuse the module's exact lchoose.
const _SURVEY_LCHOOSE = ReactiveKernelsPPLExamples.SurveyModelExample._survey_lchoose
function _survey_bind(d)
    nmax = Is(d, "nmax"); m = Is(d, "m"); k = Iv(d, "k")
    (; ns = Float64.(1:nmax),
       lc = Float64[sum(_SURVEY_LCHOOSE(n, ki) for ki in k) for n in 1:nmax],
       sk = Float64(sum(k)), m = m, log_1_nmax = -log(Float64(nmax)))
end

# Source-derived off_tu (EXPECTED Stan−Turing normalizer offset) for models whose upstream Turing
# writes an explicit prior normalizer the reference .stan omits (implicit uniform / ordinary lpdf).
# Applied as a post-definition merge so it also covers loop-registered radon_county. Constants
# VERIFIED against the pinned DPPL 6378673 posteriordb_models.jl (SHA256 a7ef985b…) + posteriordb-1.0.0
# .stan by ReactiveKernels:performance (2026-09-08); recorded RK offsets ~0 / gradients agree — these
# are Turing-side normalization metadata, NOT RK defects. The gate re-verifies measured==declared on
# the next run (historical receipts keep their own provenance; old measurements are not relabeled).
for (k, otu, why) in [
    ("kidiq-kidscore_momhs",       -log(2),     "upstream Turing truncated Cauchy(0,2.5) (+log2 half-line normalizer); reference .stan ordinary cauchy at sigma>0. Stan−Turing=−log2 (posteriordb_models.jl:467-525)"),
    ("kidiq-kidscore_momiq",       -log(2),     "upstream Turing truncated Cauchy(0,2.5) (+log2); reference .stan ordinary cauchy at sigma>0. Stan−Turing=−log2 (posteriordb_models.jl:467-525)"),
    ("kidiq-kidscore_momhsiq",     -log(2),     "upstream Turing truncated Cauchy(0,2.5) (+log2); reference .stan ordinary cauchy at sigma>0. Stan−Turing=−log2 (posteriordb_models.jl:467-525)"),
    ("kidiq-kidscore_interaction", -log(2),     "upstream Turing truncated Cauchy(0,2.5) (+log2); reference .stan ordinary cauchy at sigma>0. Stan−Turing=−log2 (posteriordb_models.jl:467-525)"),
    ("sblri-blr",                  -log(2),     "upstream Turing half-Normal(0,10) (+log2); blr.stan ordinary normal at sigma>0. Stan−Turing=−log2 (posteriordb_models.jl:799-807, blr.stan:13-14)"),
    ("Mtbh_data-Mtbh_model",        log(3),     "upstream Turing sigma~Uniform(0,3) (−log3); .stan bounds + uniform commented out. Stan−Turing=+log3 (posteriordb_models.jl:1432, Mtbh.stan:22,48)"),
    ("Mth_data-Mth_model",          log(5),     "upstream Turing sigma~Uniform(0,5) (−log5); .stan bounds + uniform commented out. Stan−Turing=+log5 (posteriordb_models.jl:1463, Mth.stan:21,39)"),
    ("election88-election88_full",  5*log(100), "five upstream Turing Uniform(0,100); .stan bounded sigmas, no density prior. Stan−Turing=+5log100 (posteriordb_models.jl:1900-1904, election88.stan:25-29)"),
    ("pilots-pilots",               3*log(100), "three upstream Turing Uniform(0,100); .stan bounded, no prior. Stan−Turing=+3log100 (pilots.stan:14-16)"),
    ("radon_mod-radon_county",      2*log(100), "two upstream Turing Uniform(0,100); .stan bounds, no prior. Stan−Turing=+2log100 (radon_county.stan:10-11)"),
    ("low_dim_gauss_mix-low_dim_gauss_mix", -log(4), "upstream Turing two truncated Normal(0,2) scales (+2log2); .stan constrained scales + ordinary normal. Stan−Turing=−log4; ordered-mu convention retained (posteriordb_models.jl:846-855, low_dim_gauss_mix.stan:7,11)"),
]
    haskey(REGISTRY, k) || error("off_tu declaration for unknown registry key $k")
    REGISTRY[k] = (; REGISTRY[k]..., off_tu = otu, off_reason = why)
end

end # module All80Registry
