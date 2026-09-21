using DifferentiationInterface
using Distributions: Normal, cdf, logcdf, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# TGI block cell vocabulary + observation likelihoods (SB-mirror of Bruno
# `joint_pk_qt_tgi_brm1`, `web-pkpd/src/brm_joint_tgi.jl` @ bruno mirror ref
# `kb-impl/Bruno-arv393-tgi` 3af2284; design doc `dev-docs/arv393-tgi-model.md`).
# Aggregate mode only; logdensity only.
#
# ORACLE ROUTE (documented per the slice contract — cheapest faithful route):
#   * BRM corpus rows for these cells: NONE EXIST (no `tgi_*` symbol in the
#     BayesianRegressionModels.jl / StanBlocks.jl shared checkouts; the BRM-side
#     RK emitter has no tgi/auc/category emission yet — parent-recorded gap).
#   * BridgeStan eval of the SB program: disproportionate for standalone cells
#     (no bruno checkout on this host, mirror-only; the SB side itself gates
#     compilation behind TGI_COMPILE=1 as expensive/opt-in) — and unnecessary:
#     bruno's own TGI_COMPILE guard PROVES the compiled Stan program's category
#     loglik against a Distributions.jl reference at rtol=1e-8
#     (`web-pkpd/test/joint_pk_qt_tgi.jl`, "compiled: category likelihood…").
#   * CHOSEN: port-verified Stan math. The `@deffun` bodies are ported
#     line-by-line into `src/tgi.jl` and verified three independent ways:
#     (1) fresh Distributions.jl-loop hand oracles transcribed from the model
#     doc + Stan source (different primitives: `cdf` vs `erfc`-log);
#     (2) bruno's guard reference `reference_category_logprob` reproduced
#     verbatim (provenance cited — the SB-validated leg);
#     (3) Stan primitives against Distributions.jl / BigFloat.
#     Committed literals below are regression goldens: every one was verified
#     against (1)+(2) before pinning. Gradients close the triangle
#     (Enzyme vs Reactant-compiled vs central finite differences).
#
# No `test/corpus/` cases: corpus pins `@rkppl` surface→plan lowering and this
# slice adds no surface syntax (cell admission + grouped-kernel emission are the
# sibling foundation slice's — `contract.jl`/`generator.jl`, untouched here).
# Goldens live as committed literals + the pinned `tgi_nadir_scan_expr` output.

const _TGI_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

function _tgi_findiff(f, u; h = cbrt(eps(Float64)))
    g = Vector{Float64}(undef, length(u))
    for i in eachindex(u)
        up = Vector{Float64}(u)
        up[i] += h
        dn = Vector{Float64}(u)
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

# --- hand oracles (fresh transcription from the model doc + Stan source) ---
#
# Different primitives from `src/tgi.jl` (`cdf`/`logpdf` vs `erfc`-log): agreement
# verifies the port, not the shared spelling.

function _hand_category_logprob(y, r, ref, c_cr, c_pr, c_pd, sigma, eps)
    pd = (c_pd + ref - r) / sigma
    pr = min((c_pr - r) / sigma, pd)
    cr = min((c_cr - r) / sigma, pr)
    Φ = Normal()
    p = y == 1 ? cdf(Φ, cr) :
        y == 2 ? cdf(Φ, pr) - cdf(Φ, cr) :
        y == 3 ? cdf(Φ, pd) - cdf(Φ, pr) : 1 - cdf(Φ, pd)
    return log((1 - eps) * p + eps / 4)
end

# bruno's `reference_category_logprob` reproduced VERBATIM (provenance:
# `web-pkpd/test/joint_pk_qt_tgi.jl`; bruno's TGI_COMPILE guard proves the
# compiled SB program against this exact formula at rtol=1e-8 — the
# SB-validated leg of the triangle).
function _bruno_reference_category_logprob(y, change, ref, c_cr, c_pr, c_pd, sigma, eps)
    pd = (c_pd + ref - change) / sigma
    pr = min((c_pr - change) / sigma, pd)
    cr = min((c_cr - change) / sigma, pr)
    Φ(z) = cdf(Normal(), z)
    p = y == 1 ? Φ(cr) : y == 2 ? Φ(pr) - Φ(cr) : y == 3 ? Φ(pd) - Φ(pr) : 1 - Φ(pd)
    log((1 - eps) * p + eps / 4)
end

function _hand_response_logprob(y, r, ref, c_pr, c_pd, sigma, eps)
    pr = min(c_pr - r, c_pd + ref - r) / sigma
    Φ = Normal()
    p = y == 1 ? cdf(Φ, pr) : 1 - cdf(Φ, pr)
    return log((1 - eps) * p + eps / 2)
end

function _hand_censored_logprob(y, mu, sigma, lloq_log)
    y <= lloq_log && return logcdf(Normal(mu, sigma), lloq_log)
    return logpdf(Normal(mu, sigma), y)
end

function _hand_ratio_loglinear(t, exposure, g, k)
    return [g * ti - k * ei for (ti, ei) in zip(t, exposure)]
end

function _hand_ratio_resistant(t, exposure, g, k, phi)
    return [g * ti + log((1 - phi) * exp(-k * ei) + phi)
            for (ti, ei) in zip(t, exposure)]
end

function _hand_nadir(r)
    out = Vector{Float64}(undef, length(r))
    current = 0.0
    for i in eachindex(r)
        out[i] = current
        current = min(current, Float64(r[i]))
    end
    return out
end

# Canonical fixtures: tumor-clock times (screening / first dose / wk8 / wk16),
# reference-unit exposures, doc-scale parameters (truth log kg -1.39, log kd
# -0.11, sigma 0.10 from the synthetic-fit record).
_tgi_times() = [-24.0 / 1344.0, 0.0, 1.0, 2.0]
_tgi_exposures() = [0.0, 0.0, 0.9, 1.8]
_tgi_gk() = (exp(-1.39), exp(-0.11))

@testset "TGI options admission" begin
    dflt = tgi_options()
    @test dflt.observation == "ordinal"
    @test dflt.structure == "log_linear"
    @test dflt.thresholds == "lugano_ct"
    @test dflt.measure == "spd"
    @test dflt.misclassification == 0.01
    @test dflt.lloq === nothing
    @test dflt.centered_subjects == false
    for obs in ("continuous", "ordinal", "binary", "none"),
        st in ("log_linear", "resistant_fraction"),
        th in ("lugano_ct", "recist", "estimated"),
        ms in ("spd", "sld")

        o = tgi_options(; observation = obs, structure = st, thresholds = th,
            measure = ms)
        @test (o.observation, o.structure, o.thresholds, o.measure) ==
            (obs, st, th, ms)
    end
    @test_throws "expected one of continuous, ordinal, binary, none" tgi_options(;
        observation = "counts")
    @test_throws "expected one of log_linear, resistant_fraction" tgi_options(;
        structure = "gompertz")
    @test_throws "expected one of lugano_ct, recist, estimated" tgi_options(;
        thresholds = "irecist")
    @test_throws "expected one of spd, sld" tgi_options(; measure = "volume")
    @test_throws "must lie in [0, 0.5)" tgi_options(; misclassification = 0.5)
    @test_throws "must lie in [0, 0.5)" tgi_options(; misclassification = -0.1)
    @test_throws "must lie in [0, 0.5)" tgi_options(; misclassification = Inf)
    @test tgi_options(; misclassification = 0.0).misclassification == 0.0
    @test_throws "must be positive and finite" tgi_options(; lloq = 0.0)
    @test_throws "must be positive and finite" tgi_options(; lloq = -5.0)
    @test tgi_options(; lloq = 5).lloq == 5.0
    @test tgi_options(; centered_subjects = true).centered_subjects == true
    @test tgi_options(; centered_subjects = false).centered_subjects == false
end

@testset "TGI threshold math" begin
    @test TGI_TIME_SCALE_H == 1344.0
    @test TGI_LOG_PR == log(0.5)
    @test TGI_LOG_PD == log(1.5)
    @test TGI_RECIST_LOG_PR == log(0.7)
    @test TGI_RECIST_LOG_PD == log(1.2)
    @test tgi_measure_dim("spd") == 2
    @test tgi_measure_dim("sld") == 1
    @test_throws "expected one of spd, sld" tgi_measure_dim("volume")
    # Default lugano_ct × spd scales by exactly 1.0 (byte-identical constants).
    dflt = tgi_options()
    @test tgi_threshold_scale(dflt) == 1.0
    c_pr, c_pd = tgi_fixed_cutpoints(dflt)
    @test c_pr == log(0.5)
    @test c_pd == log(1.5)
    # lugano_ct × sld halves the log thresholds.
    sld = tgi_options(; measure = "sld")
    @test tgi_threshold_scale(sld) == 0.5
    @test tgi_fixed_cutpoints(sld) == (0.5 * log(0.5), 0.5 * log(1.5))
    # recist × sld emits its constants directly; recist × spd doubles them.
    rsld = tgi_options(; thresholds = "recist", measure = "sld")
    @test tgi_fixed_cutpoints(rsld) == (log(0.7), log(1.2))
    rspd = tgi_options(; thresholds = "recist", measure = "spd")
    @test tgi_fixed_cutpoints(rspd) == (2.0 * log(0.7), 2.0 * log(1.2))
    # estimated: free cutpoints, baseline reference (no nadir).
    est = tgi_options(; thresholds = "estimated")
    @test_throws "no fixed cutpoints" tgi_fixed_cutpoints(est)
    @test tgi_estimated_cutpoints(-2.0, 0.5, 0.7) == (-2.0, -1.5, -0.8)
    @test tgi_uses_nadir(dflt) == true
    @test tgi_uses_nadir(rsld) == true
    @test tgi_uses_nadir(est) == false
end

@testset "TGI Stan primitives" begin
    # normal_lcdf vs Distributions across the clamped range and beyond.
    for x in [-35.0, -30.0, -21.2, -8.26, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, 21.2, 30.0, 35.0]
        @test tgi_normal_lcdf(x) ≈ logcdf(Normal(), x) rtol = 1e-12 atol = 1e-12
    end
    # log_diff_exp vs BigFloat truth.
    for (a, b) in [(-0.5, -2.0), (0.0, 0.0), (-100.0, -101.5), (-448.0, -449.0)]
        want = Float64(log(exp(big(a)) - exp(big(b))))
        @test tgi_log_diff_exp(a, b) ≈ want rtol = 1e-13
    end
    # report mixing vs the manual mixture, incl. -Inf lp and eps = 0.
    for (lp, eps, k) in [(-1.5, 0.01, 4.0), (-Inf, 0.01, 4.0),
            (-0.3, 0.0, 2.0), (-Inf, 0.0, 2.0), (-3.0, 0.2, 4.0)]
        want = log((1 - eps) * exp(lp) + eps / k)
        @test tgi_report_logprob(lp, eps, k) ≈ want rtol = 1e-14 atol = 1e-15
    end
    @test tgi_inv_logit(0.0) == 0.5
    @test tgi_inv_logit(800.0) == 1.0
    @test tgi_inv_logit(-800.0) == 0.0
    @test tgi_inv_logit(-3.0) ≈ 1 / (1 + exp(3.0)) rtol = 1e-15
end

@testset "TGI interval logprob" begin
    # Interior intervals vs the cdf difference; empty/inverted are -Inf.
    for (lo, hi) in [(-1.0, 1.0), (0.5, 2.0), (-3.0, -0.2), (-30.0, 30.0)]
        want = log(cdf(Normal(), hi) - cdf(Normal(), lo))
        # atol: the (-30, 30) oracle saturates to exactly 0.0 while the
        # log-space port correctly returns -4.9e-198.
        @test tgi_interval_logprob(lo, hi) ≈ want rtol = 1e-12 atol = 1e-15
    end
    @test tgi_interval_logprob(1.0, 1.0) == -Inf
    @test tgi_interval_logprob(2.0, 1.0) == -Inf
    # Clamping: beyond ±30 the probability is below 1e-197 either way.
    @test tgi_interval_logprob(-40.0, -31.0) ≈ tgi_interval_logprob(-30.0, -30.0)
    @test tgi_interval_logprob(-40.0, -31.0) == -Inf
    @test tgi_interval_logprob(31.0, 40.0) == -Inf
end

@testset "TGI latent structures vs hand oracle" begin
    t = _tgi_times()
    e = _tgi_exposures()
    g, k = _tgi_gk()
    rl = tgi_ratio_loglinear(t, e, g, k)
    @test rl ≈ _hand_ratio_loglinear(t, e, g, k) rtol = 1e-15
    @test rl ≈ [-0.004447773296994075, 0.0, -0.5571754171352072,
        -1.1143508342704145] rtol = 1e-14
    # A pre-dose scan has exposure 0: untreated growth, log ratio < 0.
    @test rl[1] ≈ g * t[1] rtol = 1e-15
    @test rl[1] < 0
    phi = tgi_inv_logit(-3.0)
    rr = tgi_ratio_resistant(t, e, g, k, phi)
    @test rr ≈ _hand_ratio_resistant(t, e, g, k, phi) rtol = 1e-13
    @test rr ≈ [-0.004447773296994075, 0.0, -0.5000541791402047,
        -0.9400353576202422] rtol = 1e-14
    # Resistant fraction limits: phi → 0 recovers log-linear, phi → 1 no kill.
    @test tgi_ratio_resistant(t, e, g, k, 0.0) ≈ rl rtol = 1e-15
    @test tgi_ratio_resistant(t, e, g, k, 1.0) ≈ g .* t rtol = 1e-15
    # Scalar/vector agreement.
    @test tgi_ratio_loglinear(t[3], e[3], g, k) == rl[3]
    @test tgi_ratio_resistant(t[3], e[3], g, k, phi) == rr[3]
    @test tgi_log_survival(k * e[3], phi) ≈
        log((1 - phi) * exp(-k * e[3]) + phi) rtol = 1e-14
end

@testset "TGI running nadir" begin
    @test tgi_running_nadir([0.0, -0.65, -1.21, -1.77]) == [0.0, 0.0, -0.65, -1.21]
    @test tgi_running_nadir([0.0, 0.4, 0.81, 1.2]) == [0.0, 0.0, 0.0, 0.0]
    @test tgi_running_nadir([0.0, -2.0, -1.9]) == [0.0, 0.0, -2.0]
    @test tgi_running_nadir(Float64[]) == Float64[]
    r = _hand_ratio_loglinear(_tgi_times(), _tgi_exposures(), _tgi_gk()...)
    @test tgi_running_nadir(r) == _hand_nadir(r)
    # The scan formulation emits the exact expected statement (golden pin).
    got = tgi_nadir_scan_expr(:tgi_change, :tgi_ref)
    want = Base.remove_linenums!(:(tgi_ref = scan(tgi_change; init = 0.0) do carry, x
        (min(carry, x), carry)
    end))
    @test got == want
end

@testset "TGI category vs oracles" begin
    sigma = sqrt(2.0) * 0.10
    c_cr, c_pr, c_pd = -2.3, log(0.5), log(1.5)
    # Three shapes: monotone shrinkage (deepening nadir), monotone growth
    # (PD from baseline), and deep-nadir recovery (SD interval exactly empty).
    shapes = (
        ([0.0, -0.65, -1.21, -1.77], [3, 3, 2, 2], -0.5091275105866673),
        ([0.0, 0.4, 0.81, 1.2], [3, 3, 3, 4], -6.068131269000123),
        ([0.0, -2.0, -1.9], [3, 2, 3], -6.025639178215117),
    )
    for (r, y, golden) in shapes
        ref = tgi_running_nadir(r)
        for i in eachindex(y)
            hand = _hand_category_logprob(y[i], r[i], ref[i], c_cr, c_pr,
                c_pd, sigma, 0.01)
            bruno = _bruno_reference_category_logprob(y[i], r[i], ref[i],
                c_cr, c_pr, c_pd, sigma, 0.01)
            impl = tgi_category_lpmf(y[i], r[i], ref[i], c_cr, c_pr, c_pd,
                sigma, 0.01)
            @test impl ≈ hand rtol = 1e-12 atol = 1e-13
            @test impl ≈ bruno rtol = 1e-12 atol = 1e-13
        end
        @test tgi_category_lpmfs(y, r, ref, c_cr, c_pr, c_pd, sigma, 0.01) ≈
            [tgi_category_lpmf(y[i], r[i], ref[i], c_cr, c_pr, c_pd, sigma,
                0.01) for i in eachindex(y)] rtol = 1e-15
        @test tgi_category_lpmf(y, r, ref, c_cr, c_pr, c_pd, sigma, 0.01) ≈
            sum(_hand_category_logprob(y[i], r[i], ref[i], c_cr, c_pr, c_pd,
                sigma, 0.01) for i in eachindex(y)) rtol = 1e-12
        @test tgi_category_lpmf(y, r, ref, c_cr, c_pr, c_pd, sigma, 0.01) ≈
            golden rtol = 1e-13
    end
    # The emptied SD interval stays finite through the misclassification floor.
    r = [0.0, -2.0, -1.9]
    ref = tgi_running_nadir(r)
    @test ref[3] == -2.0
    @test isfinite(tgi_category_lpmf(3, r[3], ref[3], c_cr, c_pr, c_pd,
        sigma, 0.01))
    @test tgi_category_lpmf(3, r[3], ref[3], c_cr, c_pr, c_pd, sigma, 0.01) ≈
        log(0.01 / 4) rtol = 1e-12
    # recist + estimated cutpoints take the same path.
    @test tgi_category_lpmf(2, -0.5, 0.0, -2.0, log(0.7), log(1.2), sigma,
        0.01) ≈ _hand_category_logprob(2, -0.5, 0.0, -2.0, log(0.7),
        log(1.2), sigma, 0.01) rtol = 1e-12
    ec = tgi_estimated_cutpoints(-2.0, 0.5, 0.7)
    @test tgi_category_lpmf(4, 0.3, 0.0, ec..., sigma, 0.01) ≈
        _hand_category_logprob(4, 0.3, 0.0, ec..., sigma, 0.01) rtol = 1e-12
    # eps = 0 is the pure size rule.
    @test tgi_category_lpmf(2, -1.21, -0.65, c_cr, c_pr, c_pd, sigma, 0.0) ≈
        log(cdf(Normal(), min((c_pr + 1.21) / sigma,
            (c_pd - 0.65 + 1.21) / sigma)) -
            cdf(Normal(), min((c_cr + 1.21) / sigma,
            min((c_pr + 1.21) / sigma, (c_pd - 0.65 + 1.21) / sigma)))) rtol = 1e-11
end

@testset "TGI response vs hand oracle" begin
    sigma = sqrt(2.0) * 0.10
    c_pr, c_pd = log(0.5), log(1.5)
    shapes = (
        ([0.0, -0.65, -1.21, -1.77], [0, 0, 1, 1], -0.49537341029031484),
        ([0.0, 0.4, 0.81, 1.2], [0, 0, 0, 0], -0.020050640947310107),
        ([0.0, -2.0, -1.9], [0, 1, 1], -0.030466897257920986),
    )
    for (r, y, golden) in shapes
        ref = tgi_running_nadir(r)
        for i in eachindex(y)
            @test tgi_response_lpmf(y[i], r[i], ref[i], c_pr, c_pd, sigma,
                0.01) ≈ _hand_response_logprob(y[i], r[i], ref[i], c_pr,
                c_pd, sigma, 0.01) rtol = 1e-12 atol = 1e-13
        end
        @test tgi_response_lpmf(y, r, ref, c_pr, c_pd, sigma, 0.01) ≈
            sum(_hand_response_logprob(y[i], r[i], ref[i], c_pr, c_pd,
                sigma, 0.01) for i in eachindex(y)) rtol = 1e-12
        @test tgi_response_lpmf(y, r, ref, c_pr, c_pd, sigma, 0.01) ≈
            golden rtol = 1e-13
    end
    # A responder below a deep nadir: PD precedence keeps the PR interval.
    r = [0.0, -2.0, -1.9]
    ref = tgi_running_nadir(r)
    @test tgi_response_lpmf(1, r[3], ref[3], c_pr, c_pd, sigma, 0.01) ≈
        _hand_response_logprob(1, r[3], ref[3], c_pr, c_pd, sigma, 0.01) rtol = 1e-12
end

@testset "TGI censored vs hand oracle" begin
    lloq_log = log(5.0)
    # Interior values take the density; at/below-bound values take the CDF.
    # Model means near the bound for censored rows (a censored fit's regime).
    mu = [7.5, 1.8, 7.2, 2.5]
    y = [7.6, lloq_log, 7.1, 0.5]
    for i in eachindex(y)
        @test tgi_censored_lpdf(y[i], mu[i], 0.13, lloq_log) ≈
            _hand_censored_logprob(y[i], mu[i], 0.13, lloq_log) rtol = 1e-13
    end
    @test tgi_censored_lpdf(y, mu, 0.13, lloq_log) ≈
        sum(_hand_censored_logprob(y[i], mu[i], 0.13, lloq_log)
            for i in eachindex(y)) rtol = 1e-13
    @test tgi_censored_lpdf(y, mu, 0.13, lloq_log) ≈ -27.31746484615032 rtol = 1e-13
    # A disappeared lesion (recorded 0, clamped to the bound by data prep).
    @test tgi_censored_lpdf(lloq_log, 1.8, 0.13, lloq_log) ≈
        logcdf(Normal(1.8, 0.13), lloq_log) rtol = 1e-13
    # Extreme-tail caveat (SHARED with the thin layer's censored Gaussian
    # `_gaussian_cell`, bit-for-bit the same `log(0.5*erfc(...))` form — the
    # accepted slice-1 probit caveat): past |z| ≈ 38.6 `erfc` underflows and
    # the CDF arm reads -Inf where Stan stays finite. Pinned, not fixed.
    @test tgi_censored_lpdf(lloq_log, 7.5, 0.13, lloq_log) == -Inf
end

@testset "TGI cell vocabulary admission" begin
    @test TGI_CELL_FUNCTIONS == (
        :tgi_ratio_loglinear, :tgi_ratio_resistant, :tgi_log_survival,
        :tgi_running_nadir, :tgi_interval_logprob, :tgi_report_logprob,
        :tgi_category_lpmf, :tgi_category_lpmfs,
        :tgi_response_lpmf, :tgi_response_lpmfs,
        :tgi_censored_lpdf, :tgi_censored_lpdfs,
        :tgi_normal_lcdf, :tgi_log_diff_exp, :tgi_inv_logit,
    )
    for f in TGI_CELL_FUNCTIONS
        @test isdefined(ReactiveKernelsPPL, f)
    end
end

# --- standalone @kernel parity (Enzyme + Reactant + findiff) ---
#
# Each kernel below is the grouped-kernel emission shape for one observation
# model: actives packed in one vector `u`, responses/latents bound, the
# likelihood a whole-vector `tgi_*` recipe. The nadir rides a BOUND host
# vector here (isolating likelihood gradients); the `scan` formulation is
# proved separately below (native/Enzyme/Reactant/findiff — the scan fix
# 1ad54a5 unblocked the Reactant axis). Oracle closures use
# only the `_hand_*` oracles.

_tgi_kernel_data() = (;
    t = [0.0, 1.0, 2.0],
    exposure = [0.0, 0.9, 1.8],
    t0 = -48.0 / 1344.0,
    c_pr = log(0.5),
    c_pd = log(1.5),
    eps = 0.01,
)

function _tgi_oracle_change(d, g, k)
    r = _hand_ratio_loglinear(d.t, d.exposure, g, k)
    return r .- g .* d.t0
end

@testset "TGI category standalone kernel parity" begin
    d = _tgi_kernel_data()
    y = [3, 2, 2]
    u0 = [exp(-1.39), exp(-0.11), 0.10, -2.3]
    @kernel tgi_cat_standalone(u::Vector{Float64}, t, exposure, y, ref, t0, c_pr, c_pd, eps) = begin
        g::Float64 = u[1]
        k::Float64 = u[2]
        sigma::Float64 = u[3]
        c_cr::Float64 = u[4]
        r = tgi_ratio_loglinear(t, exposure, g, k)
        change = r .- g .* t0
        total::Float64 = tgi_category_lpmf(y, change, ref, c_cr, c_pr, c_pd,
            sqrt(2.0) * sigma, eps)
        return total
    end
    change0 = _tgi_oracle_change(d, u0[1], u0[2])
    ref0 = _hand_nadir(change0)
    bound = (; d.t, d.exposure, y, ref = ref0, d.t0, d.c_pr, d.c_pd, d.eps)
    want = sum(_hand_category_logprob(y[i], change0[i], ref0[i], u0[4],
        d.c_pr, d.c_pd, sqrt(2.0) * u0[3], d.eps) for i in eachindex(y))
    kern = prepare(tgi_cat_standalone;
        have = (:u, :t, :exposure, :y, :ref, :t0, :c_pr, :c_pd, :eps),
        want = :total, bound = bound)
    @test kern(u0) ≈ want rtol = 1e-12
    prep = prepare_ad(tgi_cat_standalone, _TGI_BACKEND, u0; active = :u,
        want = :total, bound = bound)
    g = Vector{Float64}(undef, 4)
    val, _ = ReactiveKernels.ad_value_and_gradient!(prep, g, u0)
    @test val ≈ want rtol = 1e-12
    @test g ≈ _tgi_findiff(kern, u0) rtol = 1e-5 atol = 1e-7
    @test val ≈ -1.8914331391421038 rtol = 1e-12
    @test g ≈ [-11.21261441791705, 9.742906753893894, 15.180773947281388,
        -9.06352913329849e-16] rtol = 1e-8 atol = 1e-10
    traced = Reactant.to_rarray(u0)
    compiled = compile_ad_value_and_gradient(prep, traced)
    rval, rgrad = compiled(traced)
    @test Float64(rval) ≈ want rtol = 1e-11
    @test Array(rgrad) ≈ g rtol = 1e-5 atol = 1e-7
end

@testset "TGI resistant category standalone kernel parity" begin
    d = _tgi_kernel_data()
    y = [3, 3, 2]
    u0 = [exp(-1.39), exp(-0.11), 0.10, -2.3, -3.0]
    @kernel tgi_cat_res_standalone(u::Vector{Float64}, t, exposure, y, ref, t0, c_pr, c_pd, eps) = begin
        g::Float64 = u[1]
        k::Float64 = u[2]
        sigma::Float64 = u[3]
        c_cr::Float64 = u[4]
        phi::Float64 = tgi_inv_logit(u[5])
        r = tgi_ratio_resistant(t, exposure, g, k, phi)
        change = r .- g .* t0
        total::Float64 = tgi_category_lpmf(y, change, ref, c_cr, c_pr, c_pd,
            sqrt(2.0) * sigma, eps)
        return total
    end
    r0 = _hand_ratio_resistant(d.t, d.exposure, u0[1], u0[2],
        1 / (1 + exp(3.0)))
    change0 = r0 .- u0[1] .* d.t0
    ref0 = _hand_nadir(change0)
    bound = (; d.t, d.exposure, y, ref = ref0, d.t0, d.c_pr, d.c_pd, d.eps)
    want = sum(_hand_category_logprob(y[i], change0[i], ref0[i], u0[4],
        d.c_pr, d.c_pd, sqrt(2.0) * u0[3], d.eps) for i in eachindex(y))
    kern = prepare(tgi_cat_res_standalone;
        have = (:u, :t, :exposure, :y, :ref, :t0, :c_pr, :c_pd, :eps),
        want = :total, bound = bound)
    @test kern(u0) ≈ want rtol = 1e-12
    prep = prepare_ad(tgi_cat_res_standalone, _TGI_BACKEND, u0; active = :u,
        want = :total, bound = bound)
    g = Vector{Float64}(undef, 5)
    val, _ = ReactiveKernels.ad_value_and_gradient!(prep, g, u0)
    @test val ≈ want rtol = 1e-12
    @test g ≈ _tgi_findiff(kern, u0) rtol = 1e-5 atol = 1e-7
    @test val ≈ -0.15177523861613115 rtol = 1e-12
    @test g ≈ [-0.3213871348550686, 0.14150102331502623, -4.142239970572166,
        -1.3350617560764113e-20, -0.05097945903218263] rtol = 1e-8 atol = 1e-10
    traced = Reactant.to_rarray(u0)
    compiled = compile_ad_value_and_gradient(prep, traced)
    rval, rgrad = compiled(traced)
    @test Float64(rval) ≈ want rtol = 1e-11
    @test Array(rgrad) ≈ g rtol = 1e-5 atol = 1e-7
end

@testset "TGI response standalone kernel parity" begin
    d = _tgi_kernel_data()
    y = [0, 1, 1]
    u0 = [exp(-1.39), exp(-0.11), 0.10]
    @kernel tgi_resp_standalone(u::Vector{Float64}, t, exposure, y, ref, t0, c_pr, c_pd, eps) = begin
        g::Float64 = u[1]
        k::Float64 = u[2]
        sigma::Float64 = u[3]
        r = tgi_ratio_loglinear(t, exposure, g, k)
        change = r .- g .* t0
        total::Float64 = tgi_response_lpmf(y, change, ref, c_pr, c_pd,
            sqrt(2.0) * sigma, eps)
        return total
    end
    change0 = _tgi_oracle_change(d, u0[1], u0[2])
    ref0 = _hand_nadir(change0)
    bound = (; d.t, d.exposure, y, ref = ref0, d.t0, d.c_pr, d.c_pd, d.eps)
    want = sum(_hand_response_logprob(y[i], change0[i], ref0[i], d.c_pr,
        d.c_pd, sqrt(2.0) * u0[3], d.eps) for i in eachindex(y))
    kern = prepare(tgi_resp_standalone;
        have = (:u, :t, :exposure, :y, :ref, :t0, :c_pr, :c_pd, :eps),
        want = :total, bound = bound)
    @test kern(u0) ≈ want rtol = 1e-12
    prep = prepare_ad(tgi_resp_standalone, _TGI_BACKEND, u0; active = :u,
        want = :total, bound = bound)
    g = Vector{Float64}(undef, 3)
    val, _ = ReactiveKernels.ad_value_and_gradient!(prep, g, u0)
    @test val ≈ want rtol = 1e-12
    @test g ≈ _tgi_findiff(kern, u0) rtol = 1e-5 atol = 1e-7
    @test val ≈ -1.8677542077277782 rtol = 1e-12
    @test g ≈ [-11.0324322656463, 9.588048371182285, 15.15160271843938] rtol = 1e-8 atol = 1e-10
    traced = Reactant.to_rarray(u0)
    compiled = compile_ad_value_and_gradient(prep, traced)
    rval, rgrad = compiled(traced)
    @test Float64(rval) ≈ want rtol = 1e-11
    @test Array(rgrad) ≈ g rtol = 1e-5 atol = 1e-7
end

@testset "TGI censored standalone kernel parity" begin
    d = _tgi_kernel_data()
    # Shrinking tumor near the detection limit (interior / bound / below).
    ylog = [2.1, log(5.0), 0.5]
    lloq_log = log(5.0)
    u0 = [exp(-1.39), exp(-0.11), 2.0, 0.13]
    @kernel tgi_cens_standalone(u::Vector{Float64}, t, exposure, ylog, lloq_log) = begin
        g::Float64 = u[1]
        k::Float64 = u[2]
        b0::Float64 = u[3]
        sigma::Float64 = u[4]
        r = tgi_ratio_loglinear(t, exposure, g, k)
        mu = b0 .+ r
        total::Float64 = tgi_censored_lpdf(ylog, mu, sigma, lloq_log)
        return total
    end
    r0 = _hand_ratio_loglinear(d.t, d.exposure, u0[1], u0[2])
    mu0 = u0[3] .+ r0
    bound = (; d.t, d.exposure, ylog, lloq_log)
    want = sum(_hand_censored_logprob(ylog[i], mu0[i], u0[4], lloq_log)
        for i in eachindex(ylog))
    kern = prepare(tgi_cens_standalone;
        have = (:u, :t, :exposure, :ylog, :lloq_log), want = :total,
        bound = bound)
    @test kern(u0) ≈ want rtol = 1e-12
    prep = prepare_ad(tgi_cens_standalone, _TGI_BACKEND, u0; active = :u,
        want = :total, bound = bound)
    g = Vector{Float64}(undef, 4)
    val, _ = ReactiveKernels.ad_value_and_gradient!(prep, g, u0)
    @test val ≈ want rtol = 1e-12
    @test g ≈ _tgi_findiff(kern, u0) rtol = 1e-5 atol = 1e-7
    @test val ≈ 0.720081216340853 rtol = 1e-12
    @test g ≈ [-1.499788819176582, 1.3498099372589238, 4.417371514020152,
        -5.062838885055454] rtol = 1e-8 atol = 1e-10
    traced = Reactant.to_rarray(u0)
    compiled = compile_ad_value_and_gradient(prep, traced)
    rval, rgrad = compiled(traced)
    @test Float64(rval) ≈ want rtol = 1e-11
    @test Array(rgrad) ≈ g rtol = 1e-5 atol = 1e-7
end

@testset "TGI nadir scan parity" begin
    d = _tgi_kernel_data()
    u0 = [exp(-1.39), exp(-0.11)]
    # The body below is `tgi_nadir_scan_expr(:change, :nadir)` inlined (the
    # builder output is pinned above); it must equal the host loop.
    @kernel tgi_nadir_standalone(u::Vector{Float64}, t, exposure, t0) = begin
        g::Float64 = u[1]
        k::Float64 = u[2]
        r = tgi_ratio_loglinear(t, exposure, g, k)
        change = r .- g .* t0
        nadir = scan(change; init = 0.0) do carry, x
            (min(carry, x), carry)
        end
        total::Float64 = sum(nadir)
        return total
    end
    change0 = _tgi_oracle_change(d, u0[1], u0[2])
    ref0 = _hand_nadir(change0)
    bound = (; d.t, d.exposure, d.t0)
    @test ref0 ≈ [0.0, 0.0, -0.5482798705412191] rtol = 1e-14
    veckern = prepare(tgi_nadir_standalone;
        have = (:u, :t, :exposure, :t0), want = :nadir, bound = bound)
    @test veckern(u0) ≈ ref0 rtol = 1e-14
    kern = prepare(tgi_nadir_standalone;
        have = (:u, :t, :exposure, :t0), want = :total, bound = bound)
    @test kern(u0) ≈ sum(ref0) rtol = 1e-14
    prep = prepare_ad(tgi_nadir_standalone, _TGI_BACKEND, u0; active = :u,
        want = :total, bound = bound)
    g = Vector{Float64}(undef, 2)
    val, _ = ReactiveKernels.ad_value_and_gradient!(prep, g, u0)
    @test val ≈ sum(ref0) rtol = 1e-14
    @test g ≈ _tgi_findiff(kern, u0) rtol = 1e-5 atol = 1e-7
    @test val ≈ -0.5482798705412191 rtol = 1e-14
    @test g ≈ [1.0357142857142858, -0.9] rtol = 1e-8 atol = 1e-10
    traced = Reactant.to_rarray(u0)
    compiled = compile_ad_value_and_gradient(prep, traced)
    rval, rgrad = compiled(traced)
    @test Float64(rval) ≈ sum(ref0) rtol = 1e-11
    @test Array(rgrad) ≈ g rtol = 1e-5 atol = 1e-7
end
