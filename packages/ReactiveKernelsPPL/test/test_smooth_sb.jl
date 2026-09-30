# Smooth family (matrix-b): exact GP (gp_regr / gp_pois_regr), HSGP
# (accel_gp / brm_hsgp) and spline (accel_splines) programs vs the pair
# partner's BridgeStan numbers, plus the stated-hyper-prior surface the
# HSGP/spline items need: `hsgp_basis(...; length_scale=, sd=)` and
# `spline_basis(...; sd=)` (SB `length_scale(:, hsgp(x)) ~ ...` /
# `sd(:, hsgp(x)) ~ ...` / `sd(mu, s(x)) ~ ...`), and HSGP-only
# predictors (SB `0 + hsgp(x)`, location and `exp.` scale).
#
# Also Bordet builders 2-4 (pin 72d4bfc9: pooled HSGP, shape-pooled
# correlated slope, censored Student-t nu=4) and the bruno HSGP/GP
# components (pkpd_models.jl @ 0090db25: hsgp, clamped_hsgp,
# gp_effectiveness), which add the fixed `domain=` and the bounding
# `length_scale = Uniform(lo, hi)` (SB prior-bound intersection).
#
# SB numbers: briefs 2026-09-27T22-26-40-536-ftkxju (numbers) and
# 2026-09-27T22-26-41-702-k7ikuw (verdict: hierarchical_gp /
# kronecker_gp NO-COUNTERPART) on BayesianRegressionModels:rk:kernel:matrix-b
# (BRM 88c5621, StanBlocks 342436de, BridgeStan 2.9.0; full posterior,
# propto=false, Jacobian included). Formulas + data from the partner's
# run scripts (matrix-b-smooth.jl, -reruns2.jl, -numbers5.jl).
#
# Every leg compares DIRECTLY at the same u: RK's exact GP
# (`gp_chol_latent(gp_exp_quad_cov(x, sigma, rho, 1e-9), z)`, non-centered)
# is SB's `gp(x)` emission; the HSGP basis/spectral weights are SB
# `_sb_hsgp` verbatim; the spline blocks are SB's (term-splines-stan). A
# stated length-scale prior drops the validity floor (BRM
# `_brm_hsgp_declared_rho_lower`), which the brm_hsgp/accel_gp values pin
# (SB constrains rho = exp(u) there). The banked probes are uniform
# (zeros and 0.3^n), so the n = coordinate COUNT and the layout-name
# assertions carry the mapping; the HSGP/spline math itself is pinned
# against independent oracles in test_hsgp.jl / test_spline.jl.
# (`_check_gradient` / `_GEN_BACKEND` come from test_generator.jl;
# `_irt_intercept_integral` — the exact collapsed-totals bridge — from
# test_irt.jl, both included earlier by runtests.jl.)
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using SpecialFunctions
using Statistics: std
using Test

function _sm_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

_sm_cols(nt) = Dict{Symbol,AbstractVector}(k => collect(v)
    for (k, v) in pairs(nt))
_sm_val(kern, u) = Base.invokelatest(kern, u)

const _SM_GX = [-2.0, -1.2, -0.5, 0.5, 1.2, 2.0]
const _SM_AX = [-2.0, -1.4, -0.8, -0.2, 0.4, 1.0, 1.6, 2.2]
const _SM_SX = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
    2.5, 3.0]

function _sm_mcycle()
    path = joinpath(@__DIR__, "..", "..", "..", "examples", "data",
        "mcycle.csv")
    rows = split.(readlines(path)[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    lo, hi = extrema(times)
    x = @. -1 + 2 * (times - lo) / (hi - lo)
    return (; x, x2 = copy(x), y = accel ./ std(accel))
end

const _SM_GPREGR = quote
    b0 ~ Normal(0, 1)
    rho ~ Gamma(25, 4)
    sig ~ Normal(0, 2; lower = 0)
    @plate for i in eachindex(y)
        z[i] ~ Normal(0, 1)
    end
    f = gp_chol_latent(gp_exp_quad_cov(x, sig, rho, 1e-9), z)
    mu = b0 .+ f
    ls ~ Normal(0, 1)
    sigma = exp(ls)
    y .~ Normal.(mu, sigma)
end

const _SM_GPPOIS = quote
    b0 ~ Normal(0, 1)
    rho ~ Gamma(25, 4)
    sig ~ Normal(0, 2; lower = 0)
    @plate for i in eachindex(counts)
        z[i] ~ Normal(0, 1)
    end
    f = gp_chol_latent(gp_exp_quad_cov(x, sig, rho, 1e-9), z)
    mu = b0 .+ f
    counts .~ Poisson.(exp.(mu))
end

const _SM_ACCELGP = quote
    b0 ~ StudentT(3, -13, 36)
    mu = b0 .+ hsgp(:h_x)
    hsgp_basis(:h_x, x; k = 8,
        length_scale = InverseGamma(1.124909, 0.0177),
        sd = StudentT(3, 0, 36))
    s0 ~ StudentT(3, 0, 10)
    lsig = s0 .+ hsgp(:h_x2)
    hsgp_basis(:h_x2, x2; k = 8,
        length_scale = InverseGamma(1.124909, 0.0177),
        sd = StudentT(3, 0, 36))
    y .~ Normal.(mu, exp.(lsig))
end

const _SM_BRMHSGP = quote
    mu = hsgp(:h_x)
    hsgp_basis(:h_x, x; k = 20, length_scale = LogNormal(0, 4),
        sd = LogNormal(0, 4))
    lsig = hsgp(:h_x2)
    hsgp_basis(:h_x2, x2; k = 20, length_scale = LogNormal(0, 4),
        sd = LogNormal(0, 4))
    y .~ Normal.(mu, exp.(lsig))
end

const _SM_ACCELSPL = quote
    spline_basis(:s_x, x; sd = StudentT(3, 0, 36))
    spline_basis(:s_x2, x2; sd = StudentT(3, 0, 10))
    b0 ~ StudentT(3, -13, 36)
    s0 ~ StudentT(3, 0, 10)
    mu = b0 .+ spline(:s_x)
    lsig = s0 .+ spline(:s_x2)
    y .~ Normal.(mu, exp.(lsig))
end

const _SM_DATA = (
    gpregr = (; x = _SM_GX, y = [0.5, -0.3, 0.8, -0.1, 0.4, -0.6]),
    gppois = (; x = _SM_GX, counts = [3, 1, 6, 2, 1, 4]),
    accelgp = (; x = _SM_AX, x2 = _SM_AX,
        y = [-1.2, -0.5, 0.3, 1.1, 0.8, -0.2, -0.9, -1.4]),
    accelspl = (; x = _SM_SX, x2 = _SM_SX,
        y = [-1.4, -1.2, -0.5, 0.3, 1.1, 0.8, -0.2, -0.9, -1.4, -0.7,
            0.2, 0.9]),
)

@testset "hyper-prior surface" begin
    # Stated HSGP priors land on the basis IR; a stated length scale
    # drops the validity floor (plain `exp`), the default keeps it.
    plan = lower_rkppl(_SM_ACCELGP, (:y, :x, :x2))
    hb = first(plan.hsgp_bases)
    @test hb.rho_prior == HyperPrior(:inverse_gamma,
        (arg1 = 1.124909, arg2 = 0.0177))
    @test hb.sigma_prior == HyperPrior(:student_t,
        (arg1 = 3.0, arg2 = 0.0, arg3 = 36.0))
    bound = bind_data(plan, _sm_cols(_SM_DATA.accelgp))
    lay = assign_layout(bound)
    tr = Dict(e.name => e.transform for e in lay.entries)
    @test tr[:rho_h_x] === :exp
    dplan = lower_rkppl(quote
        a ~ Normal(0, 1)
        mu = a .+ hsgp(:h_x)
        hsgp_basis(:h_x, x; k = 8)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test only(dplan.hsgp_bases).rho_prior === nothing
    dlay = assign_layout(bind_data(dplan, _sm_cols((; x = _SM_AX,
        y = _SM_DATA.accelgp.y))))
    @test Dict(e.name => e.transform for e in dlay.entries)[:rho_h_x] ===
        :floored
    # Stated spline sd prior rides the sd vector (Stan-kernel support).
    splan = lower_rkppl(_SM_ACCELSPL, (:y, :x, :x2))
    sd = only(v for v in splan.spline_vectors if v.name === :sd_s_x)
    @test sd.family === :student_t
    @test sd.args == (arg1 = 3.0, arg2 = 0.0, arg3 = 36.0)
    @test sd.support_override === :positive_stan
    # Fail-closed: proper halves, unadmitted families, non-literal args,
    # wrong arity, and unknown keywords.
    for bad in (:(HalfNormal(2)), :(Beta(1, 1)), :(Normal(0, s)),
                :(StudentT(3, 0)), :(truncated(Normal(0, 1), 0, Inf)))
        ex = quote
            a ~ Normal(0, 1)
            mu = a .+ hsgp(:h_x)
            hsgp_basis(:h_x, x; k = 8, sd = $bad)
            y .~ Normal.(mu, 1.0)
        end
        @test_throws SurfaceLoweringError lower_rkppl(ex, (:y, :x))
    end
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        mu = a .+ hsgp(:h_x)
        hsgp_basis(:h_x, x; k = 8, lengthscale = LogNormal(0, 1))
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        spline_basis(:s_x, x; sd = HalfCauchy(2))
        a ~ Normal(0, 1)
        mu = a .+ spline(:s_x)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    # A hand-built plan's hyper prior is contract-validated too.
    bad = HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, hb.fits, hb.label,
        hb.cov, hb.period, HyperPrior(:beta, (arg1 = 1.0, arg2 = 1.0)),
        nothing)
    @test_throws ContractValidationError validate_structure(
        StructuralPlan(plan.responses, plan.predictors,
            plan.population_priors, plan.parameters, plan.assignments,
            plan.columns, plan.n_obs; derived = plan.derived,
            hsgp_bases = [bad, plan.hsgp_bases[2]]))
end

const _SM_BORDET = let
    bm = [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
    t = [0.5, 2.0, 8.0, 0.5, 2.0, 8.0, 0.5, 2.0, 8.0, 0.5, 2.0, 8.0]
    d = [0.0, 10.0, 50.0, 0.0, 10.0, 50.0, 0.0, 10.0, 50.0, 0.0, 10.0,
        50.0]
    (; log_time = log.(t), log_dose = log.(d .+ 1.0),
        affectable = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        biomarker = bm, person = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        lloq = [b == 1 ? -3.0 : -2.0 for b in bm],
        uloq = [b == 1 ? 3.0 : 2.5 for b in bm],
        log_obs = [0.1, 0.5, 1.2, -0.2, 0.3, 0.9, 0.0, 0.4, 1.0, -3.0,
            0.2, 2.5])
end

# Bordet `brm_pooled_hsgp` / `brm_shapepooled_hsgp` /
# `brm_pooled_student_t` (BRM default priors throughout — RK's defaults).
_sm_bordet(slope::Bool, lik) = quote
    hsgp_basis(:h_t, log_time; k = 5)
    hsgp_basis(:h_d, log_dose; k = 5)
    r_b ~ varying_effect(biomarker, $(slope ? :([1, affectable]) : :([1])))
    r_p ~ varying_effect(person, [1])
    log_y = a .+ b_aff .* affectable .+ hsgp(:h_t) .+ hsgp(:h_d) .+ r_b .+
        r_p
    ls = c0
    log_obs .~ $lik
end

# Bordet `brm_hierarchical_terms` (builder 5): mean = base + bump * resp
# over per-series correlated parametric curves — `transient` (loc,
# log-slope, magnitude; `bump_math`) and `saturating` (loc, log-slope;
# `sigmoid_math`) — as a composed predictor v3 (data-column leaves +
# `logistic.` maps over varying-slice subs). `series` indexes the 4
# biomarker x person series (Bordet derives it the same way).
const _SM_BORDET_D5 = quote
    sigma ~ Exponential(1)
    r_base ~ varying_effect(series, [1])
    base = b0 .+ r_base
    dt ~ varying_draws(series, [1, 1, 1])
    t1 ~ varying_slice(dt, 1)
    t2 ~ varying_slice(dt, 2)
    t3 ~ varying_slice(dt, 3)
    ds ~ varying_draws(series, [1, 1])
    s1 ~ varying_slice(ds, 1)
    s2 ~ varying_slice(ds, 2)
    tl = t1
    tls = t2
    tm = t3
    dl = s1
    dls = s2
    xi = (log_time .- tl) .* exp.(tls)
    bump = logistic.(xi) .* logistic.(.-xi) .* tm
    resp = logistic.((log_dose .- dl) .* exp.(dls))
    mu = base .+ bump .* resp
    log_obs .~ censored.(Normal.(mu, sigma), lloq, uloq)
end
const _SM_BORDET_D5_DATA = (; log_time = _SM_BORDET.log_time,
    log_dose = _SM_BORDET.log_dose,
    series = _SM_BORDET.biomarker .+ (_SM_BORDET.person .- 1) .* 2,
    lloq = _SM_BORDET.lloq, uloq = _SM_BORDET.uloq,
    log_obs = _SM_BORDET.log_obs)

const _SM_BRUNO_X = [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0]
const _SM_BRUNO_Y = [0.2, -0.3, 0.5, 1.1, 0.7, -0.1, -0.6, 0.3]
# bruno hsgp: minmax axis, fixed L = 1.5 (`domain=(-1.5, 1.5)`), rho
# bounded by the K=10 validity floor (4*1.5/pi)*sqrt(log(100)/99).
_sm_bruno1(ax) = quote
    b0 ~ Normal(0, 5)
    mu = b0 .+ hsgp(:h)
    hsgp_basis(:h, $ax; k = 10, domain = (-1.5, 1.5),
        length_scale = Uniform(0.4119140661125232, 2), sd = LogNormal(0, 1))
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end
const _SM_BRUNO_EFF = quote
    b0 ~ Normal(0, 5)
    mu = b0 .+ hsgp(:h)
    hsgp_basis(:h, xd, xc; k = (8, 8), iso = false,
        domain = ((-1.5, 1.5), (-1.5, 1.5)),
        length_scale = Uniform(0.5163616086861821, 2), sd = Normal(0, 1))
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end
const _SM_BRUNO_EFF_DATA = let
    dose = repeat([0.0, 10.0, 50.0, 200.0], 4)
    conc = repeat([0.0, 100.0, 500.0, 1500.0], inner = 4)
    (; xd = @.(-1 + 2 * dose / 200.0), xc = @.(-1 + 2 * conc / 1500.0),
        y = [0.1, 0.3, 0.8, 1.2, 0.0, 0.2, 0.6, 1.0, -0.1, 0.1, 0.5, 0.9,
            -0.2, 0.0, 0.4, 0.7])
end

@testset "hsgp domain + bounding hyper prior" begin
    plan = lower_rkppl(_sm_bruno1(:x), (:y, :x))
    hb = only(plan.hsgp_bases)
    @test hb.domain == [(-1.5, 1.5)]
    @test hb.rho_prior == HyperPrior(:uniform,
        (arg1 = 0.4119140661125232, arg2 = 2.0))
    bound = bind_data(plan, _sm_cols((; x = _SM_BRUNO_X, y = _SM_BRUNO_Y)))
    @test only(bound.hsgp_bases).fits == [(0.0, 1.5)]
    e = only(en for en in assign_layout(bound).entries
        if en.name === :rho_h)
    @test (e.transform, e.lo, e.hi) == (:interval, 0.4119140661125232, 2.0)
    # Data outside the fixed domain fails at bind; `domain` excludes `c`
    # and periodic; malformed pairs fail at the surface.
    @test_throws ContractValidationError bind_data(plan,
        _sm_cols((; x = [-2.0, 0.0, 1.0], y = [0.1, 0.2, 0.3])))
    for kws in (:((; k = 4, domain = (-1.5, 1.5), c = 1.5)),
                :((; k = 4, domain = (1.5, -1.5))),
                :((; k = 4, domain = ((-1.0, 1.0), (-1.0, 1.0)))),
                :((; k = 4, domain = (-1.5, 1.5), cov = :periodic,
                    period = 2.0)),
                :((; k = 4, length_scale = Uniform(2.0, 1.0))))
        ex = quote
            a ~ Normal(0, 1)
            mu = a .+ hsgp(:h)
            hsgp_basis(:h, x; $(kws.args[1].args...))
            y .~ Normal.(mu, 1.0)
        end
        @test_throws Union{SurfaceLoweringError,ContractValidationError} bind_data(
            lower_rkppl(ex, (:y, :x)),
            _sm_cols((; x = [-1.0, 0.0, 1.0], y = [0.1, 0.2, 0.3])))
    end
end

@testset "smooth SB parity" begin
    # gp_regr: P-gpregr (zeros) / P-gpregr2 (0.3^10); SB names
    # pop_mu_beta_pop.1, gp_x_rho, gp_x_sigma, gp_x_z.1-6,
    # pop_log_sigma_beta_pop.1.
    _, _, kern, lay = _sm_query(_SM_GPREGR, _sm_cols(_SM_DATA.gpregr))
    @test lay.total == 10
    @test _sm_val(kern, zeros(10)) ≈ -105.04931360473961 atol = 1e-10
    @test _sm_val(kern, fill(0.3, 10)) ≈ -100.4985934677729 atol = 1e-10
    # gp_pois_regr: P-gppois / P-gppois2 (n = 9).
    _, _, kern, lay = _sm_query(_SM_GPPOIS, _sm_cols(_SM_DATA.gppois))
    @test lay.total == 9
    @test _sm_val(kern, zeros(9)) ≈ -116.10395556445295 atol = 1e-10
    @test _sm_val(kern, fill(0.3, 9)) ≈ -102.26650368609637 atol = 1e-10
    # accel_gp: P-accelgp / P-accelgp2 (n = 22, k = 8 per basis).
    _, _, kern, lay = _sm_query(_SM_ACCELGP, _sm_cols(_SM_DATA.accelgp))
    @test lay.total == 22
    @test _sm_val(kern, zeros(22)) ≈ -51.40896763818961 atol = 1e-10
    @test _sm_val(kern, fill(0.3, 22)) ≈ -55.66616092236859 atol = 1e-10
    # brm_hsgp: P-brmhsgp / P-brmhsgp2 (real mcycle, k = 20 per basis,
    # coefficient-free `0 + hsgp(x)` location and log-scale).
    _, _, kern, lay = _sm_query(_SM_BRMHSGP, _sm_cols(_sm_mcycle()))
    @test lay.total == 44
    @test _sm_val(kern, zeros(44)) ≈ -252.782708224855 atol = 1e-10
    @test _sm_val(kern, fill(0.3, 44)) ≈ -272.8662184101349 atol = 1e-10
    # accel_splines: P-accelspl12 (0.3^24; per basis 2 fixed + 8 pen + sd).
    _, _, kern, lay = _sm_query(_SM_ACCELSPL, _sm_cols(_SM_DATA.accelspl))
    @test lay.total == 24
    @test _sm_val(kern, fill(0.3, 24)) ≈ -68.72536513207879 atol = 1e-10
end

@testset "bordet builders 2-4 SB parity (pin 72d4bfc9)" begin
    cols = _sm_cols(_SM_BORDET)
    for (label, prog, n, banked) in (
            ("D2 brm_pooled_hsgp",
             _sm_bordet(false, :(censored.(Normal.(log_y, exp.(ls)), lloq,
                uloq))), 23, -48.83720708213348),
            ("D3 brm_shapepooled_hsgp",
             _sm_bordet(true, :(censored.(Normal.(log_y, exp.(ls)), lloq,
                uloq))), 27, -56.14227869781161),
            ("D4 brm_pooled_student_t",
             _sm_bordet(false, :(censored.(StudentT.(4.0, log_y, exp.(ls)),
                lloq, uloq))), 23, -47.95500054200704))
        _, _, kern, lay = _sm_query(prog, cols)
        @test lay.total == n
        @test _sm_val(kern, fill(0.3, n)) ≈ banked atol = 1e-10
    end
end

@testset "bordet builder 5 SB parity (pin 72d4bfc9)" begin
    # SB (D5, 0.3^35) emits the `base ~ 1 + (1 | series)` block in
    # collapsed totals (`total_scale_base_tau`, `total_base.1-4`, the
    # Normal(0, 1) intercept integrated out); every other coordinate —
    # both correlated curve blocks (L/tau/z_flat), sigma — shares RK's
    # parameterization. The exact intercept integral bridges the two.
    _, _, kern, lay = _sm_query(_SM_BORDET_D5, _sm_cols(_SM_BORDET_D5_DATA))
    @test lay.total == 36
    base = Dict{Symbol,Float64}(nm => 0.3 for nm in coordinate_names(lay))
    val, resid = _irt_intercept_integral(kern, lay, base,
        Symbol("base.Intercept"), [Symbol("xi_series.$j") for j in 1:4],
        :log_scale_series, fill(0.3, 4), 1.0)
    @test resid < 1e-10
    @test val ≈ -55.98105373202722 atol = 1e-10
    # The composed mean's data leaves ride the term's columns (the
    # data columns the tree reads elementwise in-graph).
    plan = lower_rkppl(_SM_BORDET_D5, keys(_SM_BORDET_D5_DATA))
    t = only(only(p for p in plan.predictors if p.name === :mu).terms)
    @test t.kind === ComposedTerm
    @test Set(t.columns) == Set([:log_time, :log_dose])
end

@testset "bruno HSGP/GP components SB parity (0090db25)" begin
    xc = [2 * (clamp(v, -0.5, 0.5) + 0.5) / 1.0 - 1.0 for v in _SM_BRUNO_X]
    for (label, prog, data, n, banked) in (
            ("hsgp", _sm_bruno1(:x), (; x = _SM_BRUNO_X, y = _SM_BRUNO_Y),
             14, -26.131600698044252),
            ("clamped_hsgp", _sm_bruno1(:xc), (; xc, y = _SM_BRUNO_Y), 14,
             -26.006302028374282),
            ("gp_effectiveness", _SM_BRUNO_EFF, _SM_BRUNO_EFF_DATA, 69,
             -89.99169920605078))
        _, _, kern, lay = _sm_query(prog, _sm_cols(data))
        @test lay.total == n
        @test _sm_val(kern, fill(0.3, n)) ≈ banked atol = 1e-10
    end
end

const _SM_ITEMS = (
    ("gp_regr", _SM_GPREGR, _SM_DATA.gpregr),
    ("gp_pois_regr", _SM_GPPOIS, _SM_DATA.gppois),
    ("accel_gp", _SM_ACCELGP, _SM_DATA.accelgp),
    ("brm_hsgp", _SM_BRMHSGP, _sm_mcycle()),
    ("accel_splines", _SM_ACCELSPL, _SM_DATA.accelspl),
    ("bordet D3", _sm_bordet(true, :(censored.(Normal.(log_y, exp.(ls)),
        lloq, uloq))), _SM_BORDET),
    ("bordet D4", _sm_bordet(false, :(censored.(StudentT.(4.0, log_y,
        exp.(ls)), lloq, uloq))), _SM_BORDET),
    ("bordet D5", _SM_BORDET_D5, _SM_BORDET_D5_DATA),
    ("bruno hsgp", _sm_bruno1(:x), (; x = _SM_BRUNO_X, y = _SM_BRUNO_Y)),
    ("bruno gp_effectiveness", _SM_BRUNO_EFF, _SM_BRUNO_EFF_DATA),
)

_sm_probe(n) = [0.2 * sin(1.3i) + 0.02i for i in 1:n]

@testset "smooth Enzyme-vs-findiff ($label)" for (label, prog, data) in _SM_ITEMS
    bound, built, _, lay = _sm_query(prog, _sm_cols(data))
    _check_gradient(built.spec, bound, _sm_probe(lay.total))
end

function _sm_reactant_measure(built, bound, post_q, u)
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; native, primal, val, g, rval = Float64(rval),
        rgrad = Array(rgrad))
end

# XLA legs for the basis-expansion items (HSGP / spline). The exact-GP
# items stay native-only: Reactant reverse through the dense Cholesky is
# the upstream gap documented in test_gp.jl (no XLA assertions there by
# design).
# Upstream XLA gap (the student-evidence pin precedent): the censored
# Student-t evidence arms route through `SpecialFunctions.beta_inc`,
# which has no method for a traced scalar (measured on Reactant
# 0.2.289/0.2.290). Only the Bordet D4 leg carries it; the signature
# below is exactly that gap, anything else rethrows loudly.
_sm_is_upstream_gap(e) =
    e isa MethodError && e.f === SpecialFunctions.beta_inc &&
    length(e.args) == 3 && e.args[3] isa Reactant.TracedRNumber

@testset "smooth under Reactant ($label)" for (label, prog, data) in _SM_ITEMS[3:end]
    gapped = label == "bordet D4"
    try
        bound, built, post_q, lay = _sm_query(prog, _sm_cols(data))
        fx = Base.invokelatest(_sm_reactant_measure, built, bound, post_q,
            _sm_probe(lay.total))
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
        # Self-firing pin: errors (Unexpected Pass) once upstream wires
        # beta_inc, forcing removal of the gate.
        gapped && @test_broken true
    catch e
        gapped && _sm_is_upstream_gap(e) || rethrow()
    end
end
