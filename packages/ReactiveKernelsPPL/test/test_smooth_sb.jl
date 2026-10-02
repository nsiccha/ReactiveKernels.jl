# Smooth family (matrix-b): exact GP (gp_regr / gp_pois_regr), HSGP
# (accel_gp / brm_hsgp) and spline (accel_splines) programs vs the pair
# partner's BridgeStan numbers, plus the stated-hyper-prior surface the
# HSGP/spline items need: `hsgp_basis(...; length_scale=, sd=)` and
# `spline_basis(...; sd=)` (SB `length_scale(:, hsgp(x)) ~ ...` /
# `sd(:, hsgp(x)) ~ ...` / `sd(mu, s(x)) ~ ...`), and HSGP-only
# predictors (SB `0 + hsgp(x)`, location and `exp.` scale).
#
# Also the fixed `domain=` and the bounding `length_scale = Uniform(lo, hi)`
# (SB prior-bound intersection) on a one-axis HSGP regression.
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
# `_sb_hsgp` verbatim; the spline blocks are SB's (term-splines-stan)
# minus the tps constant column (accel_splines bridges it below). A
# stated length-scale prior drops the validity floor (BRM
# `_brm_hsgp_declared_rho_lower`), which the brm_hsgp/accel_gp values pin
# (SB constrains rho = exp(u) there). The banked probes are uniform
# (zeros and 0.3^n), so the n = coordinate COUNT and the layout-name
# assertions carry the mapping; the HSGP/spline math itself is pinned
# against independent oracles in test_hsgp.jl / test_spline.jl.
# (`_check_gradient` / `_GEN_BACKEND` come from test_generator.jl,
# included earlier by runtests.jl.)
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Statistics: std
using Test

function _sm_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
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
    plan = lower_rkppl(_SM_ACCELGP, (:y, :x, :x2); conditioned = (:y, :x, :x2))
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
    end, (:y, :x); conditioned = (:y, :x))
    @test only(dplan.hsgp_bases).rho_prior === nothing
    dlay = assign_layout(bind_data(dplan, _sm_cols((; x = _SM_AX,
        y = _SM_DATA.accelgp.y))))
    @test Dict(e.name => e.transform for e in dlay.entries)[:rho_h_x] ===
        :floored
    # Stated spline sd prior rides the sd vector (Stan-kernel support).
    splan = lower_rkppl(_SM_ACCELSPL, (:y, :x, :x2); conditioned = (:y, :x, :x2))
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
            s ~ Exponential(1)
            mu = a .+ hsgp(:h_x)
            hsgp_basis(:h_x, x; k = 8, sd = $bad)
            y .~ Normal.(mu, 1.0)
        end
        if bad == :(StudentT(3, 0))
            # refused: StudentT is missing its scale argument (P3 arity).
            @test_throws SurfaceLoweringError lower_rkppl(ex, (:y, :x); conditioned = (:y, :x))
        else
            # capability: proper, bounded and sampled-argument basis
            # hyper priors (P7/P8 1cmodra; todo `0bfiemp`).
            @test_broken (lower_rkppl(ex, (:y, :x); conditioned = (:y, :x)); true)
        end
    end
    # refused: unknown keyword `lengthscale` (unsupported kwarg is a Julia MethodError, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        mu = a .+ hsgp(:h_x)
        hsgp_basis(:h_x, x; k = 8, lengthscale = LogNormal(0, 1))
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    # capability: HalfCauchy spline-sd hyper prior (proper half-distribution, honest spelling) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
        spline_basis(:s_x, x; sd = HalfCauchy(2))
        a ~ Normal(0, 1)
        mu = a .+ spline(:s_x)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x)); true)
    # A hand-built plan's hyper prior is contract-validated too.
    bad = HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, hb.fits, hb.label,
        hb.cov, hb.period, HyperPrior(:beta, (arg1 = 1.0, arg2 = 1.0)),
        nothing)
    # capability: a Beta smooth-SD prior with explicit shape arguments (P8 1cmodra; todo `0bfiemp`).
    @test_broken (validate_structure(
        StructuralPlan(plan.responses, plan.predictors,
            plan.population_priors, plan.parameters, plan.assignments,
            plan.columns, plan.n_obs; derived = plan.derived,
            hsgp_bases = [bad, plan.hsgp_bases[2]])); true)
end

const _SM_DOMAIN_X = [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0]
const _SM_DOMAIN_Y = [0.2, -0.3, 0.5, 1.1, 0.7, -0.1, -0.6, 0.3]
# One-axis HSGP on a fixed domain L = 1.5 (`domain=(-1.5, 1.5)`), rho
# bounded by the K=10 validity floor (4*1.5/pi)*sqrt(log(100)/99).
const _SM_DOMAIN_HSGP = quote
    b0 ~ Normal(0, 5)
    mu = b0 .+ hsgp(:h)
    hsgp_basis(:h, x; k = 10, domain = (-1.5, 1.5),
        length_scale = Uniform(0.4119140661125232, 2), sd = LogNormal(0, 1))
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end

@testset "hsgp domain + bounding hyper prior" begin
    plan = lower_rkppl(_SM_DOMAIN_HSGP, (:y, :x); conditioned = (:y, :x))
    hb = only(plan.hsgp_bases)
    @test hb.domain == [(-1.5, 1.5)]
    @test hb.rho_prior == HyperPrior(:uniform,
        (arg1 = 0.4119140661125232, arg2 = 2.0))
    bound = bind_data(plan, _sm_cols((; x = _SM_DOMAIN_X, y = _SM_DOMAIN_Y)))
    @test only(bound.hsgp_bases).fits == [(0.0, 1.5)]
    e = only(en for en in assign_layout(bound).entries
        if en.name === :rho_h)
    @test (e.transform, e.lo, e.hi) == (:interval, 0.4119140661125232, 2.0)
    # Data outside the fixed domain fails at bind; `domain` excludes `c`
    # and periodic; malformed pairs fail at the surface.
    # refused: data outside the fixed approximation domain
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
        # refused: over-determined boundary, domain and c both set it (P2)
        @test_throws Union{SurfaceLoweringError,ContractValidationError} bind_data(
            lower_rkppl(ex, (:y, :x); conditioned = (:y, :x)),
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
    # accel_splines: P-accelspl12 (SB 0.3^24; per basis 2 fixed + 8 pen +
    # sd). Intended divergence: RK's tps drops SB's constant null column
    # (decision `1cmodra`, prong `tps-intercept`), so RK packs 1 fixed per
    # basis (22 coordinates). Folding SB's constant coefficients (0.3
    # each) into the intercepts b0 and s0 makes the models identical: RK
    # at 0.3^22 with both intercepts at 0.6 equals the SB literal up to
    # those two priors, StudentT(3, -13, 36) and StudentT(3, 0, 10).
    # Both bases use canonical signs (1vts6mb); BRM's independently compiled
    # reference is spline_accel_parity_case at canonical 040a759f8a056a3f07f59f851b440db607e9a7d7.
    _, _, kern, lay = _sm_query(_SM_ACCELSPL, _sm_cols(_SM_DATA.accelspl))
    @test lay.total == 22
    names = coordinate_names(lay)
    @test names[1:3] ==
        [:b0, :s0, Symbol("b_s_x_fixed.1")]
    u = [0.6; 0.6; fill(0.3, 20)]
    t3(m, s, v) = logpdf(TDist(3), (v - m) / s) - log(s)
    bridge = t3(-13, 36, 0.6) - t3(-13, 36, 0.3) + t3(0, 10, 0.6) -
        t3(0, 10, 0.3)
    @test _sm_val(kern, zeros(22)) ≈ -46.468310103713605 atol = 1e-10
    @test _sm_val(kern, u) ≈ -69.1125421088841 + bridge atol = 1e-10
end

@testset "grouped HSGP surface + contract" begin
    base(kws...) = quote
        a ~ Normal(0, 1)
        hsgp_basis(:h, x; k = 6, $(kws...))
        mu = a .+ hsgp(:h)
        y .~ Normal.(mu, 1.0)
    end
    cols = (:y, :x, :g, :q)
    kw(k, v) = Expr(:kw, k, v)
    # `by` alone (shared hypers), `(1 | g)` without an intercept, and a
    # stated prior on the other hyper all lower.
    hb = only(lower_rkppl(base(kw(:by, :g)), cols; conditioned = cols).hsgp_bases)
    @test hb.by == HSGPGrouping(:g, nothing) && hb.rho_prior === nothing
    hb = only(lower_rkppl(base(kw(:by, :g),
        kw(:length_scale, :((1 | g))), kw(:sd, :(Normal(0, 1)))),
        cols; conditioned = cols).hsgp_bases)
    @test hb.rho_prior == HSGPHyperLP(false, :g)
    @test hb.sigma_prior isa HyperPrior
    # Fail closed: a hyper-predictor without `by`, grouped by another
    # column, a non-`1 + (1 | g)` formula, `by` on a 2-D / periodic basis,
    # and a non-column `by`.
    for bad in (base(kw(:length_scale, :(1 + (1 | g)))),
            base(kw(:by, :g), kw(:sd, :(1 + (1 | q)))),
            base(kw(:by, :g), kw(:sd, :(2 + (1 | g)))),
            base(kw(:by, :g), kw(:sd, :(1 + (x | g)))),
            base(kw(:by, :g), kw(:cov, QuoteNode(:periodic)),
                kw(:period, 1.0)),
            base(kw(:by, :(g .+ 1))))
        if bad in (base(kw(:by, :g), kw(:cov, QuoteNode(:periodic)),
                kw(:period, 1.0)), base(kw(:by, :(g .+ 1))))
            # capability: grouped periodic bases and computed groups
            # (P8 1cmodra; todo `0bfiemp`).
            @test_broken (lower_rkppl(bad, cols; conditioned = cols); true)
        else
            # refused: remaining hyper formulas lack a matching group or
            # contain an observation slope in a group-level slot (IR contract).
            @test_throws SurfaceLoweringError lower_rkppl(bad, cols; conditioned = cols)
        end
    end
    # capability: grouped multi-axis HSGP (by= with two axes; source: "v1 ... planned") (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        hsgp_basis(:h, x, q; k = (4, 4), by = g)
        mu = a .+ hsgp(:h)
        y .~ Normal.(mu, 1.0)
    end, cols; conditioned = cols); true)
    # A hand-built hyper-predictor without a grouping is contract-invalid.
    plan = lower_rkppl(base(kw(:by, :g),
        kw(:length_scale, :(1 + (1 | g)))), cols; conditioned = cols)
    hb = only(plan.hsgp_bases)
    bad = HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, hb.fits, hb.label,
        hb.cov, hb.period, hb.rho_prior, hb.sigma_prior, hb.domain, nothing)
    # refused: hyper-predictor without a grouping (IR contract)
    @test_throws ContractValidationError validate_structure(
        StructuralPlan(plan.responses, plan.predictors,
            plan.population_priors, plan.parameters, plan.assignments,
            plan.columns, plan.n_obs; derived = plan.derived,
            hsgp_bases = [bad]))
end

_sm_probe(n) = [0.2 * sin(1.3i) + 0.02i for i in 1:n]

const _SM_ITEMS = (
    ("gp_regr", _SM_GPREGR, _SM_DATA.gpregr),
    ("gp_pois_regr", _SM_GPPOIS, _SM_DATA.gppois),
    ("accel_gp", _SM_ACCELGP, _SM_DATA.accelgp),
    ("brm_hsgp", _SM_BRMHSGP, _sm_mcycle()),
    ("accel_splines", _SM_ACCELSPL, _SM_DATA.accelspl),
    ("hsgp domain", _SM_DOMAIN_HSGP, (; x = _SM_DOMAIN_X, y = _SM_DOMAIN_Y)),
)

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
@testset "smooth under Reactant ($label)" for (label, prog, data) in _SM_ITEMS[3:end]
    bound, built, post_q, lay = _sm_query(prog, _sm_cols(data))
    fx = Base.invokelatest(_sm_reactant_measure, built, bound, post_q,
        _sm_probe(lay.total))
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.rval ≈ fx.val rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
end
