# SB parity: matrix-b mixture (M1/M2/M3) + rate-GLM (G6/G7) legs vs the
# pair partner's BridgeStan numbers (briefs 2026-09-27T22-26-41-960-17idfk9
# (mixtures) and 2026-09-27T22-25-49-345-ia3a6q (glm) on
# BayesianRegressionModels:rk:kernel:matrix-b, BRM 88c5621, StanBlocks
# 342436de, BridgeStan 2.9.0). Full posterior at u, propto=false,
# Jacobian included. Partner formulas + data recovered from their banked
# run scripts (/tmp/matrix-b-mixtures.jl, /tmp/matrix-b-glm.jl); M2's
# simplex point machine-read from their compiled model
# (BridgeStan param_constrain at 0.3^14).
#
# Map audit (all verified in-tree): RK `:real`/`:positive`/`:interval`
# transforms are bit-identical to Stan's (affine-logistic interval,
# exp positive, identity real) with identical log-Jacobians, and
# Uniform/Beta/Gamma/Dirichlet prior densities match Stan's
# propto=false forms — so G6/G7 compare DIRECTLY at the same u, and
# mixture mus/sigmas ride identical coords. The ONE structural gap: RK
# simplex is stick-breaking, Stan's is ILR/softmax (Helmert contrasts —
# confirmed: hand-derived prediction matched machine-read
# param_constrain to 5 digits). Mixture legs therefore translate the
# probe (u_RK* = same mus/sigmas, w-block = stick-breaking inverse of
# w_SB) and assert
#   RK_value(u_RK*) - SB_banked == RK_stick_jac - SB_ILR_jac
# with SB_ILR_jac = sum(log, w) + 0.5*log(K) — confirmed empirically
# against BOTH banked M1 (gap closes exactly) and M2 (matches to ~1e-4
# by hand). Both directions of the stick map are hand-rolled below
# (independent of the implementation under test); the value assertions
# still catch implementation bugs because kern() runs the real maps.
# (`_check_gradient` / `_GEN_BACKEND` come from test_generator.jl.)
using Distributions: Normal, Dirichlet, Beta, Gamma, Binomial, Poisson,
    Bernoulli, Cauchy, logpdf
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# ---- Hand-rolled stick-breaking pair (independent of layout.jl) ----
_sb_logistic(x) = 1.0 / (1.0 + exp(-x))

# Inverse: simplex -> K-1 logits (z_j = w_j / remaining).
function _sb_unconstrain(w::AbstractVector)
    K = length(w)
    u = Vector{Float64}(undef, K - 1)
    remaining = 1.0
    for j in 1:(K - 1)
        z = w[j] / remaining
        u[j] = log(z) - log1p(-z) - log(K - j)
        remaining -= w[j]
    end
    return u
end

# Forward log-Jacobian: sum over breaks of log(r) + log(z) + log1p(-z).
function _sb_logjac(u::AbstractVector)
    K = length(u) + 1
    lr, acc = 0.0, 0.0
    for j in 1:(K - 1)
        z = _sb_logistic(u[j] + log(K - j))
        l = log1p(-z)
        acc += lr + log(z) + l
        lr += l
    end
    return acc
end

# Stan ILR simplex log-Jacobian (empirically confirmed, see header).
_stan_simplex_logjac(w::AbstractVector) = sum(log, w) + 0.5 * log(length(w))

# Stable K-term log-sum-exp over a vector of log terms.
function _sb_logsumexp(ts::AbstractVector)
    m = maximum(ts)
    return m + log(sum(exp, ts .- m))
end

function _sb_query(prog::Expr, cols::Dict{Symbol,<:AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

# Posterior at an unconstrained probe (world-age-safe call).
_sb_posterior(kern, lay, u) = Base.invokelatest(kern, u)

@testset "M1 normal_mixture SB parity" begin
    # SB: mu1,mu2 ~ Normal(0,10); w ~ Dirichlet(2,1.0);
    # y ~ Mixture([N(mu1,1), N(mu2,1)], w); y=[-2,-1.8,1.9,2.2];
    # u=zeros(3), names [mu1,mu2,w.1], value -19.003522156056047.
    # w_SB at zeros = [0.5, 0.5] (machine-read CONSTRAINED M1).
    prog = Meta.parse("""begin
        mu1 ~ Normal(0.0, 10.0)
        mu2 ~ Normal(0.0, 10.0)
        w ~ Dirichlet(2, 1.0)
        y .~ MixtureModel.([Normal.(mu1, 1.0), Normal.(mu2, 1.0)], w)
    end""")
    cols = Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2])
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    w_sb = [0.5, 0.5]
    uw = _sb_unconstrain(w_sb)
    @test length(uw) == 1 && abs(uw[1]) < 1e-15
    widx = findall(n -> !(n in (:mu1, :mu2)), names)
    @test length(widx) == 1
    u = zeros(length(names))
    u[widx] .= uw
    got = _sb_posterior(kern, lay, u)
    sb_val = -19.003522156056047
    # Cross-parity: same constrained point, Jacobian difference only.
    @test abs((got - sb_val) -
              (_sb_logjac(uw) - _stan_simplex_logjac(w_sb))) < 1e-12
    # Independent oracle at the RK probe (self-consistency).
    pr = logpdf(Normal(0, 10), 0.0) + logpdf(Normal(0, 10), 0.0) +
         logpdf(Dirichlet([1.0, 1.0]), w_sb)
    lik = sum(cols[:y]) do v
        _sb_logsumexp([log(w_sb[1]) + logpdf(Normal(0, 1), v),
            log(w_sb[2]) + logpdf(Normal(0, 1), v)])
    end
    @test abs(got - (pr + lik + _sb_logjac(uw))) < 1e-12
    _check_gradient(built.spec, bound, u)
end

@testset "M2 normal_mixture_k SB parity" begin
    # SB: mu1..5 ~ Normal(0,10); s1..5 ~ Uniform(0,10); w ~ Dirichlet(5,1.0);
    # y ~ Mixture(5 x N(mui,si), w); y=[-3,-2.5,0.1,0.5,2.0,2.8];
    # u=0.3^14, value -43.83833258031802. w_SB machine-read from the
    # partner's compiled model (param_constrain at 0.3^14); mus/sigmas ride
    # identical coords (identity + affine-logistic interval, same u).
    prog = Meta.parse("""begin
        mu1 ~ Normal(0.0, 10.0)
        mu2 ~ Normal(0.0, 10.0)
        mu3 ~ Normal(0.0, 10.0)
        mu4 ~ Normal(0.0, 10.0)
        mu5 ~ Normal(0.0, 10.0)
        s1 ~ Uniform(0.0, 10.0)
        s2 ~ Uniform(0.0, 10.0)
        s3 ~ Uniform(0.0, 10.0)
        s4 ~ Uniform(0.0, 10.0)
        s5 ~ Uniform(0.0, 10.0)
        w ~ Dirichlet(5, 1.0)
        y .~ MixtureModel.([Normal.(mu1, s1), Normal.(mu2, s2),
            Normal.(mu3, s3), Normal.(mu4, s4), Normal.(mu5, s5)], w)
    end""")
    y = [-3.0, -2.5, 0.1, 0.5, 2.0, 2.8]
    cols = Dict{Symbol,AbstractVector}(:y => y)
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    w_sb = [0.31350412057955673, 0.20511041318944392, 0.17560848639229368,
        0.15866512050066742, 0.14711185933803816]
    uw = _sb_unconstrain(w_sb)
    rest = [:mu1, :mu2, :mu3, :mu4, :mu5, :s1, :s2, :s3, :s4, :s5]
    widx = findall(n -> !(n in rest), names)
    @test length(widx) == 4
    u = [n in rest ? 0.3 : 0.0 for n in names]
    u[widx] .= uw
    got = _sb_posterior(kern, lay, u)
    sb_val = -43.83833258031802
    @test abs((got - sb_val) -
              (_sb_logjac(uw) - _stan_simplex_logjac(w_sb))) < 1e-12
    # Independent oracle at the RK probe (self-consistency).
    s_at = 10.0 * _sb_logistic(0.3)
    pr = sum(logpdf(Normal(0, 10), 0.3) for _ in 1:5) + 5 * (-log(10.0)) +
         logpdf(Dirichlet(fill(1.0, 5)), w_sb)
    lik = sum(y) do v
        _sb_logsumexp([log(w_sb[k]) + logpdf(Normal(0.3, s_at), v)
                       for k in 1:5])
    end
    ijac = log(s_at) + log(10.0 - s_at) - log(10.0)
    @test abs(got - (pr + lik + 5 * ijac + _sb_logjac(uw))) < 1e-12
    _check_gradient(built.spec, bound, u)
end

@testset "M3 low_dim_gauss_mix SB parity" begin
    # SB: mu1,mu2 ~ Normal(0,2); s1,s2 ~ Normal(0,2) (UNCONSTRAINED reals
    # threaded raw as sigmas — garbage-in-garbage-out exactly like Stan);
    # w ~ Dirichlet(2,5.0); y=[-1.5,-1.2,0.9,1.1]; value -41.77127219940857.
    # PROBE ERRATUM: the partner brief table says u=zeros, but the banked
    # value is at 0.3^5 (their reruns script; at zeros sigma=0 makes the
    # density non-finite, while 0.3^5 reconstructs the banked value).
    # w_SB = Stan K=2 ILR at z=0.3 (partner-validated simplex2).
    # M4 (low_dim_gauss_mix_collapse) is banked IDENTICAL (m4=m3 in the
    # partner script, same value) — this leg covers both.
    prog = Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        s1 ~ Normal(0.0, 2.0)
        s2 ~ Normal(0.0, 2.0)
        w ~ Dirichlet(2, 5.0)
        y .~ MixtureModel.([Normal.(mu1, s1), Normal.(mu2, s2)], w)
    end""")
    y = [-1.5, -1.2, 0.9, 1.1]
    cols = Dict{Symbol,AbstractVector}(:y => y)
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    w_sb = [0.6045031524689135, 0.3954968475310864]
    uw = _sb_unconstrain(w_sb)
    rest = [:mu1, :mu2, :s1, :s2]
    widx = findall(n -> !(n in rest), names)
    @test length(widx) == 1
    u = [n in rest ? 0.3 : 0.0 for n in names]
    u[widx] .= uw
    got = _sb_posterior(kern, lay, u)
    sb_val = -41.77127219940857
    @test abs((got - sb_val) -
              (_sb_logjac(uw) - _stan_simplex_logjac(w_sb))) < 1e-12
    pr = sum(logpdf(Normal(0, 2), 0.3) for _ in 1:4) +
         logpdf(Dirichlet([5.0, 5.0]), w_sb)
    lik = sum(y) do v
        _sb_logsumexp([log(w_sb[1]) + logpdf(Normal(0.3, 0.3), v),
            log(w_sb[2]) + logpdf(Normal(0.3, 0.3), v)])
    end
    @test abs(got - (pr + lik + _sb_logjac(uw))) < 1e-12
    _check_gradient(built.spec, bound, u)
end

@testset "G6 beta-binomial SB parity" begin
    # SB: rate ~ Beta(2,2); successes ~ Binomial(trials, rate);
    # successes=[7,4,9], trials=[10,10,12]; u=zeros(1);
    # value -7.633312211078095, grad [4.0]. Direct u-parity (same
    # logistic map + Beta density both sides).
    prog = Meta.parse("""begin
        rate ~ Beta(2.0, 2.0)
        k .~ Binomial.(n, rate)
    end""")
    cols = Dict{Symbol,AbstractVector}(:k => [7, 4, 9], :n => [10, 10, 12])
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    u = _sb_vec(names, [:rate => 0.0])
    @test abs(Base.invokelatest(kern, u) - (-7.633312211078095)) < 1e-12
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test maximum(abs.(g .- _sb_vec(names, [:rate => 4.0]))) < 1e-10
    # Independent oracle at the probe (rate=0.5).
    want = logpdf(Beta(2.0, 2.0), 0.5) +
           sum(logpdf(Binomial(n, 0.5), k)
               for (k, n) in ((7, 10), (4, 10), (9, 12))) +
           log(0.5) + log(0.5)
    @test abs(Base.invokelatest(kern, u) - want) < 1e-12
    _check_gradient(built.spec, bound, u)
end

@testset "G7 poisson-gamma SB parity" begin
    # SB: rate ~ Gamma(2,1); counts ~ Poisson(rate); counts=[3,1,6,2];
    # u=zeros(1); value -14.064157861798101, grad [9.0]. Direct u-parity
    # (same exp map + Gamma density both sides).
    prog = Meta.parse("""begin
        rate ~ Gamma(2.0, 1.0)
        y .~ Poisson.(rate)
    end""")
    cols = Dict{Symbol,AbstractVector}(:y => [3, 1, 6, 2])
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    u = _sb_vec(names, [:rate => 0.0])
    @test abs(Base.invokelatest(kern, u) - (-14.064157861798101)) < 1e-12
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test maximum(abs.(g .- _sb_vec(names, [:rate => 9.0]))) < 1e-10
    want = logpdf(Gamma(2.0, 1.0), 1.0) +
           sum(logpdf(Poisson(1.0), v) for v in (3, 1, 6, 2)) +
           log(1.0) # exp Jacobian at u=0 is log(1)=0; kept explicit
    @test abs(Base.invokelatest(kern, u) - want) < 1e-12
    _check_gradient(built.spec, bound, u)
end

# NOTE (G3/G4): glmm1_model / glmm_poisson have NO RK counterpart — the
# SB form samples (tau, group-totals) with the fixed effects ANALYTICALLY
# MARGINALIZED (brm_total: beta = conditional mean, -0.5*logdet(Q)
# adjustment; 4- and 10-dim probes), while the thin layer's varying
# effects are explicit non-centered (beta, tau, z). Marginal != joint:
# no probe translation closes the gap (verified structural wall, same
# class as the partner's HMM/occupancy walls but on the RK side). The
# GLM verdict brief records G3/G4 as NO-COUNTERPART.

@testset "G1 glm_binomial SB parity" begin
    # SB: eta ~ 1+year+year2, Normal(0,100) coefs;
    # counts ~ BinomialLogit(totals, eta); 6 rows;
    # u=[0.1,0.2,-0.1] (intercept, year, year2);
    # value -43.311049476871304. Direct u-parity (identity coefs).
    prog = Meta.parse("""begin
        a ~ Normal(0.0, 100.0)
        b1 ~ Normal(0.0, 100.0)
        b2 ~ Normal(0.0, 100.0)
        eta = a .+ b1 .* year .+ b2 .* year2
        counts .~ Binomial.(totals, logistic.(eta))
    end""")
    df = (counts = [14, 9, 11, 6, 4, 7], totals = fill(20, 6),
        year = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        year2 = [6.25, 2.25, 0.25, 0.25, 2.25, 6.25])
    cols = Dict{Symbol,AbstractVector}(:counts => df.counts,
        :totals => df.totals, :year => df.year, :year2 => df.year2)
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    @test Set(names) == Set([Symbol("eta.Intercept"), Symbol("eta.year"),
        Symbol("eta.year2")])
    u = _sb_vec(names, [Symbol("eta.Intercept") => 0.1,
        Symbol("eta.year") => 0.2, Symbol("eta.year2") => -0.1])
    @test abs(Base.invokelatest(kern, u) - (-43.311049476871304)) < 1e-12
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    want = _sb_vec(names, [Symbol("eta.Intercept") => -3.683080936679581,
        Symbol("eta.year") => -43.93322274728044,
        Symbol("eta.year2") => 22.58426637610638])
    @test maximum(abs.(g .- want)) < 1e-10
    # Independent oracle at the probe.
    eta = 0.1 .+ 0.2 .* df.year .- 0.1 .* df.year2
    wantv = sum(logpdf(Normal(0, 100), c) for c in (0.1, 0.2, -0.1)) +
            sum(logpdf(Binomial(20, _sb_logistic(e)), k)
                for (e, k) in zip(eta, df.counts))
    @test abs(Base.invokelatest(kern, u) - wantv) < 1e-12
    _check_gradient(built.spec, bound, u)
end

@testset "G2 glm_poisson SB parity" begin
    # SB: log(mu) ~ 1+year+year2+year3, Flat coefs (DELTA: slice has
    # bounded-uniform); counts ~ Poisson(mu); 6 rows;
    # u=[0.2,0.1,-0.05,0.03]; value -17.788101912439263. Direct u-parity
    # (identity coefs, no prior terms).
    prog = Meta.parse("""begin
        a ~ Flat()
        b1 ~ Flat()
        b2 ~ Flat()
        b3 ~ Flat()
        mu = a .+ b1 .* year .+ b2 .* year2 .+ b3 .* year3
        counts .~ Poisson.(exp.(mu))
    end""")
    df = (counts = [3, 1, 6, 2, 1, 4],
        year = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        year2 = [6.25, 2.25, 0.25, 0.25, 2.25, 6.25],
        year3 = [-15.625, -3.375, -0.125, 0.125, 3.375, 15.625])
    cols = Dict{Symbol,AbstractVector}(:counts => df.counts,
        :year => df.year, :year2 => df.year2, :year3 => df.year3)
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    @test Set(names) == Set([Symbol("mu.Intercept"), Symbol("mu.year"),
        Symbol("mu.year2"), Symbol("mu.year3")])
    u = _sb_vec(names, [Symbol("mu.Intercept") => 0.2,
        Symbol("mu.year") => 0.1, Symbol("mu.year2") => -0.05,
        Symbol("mu.year3") => 0.03])
    @test abs(Base.invokelatest(kern, u) - (-17.788101912439263)) < 1e-12
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    want = _sb_vec(names, [Symbol("mu.Intercept") => 10.062859779706784,
        Symbol("mu.year") => -3.8913188535023844,
        Symbol("mu.year2") => 30.397137846772317,
        Symbol("mu.year3") => -8.606116623948758])
    @test maximum(abs.(g .- want)) < 1e-10
    eta = 0.2 .+ 0.1 .* df.year .- 0.05 .* df.year2 .+ 0.03 .* df.year3
    wantv = sum(logpdf(Poisson(exp(e)), k)
                for (e, k) in zip(eta, df.counts))
    @test abs(Base.invokelatest(kern, u) - wantv) < 1e-12
    _check_gradient(built.spec, bound, u)
end

@testset "G5 logistic_regression_rhs SB parity" begin
    # SB: b0 ~ Normal(0,5); b1,b2 ~ Horseshoe(local=1, global=0.1)
    # (DELTA: slice has regularized HS + slab); mu = b0+b1*x1+b2*x2;
    # y ~ BernoulliLogit(mu); 6 rows. Direct u-parity: the thin-layer
    # horseshoe triples (raw/intercept-Normal, lambda/tau Stan-kernel
    # half-Cauchy, b = raw*lambda*tau) match Stan's emission exactly.
    prog = Meta.parse("""begin
        b0 ~ Normal(0.0, 5.0)
        b1 ~ Horseshoe(local_scale = 1.0, global_scale = 0.1)
        b2 ~ Horseshoe(local_scale = 1.0, global_scale = 0.1)
        mu = b0 .+ b1 .* x1 .+ b2 .* x2
        y .~ Bernoulli.(logistic.(mu))
    end""")
    df = (y = [0, 1, 1, 0, 1, 0],
        x1 = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        x2 = [1.0, 0.5, -0.5, 1.5, 0.0, -1.0])
    cols = Dict{Symbol,AbstractVector}(:y => df.y, :x1 => df.x1,
        :x2 => df.x2)
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    # SB order [b0, b1_raw, b1_lambda, b1_tau, b2_raw, b2_lambda, b2_tau].
    sb_names = [:b0, :b1_raw, :b1_lambda, :b1_tau,
        :b2_raw, :b2_lambda, :b2_tau]
    rk_of = Dict(:b0 => :horseshoe_mu_Intercept_normal,
        :b1_raw => :horseshoe_mu_x1_raw,
        :b1_lambda => :horseshoe_mu_x1_lambda,
        :b1_tau => :horseshoe_mu_x1_tau,
        :b2_raw => :horseshoe_mu_x2_raw,
        :b2_lambda => :horseshoe_mu_x2_lambda,
        :b2_tau => :horseshoe_mu_x2_tau)
    @test Set(names) == Set(values(rk_of))
    cases = ((fill(0.0, 7), -19.11542134761971,
        [0.0, -0.75, 0.0, -0.9801980198019802, -0.75, 0.0,
            -0.9801980198019802]),
        ([0.1, 0.5, 0.2, -0.3, 0.4, 0.1, -0.2], -19.632259916162344,
            [-0.443292992040744, -1.5463985021133373, -0.7205745712815728,
                -1.4874090156567834, -1.3113983243232483,
                -0.46422732435425496, -1.3351614013328863]))
    for (uu, val, grad) in cases
        u = _sb_vec(names, [rk_of[s] => v for (s, v) in zip(sb_names, uu)])
        @test abs(Base.invokelatest(kern, u) - val) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _sb_vec(names,
            [rk_of[s] => v for (s, v) in zip(sb_names, grad)])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    # Independent oracle at the u2 probe.
    b0v, r1, l1, t1, r2, l2, t2 = 0.1, 0.5, exp(0.2), exp(-0.3), 0.4,
        exp(0.1), exp(-0.2)
    bb1, bb2 = r1 * l1 * t1, r2 * l2 * t2
    mu = b0v .+ bb1 .* df.x1 .+ bb2 .* df.x2
    wantv = logpdf(Normal(0, 5), b0v) + logpdf(Normal(0, 1), r1) +
            logpdf(Normal(0, 1), r2) +
            logpdf(Cauchy(0, 1), l1) + logpdf(Cauchy(0, 0.1), t1) +
            logpdf(Cauchy(0, 1), l2) + logpdf(Cauchy(0, 0.1), t2) +
            sum(logpdf(Bernoulli(_sb_logistic(e)), v)
                for (e, v) in zip(mu, df.y)) +
            0.2 - 0.3 + 0.1 - 0.2 # exp Jacobians on lambda/tau
    u2 = _sb_vec(names, [rk_of[s] => v for (s, v) in
        zip(sb_names, [0.1, 0.5, 0.2, -0.3, 0.4, 0.1, -0.2])])
    @test abs(Base.invokelatest(kern, u2) - wantv) < 1e-12
    _check_gradient(built.spec, bound, u2)
end

@testset "G5m logistic minimal SB parity" begin
    # SB minimal horseshoe isolation: b0 ~ Normal(0,5),
    # b1 ~ Horseshoe(1.0, 0.1) over x1 == 0 (N=2, y=[0,1], so b1*x1 = 0
    # and mu = b0 regardless). zeros(4) -> -10.128751716069296; u2 ->
    # -10.362079506135036.
    prog = Meta.parse("""begin
        b0 ~ Normal(0.0, 5.0)
        b1 ~ Horseshoe(local_scale = 1.0, global_scale = 0.1)
        mu = b0 .+ b1 .* x1
        y .~ Bernoulli.(logistic.(mu))
    end""")
    df = (y = [0, 1], x1 = [0.0, 0.0])
    cols = Dict{Symbol,AbstractVector}(:y => df.y, :x1 => df.x1)
    bound, built, kern, lay = _sb_query(prog, cols)
    names = coordinate_names(lay)
    sb_names = [:b0, :b1_raw, :b1_lambda, :b1_tau]
    rk_of = Dict(:b0 => :horseshoe_mu_Intercept_normal,
        :b1_raw => :horseshoe_mu_x1_raw,
        :b1_lambda => :horseshoe_mu_x1_lambda,
        :b1_tau => :horseshoe_mu_x1_tau)
    @test Set(names) == Set(values(rk_of))
    cases = ((fill(0.0, 4), -10.128751716069296,
        [0.0, 0.0, 0.0, -0.9801980198019802]),
        ([0.2, -0.4, 0.3, 0.1], -10.362079506135036,
            [-0.10766799462495585, 0.4, -0.291312612451591,
                -0.9837583602379762]))
    for (uu, val, grad) in cases
        u = _sb_vec(names, [rk_of[s] => v for (s, v) in zip(sb_names, uu)])
        @test abs(Base.invokelatest(kern, u) - val) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _sb_vec(names,
            [rk_of[s] => v for (s, v) in zip(sb_names, grad)])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    _check_gradient(built.spec, bound,
        _sb_vec(names, [rk_of[s] => v for (s, v) in
            zip(sb_names, [0.2, -0.4, 0.3, 0.1])]))
end
