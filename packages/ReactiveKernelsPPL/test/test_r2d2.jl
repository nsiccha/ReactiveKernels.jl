# Flat whole-predictor R2D2 variance decomposition (SB `effect(lp,:) ~ r2d2`
# mirror): share composition, derived per-column scales, admission.
using Distributions: Normal, Beta, Exponential, Dirichlet, logpdf
using Test

function _r2d2_demo(; tau_decl = nothing)
    decl = tau_decl === nothing ? :(r2d2(mu, R2, phi)) :
        :(r2d2(mu, R2, phi, $tau_decl))
    return quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        mu = a .+ b1 .* x1 .+ b2 .* x2
        $decl
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
end

function _r2d2_cols()
    return Dict{Symbol,AbstractVector}(
        :x1 => [0.5, -1.0, 1.5, 0.0],
        :x2 => [1.0, 0.5, -0.5, 2.0],
        :y => [1.0, 2.0, 1.5, 2.5])
end

# SB `brm_r2d2_scale` formula, hand-evaluated (independent of the emitter).
_r2d2_scale(phi, r2, tau, vx) = sqrt(phi * r2 * tau^2 / vx)

@testset "r2d2 surface admission" begin
    plan = lower_rkppl(_r2d2_demo(), Set([:x1, :x2, :y]))
    @test length(plan.r2d2_priors) == 1
    rp = only(plan.r2d2_priors)
    @test rp.predictor === :mu
    @test rp.r2 === :R2
    @test rp.phi === :phi
    @test rp.tau === :r2d2_mu_tau_bsv # synthesized half-standard-Normal
    @test isempty(rp.overrides)
    @test isempty(plan.population_priors) # coverage moved to the R2D2Prior
    tau = only(p for p in plan.parameters if p.name === :r2d2_mu_tau_bsv)
    @test tau.family === :normal && tau.args == (arg1 = 0, arg2 = 1)
    @test tau.support_override === :positive
    # Explicit sampled tau + literal tau spellings.
    plan2 = lower_rkppl(quote
            R2 ~ Beta(2.0, 2.0)
            phi ~ Dirichlet([1.0, 1.0])
            tau ~ HalfNormal(1.0)
            mu = a .+ b1 .* x1 .+ b2 .* x2
            r2d2(mu, R2, phi, tau)
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, Set([:x1, :x2, :y]))
    @test only(plan2.r2d2_priors).tau === :tau
    @test !any(p -> p.name === :r2d2_mu_tau_bsv, plan2.parameters)
    plan3 = lower_rkppl(_r2d2_demo(; tau_decl = 2.5), Set([:x1, :x2, :y]))
    @test only(plan3.r2d2_priors).tau == 2.5
end

@testset "r2d2 overrides ride stated Normals" begin
    plan = lower_rkppl(quote
            R2 ~ Beta(1.0, 1.0)
            phi ~ Dirichlet([1.0])
            b1 ~ Normal(0.5, 2.0)
            mu = a .+ b1 .* x1 .+ b2 .* x2
            r2d2(mu, R2, phi)
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, Set([:x1, :x2, :y]))
    rp = only(plan.r2d2_priors)
    @test rp.overrides == Dict(:x1 => (0.5, 2.0)) # b1 leaves the simplex
    cols = _r2d2_cols()
    bound = bind_data(plan, cols)
    pred = only(p for p in bound.predictors if p.name === :mu)
    shape = ReactiveKernelsPPL.design_shape(pred, bound.columns;
        levelmaps = bound.levelmaps)
    share, fallback, loc, _ =
        ReactiveKernelsPPL.r2d2_column_scales(shape, bound.columns,
            rp.overrides)
    @test share == [0, 0, 1] # intercept, overridden b1, b2 takes share 1
    @test fallback == [1.0, 2.0, 1.0]
    @test loc == [0.0, 0.5, 0.0]
end

@testset "r2d2 share composition over factors" begin
    plan = lower_rkppl(quote
            R2 ~ Beta(1.0, 1.0)
            phi ~ Dirichlet([1.0, 1.0, 1.0])
            mu = b1 .* x1 .+ c[g]
            r2d2(mu, R2, phi)
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, Set([:x1, :g, :y]))
    cols = Dict{Symbol,AbstractVector}(
        :x1 => [0.5, -1.0, 1.5, 0.0, 1.0, -0.5],
        :g => [1, 2, 1, 2, 1, 2],
        :y => [1.0, 2.0, 1.5, 2.5, 1.0, 2.0])
    bound = bind_data(plan, cols)
    @test only(bound.vector_parameters).size == 3
    pred = only(p for p in bound.predictors if p.name === :mu)
    shape = ReactiveKernelsPPL.design_shape(pred, bound.columns;
        levelmaps = bound.levelmaps)
    rp = only(bound.r2d2_priors)
    share, _, _, varx =
        ReactiveKernelsPPL.r2d2_column_scales(shape, bound.columns,
            rp.overrides)
    @test share == [1, 2, 3] # continuous + one share per dummy
    n = 6
    @test varx[1] ≈ sum((x - sum(cols[:x1]) / n)^2 for x in cols[:x1]) / (n - 1)
    @test varx[2] ≈ 3 * 3 / (n * (n - 1)) # brm_cat_variances, level 1
    @test varx[3] ≈ 3 * 3 / (n * (n - 1)) # ... level 2
end

@testset "r2d2 e2e values and gradient" begin
    bound = bind_data(lower_rkppl(_r2d2_demo(), Set([:x1, :x2, :y])),
        _r2d2_cols())
    built = build_kernel(bound)
    # Layout: 3 coefs + R2 + sigma + tau + 1 phi stick.
    u = [0.1, -0.2, 0.3, 0.4, 0.5, -0.1, 0.2]
    b = u[1:3]
    s_r2 = 1 / (1 + exp(-u[4]))
    r2v = s_r2
    sig = exp(u[5])
    tau = exp(u[6])
    s_ph = 1 / (1 + exp(-u[7]))
    ph = [s_ph, 1 - s_ph]
    cols = _r2d2_cols()
    n = 4
    vx = [sum((x - sum(c) / n)^2 for x in c) / (n - 1)
        for c in (cols[:x1], cols[:x2])]
    sc = [_r2d2_scale(ph[k], r2v, tau, vx[k]) for k in 1:2]
    mu = b[1] .+ b[2] .* cols[:x1] .+ b[3] .* cols[:x2]
    ll = sum(logpdf.(Normal.(mu, sig), cols[:y]))
    pr = logpdf(Normal(0, 1), b[1]) +
        logpdf(Normal(0, sc[1]), b[2]) + logpdf(Normal(0, sc[2]), b[3]) +
        logpdf(Beta(1, 1), r2v) + logpdf(Exponential(1), sig) +
        logpdf(Normal(0, 1), tau) + log(2) + # :positive half renormalizer
        logpdf(Dirichlet([1.0, 1.0]), ph)
    jac = u[5] + u[6] + log(s_r2) + log1p(-s_r2) + log(s_ph) + log1p(-s_ph)
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "r2d2 fail-closed battery" begin
    cols = _r2d2_cols()
    # Declaration shape.
    bad = quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        mu = a .+ b1 .* x1 .+ b2 .* x2
        r2d2(mu, R2)
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    @test_throws SurfaceLoweringError lower_rkppl(bad, Set([:x1, :x2, :y]))
    dup = quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        mu = a .+ b1 .* x1 .+ b2 .* x2
        r2d2(mu, R2, phi)
        r2d2(mu, R2, phi)
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    @test_throws SurfaceLoweringError lower_rkppl(dup, Set([:x1, :x2, :y]))
    ghost = quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        mu = a .+ b1 .* x1 .+ b2 .* x2
        r2d2(nope, R2, phi)
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    @test_throws SurfaceLoweringError lower_rkppl(ghost, Set([:x1, :x2, :y]))
    # Parameter families (structural, via hand-built plans).
    good = lower_rkppl(_r2d2_demo(), Set([:x1, :x2, :y]))
    notbeta = StructuralPlan(good.responses, good.predictors,
        good.population_priors,
        [p.name === :R2 ? SampledParameter(:R2, :normal, (arg1 = 0.0, arg2 = 1.0),
            nothing, :R2) : p for p in good.parameters],
        good.assignments, good.columns, good.n_obs,
        roles = good.roles, derived = good.derived, levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        ranef_buckets = good.ranef_buckets,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = good.kernel_plates, r2d2_priors = good.r2d2_priors)
    @test_throws ContractValidationError validate_structure(notbeta)
    # Bind-time: phi size mismatch, decomposed-nothing, constant column.
    small_phi = quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0])
        mu = a .+ b1 .* x1 .+ b2 .* x2
        r2d2(mu, R2, phi)
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    @test_throws ContractValidationError bind_data(
        lower_rkppl(small_phi, Set([:x1, :x2, :y])), cols)
    icpt = quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0])
        mu = a .+ b1 .* x1
        b1 ~ Normal(0.0, 1.0)
        r2d2(mu, R2, phi)
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    @test_throws ContractValidationError bind_data(
        lower_rkppl(icpt, Set([:x1, :x2, :y])), cols)
    flat = Dict{Symbol,AbstractVector}(:x1 => ones(4),
        :x2 => [1.0, 0.5, -0.5, 2.0], :y => [1.0, 2.0, 1.5, 2.5])
    @test_throws ContractValidationError bind_data(
        lower_rkppl(_r2d2_demo(), Set([:x1, :x2, :y])), flat)
    # Monotonic columns are out of the flat slice.
    momodel = quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        s ~ Dirichlet([1.0, 1.0])
        mu = a .+ b1 .* x1 .+ b3 .* mo(c, s)
        r2d2(mu, R2, phi)
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    @test_throws SurfaceLoweringError lower_rkppl(momodel,
        Set([:x1, :x2, :c, :y]))
end
