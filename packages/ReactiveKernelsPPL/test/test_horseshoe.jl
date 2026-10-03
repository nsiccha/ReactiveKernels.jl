# Per-coefficient horseshoe shrinkage prior (SB `_sb_horseshoe[_scaled]`
# mirror): beta = raw * lambda * tau with raw ~ N(0,1) and Stan-kernel
# half-Cauchy local/global scales (no truncation renormalizer — SB never
# renormalizes bounds), one triple per coefficient.
using Distributions: Normal, Cauchy, Exponential, logpdf
using Test

function _horseshoe_demo()
    return quote
        a ~ Normal(0, 1)
        b1 ~ Horseshoe()
        b2 ~ Horseshoe(local_scale = 0.5, global_scale = 0.25)
        mu = a .+ b1 .* x1 .+ b2 .* x2
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
end

function _horseshoe_cols()
    return Dict{Symbol,AbstractVector}(
        :x1 => [0.5, -1.0, 1.5, 0.0],
        :x2 => [1.0, 0.5, -0.5, 2.0],
        :y => [1.0, 2.0, 1.5, 2.5])
end

@testset "horseshoe surface admission" begin
    plan = lower_rkppl(_horseshoe_demo(), Set([:x1, :x2, :y]); conditioned = Set([:x1, :x2, :y]))
    @test length(plan.horseshoe_priors) == 2
    by_addr = Dict(h.addressee => h for h in plan.horseshoe_priors)
    @test by_addr[:x1].predictor === :mu
    @test by_addr[:x1].local_scale == 1.0
    @test by_addr[:x1].global_scale == 1.0
    @test by_addr[:x2].predictor === :mu
    @test by_addr[:x2].local_scale == 0.5
    @test by_addr[:x2].global_scale == 0.25
    @test isempty(plan.population_priors) # coverage moves to triples + scalars
    got = Dict(p.name => p for p in plan.parameters)
    # Unstated intercept rides a Normal scalar (the default prior).
    icpt = got[:horseshoe_mu_Intercept_normal]
    @test icpt.family === :normal && icpt.args == (arg1 = 0.0, arg2 = 1.0)
    @test icpt.support_override === nothing
    for (addr, ls, gs) in ((:x1, 1.0, 1.0), (:x2, 0.5, 0.25))
        raw = got[Symbol(:horseshoe_mu_, addr, :_raw)]
        @test raw.family === :normal && raw.args == (arg1 = 0, arg2 = 1)
        @test raw.support_override === nothing
        lam = got[Symbol(:horseshoe_mu_, addr, :_lambda)]
        @test lam.family === :cauchy && lam.args == (arg1 = 0, arg2 = ls)
        @test lam.support_override === :positive
        tau = got[Symbol(:horseshoe_mu_, addr, :_tau)]
        @test tau.family === :cauchy && tau.args == (arg1 = 0, arg2 = gs)
        @test tau.support_override === :positive
    end
    # A stated Normal beside a horseshoe stays Normal (mixed predictor).
    mixed = lower_rkppl(quote
            b0 ~ Normal(0.0, 5.0)
            b1 ~ Horseshoe()
            mu = b0 .+ b1 .* x1
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, Set([:x1, :y]); conditioned = Set([:x1, :y]))
    @test length(mixed.horseshoe_priors) == 1
    @test isempty(mixed.population_priors)
    mgot = Dict(p.name => p for p in mixed.parameters)
    @test mgot[:horseshoe_mu_Intercept_normal].args == (arg1 = 0.0, arg2 = 5.0)
    # Both keyword spellings land on the entry (bare `:kw` and
    # `:parameters`-wrapped — neither may silently default).
    wrapped = lower_rkppl(quote
            a ~ Normal(0, 1)
            b1 ~ Horseshoe(; global_scale = 0.25)
            mu = a .+ b1 .* x1
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, Set([:x1, :y]); conditioned = Set([:x1, :y]))
    wentry = only(wrapped.horseshoe_priors)
    @test (wentry.local_scale, wentry.global_scale) == (1.0, 0.25)
    # Layout: the coefficient block is derived (no :coefficient entry);
    # triples + scalars lay out as sampled parameters.
    bound = bind_data(plan, _horseshoe_cols())
    built = build_kernel(bound)
    kinds = [(e.kind, e.name) for e in built.layout.entries]
    @test all(k -> k[1] !== :coefficient, kinds)
    @test coordinate_names(built.layout) == [
        :sigma, :horseshoe_mu_Intercept_normal,
        :horseshoe_mu_x1_raw, :horseshoe_mu_x1_lambda, :horseshoe_mu_x1_tau,
        :horseshoe_mu_x2_raw, :horseshoe_mu_x2_lambda, :horseshoe_mu_x2_tau]
    @test built.layout.total == 8
end

@testset "horseshoe e2e values and gradient" begin
    bound = bind_data(lower_rkppl(_horseshoe_demo(), Set([:x1, :x2, :y]); conditioned = Set([:x1, :x2, :y])),
        _horseshoe_cols())
    built = build_kernel(bound)
    # Layout order (pinned above): sigma, intercept scalar, x1 triple, x2 triple.
    u = [0.5, 0.1, -0.2, 0.3, 0.4, 0.15, -0.35, 0.25]
    sig = exp(u[1])
    icpt = u[2]
    raw1, lam1, tau1 = u[3], exp(u[4]), exp(u[5])
    raw2, lam2, tau2 = u[6], exp(u[7]), exp(u[8])
    b1 = raw1 * lam1 * tau1
    b2 = raw2 * lam2 * tau2
    cols = _horseshoe_cols()
    mu = icpt .+ b1 .* cols[:x1] .+ b2 .* cols[:x2]
    ll = sum(logpdf.(Normal.(mu, sig), cols[:y]))
    pr = logpdf(Normal(0, 1), icpt) +
        logpdf(Normal(0, 1), raw1) +
        logpdf(truncated(Cauchy(0, 1), 0, Inf), lam1) +
        logpdf(truncated(Cauchy(0, 1), 0, Inf), tau1) +
        logpdf(Normal(0, 1), raw2) +
        logpdf(truncated(Cauchy(0, 0.5), 0, Inf), lam2) +
        logpdf(truncated(Cauchy(0, 0.25), 0, Inf), tau2) +
        logpdf(Exponential(1), sig)
    jac = u[1] + u[4] + u[5] + u[7] + u[8]
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "horseshoe negative use carries the sign" begin
    bound = bind_data(lower_rkppl(quote
                a ~ Normal(0, 1)
                b1 ~ Horseshoe()
                mu = a .- b1 .* x1
                sigma ~ Exponential(1.0)
                y .~ Normal.(mu, sigma)
            end, Set([:x1, :y]); conditioned = Set([:x1, :y])),
        Dict{Symbol,AbstractVector}(:x1 => [0.5, -1.0, 1.5, 0.0],
            :y => [1.0, 2.0, 1.5, 2.5]))
    entry = only(bound.horseshoe_priors)
    @test entry.sign == -1
    built = build_kernel(bound)
    # Layout: sigma, intercept scalar, x1 triple.
    u = [0.5, 0.1, -0.2, 0.3, 0.4]
    sig = exp(u[1])
    b1 = -(u[3] * exp(u[4]) * exp(u[5]))
    cols = Dict{Symbol,AbstractVector}(:x1 => [0.5, -1.0, 1.5, 0.0],
        :y => [1.0, 2.0, 1.5, 2.5])
    mu = u[2] .+ b1 .* cols[:x1]
    ll = sum(logpdf.(Normal.(mu, sig), cols[:y]))
    pr = logpdf(Normal(0, 1), u[2]) + logpdf(Normal(0, 1), u[3]) +
        logpdf(truncated(Cauchy(0, 1),0,Inf), exp(u[4])) +
        logpdf(truncated(Cauchy(0, 1),0,Inf), exp(u[5])) +
        logpdf(Exponential(1), sig)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[1] + u[4] + u[5]
    _check_gradient(built.spec, bound, u)
end

@testset "horseshoe fail-closed battery" begin
    cols = _horseshoe_cols()
    # Retire the flat-slice pins with explicit libraries (user 1cmodra /
    # 10ldrvz; follow-up 17gw1zm). The companion test_shrinkage_values
    # checks their density, transforms, coordinates and reverse derivatives.
    for kind in (:indexed, :monotonic)
        case = _sh_case(kind, 7, 3)
        @test (lower_rkppl(case.ast, case.data; conditioned = (:y,)); true)
    end
    # One structured prior per predictor.
    both = quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        b1 ~ Horseshoe()
        mu = a .+ b1 .* x1 .+ b2 .* x2
        r2d2(mu, R2, phi)
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    # refused: a Horseshoe prior plus an r2d2 decomposition on one coefficient is a double prior
    @test_throws SurfaceLoweringError lower_rkppl(both, Set([:x1, :x2, :y]); conditioned = Set([:x1, :x2, :y]))
    # Unknown arguments and invalid scale literals violate the library
    # signature/domain; Bool and sampled scales are valid numeric values.
    for rhs in (:(Horseshoe(0.5)), :(Horseshoe(scale = 0.5)),
            :(Horseshoe(local_scale = 0.0)), :(Horseshoe(global_scale = -0.1)),
            :(Horseshoe(local_scale = Inf)), :(Horseshoe(local_scale = true)),
            :(Horseshoe(local_scale = s)))
        bad = quote
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            b1 ~ $rhs
            mu = a .+ b1 .* x1
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end
        if rhs in (:(Horseshoe(local_scale = true)), :(Horseshoe(local_scale = s)))
            # capability: Bool and sampled Horseshoe scale arguments
            # (10gzbm9 bool-values; P8 1cmodra; todo `0fkd9yk`).
            @test_broken (lower_rkppl(bad, Set([:x1, :y]); conditioned = Set([:x1, :y])); true)
        else
            # refused: unknown/positional arguments or a non-finite,
            # non-positive scale literal (Julia signature/domain, P3).
            @test_throws SurfaceLoweringError lower_rkppl(bad, Set([:x1, :y]); conditioned = Set([:x1, :y]))
        end
    end
    # A horseshoe coefficient aliased as a scale stays loud (the
    # single-assignment gate, ahead of scale admission).
    aliased = quote
        a ~ Normal(0, 1)
        b1 ~ Horseshoe()
        s = b1
        mu = a .+ b1 .* x1
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, s)
    end
    # capability: a link or value that can leave a slot's support; an out-of-support value has -Inf density (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (lower_rkppl(aliased, Set([:x1, :y]); conditioned = Set([:x1, :y])); true)
    # Structural coverage (hand-built plans): exactly one prior per key,
    # triples present with half-Cauchy geometry.
    good = lower_rkppl(_horseshoe_demo(), Set([:x1, :x2, :y]); conditioned = Set([:x1, :x2, :y]))
    dup = StructuralPlan(good.responses, good.predictors,
        [PopulationPrior(:mu, :x1, 0.0, 1.0)], good.parameters,
        good.assignments, good.columns, good.n_obs;
        roles = good.roles, derived = good.derived, levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        varying_draws = good.varying_draws,
        varying_slices = good.varying_slices,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = good.kernel_plates, r2d2_priors = good.r2d2_priors,
        horseshoe_priors = good.horseshoe_priors)
    # refused: population prior duplicating a horseshoe key (IR contract)
    @test_throws ContractValidationError validate_structure(dup)
    dropped = StructuralPlan(good.responses, good.predictors,
        good.population_priors,
        [p for p in good.parameters if p.name !== :horseshoe_mu_x1_lambda],
        good.assignments, good.columns, good.n_obs;
        roles = good.roles, derived = good.derived, levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        varying_draws = good.varying_draws,
        varying_slices = good.varying_slices,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = good.kernel_plates, r2d2_priors = good.r2d2_priors,
        horseshoe_priors = good.horseshoe_priors)
    # refused: horseshoe triple missing lambda (IR contract)
    @test_throws ContractValidationError validate_structure(dropped)
end
