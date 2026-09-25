# Finite-mixture response (SB `MixtureModel` mirror): surface admission,
# fail-closed battery, value parity vs Distributions.jl oracles (all 7 v1
# families), Enzyme-vs-findiff gradients, Reactant/XLA value+grad, and
# SB-parity constants. (`_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: MixtureModel, Normal, Bernoulli, Poisson, Binomial,
    NegativeBinomial, Gamma, Beta, Exponential, LogNormal, Dirichlet, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a mixture program; return
# `(bound, built, kern, layout)`.
function _mix_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_mix_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Two-term log-sum-exp for per-row oracles (stable; no new test dep).
_mlogaddexp(a::Real, b::Real) = max(a, b) + log1p(exp(-abs(a - b)))

@testset "mixture surface admission" begin
    @testset "predictor + param locations, shared scale" begin
        plan = lower_rkppl(quote
                mu1 = a1 .+ b1 .* x
                mu2 ~ Normal(0.0, 5.0)
                sigma ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)],
                    [0.3, 0.7])
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === MixtureFam
        @test r.link === IdentityLink
        @test r.mixture_family === GaussianFam
        @test r.mixture_locs == [:mu1, :mu2]
        @test r.mixture_scales == [:sigma, :sigma]
        @test r.mixture_weights == [0.3, 0.7]
        @test r.predictor === :mu1 # Anchor: first location predictor.
        @test r.scale === nothing
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "K = 1 uses the general form" begin
        plan = lower_rkppl(quote
                mu1 ~ Normal(0.0, 5.0)
                sigma ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, sigma)], [1.0])
            end, (:y,))
        r = only(plan.responses)
        @test r.mixture_family === GaussianFam
        @test r.mixture_locs == [:mu1]
        @test r.mixture_scales == [:sigma]
        @test r.mixture_weights == [1.0]
        @test r.predictor === :mu1 # Anchor: first location param.
    end
    @testset "simplex weights link the Dirichlet vector" begin
        plan = lower_rkppl(quote
                w ~ Dirichlet([1.0, 1.0])
                y .~ MixtureModel.([Normal.(-1.0, 0.5), Normal.(1.0, 0.5)], w)
            end, (:y,))
        r = only(plan.responses)
        @test r.mixture_weights === :w
        @test r.predictor === :w # Anchor: weights simplex name.
        @test length(plan.vector_parameters) == 1
        @test plan.vector_parameters[1].family === :simplex_dirichlet
    end
    @testset "Binomial literal trials, bare probs" begin
        plan = lower_rkppl(quote
                p1 ~ Beta(2.0, 2.0)
                y .~ MixtureModel.([Binomial.(10, p1), Binomial.(10, 0.2)],
                    [0.5, 0.5])
            end, (:y,))
        r = only(plan.responses)
        @test r.mixture_family === BinomialLogitFam
        @test r.mixture_locs == [:p1, 0.2]
        @test r.trials === 10
    end
    @testset "scale-predictor anchor" begin
        plan = lower_rkppl(quote
                ls = c .+ d .* x
                mu1 ~ Normal(0.0, 5.0)
                y .~ MixtureModel.([Normal.(mu1, exp.(ls)),
                    Normal.(0.0, exp.(ls))], [0.5, 0.5])
            end, (:y, :x))
        r = only(plan.responses)
        @test r.predictor === :ls # Anchor: first scale predictor.
        @test r.mixture_scales[1] isa ScalePredictorRef
    end
    @testset "alternate heads desugar per component" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                phi ~ Exponential(1.0)
                y .~ MixtureModel.([NegativeBinomial2Log.(mu, phi),
                    NegativeBinomial2Log.(mu, phi)], [0.5, 0.5])
            end, (:y, :x))
        r = only(plan.responses)
        @test r.mixture_family === NegativeBinomial2Fam
        @test r.mixture_locs == [:mu, :mu] # Sharing interns by name.
    end
end

@testset "mixture fail-closed battery" begin
    # Each entry: (label, program, error type). Surface rejections throw
    # `SurfaceLoweringError`; contract rejections (parsed but invalid)
    # throw `ContractValidationError`.
    cases = [
        ("heterogeneous families",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                lam1 ~ LogNormal(0.0, 1.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Poisson.(lam1)],
                    [0.5, 0.5])
            end), SurfaceLoweringError),
        ("heterogeneous links, same base",
            :(begin
                eta = a .+ b .* x
                eta2 = c .+ d .* x
                y .~ MixtureModel.([Bernoulli.(logistic.(eta)),
                    Bernoulli.(probit.(eta2))], [0.5, 0.5])
            end), SurfaceLoweringError),
        ("probit outside v1",
            :(begin
                eta = a .+ b .* x
                eta2 = c .+ d .* x
                y .~ MixtureModel.([Bernoulli.(probit.(eta)),
                    Bernoulli.(probit.(eta2))], [0.5, 0.5])
            end), SurfaceLoweringError),
        ("K = 0",
            :(begin
                y .~ MixtureModel.([], [1.0])
            end), SurfaceLoweringError),
        ("components not a vector",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(Normal.(mu1, s), [1.0])
            end), SurfaceLoweringError),
        ("bad arity",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s)])
            end), SurfaceLoweringError),
        ("scalar weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s)], 1.0)
            end), SurfaceLoweringError),
        ("non-numeric weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Normal.(mu1, s)],
                    [0.5, x])
            end), SurfaceLoweringError),
        ("boolean weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Normal.(mu1, s)],
                    [true, false])
            end), SurfaceLoweringError),
        ("frequency weights rejected",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ weighted.(MixtureModel.([Normal.(mu1, s),
                    Normal.(mu1, s)], [0.5, 0.5]), wt)
            end), SurfaceLoweringError),
        ("evidence rejected",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ truncated.(MixtureModel.([Normal.(mu1, s),
                    Normal.(mu1, s)], [0.5, 0.5]), 0, 10)
            end), SurfaceLoweringError),
        ("partial range rejected",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y[1:2] .~ MixtureModel.([Normal.(mu1, s),
                    Normal.(mu1, s)], [0.5, 0.5])
            end), SurfaceLoweringError),
        ("fully fixed",
            :(begin
                y .~ MixtureModel.([Normal.(-1.0, 0.5), Normal.(1.0, 0.5)],
                    [0.4, 0.6])
            end), SurfaceLoweringError),
        ("unknown location",
            :(begin
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(nope, s), Normal.(0.0, s)],
                    [0.5, 0.5])
            end), SurfaceLoweringError),
        ("assignment location",
            :(begin
                m = 2.0
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(m, s), Normal.(0.0, s)],
                    [0.5, 0.5])
            end), SurfaceLoweringError),
        ("data-column location",
            :(begin
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(x, s), Normal.(0.0, s)],
                    [0.5, 0.5])
            end), SurfaceLoweringError),
        ("wrapped param",
            :(begin
                lam1 ~ LogNormal(0.0, 1.0)
                y .~ MixtureModel.([Poisson.(exp.(lam1)), Poisson.(4.0)],
                    [0.5, 0.5])
            end), SurfaceLoweringError),
        ("bare predictor",
            :(begin
                eta = a .+ b .* x
                y .~ MixtureModel.([Poisson.(eta), Poisson.(4.0)],
                    [0.5, 0.5])
            end), SurfaceLoweringError),
        ("unknown wrapper",
            :(begin
                eta = a .+ b .* x
                y .~ MixtureModel.([Bernoulli.(foo.(eta)),
                    Bernoulli.(0.5)], [0.5, 0.5])
            end), SurfaceLoweringError),
        ("split Binomial trials",
            :(begin
                y .~ MixtureModel.([Binomial.(n1, 0.3), Binomial.(n2, 0.3)],
                    [0.5, 0.5])
            end), SurfaceLoweringError),
        ("weights length",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Normal.(mu1, s)],
                    [0.5, 0.3, 0.2])
            end), ContractValidationError),
        ("weights sum",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Normal.(mu1, s)],
                    [0.5, 0.6])
            end), ContractValidationError),
        ("negative weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Normal.(mu1, s)],
                    [1.5, -0.5])
            end), ContractValidationError),
        ("non-literal weights element",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Normal.(mu1, s)],
                    [0.5, Inf])
            end), SurfaceLoweringError),
        ("concentration length",
            :(begin
                w ~ Dirichlet([1.0, 1.0, 1.0])
                y .~ MixtureModel.([Normal.(-1.0, 0.5), Normal.(1.0, 0.5)],
                    w)
            end), ContractValidationError),
        ("Bernoulli prob domain",
            :(begin
                eta = a .+ b .* x
                y .~ MixtureModel.([Bernoulli.(logistic.(eta)),
                    Bernoulli.(1.5)], [0.5, 0.5])
            end), ContractValidationError),
        ("Poisson mean domain",
            :(begin
                lam1 ~ LogNormal(0.0, 1.0)
                y .~ MixtureModel.([Poisson.(lam1), Poisson.(-1.0)],
                    [0.5, 0.5])
            end), ContractValidationError),
        ("Gamma mean domain",
            :(begin
                a1 ~ Exponential(1.0)
                mu1 ~ Gamma(2.0, 1.0)
                y .~ MixtureModel.([Gamma.(a1, mu1 ./ a1),
                    Gamma.(a1, 0.0 ./ a1)], [0.5, 0.5])
            end), ContractValidationError),
        ("Beta mean domain",
            :(begin
                k1 ~ Exponential(1.0)
                y .~ MixtureModel.([Beta.(0.3 .* k1, (1 .- 0.3) .* k1),
                    Beta.(1.5 .* k1, (1 .- 1.5) .* k1)], [0.5, 0.5])
            end), ContractValidationError),
        ("empty weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s)], Float64[])
            end), SurfaceLoweringError),
    ]
    for (label, prog, E) in cases
        @testset "$label" begin
            @test_throws E lower_rkppl(prog, (:y, :x, :n1, :n2, :wt))
        end
    end
    @testset "programmatic shape errors" begin
        terms = TermSpec[
            TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                :intercept),
            TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term),
        ]
        priors = PopulationPrior[
            PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x, 0.0, 1.0),
        ]
        cols = Dict{Symbol,AbstractVector}(:y => zeros(9),
            :x => collect(1.0:9))
        params = SampledParameter[
            SampledParameter(:mu2, :normal, (arg1 = 0.0, arg2 = 5.0),
                nothing, :mu2),
            SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing,
                :sigma),
        ]
        function _mixresp(; f = GaussianFam, link = IdentityLink,
                locs = Union{Symbol,Real}[:mu, :mu2],
                scales = Union{Nothing,Symbol,Real,ScalePredictorRef}[
                    :sigma, :sigma],
                weights::Union{Nothing,Symbol,Vector{Float64}} = [0.3, 0.7],
                trials = nothing)
            return LikelihoodSpec(MixtureFam, link, :y, :mu, nothing,
                nothing, ResponseEvidence(:none, nothing, nothing), :y_resp,
                trials, nothing; mixture_family = f, mixture_locs = locs,
                mixture_scales = scales, mixture_weights = weights)
        end
        _mixplan(r) = StructuralPlan([r],
            [PredictorSpec(:mu, IdentityLink, terms, :mu)], priors, params,
            AssignmentSpec[], cols, 9)
        # Scale slots must match locations one-to-one.
        bad = _mixresp(scales = Union{Nothing,Symbol,Real,ScalePredictorRef}[
            :sigma])
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # The response link is the components' canonical link.
        bad = _mixresp(link = LogLink)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # Binomial mixtures require trials (surface always emits them, so
        # this is programmatic-only).
        bloc = Union{Symbol,Real}[:mu, :mu2]
        bsc = Union{Nothing,Symbol,Real,ScalePredictorRef}[nothing, nothing]
        bad = _mixresp(f = BinomialLogitFam, link = LogitLink, locs = bloc,
            scales = bsc, trials = nothing)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # ... and non-Binomial mixtures take none.
        bad = _mixresp(trials = 10)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # Non-finite weights (surface literals are finite by
        # construction, so this is programmatic-only).
        bad = _mixresp(weights = [0.5, Inf])
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # A non-simplex weights vector fails the family check.
        r = _mixresp(weights = :z)
        plan = StructuralPlan([r],
            [PredictorSpec(:mu, IdentityLink, terms, :mu)], priors, params,
            AssignmentSpec[], cols, 9; vector_parameters = VectorParameter[
                VectorParameter(:z, :vector_normal, (arg1 = [0.0, 0.0],
                    arg2 = [1.0, 1.0]), 2, :z)])
        @test_throws ContractValidationError validate_structure(plan)
        # And the valid programmatic shape passes.
        @test (validate_structure(_mixplan(_mixresp())); true)
    end
end

@testset "mixture value parity" begin
    @testset "gaussian params (driving case)" begin
        prog = quote
            mu1 ~ Normal(-2.0, 0.1)
            mu2 ~ Normal(2.0, 0.1)
            sigma ~ Exponential(1.0)
            y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)],
                [0.4, 0.6])
        end
        cols = Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2])
        _, _, kern, lay = _mix_query(prog, cols)
        @test coordinate_names(lay) == [:mu1, :mu2, :sigma]
        for q in [(mu1 = -2.0, mu2 = 2.0, sigma = 0.3),
            (mu1 = -1.5, mu2 = 1.0, sigma = 1.2)]
            got = _mix_posterior(kern, lay, q)
            mix = MixtureModel([Normal(q.mu1, q.sigma), Normal(q.mu2, q.sigma)],
                [0.4, 0.6])
            want = sum(logpdf.(mix, cols[:y])) +
                logpdf(Normal(-2, 0.1), q.mu1) +
                logpdf(Normal(2, 0.1), q.mu2) +
                logpdf(Exponential(1.0), q.sigma) + log(q.sigma)
            @test got ≈ want rtol = 1e-12
        end
    end
    @testset "gaussian predictor + param" begin
        prog = quote
            mu1 = a1 .+ b1 .* x
            mu2 ~ Normal(0.0, 5.0)
            sigma ~ Exponential(1.0)
            y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)],
                [0.3, 0.7])
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 0.5, -1.0],
            :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (mu1 = [0.5, -0.25], mu2 = 1.0, sigma = 1.5)
        got = _mix_posterior(kern, lay, q)
        eta = q.mu1[1] .+ q.mu1[2] .* x
        ll = sum(zip(cols[:y], eta)) do (yi, etai)
            _mlogaddexp(log(0.3) + logpdf(Normal(etai, q.sigma), yi),
                log(0.7) + logpdf(Normal(q.mu2, q.sigma), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), q.mu1)) +
            logpdf(Normal(0, 5), q.mu2) +
            logpdf(Exponential(1.0), q.sigma) + log(q.sigma)
        @test got ≈ want rtol = 1e-12
    end
    @testset "bernoulli predictor + literal" begin
        for ycol in ([0, 1, 1, 0], [false, true, true, false])
            prog = quote
                eta = a .+ b .* x
                y .~ MixtureModel.([Bernoulli.(logistic.(eta)),
                    Bernoulli.(0.7)], [0.5, 0.5])
            end
            x = [0.5, -1.0, 1.5, 0.0]
            cols = Dict{Symbol,AbstractVector}(:y => ycol, :x => x)
            _, _, kern, lay = _mix_query(prog, cols)
            q = (eta = [0.2, -0.4],)
            got = _mix_posterior(kern, lay, q)
            eta = q.eta[1] .+ q.eta[2] .* x
            p1 = 1 ./ (1 .+ exp.(-eta))
            ll = sum(zip(ycol, p1)) do (yi, pi)
                _mlogaddexp(log(0.5) + logpdf(Bernoulli(pi), yi),
                    log(0.5) + logpdf(Bernoulli(0.7), yi))
            end
            want = ll + sum(logpdf.(Normal(0, 1), q.eta))
            @test got ≈ want rtol = 1e-12
        end
    end
    @testset "poisson param + literal" begin
        prog = quote
            lam1 ~ LogNormal(0.0, 1.0)
            y .~ MixtureModel.([Poisson.(lam1), Poisson.(4.0)], [0.3, 0.7])
        end
        cols = Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2])
        _, _, kern, lay = _mix_query(prog, cols)
        for lam in (1.5, 4.0)
            q = (lam1 = lam,)
            got = _mix_posterior(kern, lay, q)
            mix = MixtureModel([Poisson(lam), Poisson(4.0)], [0.3, 0.7])
            want = sum(logpdf.(mix, cols[:y])) +
                logpdf(LogNormal(0, 1), lam) + log(lam)
            @test got ≈ want rtol = 1e-12
        end
    end
    @testset "binomial predictor + literal, column trials" begin
        prog = quote
            eta = a .+ b .* x
            y .~ MixtureModel.([Binomial.(n, logistic.(eta)),
                Binomial.(n, 0.25)], [0.6, 0.4])
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [3, 7, 5, 8],
            :x => x, :n => [10, 10, 10, 10])
        _, _, kern, lay = _mix_query(prog, cols)
        q = (eta = [0.2, -0.4],)
        got = _mix_posterior(kern, lay, q)
        eta = q.eta[1] .+ q.eta[2] .* x
        p1 = 1 ./ (1 .+ exp.(-eta))
        ll = sum(zip(cols[:y], p1)) do (yi, pi)
            _mlogaddexp(log(0.6) + logpdf(Binomial(10, pi), yi),
                log(0.4) + logpdf(Binomial(10, 0.25), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), q.eta))
        @test got ≈ want rtol = 1e-12
    end
    @testset "nb2 predictor + param means" begin
        prog = quote
            eta = a .+ b .* x
            mu2 ~ Gamma(2.0, 1.0)
            phi1 ~ Exponential(1.0)
            phi2 ~ Exponential(1.0)
            y .~ MixtureModel.([NegativeBinomial2.(exp.(eta), phi1),
                NegativeBinomial2.(mu2, phi2)], [0.5, 0.5])
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [2, 0, 4, 1], :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (eta = [0.3, 0.1], mu2 = 2.0, phi1 = 1.5, phi2 = 0.5)
        got = _mix_posterior(kern, lay, q)
        mu1 = exp.(q.eta[1] .+ q.eta[2] .* x)
        nb(mu, phi, y) = logpdf(NegativeBinomial(phi, phi / (phi + mu)), y)
        ll = sum(zip(cols[:y], mu1)) do (yi, m1)
            _mlogaddexp(log(0.5) + nb(m1, q.phi1, yi),
                log(0.5) + nb(q.mu2, q.phi2, yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), q.eta)) +
            logpdf(Gamma(2, 1), q.mu2) + log(q.mu2) +
            logpdf(Exponential(1.0), q.phi1) + log(q.phi1) +
            logpdf(Exponential(1.0), q.phi2) + log(q.phi2)
        @test got ≈ want rtol = 1e-11
    end
    @testset "gamma predictor + param means, shared shape" begin
        prog = quote
            eta = a .+ b .* x
            alpha ~ Exponential(1.0)
            mu2 ~ Gamma(2.0, 1.0)
            y .~ MixtureModel.([Gamma.(alpha, exp.(eta) ./ alpha),
                Gamma.(alpha, mu2 ./ alpha)], [0.5, 0.5])
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [1.5, 0.5, 2.0, 1.0],
            :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (eta = [0.2, -0.1], alpha = 2.0, mu2 = 1.5)
        got = _mix_posterior(kern, lay, q)
        mu1 = exp.(q.eta[1] .+ q.eta[2] .* x)
        ll = sum(zip(cols[:y], mu1)) do (yi, m1)
            _mlogaddexp(log(0.5) + logpdf(Gamma(q.alpha, m1 / q.alpha), yi),
                log(0.5) + logpdf(Gamma(q.alpha, q.mu2 / q.alpha), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), q.eta)) +
            logpdf(Exponential(1.0), q.alpha) + log(q.alpha) +
            logpdf(Gamma(2, 1), q.mu2) + log(q.mu2)
        @test got ≈ want rtol = 1e-12
    end
    @testset "beta predictor + literal means, shared kappa" begin
        prog = quote
            eta = a .+ b .* x
            kappa ~ Exponential(1.0)
            y .~ MixtureModel.([Beta.(logistic.(eta) .* kappa,
                    (1 .- logistic.(eta)) .* kappa),
                Beta.(0.7 .* kappa, (1 .- 0.7) .* kappa)], [0.5, 0.5])
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [0.2, 0.8, 0.4, 0.6],
            :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (eta = [0.2, -0.4], kappa = 5.0)
        got = _mix_posterior(kern, lay, q)
        mu1 = 1 ./ (1 .+ exp.(-(q.eta[1] .+ q.eta[2] .* x)))
        ll = sum(zip(cols[:y], mu1)) do (yi, m1)
            _mlogaddexp(log(0.5) + logpdf(Beta(m1 * q.kappa,
                        (1 - m1) * q.kappa), yi),
                log(0.5) + logpdf(Beta(0.7 * q.kappa, 0.3 * q.kappa), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), q.eta)) +
            logpdf(Exponential(1.0), q.kappa) + log(q.kappa)
        @test got ≈ want rtol = 1e-12
    end
    @testset "K = 1 equals the single component" begin
        prog = quote
            mu1 ~ Normal(-2.0, 0.1)
            sigma ~ Exponential(1.0)
            y .~ MixtureModel.([Normal.(mu1, sigma)], [1.0])
        end
        cols = Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2])
        _, _, kern, lay = _mix_query(prog, cols)
        q = (mu1 = -2.0, sigma = 0.3)
        got = _mix_posterior(kern, lay, q)
        want = sum(logpdf.(Normal(q.mu1, q.sigma), cols[:y])) +
            logpdf(Normal(-2, 0.1), q.mu1) +
            logpdf(Exponential(1.0), q.sigma) + log(q.sigma)
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed — this leg also covers simplex weights, whose stick-breaking
# Jacobian has no hand oracle here).
function _mix_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _mix_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "mixture Enzyme gradients" begin
    x = [0.5, -1.0, 1.5, 0.0]
    @testset "gaussian" begin
        _mix_enzyme_check(quote
                mu1 ~ Normal(-2.0, 0.1)
                mu2 ~ Normal(2.0, 0.1)
                sigma ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)],
                    [0.4, 0.6])
            end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]),
            (mu1 = -2.0, mu2 = 2.0, sigma = 0.3))
    end
    @testset "bernoulli" begin
        _mix_enzyme_check(quote
                eta = a .+ b .* x
                y .~ MixtureModel.([Bernoulli.(logistic.(eta)),
                    Bernoulli.(0.7)], [0.5, 0.5])
            end, Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0], :x => x),
            (eta = [0.2, -0.4],))
    end
    @testset "poisson" begin
        _mix_enzyme_check(quote
                lam1 ~ LogNormal(0.0, 1.0)
                y .~ MixtureModel.([Poisson.(lam1), Poisson.(4.0)],
                    [0.3, 0.7])
            end, Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2]),
            (lam1 = 1.5,))
    end
    @testset "binomial" begin
        _mix_enzyme_check(quote
                eta = a .+ b .* x
                y .~ MixtureModel.([Binomial.(n, logistic.(eta)),
                    Binomial.(n, 0.25)], [0.6, 0.4])
            end, Dict{Symbol,AbstractVector}(:y => [3, 7, 5, 8], :x => x,
                :n => [10, 10, 10, 10]), (eta = [0.2, -0.4],))
    end
    @testset "nb2" begin
        _mix_enzyme_check(quote
                eta = a .+ b .* x
                mu2 ~ Gamma(2.0, 1.0)
                phi1 ~ Exponential(1.0)
                phi2 ~ Exponential(1.0)
                y .~ MixtureModel.([NegativeBinomial2.(exp.(eta), phi1),
                    NegativeBinomial2.(mu2, phi2)], [0.5, 0.5])
            end, Dict{Symbol,AbstractVector}(:y => [2, 0, 4, 1], :x => x),
            (eta = [0.3, 0.1], mu2 = 2.0, phi1 = 1.5, phi2 = 0.5))
    end
    @testset "gamma" begin
        _mix_enzyme_check(quote
                eta = a .+ b .* x
                alpha ~ Exponential(1.0)
                mu2 ~ Gamma(2.0, 1.0)
                y .~ MixtureModel.([Gamma.(alpha, exp.(eta) ./ alpha),
                    Gamma.(alpha, mu2 ./ alpha)], [0.5, 0.5])
            end, Dict{Symbol,AbstractVector}(:y => [1.5, 0.5, 2.0, 1.0],
                :x => x), (eta = [0.2, -0.1], alpha = 2.0, mu2 = 1.5))
    end
    @testset "beta" begin
        _mix_enzyme_check(quote
                eta = a .+ b .* x
                kappa ~ Exponential(1.0)
                y .~ MixtureModel.([Beta.(logistic.(eta) .* kappa,
                        (1 .- logistic.(eta)) .* kappa),
                    Beta.(0.7 .* kappa, (1 .- 0.7) .* kappa)], [0.5, 0.5])
            end, Dict{Symbol,AbstractVector}(:y => [0.2, 0.8, 0.4, 0.6],
                :x => x), (eta = [0.2, -0.4], kappa = 5.0))
    end
    @testset "simplex weights" begin
        _mix_enzyme_check(quote
                w ~ Dirichlet([1.0, 1.0])
                mu1 ~ Normal(-2.0, 0.1)
                mu2 ~ Normal(2.0, 0.1)
                sigma ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)],
                    w)
            end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]),
            (w = [0.4, 0.6], mu1 = -2.0, mu2 = 2.0, sigma = 0.3))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the mixture plate must not unroll over observations.
function _mix_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        heads[string(st.head)] = get(heads, string(st.head), 0) + 1
    end
    return heads
end

@testset "mixture emission is O(1) in n_obs" begin
    prog = quote
        mu1 ~ Normal(-2.0, 0.1)
        mu2 ~ Normal(2.0, 0.1)
        sigma ~ Exponential(1.0)
        y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)],
            [0.4, 0.6])
    end
    h4 = _mix_statement_heads(prog,
        Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]))
    h8 = _mix_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => [-2.0, -1.8, 1.9, 2.2, -2.1, -1.9, 2.0, 2.1]))
    @test h4 == h8
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _mix_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_mix_reactant_measure, built, bound, post_q, u)
end

function _mix_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "mixture under Reactant" begin
    x = [0.5, -1.0, 1.5, 0.0]
    progs = [
        ("gaussian", quote
            mu1 ~ Normal(-2.0, 0.1)
            mu2 ~ Normal(2.0, 0.1)
            sigma ~ Exponential(1.0)
            y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)],
                [0.4, 0.6])
        end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2])),
        ("poisson", quote
            lam1 ~ LogNormal(0.0, 1.0)
            y .~ MixtureModel.([Poisson.(lam1), Poisson.(4.0)], [0.3, 0.7])
        end, Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2])),
        ("bernoulli", quote
            eta = a .+ b .* x
            y .~ MixtureModel.([Bernoulli.(logistic.(eta)),
                Bernoulli.(0.7)], [0.5, 0.5])
        end, Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0], :x => x)),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _mix_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[1]
        small = _mix_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]))
        large = _mix_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => [-2.0, -1.8, 1.9, 2.2, -2.1, -1.9, 2.0, 2.1]))
        @test small.lines == large.lines
    end
end
