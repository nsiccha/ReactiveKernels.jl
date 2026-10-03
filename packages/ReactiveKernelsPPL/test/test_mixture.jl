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
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
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
                a1 ~ Normal(0, 1)
                b1 ~ Normal(0, 1)
                mu1 = a1 .+ b1 .* x
                mu2 ~ Normal(0.0, 5.0)
                sigma ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.3, 0.7]))
            end, (:y, :x); conditioned = (:y, :x))
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
                y .~ MixtureModel.(vcat.(Normal.(mu1, sigma)), Ref([1.0]))
            end, (:y,); conditioned = (:y,))
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
                y .~ MixtureModel.(vcat.(Normal.(-1.0, 0.5), Normal.(1.0, 0.5)), Ref(w))
            end, (:y,); conditioned = (:y,))
        r = only(plan.responses)
        @test r.mixture_weights === :w
        @test r.predictor === :w # Anchor: weights simplex name.
        @test length(plan.vector_parameters) == 1
        @test plan.vector_parameters[1].family === :simplex_dirichlet
    end
    @testset "Binomial literal trials, bare probs" begin
        plan = lower_rkppl(quote
                p1 ~ Beta(2.0, 2.0)
                y .~ MixtureModel.(vcat.(Binomial.(10, p1), Binomial.(10, 0.2)), Ref([0.5, 0.5]))
            end, (:y,); conditioned = (:y,))
        r = only(plan.responses)
        @test r.mixture_family === BinomialLogitFam
        @test r.mixture_locs == [:p1, 0.2]
        @test r.trials === 10
    end
    @testset "scale-predictor anchor" begin
        plan = lower_rkppl(quote
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                ls = c .+ d .* x
                mu1 ~ Normal(0.0, 5.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, exp.(ls)),
                    Normal.(0.0, exp.(ls))), Ref([0.5, 0.5]))
            end, (:y, :x); conditioned = (:y, :x))
        r = only(plan.responses)
        @test r.predictor === :ls # Anchor: first scale predictor.
        @test r.mixture_scales[1] isa ScalePredictorRef
    end
    @testset "intercept-only scale predictor (SB log(sigma) ~ 1)" begin
        plan = lower_rkppl(quote
                c ~ Normal(0, 1)
                mu1 ~ Normal(-2.0, 0.1)
                mu2 ~ Normal(2.0, 0.1)
                sigma = c
                y .~ MixtureModel.(vcat.(Normal.(mu1, exp.(sigma)),
                    Normal.(mu2, exp.(sigma))), Ref([0.4, 0.6]))
            end, (:y,); conditioned = (:y,))
        r = only(plan.responses)
        @test r.mixture_scales == [ScalePredictorRef(:sigma, LogLink),
            ScalePredictorRef(:sigma, LogLink)]
        @test r.predictor === :sigma # Anchor: first scale predictor.
        pred = only(p for p in plan.predictors if p.name === :sigma)
        @test [t.kind for t in pred.terms] == [InterceptTerm]
        @test isempty(plan.derived)
    end
    @testset "constant location is a declared parameter (SB mu ~ 1)" begin
        # Strict declarations: the intercept-only location slot (an
        # undeclared `mu1 = c`) is gone. A constant location is a declared
        # scalar spelled bare; an alias of one fails (the battery's
        # "stated scalar loc alias").
        plan = lower_rkppl(quote
                mu1 ~ Normal(0, 1)
                mu2 ~ Normal(0.0, 5.0)
                sigma ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, sigma),
                    Normal.(mu2, sigma)), Ref([0.3, 0.7]))
            end, (:y,); conditioned = (:y,))
        r = only(plan.responses)
        @test r.mixture_locs == [:mu1, :mu2]
        @test r.predictor === :mu1 # Anchor: first location.
        @test isempty(plan.predictors)
        @test [q.name for q in plan.parameters] == [:mu1, :mu2, :sigma]
        @test isempty(plan.derived)
    end
    @testset "alternate heads desugar per component" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                phi ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(NegativeBinomial2Log.(mu, phi),
                    NegativeBinomial2Log.(mu, phi)), Ref([0.5, 0.5]))
            end, (:y, :x); conditioned = (:y, :x))
        r = only(plan.responses)
        @test r.mixture_family === NegativeBinomial2Fam
        @test r.mixture_locs == [:mu, :mu] # Sharing interns by name.
    end
end

@testset "mixture refusals and capability gaps" begin
    # Each entry: (label, program, error type). Surface rejections throw
    # `SurfaceLoweringError`; contract rejections (parsed but invalid)
    # throw `ContractValidationError`.
    cases = [
        # refused: mixture of a continuous and a discrete component has no common density (mathematically invalid)
        ("heterogeneous families",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                lam1 ~ LogNormal(0.0, 1.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Poisson.(lam1)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: per-component links (todo `1nb43fj`).
        ("heterogeneous links, same base",
            :(begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                eta = a .+ b .* x
                eta2 = c .+ d .* x
                y .~ MixtureModel.(vcat.(Bernoulli.(logistic.(eta)),
                    Bernoulli.(normcdf.(eta2))), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: probit component links (todo `1nb43fj`).
        ("probit outside v1",
            :(begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                eta = a .+ b .* x
                eta2 = c .+ d .* x
                y .~ MixtureModel.(vcat.(Bernoulli.(normcdf.(eta)),
                    Bernoulli.(normcdf.(eta2))), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # refused: empty mixture (K = 0)
        ("K = 0",
            :(begin
                y .~ MixtureModel.(vcat.(), Ref([1.0]))
            end), SurfaceLoweringError),
        # refused: MixtureModel.(Normal.(...), Ref([1.0])) broadcasts MixtureModel(::Normal, ::Float64), a Julia MethodError (P3)
        ("components not a vector",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(Normal.(mu1, s), Ref([1.0]))
            end), SurfaceLoweringError),
        # admitted: MixtureModel(components) with its standard uniform weights (todo `139j2uo`).
        ("uniform default weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s)))
            end), SurfaceLoweringError),
        # refused: scalar mixture prior is a Julia MethodError; weights are a probability vector (P3)
        ("scalar weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s)), Ref(1.0))
            end), SurfaceLoweringError),
        # refused: [0.5, x] is a Vector{Any} of a scalar and a data vector, not a probability vector (P3)
        ("non-numeric weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu1, s)), Ref([0.5, x]))
            end), SurfaceLoweringError),
        # capability: Bool numeric weights (10gzbm9 bool-values) (todo `1nb43fj`).
        ("boolean weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu1, s)), Ref([true, false]))
            end), SurfaceLoweringError),
        # capability: frequency-weighted mixture observations (todo `1nb43fj`).
        ("frequency weights rejected",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ weighted.(MixtureModel.(vcat.(Normal.(mu1, s),
                    Normal.(mu1, s)), Ref([0.5, 0.5])), wt)
            end), SurfaceLoweringError),
        # capability: truncated mixture evidence (todo `1nb43fj`).
        ("evidence rejected",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ truncated.(MixtureModel.(vcat.(Normal.(mu1, s),
                    Normal.(mu1, s)), Ref([0.5, 0.5])), 0, 10)
            end), SurfaceLoweringError),
        # capability: a partial observation range (todo `1nb43fj`).
        ("partial range rejected",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y[1:2] .~ MixtureModel.(vcat.(Normal.(mu1, s),
                    Normal.(mu1, s)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: parameter-free observations (10gzbm9 degenerate) (todo `1nb43fj`).
        ("fully fixed",
            :(begin
                y .~ MixtureModel.(vcat.(Normal.(-1.0, 0.5), Normal.(1.0, 0.5)), Ref([0.4, 0.6]))
            end), SurfaceLoweringError),
        # refused: undeclared name (P6, 05oe96l)
        ("unknown location",
            :(begin
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(nope, s), Normal.(0.0, s)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: an assigned scalar component location (todo `1nb43fj`).
        ("assignment location",
            :(begin
                m = 2.0
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(m, s), Normal.(0.0, s)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: a data-valued component location (todo `1nb43fj`).
        ("data-column location",
            :(begin
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(x, s), Normal.(0.0, s)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: a computed component location (P8 1cmodra) (todo `1nb43fj`).
        ("wrapped param",
            :(begin
                lam1 ~ LogNormal(0.0, 1.0)
                y .~ MixtureModel.(vcat.(Poisson.(exp.(lam1)), Poisson.(4.0)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: an unconstrained component rate, with -Inf outside support (10gzbm9 support-links) (todo `1nb43fj`).
        ("bare predictor",
            :(begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ MixtureModel.(vcat.(Poisson.(eta), Poisson.(4.0)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # refused: foo is undefined (UndefVarError in Julia)
        ("unknown wrapper",
            :(begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ MixtureModel.(vcat.(Bernoulli.(foo.(eta)),
                    Bernoulli.(0.5)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # capability: per-component trials columns (todo `1nb43fj`).
        ("split Binomial trials",
            :(begin
                y .~ MixtureModel.(vcat.(Binomial.(n1, 0.3), Binomial.(n2, 0.3)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
        # refused: weights length mismatches component count
        ("weights length",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu1, s)), Ref([0.5, 0.3, 0.2]))
            end), ContractValidationError),
        # refused: mixture probabilities do not sum to 1
        ("weights sum",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu1, s)), Ref([0.5, 0.6]))
            end), ContractValidationError),
        # refused: negative mixture weight
        ("negative weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu1, s)), Ref([1.5, -0.5]))
            end), ContractValidationError),
        # refused: non-finite (Inf) mixture weight
        ("non-literal weights element",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu1, s)), Ref([0.5, Inf]))
            end), SurfaceLoweringError),
        # refused: Dirichlet length mismatches component count
        ("concentration length",
            :(begin
                w ~ Dirichlet([1.0, 1.0, 1.0])
                y .~ MixtureModel.(vcat.(Normal.(-1.0, 0.5), Normal.(1.0, 0.5)), Ref(w))
            end), ContractValidationError),
        # refused: Bernoulli probability 1.5 outside [0,1]
        ("Bernoulli prob domain",
            :(begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ MixtureModel.(vcat.(Bernoulli.(logistic.(eta)),
                    Bernoulli.(1.5)), Ref([0.5, 0.5]))
            end), ContractValidationError),
        # refused: negative Poisson mean
        ("Poisson mean domain",
            :(begin
                lam1 ~ LogNormal(0.0, 1.0)
                y .~ MixtureModel.(vcat.(Poisson.(lam1), Poisson.(-1.0)), Ref([0.5, 0.5]))
            end), ContractValidationError),
        # refused: zero Gamma scale (malformed distribution)
        ("Gamma mean domain",
            :(begin
                a1 ~ Exponential(1.0)
                mu1 ~ Gamma(2.0, 1.0)
                y .~ MixtureModel.(vcat.(Gamma.(a1, mu1 ./ a1),
                    Gamma.(a1, 0.0 ./ a1)), Ref([0.5, 0.5]))
            end), ContractValidationError),
        # refused: negative Beta shape (malformed distribution)
        ("Beta mean domain",
            :(begin
                k1 ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Beta.(0.3 .* k1, (1 .- 0.3) .* k1),
                    Beta.(1.5 .* k1, (1 .- 1.5) .* k1)), Ref([0.5, 0.5]))
            end), ContractValidationError),
        # refused: empty mixture weights
        ("empty weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s)), Ref(Float64[]))
            end), SurfaceLoweringError),
        # A scalar alias over a stated prior reads like the name itself
        # (a sampled parameter — spell it bare). It never routes to the
        # intercept-only location slot, which only an undeclared name
        # reached (gone under strict declarations).
        # capability: an alias of a declared scalar location (todo `1nb43fj`).
        ("stated scalar loc alias",
            :(begin
                c ~ Normal(0.0, 5.0)
                mu1 = c
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu1, s)), Ref([0.5, 0.5]))
            end), SurfaceLoweringError),
    ]
    supported = Set(["frequency weights rejected", "evidence rejected",
        "partial range rejected", "fully fixed", "assignment location",
        "data-column location", "split Binomial trials", "stated scalar loc alias",
        "uniform default weights", "wrapped param"])
    capabilities = Set(["heterogeneous links, same base", "probit outside v1", "boolean weights", "bare predictor"])
    for (label, prog, E) in cases
        @testset "$label" begin
            if label in supported
                # Density and gradient oracles: test_response_combinations.jl.
                @test lower_rkppl(prog, (:y, :x, :n1, :n2, :wt);
                    conditioned = (:y, :x, :n1, :n2, :wt)) isa StructuralPlan
            elseif label in capabilities
                # capability: each entry above names the valid combination (todo `1nb43fj`).
                @test_broken (lower_rkppl(prog, (:y, :x, :n1, :n2, :wt); conditioned = (:y, :x, :n1, :n2, :wt)); true)
            else
                # refused: each remaining entry cites its Julia signature,
                # declaration or distribution-domain violation above (P3/P6).
                @test_throws E lower_rkppl(prog, (:y, :x, :n1, :n2, :wt); conditioned = (:y, :x, :n1, :n2, :wt))
            end
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
        # refused: scale slots must match locations one-to-one (IR contract)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # The response link is the components' canonical link.
        bad = _mixresp(link = LogLink)
        # refused: mixture response link must be the components' canonical link (IR contract)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # Binomial mixtures require trials (surface always emits them, so
        # this is programmatic-only).
        bloc = Union{Symbol,Real}[:mu, :mu2]
        bsc = Union{Nothing,Symbol,Real,ScalePredictorRef}[nothing, nothing]
        bad = _mixresp(f = BinomialLogitFam, link = LogitLink, locs = bloc,
            scales = bsc, trials = nothing)
        # refused: Binomial mixture without trials (IR contract)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # ... and non-Binomial mixtures take none.
        bad = _mixresp(trials = 10)
        # refused: non-Binomial mixture with trials (IR contract)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # Non-finite weights (surface literals are finite by
        # construction, so this is programmatic-only).
        bad = _mixresp(weights = [0.5, Inf])
        # refused: non-finite mixture weights (IR contract)
        @test_throws ContractValidationError validate_structure(_mixplan(bad))
        # A non-simplex weights vector fails the family check.
        r = _mixresp(weights = :z)
        plan = StructuralPlan([r],
            [PredictorSpec(:mu, IdentityLink, terms, :mu)], priors, params,
            AssignmentSpec[], cols, 9; vector_parameters = VectorParameter[
                VectorParameter(:z, :vector_normal, (arg1 = [0.0, 0.0],
                    arg2 = [1.0, 1.0]), 2, :z)])
        # refused: mixture weights must name a simplex parameter (IR contract)
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
            y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.4, 0.6]))
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
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            mu1 = a1 .+ b1 .* x
            mu2 ~ Normal(0.0, 5.0)
            sigma ~ Exponential(1.0)
            y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.3, 0.7]))
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 0.5, -1.0],
            :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (a1 = 0.5, b1 = -0.25, mu2 = 1.0, sigma = 1.5)
        got = _mix_posterior(kern, lay, q)
        eta = q.a1 .+ q.b1 .* x
        ll = sum(zip(cols[:y], eta)) do (yi, etai)
            _mlogaddexp(log(0.3) + logpdf(Normal(etai, q.sigma), yi),
                log(0.7) + logpdf(Normal(q.mu2, q.sigma), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), (q.a1, q.b1))) +
            logpdf(Normal(0, 5), q.mu2) +
            logpdf(Exponential(1.0), q.sigma) + log(q.sigma)
        @test got ≈ want rtol = 1e-12
    end
    @testset "bernoulli predictor + literal" begin
        for ycol in ([0, 1, 1, 0], [false, true, true, false])
            prog = quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ MixtureModel.(vcat.(Bernoulli.(logistic.(eta)),
                    Bernoulli.(0.7)), Ref([0.5, 0.5]))
            end
            x = [0.5, -1.0, 1.5, 0.0]
            cols = Dict{Symbol,AbstractVector}(:y => ycol, :x => x)
            _, _, kern, lay = _mix_query(prog, cols)
            q = (a = 0.2, b = -0.4,)
            got = _mix_posterior(kern, lay, q)
            eta = q.a .+ q.b .* x
            p1 = 1 ./ (1 .+ exp.(-eta))
            ll = sum(zip(ycol, p1)) do (yi, pi)
                _mlogaddexp(log(0.5) + logpdf(Bernoulli(pi), yi),
                    log(0.5) + logpdf(Bernoulli(0.7), yi))
            end
            want = ll + sum(logpdf.(Normal(0, 1), (q.a, q.b)))
            @test got ≈ want rtol = 1e-12
        end
    end
    @testset "poisson param + literal" begin
        prog = quote
            lam1 ~ LogNormal(0.0, 1.0)
            y .~ MixtureModel.(vcat.(Poisson.(lam1), Poisson.(4.0)), Ref([0.3, 0.7]))
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
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ MixtureModel.(vcat.(Binomial.(n, logistic.(eta)),
                Binomial.(n, 0.25)), Ref([0.6, 0.4]))
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [3, 7, 5, 8],
            :x => x, :n => [10, 10, 10, 10])
        _, _, kern, lay = _mix_query(prog, cols)
        q = (a = 0.2, b = -0.4,)
        got = _mix_posterior(kern, lay, q)
        eta = q.a .+ q.b .* x
        p1 = 1 ./ (1 .+ exp.(-eta))
        ll = sum(zip(cols[:y], p1)) do (yi, pi)
            _mlogaddexp(log(0.6) + logpdf(Binomial(10, pi), yi),
                log(0.4) + logpdf(Binomial(10, 0.25), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), (q.a, q.b)))
        @test got ≈ want rtol = 1e-12
    end
    @testset "nb2 predictor + param means" begin
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            mu2 ~ Gamma(2.0, 1.0)
            phi1 ~ Exponential(1.0)
            phi2 ~ Exponential(1.0)
            y .~ MixtureModel.(vcat.(NegativeBinomial2.(exp.(eta), phi1),
                NegativeBinomial2.(mu2, phi2)), Ref([0.5, 0.5]))
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [2, 0, 4, 1], :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (a = 0.3, b = 0.1, mu2 = 2.0, phi1 = 1.5, phi2 = 0.5)
        got = _mix_posterior(kern, lay, q)
        mu1 = exp.(q.a .+ q.b .* x)
        nb(mu, phi, y) = logpdf(NegativeBinomial(phi, phi / (phi + mu)), y)
        ll = sum(zip(cols[:y], mu1)) do (yi, m1)
            _mlogaddexp(log(0.5) + nb(m1, q.phi1, yi),
                log(0.5) + nb(q.mu2, q.phi2, yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), (q.a, q.b))) +
            logpdf(Gamma(2, 1), q.mu2) + log(q.mu2) +
            logpdf(Exponential(1.0), q.phi1) + log(q.phi1) +
            logpdf(Exponential(1.0), q.phi2) + log(q.phi2)
        @test got ≈ want rtol = 1e-11
    end
    @testset "gamma predictor + param means, shared shape" begin
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            alpha ~ Exponential(1.0)
            mu2 ~ Gamma(2.0, 1.0)
            y .~ MixtureModel.(vcat.(Gamma.(alpha, exp.(eta) ./ alpha),
                Gamma.(alpha, mu2 ./ alpha)), Ref([0.5, 0.5]))
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [1.5, 0.5, 2.0, 1.0],
            :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (a = 0.2, b = -0.1, alpha = 2.0, mu2 = 1.5)
        got = _mix_posterior(kern, lay, q)
        mu1 = exp.(q.a .+ q.b .* x)
        ll = sum(zip(cols[:y], mu1)) do (yi, m1)
            _mlogaddexp(log(0.5) + logpdf(Gamma(q.alpha, m1 / q.alpha), yi),
                log(0.5) + logpdf(Gamma(q.alpha, q.mu2 / q.alpha), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), (q.a, q.b))) +
            logpdf(Exponential(1.0), q.alpha) + log(q.alpha) +
            logpdf(Gamma(2, 1), q.mu2) + log(q.mu2)
        @test got ≈ want rtol = 1e-12
    end
    @testset "beta predictor + literal means, shared kappa" begin
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            kappa ~ Exponential(1.0)
            y .~ MixtureModel.(vcat.(Beta.(logistic.(eta) .* kappa,
                    (1 .- logistic.(eta)) .* kappa),
                Beta.(0.7 .* kappa, (1 .- 0.7) .* kappa)), Ref([0.5, 0.5]))
        end
        x = [0.5, -1.0, 1.5, 0.0]
        cols = Dict{Symbol,AbstractVector}(:y => [0.2, 0.8, 0.4, 0.6],
            :x => x)
        _, _, kern, lay = _mix_query(prog, cols)
        q = (a = 0.2, b = -0.4, kappa = 5.0)
        got = _mix_posterior(kern, lay, q)
        mu1 = 1 ./ (1 .+ exp.(-(q.a .+ q.b .* x)))
        ll = sum(zip(cols[:y], mu1)) do (yi, m1)
            _mlogaddexp(log(0.5) + logpdf(Beta(m1 * q.kappa,
                        (1 - m1) * q.kappa), yi),
                log(0.5) + logpdf(Beta(0.7 * q.kappa, 0.3 * q.kappa), yi))
        end
        want = ll + sum(logpdf.(Normal(0, 1), (q.a, q.b))) +
            logpdf(Exponential(1.0), q.kappa) + log(q.kappa)
        @test got ≈ want rtol = 1e-12
    end
    @testset "K = 1 equals the single component" begin
        prog = quote
            mu1 ~ Normal(-2.0, 0.1)
            sigma ~ Exponential(1.0)
            y .~ MixtureModel.(vcat.(Normal.(mu1, sigma)), Ref([1.0]))
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
                y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.4, 0.6]))
            end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]),
            (mu1 = -2.0, mu2 = 2.0, sigma = 0.3))
    end
    @testset "bernoulli" begin
        _mix_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ MixtureModel.(vcat.(Bernoulli.(logistic.(eta)),
                    Bernoulli.(0.7)), Ref([0.5, 0.5]))
            end, Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0], :x => x),
            (a = 0.2, b = -0.4,))
    end
    @testset "poisson" begin
        _mix_enzyme_check(quote
                lam1 ~ LogNormal(0.0, 1.0)
                y .~ MixtureModel.(vcat.(Poisson.(lam1), Poisson.(4.0)), Ref([0.3, 0.7]))
            end, Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2]),
            (lam1 = 1.5,))
    end
    @testset "binomial" begin
        _mix_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ MixtureModel.(vcat.(Binomial.(n, logistic.(eta)),
                    Binomial.(n, 0.25)), Ref([0.6, 0.4]))
            end, Dict{Symbol,AbstractVector}(:y => [3, 7, 5, 8], :x => x,
                :n => [10, 10, 10, 10]), (a = 0.2, b = -0.4,))
    end
    @testset "nb2" begin
        _mix_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                mu2 ~ Gamma(2.0, 1.0)
                phi1 ~ Exponential(1.0)
                phi2 ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(NegativeBinomial2.(exp.(eta), phi1),
                    NegativeBinomial2.(mu2, phi2)), Ref([0.5, 0.5]))
            end, Dict{Symbol,AbstractVector}(:y => [2, 0, 4, 1], :x => x),
            (a = 0.3, b = 0.1, mu2 = 2.0, phi1 = 1.5, phi2 = 0.5))
    end
    @testset "gamma" begin
        _mix_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                alpha ~ Exponential(1.0)
                mu2 ~ Gamma(2.0, 1.0)
                y .~ MixtureModel.(vcat.(Gamma.(alpha, exp.(eta) ./ alpha),
                    Gamma.(alpha, mu2 ./ alpha)), Ref([0.5, 0.5]))
            end, Dict{Symbol,AbstractVector}(:y => [1.5, 0.5, 2.0, 1.0],
                :x => x), (a = 0.2, b = -0.1, alpha = 2.0, mu2 = 1.5))
    end
    @testset "beta" begin
        _mix_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                kappa ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Beta.(logistic.(eta) .* kappa,
                        (1 .- logistic.(eta)) .* kappa),
                    Beta.(0.7 .* kappa, (1 .- 0.7) .* kappa)), Ref([0.5, 0.5]))
            end, Dict{Symbol,AbstractVector}(:y => [0.2, 0.8, 0.4, 0.6],
                :x => x), (a = 0.2, b = -0.4, kappa = 5.0))
    end
    @testset "simplex weights" begin
        _mix_enzyme_check(quote
                w ~ Dirichlet([1.0, 1.0])
                mu1 ~ Normal(-2.0, 0.1)
                mu2 ~ Normal(2.0, 0.1)
                sigma ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref(w))
            end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]),
            (w = [0.4, 0.6], mu1 = -2.0, mu2 = 2.0, sigma = 0.3))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the mixture plate must not unroll over observations.
function _mix_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
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
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.4, 0.6]))
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
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
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
            y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.4, 0.6]))
        end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2])),
        ("poisson", quote
            lam1 ~ LogNormal(0.0, 1.0)
            y .~ MixtureModel.(vcat.(Poisson.(lam1), Poisson.(4.0)), Ref([0.3, 0.7]))
        end, Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2])),
        ("bernoulli", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ MixtureModel.(vcat.(Bernoulli.(logistic.(eta)),
                Bernoulli.(0.7)), Ref([0.5, 0.5]))
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

@testset "mixture simplex weights compiled parity" begin
    prog_w = quote
        w ~ Dirichlet([1.0, 1.0])
        y .~ MixtureModel.(vcat.(Normal.(-1.0, 0.5), Normal.(1.0, 0.5)), Ref(w))
    end
    cols_w = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5])
    fx = _mix_reactant(prog_w, cols_w)
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
end

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-25T23-22-01-158-qtfwg1 on
# BayesianRegressionModels:rk:parity-fam-mixture, BRM 20f532c, StanBlocks
# 24578c3, BridgeStan 2.9.0): full posterior at u_unc, propto=false,
# Jacobian included, BridgeStan AD grads. RK layout order differs from SB
# declaration order, so pins compare by coordinate name (SB pins below are
# in SB declaration order).
_mix_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "mixture SB parity" begin
    @testset "canonical 2-gaussian" begin
        # SB: mu1 ~ Normal(-2, 0.1); mu2 ~ Normal(2, 0.1);
        # log(sigma) ~ 1;
        # y ~ MixtureModel([Normal(mu1, sigma), Normal(mu2, sigma)],
        #     [0.4, 0.6]); y = [-2.0, -1.8, 1.9, 2.2];
        # u (SB order [mu1, mu2, sigma]) = [-2.0, 2.0, log(0.3)].
        prog = quote
            c ~ Normal(0, 1)
            mu1 ~ Normal(-2.0, 0.1)
            mu2 ~ Normal(2.0, 0.1)
            sigma = c
            y .~ MixtureModel.(vcat.(Normal.(mu1, exp.(sigma)),
                Normal.(mu2, exp.(sigma))), Ref([0.4, 0.6]))
        end
        bound, built, kern, lay = _mix_query(prog,
            Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]))
        names = coordinate_names(lay)
        u = _mix_sb_vec(names, [:mu1 => -2.0, :mu2 => 2.0,
            :c => log(0.3)])
        @test abs(Base.invokelatest(kern, u) - (-1.0905162971993954)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _mix_sb_vec(names, [:mu1 => 2.222222222222222,
            :mu2 => 1.1111111111111123,
            :c => -1.796027195674063])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "poisson count leg" begin
        # SB: lambda1 ~ Exponential(1); lambda2 ~ Exponential(1);
        # y ~ MixtureModel([Poisson(lambda1), Poisson(lambda2)],
        #     [0.3, 0.7]); y = [0, 1, 3, 5, 2];
        # u = log.([1.5, 4.0]).
        prog = quote
            lambda1 ~ Exponential(1.0)
            lambda2 ~ Exponential(1.0)
            y .~ MixtureModel.(vcat.(Poisson.(lambda1), Poisson.(lambda2)), Ref([0.3, 0.7]))
        end
        bound, built, kern, lay = _mix_query(prog,
            Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2]))
        names = coordinate_names(lay)
        u = _mix_sb_vec(names, [:lambda1 => log(1.5),
            :lambda2 => log(4.0)])
        @test abs(Base.invokelatest(kern, u) - (-13.770608192734482)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _mix_sb_vec(names, [:lambda1 => -1.4238640775837066,
            :lambda2 => -5.631856052803615])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
