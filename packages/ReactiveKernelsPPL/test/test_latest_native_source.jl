using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels
using Test

# The GLM rule and half-prior helpers are created when their package loads.
# A model created later may be called from an older world, so check both the
# public latest-world sampler and raw queries with their documented barrier.
function _latest_source_fresh_model()
    mod = Module(gensym(:LatestSource))
    Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
    Core.eval(mod, quote
        generated_regression = @rkppl begin
            beta[axes(X, 2)] .~ Normal.(0, 10)
            sigma ~ HalfNormal(10)
            mu = X * beta
            y .~ Normal.(mu, sigma)
        end
    end)
    X = [1.0 0.5; 1.0 -0.4; 1.0 1.2; 1.0 -0.7]
    y = [0.3, -0.5, 1.1, -0.2]
    model = Core.eval(mod, :generated_regression)
    plan = model(; X) | (; y)
    sources = ReactiveKernelsDistributionKernels.DistributionKernelSources
    @test ReactiveKernelsPPL.PPLGeneratedModels.normal_id_glm_lpdf(X, zeros(2), 1.0, y) ≈
          sources.normal_id_glm_lpdf(X, zeros(2), 1.0, y)
    @test sources._half_normal_logpdf_on_support(1.0, 10.0) ≈
          logpdf(Normal(0.0, 10.0), 1.0) + log(2)
    built = build_kernel(plan)
    u = zeros(built.layout.total)
    raw = prepare_query(built, plan, :sampler)
    q = prepare_sampler(built, plan, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value = Base.invokelatest(raw, u)
    @test q(u) ≈ value
    likelihood = Base.invokelatest(prepare_query(built, plan, :likelihood), u)
    @test likelihood isa Float64
    pw = Base.invokelatest(prepare_query(built, plan, :pointwise), u).y
    @test length(pw) == length(y)
    @test sum(pw) ≈ likelihood
    @test Base.invokelatest(prepare_cell_query(built, plan, :y), u, 2) ≈ pw[2]
    gradient = zeros(length(u))
    gradient_value, returned = sampler_value_and_gradient!(q, gradient, u)
    @test returned === gradient
    @test gradient_value ≈ value
    h = 1e-5
    for i in eachindex(u)
        step = zeros(length(u))
        step[i] = h
        finite = (Base.invokelatest(raw, u .+ step) -
                  Base.invokelatest(raw, u .- step)) / (2h)
        @test gradient[i] ≈ finite rtol = 1e-5 atol = 1e-6
    end
    nothing
end

@testset "scoped native source calls across a fresh-model world age" begin
    captured = ReactiveKernels._KernelSourceFunction(identity, nothing, (; kept = 2.0))
    latest = ReactiveKernels._kernel_latest_source(captured)
    @test latest.captures === captured.captures
    @test latest(2.0) == captured(2.0)
    _latest_source_fresh_model()
end
