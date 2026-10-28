using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# A model call with nothing left to observe returns its binding, an
# `RKPPLBoundModel`: here, a model whose only response is a data-only
# definition (`ys = hcat(u, v); ys ~ LogDensity(…)`, rkppl-use §1 "Hidden
# Markov sequences"), and a prior-only model. Every entry point that reads a
# bound plan reads such a binding as `build_kernel` does, as the plan
# `bind_data(binding)` lowers it to (snag rkppl-kernel-exp-5efabaed).

const _BME_U = [0.4, 1.3, 0.9, 2.1]
const _BME_V = [-0.3, 0.8, 0.1, -1.2]
_bme_whole(ys, a, s) = sum(-((ys .- a) ./ s) .^ 2 ./ 2 .- log(s))
_bme_central(f, u, j; h = 1e-6) =
    (f(setindex!(copy(u), u[j] + h, j)) - f(setindex!(copy(u), u[j] - h, j))) / 2h

@testset "a binding with nothing to observe reads as its bound plan" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        ys = hcat(u, v)
        ys ~ LogDensity(_bme_whole, a, s)
    end
    b = m(; u = copy(_BME_U), v = copy(_BME_V))
    @test b isa RKPPLBoundModel
    plan = bind_data(b)
    built = build_kernel(b)
    @test coordinate_names(assign_layout(b)) == coordinate_names(assign_layout(plan)) ==
        coordinate_names(built.layout) == [:a, :s]
    @test kernel_expr(b, built.layout) == kernel_expr(plan, built.layout)
    @test kernel_expr(b, built.layout; name = :other) ==
        kernel_expr(plan, built.layout; name = :other)
    @test model_view(built; bound = b).source == model_view(built; bound = plan).source
    @test model_view(built; bound = b).source !== nothing

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q(cut) = u -> Base.invokelatest(prepare_query(built, b, cut), u)
    qp = u -> Base.invokelatest(prepare_query(built, plan, :sampler), u)
    sampler = prepare_sampler(built, b, unconstrain(built.layout, (; a = 0.0, s = 1.0)); backend)
    twin = prepare_sampler(built, plan, unconstrain(built.layout, (; a = 0.0, s = 1.0)); backend)
    ad = prepare_query_ad(prepare_ad, built, b, :sampler, backend,
        unconstrain(built.layout, (; a = 0.0, s = 1.0)))
    for (a, s) in ((0.3, 0.8), (-1.1, 2.4))
        u = unconstrain(built.layout, (; a, s))
        @test q(:likelihood)(u) ≈ _bme_whole(hcat(_BME_U, _BME_V), a, s) rtol = 1e-12
        @test q(:prior)(u) ≈ logpdf(Normal(0, 1), a) + logpdf(Exponential(1), s) rtol = 1e-12
        @test q(:sampler)(u) ≈ q(:likelihood)(u) + q(:prior)(u) + q(:log_jacobian)(u) rtol = 1e-12
        @test q(:sampler)(u) == qp(u)
        g, gp = similar(u), similar(u)
        value, _ = sampler_value_and_gradient!(sampler, g, u)
        twin_value, _ = sampler_value_and_gradient!(twin, gp, u)
        @test value ≈ q(:sampler)(u) rtol = 1e-12
        @test value == twin_value
        @test g == gp
        for j in eachindex(u)
            @test g[j] ≈ _bme_central(q(:sampler), u, j) rtol = 1e-6
        end
        ad_value, ad_gradient = ad_value_and_gradient(ad, u)
        @test ad_value ≈ value rtol = 1e-12
        @test ad_gradient ≈ g rtol = 1e-12
    end
end

@testset "a binding reads as its bound plan in one-cell queries" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        ys = log.(raw)
        ys .~ Normal.(a, 0.5)
    end
    b = m(; raw = exp.(_BME_U))
    @test b isa RKPPLBoundModel
    plan = bind_data(b)
    built = build_kernel(b)
    u = unconstrain(built.layout, (; a = 0.7))
    cells = prepare_cell_query(built, b, :ys)
    twin_cells = prepare_cell_query(built, plan, :ys)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    sampler = prepare_cell_sampler(built, b, :ys, u; backend)
    twin = prepare_cell_sampler(built, plan, :ys, u; backend)
    compact_backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    compact = prepare_compact_cell_sampler(built, b, :ys, u, [1]; backend = compact_backend)
    for i in eachindex(_BME_U)
        @test cells(u, i) ≈ logpdf(Normal(0.7, 0.5), _BME_U[i]) rtol = 1e-12
        @test cells(u, i) == twin_cells(u, i)
        g, gp = similar(u), similar(u)
        value, _ = cell_value_and_gradient!(sampler, g, u, i)
        twin_value, _ = cell_value_and_gradient!(twin, gp, u, i)
        @test value == twin_value
        @test g == gp
        @test g[1] ≈ (_BME_U[i] - 0.7) / 0.5^2 rtol = 1e-12
        compact_gradient = similar(u, 1)
        compact_value, _ = cell_value_and_gradient!(compact, compact_gradient, u, u[[1]], [1], i)
        @test compact_value ≈ value rtol = 1e-12
        @test compact_gradient ≈ g rtol = 1e-12
    end
end

@testset "a model with no inputs reads as its bound plan" begin
    prior_only = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Exponential(1)
    end
    @test prior_only isa RKPPLModel
    built = build_kernel(prior_only)
    plan = bind_data(prior_only())
    @test kernel_expr(prior_only, built.layout) == kernel_expr(plan, built.layout)
    @test coordinate_names(assign_layout(prior_only)) == [:a, :s]
    u = unconstrain(built.layout, (; a = -0.4, s = 1.7))
    @test Base.invokelatest(prepare_query(built, prior_only, :prior), u) ≈
        logpdf(Normal(0, 1), -0.4) + logpdf(Exponential(1), 1.7) rtol = 1e-12

    # A captured model conditioned before any call lowers on its call too.
    observed = condition(@rkppl(begin
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end); y = copy(_BME_V))
    @test observed isa RKPPLModel
    observed_built = build_kernel(observed)
    @test kernel_expr(observed, observed_built.layout) ==
        kernel_expr(observed(), observed_built.layout)
    v = unconstrain(observed_built.layout, (; a = 0.2))
    @test Base.invokelatest(prepare_query(observed_built, observed, :likelihood), v) ≈
        sum(logpdf.(Normal(0.2, 1), _BME_V)) rtol = 1e-12
end
