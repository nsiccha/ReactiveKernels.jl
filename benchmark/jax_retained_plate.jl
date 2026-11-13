# Consumer reproduction for snag export-an-explic-afc96e70. Run in an
# environment containing this RK checkout, its nested RKPPL, BRM and Enzyme,
# with RK_JAX_PYTHON pointing to Python/JAX.
# The permanent compiler regression is test/test_jax_flat_provenance.jl.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, Test
include(joinpath(@__DIR__, "..", "test", "jax_python.jl"))

bare(d) = @brm d begin
    mu ~ 1 + x
    onset ~ LogNormal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(mu)
        mean_cell = mu[i] .* exp.(-t[i] ./ onset)
        y[i] ~ Normal(mean_cell, sigma)
    end
end
kept(d) = @brm d begin
    mu ~ 1 + x
    onset ~ LogNormal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(mu)
        mean_cell[i] = mu[i] .* exp.(-t[i] ./ onset)
        y[i] ~ Normal(mean_cell[i], sigma)
    end
end

function raw_inputs(rk, src)
    bound = rk_bound_data(rk, src)
    prepare = only(f for f in src.functions if f.name == "prepare_data")
    Tuple(bound[Symbol(first(input))] for input in prepare.inputs)
end

function native_value_gradient(query, u)
    gradient = zero(u)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(query), Enzyme.Active,
                   Enzyme.Duplicated(u, gradient))
    query(u), gradient
end

@testset "BRM retained ragged values export through RK" begin
    @test rk_jax_available()
    @test Base.get_extension(BayesianRegressionModels,
        :BayesianRegressionModelsReactiveKernelsExt) !== nothing
    data = [
        (; x = [-0.5, 0.3, 1.2], t = [[0., 0.5], [0.3], [0.2, 0.8, 1.5]],
           y = [[0.9, 0.8], [0.7], [1., 0.8, 0.4]]),
        (; x = [-0.7, 0.1, 0.8, 1.4], t = [[0.2], [0., 0.5, 0.9], [0.3, 1.1], [0.4, 1.7]],
           y = [[0.7], [0.9, 0.8, 0.6], [0.8, 0.5], [0.6, 0.3]]),
    ]
    original = deepcopy(data)
    backends = [(RKBRMI(bare(d)), RKBRMI(kept(d))) for d in data]
    sources = [jax_source(b.model, rk_bound_plan(b); values = (:mean_cell,))
               for (_, b) in backends]
    @test first(sources).text == last(sources).text
    src = first(sources) # Reuse one emitted module for both datasets.
    for (case, (a, b)) in enumerate(backends)
        bare_src = jax_source(a.model, rk_bound_plan(a))
        queries = [prepare_query(r.model, rk_bound_plan(r), :sampler) for r in (a, b)]
        qmean, qmu, qonset = [prepare_query(b.model, rk_bound_plan(b), (name,))
                             for name in (:mean_cell, :mu, :onset)]
        for shift in (0.0, 0.2, -0.3)
            u = collect(range(-0.2, 0.4; length = a.model.layout.total)) .+ shift
            saved = copy(u)
            native = [Base.invokelatest(native_value_gradient, q, u) for q in queries]
            @test native[1][1] ≈ native[2][1] rtol = 1e-12
            @test native[1][2] ≈ native[2][2] rtol = 1e-10 atol = 1e-10
            actual = Base.invokelatest(qmean, u)
            mu, onset = Base.invokelatest(qmu, u), Base.invokelatest(qonset, u)
            expected = [mu[i] * exp(-t / onset) for i in eachindex(mu) for t in data[case].t[i]]
            @test actual ≈ expected rtol = 1e-12
            out = rk_jax_evaluate(src, :constrain, (u, raw_inputs(b, src)...); prepare = :arguments)
            @test out.value["mean_cell"] ≈ actual rtol = 1e-12
            @test length(out.value["mean_cell"]) == sum(length, data[case].t)
            for (r, module_src, reference) in zip((a, b), (bare_src, src), native)
                result = rk_jax_evaluate(module_src, :log_density,
                    (u, raw_inputs(r, module_src)...); grad = true, prepare = :arguments)
                @test result.value ≈ reference[1] rtol = 1e-12
                @test result.grad ≈ reference[2] rtol = 1e-10 atol = 1e-10
            end
            @test u == saved
            println("case=", case, " shift=", shift, " flat_length=", length(actual),
                    " density=", native[2][1])
        end
    end
    @test data == original
    println("emitted_bytes=", sizeof(src.text), " RK=", pkgversion(ReactiveKernels),
            " RKPPL=", pkgversion(ReactiveKernelsPPL), " BRM=", pkgversion(BayesianRegressionModels))
end
