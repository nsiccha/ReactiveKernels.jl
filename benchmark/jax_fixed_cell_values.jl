# Consumer reproduction for snag jax-source-value-3bf4c69e. Run in an
# environment containing this RK checkout, its nested RKPPL, BRM and Enzyme,
# with RK_JAX_PYTHON pointing to Python/JAX.
# The permanent compiler regression is test/test_jax_flat_provenance.jl.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, Test
include(joinpath(@__DIR__, "..", "test", "jax_python.jl"))

# Two measurements per unit, the second lower by a per-unit amount: the
# kept value is `vcat` of two of the cell's scalars.
kept_vcat(d) = @brm d begin
    sigma ~ Exponential(1)
    level ~ 1 + (1 | u | unit)
    fall ~ 0 + x
    @plate for i in eachindex(level)
        mean_y[i] = vcat(level[i], level[i] - fall[i])
        y[i] ~ Normal(mean_y[i], sigma)
    end
end
# The same model with the value a broadcast over a per-unit data vector.
kept_broadcast(d) = @brm d begin
    sigma ~ Exponential(1)
    level ~ 1 + (1 | u | unit)
    fall ~ 0 + x
    @plate for i in eachindex(level)
        mean_y[i] = level[i] .- fall[i] .* visit[i]
        y[i] ~ Normal(mean_y[i], sigma)
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

@testset "BRM kept vcat cell values export through RK" begin
    @test rk_jax_available()
    @test Base.get_extension(BayesianRegressionModels,
        :BayesianRegressionModelsReactiveKernelsExt) !== nothing
    data = [
        (; unit = ["a", "b", "c"], x = [0.5, 1.0, 2.0], visit = [[0.0, 1.0] for _ in 1:3],
           y = [[1.0, 0.4], [1.2, 0.3], [0.9, -0.6]]),
        (; unit = ["a", "b", "c", "d", "e"], x = [0.2, 0.7, 1.1, 1.6, 2.4],
           visit = [[0.0, 1.0] for _ in 1:5],
           y = [[0.8, 0.5], [1.1, 0.2], [0.7, 0.1], [1.3, -0.2], [0.9, -0.9]]),
    ]
    original = deepcopy(data)
    for d in data
        vc, bc = RKBRMI(kept_vcat(d)), RKBRMI(kept_broadcast(d))
        @test vc.model.layout.total == bc.model.layout.total
    end
    backends = [(RKBRMI(kept_vcat(d)), RKBRMI(kept_broadcast(d))) for d in data]
    sources = [jax_source(v.model, rk_bound_plan(v); values = (:mean_y,)) for (v, _) in backends]
    controls = [jax_source(b.model, rk_bound_plan(b); values = (:mean_y,)) for (_, b) in backends]
    # One emitted module serves both datasets.
    @test first(sources).text == last(sources).text
    src = first(sources)
    for (case, (v, b)) in enumerate(backends)
        sampler = prepare_query(v.model, rk_bound_plan(v), :sampler)
        qmean, qlevel, qfall = [prepare_query(v.model, rk_bound_plan(v), (name,))
                                for name in (:mean_y, :level, :fall)]
        for shift in (0.0, 0.2, -0.3)
            u = collect(range(-0.2, 0.4; length = v.model.layout.total)) .+ shift
            saved = copy(u)
            native = Base.invokelatest(native_value_gradient, sampler, u)
            actual = Base.invokelatest(qmean, u)
            level, fall = Base.invokelatest(qlevel, u), Base.invokelatest(qfall, u)
            expected = [m for i in eachindex(level) for m in (level[i], level[i] - fall[i])]
            @test actual ≈ expected rtol = 1e-12
            out = rk_jax_evaluate(src, :constrain, (u, raw_inputs(v, src)...); prepare = :arguments)
            @test out.value["mean_y"] ≈ actual rtol = 1e-12
            @test length(out.value["mean_y"]) == 2 * length(data[case].unit)
            control = rk_jax_evaluate(controls[case], :constrain, (u, raw_inputs(b, controls[case])...);
                                      prepare = :arguments)
            @test out.value["mean_y"] ≈ control.value["mean_y"] rtol = 1e-12
            result = rk_jax_evaluate(src, :log_density, (u, raw_inputs(v, src)...);
                                     grad = true, prepare = :arguments)
            @test result.value ≈ native[1] rtol = 1e-12
            @test result.grad ≈ native[2] rtol = 1e-10 atol = 1e-10
            @test u == saved
            println("case=", case, " shift=", shift, " flat_length=", length(actual),
                    " density=", native[1], " mean_y=", out.value["mean_y"])
        end
    end
    @test data == original
    println("emitted_bytes=", sizeof(src.text), " RK=", pkgversion(ReactiveKernels),
            " RKPPL=", pkgversion(ReactiveKernelsPPL), " BRM=", pkgversion(BayesianRegressionModels))
end
