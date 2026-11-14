using ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# A plain authored helper broadcast in a model is a named, documented Python
# function with its method's argument names, vectorized itself; the stable
# rewrites' recorded helpers stay private (snag jax-source-drops-5f297211).

"""
    jax_named_trend_mean(level, trend, t)

The mean at step `t`.
"""
jax_named_trend_mean(level, trend, t) = level + trend * t

const jax_named_trend_model = @rkppl begin
    "Level."
    level ~ Normal(level_loc, level_scale)
    "Trend per step."
    trend ~ Normal(trend_loc, trend_scale)
    "Noise SD."
    noise ~ Exponential(noise_scale)
    mu = jax_named_trend_mean.(level, trend, t)
    y .~ Normal.(mu, noise)
end

@testset "a broadcast plain helper is a named function of the model" begin
    for n in (0, 3, 19)
        t = collect(range(1.0, 3.0; length = n))
        raw = (; level_loc = 0.1, level_scale = 1.0, trend_loc = 0.0, trend_scale = 0.5,
               noise_scale = 1.0, t, y = 0.2 .* t .- 0.1)
        plan = jax_named_trend_model(; Base.structdiff(raw, (; y = raw.y))...) | (; raw.y)
        built = build_kernel(plan)
        src = jax_source(built, plan)
        text = src.text
        @test occursin("def jax_named_trend_mean(level: ArrayLike, trend: ArrayLike, t: ArrayLike)", text)
        @test occursin("The mean at step `t`.", text)
        @test occursin("jnp.vectorize(jax_named_trend_mean)(", text)
        @test !occursin(r"(?m)^def _jax_named_trend_mean", text)
        @test !occursin(r"(?m)^def rk_"m, text)
        names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
        args = Tuple(getproperty(raw, Symbol(name)) for name in names)
        u = collect(range(-0.2, 0.3; length = built.layout.total))
        if rk_jax_available()
            native = prepare_query(built, plan, :sampler)
            got = rk_jax_evaluate(src, :log_density, (u, args...); grad = true,
                                  prepare = :arguments)
            @test got.value ≈ native(u) atol = 1e-11
            finite_difference = [(native(u + 1e-5 .* (1:length(u) .== i)) -
                                  native(u - 1e-5 .* (1:length(u) .== i))) / 2e-5
                                 for i in eachindex(u)]
            @test got.grad ≈ finite_difference rtol = 2e-6 atol = 1e-7
        else
            @test_skip false
        end
    end
end
