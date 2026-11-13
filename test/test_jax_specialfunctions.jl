using ReactiveKernels, SpecialFunctions, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_trigamma_edges(x::Vector{Float64}) = begin
    values = trigamma.(x)
    total = sum(values)
    return total
end
@kernel jax_besseli_edges(x::Vector{Float64}, n::Vector{Int}) = begin
    values = besseli.(n, x)
    total = sum(values)
    return total
end
@kernel jax_trigamma_prepared(w::Float64, x::Vector{Float64}) = begin
    values = trigamma.(x)
    total = w * sum(values)
    return total
end

@testset "jax_source: SpecialFunctions edge values and derivatives" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsSpecialFunctionsExt) !== nothing
    tg = jax_source(jax_trigamma_edges; have = (:x,), want = (:values, :total))
    bi = jax_source(jax_besseli_edges; have = (:x, :n), want = (:values, :total))
    @test occursin("_rk_trigamma(x)", string(tg))
    @test occursin("_rk_besseli(n, x)", string(bi))
    if !rk_jax_available()
        @test_skip false
    else
        # Relative comparisons catch lost tiny values; a default absolute
        # tolerance would accept the original zero at 1e200 and 1e300.
        x = [0.125, 1.0, 8.0, 1e4, prevfloat(1e8), 1e8, nextfloat(1e8),
             1e9, 1e100, 1e154, 1e200, 1e300, Inf]
        src = jax_source(jax_trigamma_edges; have = (:x,), want = :values)
        got = rk_jax_evaluate(src, "kernel", (x,)).value
        @test all(isapprox.(got, trigamma.(x); rtol = 2e-14, atol = 0))
        @test got[end] == 0.0
        @test got[end-1] > 0.0
        src = jax_source(jax_trigamma_edges; have = (:x,), want = :total)
        # Derivatives beyond 1e150 are subnormal or below Float64 range;
        # compare representable derivatives and the limiting derivative at Inf.
        dx = [x[1:9]; Inf]
        result = rk_jax_evaluate(src, "kernel", (dx,); grad = true)
        @test result.value ≈ sum(trigamma, dx) rtol = 2e-14
        @test all(isapprox.(result.grad, polygamma.(2, dx); rtol = 2e-14, atol = 0))
        @test result.grad[end] == 0.0

        # Computed/prepared values take the NumPy/SciPy host translation.
        src = jax_source(jax_trigamma_prepared; have = (:w, :x), want = :total,
                         data = (; x))
        prepared = rk_jax_evaluate(src, "kernel", (2.0, x); grad = true, prepare = true)
        @test prepared.value ≈ 2sum(trigamma, x) rtol = 2e-14
        @test prepared.grad ≈ sum(trigamma, x) rtol = 2e-14

        # The same tail also avoids Float32 polygamma underflow.
        src32 = jax_source(jax_trigamma_edges; have = (:x,), want = :total,
                           float_type = Float32)
        src64 = jax_source(jax_trigamma_edges; have = (:x,), want = :total)
        for xf in ([0.125, 1.0, 1e8, Inf], [1e20], [1e30])
            single = rk_jax_float32(src32, src64, "kernel", (xf,))
            @test single.value ≈ sum(trigamma, xf) rtol = 2e-6 atol = 0
            @test !single.f64
        end

        # Runtime orders on both sides of zero, and both signed zeros in x.
        orders = [-3, -2, -1, 0, 1, 2, 3, 7, 24, 25]
        points = [-4.0, -1e-8, -1e-300, -0.0, 0.0, 1e-300, 1e-8, 4.0]
        n = repeat(orders; inner = length(points))
        x = repeat(points; outer = length(orders))
        src = jax_source(jax_besseli_edges; have = (:x, :n), want = :values)
        got = rk_jax_evaluate(src, "kernel", (x, n)).value
        @test all(isapprox.(got, besseli.(n, x); rtol = 2e-12, atol = 1e-300))
        src = jax_source(jax_besseli_edges; have = (:x, :n), want = :total)
        result = rk_jax_evaluate(src, "kernel", (x, n); grad = true)
        reference = (besseli.(n .- 1, x) .+ besseli.(n .+ 1, x)) ./ 2
        @test all(isapprox.(result.grad, reference; rtol = 2e-12, atol = 1e-14))
        @test all(iszero, result.grad[(n .== 0) .& iszero.(x)])
        @test all(==(0.5), result.grad[(abs.(n) .== 1) .& iszero.(x)])

        # Empty broadcasts retain empty outputs and gradients.
        empty = rk_jax_evaluate(src, "kernel", (Float64[], Int[]); grad = true)
        @test empty.value == 0.0
        @test isempty(empty.grad)
    end
end
