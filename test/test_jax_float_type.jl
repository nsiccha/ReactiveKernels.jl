using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# `jax_source(...; float_type = Float32)`: a float32 module of a Float64
# program, for an accelerator whose float64 is slow (snag
# jax-source-float-5e6b7604). Its float64 module is the reference.

# A lazy branch with a literal arm (`1.0`, float64 under JAX's 64-bit mode)
# and a static NumPy constant (`log(2pi)`, a float64 NumPy scalar).
@kernel jax_float_branch(y::Vector{Float64}, mu::Float64, sigma::Float64) = begin
    scale = sigma > 0 ? sigma : 1.0
    z = (y .- mu) ./ scale
    total = sum(-0.5 .* z .^ 2) - length(y) * log(scale) - 0.5 * length(y) * log(2pi)
    return total
end
# A loop whose carry starts at a literal, and an allocation.
@traceable function jax_float_squares(x::AbstractVector)
    acc = 0.0
    out = zeros(length(x))
    for i in eachindex(x)
        acc += x[i]^2
        out[i] = acc
    end
    return out
end
@kernel jax_float_loop(x::Vector{Float64}, rate::Float64) = begin
    path = jax_float_squares(x)
    total = sum(exp.(-rate .* path))
    return total
end
# `fma` rounds once, through a float64 product even in a float32 module.
@kernel jax_float_fma(x::Float64, y::Float64, z::Float64) = begin
    result = fma(x, y, z)
    return result
end
# A `Float32` program's literal arm, in the default module.
@kernel jax_float32_literal_arm(x::Vector{Float32}, s::Float32) = begin
    scale = s > 0 ? s : 1.0f0
    total = sum(x ./ scale)
    return total
end

@testset "jax_source: float_type = Float32" begin
    branch = (jax_float_branch, (:y, :mu, :sigma), :total)
    loop = (jax_float_loop, (:x, :rate), :total)
    for (spec, have, want) in (branch, loop)
        src64 = jax_source(spec; have, want)
        src32 = jax_source(spec; have, want, float_type = Float32)
        # The default module is the `Float64` one.
        @test string(jax_source(spec; have, want, float_type = Float64)) == string(src64)
        @test occursin("jax.config.update(\"jax_enable_x64\", True)", string(src64))
        # No float64 anywhere, and no 64-bit mode.
        @test !occursin("jax_enable_x64", string(src32))
        @test !occursin("float64", string(src32))
        @test occursin("dtype=jnp.float32", string(src32))
    end
    @test_throws ArgumentError jax_source(jax_float_branch; have = (:y, :mu, :sigma),
                                          want = :total, float_type = Float16)
    # `fma` keeps its once-rounding helper, so the module turns 64-bit mode on.
    fma32 = string(jax_source(jax_float_fma; have = (:x, :y, :z), want = :result,
                              float_type = Float32))
    @test occursin("_rk_fma32", fma32) && occursin("jax_enable_x64", fma32)
    # The static arm of a `Float32` program's branch is float32.
    literal = string(jax_source(jax_float32_literal_arm; have = (:x, :s), want = :total))
    @test occursin("dtype=jnp.float32))", literal)

    if !rk_jax_available()
        @info "RK_JAX_PYTHON is not set: float32 JAX parity skipped"
        @test_skip false
    else
        y = [1.2, -0.3, 2.5, 0.7, 1.9]
        for (sigma, label) in ((0.8, "then arm"), (-1.0, "literal arm"))
            args = (y, 0.4, sigma)
            src32 = jax_source(jax_float_branch; have = (:y, :mu, :sigma), want = :total,
                               float_type = Float32)
            src64 = jax_source(jax_float_branch; have = (:y, :mu, :sigma), want = :total)
            native = prepare(jax_float_branch; have = (:y, :mu, :sigma), want = :total)(args...)
            # With 64-bit mode on (the float64 module imported first), the
            # float32 module's trace holds no float64.
            r = rk_jax_float32(src32, src64, :kernel, args)
            @test r.x64 && !r.f64
            @test r.dtype == "float32" && r.grad_dtype == "float32"
            @test r.reference ≈ native rtol = 1e-12
            @test abs(r.value - native) <= 8 * eps(Float32) * max(1, abs(native))
            @test r.grad ≈ r.reference_grad rtol = 1e-5
            # Alone, it runs without 64-bit mode.
            alone = rk_jax_float32(src32, nothing, :kernel, args)
            @test !alone.x64 && alone.dtype == "float32"
            @test alone.value == r.value
        end
        args = ([0.3, 0.1, 0.7, 0.2], 0.5)
        r = rk_jax_float32(jax_source(jax_float_loop; have = (:x, :rate), want = :total,
                                      float_type = Float32),
                           jax_source(jax_float_loop; have = (:x, :rate), want = :total),
                           :kernel, args)
        @test r.x64 && !r.f64 && r.dtype == "float32"
        @test abs(r.value - r.reference) <= 8 * eps(Float32) * max(1, abs(r.reference))
        @test r.grad ≈ r.reference_grad rtol = 1e-5
        # A `Float32` program's literal arm in its float64 module.
        literal = jax_source(jax_float32_literal_arm; have = (:x, :s), want = :total)
        r = rk_jax_float32(literal, nothing, :kernel, (Float32[1, 2, 3], -1.0f0))
        @test r.x64 && !r.f64 && r.dtype == "float32" && r.value == 6
    end
end
