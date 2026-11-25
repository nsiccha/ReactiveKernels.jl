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
    # The static arm of a `Float32` program's branch is float32: selected
    # beside the float32 `s`, the Python literal takes its dtype.
    literal = string(jax_source(jax_float32_literal_arm; have = (:x, :s), want = :total))
    @test occursin("jnp.where(s > 0, s, 1.0)", literal)

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

# A parameter-dependent helper keeps the integer reductions on the device.
@traceable function jax_weighted_extrema(a, row)
    isempty(row) && return zero(a)
    a * maximum(row) + a * a * minimum(row)
end
@kernel jax_ragged_extrema(a::Float64, rows) = begin
    weights = plate(rows) do row
        jax_weighted_extrema(a, row)
    end
    total = sum(weights)
    return total
end

@testset "jax_source: padded integer extrema in both JAX precision modes" begin
    have = (:a, :rows)
    native = prepare(jax_ragged_extrema; have, want = :total)
    for rows in ([[-9, -4], [7], Int[]], [UInt64[9, 4], UInt64[7], UInt64[]],
                 [Bool[true, false], Bool[true], Bool[]])
        types = (; a = Float64, rows = typeof(rows))
        options = (; have, want = :total, types)
        src = jax_source(jax_ragged_extrema; options..., data = (; rows), float_type = Float32)
        # A larger exemplar changes the padded width, never the emitted body.
        larger = [repeat(row, 5) for row in rows]
        @test string(src) == string(jax_source(jax_ragged_extrema; options...,
            data = (; rows = larger), float_type = Float32))
        if rk_jax_available()
            ref = jax_source(jax_ragged_extrema; options..., data = (; rows))
            for reference in (nothing, ref)
                r = rk_jax_float32(src, reference, :kernel, (2.0, rows); prepare = true)
                @test r.x64 == (reference !== nothing)
                @test r.dtype == r.grad_dtype == "float32"
                @test r.value == native(2.0, rows)
                @test r.grad == sum(maximum(row) + 4 * minimum(row) for row in rows if !isempty(row))
            end
            if eltype(eltype(rows)) === Int
                # With x64 on, neither a 32-bit upper nor lower bound is neutral here.
                wide = [[-(2^40)], [2^40, 2^41]]
                @test rk_jax_float32(src, ref, :kernel, (2.0, wide); prepare = true).value ==
                    native(2.0, wide)
                # Without prepare_data the ragged mask has no reduction initial.
                plain = jax_source(jax_ragged_extrema; options..., float_type = Float32)
                @test rk_jax_float32(plain, nothing, :kernel, (2.0, rows)).value == native(2.0, rows)
            end
        else
            @test_skip rk_jax_available()
        end
    end
end

# An ODE solve per plate cell whose solution meets a float32 value in a lazy
# branch, as a likelihood guarded by its scale does. With 64-bit mode on, a
# solve in JAX's default float made the solution float64 and the branch's arms
# disagree (`cond branches must have equal output types`; snag
# evaluating-the-l-662ff8c9, Portfolio's Friberg model in an RKM bundle).
_jax_float_absorption(t, y, ka, k) = [-ka * y[1], ka * y[1] - k * y[2]]
@kernel jax_float_ode(u::Vector{Float64}, doses::Vector{Float64},
                      times::Vector{Vector{Float64}}) = begin
    ka = exp(u[1])
    k = exp(u[2])
    sigma = exp(u[3])
    values = plate(doses, times) do d, ts
        sol = rk_ode_rk45_tol(_jax_float_absorption, [d, 0.0], 0.0, ts, 1e-6, 1e-6, 10^4, ka, k)
        level = sum(sol[:, 2])
        sigma > 0 ? -0.5 * ((log(level) - log(d)) / sigma)^2 - log(sigma) : -Inf
    end
    total = sum(values)
    return total
end

@testset "jax_source: a float32 module's ODE solve is float32 in either mode" begin
    have = (:u, :doses, :times)
    src32 = jax_source(jax_float_ode; have, want = :total, float_type = Float32)
    src64 = jax_source(jax_float_ode; have, want = :total)
    # The solve's dtype is the module's float type, not JAX's default float.
    @test occursin("\n    dtype = jnp.float32\n", string(src32))
    @test occursin("\n    dtype = jnp.float64\n", string(src64))
    @test !occursin("result_type(float)", string(src32))
    @test !occursin("jax_enable_x64", string(src32))
    if rk_jax_available() && success(`$RK_JAX_PYTHON -c "import diffrax"`)
        args = ([0.2, -0.7, -1.0], [100.0, 80.0, 120.0],
                [[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0]])
        # The float64 module, imported first, turns 64-bit mode on. diffrax's
        # Dopri5 dense output itself multiplies its stages by a NumPy float64
        # coefficient vector and rounds the product to the state's dtype
        # (`FourthOrderPolynomialInterpolation`), so this trace holds float64
        # there (`r.f64`); the values and gradient are float32.
        r = rk_jax_float32(src32, src64, :kernel, args)
        @test r.x64
        @test r.dtype == "float32" && r.grad_dtype == "float32"
        @test r.value ≈ r.reference rtol = 1e-5
        @test r.grad ≈ r.reference_grad rtol = 1e-3
        alone = rk_jax_float32(src32, nothing, :kernel, args)
        @test !alone.x64 && alone.dtype == "float32"
        @test alone.value ≈ r.value rtol = 1e-6
        # The per-lane CUDA path, in Pallas' interpreter on this backend: its
        # steps are the module's own, with no float64 anywhere.
        if success(`$RK_JAX_PYTHON -c "from jax.experimental import pallas"`)
            lanes = withenv(() -> rk_jax_float32(src32, src64, :kernel, args),
                            "RK_JAX_ODE_KERNEL" => "interpret")
            @test lanes.x64 && !lanes.f64
            @test lanes.dtype == "float32" && lanes.grad_dtype == "float32"
            @test lanes.value ≈ r.value rtol = 1e-5
            @test lanes.grad ≈ r.grad rtol = 1e-3
        end
    else
        @test_skip false
    end
end
