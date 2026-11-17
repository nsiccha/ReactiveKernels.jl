using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Julia's `A[i] = v` stores `convert(eltype(A), v)`. JAX's `.at[i].set(v)` of
# a value of another dtype leaves that cast to the scatter, which performs it
# with a FutureWarning ("scatter inputs have incompatible types") and will
# refuse it in a later release (snag jax-float32-arra-f4c14d66): the emitted
# store converts the value to the array's dtype itself.

# A loop storing a branch's Float64 value into a Float32 vector.
@traceable function jax_store_losses(x::AbstractVector, weight)
    losses = zeros(Float32, length(x))
    for i in eachindex(x)
        loss = 0.0
        if x[i] > 0
            loss = weight * log(x[i]) + (1 - weight) * x[i]
        end
        losses[i] = loss
    end
    return losses
end
@kernel jax_store_branch(weight::Float64, x::Vector{Float64}) = begin
    losses = jax_store_losses(x, weight)
    total = sum(losses)
    return total
end

# A row store, an updating store and a float stored into an integer vector.
@traceable function jax_store_rows(x::AbstractVector, scale)
    out = zeros(Float32, length(x), 2)
    counts = zeros(Int, length(x))
    for i in eachindex(x)
        out[i, :] = [scale * x[i], scale + x[i]]
        out[i, 1] += scale
        counts[i] = floor(x[i])
    end
    return sum(out) + sum(counts)
end
@kernel jax_store_matrix(scale::Float64, x::Vector{Float64}) = begin
    total = jax_store_rows(x, scale)
    return total
end

# A Float64 row stored into a vector of Float32 vectors.
@traceable function jax_store_slot_rows(a::Float64, x::AbstractVector)
    rows = Vector{Vector{Float32}}(undef, 2)
    rows[1] = [a, 2a]
    rows[2] = [a + x[1], a * x[1]]
    total = 0.0
    for t in eachindex(rows)
        total += sum(rows[t])
    end
    total
end
@kernel jax_store_slots(a::Float64, x::Vector{Float64}) = begin
    total = jax_store_slot_rows(a, x)
    return total
end

# An integer stored into a Bool vector, and a Bool into a Float32 vector.
@traceable function jax_store_flags(x::AbstractVector, s)
    flags = zeros(Bool, length(x))
    weights = zeros(Float32, length(x))
    for i in eachindex(x)
        flags[i] = Int(x[i] > 0)
        weights[i] = x[i] > 1
    end
    return s * (sum(flags) + sum(weights))
end
@kernel jax_store_flag_kernel(s::Float64, x::Vector{Float64}) = begin
    total = jax_store_flags(x, s)
    return total
end

# The FutureWarning of an implicit scatter cast, raised as an error.
const JAX_STORE_STRICT = "error:scatter inputs have incompatible types:FutureWarning"

@testset "jax_source: typed array stores convert like setindex!" begin
    x = [1.7, -0.4, 2.3, 0.6, -1.1]
    cases = [
        # (spec, have, args, d total / d first argument)
        (jax_store_branch, (:weight, :x), (0.35, x), sum(log(v) - v for v in x if v > 0)),
        (jax_store_branch, (:weight, :x), (0.35, Float64[]), 0.0),
        (jax_store_matrix, (:scale, :x), (1.25, x), sum(v + 2 for v in x)),
        (jax_store_slots, (:a, :x), (0.8, x), 4 + x[1]),
        (jax_store_flag_kernel, (:s, :x), (1.5, x), count(>(0), x) + count(>(1), x)),
    ]
    for (spec, have, args, derivative) in cases
        src = jax_source(spec; have, want = :total)
        text = string(src)
        # Each store converts its value to the array's dtype explicitly.
        @test occursin(".set(jnp.asarray(", text)
        # The loops stay loops.
        spec === jax_store_slots || @test occursin("jax.lax.fori_loop", text)
        # A float32 module's floats are all float32: it holds no float64.
        src32 = jax_source(spec; have, want = :total, float_type = Float32)
        @test !occursin("float64", string(src32))
        native = prepare(spec; have, want = :total)(args...)
        if !rk_jax_available()
            @info "RK_JAX_PYTHON is not set: typed-store JAX parity skipped"
            @test_skip false
            continue
        end
        jax, single = withenv("PYTHONWARNINGS" => JAX_STORE_STRICT) do
            rk_jax_evaluate(src, :kernel, args; grad = true),
            rk_jax_float32(src32, nothing, :kernel, args)
        end
        @test jax.value ≈ native rtol = 1e-6 atol = 1e-6
        @test jax.grad ≈ derivative rtol = 1e-5 atol = 1e-6
        @test !single.f64
        @test single.value ≈ native rtol = 1e-5 atol = 1e-5
        @test single.grad ≈ derivative rtol = 1e-4 atol = 1e-5
    end
end
