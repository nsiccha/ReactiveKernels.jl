using ReactiveKernels
using Test

include(joinpath(@__DIR__, "jax_python.jl"))

# JAX source emission (todo 1l3nni7): the emitted module keeps the program's
# structure, plates, scans, lazy branches and loops, and computes the native
# values; `jax.grad` of it matches the native program's finite differences.

@kernel jax_gaussian(x::Vector{Float64}, mu::Float64, logsigma::Float64) = begin
    sigma = exp(logsigma)
    z = (x .- mu) ./ sigma
    lp = -0.5 * sum(abs2, z) - length(x) * (logsigma + 0.5 * log(2pi))
    return lp
end

@kernel jax_plate_scan(y::Vector{Float64}, x::Vector{Float64}, a::Float64, b::Float64,
                       logsigma::Float64) = begin
    sigma = exp(logsigma)
    lp = plate(y, x) do yi, xi
        if sigma > 0
            -0.5 * ((yi - a - b * xi) / sigma)^2 - log(sigma)
        else
            -Inf
        end
    end
    filtered = scan(y; init = 0.0) do acc, yi
        next = 0.9 * acc + b * yi
        (next, next)
    end
    total = sum(lp) + sum(filtered)
    return total
end

@traceable function jax_test_recurrence(R::AbstractVector, rho::Real, kappa::Real)
    T = promote_type(eltype(R), typeof(rho), typeof(kappa))
    W = Vector{T}(undef, length(R))
    C, D = one(T), zero(T)
    recover = 1 - exp(-rho)
    for t in eachindex(R)
        C = C + (1 - C) * recover
        dep = C * (1 - exp(-kappa * R[t]))
        C = C - dep
        W[t] = D + dep
        D = 0.5 * D + dep
    end
    return W
end

@kernel jax_loop(rho::Float64, R::Vector{Float64}, kappa::Float64) = begin
    W = jax_test_recurrence(R, rho, kappa)
    s = sum(abs2, W)
    return s
end

@traceable jax_test_count(x::AbstractVector, n::Integer) = begin
    acc = zero(eltype(x))
    for i in 1:n
        acc += x[i]
    end
    acc
end
@kernel jax_data_count(x::Vector{Float64}, n::Int) = begin
    s = jax_test_count(x, n)
    return s
end

@testset "jax_source: structure" begin
    src = jax_source(jax_plate_scan; have = (:a, :y, :x, :b, :logsigma), want = :total)
    text = string(src)
    @test only(src.functions).name == "kernel"
    @test first.(only(src.functions).inputs) == ["a", "y", "x", "b", "logsigma"]
    @test occursin("jax.vmap(", text)
    @test occursin("jax.lax.scan(", text)
    @test occursin("jax.lax.cond(", text)
    # Nothing iterates in Python: every loop is a JAX loop primitive, so the
    # module is the same program for every data length.
    @test !occursin(r"^\s*for "m, text)
    loop = string(jax_source(jax_loop; have = (:rho, :R, :kappa), want = :s))
    @test occursin("jax.lax.fori_loop(", loop)
    @test occursin(".at[", loop)
    @test !occursin(r"^\s*for "m, loop)
    # Several functions over one boundary, with module constants.
    multi = jax_source(jax_gaussian; have = (:mu, :x, :logsigma),
                       functions = (log_density = :lp, both = (:lp, :sigma),
                                    named = (density = :lp, scale = :sigma)),
                       constants = (DIMENSION = 2, NAMES = ["mu", "logsigma"]))
    @test [f.name for f in multi.functions] == ["log_density", "both", "named"]
    @test occursin("DIMENSION = 2", string(multi))
    @test occursin("return {\"density\": ", string(multi))
end

@testset "jax_source: refused constructs" begin
    # refused: a loop whose trip count is data has no static bound, and JAX
    # arrays and loops need static shapes (docs/src/jax.md, "Limits").
    @test_throws JAXEmissionError jax_source(jax_data_count; have = (:x, :n), want = :s)
    err = try
        jax_source(jax_data_count; have = (:x, :n), want = :s)
    catch e
        e
    end
    @test occursin("trip count depends on data", sprint(showerror, err))
end

@testset "jax_source: values and gradients" begin
    if !rk_jax_available()
        @info "RK_JAX_PYTHON is not set: JAX value and gradient parity checks skipped"
        @test_skip false
    else
        x = [0.3, -1.2, 0.8, 2.1, -0.4]
        native = prepare(jax_gaussian; have = (:mu, :x, :logsigma), want = :lp)
        src = jax_source(jax_gaussian; have = (:mu, :x, :logsigma), want = :lp)
        r = rk_jax_evaluate(src, "kernel", (0.2, x, -0.3); grad = true)
        @test r.value ≈ native(0.2, x, -0.3) rtol = 1e-12
        @test r.grad ≈ sum((x .- 0.2) ./ exp(-0.3)^2) rtol = 1e-10

        y = [0.5, -0.1, 1.3, 0.2]
        native = prepare(jax_plate_scan; have = (:a, :y, :x, :b, :logsigma), want = :total)
        src = jax_source(jax_plate_scan; have = (:a, :y, :x, :b, :logsigma), want = :total)
        r = rk_jax_evaluate(src, "kernel", (0.1, y, x[1:4], 0.7, -0.2); grad = true)
        f(a) = native(a, y, x[1:4], 0.7, -0.2)
        @test r.value ≈ f(0.1) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> f(only(u)), [0.1])) rtol = 1e-6

        R = [0.0, 1.2, 0.3, 0.0, 2.5, 0.1]
        native = prepare(jax_loop; have = (:rho, :R, :kappa), want = :s)
        src = jax_source(jax_loop; have = (:rho, :R, :kappa), want = :s)
        r = rk_jax_evaluate(src, "kernel", (0.7, R, 0.3); grad = true)
        g(rho) = native(rho, R, 0.3)
        @test r.value ≈ g(0.7) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> g(only(u)), [0.7])) rtol = 1e-6
    end
end
