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

@kernel jax_generators(w::Float64, x::Vector{Float64}) = begin
    s = sum(w * x[i]^2 for i in 1:length(x); init = 0.0)
    v = [exp(w * x[i]) for i in eachindex(x)]
    t = s + sum(v) + sum(x[i] for i in eachindex(x) if x[i] > 0)
    return t
end

# A helper written for the compiled path: `@trace` control flow, a counted
# `while` loop (a `fori_loop`, which `jax.grad` reverses), and a branch whose
# arms assign an outer local.
@traceable function jax_test_traced(x::Real)
    i = 0
    h = one(x)
    ReactiveKernels.ReactantCore.@trace while i < 20
        h = h * (1 + x / (i + 1))
        i = i + 1
    end
    result = zero(x)
    ReactiveKernels.ReactantCore.@trace if x > 0
        result = h
    else
        result = -h
    end
    return result
end
@kernel jax_traced(x::Float64) = begin
    y = jax_test_traced(x)
    return y
end
# A `while` loop whose trip count is data: `jax.lax.while_loop`.
@traceable function jax_test_doubling(x::Real)
    h = one(x)
    n = 0
    while h < x
        h = 2 * h
        n = n + 1
    end
    return n * h
end
@kernel jax_doubling(x::Float64) = begin
    y = jax_test_doubling(x)
    return y
end

# Typed allocations, `hcat`, `permutedims`, a numeric-type broadcast, and a
# postfix on a compound operand (`size(x .- w, 1)`).
@kernel jax_alloc(x::Vector{Float64}, w::Float64) = begin
    n = length(x)
    D = Float64.(hcat(ones(Float64, n), x))
    P = permutedims(D)
    c = fill(w, 2)
    s = sum(D * c) + sum(P .* w) + size(x .- w, 1)
    return s
end

# A loop whose `if` arms end in values of different types (an assignment, a
# destructuring): the branch carries only the locals they assign.
@traceable function jax_test_triangle(z::AbstractVector, k::Integer)
    out = ones(Float64, k)
    i, j, d = 1, 2, 1.0
    for p in 1:length(z)
        d = d * sqrt(1 - z[p]^2)
        if i + 1 < j
            i = i + 1
        else
            out[j] = d
            i, j, d = 1, j + 1, 1.0
        end
    end
    return out
end
@kernel jax_triangle(z::Vector{Float64}) = begin
    v = jax_test_triangle(tanh.(z), 3)
    s = sum(log.(v))
    return s
end

# A scan whose step outputs named numbers, read field by field by a plate.
@kernel jax_named_scan(y::Vector{Float64}, b::Float64) = begin
    rows = scan(y; init = 0.0) do acc, yi
        next = b * acc + yi
        (next, (level = next, step = yi - next))
    end
    level = plate(rows) do row
        row.level
    end
    s = sum(level .^ 2)
    return s
end

# Runtime checks: one under a branch that a vmapped plate cell evaluates for
# every lane (it must hold for lanes that do not take the branch), and ones
# on the data.
@traceable function jax_test_positive_log(xi::Real)
    if xi > 0
        ReactiveKernels._runtime_check(xi > 0, ArgumentError("xi must be positive"))
        log(xi)
    else
        zero(xi)
    end
end
@traceable function jax_test_checked_sum(lp::AbstractVector)
    ReactiveKernels._runtime_check(length(lp) > 0, ArgumentError("lp is empty"))
    m = sum(lp)
    ReactiveKernels._runtime_check(m < 10, ArgumentError("m is too large"))
    return m
end
@kernel jax_checked(x::Vector{Float64}) = begin
    lp = plate(x) do xi
        jax_test_positive_log(xi)
    end
    m = jax_test_checked_sum(lp)
    return m
end

# A plain helper whose compiled form is authored in place (`@trace`): its
# source is recorded for `jax_source`, and it gains no `traced` method.
"The logarithm above one, the offset below it."
ReactiveKernels.@_record_source function jax_test_recorded(x)
    result = zero(x)
    ReactiveKernels.ReactantCore.@trace if x > 1
        result = log(x)
    else
        result = x - 1
    end
    return result
end
@kernel jax_recorded(x::Float64) = begin
    y = jax_test_recorded(x)
    return y
end

# A straight-line plain helper, translated from its lowered code: its literal
# powers and `eltype` are type values there (`Val{2}`, `Float64`), which bind
# no Python local.
jax_test_plain(w, x) = zero(eltype(x)) + sum(x) * w^2 + w^-1
@kernel jax_plain(w::Float64, x::Vector{Float64}) = begin
    s = jax_test_plain(w, x)
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

# Data preparation whose shapes follow the data's values (decision 1ooufd4):
# with `data`, it is `prepare_data`, host NumPy with Python control flow.
# Helpers whose loops `@traceable`'s traced form cannot carry (`break`,
# `continue`, several ranges) record their source with `@_record_source`, as
# RKPPL's own preparation helpers do.
ReactiveKernels.@_record_source function jax_prep_codes(x::AbstractVector, levels::AbstractVector)
    codes = Vector{Int}(undef, length(x))
    for (i, v) in enumerate(x)
        c = 0
        for (j, lv) in enumerate(levels)
            if isequal(v, lv)
                c = j
                break
            end
        end
        codes[i] = c
    end
    return codes
end
ReactiveKernels.@_record_source function jax_prep_lanes(y::AbstractVector{<:Integer}, K)
    out = Int[]
    for (i, count) in enumerate(min.(y, K .- 1)), j in 1:count
        push!(out, i)
    end
    return out
end
ReactiveKernels.@_record_source function jax_prep_first(keys::AbstractVector)
    seen = Dict{eltype(keys),Int}()
    codes = zeros(Int, length(keys))
    for i in eachindex(keys)
        haskey(seen, keys[i]) && (codes[i] = seen[keys[i]]; continue)
        codes[i] = get!(seen, keys[i], length(seen) + 1)
    end
    return codes
end
ReactiveKernels.@_record_source function jax_prep_search(x::AbstractVector, target)
    found = 0
    i = 1
    while i <= length(x)
        if x[i] >= target
            found = i
            break
        end
        i += 1
    end
    return found
end
@kernel jax_prep(g::Vector{String}, y::Vector{Int}, theta::Vector{Float64}) = begin
    codes = jax_prep_codes(g, ["a", "b", "c"])
    lanes = jax_prep_lanes(y, 3)
    first = jax_prep_first(g)
    at = jax_prep_search(y, 2)
    total = sum(theta[codes]) + 2 * sum(theta[first]) + sum(lanes) * theta[2] + at * theta[1]
    return total
end
# A ragged value read through a plate, then flattened.
@kernel jax_prep_flat(R::Vector{Vector{Float64}}, a::Float64) = begin
    scaled = plate(R) do r
        a .* r
    end
    flat = reduce(vcat, scaled)
    total = sum(abs2, flat)
    return total
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
    generators = string(jax_source(jax_generators; have = (:w, :x), want = :t))
    @test occursin("jax.vmap(element", generators)
    @test occursin("jnp.where(kept", generators)
    @test !occursin(r"^\s*for "m, generators)
    @test ReactiveKernels._traceable_source(typeof(jax_test_recorded), Float64) !== nothing
    @test !any(m -> occursin("jax_test_recorded", string(m.sig)), methods(ReactiveKernels.traced))
    @test occursin("logarithm above one", string(@doc jax_test_recorded))
    checked = string(jax_source(jax_checked; have = (:x,), want = :m))
    @test occursin("jax.debug.callback(", checked)
    @test occursin("_rk_check(jnp.logical_or(", checked)
    traced = string(jax_source(jax_traced; have = (:x,), want = :y))
    @test occursin("jax.lax.fori_loop(", traced)
    @test occursin("jax.lax.cond(", traced)
    @test occursin("jax.lax.while_loop(", string(jax_source(jax_doubling; have = (:x,), want = :y)))
    plain = string(jax_source(jax_plain; have = (:w, :x), want = :s))
    @test occursin("w ** 2", plain)
    @test !occursin(r"<[a-z]", plain)
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

        native = prepare(jax_generators; have = (:w, :x), want = :t)
        src = jax_source(jax_generators; have = (:w, :x), want = :t)
        r = rk_jax_evaluate(src, "kernel", (0.4, x); grad = true)
        @test r.value ≈ native(0.4, x) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x), [0.4])) rtol = 1e-6

        native = prepare(jax_traced; have = (:x,), want = :y)
        src = jax_source(jax_traced; have = (:x,), want = :y)
        @test rk_jax_evaluate(src, "kernel", (0.3,)).value ≈ native(0.3) rtol = 1e-12
        @test rk_jax_evaluate(src, "kernel", (-0.3,)).value ≈ native(-0.3) rtol = 1e-12
        @test rk_jax_evaluate(src, "kernel", (0.3,); grad = true).grad ≈
              only(rk_fd_gradient(u -> native(only(u)), [0.3])) rtol = 1e-6
        native = prepare(jax_doubling; have = (:x,), want = :y)
        src = jax_source(jax_doubling; have = (:x,), want = :y)
        @test rk_jax_evaluate(src, "kernel", (37.0,)).value ≈ native(37.0) rtol = 1e-12
        native = prepare(jax_plain; have = (:w, :x), want = :s)
        src = jax_source(jax_plain; have = (:w, :x), want = :s)
        r = rk_jax_evaluate(src, "kernel", (0.7, x); grad = true)
        @test r.value ≈ native(0.7, x) rtol = 1e-12
        @test r.grad ≈ 2 * 0.7 * sum(x) - 0.7^-2 rtol = 1e-10

        for (kernel, have, want, args) in (
                (jax_alloc, (:x, :w), :s, (x, 0.7)),
                (jax_triangle, (:z,), :s, ([0.3, -0.5, 0.2],)),
                (jax_named_scan, (:y, :b), :s, (y, 0.7)),
                (jax_checked, (:x,), :m, (x,)),
                (jax_recorded, (:x,), :y, (2.0,)), (jax_recorded, (:x,), :y, (0.5,)))
            native = prepare(kernel; have, want)
            src = jax_source(kernel; have, want)
            @test rk_jax_evaluate(src, "kernel", args).value ≈ native(args...) rtol = 1e-12
        end
        # A check that fails on the data raises when the module runs.
        src = jax_source(jax_checked; have = (:x,), want = :m)
        @test_throws "m is too large" rk_jax_evaluate(src, "kernel", ([5e4, 3e5],))

        R = [0.0, 1.2, 0.3, 0.0, 2.5, 0.1]
        native = prepare(jax_loop; have = (:rho, :R, :kappa), want = :s)
        src = jax_source(jax_loop; have = (:rho, :R, :kappa), want = :s)
        r = rk_jax_evaluate(src, "kernel", (0.7, R, 0.3); grad = true)
        g(rho) = native(rho, R, 0.3)
        @test r.value ≈ g(0.7) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> g(only(u)), [0.7])) rtol = 1e-6
    end
end

@testset "jax_source: prepare_data" begin
    g = ["b", "a", "c", "a", "b", "b"]
    y = [1, 3, 2, 0, 3, 1]
    src = jax_source(jax_prep; have = (:theta, :g, :y), want = :total, data = (; g, y))
    text = string(src)
    @test [f.name for f in src.functions] == ["prepare_data", "kernel"]
    @test first.(src.functions[1].inputs) == ["g", "y"]
    @test first(first.(src.functions[2].inputs)) == "theta"
    # The preparation is host code with real loops; the function is JAX code.
    prepare = text[findfirst("def prepare_data", text).start:end]
    @test occursin(r"^\s*for .* in enumerate\("m, text)
    @test occursin("break", text) && occursin(".append(", text) && occursin(".setdefault(", text)
    @test !occursin("jnp.", split(prepare, "\ndef kernel")[1])
    @test occursin("PREPARED = {", text)
    # Without `data`, the same preparation has no JAX translation.
    @test_throws JAXEmissionError jax_source(jax_prep; have = (:theta, :g, :y), want = :total)
    R = [[0.5, 1.0], [2.0], [0.1, 0.2, 0.3]]
    flat = jax_source(jax_prep_flat; have = (:a, :R), want = :total, data = (; R))
    @test "R_flat" in first.(flat.functions[2].inputs)
    @test occursin("jnp.nonzero(", string(flat))
    if !rk_jax_available()
        @test_skip false
    else
        theta = [0.3, -1.2, 2.0]
        native = sum(theta[jax_prep_codes(g, ["a", "b", "c"])]) +
                 2 * sum(theta[jax_prep_first(g)]) + sum(jax_prep_lanes(y, 3)) * theta[2] +
                 jax_prep_search(y, 2) * theta[1]
        r = rk_jax_evaluate(src, "kernel", (theta, g, y); grad = true, prepare = true)
        @test r.value ≈ native rtol = 1e-12
        @test r.grad ≈ rk_fd_gradient(t -> sum(t[jax_prep_codes(g, ["a", "b", "c"])]) +
                 2 * sum(t[jax_prep_first(g)]) + sum(jax_prep_lanes(y, 3)) * t[2] +
                 jax_prep_search(y, 2) * t[1], theta) rtol = 1e-6
        r = rk_jax_evaluate(flat, "kernel", (0.7, R); grad = true, prepare = true)
        @test r.value ≈ sum(abs2, 0.7 .* reduce(vcat, R)) rtol = 1e-12
        @test r.grad ≈ 2 * 0.7 * sum(abs2, reduce(vcat, R)) rtol = 1e-10
        # A loop whose trip count is data runs on the host.
        counted = jax_source(jax_data_count; have = (:x, :n), want = :s, data = (x = [1.0, 2.0, 4.0], n = 2))
        @test occursin("for i in range(1, n + 1):", string(counted))
    end
end
