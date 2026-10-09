using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# `@plate for` / `@scan` loop syntax desugars onto the authored plate and scan
# operations (user decision `05cugvn`): a loop prepares the same kernel as its
# call form, with reads at the loop index zipped and other values captured.

function _loop_program(k)
    names = Dict{String,String}()
    replace(string(readable_code(k)),
            r"var\"[^\"]*\"" => m -> get!(names, m, "v$(length(names) + 1)"))
end

@kernel _loop_standardize(y, mu, sigma) = begin
    @plate for i in eachindex(y, mu)
        z[i] = (y[i] - mu[i]) / sigma
    end
    return z
end
@kernel _call_standardize(y, mu, sigma) = begin
    z = plate(eachindex(y, mu), y, mu) do i, y_i, mu_i
        z_i = (y_i - mu_i) / sigma
        z_i
    end
    return z
end

# The domain iterates the one array the cells read: exactly `plate(c) do`, so a
# scan feeding the plate still streams through it.
@kernel _loop_streamed(xs::Vector{Float64}, scale) = begin
    cumulative = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    @plate for t in eachindex(cumulative)
        pointwise[t] = -0.5 * (cumulative[t] / scale)^2
    end
    total::Float64 = sum(pointwise)
    return total
end
@kernel _call_streamed(xs::Vector{Float64}, scale) = begin
    cumulative = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    pointwise = plate(cumulative) do cumulative_t
        pointwise_t = -0.5 * (cumulative_t / scale)^2
        pointwise_t
    end
    total::Float64 = sum(pointwise)
    return total
end

# A gather by a data index, a bare use of the index, two outputs (one typed)
# that share a cell local.
@kernel _loop_outputs(x, idx, w) = begin
    @plate for i in eachindex(idx)
        g = x[idx[i]]
        a[i]::Float64 = g * w[1] + i
        b[i] = a[i] + 1
    end
    return (a, b)
end

# A nested loop over each group's observations, including an empty group.
@kernel _loop_nested(groups, w) = begin
    @plate for g in eachindex(groups)
        obs = groups[g]
        @plate for j in eachindex(obs)
            s[j] = obs[j] * w[g]
        end
        tot[g] = sum(s; init = 0.0)
    end
    return tot
end

# A loop over a domain that is not the arrays' own iterator: the domain is a
# plate argument, checked against every array read at the index.
@kernel _loop_range(y, mu, n) = begin
    @plate for i in 1:n
        r[i] = y[i] - mu[i]
    end
    return r
end

# The bound reader of `test_plate_closures.jl` as a loop. `live` is also read
# whole, so `live[s]` stays a gather from the captured vector.
@kernel _loop_reader(live, kinds_by_subject, read_idx) = begin
    @plate for s in eachindex(kinds_by_subject, read_idx)
        kinds = kinds_by_subject[s]
        read_positions = findall(isone, kinds)
        observation_operations = read_positions[read_idx[s]]
        out[s] = sum(live[observation_operations]; init = 0.0) * live[s]
    end
    total::Float64 = sum(out)
end
@kernel _call_reader(live, kinds_by_subject, read_idx) = begin
    out = plate(eachindex(kinds_by_subject, read_idx), kinds_by_subject, read_idx) do s, kinds_by_subject_s, read_idx_s
        kinds = kinds_by_subject_s
        read_positions = findall(isone, kinds)
        observation_operations = read_positions[read_idx_s]
        out_s = sum(live[observation_operations]; init = 0.0) * live[s]
        out_s
    end
    total::Float64 = sum(out)
end

@kernel _scan_ar1(x, phi, s) = begin
    @scan begin
        a[1] = s
        for t in 2:length(x)
            a[t] = phi * a[t - 1] + x[t]
        end
    end
    return a
end
@kernel _call_ar1(x, phi, s) = begin
    a = scan(2:length(x); init = s, include_init = true) do a_prev, t
        a_t = phi * a_prev + x[t]
        (a_t, a_t)
    end
    return a
end

@kernel _scan_level(z, beta, sigma) = begin
    @scan begin
        level[1] = 0.0
        increment[1] = 0.0
        for t in 2:length(z)
            increment[t] = beta * increment[t - 1] + sigma * z[t]
            level[t] = level[t - 1] + increment[t]
        end
    end
    return (level, increment)
end

@kernel _scan_fib(n) = begin
    @scan begin
        f[1] = 1.0
        f[2] = f[1]
        for t in 3:n
            f[t] = f[t - 1] + f[t - 2]
        end
    end
    return f
end

@testset "@plate for loops" begin
    y, mu = [1.0, 2.0, 4.0], [0.5, 0.5, 1.0]
    @test prepare(_loop_standardize)(y, mu, 2.0) == (y .- mu) ./ 2.0
    @test _loop_program(prepare(_loop_standardize)) ==
          _loop_program(prepare(_call_standardize))

    xs = [0.5, -1.0, 2.0, 0.25]
    streamed = prepare(_loop_streamed)
    @test streamed(xs, 1.5) ≈ sum(v -> -0.5 * (v / 1.5)^2, cumsum(xs))
    @test _loop_program(streamed) == _loop_program(prepare(_call_streamed))
    @test !occursin("similar", _loop_program(streamed))

    x, idx = [10.0, 20.0, 30.0], [3, 1]
    @test prepare(_loop_outputs)(x, idx, [2.0]) == ([61.0, 22.0], [62.0, 23.0])
    @test prepare(_loop_nested)([[1.0, 2.0], Float64[], [3.0]], [1.0, 2.0, 3.0]) ==
          [3.0, 0.0, 9.0]

    @test prepare(_loop_range)([1.0, 2.0], [0.5, 1.0], 2) == [0.5, 1.0]
    # Rejected data: an array read at the loop index has the loop's indices; a
    # Julia loop would raise `BoundsError`, and a zipped singleton must not be
    # repeated across the domain instead.
    @test_throws DimensionMismatch prepare(_loop_range)([1.0, 2.0], [0.5], 2)
    @test_throws DimensionMismatch prepare(_loop_range)([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], 2)
end

@testset "@plate for under bound= and native Reverse" begin
    caches(p) = sort([only(r.outputs).name for r in p.recipes
                      if r.op isa ReactiveKernels._BoundConstant &&
                         startswith(String(only(r.outputs).name), "bound_plate_")])
    kinds = [[1, 2, 1, 1, 3], [2, 1, 1, 3, 1, 1], [3, 3]]
    read_idx = [[1, 3], [2, 4], Int[]]
    data = (; kinds_by_subject = kinds, read_idx)
    k = prepare(_loop_reader; bound = data)
    r = prepare(_call_reader; bound = data)
    @test !isempty(caches(k.plan))
    @test caches(k.plan) == caches(r.plan)
    @test _loop_program(k) == _loop_program(r)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    live = [0.5, -1.0, 2.0, 3.0, 0.0, 7.0]
    value_gradient(kernel) = ad_value_and_gradient(
        prepare_ad(kernel, backend, live; active = :live), live)
    @test value_gradient(k) == value_gradient(r)
    @test first(value_gradient(k)) == prepare(_loop_reader)(live, kinds, read_idx)
end

@testset "@scan loops" begin
    x = [0.0, 1.0, -0.5, 2.0]
    reference = foldl((a, t) -> push!(a, 0.7 * a[end] + x[t]), 2:4; init = [0.3])
    @test prepare(_scan_ar1)(x, 0.7, 0.3) ≈ reference
    @test _loop_program(prepare(_scan_ar1)) == _loop_program(prepare(_call_ar1))

    z = [0.0, 0.5, -1.0, 0.25]
    increment, level = [0.0], [0.0]
    for t in 2:4
        push!(increment, 0.9 * increment[end] + 0.1 * z[t])
        push!(level, level[end] + increment[end])
    end
    L, I = prepare(_scan_level)(z, 0.9, 0.1)
    @test L ≈ level
    @test I ≈ increment
    @test prepare(_scan_fib)(7) == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    @test prepare(_scan_fib)(2) == [1.0, 1.0]

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    @kernel _scan_total(x::Vector{Float64}, phi::Float64) = begin
        @scan begin
            a[1] = 0.0
            for t in 2:length(x)
                a[t] = phi * a[t - 1] + x[t]
            end
        end
        total::Float64 = sum(a)
    end
    phi = 0.6
    total(phi) = sum(foldl((a, t) -> push!(a, phi * a[end] + x[t]), 2:4; init = [0.0]))
    prepared = prepare_ad(prepare(_scan_total; want = :total), backend, x, phi; active = :phi)
    value, gradient = ad_value_and_gradient(prepared, x, phi)
    @test value ≈ total(phi)
    @test gradient ≈ (total(phi + 1e-6) - total(phi - 1e-6)) / 2e-6 rtol = 1e-6
end

@testset "loop syntax refusals" begin
    expand(ex) = try
        m = Module()
        Core.eval(m, :(using ReactiveKernels))
        Core.eval(m, ex)
        nothing
    catch err
        err isa LoadError ? err.error : err
    end
    # refused: outputs are written at exactly the loop index (decision 05cugvn)
    @test expand(:(@kernel _k(x) = begin
        @plate for i in eachindex(x)
            r[i + 1] = x[i]
        end
        return r
    end)) isa ArgumentError
    # refused: a loop with no output defines nothing (decision 05cugvn)
    @test expand(:(@kernel _k(x) = begin
        @plate for i in eachindex(x)
            v = x[i]
        end
        return x
    end)) isa ArgumentError
    # refused: a lag beyond the seeded depth has no carried value (decision 05cugvn)
    @test expand(:(@kernel _k(x) = begin
        @scan begin
            a[1] = 0.0
            for t in 2:length(x)
                a[t] = a[t - 2] + x[t]
            end
        end
        return a
    end)) isa ArgumentError
    # refused: a step reads the current value only after writing it (decision 05cugvn)
    @test expand(:(@kernel _k(x) = begin
        @scan begin
            a[1] = 0.0
            for t in 2:length(x)
                b = a[t]
                a[t] = b + x[t]
            end
        end
        return a
    end)) isa ArgumentError
    # refused: a carried array needs its seeds before the loop (decision 05cugvn)
    @test expand(:(@kernel _k(x) = begin
        @scan begin
            a[1] = 0.0
            for t in 2:length(x)
                a[t] = a[t - 1] + x[t]
                c[t] = a[t]
            end
        end
        return a
    end)) isa ArgumentError
    # Outside a @kernel body the macros only explain where they belong.
    @test expand(quote
        @plate for i in 1:2
            r[i] = i
        end
    end) isa ArgumentError
    @test expand(quote
        @scan begin
            a[1] = 0
            for t in 2:3
                a[t] = a[t - 1]
            end
        end
    end) isa ArgumentError
end
