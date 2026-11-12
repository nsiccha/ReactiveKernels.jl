using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@traceable function jax_host_apply(f, x)
    g = f
    g(x)
end
@traceable function jax_host_fold(f, xs)
    total = 0.0
    for x in xs
        total += jax_host_apply(f, x)
    end
    total
end
@traceable jax_host_convert(T, x) = convert(T, x)
jax_host_capture(b, xs) = jax_host_fold(x -> b * x, xs)
@traceable function jax_host_pair_apply(pair, xs)
    f = getfield(pair, 1)
    b = getfield(pair, 2)
    b * jax_host_fold(f, xs)
end
jax_host_pair(b, xs) = jax_host_pair_apply((exp, b), xs)
@kernel jax_host_functions(a::Float64, xs::Vector{Float64}) = begin
    first = jax_host_fold(exp, xs)
    second = jax_host_fold(sin, xs)
    total = a * (first + second)
    return total
end
@kernel jax_host_captured(a::Float64, b::Float64, xs::Vector{Float64}) = begin
    result = jax_host_capture(b, xs)
    total = a * result
    return total
end
@kernel jax_host_aggregate(a::Float64, b::Float64, xs::Vector{Float64}) = begin
    result = jax_host_pair(b, xs)
    total = a * result
    return total
end
@kernel jax_host_types(a::Float64, x::Float32) = begin
    result = jax_host_convert(Float64, x)
    total = a * result
    return total
end
@kernel jax_host_sqrt(a::Float64, x::Float64) = begin
    value = 1 / sqrt(exp(x))
    total = a * value
    return total
end
@kernel jax_host_scan(a::Float64, xs::Vector{Float64}) = begin
    bounds = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    value = isempty(xs) ? 0.0 : 1 / sqrt(exp(bounds[end]))
    total = a * value
    return total
end
@kernel jax_host_ragged(a::Float64, xs::Vector{Vector{Float64}}) = begin
    cells = plate(xs) do row
        jax_host_fold(exp, row)
    end
    total = a * sum(cells)
    return total
end

@testset "jax_source: host helper function and type arguments" begin
    specs = ((jax_host_functions, (:a, :xs), (:xs,)),
             (jax_host_captured, (:a, :b, :xs), (:b, :xs)),
             (jax_host_aggregate, (:a, :b, :xs), (:b, :xs)),
             (jax_host_scan, (:a, :xs), (:xs,)))
    lengths = (0, 1, 19, 73)
    for (spec, have, data) in specs
        sources = String[]
        for n in lengths
            xs = collect(range(0.1, 0.3; length = max(2, n)))[1:n]
            bound = :b in have ? (; b = 0.7 + n / 100, xs) : (; xs)
            source = jax_source(spec; have, want = :total, data = bound)
            push!(sources, string(source))
            @test !occursin(r"<(function|callable|closure|type)", string(source))
            @test occursin(r"\bfor\b", string(source))
            if rk_jax_available()
                args = (1.7, values(bound)...)
                original = deepcopy(args)
                native = prepare(spec; have, want = :total)
                r = rk_jax_evaluate(source, :kernel, args; grad = true, prepare = :arguments)
                @test r.value ≈ native(args...)
                # Ordinary native Reverse through the same prepared kernel.
                derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                    Enzyme.Active(first(args)), map(Enzyme.Const, Base.tail(args))...)[1][1]
                @test only(r.grad) ≈ derivative
                @test args == original
            end
        end
        # Data lengths, numeric captures and scan capacities stay runtime inputs.
        @test all(==(first(sources)), sources)
    end
    if !rk_jax_available()
        @test_skip false
    else
        for x in (-3.0, 0.25, 800.0)
            source = jax_source(jax_host_sqrt; have = (:a, :x), want = :total, data = (:x,))
            native = prepare(jax_host_sqrt; have = (:a, :x), want = :total)
            r = rk_jax_evaluate(source, :kernel, (2.0, x); grad = true, prepare = true)
            @test r.value ≈ native(2.0, x)
            @test only(r.grad) ≈ native(1.0, x)
            device = jax_source(jax_host_sqrt; have = (:x, :a), want = :total)
            d = rk_jax_evaluate(device, :kernel, (x, 2.0); grad = true)
            native_device = prepare(jax_host_sqrt; have = (:x, :a), want = :total)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native_device, Enzyme.Active,
                Enzyme.Active(x), Enzyme.Const(2.0))[1][1]
            @test d.value ≈ r.value
            @test only(d.grad) ≈ derivative
        end
        typed = jax_source(jax_host_types; have = (:a, :x), want = :total, data = (:x,))
        @test rk_jax_evaluate(typed, :kernel, (2.0, 0.25f0); prepare = true).value == 0.5
        ragged = jax_source(jax_host_ragged; have = (:a, :xs), want = :total, data = (:xs,))
        rows = [Float64[], [0.2], [0.1, -0.2, 0.5]]
        r = rk_jax_evaluate(ragged, :kernel, (2.0, rows); grad = true, prepare = :arguments)
        expected = sum(row -> sum(exp, row; init = 0.0), rows)
        @test r.value ≈ 2 * expected
        @test only(r.grad) ≈ expected
    end
end

# A trailing Vararg formal (snag traceable-on-a-v-e940f11c): the recorded
# source reads it as the tuple of the trailing arguments, in `prepare_data`
# (a data-only design merging several time vectors) and in the functions.
# `sum(f, args[; init])` folds the tuple's fixed length.
@traceable function jax_host_union(args::Vararg{Any,N}) where {N}
    out = Float64[]
    for a in args
        for v in a
            push!(out, v)
        end
    end
    unique!(sort!(out))
end
@traceable jax_host_count(args...) = sum(length, args; init = 0)
@traceable jax_host_rest_scale(w, rest...) = w * (sum(sum, rest) + length(rest))
@kernel jax_host_vararg(a::Float64, t1::Vector{Float64}, t2::Vector{Float64},
                        t3::Vector{Float64}) = begin
    times = jax_host_union(t1, t2, t3)
    n_points = jax_host_count(t1, t2, t3)
    spread = jax_host_rest_scale(a, times, t1)
    total = spread + length(times) / n_points
    return total
end

@testset "jax_source: a @traceable helper with a Vararg argument" begin
    data = ([0.0, 0.5, 2.0], [0.5, 1.0], [3.0, 0.0, 1.0, 0.25])
    native = prepare(jax_host_vararg; have = (:a, :t1, :t2, :t3), want = :total)
    expected = native(1.3, data...)
    times = sort!(unique!(vcat(data...)))
    @test expected ≈ 1.3 * (sum(times) + sum(data[1]) + 2) + length(times) / 9
    prepared = jax_source(jax_host_vararg; have = (:a, :t1, :t2, :t3), want = :total,
                          data = (:t1, :t2, :t3))
    # The merge runs once per dataset on the host; the scale, which reads the
    # parameter, in the functions.
    @test occursin("def prepare_data", string(prepared))
    # The loop over each argument's data stays a loop.
    @test occursin(r"\bfor\b", string(prepared))
    if !rk_jax_available()
        @test_skip false
    else
        r = rk_jax_evaluate(prepared, :kernel, (1.3, data...); grad = true,
                            prepare = :arguments)
        @test r.value ≈ expected
        derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
            Enzyme.Active(1.3), map(Enzyme.Const, data)...)[1][1]
        @test only(r.grad) ≈ derivative
    end
end
