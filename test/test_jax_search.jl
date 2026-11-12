using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# The positional kernel and its captured predicate are the public spelling
# used for grouped model labels. All preprocessing stays in the emitted module.
@kernel jax_label_indices(values, levels) = begin
    indices = plate(values) do value
        Int(findfirst(isequal(value), levels))
    end
    return indices
end
@kernel jax_label_total(theta::Float64, values, levels) = begin
    indices = jax_label_indices(values, levels)
    total = theta^2 * sum(indices)
    return total
end

@traceable function jax_first_or_zero(values, target)
    index = findfirst(isequal(target), values)
    index === nothing ? 0 : index
end
@kernel jax_search_total(theta::Float64, values, target) = begin
    index = jax_first_or_zero(values, target)
    total = theta * index
    return total
end

jax_search_guard(x) = x < 0 ? error("searched past the first match") : x == 2
@traceable function jax_guarded_search(values)
    index = findfirst(jax_search_guard, values)
    index === nothing ? 0 : index
end
@kernel jax_search_lazy(theta::Float64, values) = begin
    index = jax_guarded_search(values)
    total = theta * index
    return total
end

@kernel jax_fixed_predicates(theta::Float64, threshold, values) = begin
    first_below = findfirst(<(threshold), values)
    first_above = findfirst(Base.Fix1(<, threshold), values)
    total = theta * (Int(first_above) + Int(first_below))
    return total
end
@kernel jax_ragged_search(theta::Float64, rows, target) = begin
    indices = plate(rows) do row
        jax_first_or_zero(row, target)
    end
    total = theta * sum(indices)
    return total
end
@traceable function jax_identity_index(values)
    index = findfirst(identity, values)
    index === nothing ? 0 : index
end
@kernel jax_identity_search(theta::Float64, values) = begin
    index = jax_identity_index(values)
    total = theta * index
    return total
end

@testset "jax_source: host findfirst and captured predicates" begin
    values = repeat(["r1", "r2", "r3"], 10)
    levels = ["r1", "r2", "r3"]
    native = prepare(jax_label_total; have = (:theta, :values, :levels), want = :total)
    source = jax_source(jax_label_total; have = (:theta, :values, :levels), want = :total,
                        data = (; values, levels))
    text = string(source)
    @test occursin("enumerate(", text)
    @test occursin("break", text)
    @test !occursin("findfirst threw", text)
    @test !occursin("<closure>", text)
    # Both observation count and number of levels are data, never copies of
    # the cell or search body in RK emission.
    for n in (0, 1, 73)
        other = ["label-$i" for i in 1:max(1, n)]
        s = jax_source(jax_label_total; have = (:theta, :values, :levels), want = :total,
                       data = (values = other[1:n], levels = other))
        @test string(s) == text
    end
    if !rk_jax_available()
        @test_skip false
    else
        for (xs, ls) in ((values, levels), (["r3", "r1", "r3"], reverse(levels)),
                         (String[], levels), (["r1"], ["r1", "r1"]))
            for theta in (0.4, 1.3)
                result = rk_jax_evaluate(source, :kernel, (theta, xs, ls);
                                         grad = true, prepare = :arguments)
                @test result.value ≈ native(theta, xs, ls)
                derivative = (native(theta + 1e-6, xs, ls) - native(theta - 1e-6, xs, ls)) / 2e-6
                @test result.grad ≈ derivative rtol = 1e-8 atol = 1e-9
            end
        end
        @test_throws Exception native(0.4, ["absent"], levels)
        @test_throws ErrorException rk_jax_evaluate(source, :kernel, (0.4, ["absent"], levels);
                                                   prepare = true)
        for (xs, target, expected) in ((Float64[], 0.0, 0), ([0.0, -0.0, NaN], -0.0, 2),
                                      ([0.0, NaN, NaN], NaN, 2), ([1.0, 2.0], 3.0, 0),
                                      ([missing, "r1", missing], missing, 1))
            s = jax_source(jax_search_total; have = (:theta, :values, :target), want = :total,
                           data = (values = xs, target = target))
            result = rk_jax_evaluate(s, :kernel, (0.5, xs, target); grad = true, prepare = true)
            @test result.value == 0.5 * expected
            @test result.grad == expected
        end
        lazy = jax_source(jax_search_lazy; have = (:theta, :values), want = :total,
                          data = (values = [1, 2, -1],))
        @test rk_jax_evaluate(lazy, :kernel, (0.5, [1, 2, -1]); prepare = true).value == 1
        @test_throws ErrorException rk_jax_evaluate(lazy, :kernel, (0.5, [1, -1, 2]); prepare = true)
        @test rk_jax_evaluate(lazy, :kernel, (0.5, Int[]); prepare = true).value == 0
        fixed = jax_source(jax_fixed_predicates; have = (:theta, :threshold, :values), want = :total,
                           data = (threshold = 2.0, values = [3.0, 1.0]))
        for (threshold, xs, expected) in ((2.0, [3.0, 1.0], 3), (4.0, [3.0, 2.0, 5.0], 4))
            result = rk_jax_evaluate(fixed, :kernel, (0.5, threshold, xs); grad = true, prepare = :arguments)
            @test result.value == 0.5 * expected
            @test result.grad == expected
        end
        rows = [[1.0, 2.0, 3.0], Float64[], [3.0], [2.0, 2.0]]
        ragged = jax_source(jax_ragged_search; have = (:theta, :rows, :target), want = :total,
                            data = (rows = rows, target = 2.0))
        for (rs, expected) in ((rows, 3), (Vector{Float64}[], 0), ([Float64[], [2.0]], 1))
            result = rk_jax_evaluate(ragged, :kernel, (0.5, rs, 2.0); grad = true, prepare = :arguments)
            @test result.value == 0.5 * expected
            @test result.grad == expected
        end
        for xs in ([false, true, true], [0, 1, 1])
            s = jax_source(jax_identity_search; have = (:theta, :values), want = :total,
                           data = (values = xs,))
            if eltype(xs) === Bool
                @test rk_jax_evaluate(s, :kernel, (0.5, xs); prepare = true).value == 1
            else
                @test_throws TypeError findfirst(identity, xs)
                @test_throws ErrorException rk_jax_evaluate(s, :kernel, (0.5, xs); prepare = true)
                @test rk_jax_evaluate(s, :kernel, (0.5, Int[]); prepare = true).value == 0
            end
        end
    end
end
