module TestJSIterators

# `js_source` of lazy iterators: comprehensions with a condition, generators,
# `Iterators.filter`, and the functions that consume them, each module checked
# in Node against the native kernel.

using ReactiveKernels
using Test

const NODE = ReactiveKernels._js_default_node() !== nothing

# Plain helpers, translated from their typed code: an index selection by a
# threshold, the nearest admissible entry to a target (keyword default, early
# return, `isempty`, `argmin`), and the highest admissible index (`init`).
js_admissible(idx, score, limit) = [k for k in idx if score[k] <= limit]
function js_nearest(target, value, score, idx; limit = 0.9)
    ok = [k for k in idx if score[k] <= limit]
    isempty(ok) && return 0
    return ok[argmin(abs.(value[ok] .- target))]
end
js_highest(score, idx, limit) = maximum((k for k in idx if score[k] <= limit); init = 0)

# `findmin`/`findmax` give a tuple, read here by field.
js_findmin_value(x) = getfield(findmin(x), 1)
js_findmin_index(x) = getfield(findmin(x), 2)
js_findmax_value(x) = getfield(findmax(x), 1)
js_findmax_index(x) = getfield(findmax(x), 2)

@kernel js_iter_select(idx, score, value, target, limit) = begin
    kept = js_admissible(idx, score, limit)
    nearest = js_nearest(target, value, score, idx)
    highest = js_highest(score, idx, limit)
    return nearest
end
# Reductions of generators, filters and vectors. Julia reduces an empty
# filter or a generator of `identity` to the operation's identity, and any other
# empty generator to an error.
js_total(x) = sum(v for v in x)
js_kept(x) = sum(v for v in x if v > 0)
js_finite(x) = sum(Iterators.filter(isfinite, x))
js_shifted(x) = sum((v for v in x); init = 1.0)
js_positive_squares(x) = sum(v^2 for v in x if v > 0)
js_positives(x) = sum(v > 0 for v in x)
js_smallest(x) = minimum(abs(v) for v in x)
js_capped(x) = maximum((v for v in x if v < 0); init = -Inf)
js_least(x) = minimum((2v for v in x); init = 10.0)
js_npositive(x) = count(v -> v > 0, x)
js_nlarge(x) = count(abs(v) > 1 for v in x)
js_anylarge(x) = any(v -> v > 2, x)
js_allsmall(x) = all(abs(v) < 5 for v in x)
js_nonehigh(x) = isempty(v for v in x if v > 10)
js_nonan(x) = isempty(Iterators.filter(isnan, x))
js_odd_squares(n) = sum(k^2 for k in 1:n if k % 2 == 1)
js_thirds(n) = [k for k in 1:n if k % 3 == 0]
js_nested(x) = sum(y for y in (2v for v in x) if y > 1)
js_shifted_positive(x) = [y + 1 for y in (v for v in x if v > 0)]
js_positive(x) = filter(v -> v > 0, x)

@kernel js_iter_sums(x) = begin
    inline = sum(v for v in x if v != 1) + count(v -> v > 1, x)
    total = js_total(x)
    kept = js_kept(x)
    finite = js_finite(x)
    shifted = js_shifted(x)
    folded = sum(x; init = 0.0)
    return total
end
@kernel js_iter_mapped_sums(x) = begin
    positive_squares = js_positive_squares(x)
    positives = js_positives(x)
    return positives
end
@kernel js_iter_extrema(x) = begin
    smallest = js_smallest(x)
    capped = js_capped(x)
    top = maximum(x; init = 0.0)
    least = js_least(x)
    return smallest
end
@kernel js_iter_counts(x) = begin
    npositive = js_npositive(x)
    nlarge = js_nlarge(x)
    anylarge = js_anylarge(x)
    allsmall = js_allsmall(x)
    nonehigh = js_nonehigh(x)
    nonan = js_nonan(x)
    blank = isempty(x)
    return npositive
end
@kernel js_iter_ranges(n) = begin
    odd_squares = js_odd_squares(n)
    thirds = js_thirds(n)
    none = isempty(1:n)
    return thirds
end
@kernel js_iter_nested(x) = begin
    nested = js_nested(x)
    shifted = js_shifted_positive(x)
    positive = js_positive(x)
    return nested
end
@kernel js_iter_find(x) = begin
    minvalue = js_findmin_value(x)
    minindex = js_findmin_index(x)
    maxvalue = js_findmax_value(x)
    maxindex = js_findmax_index(x)
    lowest = argmin(x)
    highest = argmax(x)
    return lowest
end
# A condition that is not a `Bool`, and a generator of mixed element types.
js_truthy(x) = [v for v in x if v]
js_mixed(x) = Float64(sum(v > 0 ? v : 0 for v in x))
@kernel js_iter_truthy(x) = begin
    y = js_truthy(x)
    return y
end
@kernel js_iter_mixed(x) = begin
    y = js_mixed(x)
    return y
end
@kernel js_iter_find_int(k) = begin
    lowest = argmin(k)
    highest = argmax(k)
    return lowest
end

const V = Vector{Float64}
const SPREAD = [sin(k) * 10.0^(k % 7) for k in 1:3001]
const CASES = [
    (:js_iter_select, js_iter_select, (:idx, :score, :value, :target, :limit), (:kept, :nearest, :highest),
     (; idx = Vector{Int}, score = V, value = V, target = Float64, limit = Float64),
     [([1, 2, 3, 4, 5], [0.0, 0.0, 0.4, 0.8, 1.0], [0.05, 0.1, 0.25, 0.4, 0.5], 0.3, 0.5),
      ([1, 2, 3, 4, 5], [0.0, 0.0, 0.4, 0.8, 1.0], [0.05, 0.1, 0.25, 0.4, 0.5], 0.3, 0.95),
      ([5, 3, 1], [0.0, 0.0, 0.4, 0.8, 1.0], [0.05, 0.1, 0.25, 0.4, 0.5], 0.0, 0.5),
      ([4, 5], [0.0, 0.0, 0.4, 0.95, 1.0], [0.05, 0.1, 0.25, 0.4, 0.5], 0.3, 0.5),
      (Int[], Float64[], Float64[], 0.3, 0.5),
      ([1, 2, 3], [0.0, 0.0], [0.1, 0.2, 0.3], 0.3, 0.5)]),          # score[3]: a BoundsError
    (:js_iter_sums, js_iter_sums, (:x,), (:total, :kept, :finite, :shifted, :folded, :inline), (; x = V),
     [(SPREAD,), ([-0.0],), ([1.0, NaN, -2.0, Inf],), (Float64[],)]),
    (:js_iter_mapped_sums, js_iter_mapped_sums, (:x,), (:positive_squares, :positives), (; x = V),
     [(SPREAD,), ([1.0, -2.0],), ([-1.0],), (Float64[],)]),
    (:js_iter_extrema, js_iter_extrema, (:x,), (:smallest, :capped, :top, :least), (; x = V),
     [([1.5, -2.0, 0.25, -0.5],), ([3.0, 2.0],), ([-1.0, NaN],), ([-0.0, 0.0],), (Float64[],)]),
    (:js_iter_counts, js_iter_counts, (:x,), (:npositive, :nlarge, :anylarge, :allsmall, :nonehigh, :nonan, :blank),
     (; x = V), [([1.5, -2.0, 0.25, 3.0],), ([0.5, -0.5, 11.0],), ([NaN, 1.0],), ([-6.0],), (Float64[],)]),
    (:js_iter_ranges, js_iter_ranges, (:n,), (:odd_squares, :thirds, :none), (; n = Int), [(10,), (1,), (0,), (-3,)]),
    (:js_iter_nested, js_iter_nested, (:x,), (:nested, :shifted, :positive), (; x = V),
     [([1.0, -0.25, 0.75, 2.0],), ([0.5],), (Float64[],)]),
    (:js_iter_find, js_iter_find, (:x,), (:minvalue, :minindex, :maxvalue, :maxindex, :lowest, :highest), (; x = V),
     [([2.0, 0.5, 0.5, 3.0, 3.0],), ([1.0, NaN, 0.5, NaN],), ([-0.0, 0.0],), ([0.0, -0.0],), ([-Inf, Inf],),
      (Float64[],)]),
    (:js_iter_find_int, js_iter_find_int, (:k,), (:lowest, :highest), (; k = Vector{Int}),
     [([3, 1, 1, 7, 7],), ([4],), (Int[],)]),
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) ||
    (iszero(a) ? iszero(b) && signbit(a) == signbit(b) : isapprox(a, b; rtol = 1e-12, atol = 1e-15))
agree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(agree.(a, b))
agree(a, b) = false

# The code of one function of a module.
function body(source, name)
    i = findfirst("function " * name * "(", source)
    source[i[1]:findnext("\n}\n", source, i[1])[end]]
end

@testset "js_source of lazy iterators" begin
    @testset "Node agrees with native: $name ($vectors)" for (name, k, have, want, types, cases) in CASES,
                                                             vectors in (:array, :float64array)
        source = ReactiveKernels.js_source(k; have, want, types, name)
        if !NODE
            @test_skip "no node on the PATH"
            continue
        end
        native = prepare(k; have, want)
        outcomes = ReactiveKernels.js_run(source, name, cases; vectors)
        for (c, r) in zip(cases, outcomes)
            expected = try
                native(c...)
            catch e
                e
            end
            @test !r.inputs_mutated
            if expected isa Exception
                @test r.julia_error && r.error == string(nameof(typeof(expected)))
            else
                @test r.error === nothing
                r.error === nothing || continue
                @test typeof(r.value) == typeof(expected)
                @test all(agree(expected[j], r.value[j]) for j in eachindex(want))
            end
        end
    end

    @testset "a filtered comprehension is one loop" begin
        source = ReactiveKernels.js_source(js_iter_select; have = (:idx, :score, :value, :target, :limit),
                                           want = :kept, types = (; idx = Vector{Int}, score = V, value = V,
                                                                   target = Float64, limit = Float64))
        admissible = body(source, "js_admissible")
        @test count("for (let", admissible) == 1
        @test occursin("continue;", admissible) && occursin(".push(", admissible)
    end

    @testset "sums of iterators fold from the left, as Julia's do" begin
        # Pairwise and sequential sums of SPREAD differ; the module matches
        # Julia's sequential one exactly.
        @test sum(v for v in SPREAD) != sum(SPREAD)
        if NODE
            source = ReactiveKernels.js_source(js_iter_sums; have = (:x,), want = :total, types = (; x = V))
            r = only(ReactiveKernels.js_run(source, :kernel, [(SPREAD,)]))
            @test r.value === sum(v for v in SPREAD)
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "unsupported iterators are named" begin
        e = try
            ReactiveKernels.js_source(js_iter_truthy; have = (:x,), want = :y, types = (; x = V))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError && occursin("must return a `Bool`", e.message)
        e = try
            ReactiveKernels.js_source(js_iter_mixed; have = (:x,), want = :y, types = (; x = V))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError && occursin("concrete element type", e.message)
    end
end

end
