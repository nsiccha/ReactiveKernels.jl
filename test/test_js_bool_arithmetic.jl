module TestJSBoolArithmetic

# `js_source` of arithmetic on `Bool`s, which Julia treats as the numbers 0 and
# 1: a share of curves above a line written as a sum of comparisons, a
# weighted sum of them or a count over a tuple, `Bool` operands of `+`, `-`,
# `*`, `/`, `^` and the comparisons next to numbers, and sums and vector
# arithmetic of `Bool` vectors. Each module is checked in Node against the
# native kernel: values, types (`true * true` is a `Bool`, `true + true` an
# `Int`) and signed zeros (`false * -Inf` is `-0.0`, `false + -0.0` is `-0.0`).

using ReactiveKernels
using Test

const NODE = ReactiveKernels._js_default_node() !== nothing

# The share of five curves above a line: a sum of comparisons, and a weighted
# sum of them per position.
@kernel js_bool_share(line, d1, d2, d3, d4, d5) = begin
    above = ((d1 .> line) .+ (d2 .> line) .+ (d3 .> line) .+ (d4 .> line) .+ (d5 .> line)) ./ 5
    return above
end
@kernel js_bool_weighted(line, weight, d1, d2, d3, d4, d5) = begin
    above = [sum(weight .* ([d1[j], d2[j], d3[j], d4[j], d5[j]] .> line)) for j in eachindex(d1)]
    return above
end
# The same share as a count of a partially applied comparison over a tuple.
@kernel js_bool_counted(line, d1, d2, d3, d4, d5) = begin
    above = [count(>(line), (d1[j], d2[j], d3[j], d4[j], d5[j])) / 5 for j in eachindex(d1)]
    return above
end

# Scalar arithmetic with `Bool` operands, in a plain helper and in the recipe.
js_bool_votes(a, b, c) = a + b + c
@kernel js_bool_scalars(p, q, x, k) = begin
    sum2 = p + q
    diff = p - q
    both = p * q
    ratio = p / q
    power = p^q
    neg = -p
    scaled = x * p
    scaled_left = p * x
    shifted = p + x
    shifted_left = x + p
    lowered = x - p
    over = x / p
    count_k = p + k
    times_k = k * q
    votes = js_bool_votes(p, q, x > 0)
    larger = max(p, x)
    below = p < x
    same = p == k
    as_float = float(p)
    recip = float(k)^-1     # a power of a `Float64`, not an integer power
    return sum2
end

# Sums and vector arithmetic of `Bool` vectors, and reductions of comparisons.
@kernel js_bool_vectors(b, c, x, w) = begin
    total = sum(b)
    added = b + c
    taken = b - c
    negated = -b
    weighted = sum(w .* b)
    masked = x .* b
    doubled = b * 2.0
    halved = b / 2
    # a comparison broadcast is a `BitVector`, read like a `Vector{Bool}`
    n_above = sum(x .> 1.0)
    n_count = count(x .> 1.0)
    some = any(x .> 1.0)
    every = all(b .| c)
    above_weight = sum(w .* (x .> 1.0))
    return total
end

const V = Vector{Float64}
const CURVES = ([0.05, 0.12, 0.25, 0.34, 0.45], [0.1, 0.2, 0.31, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.32, 0.5, 0.6, 0.7], [0.4, 0.5, 0.6, 0.7, 0.8])
const ODD = ([NaN, -0.0, Inf], [1.0, 0.0, -Inf], [0.5, -1.0, 2.0], [NaN, NaN, NaN], [3.0, 0.0, -0.0])
const CASES = [
    (:js_bool_share, js_bool_share, (:line, :d1, :d2, :d3, :d4, :d5), (:above,),
     (; line = Float64, d1 = V, d2 = V, d3 = V, d4 = V, d5 = V),
     [(0.33, CURVES...), (0.0, ODD...), (NaN, CURVES...),
      (0.5, Float64[], Float64[], Float64[], Float64[], Float64[])]),
    (:js_bool_counted, js_bool_counted, (:line, :d1, :d2, :d3, :d4, :d5), (:above,),
     (; line = Float64, d1 = V, d2 = V, d3 = V, d4 = V, d5 = V),
     [(0.33, CURVES...), (0.0, ODD...), (NaN, CURVES...),
      (0.5, Float64[], Float64[], Float64[], Float64[], Float64[])]),
    (:js_bool_weighted, js_bool_weighted, (:line, :weight, :d1, :d2, :d3, :d4, :d5), (:above,),
     (; line = Float64, weight = V, d1 = V, d2 = V, d3 = V, d4 = V, d5 = V),
     [(0.33, fill(0.2, 5), CURVES...), (0.0, [-0.0, Inf, NaN, -1.0, 2.0], ODD...),
      (0.33, fill(0.2, 4), CURVES...)]),   # a weight per curve missing: a DimensionMismatch
    (:js_bool_scalars, js_bool_scalars, (:p, :q, :x, :k),
     (:sum2, :diff, :both, :ratio, :power, :neg, :scaled, :scaled_left, :shifted, :shifted_left, :lowered, :over,
      :count_k, :times_k, :votes, :larger, :below, :same, :as_float, :recip),
     (; p = Bool, q = Bool, x = Float64, k = Int),
     [(p, q, x, k) for (p, q) in ((true, true), (true, false), (false, true), (false, false))
      for (x, k) in ((2.5, 3), (-0.0, 1), (-Inf, 0), (NaN, -2), (0.0, 1))]),
    (:js_bool_vectors, js_bool_vectors, (:b, :c, :x, :w),
     (:total, :added, :taken, :negated, :weighted, :masked, :doubled, :halved, :n_above, :n_count, :some, :every,
      :above_weight),
     (; b = Vector{Bool}, c = Vector{Bool}, x = V, w = V),
     [([true, false, true], [true, true, false], [1.5, -Inf, NaN], [0.25, 0.5, -0.0]),
      (Bool[], Bool[], Float64[], Float64[]),
      (fill(true, 2000), [isodd(k) for k in 1:2000], collect(1.0:2000.0), fill(0.5, 2000)),
      ([true], [true, false], [1.0], [1.0])]),   # b + c: a DimensionMismatch
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) ||
    (iszero(a) ? iszero(b) && signbit(a) == signbit(b) : isapprox(a, b; rtol = 1e-12, atol = 1e-15))
agree(a::Bool, b::Bool) = a === b
agree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(agree.(a, b))
agree(a, b) = false

@testset "js_source of Bool arithmetic" begin
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
                # one WANT is its value, several a tuple of them
                outputs(x) = length(want) == 1 ? (x,) : x
                @test all(agree(outputs(expected)[j], outputs(r.value)[j]) for j in eachindex(want))
            end
        end
    end

    @testset "a Bool is 0 or 1 where Julia converts it, in one loop over the curves" begin
        source = ReactiveKernels.js_source(js_bool_share; have = CASES[1][3], want = :above, types = CASES[1][5],
                                           name = :js_bool_share)
        main = source[findfirst("export function js_bool_share(", source)[1]:end]
        @test occursin(" ? 1 : 0", main)
        @test count("for (let", main) == 1 && occursin("_rk_bclen(d1.length, d2.length", main)
    end
end

end
