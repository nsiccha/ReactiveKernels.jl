module TestJSFunctions

# `js_source` of the Base idioms plain helpers use over a window of a series:
# comparison predicates (`==(x)`, `>=(x)`, …) given to the functions that take
# one, `findfirst`/`findlast`, `sum`/`maximum`/`minimum` with a function,
# `x[a:b]`, `collect`, and `merge` of named tuples. Each module is checked in
# Node against the native kernel, errors included.

using ReactiveKernels
using Test

const NODE = ReactiveKernels._js_default_node() !== nothing

# Named tuples: overriding a field, several tuples, Julia's field order (the
# first tuple's names, then each later one's new names; later values win).
js_override(a, b) = merge((; a = 1.0, b), (; a))
js_sumfields(t) = t.a + t.b
js_merge3(a, b, c) = merge((; a, b), (; c, a = 2a), (; b = c))
js_merged_first(a, b, c) = ((h, _, _) = js_merge3(a, b, c); h)
js_merged_last(a, b, c) = ((_, _, t) = js_merge3(a, b, c); t)
js_merged_b(a, b, c) = js_merge3(a, b, c).b

@kernel js_fn_merge(a, b, c) = begin
    total = js_sumfields(js_override(a, b))
    head = js_merged_first(a, b, c)
    tail = js_merged_last(a, b, c)
    middle = js_merged_b(a, b, c)
    return total
end

# Searching: a key's position or an `ArgumentError`, the last element below a
# level, the first `true`, and positions in a range with runtime bounds.
function js_lookup(key, keys, values)
    k = findfirst(==(key), keys)
    k === nothing && throw(ArgumentError("the data do not cover the chosen key"))
    return values[k]
end
function js_last_below(x, level)
    k = findlast(<(level), x)
    return isnothing(k) ? 0 : k
end
js_first_true(b) = (k = findfirst(b); k === nothing ? -1 : k)
js_last_true(b) = (k = findlast(b); k === nothing ? -1 : k)
js_first_reaching(n, c) = (k = findfirst(>=(c), 3:n); k === nothing ? 0 : k)
js_last_unequal(n, c) = (k = findlast(!=(c), 1:n); k === nothing ? 0 : k)
js_first_same(x, v) = (k = findfirst(isequal(v), x); k === nothing ? 0 : k)

@kernel js_fn_find(key, keys, values, x, level, b, n, c) = begin
    found = js_lookup(key, keys, values)
    below = js_last_below(x, level)
    first_true = js_first_true(b)
    last_true = js_last_true(b)
    reaching = js_first_reaching(n, c)
    unequal = js_last_unequal(n, c)
    same = js_first_same(x, level)
    return found
end

# Predicates given to `count`, `any`, `all` and `filter`.
js_at_least(x, level) = count(>=(level), x)
js_above(x, level) = count(>(level), x)
js_any_equal(x, v) = any(==(v), x)
js_all_below(x, c) = all(<(c), x)
js_all_at_most(x, c) = all(<=(c), x)
js_n_unequal(x, v) = count(!=(v), x)
js_n_nan(x) = count(isequal(NaN), x)
js_equal_to(x, v) = filter(==(v), x)

@kernel js_fn_predicates(x, level) = begin
    at_least = js_at_least(x, level)
    above = js_above(x, level)
    any_equal = js_any_equal(x, level)
    all_below = js_all_below(x, level)
    all_at_most = js_all_at_most(x, level)
    unequal = js_n_unequal(x, level)
    nans = js_n_nan(x)
    equal = js_equal_to(x, level)
    return at_least
end

# Reductions with a function first: Base's own functions, closures with and
# without captured values, `Bool`-valued functions, and a range source.
js_rms(x, y) = sqrt(sum(abs2, x .- y) / length(x))
js_twice(x) = sum(v -> 2v, x)
js_scaled(x, c) = sum(v -> v * c, x)
js_nan_total(x) = sum(isnan, x)
js_largest_size(x) = maximum(abs, x)
js_least_offset(x, c) = minimum(v -> v - c, x)
js_any_nan(x) = maximum(isnan, x)
js_all_nan(x) = minimum(isnan, x)
js_squares_to(n) = sum(abs2, 1:n)

@kernel js_fn_reductions(x, y, c) = begin
    rms = js_rms(x, y)
    twice = js_twice(x)
    scaled = js_scaled(x, c)
    nan_total = js_nan_total(x)
    largest_size = js_largest_size(x)
    least_offset = js_least_offset(x, c)
    any_nan = js_any_nan(x)
    all_nan = js_all_nan(x)
    return rms
end
@kernel js_fn_range_reduction(n) = begin
    squares = js_squares_to(n)
    return squares
end

# Windows and copies.
js_window(x, from, to) = x[from:to]
js_head(x) = x[1:2]
js_positions(from, to) = collect(from:to)
js_copy(x) = collect(x)
js_window_sum(x, from, to) = sum(x[from:to])

@kernel js_fn_windows(x, from, to) = begin
    window = js_window(x, from, to)
    head = js_head(x)
    positions = js_positions(from, to)
    copied = js_copy(x)
    window_sum = js_window_sum(x, from, to)
    return window
end

# A predicate that is not a `Bool`.
js_truthy_search(x) = findfirst(v -> v, x)
@kernel js_fn_truthy(x) = begin
    k = js_truthy_search(x)
    return k
end

const V = Vector{Float64}
const SERIES = [3.0, 12.5, 7.25, 20.0, 1.5, 11.0]
const CASES = [
    (:js_fn_merge, js_fn_merge, (:a, :b, :c), (:total, :head, :tail, :middle), (; a = Float64, b = Float64, c = Float64),
     [(5.0, 2.0, 0.5), (-0.0, NaN, 3.0)]),
    (:js_fn_find, js_fn_find, (:key, :keys, :values, :x, :level, :b, :n, :c),
     (:found, :below, :first_true, :last_true, :reaching, :unequal, :same),
     (; key = Int, keys = Vector{Int}, values = Vector{Int}, x = V, level = Float64, b = Vector{Bool}, n = Int, c = Int),
     [(2020, [2019, 2020, 2021], [10, 20, 30], SERIES, 10.0, [false, true, true, false], 9, 5),
      (2021, [2021, 2021], [7, 8], [NaN, 1.0, NaN], NaN, [true], 2, 1),
      (2019, [2019], [1], Float64[], 0.0, Bool[], 0, 0),
      (2022, [2019, 2020, 2021], [10, 20, 30], SERIES, 10.0, Bool[], 9, 5)]),   # the key is missing
    (:js_fn_predicates, js_fn_predicates, (:x, :level),
     (:at_least, :above, :any_equal, :all_below, :all_at_most, :unequal, :nans, :equal), (; x = V, level = Float64),
     [(SERIES, 11.0), ([NaN, 1.0, NaN, -0.0], 0.0), ([2.0, 2.0], 2.0), (Float64[], 1.0), ([NaN], NaN)]),
    (:js_fn_reductions, js_fn_reductions, (:x, :y, :c),
     (:rms, :twice, :scaled, :nan_total, :largest_size, :least_offset, :any_nan, :all_nan), (; x = V, y = V, c = Float64),
     [(SERIES, reverse(SERIES), 0.5), ([1.0, NaN, -3.0], [0.0, 0.0, 0.0], 2.0), ([-0.0, 0.0], [0.0, -0.0], 0.0),
      ([sin(k) * 10.0^(k % 7) for k in 1:3001], [cos(k) for k in 1:3001], 1.5),
      ([1.0, 2.0], [1.0, 2.0, 3.0], 1.0),  # x .- y: a DimensionMismatch
      (Float64[], Float64[], 1.0)]),  # an empty sum of a closure: Julia's MethodError
    (:js_fn_range_reduction, js_fn_range_reduction, (:n,), (:squares,), (; n = Int), [(10,), (1,), (0,)]),
    (:js_fn_windows, js_fn_windows, (:x, :from, :to), (:window, :head, :positions, :copied, :window_sum),
     (; x = V, from = Int, to = Int),
     [(SERIES, 2, 4), (SERIES, 3, 2), (SERIES, 6, 6), (SERIES, 5, 9),   # 5:9 is out of bounds
      ([1.0], 1, 1)]),                                                   # x[1:2] is out of bounds
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) ||
    (iszero(a) ? iszero(b) && signbit(a) == signbit(b) : isapprox(a, b; rtol = 1e-12, atol = 1e-15))
agree(a::Bool, b::Bool) = a === b
agree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(agree.(a, b))
agree(a, b) = false

# The code of one function of a module.
function body(source, name)
    i = findfirst("function " * name * "(", source)
    source[i[1]:findnext("\n}\n", source, i[1])[end]]
end

@testset "js_source of plain-helper idioms" begin
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

    @testset "a missing key is native's ArgumentError" begin
        name, k, have, want, types, cases = CASES[2]
        if NODE
            source = ReactiveKernels.js_source(k; have, want = :found, types, name)
            r = only(ReactiveKernels.js_run(source, name, [cases[4]]))
            @test r.julia_error && r.error == "ArgumentError" && r.message == "the data do not cover the chosen key"
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "a search stops at its match and a predicate is called, not built" begin
        source = ReactiveKernels.js_source(js_fn_find; have = (:key, :keys, :values, :x, :level, :b, :n, :c),
                                           want = :found, types = CASES[2][5])
        lookup = body(source, "js_lookup")
        @test count("for (let", lookup) == 1 && occursin("break ", lookup)
        @test !occursin("Fix2", source)
    end

    @testset "a predicate that is not a Bool is named" begin
        e = try
            ReactiveKernels.js_source(js_fn_truthy; have = (:x,), want = :k, types = (; x = V))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError && occursin("must return a `Bool`", e.message)
    end
end

end
