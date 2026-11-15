module TestJSSource

using ReactiveKernels
using Test
using Base: OneTo
using LogExpFunctions: log1pexp, logistic

const NODE = ReactiveKernels._js_default_node() !== nothing

# Plain Julia helpers without annotations: branches, early returns and loops
# are translated from their typed code.
js_piecewise(x) = x < 0 ? -x : x <= 1 ? x^2 : 2x - 1
function js_early(x)
    x <= 1 && return log1p(x)
    return sqrt(x) + exp(-x) / x
end
function js_logistic_map(x0, r, n)
    x = x0
    for _ in 1:n
        x = r * x * (1 - x)
    end
    x
end
function js_cube_root(target, lo, hi)
    while hi - lo > 1e-12
        mid = (lo + hi) / 2
        if mid^3 > target
            hi = mid
        else
            lo = mid
        end
    end
    (lo + hi) / 2
end
function js_weighted_total(v)
    s = 0.0
    for i in eachindex(v)
        s += v[i] * i
    end
    s
end
js_squares(v) = [x^2 for x in v]
js_c_sin(x) = ccall(:sin, Float64, (Float64,), x)

# Loops over a range written in the helper (`for _ in 1:60`): the range is known
# while translating, needs no kernel input, and a loop over an empty one, like a
# branch on a constant, is dead code.
js_decay(x, k) = exp(-k * x)
function js_half_life(k)
    low, high = 0.0, 1.0
    while js_decay(high, k) > 0.5
        high *= 2
    end
    for _ in 1:60
        middle = (low + high) / 2
        js_decay(middle, k) > 0.5 ? (low = middle) : (high = middle)
    end
    (low + high) / 2
end
function js_fixed_series(x)
    s = 0.0
    for i in 1:12
        s += x^i / i
    end
    for i in OneTo(3)
        s = s * i + 1
    end
    for i in -3:2
        s += i * x
    end
    for _ in 1:0
        s = -s
    end
    s
end
const JS_CHECKED = false
function js_unchecked(x)
    if JS_CHECKED
        x < 0 && error("negative input")
    end
    sqrt(abs(x))
end
function js_range_sum(r, x)
    isempty(r) && return x
    s = 0.0
    for i in r
        s += i * x
    end
    s
end
function js_chosen_range(flag, n)
    r = flag ? (1:3) : (1:n)
    s = 0
    for i in r
        s += i
    end
    s
end
js_powers(x) = x .^ (0:4)
js_scaled(x) = [x / k for k in 1:5]
js_squares_scaled(x) = [k^2 for k in 1:5] .* x
js_multiples(x) = map(k -> k * x, 1:4)

# Keyword helpers (a required keyword, a default, a default computed from
# another keyword), named tuples built and read by field, and destructuring.
function js_kw_split(x, y; share, offset = 0.0, scale = 2 * share)
    lo, hi = zeros(length(x)), zeros(length(x))
    for i in eachindex(x)
        lo[i] = share * x[i] + offset
        hi[i] = scale * y[i]
    end
    return (; lo, hi = hi)
end
js_kw_total(r; weight) = weight * sum(r.lo) + sum(r.hi)
js_record_mean(rec) = (sum(rec.a) + sum(rec.b)) / (length(rec.a) + length(rec.b))
js_kw_need(x; a) = a * x
# a keyword `b` that `js_kw_need` does not take, and a missing `a`: Julia's
# MethodError and UndefKeywordError, when the branch runs
js_kw_unknown(x) = x > 0 ? js_kw_need(x; a = 2.0) : js_kw_need(x; a = 1.0, b = 3.0)
js_kw_missing(x) = x < 10 ? js_kw_need(x; a = 2.0) : js_kw_need(x)
# a keyword call reaches the keyword method, not a more specific one without
# keywords; a function without keyword methods is a MethodError
js_kw_pick(x; a) = a * x
js_kw_pick(x::Float64) = -x
js_kw_specific(x) = js_kw_pick(x; a = 3.0) + js_kw_pick(x)
js_kw_none(x) = x
js_kw_none_use(x) = x > -5 ? x : js_kw_none(x; a = 1.0)
# a generator over a range written in the source, destructured
function js_buffers(x)
    n = length(x)
    a, b, c = (Vector{Float64}(undef, n) for _ in 1:3)
    for i in 1:n
        a[i] = x[i]
        b[i] = 2 * x[i]
        c[i] = a[i] + b[i]
    end
    out = similar(x)
    for i in eachindex(out)
        out[i] = c[i] - a[i]
    end
    return out
end
js_window_means(x, w) =
    [sum(@view x[(w * (k - 1) + 1):(w * k)]) / length(@view x[(w * (k - 1) + 1):(w * k)]) for k in 1:(length(x) ÷ w)]
js_view_second(x) = (v = @view x[2:end]; v[1])
# `log1pexp` destructures thresholds that depend only on the argument's type
js_soft(x, s) = s * log1pexp(x / s)

@kernel js_scalar(a, b) = begin
    c = a * b + exp(-a) / (1 + b^2)
    return c
end
@kernel js_fused(x, a, b) = begin
    y = a .* x .+ b
    z = exp.(-y) .* sin.(x) ./ (1 .+ x .^ 2)
    return z
end
@kernel js_helpers(x) = begin
    p = js_piecewise.(x)
    e = js_early.(abs.(x) .+ 0.5)
    q = p .+ e
    return q
end
@kernel js_loops(x0, r, n, target) = begin
    y = js_logistic_map(x0, r, n)
    w = js_cube_root(target, 0.0, max(target, 1.0))
    return y
end
@kernel js_reductions(x) = begin
    s = sum(x)
    m = maximum(x)
    t = js_weighted_total(x)
    f = x[1] - x[end]
    return s
end
@kernel js_switch(x, flag, k) = begin
    y = ifelse(flag, x * k, x / k)
    return y
end
@kernel js_difference(x, y) = begin
    d = x .- y
    return d
end
@kernel js_comprehension(x) = begin
    y = js_squares(x)
    return y
end
@kernel js_branch(a, b) = begin
    y = if a > b
        a - b
    else
        log(b - a - 1)
    end
    return y
end
@kernel js_scan(x, decay, initial) = begin
    trajectory = scan(x; init = initial) do previous, observation
        current = decay * previous + observation
        (current, current)
    end
    return trajectory
end
@kernel js_plate(y, mu) = begin
    lp = plate(y, mu) do yi, mi
        -(yi - mi)^2 / 2
    end
    total = sum(lp)
    return total
end
@kernel js_fixed_ranges(x, k, flag, n) = begin
    h = js_half_life(k)
    s = js_fixed_series(x)
    u = js_unchecked(x)
    e = js_range_sum(1:0, x) + js_range_sum(1:3, x)
    c = js_chosen_range(flag, n)
    p = plate(1:4) do i
        i * x
    end
    return h
end
@kernel js_fixed_vectors(x) = begin
    p = js_powers(x)
    s = js_scaled(x)
    q = js_squares_scaled(x)
    m = js_multiples(x)
    return p
end
@kernel js_c_call(x) = begin
    y = js_c_sin(x)
    return y
end
@kernel js_fma(a, b) = begin
    y = fma(a, b, 1.0)
    return y
end
@kernel js_keywords(x, y, share, weight) = begin
    r = js_kw_split(x, y; share)
    lo = r.lo
    hi = r.hi
    t = js_kw_total(r; weight)
    return (; lo, hi, t)
end
@kernel js_record(a, b, s) = begin
    rec = (; a, b, s)
    m = js_record_mean(rec)
    return m
end
@kernel js_keyword_errors(x) = begin
    u = js_kw_unknown(x)
    m = js_kw_missing(x)
    p = js_kw_specific(x)
    q = js_kw_none_use(x)
    return u
end
@kernel js_destructure(x) = begin
    out = js_buffers(x)
    return out
end
@kernel js_views(x, w) = begin
    means = js_window_means(x, w)
    second = js_view_second(x)
    return (; means, second)
end
@kernel js_gather(x, index) = begin
    g = x[index]
    e = exp.(x[index])
    return (; g, e)
end
@kernel js_logexp(x, s) = begin
    a = js_soft.(x, s)
    b = logistic.(x)
    return (; a, b)
end

const V = Vector{Float64}
const CASES = [
    (:js_scalar, js_scalar, (:a, :b), :c, (; a = Float64, b = Float64),
     [(0.5, 2.0), (-3.0, 0.0), (1e-300, 1e10)]),
    (:js_fused, js_fused, (:x, :a, :b), :z, (; x = V, a = Float64, b = Float64),
     [(collect(range(-3, 3; length = 41)), 0.7, -0.2), (Float64[], 1.0, 2.0), ([1.0], 0.0, 0.0)]),
    (:js_helpers, js_helpers, (:x,), :q, (; x = V), [(collect(range(-2, 3; length = 51)),), (Float64[],)]),
    (:js_loops, js_loops, (:x0, :r, :n, :target), (:y, :w), (; x0 = Float64, r = Float64, n = Int, target = Float64),
     [(0.3, 3.7, 0, 2.0), (0.3, 3.7, 1, 0.5), (0.3, 3.7, 50, 27.0), (0.2, 2.5, -4, 1.0)]),
    (:js_reductions, js_reductions, (:x,), (:s, :m, :t, :f), (; x = V),
     [(collect(range(-2, 3; length = 2001)),), ([4.0],), (Float64[],)]),
    (:js_switch, js_switch, (:x, :flag, :k), :y, (; x = Float64, flag = Bool, k = Int),
     [(3.0, true, 4), (3.0, false, 4), (3.0, false, 0)]),
    (:js_difference, js_difference, (:x, :y), :d, (; x = V, y = V),
     [([1.0, 2.0, 3.0], [0.5, 0.25, 0.125]), ([1.0, 2.0, 3.0], [1.0]), ([2.0], [1.0, 2.0, 3.0]),
      ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])]),
    (:js_comprehension, js_comprehension, (:x,), :y, (; x = V), [([1.0, -2.0, 3.5],), (Float64[],)]),
    # The arm not taken is not evaluated: `log` of a negative number is never
    # computed for a > b.
    (:js_branch, js_branch, (:a, :b), :y, (; a = Float64, b = Float64), [(5.0, 1.0), (1.0, 4.0), (1.0, 1.5)]),
    (:js_scan, js_scan, (:x, :decay, :initial), :trajectory, (; x = V, decay = Float64, initial = Float64),
     [([1.0, 0.0, 0.0, 2.0], 0.5, 0.0), (Float64[], 0.5, 1.0), (collect(range(0, 1; length = 1000)), 0.9, 2.0)]),
    (:js_plate, js_plate, (:y, :mu), :total, (; y = V, mu = V),
     [([1.0, 2.0, 3.0], [0.5, 2.5, 3.0]), ([1.0, 2.0], [0.0]), (Float64[], Float64[]), ([1.0, 2.0, 3.0], [1.0, 2.0])]),
    (:js_fixed_ranges, js_fixed_ranges, (:x, :k, :flag, :n), (:h, :s, :u, :e, :c, :p),
     (; x = Float64, k = Float64, flag = Bool, n = Int),
     [(0.5, 0.3, true, 10), (-0.9, 2.0, false, 0), (2.0, 1e-3, false, 7)]),
    (:js_fixed_vectors, js_fixed_vectors, (:x,), (:p, :s, :q, :m), (; x = Float64), [(1.5,), (-0.25,)]),
    (:js_keywords, js_keywords, (:x, :y, :share, :weight), (:lo, :hi, :t),
     (; x = V, y = V, share = Float64, weight = Float64),
     [([1.0, 2.0, 3.0], [0.5, 0.25, 0.0], 0.3, 2.0), (Float64[], Float64[], 0.1, 1.0), ([1.0, 2.0], [1.0], 0.5, 1.0)]),
    (:js_record, js_record, (:a, :b, :s), :m, (; a = V, b = V, s = Float64),
     [([1.0, 2.0], [3.0], 0.5), (Float64[], [2.0, 4.0], 1.0)]),
    (:js_keyword_errors, js_keyword_errors, (:x,), (:u, :m, :p, :q), (; x = Float64),
     [(1.5,), (-1.0,), (12.0,), (-6.0,)]),
    (:js_destructure, js_destructure, (:x,), :out, (; x = V), [([1.0, -2.0, 3.5],), (Float64[],)]),
    (:js_views, js_views, (:x, :w), (:means, :second), (; x = V, w = Int),
     [(collect(1.0:12.0), 3), (collect(1.0:13.0), 4), ([1.0], 1), (collect(range(0, 1; length = 5000)), 2500)]),
    (:js_gather, js_gather, (:x, :index), (:g, :e), (; x = V, index = Vector{Int}),
     [([1.0, 2.0, 3.0], [3, 1, 1, 2]), ([1.0, 2.0], Int[]), ([1.0, 2.0], [1, 3])]),
    (:js_logexp, js_logexp, (:x, :s), (:a, :b), (; x = V, s = Float64),
     [(collect(range(-800, 800; length = 33)), 1.0), ([-40.0, -20.0, 0.0, 20.0, 35.0], 0.5)]),
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) || isapprox(a, b; rtol = 1e-12, atol = 1e-15)
agree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(agree.(a, b))
agree(a, b) = false

@testset "js_source" begin
    @testset "emits a module for $name" for (name, k, have, want, types, _) in CASES
        source = ReactiveKernels.js_source(k; have, want, types, name)
        @test occursin("export function " * string(name) * "(" * join(string.(have), ", ") * ")", source)
        @test occursin("export const INPUTS", source) && occursin("export class JuliaError", source)
    end

    @testset "Node agrees with native: $name ($vectors)" for (name, k, have, want, types, cases) in CASES,
                                                             vectors in (:array, :float64array)
        if !NODE
            @test_skip "no node on the PATH"
            continue
        end
        source = ReactiveKernels.js_source(k; have, want, types, name)
        wants = want isa Symbol ? (want,) : want
        native = prepare(k; have, want = wants)
        outcomes = ReactiveKernels.js_run(source, name, cases; vectors)
        for (c, r) in zip(cases, outcomes)
            expected = try
                native(c...)
            catch e
                e
            end
            @test !r.inputs_mutated
            if expected isa Exception
                # The same Julia exception type, thrown by the JavaScript.
                @test r.julia_error && r.error == string(nameof(typeof(expected)))
            else
                @test r.error === nothing
                r.error === nothing || continue
                # The native shape and types, and the native values.
                @test typeof(r.value) == typeof(expected)
                if want isa Symbol
                    @test agree(expected, r.value)
                else
                    @test all(agree(expected[j], r.value[j]) for j in eachindex(wants))
                end
            end
        end
    end

    @testset "branching helpers need no annotation" begin
        source = ReactiveKernels.js_source(js_helpers; have = (:x,), want = :q, types = (; x = V))
        @test occursin("function js_piecewise(x) {", source)
        @test occursin("function js_early(x) {", source)
        @test occursin("if (x <= 1) {", source)
    end

    @testset "data-dependent iteration stays a loop" begin
        loops = ReactiveKernels.js_source(js_loops; have = (:x0, :r, :n, :target), want = :y,
                                          types = (; x0 = Float64, r = Float64, n = Int, target = Float64))
        @test occursin("while (true)", loops)
        scan = ReactiveKernels.js_source(js_scan; have = (:x, :decay, :initial), want = :trajectory,
                                         types = (; x = V, decay = Float64, initial = Float64))
        body = scan[findfirst("export function kernel(", scan)[1]:end]
        body = body[1:findfirst("\n}", body)[1]]
        @test count("for (let", body) == 1
        @test occursin("x.length", body)
    end

    @testset "a range written in a helper is read as literals" begin
        source = ReactiveKernels.js_source(js_fixed_ranges; have = (:x, :k, :flag, :n), want = (:h, :s, :u, :e, :c, :p),
                                           types = (; x = Float64, k = Float64, flag = Bool, n = Int))
        body(name) = (i = findfirst("function " * name * "(", source);
                      source[i[1]:findnext("\n}\n", source, i[1])[end]])
        half = body("js_half_life")
        @test occursin("< 60 ?", half) && !occursin("_rk_range", half) && !occursin(".stop", half)
        # dead code is left out: the loop over `1:0`, the branch on `JS_CHECKED`
        # and, in the helper specialized on `1:0`, everything after `isempty`
        @test count("while (true)", body("js_fixed_series")) == 3
        @test !occursin("negative input", source) && !occursin("if (", body("js_unchecked"))
        @test occursin(r"function js_range_sum(_\d+)?\(x\) \{\n(  let \w+;\n)*  return x;\n\}", source)
        # a range assigned in a branch is a value at run time
        @test occursin("_rk_range(1, 3)", body("js_chosen_range"))
    end

    @testset "keyword calls, named tuples and destructuring" begin
        body(source, name) = (i = findfirst("function " * name * "(", source);
                              source[i[1]:findnext("\n}\n", source, i[1])[end]])
        kw = ReactiveKernels.js_source(js_keywords; have = (:x, :y, :share, :weight), want = (:lo, :hi, :t),
                                       types = (; x = V, y = V, share = Float64, weight = Float64))
        # the keyword sorter is inlined: the body function takes the keyword
        # values, the default `scale = 2 * share` computed in the module
        @test occursin(r"function js_kw_split_\d+\(share, offset, scale, x, y\) \{", kw)
        @test occursin("2 * share", kw) && !occursin("kwcall", kw) && !occursin("structdiff", kw)
        # a named tuple is an object, read by field
        @test occursin("return { lo: lo, hi: hi };", body(kw, "js_kw_split_" * match(r"js_kw_split_(\d+)", kw)[1]))
        # the loop over the data stays a loop; the destructured generator over
        # `1:3`, written in the source, is three allocations
        buffers = body(ReactiveKernels.js_source(js_destructure; have = (:x,), want = :out, types = (; x = V)),
                       "js_buffers")
        @test count("new Array(n).fill(0)", buffers) == 3 && count("while (true)", buffers) == 2
        # thresholds known from the argument's type are literals
        soft = ReactiveKernels.js_source(js_logexp; have = (:x, :s), want = (:a, :b), types = (; x = V, s = Float64))
        @test occursin("x < -745.1332191019412", soft) && !occursin("indexed_iterate", soft)
    end

    @testset "unsupported constructs are named" begin
        e = try
            ReactiveKernels.js_source(js_c_call; have = (:x,), want = :y, types = (; x = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("C function `sin`", e.message) && occursin("js_c_sin", e.message)
        e = try
            ReactiveKernels.js_source(js_fma; have = (:a, :b), want = :y, types = (; a = Float64, b = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("`fma`", e.message)
    end

    @testset "inputs and names" begin
        @test_throws ArgumentError ReactiveKernels.js_source(js_switch; have = (:x, :flag, :k), want = :y,
            types = (; x = Float64, flag = Bool, k = Int), name = :switch)
        if NODE
            source = ReactiveKernels.js_source(js_scalar; have = (:a, :b), want = :c,
                                               types = (; a = Float64, b = Float64), name = :scalar)
            r = only(ReactiveKernels.js_run(source, :scalar, [([1.0], 2.0)]))
            @test r.julia_error && r.error == "ArgumentError" && occursin("`a` must be a number", r.message)
        else
            @test_skip "no node on the PATH"
        end
    end
end

end
