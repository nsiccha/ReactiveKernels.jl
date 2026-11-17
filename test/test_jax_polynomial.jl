module JAXPolynomialTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function prefix_polynomial(x, coefficients, degree)
    evalpoly(x, coefficients[1:(degree + 1)])
end
@kernel polynomial(x::Float64, coefficients::Vector{Float64}, degree::Int) = begin
    value = prefix_polynomial(x, coefficients, degree)
    full = evalpoly(x, coefficients)
    return value, full
end

@testset "runtime polynomial prefixes" begin
    functions = (value=:value, full=:full)
    src = jax_source(polynomial; have=(:x, :coefficients, :degree), functions)
    @test occursin("jax.lax.fori_loop(", string(src))
    @test string(jax_source(polynomial; have=(:x, :coefficients, :degree), functions,
                           data=(coefficients=ones(3),))) ==
          string(jax_source(polynomial; have=(:x, :coefficients, :degree), functions,
                           data=(coefficients=ones(129),)))
    cases = (([2.0], 0), ([2.0, NaN, NaN], 0), ([1.0, 2.0, 3.0, NaN], 2), (ones(129), 128))
    for (coefficients, degree) in cases
        native = prefix_polynomial(0.3, coefficients, degree)
        derivative = Enzyme.autodiff(Enzyme.Reverse, prefix_polynomial, Enzyme.Active,
            Enzyme.Active(0.3), Enzyme.Const(coefficients), Enzyme.Const(degree))[1][1]
        if rk_jax_available()
            got = rk_jax_evaluate(src, :value, (0.3, coefficients, degree); grad=true)
            @test got.value ≈ native
            @test got.grad ≈ derivative
        end
    end
    if rk_jax_available()
        for x in (Inf, NaN)
            @test rk_jax_evaluate(src, :value, (x, [2.0, NaN], 0)).value == 2.0
            @test rk_jax_evaluate(src, :full, (x, [2.0], 0)).value == 2.0
        end
        @test_throws "prefix exceeds array length" rk_jax_evaluate(src, :value, (0.3, [1.0], 1))
        @test_throws "at least one coefficient" rk_jax_evaluate(src, :value, (0.3, [1.0], -1))
        @test_throws "at least one coefficient" rk_jax_evaluate(src, :full, (0.3, Float64[], 0))
    end
end

# A window of a vector between two lengths that are data: `x[(p + 1):(p + q)]`
# beside the prefix `x[1:p]`, of each ragged row of a plate.
@kernel split_rows(a::Float64, values::Vector{Vector{Float64}}, firsts::Vector{Vector{Float64}},
                   seconds::Vector{Vector{Float64}}) = begin
    totals = plate(values, firsts, seconds, Ref(a)) do x, first, second, a
        p = length(first)
        q = length(second)
        head = exp.(a .* x[1:p])
        tail = exp.(a .* x[(p + 1):(p + q)] .+ second)
        sum(head .* first) + 2 * sum(tail)
    end
    total = sum(totals)
end
function split_rows_native(a, values, firsts, seconds)
    total = 0.0
    for (x, first, second) in zip(values, firsts, seconds)
        p, q = length(first), length(second)
        total += sum(exp.(a .* x[1:p]) .* first) + 2 * sum(exp.(a .* x[(p + 1):(p + q)]) .* exp.(second))
    end
    total
end

# The window scanned beside the row of its length, stored narrower.
@kernel window_scan(a::Float64, values::Vector{Vector{Float64}}, firsts::Vector{Vector{Float64}},
                    seconds::Vector{Vector{Float64}}) = begin
    totals = plate(values, firsts, seconds, Ref(a)) do x, first, second, a
        p = length(first)
        q = length(second)
        path = scan(x[(p + 1):(p + q)], second; init = 1.0) do state, v, w
            next = state * exp(a * v) + w
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
end
function window_scan_native(a, values, firsts, seconds)
    total = 0.0
    for (x, first, second) in zip(values, firsts, seconds)
        p, q = length(first), length(second)
        state = 1.0
        for (v, w) in zip(x[(p + 1):(p + q)], second)
            state = state * exp(a * v) + w
            total += state
        end
    end
    total
end

# A window between ends known while tracing, of rows padded to different
# lengths. `p` is the length of a mapped dense cell (an earlier plate's
# vector cells of one length), so the range stays in the cell, as a model's
# `x[(p + 1):(p + q)]` with `p`, `q` the lengths of fixed grids does: a slice
# of static length, checked against each row's own length.
@kernel fixed_window(a::Float64, rows::Vector{Vector{Float64}}) = begin
    grids = plate(rows, Ref(a)) do x, a
        [a, 2 * a]
    end
    totals = plate(rows, grids, Ref(a)) do x, g, a
        p = length(g)
        head = exp.(a .* x[1:p])
        tail = exp.(a .* x[(p + 1):(p + 2)])
        sum(head .* g) + 2 * sum(tail) + sum(x[(p + 1):p])
    end
    total = sum(totals)
end
function fixed_window_native(a, rows)
    total = 0.0
    for x in rows
        g = [a, 2 * a]
        p = length(g)
        total += sum(exp.(a .* x[1:p]) .* g) + 2 * sum(exp.(a .* x[(p + 1):(p + 2)]))
    end
    total
end

# A window whose ends are unrelated integers: nothing bounds its length but
# the vector itself.
@kernel free_window(a::Float64, x::Vector{Float64}, lo::Int, hi::Int) = begin
    total = sum(exp.(a .* x[lo:hi]))
end
free_window_native(a, x, lo, hi) = sum(exp.(a .* x[lo:hi]))

@testset "runtime windows of padded rows" begin
    have = (:a, :values, :firsts, :seconds)
    src = jax_source(split_rows; have, want = :total)
    native = prepare(split_rows; have, want = :total)
    row(n) = Float64[sin(0.7 * k) for k in 1:n]
    datasets = (([row(5), row(2), row(3)], [row(2), Float64[], row(3)], [row(3), row(2), Float64[]]),
                ([Float64[], row(1)], [Float64[], Float64[]], [Float64[], row(1)]),
                ([row(40), row(7)], [row(11), row(7)], [row(29), Float64[]]),
                (Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]))
    prepared = jax_source(split_rows; have, want = :total,
                          data = NamedTuple{Base.tail(have)}(first(datasets)))
    for data in datasets
        # Row counts and lengths are data: they never specialize the source.
        @test prepared.text == jax_source(split_rows; have, want = :total,
                                         data = NamedTuple{Base.tail(have)}(data)).text
        value = native(0.3, data...)
        @test value ≈ split_rows_native(0.3, data...)
        derivative = Enzyme.autodiff(Enzyme.Reverse, split_rows_native, Enzyme.Active,
            Enzyme.Active(0.3), map(Enzyme.Const, data)...)[1][1]
        if rk_jax_available()
            got = rk_jax_evaluate(src, :kernel, (0.3, data...); grad = true)
            @test got.value ≈ value rtol = 1e-12 atol = 1e-12
            @test got.grad ≈ derivative rtol = 1e-12 atol = 1e-12
            for mode in (true, :arguments)
                got = rk_jax_evaluate(prepared, :kernel, (0.3, data...); grad = true, prepare = mode)
                @test got.value ≈ value rtol = 1e-12 atol = 1e-12
                @test got.grad ≈ derivative rtol = 1e-12 atol = 1e-12
            end
        end
    end
    # The window and the row of its length are stored with different widths:
    # the broadcast widens the narrower one.
    @test occursin("_rk_same_width(", src.text)
    scanned = jax_source(window_scan; have, want = :total)
    @test occursin("scan_width = min(", scanned.text)
    for data in datasets
        value = prepare(window_scan; have, want = :total)(0.3, data...)
        @test value ≈ window_scan_native(0.3, data...)
        derivative = Enzyme.autodiff(Enzyme.Reverse, window_scan_native, Enzyme.Active,
            Enzyme.Active(0.3), map(Enzyme.Const, data)...)[1][1]
        if rk_jax_available()
            got = rk_jax_evaluate(scanned, :kernel, (0.3, data...); grad = true)
            @test got.value ≈ value rtol = 1e-12 atol = 1e-12
            @test got.grad ≈ derivative rtol = 1e-12 atol = 1e-12
        end
    end

    fixed = jax_source(fixed_window; have = (:a, :rows), want = :total)
    # The window is sliced in the cell, not gathered by a hoisted range.
    @test occursin("_rk_check(", fixed.text) && occursin("range exceeds array bounds", fixed.text)
    for rows in ([row(4), row(6), row(5)], [row(4)], [row(40), row(13)], Vector{Float64}[])
        value = prepare(fixed_window; have = (:a, :rows), want = :total)(0.3, rows)
        @test value ≈ fixed_window_native(0.3, rows)
        derivative = Enzyme.autodiff(Enzyme.Reverse, fixed_window_native, Enzyme.Active,
            Enzyme.Active(0.3), Enzyme.Const(rows))[1][1]
        if rk_jax_available()
            got = rk_jax_evaluate(fixed, :kernel, (0.3, rows); grad = true)
            @test got.value ≈ value rtol = 1e-12 atol = 1e-12
            @test got.grad ≈ derivative rtol = 1e-12 atol = 1e-12
        end
    end
    # A row shorter than the window is Julia's `BoundsError`.
    @test_throws BoundsError fixed_window_native(0.3, [row(4), row(3)])
    rk_jax_available() && @test_throws "range exceeds array bounds" rk_jax_evaluate(
        fixed, :kernel, (0.3, [row(4), row(3)]))

    free = jax_source(free_window; have = (:a, :x, :lo, :hi), want = :total)
    x = row(7)
    for (lo, hi) in ((2, 5), (1, 7), (5, 5), (3, 2), (9, 4))
        value = free_window_native(0.3, x, lo, hi)
        derivative = Enzyme.autodiff(Enzyme.Reverse, free_window_native, Enzyme.Active,
            Enzyme.Active(0.3), Enzyme.Const(x), Enzyme.Const(lo), Enzyme.Const(hi))[1][1]
        if rk_jax_available()
            got = rk_jax_evaluate(free, :kernel, (0.3, x, lo, hi); grad = true)
            @test got.value ≈ value rtol = 1e-12 atol = 1e-12
            @test got.grad ≈ derivative rtol = 1e-12 atol = 1e-12
        end
    end
    # A nonempty window outside its vector is Julia's `BoundsError`.
    for (lo, hi) in ((0, 3), (4, 9))
        @test_throws BoundsError free_window_native(0.3, x, lo, hi)
        rk_jax_available() &&
            @test_throws "range exceeds array bounds" rk_jax_evaluate(free, :kernel, (0.3, x, lo, hi))
    end
end
end
