module TestJSTuplesMatrices

# `js_source` of predicates over tuples (`all(f, t)`, `any`, `count`), matrix
# literals of numbers (`[a b; c d]`, `[a b]`, `Float64[a b; c d]`), matrices
# built by writing their elements (`Matrix{T}(undef, m, n)`, `m[i, j] = x`,
# `m[:, j] = v`, `m[i, :] = v`) and `diag`, each module checked in Node against
# the native kernel (snag js-source-all-pr-b15faf4f).

using ReactiveKernels
using LinearAlgebra: diag
using Test

const NODE = ReactiveKernels._js_default_node() !== nothing

# Predicates over a tuple call the function on each element with that
# element's own type, and `all`/`any` stop at the first element that decides
# them: `b[1]` is not read once `a[1] <= 0`.
js_same_lengths(a, b) = all(v -> length(v) == length(a), (a, b))
js_first_positive(a, b) = all(v -> v[1] > 0, (a, b))
js_any_negative(x, k) = any(<(0), (x, k))
js_count_large(x, y, z) = count(v -> abs(v) > 1, (x, y, z))
js_all_flags(p, q) = all((p, q))
js_any_flags(p, q) = any((p, q))
js_count_flags(p, q) = count((p, q))
js_static_rows(x) = all(>=(1), (2, 2)) ? x : -x
js_empty_tuple(x) = all(<(0), ()) && !any(<(0), ()) ? x + count(<(0), ()) : -x

# Tuple length inside a called validation helper must translate even when
# Julia only needs the call in the exceptional branch of inferred code.
function js_check_axis_count(domain, d)
    length(domain) == d || throw(ArgumentError("expected $d axes, got $(length(domain))"))
    nothing
end
function js_domain_width(lo, hi, d)
    domain = ((lo, hi), (lo - 1, hi + 1))
    js_check_axis_count(domain, d)
    domain[1][2] - domain[1][1] + domain[2][2] - domain[2][1]
end
function js_fixed_domain_width(d)
    domain = ((-2.0, 2.0), (-3.0, 3.0))
    js_check_axis_count(domain, d)
    domain[d][2] - domain[d][1]
end
@kernel js_tuple_length_validation(lo, hi, d) = begin
    width = js_domain_width(lo, hi, d)
    fixed = js_fixed_domain_width(d)
    return width
end

function js_fill_owned(v, x)
    out = copy(v)
    alias = out
    fill!(out, x)
    alias
end
function js_fill_ints(n, x)
    out = Vector{Int}(undef, n)
    fill!(out, x)
end
function js_fill_flags(n, flag)
    out = Vector{Bool}(undef, n)
    fill!(out, flag)
end
@kernel js_fill_vectors(v, x, n, flag) = begin
    floats = js_fill_owned(v, x)
    ints = js_fill_ints(n, x)
    flags_as_ints = js_fill_ints(n, flag)
    flags = js_fill_flags(n, flag)
    flags_as_floats = js_fill_owned(v, flag)
    return floats
end

# A keyword sorter reuses an anonymous local for differently typed values.
# Its earlier SSA reads, and a saved reference to a reassigned argument,
# must retain the value they read.
js_kw_values(v; k = [1, 2], c = 1.5, domain = ((0.0, 1.0),), by = nothing) =
    sum(v) + sum(k) + c + domain[1][2] + (by === nothing ? 0.0 : sum(by))
js_kw_caller(v, x) = js_kw_values(v; k = [2, 3], domain = ((x, x + 1),))
function js_saved_array(v, x)
    saved = v
    v = [x]
    sum(saved) + sum(v)
end
function js_fill_if(v, x, flag)
    out = copy(v)
    flag && fill!(out, x)
    out
end
@kernel js_slot_snapshots(v, x, flag) = begin
    keywords = js_kw_caller(v, x)
    saved = js_saved_array(v, x)
    lazy = js_fill_if([1, 2], x, flag)
    return keywords
end

@kernel js_tuple_preds(a, b, x, k, y, z, p, q) = begin
    same = js_same_lengths(a, b)
    first_positive = js_first_positive(a, b)
    any_negative = js_any_negative(x, k)
    nlarge = js_count_large(x, y, z)
    both = js_all_flags(p, q)
    either = js_any_flags(p, q)
    nflags = js_count_flags(p, q)
    rows = js_static_rows(x)
    empty = js_empty_tuple(x)
    return same
end

# Matrix literals of numbers: in a helper they call Base's `hvcat`, `hcat` and
# `typed_hvcat`; in a kernel's recipe RK's `_native_hvcat` and `_native_hcat`.
# Julia promotes the operands to one element type (a `Bool` among numbers is
# `0` or `1`).
js_rotation(t) = [cos(t) -sin(t); sin(t) cos(t)]
js_row(a, b) = [a b 1]
js_typed(a, k) = Float64[a k; k a]
js_int_grid(i, j) = [i j; j i]
js_flags(p) = [p !p; true p]
js_mixed_flags(p, i) = [p i]

@kernel js_matrix_literals(t, a, b, k, i, j, p, v) = begin
    rotation = js_rotation(t)
    row = js_row(a, b)
    typed = js_typed(a, k)
    grid = js_int_grid(i, j)
    flags = js_flags(p)
    mixed = js_mixed_flags(p, i)
    factor = [1.0 0.0; v sqrt(1.0 - v * v)]
    recipe_row = [a b]
    return rotation
end

# A layout `hvcat` rejects: Julia's error, raised where the literal is.
js_bad_layout(a) = hvcat((2, 1), a, a, a)
@kernel js_matrix_bad_layout(a) = begin
    m = js_bad_layout(a)
    return m
end

# Matrices written element by element, by column and by row.
function js_fill_grid(n, m, x)
    out = Matrix{Float64}(undef, n, m)
    for j in 1:m, i in 1:n
        out[i, j] = x * i + j
    end
    out
end
function js_set_columns(v)
    out = zeros(length(v), 2)
    out[:, 1] = v
    out[1, :] = [v[1], 7.0]
    out[end, 2] = length(v)
    out
end
function js_int_cells(n)
    out = Matrix{Int}(undef, n, n)
    for i in 1:n, j in 1:n
        out[i, j] = i * j
    end
    out
end
function js_flag_cells(n)
    out = Matrix{Bool}(undef, n, 1)
    for i in 1:n
        out[i, 1] = isodd(i)
    end
    out
end
js_diag_of(A) = diag(A)

@kernel js_matrix_writes(n, m, x, v, A) = begin
    grid = js_fill_grid(n, m, x)
    columns = js_set_columns(v)
    ints = js_int_cells(n)
    flags = js_flag_cells(n)
    d = js_diag_of(A)
    return grid
end

# Writes outside the matrix, and a column of the wrong length.
js_write_outside(n) = (out = zeros(n, n); out[n + 1, 1] = 1.0; out)
js_column_mismatch(n) = (out = zeros(n, 2); out[:, 2] = ones(n + 1); out)
@kernel js_matrix_bad_writes(n) = begin
    outside = js_write_outside(n)
    mismatch = js_column_mismatch(n)
    return outside
end

# A tuple of values of one type read at a position computed at run time (a
# domain pair per axis, an axis vector), and a closure built on every pass of a
# loop: it holds the values it captured on that pass.
function js_inside_axes(a, b, lo1, hi1, lo2, hi2)
    axes_ = (a, b)
    domain = ((lo1, hi1), (lo2, hi2))
    inside = 0
    width = 0.0
    for j in 1:2
        lo, hi = domain[j][1], domain[j][2]
        all(v -> lo <= v <= hi, axes_[j]) && (inside += 1)
        width += (hi - lo) * length(axes_[j])
    end
    inside + width
end
js_axis_at(a, b, j) = sum((a, b)[j])

@kernel js_tuple_reads(a, b, lo1, hi1, lo2, hi2, j) = begin
    inside = js_inside_axes(a, b, lo1, hi1, lo2, hi2)
    picked = js_axis_at(a, b, j)
    return inside
end

# Matrix slices read in place or as a whole, and `extrema`.
js_col_spans(A) = [(e = extrema(view(A, :, j)); e[2] - e[1]) for j in 1:size(A, 2)]
js_row_total(A, i) = sum(view(A, i, :))
js_slice_reads(A, i, j) = view(A, :, j)[i] + length(view(A, i, :)) + maximum(view(A, :, j))
js_spread(x) = (e = extrema(x); e[2] - e[1])

@kernel js_matrix_slices(A, i, j, x) = begin
    spans = js_col_spans(A)
    total = js_row_total(A, i)
    reads = js_slice_reads(A, i, j)
    spread = js_spread(x)
    return spans
end

# A tuple predicate that may not return a `Bool` is refused, naming it.
js_not_bool(x) = all(v -> v > 0 ? true : nothing, (x, x))
@kernel js_tuple_not_bool(x) = begin
    y = js_not_bool(x)
    return y
end

const V = Vector{Float64}
const M = Matrix{Float64}
const CASES = [
    (:js_tuple_length_validation, js_tuple_length_validation, (:lo, :hi, :d), (:width, :fixed),
     (; lo = Float64, hi = Float64, d = Int), [(0.0, 1.0, 2), (-3.0, 5.0, 2), (0.0, 1.0, 1)]),
    (:js_fill_vectors, js_fill_vectors, (:v, :x, :n, :flag),
     (:floats, :ints, :flags_as_ints, :flags, :flags_as_floats),
     (; v = V, x = Float64, n = Int, flag = Bool),
     [([1.0, 2.0], -2.0, 3, true), (Float64[], 0.0, 0, false),
      (ones(31), -0.0, 19, false), ([1.0], 0.5, 2, true),
      (Float64[], 0.5, 0, true)]),
    (:js_slot_snapshots, js_slot_snapshots, (:v, :x, :flag), (:keywords, :saved, :lazy),
     (; v = V, x = Float64, flag = Bool),
     [([1.0, 2.0], 3.0, true), (Float64[], 0.5, false), (ones(31), -2.0, false),
      ([1.0], 0.5, true)]),
    (:js_tuple_preds, js_tuple_preds, (:a, :b, :x, :k, :y, :z, :p, :q),
     (:same, :first_positive, :any_negative, :nlarge, :both, :either, :nflags, :rows, :empty),
     (; a = V, b = V, x = Float64, k = Int, y = Float64, z = Float64, p = Bool, q = Bool),
     [([1.0, 2.0], [3.0, 4.0], 0.5, 2, -3.0, 1.5, true, true),
      ([1.0], [3.0, 4.0], -0.5, 0, 0.25, -0.5, true, false),
      ([-1.0], Float64[], 2.0, -1, 1.0, NaN, false, false),   # b[1] is never read
      ([1.0], Float64[], 2.0, 3, 4.0, 5.0, false, true)]),    # b[1]: a BoundsError
    (:js_matrix_literals, js_matrix_literals, (:t, :a, :b, :k, :i, :j, :p, :v),
     (:rotation, :row, :typed, :grid, :flags, :mixed, :factor, :recipe_row),
     (; t = Float64, a = Float64, b = Float64, k = Int, i = Int, j = Int, p = Bool, v = Float64),
     [(0.3, 1.5, -2.0, 4, 2, -3, true, 0.5), (-1.0, 0.0, -0.0, -7, 0, 1, false, -0.25),
      (0.0, NaN, Inf, 0, 5, 5, true, 1.0), (2.0, 1.0, 2.0, 1, 1, 1, false, 1.5)]),  # sqrt(1 - v^2): DomainError
    (:js_matrix_bad_layout, js_matrix_bad_layout, (:a,), (:m,), (; a = Float64), [(1.0,)]),
    (:js_matrix_writes, js_matrix_writes, (:n, :m, :x, :v, :A), (:grid, :columns, :ints, :flags, :d),
     (; n = Int, m = Int, x = Float64, v = V, A = M),
     [(2, 3, 0.5, [1.0, -2.0, 3.0], [1.0 2.0 3.0; 4.0 5.0 6.0]), (0, 2, 1.0, [4.0], zeros(0, 3)),
      (3, 0, -1.0, [1.0, 2.0], [1.0 2.0; 3.0 4.0; 5.0 6.0]),
      (2, 2, 1.0, Float64[], [1.0;;]),       # v[1]: a BoundsError
      (-1, 2, 1.0, [1.0], [1.0;;])]),        # Matrix{Float64}(undef, -1, 2): an ArgumentError
    (:js_matrix_bad_writes, js_matrix_bad_writes, (:n,), (:outside, :mismatch), (; n = Int), [(2,), (0,)]),
    (:js_tuple_reads, js_tuple_reads, (:a, :b, :lo1, :hi1, :lo2, :hi2, :j), (:inside, :picked),
     (; a = V, b = V, lo1 = Float64, hi1 = Float64, lo2 = Float64, hi2 = Float64, j = Int),
     [([0.0, 0.5], [1.0, 2.0, 3.0], -1.0, 1.0, 0.0, 2.5, 1), ([0.0, 0.5], [1.0, 2.0, 3.0], -1.0, 1.0, 0.0, 3.5, 2),
      (Float64[], [NaN], 0.0, 1.0, 0.0, 1.0, 2),
      ([0.0], [1.0], 0.0, 1.0, 0.0, 1.0, 3)]),   # (a, b)[3]: a BoundsError
    (:js_matrix_slices, js_matrix_slices, (:A, :i, :j, :x), (:spans, :total, :reads, :spread),
     (; A = M, i = Int, j = Int, x = V),
     [([1.0 -2.0 0.5; 3.0 NaN -1.0], 2, 3, [2.0, -1.0, 0.25]), ([4.0 5.0], 1, 2, [-0.0, 0.0]),
      ([1.0 2.0; 3.0 4.0], 3, 1, [1.0]),        # view(A, 3, :): a BoundsError
      ([1.0 2.0; 3.0 4.0], 1, 1, Float64[]),     # extrema(Float64[]): an error
      (zeros(0, 2), 1, 1, [1.0])]),              # extrema of an empty column
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) ||
    (iszero(a) ? iszero(b) && signbit(a) == signbit(b) : isapprox(a, b; rtol = 1e-12, atol = 1e-15))
agree(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(agree.(a, b))
agree(a, b) = false

# The code of one function of a module.
function body(source, name)
    i = findfirst("function " * name * "(", source)
    source[i[1]:findnext("\n}\n", source, i[1])[end]]
end

@testset "js_source of tuple predicates and matrices" begin
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

    @testset "a tuple predicate is unrolled and stops at the deciding element" begin
        source = ReactiveKernels.js_source(js_tuple_preds; have = (:a, :b, :x, :k, :y, :z, :p, :q),
                                           want = :first_positive,
                                           types = (; a = V, b = V, x = Float64, k = Int, y = Float64, z = Float64,
                                                    p = Bool, q = Bool))
        code = body(source, "js_first_positive")
        @test !occursin("for (let", code)
        @test occursin("break", code)
        # a predicate known while translating leaves no code
        rows = ReactiveKernels.js_source(js_tuple_preds; have = (:a, :b, :x, :k, :y, :z, :p, :q), want = :rows,
                                         types = (; a = V, b = V, x = Float64, k = Int, y = Float64, z = Float64,
                                                  p = Bool, q = Bool))
        @test !occursin("break", body(rows, "js_static_rows"))
    end

    @testset "a matrix literal is its rows" begin
        source = ReactiveKernels.js_source(js_matrix_literals; have = (:t, :a, :b, :k, :i, :j, :p, :v),
                                           want = :factor,
                                           types = (; t = Float64, a = Float64, b = Float64, k = Int, i = Int,
                                                    j = Int, p = Bool, v = Float64))
        @test occursin("[[1.0, 0.0], [v, ", source)
    end

    @testset "a non-Bool predicate is named" begin
        e = try
            ReactiveKernels.js_source(js_tuple_not_bool; have = (:x,), want = :y, types = (; x = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError && occursin("must return a `Bool`", e.message)
    end
end

end
