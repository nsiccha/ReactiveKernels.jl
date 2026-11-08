using ReactiveKernels, DifferentiationInterface, Test
import Enzyme

# A slice a plate cell passes only to calls that read it runs as a view
# (src/readonly_slices.jl), and every other slice keeps its copy. A refusal costs
# only the copy, so the tests below pin both directions: the shapes that must read
# in place, and the callees that could tell a view from a copy or would write
# through it, which must see the copy.
module ReadonlySliceCallees
import LinearAlgebra
export wave, buffered, through_helper, loops, aliases, writes, helper_writes, fills, stashes,
    returns_row, returns_wrapped, branches, reads_parent, names_type, methods_differ,
    vector_only, helper_branches, dotted, through_blas, closes_over, sums, total_of

# The reported shape: one row of a coefficient array, read once per coefficient.
function wave(frequency, coefficients, coordinate)
    total = 0.0
    for j in axes(frequency, 2)
        phase = frequency[1, j] * coordinate
        total += coefficients[2j - 1] * cos(phase) + coefficients[2j] * sin(phase)
    end
    total / sqrt(size(frequency, 2))
end
# A buffer of its own that it writes: Julia's effects call this not effect-free.
function buffered(frequency, coefficients, coordinate)
    raw = zeros(2)
    for j in axes(frequency, 2)
        raw[1 + j % 2] += coefficients[2j - 1] * cos(frequency[1, j] * coordinate)
    end
    sum(raw)
end
inner(x, k) = x[k] + x[1] + length(x)
# A fresh vector of a slice's values, from a frame of its own.
@noinline fresh_of(x) = [x[1], x[2]]
through_helper(frequency, coefficients, coordinate) = coordinate * inner(coefficients, 2)
function loops(frequency, coefficients, coordinate)
    total = 0.0
    for v in coefficients
        total += v * coordinate
    end
    total
end
aliases(frequency, coefficients, coordinate) = (row = coefficients; coordinate * row[1])

# Over a view each of these writes the caller's array, or leaks it.
function writes(frequency, coefficients, coordinate)
    coefficients[1] = 0.0
    wave(frequency, coefficients, coordinate)
end
write_helper!(x) = (x[1] = 0.0; x)
function helper_writes(frequency, coefficients, coordinate)
    write_helper!(coefficients)
    wave(frequency, coefficients, coordinate)
end
function fills(frequency, coefficients, coordinate)
    fill!(coefficients, 0.0)
    coordinate
end
const STASH = Any[]
function stashes(frequency, coefficients, coordinate)
    push!(STASH, coefficients)
    wave(frequency, coefficients, coordinate)
end
returns_row(frequency, coefficients, coordinate) = coefficients
returns_wrapped(frequency, coefficients, coordinate) = (coefficients, coordinate)

# Each of these computes something else over a view.
branches(frequency, coefficients, coordinate) =
    coefficients isa Vector ? wave(frequency, coefficients, coordinate) : 0.0 * coordinate
reads_parent(frequency, coefficients, coordinate) = coordinate * parent(coefficients)[1]
names_type(frequency, coefficients, coordinate) =
    coordinate * length(string(typeof(coefficients)))
methods_differ(frequency, coefficients, coordinate) =
    wave(frequency, coefficients, coordinate)
methods_differ(frequency, coefficients::Vector{Float64}, coordinate) =
    2.0 * wave(frequency, coefficients, coordinate)
vector_only(frequency, coefficients::Vector{Float64}, coordinate) =
    wave(frequency, coefficients, coordinate)
branch_in_helper(x) = x isa Vector ? x[1] : 0.0
helper_branches(frequency, coefficients, coordinate) =
    coordinate * branch_in_helper(coefficients)

# Reads the proof cannot follow: these keep the copy and compute what they did.
dotted(frequency, coefficients, coordinate) = sum(coefficients .* coordinate)
# BLAS receives the slice itself. (A callee that first copies part of it,
# `frequency * coefficients[1:k]`, only reads it, which Julia 1.12 and later prove.)
through_blas(frequency, coefficients, coordinate) =
    LinearAlgebra.dot(coefficients, coefficients) * coordinate
closes_over(frequency, coefficients, coordinate) =
    sum(i -> coefficients[i] * coordinate, 1:length(coefficients))
# Base may fold a view in another order than an array (on Julia 1.13 even a row),
# so `sum` of the slice keeps the copy, and its value, bit for bit.
sums(frequency, coefficients, coordinate) = sum(coefficients) * coordinate

# An in-order total that reads its argument only through `iterate`.
function total_of(x)
    total = 0.0
    for v in x
        total += v
    end
    total
end
end

module ReadonlySliceKernels
using ReactiveKernels
using ..ReadonlySliceCallees

_total(x::Number) = x
_total(x::AbstractArray) = sum(x)
_total(x::Tuple) = sum(_total, x)

for callee in (:wave, :buffered, :through_helper, :loops, :aliases, :writes, :helper_writes,
        :fills, :stashes, :returns_row, :returns_wrapped, :branches, :reads_parent, :names_type,
        :methods_differ, :vector_only, :helper_branches, :dotted, :through_blas, :closes_over,
        :sums)
    name = Symbol(:rows_, callee)
    @eval @kernel $name(z::AbstractVector{Float64}, frequency::Matrix{Float64}, block::Vector{Int},
                        coordinate::Vector{Float64}, blocks::Int) = begin
        coefficients = reshape(z, blocks, length(z) ÷ blocks)
        field = plate(block, coordinate) do b, c
            row = coefficients[b, :]
            $callee(frequency, row, c)
        end
        total = sum(_total, field; init = 0.0)
        return total
    end
end

# The other slice forms a cell reads.
@kernel columns(z::AbstractVector{Float64}, frequency::Matrix{Float64}, column::Vector{Int},
                coordinate::Vector{Float64}, blocks::Int) = begin
    coefficients = reshape(z, blocks, length(z) ÷ blocks)
    field = plate(column, coordinate) do k, c
        col = coefficients[:, k]
        total_of(col) * c
    end
    total = sum(field)
    return total
end
@kernel gathers(z::Vector{Float64}, groups::Vector{Vector{Int}}, coordinate::Vector{Float64}) = begin
    field = plate(groups, coordinate) do rows, c
        gathered = z[rows]
        total_of(gathered) * c
    end
    total = sum(field)
    return total
end
@kernel row_range(z::AbstractVector{Float64}, block::Vector{Int}, coordinate::Vector{Float64},
                  blocks::Int) = begin
    coefficients = reshape(z, blocks, length(z) ÷ blocks)
    span = 2:3
    field = plate(block, coordinate) do b, c
        part = coefficients[b, span]
        total_of(part) * c
    end
    total = sum(field)
    return total
end
# A slice nested in a call is one recipe with that call here, so it keeps its
# copy. `@rkppl` splits it into a recipe of its own.
@kernel nested_slice(z::AbstractVector{Float64}, frequency::Matrix{Float64}, block::Vector{Int},
                     coordinate::Vector{Float64}, blocks::Int) = begin
    coefficients = reshape(z, blocks, length(z) ÷ blocks)
    field = plate(block, coordinate) do b, c
        wave(frequency, coefficients[b, :], c)
    end
    total = sum(field)
    return total
end
# A scalar index reads a number, and the cell's result is its slice.
@kernel scalars(z::AbstractVector{Float64}, block::Vector{Int}, coordinate::Vector{Float64},
                blocks::Int) = begin
    coefficients = reshape(z, blocks, length(z) ÷ blocks)
    field = plate(block, coordinate) do b, c
        coefficients[b, 1] * c
    end
    total = sum(field)
    return total
end
@kernel cell_result(z::AbstractVector{Float64}, block::Vector{Int}, blocks::Int) = begin
    coefficients = reshape(z, blocks, length(z) ÷ blocks)
    field = plate(block) do b
        coefficients[b, :]
    end
    total = sum(sum, field)
    return total
end
# Two readers, one of which writes.
@kernel two_readers(z::AbstractVector{Float64}, frequency::Matrix{Float64}, block::Vector{Int},
                    coordinate::Vector{Float64}, blocks::Int) = begin
    coefficients = reshape(z, blocks, length(z) ÷ blocks)
    field = plate(block, coordinate) do b, c
        row = coefficients[b, :]
        wave(frequency, row, c) + writes(frequency, row, c)
    end
    total = sum(field)
    return total
end
# A slice of data bound at preparation holds no differentiated value.
@kernel of_bound(z::Vector{Float64}, data::Matrix{Float64}, block::Vector{Int},
                 frequency::Matrix{Float64}) = begin
    field = plate(block) do b
        row = data[b, :]
        wave(frequency, row, z[1])
    end
    total = sum(field)
    return total
end
end

const K = ReadonlySliceKernels
_in_place(kernel) = occursin("_readonly_slice", sprint(print, code_expr(kernel)))
_baseline(f) = withenv(f, "RK_READONLY_SLICES" => "0")
# Values are compared bit for bit. A gradient can differ from the copy's in its
# last bit: the view's adjoint accumulates straight into the parent's shadow,
# where Enzyme's contracted derivative arithmetic (the dogfood default, `nsz` and
# `contract`) may fuse a multiply-add that the copy's shadow rounds twice. Under
# strict derivative arithmetic (`Enzyme.API.fast_math_flags!()`) they are identical.
_same_gradient(a, b) = isapprox(a, b; rtol = 1e-13)

function _rows_arguments(; n = 6, blocks = 2, F = 3)
    z = collect(range(-0.3, 0.3; length = blocks * 2F))
    frequency = reshape(sin.(1:F), 1, F)
    block = [mod1(s, blocks) for s in 1:n]
    coordinate = cos.((1:n) ./ 5)
    (z, frequency, block, coordinate, blocks)
end

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const IN_PLACE = (:wave, :buffered, :through_helper, :loops, :aliases)
const REFUSED = (:writes, :helper_writes, :fills, :stashes, :returns_row, :returns_wrapped,
                 :branches, :reads_parent, :names_type, :methods_differ, :vector_only,
                 :helper_branches, :dotted, :through_blas, :closes_over, :sums)

@testset "a slice only reads in place when the proof holds" begin
    # A view of the shape a plate cell reads.
    A = reshape(view(collect(1.0:12.0), 1:12), 3, 4)
    V = typeof(view(A, 2, :))
    reads(f, T...) = ReactiveKernels._reads_only(Tuple{typeof(f),T...}, 2)
    @test reads(x -> x[2] + length(x), V)
    @test reads(x -> (s = 0.0; for v in x; s += v; end; s), V)
    @test !reads(x -> x, V)
    @test !reads(x -> (x, 1.0), V)
    @test !reads(x -> (x[1] = 0.0; 1.0), V)
    @test !reads(x -> (fill!(x, 0.0); 1.0), V)
    @test !reads(x -> (x .= 0.0; 1.0), V)
    @test !reads(x -> (unsafe_store!(pointer(x), 0.0); 1.0), V)
    @test !reads(x -> (push!(ReadonlySliceCallees.STASH, x); 1.0), V)
    # A caught exception can hold what the frame threw.
    @test !reads(x -> try; x[100]; catch e; e isa BoundsError ? 1.0 : 2.0; end, V)
    # A frame with no unique method is no proof.
    @test !ReactiveKernels._reads_only(Tuple{typeof(ReadonlySliceCallees.vector_only),Any,V,Any}, 3)

    # Snag readonly-slice-r-cc6a354e: a vararg frame, the shape of every authored
    # cell call (`_KernelSource`), follows only the slice's element of its argument
    # tuple, as a fixed-arity frame follows only its argument. So another argument
    # handed to BLAS does not end the proof for the slice.
    M = [1.0 2.0; 3.0 4.0]
    second(f) = ReactiveKernels._reads_only(Tuple{typeof(f),Matrix{Float64},V}, 3)
    mixes(M, x) = sum(M * [x[1], x[2]])
    forwards(args...) = mixes(args[1], args[2])
    @test second(mixes) && second(forwards)
    @test forwards(M, view(A, 2, :)) == mixes(M, view(A, 2, :))
    # Its own element still may not be written, stored or returned, and a tuple
    # used other than by constant `getfield` is followed whole.
    @test !second((args...) -> (args[2][1] = 0.0; 1.0))
    @test !second((args...) -> args[2])
    @test !second((args...) -> (push!(ReadonlySliceCallees.STASH, args); 1.0))
    @test !second((args...) -> (args, 1.0))
    empty!(ReadonlySliceCallees.STASH)
    # A call's result is derived from the slice only when the callee may return
    # something derived from it: a fresh vector computed from its values (a
    # vector-valued cell) is not, and what a helper hands back of it is.
    fresh(M, x) = M * [x[1], x[2]]
    @test second(fresh) && second((args...) -> fresh(args[1], args[2]))
    @test second((M, x) -> ReadonlySliceCallees.fresh_of(x))
    @test second((M, x) -> M * ReadonlySliceCallees.fresh_of(x))
    @test second((M, x) -> ReadonlySliceCallees.inner(x, 2) * M)
    @test second((M, x) -> identity(x)[1] * M)
    @test !second((M, x) -> identity(x))
    @test !second((M, x) -> (identity(x)[1] = 0.0; M))
end

@testset "every reader must only read: $callee" for callee in IN_PLACE
    args = _rows_arguments()
    kernel = getfield(K, Symbol(:rows_, callee))
    prepared = prepare(kernel; exemplars = args)
    @test _in_place(prepared)
    copied = _baseline(() -> prepare(kernel; exemplars = args))
    @test !_in_place(copied)
    z = copy(args[1])
    @test prepared(args...) == copied(args...)
    @test args[1] == z
    ad = prepare_ad(prepared, BACKEND, args...; active = :z)
    ad_copied = prepare_ad(copied, BACKEND, args...; active = :z)
    value, gradient = ad_value_and_gradient(ad, args...)
    value_copied, gradient_copied = ad_value_and_gradient(ad_copied, args...)
    @test value == value_copied
    @test _same_gradient(gradient, gradient_copied)
    @test args[1] == z
end

@testset "a callee that could tell, or would write, sees the copy: $callee" for callee in REFUSED
    args = _rows_arguments()
    kernel = getfield(K, Symbol(:rows_, callee))
    expected = _baseline(() -> prepare(kernel; exemplars = args))
    prepared = prepare(kernel; exemplars = args)
    @test !_in_place(prepared)
    z = copy(args[1])
    @test isequal(prepared(args...), expected(args...))
    # A write lands in the callee's own copy, never in the caller's array.
    @test args[1] == z
    empty!(ReadonlySliceCallees.STASH)
end

@testset "slice forms" begin
    z = collect(range(-0.3, 0.3; length = 12))
    n = 6
    coordinate = cos.((1:n) ./ 5)
    frequency = reshape(sin.(1:3), 1, 3)
    block = [mod1(s, 2) for s in 1:n]
    @testset "$name" for (name, args, in_place) in (
            (:columns, (z, frequency, [mod1(s, 6) for s in 1:n], coordinate, 2), true),
            (:gathers, (z, [[1, 3], [2], Int[], [4, 5, 6], [7], [8, 9]], coordinate), true),
            (:row_range, (z, block, coordinate, 2), true),
            (:nested_slice, (z, frequency, block, coordinate, 2), false),
            (:scalars, (z, block, coordinate, 2), false),
            (:cell_result, (z, block, 2), false))
        kernel = getfield(K, name)
        prepared = prepare(kernel; exemplars = args)
        copied = _baseline(() -> prepare(kernel; exemplars = args))
        @test _in_place(prepared) == in_place
        @test prepared(args...) == copied(args...)
        in_place || continue
        ad = prepare_ad(prepared, BACKEND, args...; active = :z)
        ad_copied = prepare_ad(copied, BACKEND, args...; active = :z)
        @test _same_gradient(ad_value_and_gradient(ad, args...)[2],
                             ad_value_and_gradient(ad_copied, args...)[2])
    end
    # One reader that writes keeps the copy for both.
    args = (z, frequency, block, coordinate, 2)
    two = prepare(K.two_readers; exemplars = args)
    @test !_in_place(two)
    @test two(args...) == _baseline(() -> prepare(K.two_readers; exemplars = args))(args...)
    @test args[1] == collect(range(-0.3, 0.3; length = 12))
end

@testset "what the lowering needs" begin
    args = _rows_arguments()
    # No exemplars, no concrete types: the program copies, as it always did.
    @test !_in_place(prepare(K.rows_wave))
    # Bound data keeps its copy.
    data = reshape(collect(1.0:12.0), 2, 6)
    frequency = reshape(sin.(1:3), 1, 3)
    block = [1, 2, 1]
    bound = prepare(K.of_bound; have = (:z, :data, :block, :frequency), want = :total,
        bound = (; data, block, frequency), exemplars = ([0.5],))
    @test !_in_place(bound)
    @test bound([0.5]) ≈ sum(ReadonlySliceCallees.wave(frequency, data[b, :], 0.5) for b in block)
end

@testset "the view keeps the slice's bounds check and empty cells" begin
    args = _rows_arguments()
    prepared = prepare(K.rows_wave; exemplars = args)
    @test _in_place(prepared)
    z, frequency, block, coordinate, blocks = args
    @test_throws BoundsError prepared(z, frequency, [1, 3, 1, 1, 1, 1], coordinate, blocks)
    copied = _baseline(() -> prepare(K.rows_wave; exemplars = args))
    @test_throws BoundsError copied(z, frequency, [1, 3, 1, 1, 1, 1], coordinate, blocks)
    # No cells, and rows of no coefficients.
    @test prepared(z, frequency, Int[], Float64[], blocks) == 0.0
    no_frequency = reshape(Float64[], 1, 0)
    @test isequal(prepared(z, no_frequency, block, coordinate, blocks),
                  copied(z, no_frequency, block, coordinate, blocks))
end

@testset "bytes per gradient do not hold a copy of each row" begin
    bytes(kernel, args) = begin
        ad = prepare_ad(kernel, BACKEND, args...; active = :z)
        ad_value_and_gradient(ad, args...)
        minimum(@allocated(ad_value_and_gradient(ad, args...)) for _ in 1:3)
    end
    sized(F) = _rows_arguments(; n = 200, blocks = 4, F)
    small, large = sized(8), sized(64)
    in_place(args) = prepare(K.rows_wave; exemplars = args)
    copying(args) = _baseline(() -> prepare(K.rows_wave; exemplars = args))
    @test _in_place(in_place(small)) && !_in_place(copying(small))
    # The copy's bytes grow with the row, by rows × length × 8 at the least. The
    # view's grow only by the gradient vector and its shadow, of the point's length.
    @test bytes(copying(large), large) - bytes(copying(small), small) >= 200 * (128 - 16) * 8
    @test bytes(in_place(large), large) - bytes(in_place(small), small) <= 2 * (512 - 64) * 8 + 256
end
