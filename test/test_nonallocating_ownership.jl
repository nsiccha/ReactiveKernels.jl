using ReactiveKernels
using MutatingFunctions
using LinearAlgebra: Symmetric, cholesky, PosDefException
using Test
using ReactiveKernels: plate, scan

# Cache ownership and result typing of `prepare_nonallocating`: a cache slot is
# a write destination only for storage the kernel owns, a declared plate output
# fixes the buffer's element type, and exemplar arguments type the program from
# the runtime argument types.

function ownership_allocations(k, args)
    k(args...)
    k(args...)
    @allocated k(args...)
end

# A lazy observation domain whose representation depends on the plan type: a
# range for one, row slices of a stored index matrix for the other.
struct RangeIndexPlan
    n::Int
    shifts::Vector{Int}
end
struct RowIndexPlan
    rows::Matrix{Int}
end
ownership_domain(p::RangeIndexPlan) = 1:p.n
ownership_domain(p::RowIndexPlan) = eachrow(p.rows)
ownership_lag(t, p::RangeIndexPlan, j) = t - p.shifts[j]
ownership_lag(row, ::RowIndexPlan, j) = row[j]

const LAZY_DOMAIN = @kernel lazy_domain(plan, w::Vector{Float64},
                                        u::Vector{Float64}) = begin
    observations = ownership_domain(plan)
    y::Vector{Float64} = plate(observations) do t
        sum(w[j] * get(u, ownership_lag(t, plan, j), 0.0) for j in eachindex(w);
            init = 0.0)
    end
    odd::Vector{Float64} = y[1:2:end]
    even = y[2:2:end]
    trajectory::AbstractVector{Float64} = scan(even; init = 0.0,
                                               include_init = true) do c, v
        next = 0.5 * c + v
        (next, next)
    end
    return odd, trajectory
end

const FIELD_SCALE = @kernel field_scale(sched, factor::Float64) = begin
    xs = sched.xs
    ys::Vector{Float64} = xs .* factor
    total::Float64 = sum(ys)
end

const TYPED_FIELD = @kernel typed_field(
        sched::NamedTuple{(:xs,),Tuple{Vector{Float64}}}) = begin
    xs = sched.xs
    total::Float64 = sum(xs)
end

const ROW_COUNT = @kernel row_count(plan) = begin
    rows = ownership_domain(plan)
    n = length(rows)
end

@traceable function ownership_withrow(A, i, v)
    B = copy(A)
    for j in eachindex(v)
        B[i + 1, j] = v[j]
    end
    B
end

const PUT_ROW = @kernel ownership_putrow(A, i, v) = begin
    result = ownership_withrow(A, i, v)
end

const OWNED_CHOLESKY = @kernel ownership_cholesky(A, triangle) = begin
    result = cholesky(Symmetric(A, triangle))
end
const CHOLESKY_ENTRY = @kernel ownership_cholesky_entry(A) = begin
    factor = cholesky(Symmetric(A)).L
    entry = factor[2, 1]
end
const OWNED_LOWER = @kernel ownership_lower(A) = begin
    result = rk_cholesky_lower(A)
end

@testset "nonallocating cache ownership and result typing" begin
    @testset "Cholesky workspaces respect triangle, shape and input ownership" begin
        A = [4.0 1.0; 2.0 5.0]
        chol = prepare_nonallocating(OWNED_CHOLESKY, A, :U; want=:result)
        lower = prepare_nonallocating(OWNED_LOWER, A; want=:result)
        original = copy(A)
        first_chol = chol(A, :U)
        first_lower = lower(A)
        entry = prepare_nonallocating(CHOLESKY_ENTRY, A; want=:entry)
        @test entry(A) == cholesky(Symmetric(A)).L[2, 1]
        # LinearAlgebra 1.10 materializes the transposed L of an upper
        # factor. Later versions return a view over its factors.
        if Base.mightalias(parent(first_chol.L), first_chol.factors)
            @test ownership_allocations(entry, (A,)) == 0
        end
        @test first_chol.L == cholesky(Symmetric(A)).L
        @test first_lower == rk_cholesky_lower(A)
        @test A == original
        A[2, 1] = -0.5
        original = copy(A)
        changed_chol = chol(A, :L)
        parent(first_lower)[1, 2] = NaN
        changed_lower = lower(A)
        @test changed_chol.factors === first_chol.factors
        @test parent(changed_lower) === parent(first_lower)
        @test changed_chol.L == cholesky(Symmetric(A, :L)).L
        @test parent(changed_lower) == parent(rk_cholesky_lower(A))
        @test A == original

        # A previous result can itself be the next call's covariance.
        for (run, reference, storage) in (
                (B -> chol(B, :L).L, B -> cholesky(Symmetric(B, :L)).L, changed_chol.factors),
                (lower, rk_cholesky_lower, parent(changed_lower)))
            before = copy(storage)
            factor = run(storage)
            @test isequal(storage, before)
            @test factor == reference(before)
        end
        for B in (zeros(0, 0), [9.0;;], [6.0 0.0; 0.0 3.0])
            @test chol(B, :U).L == cholesky(Symmetric(B)).L
            @test lower(B) == rk_cholesky_lower(B)
        end
        B = view(Float32.(A), :, :)
        chol32 = prepare_nonallocating(OWNED_CHOLESKY, B, :U; want=:result)
        lower32 = prepare_nonallocating(OWNED_LOWER, B; want=:result)
        f32, l32 = chol32(B, :U), lower32(B)
        B[1, 1] += 1
        @test chol32(B, :U).factors === f32.factors
        @test parent(lower32(B)) === parent(l32)
        @test f32.L == cholesky(Symmetric(B)).L
        @test l32 == rk_cholesky_lower(B)
        @test_throws PosDefException lower([-1.0;;])
        @test_throws PosDefException chol([-1.0;;], :U)
        @test lower([9.0;;])[1, 1] == chol([9.0;;], :U).L[1, 1] == 3.0
    end

    @testset "indexed writes fill an independent copy" begin
        p = prepare(PUT_ROW; have = (:A, :i, :v), want = :result)
        A = reshape(collect(1.0:32), 4, 8)
        original = copy(A)
        v = ones(8)
        run = prepare_nonallocating(p, A, 0, v)
        @test run(A, 0, v) == ownership_withrow(A, 0, v)
        @test ownership_allocations(run, (A, 0, v)) == 0
        # Runtime extents, including a partial row and an empty index domain.
        for (matrix, i, row) in ((A, 2, [3.0, 4.0]), (A, 99, Float64[]),
                                 (zeros(2, 19), 1, ones(19)))
            @test run(matrix, i, row) == ownership_withrow(matrix, i, row)
            @test ownership_allocations(run, (matrix, i, row)) == 0
        end
        @test_throws BoundsError run(A, 4, v)
        @test_throws BoundsError run(A, 0, ones(9))
        old = run(A, 0, v)
        saved = copy(old)
        next = run(old, 2, v)
        @test next == ownership_withrow(saved, 2, v)
        @test !Base.mightalias(next, old)
        @test old == saved

        # The source row overlaps the row being replaced, in reverse order.
        row = view(A, 2, 8:-1:1)
        views = prepare_nonallocating(p, A, 1, row)
        borrowed = views(A, 1, row)
        @test borrowed == ownership_withrow(original, 1, row)
        @test ownership_allocations(views, (A, 1, row)) == 0
        @test A == original
        # A view into the previous destination is also a live input.
        rhs = view(borrowed, 2, 8:-1:1)
        snapshot = copy(borrowed)
        expected = ownership_withrow(A, 0, rhs)
        result = views(A, 0, rhs)
        @test result == expected
        @test !Base.mightalias(result, rhs)
        @test borrowed == snapshot
    end
    @testset "a range-valued step is recomputed, not written into" begin
        k = prepare_nonallocating(LAZY_DOMAIN; want = :y)
        reference = prepare(LAZY_DOMAIN; want = :y)
        first_args = (RangeIndexPlan(9, [0, 2, 4]), [1.0, 0.5, 0.25],
                      collect(1.0:9.0))
        second_args = (RangeIndexPlan(11, [0, 3]), [2.0, 1.0], collect(1.0:11.0))
        @test k(first_args...) == reference(first_args...)
        @test k(first_args...) == reference(first_args...)
        @test k(second_args...) == reference(second_args...)
        rows = (RowIndexPlan([1 3; 2 4; 5 0]), [1.0, 0.25], collect(1.0:6.0))
        @test k(rows...) == reference(rows...)
        @test k(first_args...) == reference(first_args...)
    end

    @testset "steps never write into caller-owned inputs" begin
        for (label, k) in (("untyped", prepare_nonallocating(FIELD_SCALE;
                                                              want = :total)),
                           ("exemplar", prepare_nonallocating(FIELD_SCALE,
                                (; xs = [0.0], tag = 0), 1.0; want = :total)))
            @testset "$label field read" begin
                first_input = (; xs = [1.0, 2.0, 3.0], tag = 1)
                second_input = (; xs = [10.0, 20.0, 30.0], tag = 2)
                @test k(first_input, 2.0) == 12.0
                @test k(second_input, 2.0) == 120.0
                @test first_input.xs == [1.0, 2.0, 3.0]
                @test second_input.xs == [10.0, 20.0, 30.0]
            end
        end
        typed = prepare_nonallocating(TYPED_FIELD)
        first_input = (; xs = [1.0, 2.0, 3.0])
        second_input = (; xs = [10.0, 20.0, 30.0])
        @test typed(first_input) == 6.0
        @test typed(second_input) == 60.0
        @test first_input.xs == [1.0, 2.0, 3.0]

        counted = prepare_nonallocating(ROW_COUNT)
        first_plan = RowIndexPlan([1 2; 3 4])
        second_plan = RowIndexPlan([5 6; 7 8; 9 10])
        @test counted(first_plan) == 2
        @test counted(second_plan) == 3
        @test first_plan.rows == [1 2; 3 4]
        @test counted(RangeIndexPlan(4, Int[])) == 4
    end

    @testset "reactive in-place slots never write into caller inputs" begin
        g = Graph()
        source = value!(g, :source, Any)
        field = value!(g, :field, Vector{Float64})
        add!(g, source => field, s -> s.xs)
        program = prepare_reactive_nonallocating(g; have = (source,),
                                                 want = (field,))
        first_input = (; xs = [1.0, 2.0])
        second_input = (; xs = [10.0, 20.0])
        state = program(first_input)
        handle = statevalue(state, field)
        @test get!(state, handle) == [1.0, 2.0]
        set!(state, statevalue(state, source), second_input)
        @test get!(state, handle) == [10.0, 20.0]
        @test first_input.xs == [1.0, 2.0]
        @test !occursin("__cache_apply__((__slots__",
                        string(code_expr(program, field)))
    end

    @testset "a reshape shares only the current producer's storage" begin
        spec = @kernel shaped_values(x::Vector{Float64}, m::Int, n::Int) = begin
            data = x .* 2
            shaped = reshape(data, m, n)
        end
        k = prepare_nonallocating(spec; want = :shaped)
        for (m, n) in ((3, 4), (4, 3), (1, 7), (0, 5))
            x = Float64.(1:m*n)
            y = k(x, m, n)
            @test y == reshape(2 .* x, m, n)
            @test k(x, m, n) === y
            @test ownership_allocations(k, (x, m, n)) == 0
            @test x == Float64.(1:m*n)
        end
        @test_throws DimensionMismatch k(zeros(7), 3, 2)

        @static if hasfield(Array, :ref)
            both = prepare_nonallocating(spec; want = (:data, :shaped))
            x = collect(1.0:4.0)
            data, _ = both(x, 2, 2)
            sizehint!(data, 30)
            data, original = both(x, 2, 2)
            popfirst!(data)
            push!(data, -1.0)
            data, current = both(x, 2, 2)
            @test current !== original
            @test current == reshape(2 .* x, 2, 2)
            @test Base.mightalias(data, current)
        end

        borrowed = @kernel shaped_input(x::Vector{Float64}) = begin
            shaped = reshape(x, 2, 2)
        end
        plain = prepare_nonallocating(borrowed; want = :shaped)
        a, b = ones(4), zeros(4)
        aview = plain(a)
        bview = plain(b)
        aview[1] = 2
        @test a == [2, 1, 1, 1]
        @test all(iszero, bview)
        @test Base.mightalias(b, bview)
    end

    @testset "registered destinations still reuse owned buffers" begin
        g = Graph()
        x = value!(g, :x, Vector{Float64})
        copied = value!(g, :copied, Vector{Float64})
        add!(g, x => copied, copy)
        k = prepare_nonallocating(plan(g; have = (x,), want = (copied,)))
        input = [1.0, 2.0]
        first_result = k(input)
        @test first_result !== input
        @test k([3.0, 4.0]) === first_result
        @test first_result == [3.0, 4.0]
        @test input == [1.0, 2.0]
    end

    @testset "a declared plate output fixes the buffer element type" begin
        args = (RangeIndexPlan(9, [0, 2, 4]), [1.0, 0.5, 0.25], collect(1.0:9.0))
        for want in (:y, :odd, (:odd, :trajectory))
            k = prepare_nonallocating(LAZY_DOMAIN; want = want)
            reference = prepare(LAZY_DOMAIN; want = want)(args...)
            result = k(args...)
            @test typeof(result) == typeof(reference)
            @test result == reference
        end
        k = prepare_nonallocating(LAZY_DOMAIN; want = :y)
        y = k(args...)
        @test y isa Vector{Float64}
        # The kernel keeps that buffer, of the declared element type.
        @test any(slot -> slot isa Base.RefValue && slot[] === y, k.caches)
        @test k(args...) === y
    end

    @testset "a gathered power broadcast only writes its own result" begin
        spec = @kernel gathered_powers(x::Vector{Float64}, g::Vector{Int},
                powers::Vector{Int}) = begin
            probability = x[g] .^ powers .* x[g]
            terms = plate(eachindex(g)) do i
                probability[i]
            end
            total::Float64 = sum(terms)
        end
        k = prepare_nonallocating(spec; have = (:x, :g, :powers), want = :total)
        g = [3, 1, 1, 2]
        powers = [0, 3, 1, 2]
        x = [0.2, 0.3, 0.4]
        other = reverse(x)
        for values in (x, other, x)
            @test k(values, g, powers) == sum(values[g] .^ powers .* values[g])
            @test ownership_allocations(k, (values, g, powers)) == 0
        end
        @test x == [0.2, 0.3, 0.4]
        @test other == [0.4, 0.3, 0.2]
    end

    @testset "exemplar arguments type the program" begin
        args = ((; xs = [1.0, 2.0, 3.0], tag = 1), 2.0)
        exemplar = prepare_nonallocating(FIELD_SCALE, args...; want = :ys)
        untyped = prepare_nonallocating(FIELD_SCALE; want = :ys)
        # After the constants, callables and caches tables and the cache driver.
        signature = code_expr(exemplar).args[1].args
        @test signature[5] == Expr(:(::), :sched, typeof(args[1]))
        @test signature[6] == Expr(:(::), :factor, Float64)
        @test exemplar(args...) == prepare(FIELD_SCALE; want = :ys)(args...)
        @test exemplar((; xs = [4.0, 5.0], tag = 3), 0.5) == [2.0, 2.5]
        @test ownership_allocations(exemplar, args) == 0
        println("NONALLOCATING_ALLOC_BYTES\texemplar_field_scale\t",
                ownership_allocations(exemplar, args), "\tuntyped\t",
                ownership_allocations(untyped, args))

        @test_throws ArgumentError exemplar((; xs = [1.0f0], tag = 1), 2.0)
        @test_throws ArgumentError prepare_nonallocating(FIELD_SCALE,
                                                         args[1]; want = :ys)
        @test_throws ArgumentError prepare_nonallocating(TYPED_FIELD,
                                                         (; xs = [1]))

        plan_args = (RangeIndexPlan(9, [0, 2, 4]), [1.0, 0.5, 0.25],
                     collect(1.0:9.0))
        k = prepare_nonallocating(LAZY_DOMAIN, plan_args...;
                                  want = (:odd, :trajectory))
        reference = prepare(LAZY_DOMAIN; want = (:odd, :trajectory))
        @test k(plan_args...) == reference(plan_args...)
        other = (RangeIndexPlan(11, [0, 3]), [2.0, 1.0], collect(1.0:11.0))
        @test k(other...) == reference(other...)
        @test_throws ArgumentError k(RowIndexPlan([1 2; 3 4]), [1.0, 0.5],
                                     collect(1.0:4.0))

        prepared = prepare(FIELD_SCALE; have = (:sched, :factor), want = :ys)
        from_prepared = prepare_nonallocating(prepared, args...)
        @test from_prepared(args...) == prepared(args...)
        @test ownership_allocations(from_prepared, args) == 0
    end
end

module BufferedIntoTests

using ReactiveKernels, MutatingFunctions, Test

@kernel product(A, x) = begin
    y = A * x
end
@kernel typed_product(A::Matrix{Float64}, x::Vector{Float64}) = begin
    y = A * x
end
@kernel joined(a, b, c) = begin
    y = (a .- b) .+ c
end
@kernel centered(a, b, c) = begin
    y = a .- (b .+ c) ./ 2
end
@kernel staged(A, x, c) = begin
    ax = A * x
    y = ax .- c
end
@kernel pointwise(a, h, g) = begin
    y = a .+ (h / 2) .* g
end
@kernel ragged_copy(x) = begin
    y = reverse(x)
end
@kernel product_entry(A, x, i) = begin
    ax = A * x
    y = fill(sum(view(ax, i)), 1)
end

@noinline writer_bytes(w::W, dst, xs::T) where {W,T} = @allocated w(dst, xs...)
@noinline retained_bytes(f::F, dst, xs::T) where {F,T} = @allocated copyto!(dst, f(xs...))

# A destination is allowed to retain the source passed to its copy protocol.
struct RetainsSource{T} <: AbstractVector{T}
    data::Vector{T}
    source::Base.RefValue{Vector{T}}
end
Base.size(d::RetainsSource) = size(d.data)
function Base.copyto!(d::RetainsSource{T}, src::Vector{T}) where {T}
    d.source[] = src
    copyto!(d.data, src)
    d
end

@testset "caller-owned writers reuse private numerical results" begin
    for T in (Float32, Float64)
        x = T.(1:8)
        b, c = x .- 1, 2x
        A = reshape(T.(1:64), 8, 8)
        for (spec, xs, expected) in (
                (product, (A, x), A * x),
                (joined, (x, b, c), (x .- b) .+ c),
                (centered, (x, b, c), x .- (b .+ c) ./ 2),
                (staged, (A, x, c), A * x .- c),
                (pointwise, (x, T(0.25), c), x .+ T(0.125) .* c))
            k = prepare(spec; have = Tuple(spec.have_names), want = :y)
            w, f = prepare_into(k, xs...), prepare_nonallocating(k, xs...)
            dst = similar(expected)
            @test (@inferred w(dst, xs...)) === dst
            @test isequal(dst, expected)
            writer_bytes(w, dst, xs)
            retained_bytes(f, dst, xs)
            @test writer_bytes(w, dst, xs) == retained_bytes(f, dst, xs) == 0
        end
    end

    x = collect(1.:8.)
    A = reshape(collect(1.:64.), 8, 8)
    k = prepare(product; have = (:A, :x), want = :y)
    w = prepare_into(k, A, x)
    kept = w(similar(x), A, x)
    declared = prepare_into(typed_product)
    @test declared(similar(x), A, x) == kept
    writer_bytes(declared, similar(x), (A, x))
    @test writer_bytes(declared, similar(x), (A, x)) == 0
    for n in (0, 1, 5, 64, 8)
        m, a = reshape(collect(1.:n^2), n, n), ones(n)
        @test w(similar(a), m, a) == m * a
        @test kept == A * x
        @test writer_bytes(w, similar(a), (m, a)) == 0
    end
    @test typeof(w.scratch.kernel.f) === typeof(prepare_into(k, zeros(3, 3), zeros(3)).scratch.kernel.f)
    @test w(zeros(Float32, 8), Float32.(A), Float32.(x)) == Float32.(A) * Float32.(x)
    store = [x; zeros(8)]
    @test w(view(store, 9:16), A, view(store, 1:8)) == kept
    @test_throws ArgumentError w(view(store, 2:9), A, view(store, 1:8))
    @test_throws ArgumentError w(x, A, x)
    bound = prepare_into(product, x; bound = (; A))
    @test bound(similar(x), x) == kept
    @test_throws ArgumentError bound(view(A, :, 1), x)
    ints = fill(71, 2)
    @test_throws InexactError w(ints, [1.0 0; 0 0.5], [1., 1.])
    @test ints == [71, 71]
    @test_throws DimensionMismatch w(ints, A, x)
    @test ints == [71, 71]
    @test !(@atomic w.scratch.busy)
    @test w(ints, [1.0 0; 0 2.0], [2., 3.]) === ints
    @test ints == [2, 6]

    dst = RetainsSource(zeros(8), Ref(Float64[]))
    @test w(dst, A, x) === dst
    w(zeros(8), A, -x)
    @test dst.source[] == A * x

    entry_kernel = prepare(product_entry; have = (:A, :x, :i), want = :y)
    entry! = prepare_into(entry_kernel, A, x, 1)
    @test entry!.scratch !== nothing
    @test entry!(zeros(1), A, x, 1) == kept[1:1]
    err = try
        entry!(zeros(1), A, x, length(x) + 1)
        nothing
    catch e
        e
    end
    @test err isa BoundsError
    @test entry!(zeros(1), A, -x, 1) == -kept[1:1]
    @test err.a == kept

    rows = [[1.0], [2.0, 3.0]]
    write_rows! = prepare_into(ragged_copy, rows)
    saved = write_rows!(similar(rows), rows)
    @test saved[1] === rows[2]
    write_rows!(similar(rows), [[4.0], [5.0]])
    @test saved == [[2.0, 3.0], [1.0]]

    # Enter the public writer while another call holds its workspace. The
    # second call can finish immediately and leaves the first result intact.
    w(zeros(8), A, x)
    @atomic w.scratch.busy = true
    try
        @test w(zeros(8), A, -x) == -(A * x)
        @test only(w.scratch.kernel.caches)[] == A * x
    finally
        @atomic w.scratch.busy = false
    end

    passed = prepare(product; have = (:A, :x), want = :y,
        passes = (ast -> Expr(:function, ast.args[1], Expr(:block, :(return x))),))
    @test prepare_into(passed, A, x)(similar(x), A, x) == x

    function change_outer(ast)
        local_body = Expr(:block, Expr(:(=), :z, Expr(:call, GlobalRef(Base, :reverse), :x)), :z)
        body = Expr(:block, :(z = x), Expr(:(=), :tmp, Expr(:let, Expr(:block), local_body)),
                    :(return z))
        Expr(:function, ast.args[1], body)
    end
    scoped = prepare(product; have = (:A, :x), want = :y, passes = (change_outer,))
    @test prepare_into(scoped, A, x)(similar(x), A, x) == scoped(A, x) == reverse(x)
end

end
