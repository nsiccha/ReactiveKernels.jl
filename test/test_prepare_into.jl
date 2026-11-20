module PrepareIntoTests

using ReactiveKernels
using Test

@kernel step(x, h, g) = begin
    y = x .+ (h / 2) .* g
end
@kernel centered(x) = begin
    y = x .- sum(x)
end
@kernel shifted_by_total(x, g) = begin
    total = sum(g)
    y = x .+ total
end
@kernel keyword_step(x; h = 1) = begin
    y = x .+ h
end
@kernel guarded(x, i) = begin
    y = if i == 0
        x
    else
        x .+ x[i]
    end
end
@kernel field_read(record) = begin
    y = record.x
end
const observed = [7., 0., 0.]
read_first(x) = x + observed[1]
@kernel read_before_writing(x) = begin
    y = read_first.(x)
end

struct Unread <: AbstractVector{Any} end
Base.size(::Unread) = (3,)
Base.getindex(::Unread, i::Int) = error("inactive input was read")

function warm_bytes(f, dst, x, h, g)
    f(dst, x, h, g)
    @allocated f(dst, x, h, g)
end

@testset "caller-owned prepared outputs" begin
    x = [1., 2., 4.]
    g = [3., 2., 1.]
    kernel = prepare(step)
    writer = prepare_into(kernel, x, 0.25, g)
    @test typeof(writer.program.f) === typeof(prepare_into(kernel, zeros(64), 0.5, ones(64)).program.f)
    for n in (0, 1, 5, 8, 16, 64)
        a, b, weights = zeros(n), zeros(n), cos.(1.0:n)
        expected = copy(a)
        for i in 1:9
            h = 0.125 * i
            @test writer(b, a, h, weights) === b
            @test a == expected
            expected = expected .+ (h / 2) .* weights
            @test b == expected
            a, b = b, a
        end
    end
    @test warm_bytes(writer, similar(x), x, 0.125, g) == 0
    # Preparing an existing kernel must preserve passes already applied to it.
    return_x(ast) = Expr(:function, ast.args[1], Expr(:block, :(return x)))
    passed = prepare(step; passes = (return_x,))
    @test prepare_into(passed)(similar(x), x, 0.25, g) == passed(x, 0.25, g) == x
    @test_throws ArgumentError writer(x, x, 0.125, g)
    @test_throws ArgumentError writer(g, x, 0.125, g)
    GC.@preserve x begin
        partial = unsafe_wrap(Vector{Float64}, pointer(x, 2), length(x) - 1)
        @test_throws ArgumentError writer(partial, x, 0.125, g)
    end
    @test writer(similar(x), x, 1, g) == kernel(x, 1, g)
    typed = prepare_into(step, x, 0.25, g)
    @test_throws ArgumentError typed(similar(x), x, 1, g)
    wrong = fill(-1., 2)
    @test_throws DimensionMismatch writer(wrong, x, 1., g)
    @test wrong == fill(-1., 2)

    center! = prepare_into(centered)
    store = collect(1.:8.)
    src, dst = view(store, 1:4), view(store, 5:8)
    @test center!(dst, src) === dst
    @test dst == src .- sum(src)
    before = copy(store)
    @test_throws ArgumentError center!(view(store, 3:6), src)
    GC.@preserve store begin
        offset = unsafe_wrap(Vector{Float64}, pointer(store, 3), 6)
        @test_throws ArgumentError center!(view(offset, 1:2:5), view(store, 1:2:5))
    end
    @test store == before
    # Check all conversions before storing any of them.
    integers = fill(9, 2)
    @test_throws InexactError center!(integers, [0.5, 2.])
    @test integers == [9, 9]
    @test center!(integers, [1., 2.]) === integers
    @test integers == [-2,-1]

    for bound! in (prepare_into(step; bound = (; g)),
                   prepare_into(prepare(step; bound = (; g))),
                   prepare_into(plan(step); bound = step.g => g))
        @test bound!(similar(x), x, 0.5) == kernel(x, 0.5, g)
        @test_throws ArgumentError bound!(g, x, 0.5)
        @test g == [3., 2., 1.]
    end
    hoisted! = prepare_into(shifted_by_total; bound = (; g))
    @test hoisted!(similar(x), x) == x .+ sum(g)
    @test_throws ArgumentError hoisted!(g, x)
    source = (; x)
    borrow! = prepare_into(field_read)
    @test borrow!(similar(x), source) == x
    @test_throws ArgumentError borrow!(x, source)
    nested = Any[nothing, g]
    nested[1] = nested
    @test_throws ArgumentError borrow!(g, (; x, nested))
    @test borrow!(similar(x), (; x, nested)) == x
    @test borrow!(similar(x), (; x, inactive = Unread())) == x
    expected = read_first.(x)
    @test prepare_into(read_before_writing)(observed, x) == expected

    kw! = prepare_into(keyword_step)
    @test kw!(similar(x), x) == x .+ 1
    @test kw!(similar(x), x; h = 2) == x .+ 2
    @test_throws ArgumentError kw!(x, x)
    branch! = prepare_into(guarded)
    @test branch!(Float64[], Float64[], 0) == Float64[]
    @test branch!(similar(x), x, 2) == x .+ x[2]
    @test_throws BoundsError branch!(similar(x), x, 4)

    graph = Graph()
    a, b, c = (value!(graph, s, Vector{Float64}) for s in (:a, :b, :c))
    calls = Ref(0)
    add!(graph, a => (b, c), v -> (calls[] += 1; (reverse(v), v)))
    other! = prepare_into(graph; have = (a,), want = (b,))
    calls[] = 0
    @test other!(similar(x), x) == reverse(x)
    @test calls[] == 1
    @test_throws ArgumentError prepare_into(graph; have = (a,), want = (b,c))
end

end
