module GlobalRefCompositionTests
using ReactiveKernels, Test

@kernel pair(x::Float64, y::Float64) = begin
    total::Float64 = x + y
    squared::Float64 = abs2(total)
    alias::Float64 = total
    return total, squared, alias
end
const pair_alias = pair

@kernel plus(x, y) = begin
    total = x + y
    return total
end

@kernel recurrence(xs, gain) = begin
    updates = scan(xs; init=0.0) do carry, x
        next = carry + x * gain
        (next, next)
    end
    return updates
end

function resolved(def)
    Core.eval(@__MODULE__, Expr(:macrocall, GlobalRef(ReactiveKernels, Symbol("@kernel")),
        LineNumberNode(1), def))
end

@testset "resolved kernel bindings splice the authored graph" begin
    for name in (:pair, :pair_alias)
        head = GlobalRef(@__MODULE__, name)
        spec = resolved(:(parent(x::Float64) = begin
            (a::Float64, b::Float64, c::Float64) = $head(x, x)
            return a, b, c
        end))
        @test length(spec.graph.recipes) == 2
        @test all(r -> any(c -> r.op === c.op, pair.graph.recipes), spec.graph.recipes)
        @test prepare(spec)(2.0) == (4.0, 16.0, 4.0)
        @test prepare(spec; want=:a)(3.0) == 6.0
        @test ReactiveKernels.canon_id(spec.graph, spec.a.id) ==
              ReactiveKernels.canon_id(spec.graph, spec.c.id)
    end
    @test ReactiveKernels._kernel_resolve_binding(Main, GlobalRef(@__MODULE__, :pair)) === pair
    @test ReactiveKernels._kernel_resolve_binding(Main, GlobalRef(@__MODULE__, :missing_binding)) === nothing

    # Untyped child ports compose with typed caller ports, preserving the
    # declarations through ordinary identity/conversion recipes.
    head = GlobalRef(@__MODULE__, :recurrence)
    for form in (head, :recurrence)
        spec = resolved(:(parent(xs::Vector{Float64}, g::Float64) = begin
            result::Vector{Float64} = $form(xs, g)
            return result
        end))
        @test count(r -> r.op isa ReactiveKernels._AuthoredScanOp, spec.graph.recipes) == 1
        @test ReactiveKernels.valtype(spec.result) === Vector{Float64}
        for n in (0, 1, 17)
            xs = sin.(1:n)
            @test prepare(spec)(xs, 0.3) ≈ cumsum(xs .* 0.3)
            @test prepare(spec; bound=(; xs))(0.3) ≈ cumsum(xs .* 0.3)
        end
    end
end

@testset "computed graph arguments and nested values splice transparently" begin
    for head in (:plus, GlobalRef(@__MODULE__, :plus),
                 :(GlobalRefCompositionTests.plus))
        spec = resolved(:(parent(x::Float64) = begin
            first = $head(x + 1, 2.0)
            second = 3 * $head($head(x, x), first)
            return second
        end))
        named = resolved(:(named(x::Float64) = begin
            arg = x + 1
            constant = 2.0
            first = $head(arg, constant)
            twice = $head(x, x)
            combined = $head(twice, first)
            second = 3 * combined
            return second
        end))
        for x in (-0.4, 2.0)
            @test prepare(spec)(x) == prepare(named)(x) == 9 * x + 9
        end
        @test length(spec.graph.recipes) == length(named.graph.recipes)
        @test !occursin("plus(", sprint(show, code_expr(prepare(spec))))
    end

    # Lift a graph value in indexing/tuple expressions, and retain the child
    # scan for a computed input instead of preparing it as an opaque callable.
    spec = resolved(:(parent(xs, gain) = begin
        result = (recurrence(xs, gain + 1), plus(gain, 2))[1]
        return result
    end))
    @test count(r -> r.op isa ReactiveKernels._AuthoredScanOp,
                spec.graph.recipes) == 1
    function loop_count(ex)
        ex isa Expr || return 0
        (ex.head === :for) + sum(loop_count, ex.args; init=0)
    end
    counts = Int[]
    for n in (0, 3, 17)
        xs = sin.(1:n)
        p = prepare(spec; bound=(; xs))
        @test p(0.3) ≈ cumsum(xs .* 1.3)
        push!(counts, loop_count(code_expr(p)))
    end
    @test counts == [1, 1, 1]

    # Lazy arms and deferred lexical scopes must not be lifted to unconditional
    # caller recipes: their access and argument evaluation stay selected/local.
    @kernel guarded(x, values) = begin
        result = x > 0 ? plus(x, values[1]) : 0.0
        return result
    end
    @test prepare(guarded)(-1.0, Float64[]) == 0.0
    @test prepare(guarded)(2.0, [3.0]) == 5.0
    @kernel deferred(x) = begin
        result = map(i -> plus(i, x), 1:3)
        return result
    end
    @test prepare(deferred)(2.0) == [3.0, 4.0, 5.0]
end
end
