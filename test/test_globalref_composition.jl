module GlobalRefCompositionTests
using ReactiveKernels, Test

@kernel pair(x::Float64, y::Float64) = begin
    total::Float64 = x + y
    squared::Float64 = abs2(total)
    alias::Float64 = total
    return total, squared, alias
end
const pair_alias = pair

@kernel recurrence(xs, gain) = begin
    updates = scan(xs, Ref(gain); init=0.0) do carry, x, g
        next = carry + x * g
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
end
