module CyclicPlannerADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel cyclic_scalar(x::Float64) = begin
    @recipe (cost = 1) a::Float64 = 2x
    @recipe (cost = 0) (a, b) = (a, a^2)
    @recipe (cost = 2) b::Float64 = 4x^2
    return b
end

@kernel cyclic_scan(q::Vector{Float64}, xs::Vector{Float64}) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        next = carry + q[1]*x
        (next, next)
    end
    @recipe (cost = 0) (trajectory, total) = (trajectory, sum(abs2, trajectory))
    return total
end

_expr_nodes(ex) = ex isa Expr ? 1 + sum(_expr_nodes, ex.args; init = 0) : 0
_loops(ex) = ex isa Expr ? Int(ex.head === :for) + sum(_loops, ex.args; init = 0) : 0

@testset "grounded cyclic plans retain primal and ordinary reverse semantics" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    scalar = prepare(cyclic_scalar)
    @test (scalar.plan.cost, length(scalar.plan.recipes)) == (1.0, 2)
    scalar_ad = prepare_ad(scalar, backend, 3.0; active = :x)
    for x in (-2.0, 0.0, 3.0)
        @test scalar(x) == 4x^2
        @test ad_gradient(scalar_ad, x) == 8x
    end

    for bound in (false, true)
        sizes = Int[]
        loops = Int[]
        for n in (0, 1, 7, 31)
            xs = sin.(1:n)
            saved = copy(xs)
            k = bound ? prepare(cyclic_scan; bound = (; xs)) : prepare(cyclic_scan)
            # The self-output recipe is grounded by the real scan, not a
            # per-element expansion or an alternate mathematical objective.
            @test count(r -> r.op isa ReactiveKernels._AuthoredScanOp, k.plan.recipes) == 1
            push!(sizes, _expr_nodes(code_expr(k)))
            push!(loops, _loops(code_expr(k)))
            args = bound ? ([0.7],) : ([0.7], xs)
            ad = prepare_ad(k, backend, args...; active = :q)
            coefficient = sum(abs2, cumsum(xs))
            for scale in (-0.4, 0.0, 0.7)
                q = [scale]
                args = bound ? (q,) : (q, xs)
                @test k(args...) ≈ scale^2 * coefficient
                @test ad_gradient(ad, args...) ≈ [2scale * coefficient]
                @test q == [scale]
                @test xs == saved
            end
        end
        @test all(==(first(sizes)), sizes)
        @test all(==(first(loops)), loops)
        @test first(loops) >= 1
    end
end
end
