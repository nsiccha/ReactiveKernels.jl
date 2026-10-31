module NestedPlatePartialEvaluationTests
using ReactiveKernels, Enzyme, Test

@kernel nested_terms(groups::Vector{Vector{Int}}, cuts::Vector{Vector{Float64}},
                     x::Float64) = begin
    terms = plate(groups, cuts) do group, cut
        rows = plate(group) do y
            y > 0 ? x * log(cut[1]) : x * log(cut[end]) + length(cut)
        end
        sum(rows)
    end
    return sum(terms)
end

@kernel nested_rows(groups::Vector{Vector{Int}}, cuts::Vector{Vector{Float64}},
                    x::Float64) = begin
    rows = plate(groups, cuts) do group, cut
        plate(group) do y
            x * log(cut[1]) + y
        end
    end
    return rows
end

function oracle(groups, cuts, x)
    total, slope = 0.0, 0.0
    for (group, cut) in zip(groups, cuts), y in group
        l = y > 0 ? log(cut[1]) : log(cut[end])
        total += x*l + (y > 0 ? 0 : length(cut))
        slope += l
    end
    total, slope
end

head_count(ex, head) = ex isa Expr ?
    (ex.head === head) + sum(arg -> head_count(arg, head), ex.args; init=0) : 0
scalar_caches(kernel) = [r.op.value for r in kernel.plan.recipes
    if r.op isa ReactiveKernels._BoundConstant && r.op.value isa Vector{Float64}]

@testset "nested shared bound scalars preserve lazy cells" begin
    for (groups, cuts) in (
        ([[1,0], [0], Int[]], [[1.5,2.5], [-1.0,4.0], Float64[]]),
        ([[0]], [[0.0,4.0]]),
        ([[0]], [[NaN,4.0]]),
        ([Int[]], [Float64[]]),
        (Vector{Vector{Int}}(), Vector{Vector{Float64}}()),
        ([[1], [0,1,0]], [[2.0], [3.0,5.0]]))
        saved = deepcopy((groups,cuts))
        kernel = prepare(nested_terms; bound=(;groups,cuts), type_stability=:error)
        for x in (0.3, 1.7)
            value, slope = oracle(groups,cuts,x)
            @test kernel(x) ≈ value
            @test only(only(Enzyme.autodiff(Enzyme.Reverse, kernel,
                Enzyme.Active, Enzyme.Active(x)))) ≈ slope
        end
        @test isequal((groups,cuts), saved)
    end
    # A failed speculative read stays under the original loop and branch.
    groups, cuts = [[1]], [[-1.0,4.0]]
    throwing = prepare(nested_terms; bound=(;groups,cuts))
    @test_throws DomainError throwing(0.3)
    groups, cuts = [[0]], [Float64[]]
    bounds = prepare(nested_terms; bound=(;groups,cuts))
    @test_throws BoundsError bounds(0.3)
    # Declining a lift also preserves selected nonfinite results.
    for low in (0.0, NaN)
        kernel = prepare(nested_terms; bound=(;groups=[[1]], cuts=[[low,4.0]]))
        @test isequal(kernel(0.3), 0.3*log(low))
        @test isequal(only(only(Enzyme.autodiff(Enzyme.Reverse, kernel,
            Enzyme.Active, Enzyme.Active(0.3)))), log(low))
    end
end

@testset "nested scalar caches retain runtime loops and rebind" begin
    original = plan(nested_terms)
    graph_values, graph_recipes = copy(original.graph.values), copy(original.graph.recipes)
    first_kernel = nothing
    for n in (2, 29)
        groups = [isodd(i) ? [1,0] : [0,1,0] for i in 1:n]
        cuts = [[1.0+i/10, 2.0+i/5] for i in 1:n]
        kernel = prepare(nested_terms; bound=(;groups,cuts), type_stability=:error)
        @test kernel(0.7) ≈ first(oracle(groups,cuts,0.7))
        ast = code_expr(kernel)
        @test head_count(ast, :for) == 2
        @test log.(first.(cuts)) in scalar_caches(kernel)
        @test log.(last.(cuts)) in scalar_caches(kernel)
        if first_kernel === nothing
            first_kernel = kernel
        else
            @test typeof(kernel.f) === typeof(first_kernel.f)
            @test head_count(ast, :if) == head_count(code_expr(first_kernel), :if)
        end
    end
    @test original.graph.values == graph_values
    @test original.graph.recipes == graph_recipes
    @test isempty(scalar_caches(prepare(nested_terms)))
    # Rebinding after a refused speculation must not retain stale cached data
    # or the previous binding's refusal decision.
    rebound = prepare(nested_terms; bound=(;groups=[[1,0]], cuts=[[2.0,5.0]]))
    @test rebound(0.7) ≈ first(oracle([[1,0]], [[2.0,5.0]], 0.7))
    @test [log(2.0)] in scalar_caches(rebound)
    @test [log(5.0)] in scalar_caches(rebound)
end

@testset "nested cached scalars do not share result storage" begin
    groups, cuts = [[1,2], Int[], [3]], [[2.0], Float64[], [4.0]]
    saved = deepcopy((groups,cuts))
    kernel = prepare(nested_rows; bound=(;groups,cuts), type_stability=:error)
    expected = [[0.3*log(2)+1, 0.3*log(2)+2], Float64[], [0.3*log(4)+3]]
    first_result = kernel(0.3)
    @test first_result ≈ expected
    first_result[1][1] = -100.0
    @test kernel(0.3) ≈ expected
    @test (groups,cuts) == saved
end
end
