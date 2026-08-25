using ReactiveKernels
using MutatingFunctions
using Test

function kernel_allocations(k, x)
    k(x)
    @allocated k(x)
end

function nonallocating_op_call_indices(ast)
    indices = Int[]
    function visit(node)
        node isa Expr || return
        if node.head === :ref && length(node.args) == 2 &&
           node.args[1] === :__ops__ && node.args[2] isa Int
            push!(indices, node.args[2])
        end
        foreach(visit, node.args)
    end
    visit(ast)
    indices
end

@testset "MutatingFunctions-backed non-allocating preparation" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) !== nothing
    @testset "registered operations reuse typed per-recipe caches" begin
        g = Graph()
        x = value!(g, :x, Vector{Float64})
        copied = value!(g, :copied, Vector{Float64})
        reversed = value!(g, :reversed, Vector{Float64})
        add!(g, x => copied, copy)
        add!(g, copied => reversed, reverse)

        p = plan(g; have = (x,), want = (reversed,))
        ordinary = prepare(p)
        k = prepare_nonallocating(p)
        input = [1.0, 2.0, 3.0]

        @test k isa NonAllocatingKernel
        @test inputs(k) == inputs(p)
        @test outputs(k) == outputs(p)
        @test ordinary(input) == k(input) == [3.0, 2.0, 1.0]
        @test @inferred(k(input)) == [3.0, 2.0, 1.0]

        first_result = k(input)
        @test first_result !== input
        @test k.caches[1][] !== input
        @test k.caches[2][] !== input
        second_result = k([4.0, 5.0])
        @test first_result === second_result
        @test second_result == [5.0, 4.0]
        allocated = kernel_allocations(k, input)
        println("NONALLOCATING_ALLOC_BYTES\tcopy_reverse\t", allocated)
        @test allocated == 0

        ast = code_expr(k)
        @test occursin("__caches__", string(ast))
        @test occursin("__cache_apply__", string(ast))
        @test nonallocating_op_call_indices(ast) == [1, 2]
        @test [r.id for r in k.plan.recipes] == [r.id for r in p.recipes]
    end

    @testset "no-recipe plans return the caller input directly" begin
        g = Graph()
        x = value!(g, :x, Vector{Float64})
        p = plan(g; have = (x,), want = (x,))
        k = prepare_nonallocating(p)
        input = [1.0, 2.0]

        @test isempty(k.caches)
        @test k(input) === input
        allocated = kernel_allocations(k, input)
        println("NONALLOCATING_ALLOC_BYTES\tempty_plan\t", allocated)
        @test allocated == 0
    end

    @testset "passes run before the cache rewrite" begin
        g = Graph()
        x = value!(g, :x, Vector{Int})
        y = value!(g, :y, Vector{Int})
        add!(g, x => y, copy)
        p = plan(g; have = (x,), want = (y,))
        ran = Ref(false)
        inspect_standard_lowering(ast) = begin
            ran[] = !occursin("__caches__", string(ast))
            ast
        end
        k = prepare_nonallocating(p; passes = (inspect_standard_lowering,))
        @test ran[]
        @test k([1, 2]) == [1, 2]
        @test occursin("__caches__", string(code_expr(k)))
    end

    @testset "hidden arguments cannot collide with graph value names" begin
        g = Graph()
        x = value!(g, :__caches__, Vector{Int})
        y = value!(g, :__cache_apply__, Vector{Int})
        add!(g, x => y, copy)
        k = prepare_nonallocating(g; have = (x,), want = (y,))
        @test k([1, 2, 3]) == [1, 2, 3]
    end

    @testset "multi-output recipes fail at preparation" begin
        g = Graph()
        x = value!(g, :x, Float64)
        y = value!(g, :y, Float64)
        z = value!(g, :z, Float64)
        add!(g; inputs = x, outputs = (y, z), op = x -> (x + 1, x + 2))
        p = plan(g; have = (x,), want = (y, z))
        err = try
            prepare_nonallocating(p)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("single-output recipes", sprint(showerror, err))
        @test occursin("recipe 1", sprint(showerror, err))
    end
end
