using ReactiveKernels
using MutatingFunctions
using Test

function kernel_allocations(k, x)
    k(x)
    @allocated k(x)
end

# Fused captured sources resolve their free symbols in the authoring module.
# This module deliberately shadows `vcat`; a name-based Base.vcat rewrite would
# silently change semantics, so parity here pins exact-binding resolution.
module ShadowedVcatFixture
using ReactiveKernels
using MutatingFunctions
const vcat = (args...) -> Base.vcat(Base.reverse(args)...)
const SPEC = @kernel shadowed(a::Vector{Float64}, b::Vector{Float64}) = begin
    stacked = vcat(a, b)
    total::Float64 = sum(stacked)
end
function check_parity()
    ordinary = prepare(SPEC; have = (:a, :b), want = :total)
    k = prepare_nonallocating(SPEC; have = (:a, :b), want = :total)
    a, b = [1.0, 2.0], [10.0, 20.0, 30.0]
    ordinary(a, b) == k(a, b) == sum(Base.vcat(b, a))
end
end # module ShadowedVcatFixture

# A captured source reading a non-const global cannot be decomposed soundly
# (the binding may change between preparation and call); the whole-recipe
# fallback keeps the original closure semantics.
module NonConstGlobalFixture
using ReactiveKernels
using MutatingFunctions
scale = 2.0
const SPEC = @kernel scaled(x::Vector{Float64}) = begin
    grown = scale .* x
    total::Float64 = sum(grown)
end
function check_parity()
    ordinary = prepare(SPEC; have = (:x,), want = :total)
    k = prepare_nonallocating(SPEC; have = (:x,), want = :total)
    x = [1.0, 2.0, 3.0]
    first = ordinary(x) == k(x) == 2.0 * sum(x)
    global scale = 3.0
    second = ordinary(x) == k(x) == 3.0 * sum(x)
    global scale = 2.0
    first && second
end
end # module NonConstGlobalFixture

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

    @testset "fused captured sources decompose into destination-passing steps" begin
        # The four MNIST op classes in one authored kernel: lazy reshape/view
        # coefficient unpacking, range-slice extraction, matmul + broadcast,
        # and vcat over a zeros-constructed row.
        spec = @kernel mnist_shaped(u::Vector{Float64}, X::Matrix{Float64}) = begin
            d::Int = size(X, 2)
            r::Int = 3
            W::Matrix{Float64} = reshape(view(u, 1:(r * d)), r, d)
            b::Vector{Float64} = u[(r * d + 1):length(u)]
            scores = W * transpose(X) .+ b
            padded = vcat(zeros(1, size(scores, 2)), scores)
            total::Float64 = sum(padded)
        end
        steady(k, args...) = (k(args...); k(args...); @allocated k(args...))
        data(n, d) = (rand(3 * d + 3), rand(n, d))

        ordinary = prepare(spec; have = (:u, :X), want = :total)
        k = prepare_nonallocating(spec; have = (:u, :X), want = :total)
        u, X = data(8, 5)
        @test k(u, X) == ordinary(u, X)

        # Every fused captured source decomposed: no opaque fused closure is
        # left in the step table, and the destination steps are present.
        @test !any(op -> op isa ReactiveKernels._KernelSourceOp, k.ops)
        @test any(op -> op isa ReactiveKernels._MaterializeStep, k.ops)
        @test any(op -> op isa ReactiveKernels._ConcatenateStep, k.ops)
        @test any(op -> op isa ReactiveKernels._FillConstructorStep, k.ops)
        @test any(op -> op isa ReactiveKernels._MatMulStep, k.ops)

        small = steady(k, data(8, 5)...)
        large = steady(k, data(200, 5)...)
        println("NONALLOCATING_ALLOC_BYTES\tmnist_shaped_small\t", small)
        println("NONALLOCATING_ALLOC_BYTES\tmnist_shaped_large\t", large)
        @test small == large            # zero data-sized reallocation
        @test small <= 512              # fixed near-zero per-call residue

        # Cache reseeding on batch-size changes preserves exact parity.
        for n in (8, 32, 8, 200)
            un, Xn = data(n, 5)
            @test k(un, Xn) == ordinary(un, Xn)
        end
    end

    @testset "authored plates keep exact pointwise semantics" begin
        # An untyped plate body materializes through an eltype-`Any` cache in
        # both execution paths, so only value semantics are pinned here; the
        # zero-allocation plate coverage for typed (Float64-endpoint) bodies
        # is exercised on the real MNIST graph in test_nonallocating_mnist.jl.
        spec = @kernel plated(x::Vector{Float64}) = begin
            terms = plate(x) do value
                value * value
            end
            total::Float64 = sum(terms)
        end
        k = prepare_nonallocating(spec; have = (:x,), want = :total)
        ordinary = prepare(spec; have = (:x,), want = :total)
        pointwise = prepare(spec; have = (:x,), want = :terms)
        x = rand(64)
        # The non-allocating kernel materializes the plate then sums it; the
        # ordinary kernel fuses the reduction into an accumulator loop. The
        # materialized total is bit-exactly `sum` of the plain pointwise
        # values; against the fused accumulation only association differs.
        @test k(x) == sum(pointwise(x))
        @test isapprox(k(x), ordinary(x); rtol = 1e-12)
        for n in (64, 16, 256)
            xn = rand(n)
            @test k(xn) == sum(pointwise(xn))
        end
    end

    @testset "a PreparedKernel reuses its selected plan" begin
        g = Graph()
        x = value!(g, :x, Vector{Float64})
        copied = value!(g, :copied, Vector{Float64})
        add!(g, x => copied, copy)
        prepared = prepare(g; have = (x,), want = (copied,))
        k = prepare_nonallocating(prepared)
        @test k isa NonAllocatingKernel
        @test k([1.0, 2.0]) == [1.0, 2.0]
        @test [r.id for r in k.plan.recipes] == [r.id for r in prepared.plan.recipes]
    end

    @testset "decomposition resolves the authoring module's own bindings" begin
        # A shadowed `vcat`: the decomposed step must call the module's
        # binding, never a name-based Base.vcat guess.
        parity = ShadowedVcatFixture.check_parity()
        @test parity
        # A non-const global in the captured source keeps the whole-recipe
        # fallback and stays correct.
        @test NonConstGlobalFixture.check_parity()
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
