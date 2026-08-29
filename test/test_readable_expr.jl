using ReactiveKernels
using Test

@testset "display-only readable kernel expressions" begin
    @kernel readable_example(x::Float64) = begin
        shifted::Float64 = x + 1
        scaled::Float64 = exp(shifted)
        result::Float64 = 2 * scaled + sin(x)
        return result
    end

    kernel = prepare(readable_example)
    raw = code_expr(kernel)
    raw_before = deepcopy(raw)
    shown = ReactiveKernels._readable_expr(raw, kernel.plan)
    source = sprint(Base.show_unquoted, shown; context = :limit => false)

    @test raw == raw_before == code_expr(kernel)
    @test occursin("__ops__[1]", string(raw))
    @test !occursin("__ops__", source)
    @test occursin("shifted = x + 1", source)
    @test occursin("scaled = exp(shifted)", source)
    @test occursin("sin(x)", source)
    @test kernel(0.25) ≈ 2exp(1.25) + sin(0.25)

    source_recipes = filter(
        recipe -> recipe.op isa ReactiveKernels._KernelSourceOp,
        kernel.plan.recipes,
    )
    @test !isempty(source_recipes)
    @test all(recipe -> !(recipe.source isa ReactiveKernels._NoKernelSource),
              source_recipes)

    @testset "named low-level operations replace positional slots" begin
        graph = Graph()
        x = value!(graph, :x, Float64)
        y = value!(graph, :y, Float64)
        total = value!(graph, :total, Float64)
        add!(graph, (x, y) => total, +)
        selected = plan(graph; have = (x, y), want = total)
        named = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(code_expr(selected), selected);
            context = :limit => false,
        )
        @test !occursin("__ops__", named)
        @test occursin("total = x + y", named)
    end

    @testset "colliding authored names stay explicit" begin
        @kernel collision_example(__ops__::Float64) = begin
            result::Float64 = 2 * __ops__ + 1
            return result
        end
        collision_kernel = prepare(collision_example)
        collision_source = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(
                code_expr(collision_kernel), collision_kernel.plan,
            );
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", collision_source)
        @test occursin("let __ops__ =", collision_source)
        @test occursin("2__ops__ + 1", replace(collision_source, " * " => ""))
        @test collision_kernel(3.0) == 7.0
    end

    @testset "cache plumbing is omitted from the readable copy" begin
        nonallocating_ast = ReactiveKernels._nonallocating_ast(raw)
        nonallocating_source = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(nonallocating_ast, kernel.plan);
            context = :limit => false,
        )
        @test !occursin("__ops__", nonallocating_source)
        @test !occursin("__caches__", nonallocating_source)
        @test !occursin("__cache_apply__", nonallocating_source)
        @test occursin("shifted = x + 1", nonallocating_source)
    end

    @testset "legacy source wrappers retain their constructor contract" begin
        legacy = ReactiveKernels._KernelSourceOp(
            Val(:legacy), Val(:fused), identity,
        )
        @test legacy(2) == 2
        @test fieldcount(typeof(legacy)) == 2
    end
end
