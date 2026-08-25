module AuthoringScopeFixture
using ..ReactiveKernels

const GLOBAL_OFFSET = 4.0
helper(x) = x + GLOBAL_OFFSET
withkw(tagged; property = 0.0) = tagged.shadow

function make_spec(local_offset)
    @kernel begin
        x::Float64
        shadow::Float64
        label::Float64
        property::Float64
        y::Float64 = helper(x) + local_offset +
            sum(shadow for shadow in (1.0, 2.0)) +
            sum(map(label -> label + 1.0, (1.0, 2.0)))
        tagged::NamedTuple{(:shadow,),Tuple{Float64}} = (; shadow = y)
        selected::Float64 = tagged.shadow
        kw::Float64 = withkw(tagged; property = 99.0)
        return (selected, kw)
    end
end
end

module AuthoringNameCollisionFixture
using ..ReactiveKernels

const Dict = :shadowed
const Symbol = :shadowed

make_module_spec() = @kernel begin
    x::Int
    y::Int = x + 1
    return y
end

function make_local_spec(Dict, Symbol)
    @kernel begin
        x::Int
        y::Int = x + 2
        return y
    end
end
end

module AuthoringForwardFixture
using ..ReactiveKernels

const x = 40

qualified_global() = @kernel begin
    x::Int
    y::Int = AuthoringForwardFixture.x + 2
    return y
end
end

@testset "Declarative kernel authoring" begin
    @testset "typed ports, zero-execution construction, and metadata" begin
        calls = Ref(0)
        pair(x) = (calls[] += 1; (x + 1, x + 2))
        direct(x) = (calls[] += 1; x)
        finish(a, b) = (calls[] += 1; a + b)

        spec = @kernel begin
            x::Float64
            @recipe (cost = 1.2, cse_key = :pair) (a::Float64, b::Float64) = pair(x)
            @recipe (cost = 1.0) a = direct(x)
            out::Float64 = finish(a, b)
            return out
        end

        @test calls[] == 0
        @test spec isa KernelSpec
        @test keys(spec) == (:x, :a, :b, :out)
        @test hasproperty(spec, :graph)
        @test hasproperty(spec, :x)
        @test inputs(spec) == (spec[:x],)
        @test outputs(spec) == (spec[:out],)
        @test ReactiveKernels.valtype(spec[:x]) === Float64
        @test kernel_graph(spec).recipes[1].cost == 1.2
        @test kernel_graph(spec).recipes[1].cse_key === :pair
        @test kernel_graph(spec).recipes[1].op === pair
        @test length(kernel_graph(spec).recipes[1].outputs) == 2
        @test length(kernel_graph(spec).producers[spec[:a].id]) == 2

        planned = plan(spec)
        @test [r.id for r in planned.recipes] == [1, 3]
        @test prepare(spec)(2.0) == 7.0
        @test calls[] == 2
        @test ReactiveState(spec; materialize = :a).graph === kernel_graph(spec)
        @test visualize(spec).subject === kernel_graph(spec)
        @test occursin("digraph ReactiveKernels", dot_source(spec))

        only_a = prepare(spec; want = :a)
        @test only_a(9.0) == 9.0
        @test inputs(only_a) == (spec[:x],)
        @test outputs(only_a) == (spec[:a],)
    end

    @testset "caller scope and dependency hygiene" begin
        scoped = AuthoringScopeFixture.make_spec(3.0)
        first_recipe = kernel_graph(scoped).recipes[1]
        @test Tuple(v.name for v in first_recipe.inputs) == (:x,)
        @test Tuple(v.name for v in kernel_graph(scoped).recipes[2].inputs) == (:y,)
        @test Tuple(v.name for v in kernel_graph(scoped).recipes[3].inputs) == (:tagged,)
        @test Tuple(v.name for v in kernel_graph(scoped).recipes[4].inputs) == (:tagged,)
        @test prepare(scoped)(2.0, 10.0, 20.0, 30.0) == (17.0, 17.0)

        reserved = @kernel begin
            __ops__::Int
            x::Int
            __caches__::Int = __ops__ + x
            return __caches__
        end
        @test prepare(reserved)(2, 3) == 5

        field_collision = @kernel begin
            graph::Int
            out::Int = graph + 1
            return out
        end
        @test field_collision.graph === kernel_graph(field_collision)
        @test field_collision[:graph] isa Value{Int}
        @test prepare(field_collision)(2) == 3

        @test prepare(AuthoringNameCollisionFixture.make_module_spec())(2) == 3
        @test prepare(AuthoringNameCollisionFixture.make_local_spec(1, 2))(2) == 4

        forward_input = @kernel begin
            y::Int = x + 1
            x::Int
            return y
        end
        @test prepare(forward_input)(2) == 3

        forward_intermediate = @kernel begin
            z::Int = y + 1
            y::Int = x + 1
            x::Int
            return z
        end
        @test Tuple(v.name for v in kernel_graph(forward_intermediate).recipes[1].inputs) ==
              (:y,)
        @test prepare(forward_intermediate)(2) == 4

        forward_alternative = @kernel begin
            @recipe (cost = 0) y = x + 1
            @recipe (cost = 2) y::Int = x + 2
            x::Int
            return y
        end
        @test length(ReactiveKernels.producers_of(
            kernel_graph(forward_alternative), forward_alternative.y.id,
        )) == 2
        @test prepare(forward_alternative)(2) == 3

        qualified = AuthoringForwardFixture.qualified_global()
        @test isempty(only(kernel_graph(qualified).recipes).inputs)
        @test prepare(qualified)(2) == 42

        shadowed = @kernel begin
            source::Int
            @recipe (effectful = true) a::Int = source + 100
            sequential::Int = begin
                a = 4
                a + 1
            end
            branched::Int = begin
                if source > 0
                    a = 6
                else
                    a = 7
                end
                a
            end
            local_function::Int = begin
                a(x) = x + 8
                a(1)
            end
            declared::Int = begin
                local a = 10
                a + 1
            end
            return (sequential, branched, local_function, declared)
        end
        shadow_recipes = kernel_graph(shadowed).recipes[2:end]
        @test isempty(shadow_recipes[1].inputs)
        @test Tuple(v.name for v in shadow_recipes[2].inputs) == (:source,)
        @test isempty(shadow_recipes[3].inputs)
        @test isempty(shadow_recipes[4].inputs)
        @test prepare(shadowed)(1) == (5, 6, 9, 11)
    end

    @testset "pure named-port composition and explicit boundaries" begin
        base = @kernel begin
            x::Float64
            y::Float64 = x + 1
            return y
        end
        fragment = @kernel begin
            y::Float64
            diagnostic::Float64 = 2y
            return diagnostic
        end

        base_version = kernel_graph(base).version
        fragment_version = kernel_graph(fragment).version
        combined = merge(base, fragment)
        @test combined !== base
        @test kernel_graph(combined) !== kernel_graph(base)
        @test combined[:x] != base[:x]
        @test combined[:y] != fragment[:y]
        @test kernel_graph(base).version == base_version
        @test kernel_graph(fragment).version == fragment_version
        @test Tuple(v.name for v in inputs(combined)) == (:x,)
        @test Tuple(v.name for v in outputs(combined)) == (:y,)
        @test prepare(combined)(2.0) == 3.0
        @test prepare(combined; want = (:y, :diagnostic))(2.0) == (3.0, 6.0)

        replaced = merge(base, fragment; boundary = :fragment)
        @test Tuple(v.name for v in inputs(replaced)) == (:y,)
        @test Tuple(v.name for v in outputs(replaced)) == (:diagnostic,)
        @test prepare(replaced)(4.0) == 8.0
        @test_throws ArgumentError merge(base, fragment; boundary = :ambiguous)

        fragment_reordered = @kernel begin
            spare::Int
            y::Float64
            diagnostic::Float64 = 2y
            return diagnostic
        end
        combined_reordered = merge(base, fragment_reordered)
        @test prepare(combined_reordered; want = :diagnostic)(2.0) == 6.0

        mismatch_a = @kernel begin
            y::Int
            extra::Float64
        end
        mismatch_b = @kernel begin
            extra::Float64
            y::Int
        end
        message_a = try
            merge(base, mismatch_a)
            ""
        catch err
            sprint(showerror, err)
        end
        message_b = try
            merge(base, mismatch_b)
            ""
        catch err
            sprint(showerror, err)
        end
        @test message_a == message_b
        @test occursin("port :y", message_a)
        @test occursin("Float64", message_a)
        @test occursin("Int64", message_a)

        cse_base = @kernel begin
            x::Int
            @recipe (cse_key = :identity) a::Int = identity(x)
            @recipe (cse_key = :identity) b::Int = identity(x)
            return (a, b)
        end
        cse_fragment = @kernel begin
            b::Int
            doubled::Int = 2b
            return doubled
        end
        cse_combined = merge(cse_base, cse_fragment)
        @test canon_id(kernel_graph(cse_combined), cse_combined[:a].id) ==
              canon_id(kernel_graph(cse_combined), cse_combined[:b].id)
        @test prepare(cse_combined; want = :a)(3) == 3
        @test prepare(cse_combined; want = :b)(3) == 3
        @test prepare(cse_combined; want = :doubled)(3) == 6

        cheap_base = @kernel begin
            x::Int
            @recipe (cost = 0) y::Int = x + 1
            return y
        end
        alias_fragment = @kernel begin
            x::Int
            @recipe (cost = 10, cse_key = :expensive) a::Int = x + 10
            @recipe (cost = 10, cse_key = :expensive) y::Int = x + 10
            return y
        end
        alias_merged = merge(cheap_base, alias_fragment)
        alias_graph = kernel_graph(alias_merged)
        @test sort(ReactiveKernels.producers_of(
            alias_graph, alias_merged.y.id,
        )) == [1, 2]
        @test plan(alias_merged).cost == 0
        @test prepare(alias_merged)(2) == 3

        alias_reversed = merge(alias_fragment, cheap_base)
        reversed_graph = kernel_graph(alias_reversed)
        @test sort(ReactiveKernels.producers_of(
            reversed_graph, alias_reversed.y.id,
        )) == [1, 2]
        @test plan(alias_reversed; have = :x, want = :y).cost == 0
        @test prepare(alias_reversed; have = :x, want = :y)(2) == 3
    end

    @testset "low-level equivalence and unchanged hot path" begin
        declarative = @kernel begin
            x::Float64
            a::Float64 = x + 1
            b::Float64 = 2a
            return b
        end

        low_graph = Graph()
        low_x = value!(low_graph, :x, Float64)
        low_a = value!(low_graph, :a, Float64)
        low_b = value!(low_graph, :b, Float64)
        add!(low_graph, low_x => low_a, x -> x + 1)
        add!(low_graph, low_a => low_b, a -> 2a)
        low_plan = plan(low_graph; have = (low_x,), want = (low_b,))
        authored_plan = plan(declarative)

        recipe_shape(r) = (
            Tuple(v.name for v in r.inputs),
            Tuple(v.name for v in r.outputs),
            r.cost,
            r.cse_key,
            r.effectful,
        )
        @test recipe_shape.(authored_plan.recipes) == recipe_shape.(low_plan.recipes)
        @test code_expr(authored_plan) == code_expr(low_plan)

        authored = prepare(authored_plan)
        low = prepare(low_plan)
        @test Tuple(v.name for v in inputs(authored)) == (:x,)
        @test Tuple(v.name for v in outputs(authored)) == (:b,)
        @test authored(2.0) == low(2.0) == 6.0

        allocated(k, x) = (k(x); @allocated k(x))
        allocated(authored, 2.0)
        allocated(low, 2.0)
        @test allocated(authored, 2.0) == 0
        @test allocated(low, 2.0) == 0
    end

    @testset "informative declaration errors" begin
        @test_throws ArgumentError @kernel begin
            x::Int
            x::Float64
        end
        untyped = quote
            @kernel begin
                x::Float64
                y = x + 1
            end
        end
        error = try
            macroexpand(@__MODULE__, untyped)
            nothing
        catch err
            err
        end
        @test error isa ArgumentError
        @test occursin("port :y", sprint(showerror, error))
        @test occursin("type annotation", sprint(showerror, error))
        @test_throws KeyError (@kernel begin x::Int end)[:missing]
    end
end
