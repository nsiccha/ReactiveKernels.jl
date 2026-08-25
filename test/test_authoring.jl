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

module AuthoringComputedCalleeFixture
struct CallableSource
    calls::Base.RefValue{Int}
end

mutable struct Box
    value::Int
end

function Base.getproperty(source::CallableSource, name::Symbol)
    name === :op || return getfield(source, name)
    getfield(source, :calls)[] += 1
    identity
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

        explicit_values = prepare(spec; have = spec.x, want = spec.out)
        @test explicit_values(2.0) == 7.0
        forged_have_name = Value{Float64}(spec.x.id, :forged)
        forged_have_type = Value{Int}(spec.x.id, spec.x.name)
        forged_want_name = Value{Float64}(spec.out.id, :forged)
        forged_want_type = Value{Int}(spec.out.id, spec.out.name)
        @test_throws ArgumentError prepare(
            spec; have = forged_have_name, want = spec.out,
        )
        @test_throws ArgumentError prepare(
            spec; have = forged_have_type, want = spec.out,
        )
        @test_throws ArgumentError prepare(
            spec; have = spec.x, want = forged_want_name,
        )
        @test_throws ArgumentError prepare(
            spec; have = spec.x, want = forged_want_type,
        )

        computed_calls = Ref(0)
        makeop() = (computed_calls[] += 1; identity)
        computed_callee = @kernel begin
            x::Int
            y::Int = makeop()(x)
            return y
        end
        @test computed_calls[] == 0
        computed_kernel = prepare(computed_callee)
        @test computed_calls[] == 0
        @test computed_kernel(3) == 3
        @test computed_calls[] == 1

        property_calls = Ref(0)
        callable_source = AuthoringComputedCalleeFixture.CallableSource(property_calls)
        property_callee = @kernel begin
            x::Int
            y::Int = callable_source.op(x)
            return y
        end
        @test property_calls[] == 0
        property_kernel = prepare(property_callee)
        @test property_calls[] == 0
        @test property_kernel(4) == 4
        @test property_calls[] == 1
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
        @test :graph in propertynames(field_collision)
        @test count(name -> name === :graph, propertynames(field_collision)) == 1
        @test prepare(field_collision)(2) == 3

        storage_collision = @kernel begin
            ports::Int
            port_order::Int
            have_names::Int
            want_names::Int
            out::Int = ports + port_order + have_names + want_names
            return out
        end
        @test storage_collision.ports isa Dict
        @test storage_collision.port_order isa Vector{Symbol}
        @test storage_collision.have_names isa Vector{Symbol}
        @test storage_collision.want_names isa Vector{Symbol}
        for name in (:ports, :port_order, :have_names, :want_names)
            @test storage_collision[name] isa Value{Int}
            @test name in propertynames(storage_collision)
            @test count(==(name), propertynames(storage_collision)) == 1
        end
        @test prepare(storage_collision)(1, 2, 3, 4) == 10

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

        long_form_function = @kernel begin
            source::Int
            @recipe (effectful = true) a::Int = source + 100
            result::Int = begin
                function a(x)
                    x + 8
                end
                a(1)
            end
            return result
        end
        @test isempty(kernel_graph(long_form_function).recipes[2].inputs)
        @test prepare(long_form_function)(1) == 9

        short_default_capture = @kernel begin
            a::Int
            y::Int = begin
                f(; x = a) = x + 1
                f()
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(short_default_capture).recipes,
        ).inputs) == (:a,)
        @test prepare(short_default_capture)(2) == 3

        long_default_capture = @kernel begin
            a::Int
            y::Int = begin
                function f(x = a)
                    x + 2
                end
                f()
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(long_default_capture).recipes,
        ).inputs) == (:a,)
        @test prepare(long_default_capture)(2) == 4

        shadowed_default = @kernel begin
            source::Int
            @recipe (effectful = true) a::Int = source + 100
            y::Int = begin
                f(; x = begin
                    a = 4
                    a
                end) = x + 1
                f()
            end
            return y
        end
        @test isempty(kernel_graph(shadowed_default).recipes[2].inputs)
        @test prepare(shadowed_default)(1) == 5

        earlier_parameter_default = @kernel begin
            source::Int
            @recipe (effectful = true) x::Int = source + 100
            y::Int = begin
                f(x, y = x) = y
                f(4)
            end
            return y
        end
        @test isempty(kernel_graph(earlier_parameter_default).recipes[2].inputs)
        @test prepare(earlier_parameter_default)(1) == 4

        for_scope = @kernel begin
            i::Int
            y::Int = begin
                before = i
                for i in 1:2
                    nothing
                end
                before + i
            end
            return y
        end
        @test Tuple(v.name for v in only(kernel_graph(for_scope).recipes).inputs) == (:i,)
        @test prepare(for_scope)(10) == 20

        generator_scope = @kernel begin
            i::Int
            y::Int = sum(i for i in 1:2) + i
            return y
        end
        @test Tuple(v.name for v in only(kernel_graph(generator_scope).recipes).inputs) ==
              (:i,)
        @test prepare(generator_scope)(10) == 13

        filtered_generator = @kernel begin
            source::Int
            xs::Vector{Int}
            @recipe (effectful = true) x::Int = source + 100
            y::Vector{Int} = [x for x in xs if isodd(x)]
            return y
        end
        @test Tuple(v.name for v in kernel_graph(filtered_generator).recipes[2].inputs) ==
              (:xs,)
        @test prepare(filtered_generator)(1, [1, 2, 3]) == [1, 3]

        filtered_noncollision = @kernel begin
            xs::Vector{Int}
            threshold::Int
            y::Vector{Int} = [x for x in xs if x > threshold]
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(filtered_noncollision).recipes,
        ).inputs) == Set((:xs, :threshold))
        @test prepare(filtered_noncollision)([1, 2, 3], 1) == [2, 3]

        comma_outer_scope = @kernel begin
            x::Int
            xs::Vector{Int}
            y::Matrix{Int} = [x + y for x in xs, y in 1:x]
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(comma_outer_scope).recipes,
        ).inputs) == Set((:x, :xs))
        @test prepare(comma_outer_scope)(3, [2, 3]) == [3 4 5; 4 5 6]

        comma_filtered_noncollision = @kernel begin
            xs::Vector{Int}
            ys::Vector{Int}
            threshold::Int
            z::Vector{Int} = [
                x + y for x in xs, y in ys if x + y > threshold
            ]
            return z
        end
        @test Set(v.name for v in only(
            kernel_graph(comma_filtered_noncollision).recipes,
        ).inputs) == Set((:xs, :ys, :threshold))
        @test prepare(comma_filtered_noncollision)([1, 2], [2, 3], 3) == [4, 4, 5]

        comma_three_iterators = @kernel begin
            x::Int
            y::Int
            xs::Vector{Int}
            z::Vector{Int} = [
                x + y + z for x in xs, y in 1:x, z in 1:y if z < x
            ]
            return z
        end
        @test Set(v.name for v in only(
            kernel_graph(comma_three_iterators).recipes,
        ).inputs) == Set((:x, :y, :xs))
        @test prepare(comma_three_iterators)(3, 2, [2, 3]) ==
              [4, 5, 5, 6, 6, 7, 6, 7, 8]

        nested_filtered_generator = @kernel begin
            source::Int
            xs::Vector{Int}
            @recipe (effectful = true) x::Int = source + 100
            @recipe (effectful = true) y::Int = source + 200
            z::Vector{Int} = [
                x + y for x in xs if x > 0 for y in 1:x if y < x
            ]
            return z
        end
        @test Tuple(
            v.name for v in kernel_graph(nested_filtered_generator).recipes[3].inputs
        ) == (:xs,)
        @test prepare(nested_filtered_generator)(1, [2, 3]) == [3, 4, 5]

        while_scope = @kernel begin
            a::Int
            y::Int = begin
                before = a
                while false
                    a = 1
                end
                before + a
            end
            return y
        end
        @test Tuple(v.name for v in only(kernel_graph(while_scope).recipes).inputs) ==
              (:a,)
        @test prepare(while_scope)(10) == 20

        catch_only = @kernel begin
            source::Int
            @recipe (effectful = true) a::Int = source + 100
            caught::Int = try
                error("expected")
            catch a
                5
            end
            return caught
        end
        @test isempty(kernel_graph(catch_only).recipes[2].inputs)
        @test prepare(catch_only)(1) == 5

        catch_outer = @kernel begin
            a::Int
            y::Int = begin
                caught = try
                    error("expected")
                catch a
                    5
                end
                caught + a
            end
            return y
        end
        @test Tuple(v.name for v in only(kernel_graph(catch_outer).recipes).inputs) ==
              (:a,)
        @test prepare(catch_outer)(10) == 15

        no_read_assignment = @kernel begin
            source::Int
            @recipe (effectful = true) a::Int = source + 100
            y::Int = begin
                a = 4
                5
            end
            return y
        end
        @test isempty(kernel_graph(no_read_assignment).recipes[2].inputs)
        @test prepare(no_read_assignment)(1) == 5

        read_modify_assignment = @kernel begin
            a::Int
            y::Int = begin
                a = a + 1
                a
            end
            return y
        end
        @test Tuple(
            v.name for v in only(kernel_graph(read_modify_assignment).recipes).inputs
        ) == (:a,)
        @test prepare(read_modify_assignment)(10) == 11

        indexed_assignment = @kernel begin
            a::Vector{Int}
            i::Int
            y::Int = begin
                b = copy(a)
                b[i] = 9
                sum(b)
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(indexed_assignment).recipes,
        ).inputs) == (:a, :i)
        @test prepare(indexed_assignment)([1, 2], 2) == 10

        rhs_before_target = @kernel begin
            source::Int
            a::Vector{Int}
            @recipe (effectful = true) i::Int = source + 100
            y::Vector{Int} = begin
                b = copy(a)
                b[i] = (i = 1; 9)
                b
            end
            return y
        end
        @test Tuple(v.name for v in kernel_graph(rhs_before_target).recipes[2].inputs) ==
              (:a,)
        @test prepare(rhs_before_target)(1, [2, 3]) == [9, 3]

        rhs_before_typed_target = @kernel begin
            @recipe (effectful = true) T::DataType = Int
            y::Int = begin
                x::T = (T = Int; 2)
                x
            end
            return y
        end
        @test isempty(kernel_graph(rhs_before_typed_target).recipes[2].inputs)
        @test prepare(rhs_before_typed_target; have = ())() == 2

        typed_let = @kernel begin
            T::DataType
            y::Int = let x::T = 2
                x
            end
            return y
        end
        @test Tuple(v.name for v in only(kernel_graph(typed_let).recipes).inputs) ==
              (:T,)
        @test prepare(typed_let)(Int) == 2

        declared_typed_let = @kernel begin
            T::DataType
            y::Int = let x::T
                x = 2
                x
            end
            return y
        end
        @test Tuple(
            v.name for v in only(kernel_graph(declared_typed_let).recipes).inputs
        ) == (:T,)
        @test prepare(declared_typed_let)(Int) == 2

        typed_for = @kernel begin
            T::DataType
            xs::Vector{Int}
            y::Int = begin
                total = 0
                for x::T in xs
                    total += x
                end
                total
            end
            return y
        end
        @test Set(v.name for v in only(kernel_graph(typed_for).recipes).inputs) ==
              Set((:T, :xs))
        @test prepare(typed_for)(Int, [1, 2]) == 3

        typed_comprehension = @kernel begin
            T::DataType
            xs::Vector{Int}
            y::Vector{Int} = [x for x::T in xs]
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(typed_comprehension).recipes,
        ).inputs) == Set((:T, :xs))
        @test prepare(typed_comprehension)(Int, [1, 2]) == [1, 2]

        filtered_typed_comprehension = @kernel begin
            T::DataType
            xs::Vector{Int}
            y::Vector{Int} = [x for x::T in xs if isodd(x)]
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(filtered_typed_comprehension).recipes,
        ).inputs) == Set((:T, :xs))
        @test prepare(filtered_typed_comprehension)(Int, [1, 2, 3]) == [1, 3]

        target_receivers = @kernel begin
            a::Vector{Int}
            box::AuthoringComputedCalleeFixture.Box
            y::Int = begin
                copy(a)[1] = 9
                deepcopy(box).value = 8
                17
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(target_receivers).recipes,
        ).inputs) == (:a, :box)
        @test prepare(target_receivers)(
            [1, 2], AuthoringComputedCalleeFixture.Box(1),
        ) == 17

        property_assignment = @kernel begin
            box::AuthoringComputedCalleeFixture.Box
            field_value::Int
            value::Int
            y::Int = begin
                local_box = deepcopy(box)
                local_box.value = field_value
                local_box.value
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(property_assignment).recipes,
        ).inputs) == (:box, :field_value)
        @test prepare(
            property_assignment; have = (:box, :field_value),
        )(AuthoringComputedCalleeFixture.Box(1), 9) == 9

        named_tuple_assignment = @kernel begin
            a::Int
            x::Int
            y::NamedTuple{(:a,:b),Tuple{Int,Int}} = (a = x, b = a)
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(named_tuple_assignment).recipes,
        ).inputs) == Set((:a, :x))
        @test prepare(named_tuple_assignment)(3, 4) == (a = 4, b = 3)

        semicolon_named_tuple = @kernel begin
            a::Int
            x::Int
            y::NamedTuple{(:a,:b),Tuple{Int,Int}} = (; a = x, b = a)
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(semicolon_named_tuple).recipes,
        ).inputs) == Set((:a, :x))
        @test prepare(semicolon_named_tuple)(3, 4) == (a = 4, b = 3)

        destructuring_assignment = @kernel begin
            a::Int
            b::Int
            y::Int = begin
                (a, b) = (2, 3)
                a + b
            end
            return y
        end
        @test isempty(only(kernel_graph(destructuring_assignment).recipes).inputs)
        @test prepare(destructuring_assignment; have = ())() == 5

        mixed_assignment = @kernel begin
            source::Int
            @recipe (effectful = true) x::Int = source + 100
            y::Int = begin
                b = [0]
                (x, b[1]) = (2, 3)
                x + b[1]
            end
            return y
        end
        @test isempty(kernel_graph(mixed_assignment).recipes[2].inputs)
        @test prepare(mixed_assignment)(1) == 5

        typed_assignment = @kernel begin
            T::DataType
            y::Int = begin
                x::T = 2
                x
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(typed_assignment).recipes,
        ).inputs) == (:T,)
        @test prepare(typed_assignment)(Int) == 2

        initialized_typed_local = @kernel begin
            T::DataType
            y::Int = begin
                local x::T = 2
                x
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(initialized_typed_local).recipes,
        ).inputs) == (:T,)
        @test prepare(initialized_typed_local)(Int) == 2

        declared_typed_local = @kernel begin
            T::DataType
            y::Int = begin
                local x::T
                x = 2
                x
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(declared_typed_local).recipes,
        ).inputs) == (:T,)
        @test prepare(declared_typed_local)(Int) == 2

        one_branch_assignment = @kernel begin
            a::Int
            flag::Bool
            y::Int = begin
                if flag
                    a = 4
                end
                a + 1
            end
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(one_branch_assignment).recipes,
        ).inputs) == Set((:a, :flag))
        @test prepare(one_branch_assignment)(2, false) == 3
        @test prepare(one_branch_assignment)(2, true) == 5

        both_branch_assignment = @kernel begin
            a::Int
            flag::Bool
            y::Int = begin
                if flag
                    a = 4
                else
                    a = 5
                end
                a + 1
            end
            return y
        end
        @test Tuple(v.name for v in only(
            kernel_graph(both_branch_assignment).recipes,
        ).inputs) == (:flag,)
        @test prepare(both_branch_assignment; have = :flag)(true) == 5
        @test prepare(both_branch_assignment; have = :flag)(false) == 6

        read_before_branch_assignment = @kernel begin
            a::Int
            flag::Bool
            y::Int = begin
                before = a
                if flag
                    a = 4
                else
                    a = 5
                end
                before + a
            end
            return y
        end
        @test Set(v.name for v in only(
            kernel_graph(read_before_branch_assignment).recipes,
        ).inputs) == Set((:a, :flag))
        @test prepare(read_before_branch_assignment)(2, true) == 6
        @test prepare(read_before_branch_assignment)(2, false) == 7

        static_lambda = @kernel begin
            f::Function = (x::Base.Int -> x + 1)
            return f
        end
        static_function = prepare(static_lambda; have = ())()
        @test static_function(2) == 3

        lambda_binder_collision = @kernel begin
            source::Int
            @recipe (effectful = true) x::Int = source + 100
            f::Function = (x::Int -> x + 1)
            return f
        end
        @test isempty(kernel_graph(lambda_binder_collision).recipes[2].inputs)
        collision_function = prepare(lambda_binder_collision)(1)
        @test collision_function(2) == 3
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

        single_alias = @kernel begin
            x::Int
            @recipe (cost = 0) y::Int = x + 1
            @recipe (cost = 10, cse_key = :single_alias) a::Int = x + 10
            @recipe (cost = 10, cse_key = :single_alias) y = x + 10
            return y
        end
        single_alias_graph = kernel_graph(single_alias)
        @test sort(ReactiveKernels.producers_of(
            single_alias_graph, single_alias.y.id,
        )) == [1, 2]
        @test plan(single_alias).cost == 0
        @test prepare(single_alias)(2) == 3

        repeated_output = @kernel begin
            x::Int
            @recipe (cse_key = :repeated_output) y::Int = identity(x)
            @recipe (cse_key = :repeated_output) y = identity(x)
            return y
        end
        @test length(kernel_graph(repeated_output).recipes) == 1
        @test !haskey(kernel_graph(repeated_output).aliases, repeated_output.y.id)
        @test prepare(repeated_output)(3) == 3

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

        typed_base = @kernel begin
            x::Int
            @recipe (cse_key = :typed_cross_spec) a::Int = identity(x)
            return a
        end
        typed_fragment = @kernel begin
            x::Int
            @recipe (cse_key = :typed_cross_spec) b::String = string(x)
            return b
        end
        base_snapshot = (
            kernel_graph(typed_base).version,
            length(kernel_graph(typed_base).recipes),
            copy(kernel_graph(typed_base).producers),
            copy(kernel_graph(typed_base).aliases),
        )
        fragment_snapshot = (
            kernel_graph(typed_fragment).version,
            length(kernel_graph(typed_fragment).recipes),
            copy(kernel_graph(typed_fragment).producers),
            copy(kernel_graph(typed_fragment).aliases),
        )
        typed_messages = map(1:2) do _
            try
                merge(typed_base, typed_fragment)
                ""
            catch err
                @test err isa ArgumentError
                sprint(showerror, err)
            end
        end
        @test typed_messages[1] == typed_messages[2]
        @test occursin("structural CSE output type mismatch", typed_messages[1])
        @test occursin("position 1", typed_messages[1])
        @test base_snapshot == (
            kernel_graph(typed_base).version,
            length(kernel_graph(typed_base).recipes),
            kernel_graph(typed_base).producers,
            kernel_graph(typed_base).aliases,
        )
        @test fragment_snapshot == (
            kernel_graph(typed_fragment).version,
            length(kernel_graph(typed_fragment).recipes),
            kernel_graph(typed_fragment).producers,
            kernel_graph(typed_fragment).aliases,
        )
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

        typed_cse_error = try
            @kernel begin
                x::Int
                @recipe (cse_key = :claimed_same) a::Int = identity(x)
                @recipe (cse_key = :claimed_same) b::String = string(x)
                return b
            end
            nothing
        catch err
            err
        end
        @test typed_cse_error isa ArgumentError
        @test occursin(
            "structural CSE output type mismatch",
            sprint(showerror, typed_cse_error),
        )

        multi_output_type_error = try
            @kernel begin
                x::Int
                @recipe (cse_key = :typed_pair) (a::Int, b::String) = (x, string(x))
                @recipe (cse_key = :typed_pair) (c::Int, d::Int) = (x, x)
                return c
            end
            nothing
        catch err
            err
        end
        @test multi_output_type_error isa ArgumentError
        @test occursin("position 2", sprint(showerror, multi_output_type_error))

        invalid_signatures = (
            quote
                @kernel begin
                    T::DataType
                    y::Int = begin
                        f(x::T) = x + 1
                        f(2)
                    end
                    return y
                end
            end,
            quote
                @kernel begin
                    T::DataType
                    y::Int = begin
                        function f(x::T)
                            x + 1
                        end
                        f(2)
                    end
                    return y
                end
            end,
            quote
                @kernel begin
                    T::DataType
                    y::DataType = begin
                        f(x::Q) where {Q<:T} = Q
                        f(2)
                    end
                    return y
                end
            end,
            quote
                @kernel begin
                    T::DataType
                    y::Int = begin
                        T = Int
                        f(x::T) = x + 1
                        f(2)
                    end
                    return y
                end
            end,
            quote
                @kernel begin
                    T::DataType
                    y::Int = begin
                        local T = Int
                        function f(x::T)
                            x + 1
                        end
                        f(2)
                    end
                    return y
                end
            end,
            quote
                @kernel begin
                    T::DataType
                    f::Function = (x::T -> x)
                    return f
                end
            end,
            quote
                @kernel begin
                    T::DataType
                    xs::Vector{Int}
                    y::Vector{Int} = map(xs) do x::T
                        x
                    end
                    return y
                end
            end,
        )
        for invalid in invalid_signatures
            signature_error = try
                macroexpand(@__MODULE__, invalid)
                nothing
            catch err
                err
            end
            @test signature_error isa ArgumentError
            @test occursin(
                "cannot appear in a nested local function signature annotation",
                sprint(showerror, signature_error),
            )
            @test occursin("runtime isa/convert", sprint(showerror, signature_error))
        end

        where_collision = @kernel begin
            S::DataType
            y::DataType = begin
                f(x::S) where S = S
                f(2)
            end
            return y
        end
        @test isempty(only(kernel_graph(where_collision).recipes).inputs)
        @test prepare(where_collision; have = ())() === Int
        @test_throws KeyError (@kernel begin x::Int end)[:missing]
    end
end
