module SourceWorldAgeTests
using ReactiveKernels, Enzyme, Test

@kernel plus(x, y) = begin
    result = x + y
    return result
end
const plus_alias = plus
const OFFSET = 0.75

function authored(ex)
    Core.eval(@__MODULE__, Expr(:macrocall,
        GlobalRef(ReactiveKernels, Symbol("@kernel")), LineNumberNode(1), ex))
end

const SCOPED_SOURCE = :(scoped_source(x) = begin
    result = begin
        f(value=x) = 2 * value
        f() + sum(i for i in 1:3 if isodd(i))
    end
    return result
end)
const scoped_ad_kernel = prepare(authored(SCOPED_SOURCE))

# The builder and its caller remain in the world they entered. No caller-side
# latest-world barrier may make a newly authored recipe executable for RK.
function immediate(head, x)
    spec = authored(:(dynamic(x) = begin
        result = $head(x + 1, 2 * x) + OFFSET
        return result
    end))
    kernel = prepare(spec)
    @test kernel(x) ≈ 3 * x + 1 + OFFSET
    @test only(only(Enzyme.autodiff(Enzyme.Reverse, kernel,
        Enzyme.Active, Enzyme.Active(x)))) == 3
    @test kernel(x + 0.5) ≈ 3 * (x + 0.5) + 1 + OFFSET
    spec
end

@testset "authored recipe bodies execute in their builder's world" begin
    for head in (:plus, :plus_alias, GlobalRef(@__MODULE__, :plus),
                 :(SourceWorldAgeTests.plus)), x in (-0.4, 2.0)
        immediate(head, x)
    end
end

@testset "authored source preserves captures and lexical scopes" begin
    spec = Core.eval(@__MODULE__, quote
        let offset = 1.0
            spec = @kernel captured(x, values) = begin
                result = x > 0 ? plus(x + offset, values[1]) : -x
                return result
            end
            # The already-created graph must read the shared lexical cell
            # when called, rather than freezing its old contents.
            offset = 3.0
            spec
        end
    end)
    kernel = prepare(spec)
    values = [4.0]
    @test kernel(-2.0, Float64[]) == 2.0
    @test kernel(2.0, values) == 9.0
    @test values == [4.0]
    @test prepare(spec; bound=(; values))(2.0) == 9.0

    scoped = Core.eval(@__MODULE__, quote
        let offset = 3.0
            @kernel scoped(x) = begin
                result = sum(map(i -> let offset = i
                    plus(x, offset)
                end, 1:3)) + offset + OFFSET
                return result
            end
        end
    end)
    @test prepare(scoped)(2.0) == 15.75

    reduced = authored(:(reduced(x, values) = begin
        result = sum(x * get(values, i, 0.0) for i in 1:4; init=0.0)
        return result
    end))
    @test prepare(reduced)(2.0, [3.0, 4.0]) == 14.0
    @test prepare(reduced)(2.0, Float64[]) == 0.0

    checked = authored(:(checked(x) = begin
        result = begin
            x > 0 || throw(ArgumentError("positive input required"))
            x * x + 1
        end
        return result
    end))
    @test prepare(checked)(2.0) == 5.0
    @test_throws ArgumentError prepare(checked)(-2.0)
    @test prepare(checked; on_error=:ignore)(-2.0) == 5.0
end

function immediate_scopes(x)
    @test prepare(authored(SCOPED_SOURCE))(x) == 2 * x + 4
    matrix = authored(:(matrix_source(x) = begin
        result = [x + i + j for i in 1:2, j in 1:3]
        return result
    end))
    @test prepare(matrix)(x) == [x + i + j for i in 1:2, j in 1:3]
    caught = authored(:(caught_source(x) = begin
        result = try
            error("boom")
        catch caught
            x + length(caught.msg)
        end
        return result
    end))
    @test prepare(caught)(x) == x + 4
end

@testset "Julia scopes retain immediate primal and ordinary native reverse" begin
    for x in (-0.4, 2.0)
        immediate_scopes(x)
        @test only(only(Enzyme.autodiff(Enzyme.Reverse, scoped_ad_kernel,
            Enzyme.Active, Enzyme.Active(x)))) == 2
    end
end

function immediate_wide()
    # Width is authored schema, independent of the point's runtime values.
    # One vector input keeps the public AD boundary narrow while the fused
    # recipe consumes forty scalar ports, beyond Julia's tuple-splat cliff.
    lanes = [Symbol(:lane_, i) for i in 1:40]
    statements = [:( $(lane) = q[$i]) for (i, lane) in enumerate(lanes)]
    total = Expr(:call, :+, [Expr(:call, :*, i, lane)
                            for (i, lane) in enumerate(lanes)]...)
    definition = Expr(:(=), Expr(:call, :wide, :q), Expr(:block,
        statements..., Expr(:(=), :result, total), Expr(:return, :result)))
    kernel = prepare(authored(definition))
    point = fill(0.25, 40)
    shadow = zeros(40)
    @test kernel(point) == 205.0
    Enzyme.autodiff(Enzyme.Reverse, kernel, Enzyme.Active,
                   Enzyme.Duplicated(point, shadow))
    @test shadow == Float64.(1:40)
    @test point == fill(0.25, 40)
    @test kernel(fill(-0.5, 40)) == -410.0
end

@testset "immediate ordinary reverse retains a wide fused call" begin
    immediate_wide()
end

function immediate_namespace()
    namespace = Module(gensym(:SourceWorldAgeNamespace))
    Core.eval(namespace, :(import ReactiveKernels))
    spec = Core.eval(namespace, :(ReactiveKernels.@kernel fresh(x) = begin
        result = x + 1
        return result
    end))
    @test Core.eval(namespace, :(isdefined(@__MODULE__, :fresh)))
    @test prepare(spec)(2.0) == 3.0
    # Binding discovery has its own world boundary on newer Julia. Direct
    # execution uses the returned graph value, without rediscovering its name.
    println("SOURCE_WORLD_AGE_RUNNING_BINDING_VISIBLE=", isdefined(namespace, :fresh))
end

@testset "a returned graph executes before namespace binding discovery" begin
    immediate_namespace()
end
end
