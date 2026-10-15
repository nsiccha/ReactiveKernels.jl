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

# A history-scan step large enough that inlining it is a cost decision: a
# max-shifted log-sum-exp over the previous step's K outputs.
@kernel scan_logsumexp(log_emissions, log_transition) = begin
    K = size(log_emissions, 2)
    T = size(log_emissions, 1)
    forward = scan(1:(T * K); init = 0, history = 0.0) do carry, cell, earlier
        t = (cell - 1) ÷ K + 1
        j = (cell - 1) % K + 1
        previous = (t - 2) * K
        shift = t == 1 ? 0.0 : maximum(earlier[previous + i] for i in 1:K)
        incoming = t == 1 ? 0.0 : isfinite(shift) ?
            shift + log(sum(exp(earlier[previous + i] - shift + log_transition[i, j])
                for i in 1:K; init = 0.0)) : shift
        (carry, incoming + log_emissions[t, j])
    end
    total = sum(forward)
    return total
end

function out_of_line_source_calls(kernel, args)
    kernel isa ReactiveKernels.PreparedKernel || throw(ArgumentError(
        "expected a PreparedKernel, got $(typeof(kernel))"))
    code, _ = only(Base.code_typed(kernel, Tuple{map(typeof, args)...}; optimize = true))
    count(code.code) do statement
        statement isa Expr && statement.head === :invoke || return false
        instance = statement.args[1]
        instance isa Core.CodeInstance && (instance = instance.def)
        instance.def.name in (:_KernelSourceFunction, :_ExecutableSourceFunction)
    end
end

# `prepare_sampler` converts every operation of a prepared kernel to the
# latest-world call mode. That mode once left a large step body behind a call
# per scan step, which made native Enzyme reverse through an HMM forward pass
# 2.5–3× slower than the same kernel in the guarded mode (snag
# `rkppl-hmm-forwar-89257681`). Both modes inline the source call.
@testset "latest-world source calls are inlined into their operations" begin
    e = [sin(3t + k) for t in 1:40, k in 1:3]
    lt = log.([0.8 0.1 0.1; 0.2 0.6 0.2; 0.3 0.3 0.4])
    guarded = prepare(scan_logsumexp)
    latest = ReactiveKernels._kernel_latest_prepared(guarded)
    @test out_of_line_source_calls(guarded, (e, lt)) == 0
    @test out_of_line_source_calls(latest, (e, lt)) == 0
    @test Base.invokelatest(latest, e, lt) == guarded(e, lt)
    function gradient_of(kernel)
        de = zero(e)
        Base.invokelatest(Enzyme.autodiff, Enzyme.Reverse, Enzyme.Const(kernel),
            Enzyme.Active, Enzyme.Duplicated(e, de), Enzyme.Const(lt))
        de
    end
    @test gradient_of(latest) == gradient_of(guarded)
end
end
