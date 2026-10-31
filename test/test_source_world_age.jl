module SourceWorldAgeTests
using ReactiveKernels, Enzyme, Test

# A consumer's cache binding is newer than RK's generated source-call method.
# This is also the package-image case: its cache belongs to the consumer,
# while the source record's call method belongs to the dependency.
const RGF = ReactiveKernels.RuntimeGeneratedFunctions
RGF.init(@__MODULE__)
const consumer_companion = RGF.drop_expr(RGF.RuntimeGeneratedFunction(
    @__MODULE__, @__MODULE__,
    :((captures, x, location, scale) -> captures.offset + (x - location) / scale)))
const consumer_source = ReactiveKernels._KernelSource(
    (; offset=0.25), consumer_companion, nothing)

@testset "source companions read newer consumer caches" begin
    @test (@inferred consumer_source(3.0, 1.0, 2.0)) == 1.25
    @test only(only(Enzyme.autodiff(Enzyme.Reverse,
        x -> consumer_source(x, 1.0, 2.0), Enzyme.Active, Enzyme.Active(3.0)))) == 0.5
end

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

# A plate's buffer is typed from its cells' sources before the loop runs, so
# that typing reads the builder's fresh sources in the builder's world too.
# Typed through the authored closure, whose method the builder's world does not
# see, a cell type inferred `Union{}`: the primal raised a `MethodError` in
# `_static_plate_axis_class(::Type{Union{}})`, native Reverse an
# `EnzymeRuntimeActivityError`, and a nested NLME fit that differentiated its
# kernel in the fit's own world failed compiling it (snag
# `brm-canonical-44-a1653c2a`).
function immediate_plates(groups, s)
    flat = authored(:(flat_cells(groups, s::Float64) = begin
        cells = plate(groups) do g
            scaled = g .* s
            sum(scaled)
        end
        return cells
    end))
    nested = authored(:(nested_cells(groups, s::Float64) = begin
        cells = plate(groups) do g
            scaled = g .* s
            shifted = plate(scaled) do v
                v + s
            end
            sum(shifted)
        end
        return cells
    end))
    for (spec, slopes) in ((flat, sum.(groups)),
                           (nested, sum.(groups) .+ length.(groups)))
        kernel = prepare(spec)
        @test kernel(groups, s) ≈ s .* slopes
        @test only(only(Enzyme.autodiff(Enzyme.Reverse,
            Enzyme.Const(t -> sum(kernel(groups, t))), Enzyme.Active,
            Enzyme.Active(s)))) ≈ sum(slopes)
    end
end

@testset "plate kernels execute and differentiate in their builder's world" begin
    for s in (0.5, 2.0)
        immediate_plates([[1.0, 2.0], [3.0], Float64[]], s)
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
        instance.def.name === :_KernelSource
    end
end

# A scan step's source call is inlined into its operation: a large step body
# behind a call per scan step once made native Enzyme reverse through an HMM
# forward pass 2.5–3× slower than the same kernel with the step inlined (snag
# `rkppl-hmm-forwar-89257681`). `prepare_sampler`'s latest-world preparation
# is the same kernel: a source record is callable from any world.
@testset "source calls are inlined into their operations" begin
    e = [sin(3t + k) for t in 1:40, k in 1:3]
    lt = log.([0.8 0.1 0.1; 0.2 0.6 0.2; 0.3 0.3 0.4])
    kernel = prepare(scan_logsumexp)
    @test ReactiveKernels._kernel_latest_prepared(kernel) === kernel
    @test out_of_line_source_calls(kernel, (e, lt)) == 0
    de = zero(e)
    Base.invokelatest(Enzyme.autodiff, Enzyme.Reverse, Enzyme.Const(kernel),
        Enzyme.Active, Enzyme.Duplicated(e, de), Enzyme.Const(lt))
    @test isfinite(sum(de))
end

# The record's call splices the companion's body into its own generated
# method, whose static parameters are named as no source can spell a local:
# a model's `R = Z * F'` in a spliced body once failed to lower ("local
# variable name R conflicts with a static parameter").
@kernel static_parameter_names(x) = begin
    result = begin
        R = x .* 2
        C = sum(R)
        N = length(R)
        K = C / N
        K + first(R)
    end
    return result
end

@testset "a spliced source body may bind the call's static parameter names" begin
    x = [0.5, 1.5, 2.5]
    expected = 2 * sum(x) / length(x) + 2 * first(x)
    @test prepare(static_parameter_names)(x) ≈ expected
    recipe = only(plan(static_parameter_names; want = :result).recipes)
    record = ReactiveKernels._source_record(getfield(recipe.op, :f))
    @test record isa ReactiveKernels._KernelSource
    @test record(x) ≈ expected
end

# Authored parameters share the spliced method's scope. They may name its
# argument variables or the builtin used to unpack them, without changing
# those reads or degrading a plate's inferred element type.
@kernel source_parameter_names(getfield::Float64, f::Float64, args::Float64) = begin
    result = getfield * getfield + 3f + 5args
    return result
end
@kernel source_parameter_cells(values) = begin
    result = plate(values) do getfield
        getfield * getfield + 1
    end
    return result
end

@testset "source parameters cannot shadow generated argument reads" begin
    recipe = only(plan(source_parameter_names).recipes)
    record = ReactiveKernels._source_record(recipe.op.f)
    @test @inferred(record(2.0, 3.0, 4.0)) == 33.0
    for mode in ("0", "1")
        withenv("RK_INLINE_SOURCES" => mode) do
            kernel = prepare(source_parameter_names)
            @test @inferred(kernel(2.0, 3.0, 4.0)) == 33.0
            @test only(Enzyme.autodiff(Enzyme.Reverse, kernel, Enzyme.Active,
                Enzyme.Active(2.0), Enzyme.Active(3.0), Enzyme.Active(4.0))) ==
                (4.0, 3.0, 5.0)
            cells = prepare(source_parameter_cells)
            @test cells([1.0, 2.0]) == [2.0, 5.0]
            @test cells(Float64[]) isa Vector{Float64}
        end
    end
end
end
