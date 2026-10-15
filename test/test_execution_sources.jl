module ExecutionSourceTests
using ReactiveKernels, Enzyme, Test
const RK = ReactiveKernels

@testset "execution shares code while preserving definition provenance" begin
    mod = Module(gensym(:ExecutionSource))
    Core.eval(mod, :(using ReactiveKernels; shift(x) = x + 1.0))
    definition = :(model(x::Float64) = begin
        value::Float64 = shift(x) + 2.0
        return value
    end)
    specs = [RK._kernel_eval_definition(mod, deepcopy(definition)) for _ in 1:2]
    kernels = [prepare(spec) for spec in specs]
    @test typeof(kernels[1].f) === typeof(kernels[2].f)
    @test typeof(kernels[1].ops) === typeof(kernels[2].ops)
    @test Base.invokelatest(kernels[1], 2.0) == Base.invokelatest(kernels[2], 2.0) == 5.0
    source = [only(filter(r -> r.op isa RK._KernelSourceOp, spec.graph.recipes)) for spec in specs]
    @test RK._kernel_source_token(source[1].op) !== RK._kernel_source_token(source[2].op)
    for kernel in kernels, (op, recipe) in zip(kernel.ops, kernel.lowered_recipes)
        op isa RK._KernelSourceOp || continue
        @test RK.kernel_sourceop_token(op) === RK.kernel_sourceop_token(recipe.op)
        @test RK._kernel_endpoint_source_module(op) === RK._kernel_endpoint_source_module(recipe.op)
        @test RK._source_parameter_names(op, recipe.source, length(recipe.inputs)) ==
            RK._source_parameter_names(recipe.op, recipe.source, length(recipe.inputs))
    end
    # Generated-source reuse must still observe ordinary Julia method changes.
    Core.eval(mod, :(shift(x) = 3.0x))
    newer = prepare(RK._kernel_eval_definition(mod, deepcopy(definition)))
    @test Base.invokelatest(newer, 2.0) == 8.0
    @test Base.invokelatest(kernels[1], 2.0) == 8.0
end

@testset "execution branch metadata preserves host branch behavior" begin
    call(x, scale) = scale > 0 ? x * log(scale) : -x
    branch = RK._KernelBranch(Val((2,)), Val((1, 2)), Val((1,)), call,
        scale -> scale > 0, (x, scale) -> x * log(scale), x -> -x)
    source = RK._KernelSourceOp(Val(gensym(:branch)), Val(:fused), branch, branch)
    op = RK._execution_source(source)
    @test RK.kernel_sourceop_token(op) === RK.kernel_sourceop_token(source)
    @test RK._tensorized_plate_dispatch(op, ([2.0, 3.0], 2.0)) ≈
        [2.0, 3.0] .* log(2.0)
    @test RK._tensorized_plate_dispatch(op, ([2.0, 3.0], -2.0)) ==
        [-2.0, -3.0]
end

@testset "runtime source conversion preserves lexical captures" begin
    function make(offset)
        @kernel captured(x::Float64) = begin
            y::Float64 = x + offset
            return y
        end
    end
    a, b = prepare(make(1.0)), prepare(make(4.0))
    @test typeof(a.ops) === typeof(b.ops)
    @test a(2.0) == 3.0
    @test b(2.0) == 6.0
    function make_ref(offset)
        @kernel captured_ref(x::Float64) = begin
            y::Float64 = x + offset[]
            return y
        end
    end
    offset = Ref(1.0)
    live = prepare(make_ref(offset))
    @test live(2.0) == 3.0
    offset[] = 4.0
    @test live(2.0) == 6.0
    @test only(only(Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(live),
        Enzyme.Active, Enzyme.Active(2.0)))) == 1.0
end
end
