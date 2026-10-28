using ReactiveKernels, Reactant, Enzyme, Test

# The consumer image's first-use wrapper holds generated code, so Reactant
# must treat it as static metadata rather than traversing its expression.
module _PrecompiledNativeBackendFixture
using ReactiveKernels
const RGF = ReactiveKernels.RuntimeGeneratedFunctions
RGF.init(@__MODULE__)
const EXPR = :((x,) -> x * x + exp(x))
const QUALIFIED = ReactiveKernels._native_parameter_names(EXPR,
    macroexpand(ReactiveKernels, Expr(:macrocall,
        GlobalRef(ReactiveKernels, Symbol("@_native_context")), LineNumberNode(0), EXPR)))
const NATIVE = RGF.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__,
    QUALIFIED)
const WARMED = ReactiveKernels._PrecompileWarmFunction(NATIVE)
end

@testset "precompiled native callable backend compatibility" begin
    f = _PrecompiledNativeBackendFixture.WARMED
    @test (@inferred f(0.7)) ≈ 0.7^2 + exp(0.7)
    derivative = only(only(Enzyme.autodiff(Enzyme.Reverse, f,
        Enzyme.Active, Enzyme.Active(0.7))))
    @test derivative ≈ 1.4 + exp(0.7)
    traced = Reactant.to_rarray(0.7; track_numbers = true)
    compiled = Reactant.@compile sync = true f(traced)
    @test Float64(compiled(traced)) ≈ f(0.7)
    dropped = ReactiveKernels.RuntimeGeneratedFunctions.drop_expr(f)
    @test (@inferred dropped(0.9)) ≈ 0.9^2 + exp(0.9)
    @test ReactiveKernels._sm_compiled_call(f) === f
end
