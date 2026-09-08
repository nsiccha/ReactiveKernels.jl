using ReactiveKernels, Reactant, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme

isdefined(@__MODULE__, :InnerPlatePartialEvaluation) ||
    include("fixtures/inner_plate_partial_evaluation.jl")

@testset "Inner partial evaluation on loaded Reactant" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing
    C = InnerPlatePartialEvaluation
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for n in (3, 32), (spec, data) in (
            (C.pure, collect(range(1.0, 4.0; length=n))),
            (C.boolean, [i - 2 for i in 1:n]))
        q = [0.3, -0.1]
        for bound in (false, true)
            args = bound ? (q,) : (q, data)
            kernel = prepare(spec; bound=bound ? (; data) : NamedTuple())
            prepared = prepare_ad(kernel, backend, args...; active=:q)
            value, gradient = ad_value_and_gradient(prepared, args...)
            traced = map(Reactant.to_rarray, args)
            primal = Reactant.compile(kernel, traced; sync=true)
            both = compile_ad_value_and_gradient(prepared, traced...; sync=true)
            @test Float64(primal(traced...)) ≈ value
            result, derivative = both(traced...)
            @test Float64(result) ≈ value
            @test Array(derivative) ≈ gradient
        end
    end
end
