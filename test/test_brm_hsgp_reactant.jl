include("test_brm_hsgp.jl")
using Reactant
import Enzyme
using DifferentiationInterface: AutoEnzyme

@testset "BRM motorcycle Reactant value and gradient" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing
    data = BRMHSGPExample.motorcycle_data(joinpath(@__DIR__, "..", "examples", "data", "mcycle.csv"))
    kernel = BRMHSGPExample.prepare_model(data)
    q = 0.04sin.(collect(1.0:44.0))
    q[[1, 23]] .= -2.0
    c = zeros(40)
    backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
    prepared = prepare_ad(kernel, backend, q, c; active=:q)
    rq, rc = Reactant.to_rarray(q), Reactant.to_rarray(c)
    primal = Reactant.@compile sync=true kernel(rq, rc)
    compiled = compile_ad_value_and_gradient(prepared, rq, rc)
    for c in (zeros(40), ones(40), collect(range(0, 1; length=40)),
              repeat([0.0, 0.25, 0.6, 1.0], 10))
        value, gradient = compiled(rq, Reactant.to_rarray(c))
        @test Float64(value) ≈ motorcycle_reference(q, c, data) atol=2e-10 rtol=2e-12
        @test Float64(primal(rq, Reactant.to_rarray(c))) ≈ Float64(value) atol=2e-10 rtol=2e-12
        native_value, native_gradient = ad_value_and_gradient!(prepared, similar(q), q, c)
        @test native_value ≈ Float64(value) atol=2e-10 rtol=2e-12
        @test native_gradient ≈ Array(gradient) atol=2e-9 rtol=2e-10
        g = Array(gradient)
        # An independent central finite difference checks all 44 coordinates.
        for i in eachindex(q)
            plus, minus = copy(q), copy(q)
            plus[i] += 1e-5
            minus[i] -= 1e-5
            fd = (motorcycle_reference(plus, c, data) - motorcycle_reference(minus, c, data)) / 2e-5
            @test g[i] ≈ fd atol=3e-7 rtol=2e-6
        end
    end
end
