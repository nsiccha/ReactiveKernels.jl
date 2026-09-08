using ReactiveKernels, Reactant, Test
using DifferentiationInterface
import Enzyme
isdefined(@__MODULE__, :AuthoredPlateChains) ||
    include("fixtures/authored_plate_chains.jl")

@testset "Authored plate chain Reactant parity" begin
    C = AuthoredPlateChains
    for n in (8, 32)
        q = [0.7]
        x = collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        rq, rx, ry = Reactant.to_rarray.((q, x, y))
        for want in (:total, :pointwise)
            kernel = prepare(C.chain; want)
            compiled = Reactant.@compile sync = true kernel(rq, rx, ry)
            result = compiled(rq, rx, ry)
            host_result = result isa AbstractArray ? Array(result) : Float64(result)
            @test host_result ≈ kernel(q, x, y)
        end
        kernel = prepare(C.chain; bound = (; x, y))
        compiled = Reactant.@compile sync = true kernel(rq)
        @test Float64(compiled(rq)) ≈ kernel(q)
        ad = prepare_ad(kernel, AutoEnzyme(; mode = Enzyme.Reverse), q; active = :q)
        compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rq)
        value, gradient = compiled_ad(ad, rq)
        @test Float64(value) ≈ kernel(q)
        @test Array(gradient) ≈ ad_gradient(ad, q)
        hlo = repr(Reactant.@code_hlo optimize = true kernel(rq))
        @test occursin("stablehlo", hlo)
    end
end
