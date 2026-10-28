module GraphDensityReactantTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Reactant
import DifferentiationInterface as DI
import Enzyme
include("fixtures/graph_density.jl")
using .GraphDensityFixtures

@testset "Reactant: composed caller density values, reverse and batched growth" begin
    for rowwise in (false, true), n in (0, 1, 5, 19)
        fx = GraphDensityFixtures.fixture(n; rowwise)
        saved = deepcopy(fx.data)
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        kernel = sampler.kernel
        reverse(v) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
        input = Reactant.to_rarray(fx.u)
        for (direction, fn) in ((:primal, kernel), (:reverse, reverse))
            mlir = repr(Reactant.@code_hlo optimize=:all fn(input))
            compiled = Reactant.@compile optimize=:all fn(input)
            hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            if haskey(ENV, "RKPPL_GRAPH_DENSITY_IR_DIR")
                dir = ENV["RKPPL_GRAPH_DENSITY_IR_DIR"]
                mkpath(dir)
                write(joinpath(dir, "$rowwise-$n-$direction.mlir"), mlir)
                write(joinpath(dir, "$rowwise-$n-$direction.hlo"), hlo)
            end
            for shift in (0.0, 0.07)
                u = fx.u .+ shift
                ru = Reactant.to_rarray(u)
                expected = GraphDensityFixtures.oracle(fx, u)
                if direction === :primal
                    @test Float64(compiled(ru)) ≈ expected.posterior rtol=2e-11
                else
                    _, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
                    @test Array(compiled(ru)) ≈ gradient rtol=2e-9 atol=2e-10
                end
                @test Array(ru) == u
            end
            println("GRAPH_DENSITY_IR rowwise=",rowwise," n=",n," direction=",direction)
        end
        @test Array(input) == fx.u
        @test fx.data == saved
    end
end
end
