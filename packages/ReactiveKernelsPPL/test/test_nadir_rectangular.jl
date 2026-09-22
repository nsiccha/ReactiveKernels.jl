module NadirRectangularTests
using ReactiveKernelsPPL, Reactant, Test
import Enzyme

@testset "ragged nadir retains one loop and reverse parity" begin
    for ends in ([0, 2, 2, 5], [5], [0, 0])
        x = [0.5, -1.0, -0.2, -0.3, 0.2][1:ends[end]]
        f = x -> tgi_segmented_nadir(x, ends)
        expected = f(x)
        rx = Reactant.to_rarray(x)
        hlo = repr(Reactant.@code_hlo optimize=false f(rx))
        @test count("stablehlo.while", hlo) == (isempty(x) ? 0 : 1)
        compiled = Reactant.@compile f(rx)
        @test Array(compiled(rx)) == expected
        isempty(x) && continue
        loss = x -> sum(f(x))
        gradient = x -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(loss), x))
        gref = gradient(x)
        compiled_gradient = Reactant.@compile gradient(rx)
        @test Array(compiled_gradient(rx)) == gref
    end
end
end
