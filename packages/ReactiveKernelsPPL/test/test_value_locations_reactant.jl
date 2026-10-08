# Compiled (Reactant) checks for the cases in test_value_locations.jl.
using Reactant

@testset "value location increment 2: Reactant and data-length invariance" begin
    for (body, data, optimize) in _vl_increment2_programs(_kinv_levels(3, 9))
        fx = _bare_reactant(_vl_program(body), data; optimize)
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
        large = _bare_reactant(_vl_program(body), _vl_bigger(data); optimize)
        @test large.lines == fx.lines
        @test large.primal ≈ large.native rtol = 1e-9
        @test large.rgrad ≈ large.g rtol = 1e-7 atol = 1e-9
    end
end
