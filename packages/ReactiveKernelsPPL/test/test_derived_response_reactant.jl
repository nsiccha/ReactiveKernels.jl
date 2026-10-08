# Compiled (Reactant) checks for the cases in test_derived_response.jl
# (`_pv_reactant` comes from test_prior_vocab_reactant.jl).
using Reactant

_dr_bigcols() = Dict{Symbol,AbstractVector}(
    :earn => [2.0^i for i in 0:11], :x => repeat(_DR_X, 2))

@testset "derived response under Reactant" begin
    # Vector-mu Normal-id: default pipeline (narrowed §7n scope —
    # scalar-mu only).
    fx = _pv_reactant(_DR_M1, _dr_cols())
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
    # The derived local is a retained broadcast: doubling n_obs adds no
    # HLO lines (core constraint 1 — no data-derived unrolling).
    @test _pv_reactant(_DR_M1, _dr_bigcols()).lines == fx.lines
end
