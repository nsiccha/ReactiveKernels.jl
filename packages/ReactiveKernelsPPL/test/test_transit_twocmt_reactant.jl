using Reactant
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_transit_reactant_response(ts, p) = transit_twocmt_rule(ts, p)

@testset "transit Reactant primal retains loops and lazy regimes" begin
    # Zero lags require the logarithmic arms to remain inactive. These lags
    # also straddle the power-series, gamma-series, tail and Watson regimes.
    grid = [0.0, 5.3, 5.5, 24.0, 325.0, 614.0, 616.0]
    p = [0.08, 0.15, 0.05, 0.2, 1.2]
    counts = Tuple{Int,Int}[]
    for n in (7, 29)
        ts = repeat(grid, cld(n, length(grid)))[1:n]
        rt, rp = Reactant.to_rarray(ts), Reactant.to_rarray(p)
        compiled = Reactant.@compile _transit_reactant_response(rt, rp)
        native = _transit_reactant_response(ts, p)
        value = Array(compiled(rt, rp))
        @test all(isfinite, value)
        @test value ≈ native rtol=2e-13 atol=2e-14
        @test first(value) == 0.0

        # Reuse one executable across both rationalized disposition arms,
        # different numerical regimes and the shape-one zero-lag arm.
        for p2 in ([0.08, 0.15, 0.5, 0.2, 1.2],
                   [0.08, 0.15, 0.05, 0.4, 2.0],
                   [0.08, 0.15, 0.05, 0.2, 1.0])
            rp2 = Reactant.to_rarray(p2)
            @test Array(compiled(rt, rp2)) ≈
                _transit_reactant_response(ts, p2) rtol=2e-13 atol=2e-14
            @test Array(rp2) == p2
        end
        # Every lag changes, including zero becoming positive. A frozen
        # input or shape-only control would fail this executable reuse.
        ts2 = ts .+ 0.75
        rt2 = Reactant.to_rarray(ts2)
        @test Array(compiled(rt2, rp)) ≈
            _transit_reactant_response(ts2, p) rtol=2e-13 atol=2e-14
        @test Array(rt2) == ts2
        @test Array(rt) == ts
        @test Array(rp) == p

        hlo = String(Reactant.@code_hlo _transit_reactant_response(rt, rp))
        push!(counts, (count("stablehlo.while", hlo), count("stablehlo.if", hlo)))
    end
    @test first(counts) == last(counts)
    @test all(>(0), first(counts))
end

@testset "transit Reactant empty lags and explicit accuracy controls" begin
    p = [0.08, 0.15, 0.05, 0.2, 1.2]
    rt, rp = Reactant.to_rarray(Float64[]), Reactant.to_rarray(p)
    compiled = Reactant.@compile _transit_reactant_response(rt, rp)
    @test isempty(Array(compiled(rt, rp)))
    @test isempty(Array(rt))
    @test Array(rp) == p

    loose = prepare_transit_twocmt_rule(; series_rtol=1e-6, watson_terms=4)
    ts = [0.0, 5.3, 5.5, 24.0, 325.0, 614.0, 616.0]
    rt = Reactant.to_rarray(ts)
    compiled_loose = Reactant.@compile loose(rt, rp)
    @test Array(compiled_loose(rt, rp)) ≈ loose(ts, p) rtol=2e-13 atol=2e-14
end
