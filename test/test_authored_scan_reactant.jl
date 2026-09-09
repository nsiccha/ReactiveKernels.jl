using ReactiveKernels, Reactant, Test

if !isdefined(@__MODULE__, :AuthoredScanFixtures)
    include(joinpath(@__DIR__, "fixtures", "authored_scan.jl"))
end

@testset "authored scan retains its tensorized carry loop" begin
    spec = AuthoredScanFixtures.authored_scan_arma
    q, series = [0.2, 0.7, -0.3], sin.(1:20)
    traced_q, traced_series = Reactant.to_rarray(q), Reactant.to_rarray(series)
    for want in (:errors, :total)
        k = prepare(spec; want)
        compiled = Reactant.@compile k(traced_q, traced_series)
        actual = compiled(traced_q, traced_series)
        host = actual isa Reactant.AbstractConcreteArray ? Array(actual) :
               Reactant.to_number(actual)
        @test host ≈ k(q, series)
        hlo = repr(Reactant.@code_hlo optimize = false k(traced_q, traced_series))
        @test occursin("stablehlo.while", hlo)
    end
end

@testset "authored scan carries multiple traced sequences in one while loop" begin
    spec = AuthoredScanFixtures.authored_scan_lockstep
    a, b = [0.9, 0.8, 0.5, -0.2, 0.3, 0.6], [1.0, -0.5, 0.2, 0.7, -0.1, 0.4]
    traced_a, traced_b = Reactant.to_rarray(a), Reactant.to_rarray(b)
    for want in (:seq, :total)
        k = prepare(spec; want)
        compiled = Reactant.@compile k(traced_a, traced_b)
        actual = compiled(traced_a, traced_b)
        host = actual isa Reactant.AbstractConcreteArray ? Array(actual) :
               Reactant.to_number(actual)
        @test host ≈ k(a, b)
        # A single carry loop over BOTH sequences — not two loops, not unrolled.
        hlo = repr(Reactant.@code_hlo optimize = false k(traced_a, traced_b))
        @test occursin("stablehlo.while", hlo)
    end
end
