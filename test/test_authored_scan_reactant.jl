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

@testset "authored eachrow scan keeps one while loop" begin
    spec = AuthoredScanFixtures.authored_scan_eachrow
    M = hcat(collect(1.0:12.0), fill(0.5, 12), fill(0.25, 12))
    traced_M, traced_gain = Reactant.to_rarray(M), Reactant.to_rarray(0.5)
    @test AuthoredScanFixtures._authored_scan_eachrow_reference(M, 0.5) ≈
        prepare(spec; want = :seq)(M, 0.5)
    for want in (:seq, :total)
        k = prepare(spec; want)
        compiled = Reactant.@compile k(traced_M, traced_gain)
        actual = compiled(traced_M, traced_gain)
        host = actual isa Reactant.AbstractConcreteArray ? Array(actual) :
               Reactant.to_number(actual)
        @test host ≈ k(M, 0.5)
        # Exactly one carry loop — the snag's regression: this shape used to
        # fall through to the generic loop and trace fully unrolled (0 whiles).
        hlo = repr(Reactant.@code_hlo optimize = false k(traced_M, traced_gain))
        @test count("stablehlo.while", hlo) == 1
    end
    # N == 1 runs the eager first step with an empty loop body.
    M1 = reshape([1.0, 0.5, 0.25], 1, 3)
    k1 = prepare(spec; want = :total)
    compiled1 = Reactant.@compile k1(Reactant.to_rarray(M1), traced_gain)
    @test Reactant.to_number(compiled1(Reactant.to_rarray(M1), traced_gain)) ≈ k1(M1, 0.5)
    hlo1 = repr(Reactant.@code_hlo optimize = false k1(Reactant.to_rarray(M1), traced_gain))
    @test count("stablehlo.while", hlo1) <= 1
end
