using ReactiveKernels, Reactant, Test
import Enzyme
using DifferentiationInterface: AutoEnzyme

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

@testset "output-before-update scan with a concrete init keeps one while loop" begin
    # The running-nadir shape: the body emits the PREVIOUS carry, so the eager
    # first step returns the concrete host `init` as its output — still a
    # scalar per-step output, which used to throw at `_scan_output_buffer`.
    @kernel scmin_nadir(x) = begin
        out = scan(x; init = 0.0) do carry, c
            (min(carry, c), carry)
        end
        total::Float64 = sum(out)
        return total
    end
    x = [0.5, -1.0, 0.25]
    traced_x = Reactant.to_rarray(x)
    for want in (:out, :total)
        k = prepare(scmin_nadir; want)
        compiled = Reactant.@compile k(traced_x)
        actual = compiled(traced_x)
        host = actual isa Reactant.AbstractConcreteArray ? Array(actual) :
               Reactant.to_number(actual)
        @test host ≈ k(x)
        hlo = repr(Reactant.@code_hlo optimize = false k(traced_x))
        @test count("stablehlo.while", hlo) == 1
    end
    # The reported operation: Reactant-compiled value-and-gradient matches
    # native reverse Enzyme exactly.
    prepared = prepare_ad(
        scmin_nadir, AutoEnzyme(; mode = Enzyme.Reverse), x; active = :x, want = :total)
    gref = similar(x)
    vref, gref = ad_value_and_gradient!(prepared, gref, x)
    compiled_both = compile_ad_value_and_gradient(prepared, traced_x)
    value, gradient = compiled_both(traced_x)
    @test Float64(value) ≈ vref
    @test Array(gradient) ≈ gref
    # N == 1 runs the eager first step with an empty loop body.
    x1 = [0.5]
    k1 = prepare(scmin_nadir; want = :total)
    compiled1 = Reactant.@compile k1(Reactant.to_rarray(x1))
    @test Reactant.to_number(compiled1(Reactant.to_rarray(x1))) ≈ k1(x1)
    # A genuinely non-scalar per-step output stays a loud ArgumentError: native
    # supports it, the Reactant `while` path does not.
    @kernel tuplescan_out(x) = begin
        out = scan(x; init = 0.0) do carry, c
            (carry + c, (carry, c))
        end
        return out
    end
    @test prepare(tuplescan_out)(x) isa Vector
    @test_throws ArgumentError Reactant.@compile prepare(tuplescan_out)(traced_x)
end

@testset "bound host sequences retain the while loop with a length-independent program" begin
    spec = AuthoredScanFixtures.authored_scan_arma
    q = [0.2, 0.7, -0.3]
    traced_q = Reactant.to_rarray(q)
    sizes = Int[]
    for n in (8, 16)
        series = sin.(1:n)
        k = prepare(spec; want = :errors, bound = (; series))
        compiled = Reactant.@compile k(traced_q)
        @test Array(compiled(traced_q)) ≈ k(q)
        hlo = repr(Reactant.@code_hlo optimize = false k(traced_q))
        @test count("stablehlo.while", hlo) == 1
        push!(sizes, count("\n", hlo))
    end
    # Only tensor shapes differ between the two programs: no per-step copy.
    @test sizes[1] == sizes[2]
end

@testset "a traced sequence beside a bound sequence shares one while loop" begin
    spec = AuthoredScanFixtures.authored_scan_lockstep
    a, b = [0.9, 0.8, 0.5, -0.2, 0.3, 0.6], [1.0, -0.5, 0.2, 0.7, -0.1, 0.4]
    k = prepare(spec; want = :seq, bound = (; b))
    traced_a = Reactant.to_rarray(a)
    compiled = Reactant.@compile k(traced_a)
    @test Array(compiled(traced_a)) ≈ k(a)
    hlo = repr(Reactant.@code_hlo optimize = false k(traced_a))
    @test count("stablehlo.while", hlo) == 1
end

@testset "eachrow scans retain one while loop independent of length and width" begin
    spec = AuthoredScanFixtures.authored_scan_eachrow
    gain = Reactant.to_rarray(0.5; track_numbers = true)
    bound_sizes = Int[]
    traced_sizes = Int[]
    for K in (3, 6), n in (4, 8)
        M = hcat(collect(1.0:n), fill(0.5, n), fill(0.25, n), fill(0.1, n, K - 3))
        # A bound (host) matrix is lifted into the traced program.
        bound = prepare(spec; want = :total, bound = (; mat = M))
        compiled_bound = Reactant.@compile bound(gain)
        @test Reactant.to_number(compiled_bound(gain)) ≈ bound(0.5)
        bound_hlo = repr(Reactant.@code_hlo optimize = false bound(gain))
        @test count("stablehlo.while", bound_hlo) == 1
        push!(bound_sizes, count("\n", bound_hlo))
        # A traced matrix gathers each row as one dynamic slice, so the row
        # width never unrolls the step body.
        traced = prepare(spec; want = :total)
        traced_M = Reactant.to_rarray(M)
        compiled_traced = Reactant.@compile traced(traced_M, gain)
        @test Reactant.to_number(compiled_traced(traced_M, gain)) ≈ traced(M, 0.5)
        traced_hlo = repr(Reactant.@code_hlo optimize = false traced(traced_M, gain))
        @test count("stablehlo.while", traced_hlo) == 1
        push!(traced_sizes, count("\n", traced_hlo))
    end
    @test allequal(bound_sizes)
    @test allequal(traced_sizes)
end
