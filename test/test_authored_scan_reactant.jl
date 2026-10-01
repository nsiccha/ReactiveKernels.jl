using ReactiveKernels, Reactant, Test
import Enzyme
using DifferentiationInterface: AutoEnzyme

if !isdefined(@__MODULE__, :AuthoredScanFixtures)
    include(joinpath(@__DIR__, "fixtures", "authored_scan.jl"))
end

@testset "prepared authored scan runs only in the live lazy arm" begin
    mat = [1.0 0.2 0.1; 2.0 0.3 0.4; 3.0 0.5 0.6]
    positions = collect(1:3)
    traced_mat = Reactant.to_rarray(mat)
    traced_positions = Reactant.to_rarray(positions)
    weights = prepare(AuthoredScanFixtures.authored_scan_lazy_branch; want = :weights)
    compiled = Reactant.@compile weights(traced_mat, traced_positions, 3)
    @test Array(compiled(traced_mat, traced_positions, 3)) == weights(mat, positions, 3)
    hlo = repr(Reactant.@code_hlo optimize = false weights(traced_mat, traced_positions, 3))
    @test count("stablehlo.while", hlo) == 1

    empty_mat = Reactant.to_rarray(zeros(Float64, 0, 1))
    empty_positions = Reactant.to_rarray(Int[])
    total = prepare(AuthoredScanFixtures.authored_scan_lazy_branch; want = :total)
    empty_compiled = Reactant.@compile total(empty_mat, empty_positions, 0)
    @test Reactant.to_number(empty_compiled(empty_mat, empty_positions, 0)) == 0.0
    empty_hlo = repr(Reactant.@code_hlo optimize = false total(empty_mat, empty_positions, 0))
    @test count("stablehlo.while", empty_hlo) == 0

    qualified = AuthoredScanFixtures.QualifiedScanBinding.prepared
    traced_xs = Reactant.to_rarray([1.0, 2.0, 3.0])
    qualified_compiled = Reactant.@compile qualified(traced_xs)
    @test Array(qualified_compiled(traced_xs)) == [1.0, 3.0, 6.0]
    qualified_hlo = repr(Reactant.@code_hlo optimize = false qualified(traced_xs))
    @test count("stablehlo.while", qualified_hlo) == 1
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
        @test count("stablehlo.while", hlo) == 1
    end
    # The program is independent of the sequence length, traced or bound.
    traced_sizes, bound_sizes = Int[], Int[]
    for n in (6, 12)
        a, b = sin.(1:n), cos.(1:n)
        k = prepare(spec; want = :seq)
        traced_hlo = repr(Reactant.@code_hlo optimize = false k(
            Reactant.to_rarray(a), Reactant.to_rarray(b)))
        @test count("stablehlo.while", traced_hlo) == 1
        push!(traced_sizes, count("\n", traced_hlo))
        kb = prepare(spec; want = :seq, bound = (; b))
        bound_hlo = repr(Reactant.@code_hlo optimize = false kb(Reactant.to_rarray(a)))
        @test count("stablehlo.while", bound_hlo) == 1
        push!(bound_sizes, count("\n", bound_hlo))
    end
    @test allequal(traced_sizes)
    @test allequal(bound_sizes)
end

@kernel authored_scan_reactant_doses(mgs::Vector{Float64}, units::Matrix{Float64}, n::Int) = begin
    slots = collect(1:n)
    weights = scan(mgs, eachrow(units), Ref(slots);
            init = (; prior = zeros(Float64, n), index = 1)) do carry, mg, u, positions
        weight = mg / (1 + sum(carry.prior .* u))
        next = ifelse.(positions .== carry.index, weight, carry.prior)
        ((; prior = next, index = carry.index + 1), weight)
    end
    total::Float64 = 1.0 + sum(weights)
    return total
end

@kernel authored_scan_reactant_turnover(pd, conc_mid::Vector{Float64}, dts::Vector{Float64}) = begin
    kin = pd.baseline * pd.kout
    updated = scan(conc_mid, dts, Ref(pd), Ref(kin);
            init = pd.baseline) do previous, concentration, dt, parameters, input_rate
        c2 = parameters.kout * (1 + concentration /
            (parameters.theta1 * concentration + parameters.theta2))
        steady = input_rate / c2
        next = (previous - steady) * exp(-c2 * dt) + steady
        (next, next)
    end
    trajectory = vcat([pd.baseline], updated)
    return trajectory
end

@testset "an empty traced sequence compiles to no loop and an empty result" begin
    pd = (; baseline = 100.0, kout = 0.3, theta1 = 1.2, theta2 = 40.0)
    traced_pd = Reactant.to_rarray(pd; track_numbers = true)
    for n in (0, 4)
        conc, dts = collect(1.0:n), fill(0.1, n)
        whiles = n == 0 ? 0 : 1
        # Every sequence traced, then the whole schedule bound as host data.
        k = prepare(authored_scan_reactant_turnover)
        traced = (Reactant.to_rarray(conc), Reactant.to_rarray(dts))
        compiled = Reactant.@compile k(traced_pd, traced...)
        @test Array(compiled(traced_pd, traced...)) ≈ k(pd, conc, dts)
        hlo = repr(Reactant.@code_hlo optimize = false k(traced_pd, traced...))
        @test count("stablehlo.while", hlo) == whiles
        kb = prepare(authored_scan_reactant_turnover; bound = (; conc_mid = conc, dts))
        compiled_bound = Reactant.@compile kb(traced_pd)
        @test Array(compiled_bound(traced_pd)) ≈ kb(pd)
        bound_hlo = repr(Reactant.@code_hlo optimize = false kb(traced_pd))
        @test count("stablehlo.while", bound_hlo) == whiles

        # A vector carry over a 1-D sequence beside matrix rows.
        mgs, units = collect(1.0:n), [i < j ? 0.5 : 0.0 for j in 1:n, i in 1:n]
        kd = prepare(authored_scan_reactant_doses)
        traced_doses = (Reactant.to_rarray(mgs), Reactant.to_rarray(units))
        compiled_doses = Reactant.@compile kd(traced_doses..., n)
        @test Reactant.to_number(compiled_doses(traced_doses..., n)) ≈ kd(mgs, units, n)
        doses_hlo = repr(Reactant.@code_hlo optimize = false kd(traced_doses..., n))
        @test count("stablehlo.while", doses_hlo) == whiles
    end
    # The empty scan port itself as the program output: an empty traced input of
    # the output type is forwarded, which XLA exports (§7l).
    k = prepare(authored_scan_reactant_turnover; want = :updated)
    empty = (Reactant.to_rarray(Float64[]), Reactant.to_rarray(Float64[]))
    compiled = Reactant.@compile k(traced_pd, empty...)
    @test Array(compiled(traced_pd, empty...)) == Float64[]
end

@kernel authored_scan_reactant_shared(xs::Vector{Float64}, amounts::Vector{Float64}) = begin
    path::Vector{Float64} = scan(xs, Ref(amounts); init = 0.0) do carry, x, a
        next = carry + x + sum(a; init = 0.0)
        (next, next)
    end
    return path
end

@testset "a zero-sized traced shared operand compiles in the scan loop" begin
    # The loop reads it through a fresh tracer; rebinding the caller's argument
    # to a loop result made XLA export fail on `tensor.empty` (§7l).
    k = prepare(authored_scan_reactant_shared)
    xs = [1.0, 2.0, 3.0]
    for amounts in (Float64[], [0.5, 0.25])
        traced = (Reactant.to_rarray(xs), Reactant.to_rarray(amounts))
        compiled = Reactant.@compile k(traced...)
        @test Array(compiled(traced...)) ≈ k(xs, amounts)
        hlo = repr(Reactant.@code_hlo optimize = false k(traced...))
        @test count("stablehlo.while", hlo) == 1
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

@kernel authored_scan_reactant_turnover_init(pd, conc_mid::Vector{Float64}, dts::Vector{Float64}) = begin
    kin = pd.baseline * pd.kout
    trajectory = scan(conc_mid, dts, Ref(pd), Ref(kin);
            init = pd.baseline, include_init = true) do previous, concentration, dt, parameters, input_rate
        c2 = parameters.kout * (1 + concentration /
            (parameters.theta1 * concentration + parameters.theta2))
        steady = input_rate / c2
        next = (previous - steady) * exp(-c2 * dt) + steady
        (next, next)
    end
    return trajectory
end

@testset "an init-including scan keeps one while loop and one n + 1 buffer" begin
    pd = (; baseline = 100.0, kout = 0.3, theta1 = 1.2, theta2 = 40.0)
    traced_pd = Reactant.to_rarray(pd; track_numbers = true)
    included = prepare(authored_scan_reactant_turnover_init)
    concatenated = prepare(authored_scan_reactant_turnover)
    traced_sizes, bound_sizes = Int[], Int[]
    for n in (0, 1, 4, 8)
        conc, dts = abs.(sin.(1:n)) .* 50, fill(0.1, n)
        expected = concatenated(pd, conc, dts)
        @test included(pd, conc, dts) == expected
        traced = (Reactant.to_rarray(conc), Reactant.to_rarray(dts))
        compiled = Reactant.@compile included(traced_pd, traced...)
        result = Array(compiled(traced_pd, traced...))
        @test length(result) == n + 1
        @test result ≈ expected
        # Changed inputs reuse the executable.
        changed = (Reactant.to_rarray(conc .* 2), Reactant.to_rarray(dts))
        @test Array(compiled(traced_pd, changed...)) ≈ included(pd, conc .* 2, dts)
        hlo = repr(Reactant.@code_hlo optimize = false included(traced_pd, traced...))
        @test count("stablehlo.while", hlo) == (n == 0 ? 0 : 1)
        # Bound host sequences are lifted into the program as constants.
        kb = prepare(authored_scan_reactant_turnover_init; bound = (; conc_mid = conc, dts))
        compiled_bound = Reactant.@compile kb(traced_pd)
        @test Array(compiled_bound(traced_pd)) ≈ expected
        bound_hlo = repr(Reactant.@code_hlo optimize = false kb(traced_pd))
        @test count("stablehlo.while", bound_hlo) == (n == 0 ? 0 : 1)
        if n >= 4
            push!(traced_sizes, count("\n", hlo))
            push!(bound_sizes, count("\n", bound_hlo))
        end
    end
    # The program is independent of the sequence length.
    @test allequal(traced_sizes)
    @test allequal(bound_sizes)
    # The concatenated spelling keeps a second buffer the included one does not.
    conc, dts = abs.(sin.(1:8)) .* 50, fill(0.1, 8)
    traced = (Reactant.to_rarray(conc), Reactant.to_rarray(dts))
    included_hlo = repr(Reactant.@code_hlo optimize = false included(traced_pd, traced...))
    concatenated_hlo = repr(Reactant.@code_hlo optimize = false concatenated(traced_pd, traced...))
    @test !occursin("stablehlo.concatenate", included_hlo)
    @test occursin("stablehlo.concatenate", concatenated_hlo)

    # Reverse gradients through the seed and the steps match native Enzyme.
    @kernel included_decay(x::Vector{Float64}, seed::Float64) = begin
        path = scan(x; init = seed, include_init = true) do carry, v
            next = 0.5 * carry + v
            (next, next)
        end
        total::Float64 = sum(abs2, path)
        return total
    end
    x, seed = [0.5, -1.0, 0.25, 2.0], 1.5
    for active in (:x, :seed)
        prepared = prepare_ad(included_decay, AutoEnzyme(; mode = Enzyme.Reverse),
                              x, seed; active, want = :total)
        vref, gref = ad_value_and_gradient(prepared, x, seed)
        traced_args = (Reactant.to_rarray(x), Reactant.to_rarray(seed; track_numbers = true))
        compiled_both = compile_ad_value_and_gradient(prepared, traced_args...)
        value, gradient = compiled_both(traced_args...)
        @test Float64(value) ≈ vref
        host_gradient = gradient isa Reactant.AbstractConcreteArray ? Array(gradient) :
            Reactant.to_number(gradient)
        @test host_gradient ≈ gref
    end

    # The seed is element 1 of the scalar buffer: a compound seed is refused
    # loudly under Reactant (natively it is the vcat's element type).
    @kernel included_named_carry(x) = begin
        path = scan(x; init = (; total = 0.0), include_init = true) do carry, v
            next = carry.total + v
            ((; total = next), next)
        end
        return path
    end
    @test prepare(included_named_carry)([1.0, 2.0]) == Any[(; total = 0.0), 1.0, 3.0]
    @test_throws "only for a scalar carry seed" Reactant.@compile prepare(
        included_named_carry)(Reactant.to_rarray([1.0, 2.0]))
end
