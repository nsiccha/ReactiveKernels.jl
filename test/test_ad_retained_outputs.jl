module ADRetainedOutputsTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# `prepare_ad(...; retain)`: one reverse sweep returns the scalar objective,
# its gradient, and the retained WANT values the same primal sweep computed.

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

@kernel scored(q::Vector{Float64}; data::Vector{Float64}) = begin
    terms = plate(q, data) do qi, di
        term::Float64 = -(di - qi)^2 / 2
        return term
    end
    penalty::Float64 = sum(abs2, q) / 10
    density::Float64 = sum(terms) - penalty
end

expected_terms(q, data) = -(data .- q) .^ 2 ./ 2
expected_density(q, data) = sum(expected_terms(q, data)) - sum(abs2, q) / 10
expected_gradient(q, data) = (data .- q) .- q ./ 5

# A backend that `ad_retains_primal_sweep` does not declare.
struct UndeclaredBackend <: DifferentiationInterface.AbstractADType end

@testset "retained WANT outputs from one reverse sweep" begin
    data = [2.0, -1.0, 0.5, 1.5]
    q0 = [0.3, -0.4, 0.2, 0.0]

    @testset "authored KernelSpec" begin
        prepared = prepare_ad(scored, BACKEND, q0; data, active = :q,
                              want = :density, retain = (:terms, :penalty))
        @test prepared isa PreparedADKernel
        destination = fill(NaN, length(q0))
        previous = nothing
        for (q, d) in ((q0, data), ([0.1, 0.2, -0.3, 0.9], [1.0, 3.0, -2.0, 0.5]))
            value, gradient, retained = ad_value_gradient_and_retained!(
                prepared, destination, q; data = d)
            @test gradient === destination
            @test value ≈ expected_density(q, d)
            @test gradient ≈ expected_gradient(q, d)
            @test keys(retained) == (:terms, :penalty)
            @test retained.terms ≈ expected_terms(q, d)
            @test retained.penalty ≈ sum(abs2, q) / 10
            # A later call stores fresh values and leaves earlier ones intact.
            if previous !== nothing
                snapshot, stored = previous
                @test stored.terms == snapshot
            end
            previous = (copy(retained.terms), retained)

            value_o, gradient_o, retained_o =
                ad_value_gradient_and_retained(prepared, q; data = d)
            @test value_o == value
            @test gradient_o ≈ gradient
            @test retained_o.terms == retained.terms

            # The retaining preparation still answers the ordinary call.
            value_p, _ = ad_value_and_gradient!(prepared, destination, q; data = d)
            @test value_p == value
            @test destination ≈ expected_gradient(q, d)
        end
    end

    @testset "low-level PreparedKernel with several WANT ports" begin
        kernel = prepare(scored; have = (:q, :data),
                         want = (:terms, :density))
        prepared = prepare_ad(kernel, BACKEND, q0, data; active = :q,
                              retain = (:terms,))
        gradient = similar(q0)
        value, _, retained = ad_value_gradient_and_retained!(
            prepared, gradient, q0, data)
        @test value ≈ expected_density(q0, data)
        @test gradient ≈ expected_gradient(q0, data)
        @test retained.terms ≈ expected_terms(q0, data)
        @test retained.terms == kernel(q0, data)[1]
    end

    @testset "invalid retention fails loudly" begin
        # refused: the retained name must be a WANT port of the kernel, or the
        # objective would be ambiguous.
        kernel = prepare(scored; have = (:q, :data), want = (:terms, :density))
        @test_throws ArgumentError prepare_ad(kernel, BACKEND, q0, data;
                                              active = :q, retain = (:missing,))
        # refused: retaining every WANT leaves no objective to differentiate.
        @test_throws ArgumentError prepare_ad(kernel, BACKEND, q0, data;
                                              active = :q,
                                              retain = (:terms, :density))
        # refused: a backend not declared by `ad_retains_primal_sweep` may
        # evaluate perturbed points or restore writes, so its retained values
        # could be silently wrong (dev §1: no silent wrong results).
        @test !ad_retains_primal_sweep(UndeclaredBackend())
        @test ad_retains_primal_sweep(BACKEND)
        @test_throws ArgumentError prepare_ad(kernel, UndeclaredBackend(),
                                              q0, data; active = :q,
                                              retain = (:terms,))
        # refused: nothing is retained without `retain`, so there is no
        # retained value to return.
        plain = prepare_ad(scored, BACKEND, q0; data, active = :q,
                           want = :density)
        @test_throws ArgumentError ad_value_gradient_and_retained!(
            plain, similar(q0), q0; data)
    end
end

end # module
