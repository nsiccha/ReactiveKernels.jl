using ReactiveKernels
using Reactant
using Test
import Enzyme
using DifferentiationInterface: AutoEnzyme
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    build_eight_schools_graph, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA

# Reactant-compiled AD is the AD analog of the primal Reactant path: it takes a
# native `PreparedADKernel` (which owns the scalar-WANT / single-active-port
# validation and the authored-order reorder) and compiles a
# DifferentiationInterface gradient / value-and-gradient through Reactant. The
# differentiation engine stays the caller's DI backend — `AutoEnzyme` here —
# exactly as on the native reverse pass, and the compiled result must match the
# native gradient bit-for-bit.

const AD_REACTANT_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
_trace(x) = Reactant.to_rarray(x)

@testset "Reactant-compiled AD" begin
    @testset "standalone objective: authored defaults, keywords, parity" begin
        @kernel objective(q::Vector{Float64}, scale::Float64 = 1.25;
                          data::Vector{Float64}, offset::Float64 = 0.0) = begin
            density::Float64 = sum(q .* data) - scale * sum(abs2, q) + offset
        end

        q = [0.3, -0.4, 0.2]
        data = [2.0, -1.0, 0.5]
        prepared = prepare_ad(
            objective, AD_REACTANT_BACKEND, q; data, active = :q, want = :density)

        gref = similar(q)
        vref, gref = ad_value_and_gradient!(prepared, gref, q; data)

        # The selected HAVE boundary in authored order — supply each traced.
        names = Tuple(input.name for input in inputs(prepared))
        host = map(names) do name
            name === :q ? q :
            name === :scale ? 1.25 :
            name === :data ? data : 0.0
        end
        traced = map(_trace, host)

        compiled_gradient = compile_ad_gradient(prepared, traced...)
        gradient = Array(compiled_gradient(traced...))
        @test gradient ≈ gref
        @test gradient ≈ data .- 2(1.25) .* q

        compiled_both = compile_ad_value_and_gradient(prepared, traced...)
        value, both_gradient = compiled_both(traced...)
        @test Float64(value) ≈ vref
        @test Array(both_gradient) ≈ gref

        # A host (non-traced) active argument, or an arity mismatch, is a clear
        # ArgumentError rather than a bare MethodError.
        @test_throws ArgumentError compile_ad_gradient(prepared, q)
        @test_throws ArgumentError compile_ad_gradient(prepared, _trace(q), _trace(data))
    end

    @testset "Eight Schools supported boundaries match native reverse pass" begin
        model = build_eight_schools_graph()
        observations = Float64.(EIGHT_SCHOOLS_Y)
        observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
        theta = 0.25 .* collect(1.0:8.0)
        unconstrained = [1.5, log(2.0), theta...]

        boundaries = (
            (name = "minimal_likelihood/likelihood",
             have = (:θ, :observations, :observation_scales),
             want = :likelihood, active = :θ,
             args = (theta, observations, observation_scales)),
            (name = "packed_unconstrained/likelihood",
             have = (:unconstrained, :observations, :observation_scales),
             want = :likelihood, active = :unconstrained,
             args = (unconstrained, observations, observation_scales)),
        )

        for boundary in boundaries
            kernel = prepare(model; have = boundary.have, want = boundary.want)
            prepared = prepare_ad(
                kernel, AD_REACTANT_BACKEND, boundary.args...;
                active = boundary.active)

            reference = similar(boundary.args[1])
            value, reference =
                ad_value_and_gradient!(prepared, reference, boundary.args...)

            traced = map(_trace, boundary.args)
            gradient = Array(compile_ad_gradient(prepared, traced...)(traced...))
            @test gradient ≈ reference

            compiled_value, compiled_gradient =
                compile_ad_value_and_gradient(prepared, traced...)(traced...)
            @test Float64(compiled_value) ≈ value
            @test Array(compiled_gradient) ≈ reference
        end
    end

    @testset "primal-uncompilable boundary propagates the @compile error" begin
        # packed_unconstrained/{joint,prior} fail primal Reactant compilation with
        # "Scalar indexing is disallowed." (see the eight-schools-reactant receipt);
        # the gradient must surface that error unchanged rather than degrade or hide.
        model = build_eight_schools_graph()
        observations = Float64.(EIGHT_SCHOOLS_Y)
        observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
        theta = 0.25 .* collect(1.0:8.0)
        unconstrained = [1.5, log(2.0), theta...]

        kernel = prepare(model;
            have = (:unconstrained, :observations, :observation_scales),
            want = :posterior)
        prepared = prepare_ad(kernel, AD_REACTANT_BACKEND,
            unconstrained, observations, observation_scales;
            active = :unconstrained)
        traced = map(_trace, (unconstrained, observations, observation_scales))
        @test_throws Exception compile_ad_gradient(prepared, traced...)
    end
end
