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

    @testset "partially-evaluated AD kernels compile over the remaining ports" begin
        @kernel bound_objective(q::Vector{Float64}, data::Vector{Float64}) = begin
            shifted = data .- 1.0
            density::Float64 = sum(q .* shifted) - 0.5 * sum(abs2, q)
        end

        q = [0.3, -0.4, 0.2]
        data = [2.0, -1.0, 0.5]
        prepared = prepare_ad(
            bound_objective, AD_REACTANT_BACKEND, q;
            active = :q, want = :density, bound = (; data))
        @test Tuple(input.name for input in inputs(prepared.kernel)) == (:q,)

        gref = similar(q)
        vref, gref = ad_value_and_gradient!(prepared, gref, q)
        @test gref ≈ (data .- 1.0) .- q

        traced_q = _trace(q)
        compiled_gradient = compile_ad_gradient(prepared, traced_q)
        @test Array(compiled_gradient(traced_q)) ≈ gref

        compiled_both = compile_ad_value_and_gradient(prepared, traced_q)
        value, both_gradient = compiled_both(traced_q)
        @test Float64(value) ≈ vref
        @test Array(both_gradient) ≈ gref

        # The benchmark-facing compiler ABI exposes the same bound array as an
        # explicitly transferred inactive operand, avoiding a compiler literal
        # while retaining the q-only public PreparedADKernel above.
        _, bound_arrays =
            ReactiveKernels._externalize_bound_arrays(prepared.kernel)
        traced_bound = map(_trace, bound_arrays)
        compiled_externalized =
            ReactiveKernels._reactant_compile_ad_externalized(
                Val(:value_and_gradient), prepared,
                (traced_q,), traced_bound; sync = true)
        external_value, external_gradient =
            compiled_externalized(traced_q, traced_bound...)
        @test Float64(external_value) ≈ vref
        @test Array(external_gradient) ≈ gref
    end

    @testset "Eight Schools supported boundaries match native reverse pass" begin
        model = build_eight_schools_graph()
        observations = Float64.(EIGHT_SCHOOLS_Y)
        observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
        theta = 0.25 .* collect(1.0:8.0)
        unconstrained = [1.5, log(2.0), theta...]

        boundaries = (
            (name = "packed_unconstrained/joint",
             have = (:unconstrained, :observations, :observation_scales),
             want = :posterior, active = :unconstrained,
             args = (unconstrained, observations, observation_scales)),
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

            # Host preparation deliberately freezes the fast native-only AD
            # body. Reactant compilation must reconstruct the full kernel
            # selector instead of tracing this call, so the tensorized body is
            # selected once the arguments below become Reactant values.
            @test prepared.call isa ReactiveKernels._ADNativeKernelCall

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
end
