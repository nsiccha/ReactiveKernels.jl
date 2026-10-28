using ReactiveKernels
using Reactant
using Reactant: @compile
using Test
import Enzyme
using DifferentiationInterface: AutoEnzyme
using ReactiveKernelsPPLExamples.SumToZeroExample:
    evaluate_sum_to_zero_source, build_sum_to_zero_graph
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA

_sum_to_zero_host(value::Reactant.AbstractConcreteArray) = Array(value)
_sum_to_zero_host(value::Reactant.AbstractConcreteNumber) =
    Reactant.to_number(value)
_sum_to_zero_host(value::NamedTuple) =
    NamedTuple{keys(value)}(map(_sum_to_zero_host, values(value)))
_sum_to_zero_host(value::Tuple) = map(_sum_to_zero_host, value)
_sum_to_zero_host(value) = value

@testset "sum-to-zero docs kernels compile through Reactant" begin
    artifact = evaluate_sum_to_zero_source()
    inputs = Tuple(artifact.inputs)
    traced_inputs = map(inputs) do value
        value isa AbstractArray ? Reactant.to_rarray(value) :
        Reactant.to_rarray(value; track_numbers = true)
    end

    compiled_hot = @compile sync = true artifact.kernel(traced_inputs...)
    compiled_output = compiled_hot(traced_inputs...)
    actual_parameters, actual_prior, actual_likelihood =
        _sum_to_zero_host(compiled_output)
    expected_parameters, expected_prior, expected_likelihood = artifact.output
    @test actual_parameters.α_s2z ≈ expected_parameters.α_s2z
    @test actual_parameters.τ ≈ expected_parameters.τ
    @test actual_parameters.effects_s2z ≈ expected_parameters.effects_s2z
    @test actual_prior ≈ expected_prior
    @test actual_likelihood ≈ expected_likelihood

    parameters = first(compiled_output)
    recovery_inputs = (
        parameters,
        Reactant.to_rarray(
            artifact.recovery_inputs.α_prior_sd; track_numbers = true),
        Reactant.to_rarray(
            artifact.recovery_inputs.reconstruction_innovation;
            track_numbers = true),
    )
    compiled_recovery = @compile sync = true artifact.recovery_kernel(
        recovery_inputs...)
    compiled_recovered = _sum_to_zero_host(compiled_recovery(recovery_inputs...))
    @test compiled_recovered.α_bayes ≈ artifact.recovered.α_bayes
    @test compiled_recovered.effects_bayes ≈ artifact.recovered.effects_bayes
    @test compiled_recovered.realized_effect_mean ≈
        artifact.recovered.realized_effect_mean
end

# The sampler-facing bound posterior is the matched benchmark cell.  Its
# K-lane plates keep the batched lowering whatever K is (a lane count is a
# data length; docs/src/constraints.md), so the emitted program must not grow
# with the number of schools, and the automatic AD compile must keep the
# observation arrays embedded rather than as hidden runtime operands.
@testset "bound sum-to-zero posterior lowers to a lane-count-independent program" begin
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
    observations = Float64.(EIGHT_SCHOOLS_Y)
    observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
    α_prior_sd = 5.0
    q = [0.5, log(2.0), (0.25 .* collect(1.0:7.0))...]
    kernel = prepare(
        build_sum_to_zero_graph();
        have = (:unconstrained, :observations, :observation_scales, :α_prior_sd),
        want = :posterior,
        bound = (; observations, observation_scales, α_prior_sd),
    )
    q_traced = Reactant.to_rarray(q)

    compiled = @compile sync = true kernel(q_traced)
    @test Float64(compiled(q_traced)) ≈ kernel(q) atol = 1e-10

    # Twice the schools: identical unoptimized program, only tensor shapes
    # differ.
    program_lines(K) = begin
        obs = vcat(observations, observations)[1:K]
        scales = vcat(observation_scales, observation_scales)[1:K]
        wide = prepare(
            build_sum_to_zero_graph();
            have = (:unconstrained, :observations, :observation_scales, :α_prior_sd),
            want = :posterior,
            bound = (; observations = obs, observation_scales = scales, α_prior_sd),
        )
        q_wide = Reactant.to_rarray([0.5, log(2.0), (0.25 .* collect(1.0:(K - 1)))...])
        count("\n", repr(Reactant.@code_hlo optimize = false wide(q_wide)))
    end
    @test program_lines(8) == program_lines(16)

    backend = AutoEnzyme(
        ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
    prepared = prepare_ad(kernel, backend, q; active = :unconstrained)
    native_value, native_gradient = ad_value_and_gradient!(prepared, similar(q), q)
    compiled_ad = compile_ad_value_and_gradient(prepared, q_traced)
    @test !(compiled_ad isa ext._ExternalizedADExecutable)
    value, gradient = compiled_ad(q_traced)
    @test Float64(value) ≈ native_value atol = 1e-10
    @test Array(gradient) ≈ native_gradient atol = 1e-10
end
