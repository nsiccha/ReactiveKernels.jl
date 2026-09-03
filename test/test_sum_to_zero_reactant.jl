using ReactiveKernels
using Reactant
using Reactant: @compile
using Test
using ReactiveKernelsPPLExamples.SumToZeroExample:
    evaluate_sum_to_zero_source

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
