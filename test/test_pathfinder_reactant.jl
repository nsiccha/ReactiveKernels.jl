using TOML

if !isdefined(Main, :PathfinderKernelAuthoringFixture)
    include(joinpath(@__DIR__, "..", "benchmark", "pathfinder_kernel_authoring_fixture.jl"))
end
using .PathfinderKernelAuthoringFixture

@testset "Pathfinder candidate compiles once and replays the path through Reactant" begin
    inputs = pathfinder_fixture_inputs()
    kernel = PATHFINDER_CANDIDATE
    tolerance = Reactant.to_rarray(
        inputs.curvature_tolerance; track_numbers = true)
    identity = Reactant.to_rarray(inputs.identity)

    first_step = inputs.positions[:, 2] .- inputs.positions[:, 1]
    first_delta = inputs.gradients[:, 1] .- inputs.gradients[:, 2]
    position = Reactant.to_rarray(inputs.positions[:, 2])
    gradient = Reactant.to_rarray(inputs.gradients[:, 2])
    alpha = Reactant.to_rarray(inputs.initial_alpha)
    step = Reactant.to_rarray(first_step)
    gradient_delta = Reactant.to_rarray(first_delta)
    elbo_noise = Reactant.to_rarray(inputs.elbo_standard_draws[:, :, 1])
    output_noise = Reactant.to_rarray(inputs.output_standard_draws[:, :, 1])

    compiled = @compile kernel(
        inputs.logdensity,
        position,
        gradient,
        alpha,
        step,
        gradient_delta,
        identity,
        elbo_noise,
        output_noise,
        tolerance,
    )

    native = run_pathfinder_fixture()
    compiled_candidates = NamedTuple[]
    for candidate_index in 1:3
        path_index = candidate_index + 1
        position = Reactant.to_rarray(inputs.positions[:, path_index])
        gradient = Reactant.to_rarray(inputs.gradients[:, path_index])
        step = Reactant.to_rarray(
            inputs.positions[:, path_index] .-
            inputs.positions[:, path_index - 1])
        gradient_delta = Reactant.to_rarray(
            inputs.gradients[:, path_index - 1] .-
            inputs.gradients[:, path_index])
        elbo_noise = Reactant.to_rarray(
            inputs.elbo_standard_draws[:, :, candidate_index])
        output_noise = Reactant.to_rarray(
            inputs.output_standard_draws[:, :, candidate_index])
        values = compiled(
            inputs.logdensity,
            position,
            gradient,
            alpha,
            step,
            gradient_delta,
            identity,
            elbo_noise,
            output_noise,
            tolerance,
        )
        candidate = NamedTuple{PATHFINDER_OUTPUTS}(values)
        push!(compiled_candidates, candidate)
        expected = native.candidates[candidate_index]
        @test Array(candidate.alpha_next) ≈ expected.alpha_next rtol=1e-12 atol=1e-12
        @test candidate.curvature_accepted == expected.curvature_accepted
        @test Array(candidate.covariance) ≈ expected.covariance rtol=1e-12 atol=1e-12
        @test Array(candidate.mean) ≈ expected.mean rtol=1e-12 atol=1e-12
        @test Array(candidate.elbo_draws) ≈ expected.elbo_draws rtol=1e-12 atol=1e-12
        @test Array(candidate.log_q) ≈ expected.log_q rtol=1e-12 atol=1e-12
        @test candidate.elbo ≈ expected.elbo rtol=1e-12 atol=1e-12
        @test Array(candidate.output_draws) ≈ expected.output_draws rtol=1e-12 atol=1e-12
        alpha = candidate.alpha_next
    end

    compiled_elbos = map(candidate -> Float64(candidate.elbo), compiled_candidates)
    compiled_best = argmax(compiled_elbos)
    @test compiled_best == native.best_index
    @test Array(compiled_candidates[compiled_best].output_draws) ≈ native.draws
end
