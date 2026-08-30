using ReactiveKernels
using Test

if !isdefined(@__MODULE__, :ReactiveHMCExamples)
    include(joinpath(@__DIR__, "..", "examples", "preexisting_reactivehmc.jl"))
end
if !isdefined(@__MODULE__, :ReactiveHMCIntegratorFixture)
    include(joinpath(@__DIR__, "..", "benchmark",
                     "reactivehmc_integrator_kernel_fixture.jl"))
end

@testset "all six endpoints drive source-authored static-loop integrators" begin
    euclidean = ReactiveHMCExamples.euclidean_examples()
    riemannian = ReactiveHMCExamples.riemannian_examples()
    softabs = ReactiveHMCExamples.softabs_examples()
    cases = (
        euclidean.gaussian,
        euclidean.relativistic,
        riemannian.gaussian,
        riemannian.relativistic,
        softabs.gaussian,
        softabs.relativistic,
    )
    stepsize = 0.06
    n_fi_steps = 2

    for kernels in cases
        for (source, oracle) in (
                ReactiveHMCIntegratorFixture.generalized_leapfrog! =>
                    ReactiveHMCExamples.generalized_leapfrog!,
                ReactiveHMCIntegratorFixture.implicit_midpoint! =>
                    ReactiveHMCExamples.implicit_midpoint!,
            )
            transition = compile_state_transition(
                kernels.spec,
                partial(source; stepsize, n_fi_steps),
                values(kernels.sources),
            )
            state = initial_transition_state(transition)
            independent = initial_transition_state(transition)
            groups = typeof(transition).parameters[2]
            for group in groups
                leader = first(group)
                for alias in Base.tail(group)
                    @test getfield(state, alias) === getfield(state, leader)
                    @test getfield(independent, alias) ===
                          getfield(independent, leader)
                end
            end
            for name in propertynames(state)
                value = getfield(state, name)
                value isa AbstractArray && @test value !== getfield(independent, name)
            end
            for name in propertynames(kernels.sources)
                endswith(String(name), "_f") || continue
                @test getfield(state, name) ===
                      getfield(kernels.sources, name)
                @test getfield(independent, name) ===
                      getfield(kernels.sources, name)
            end
            actual = transition(state)
            expected = oracle(
                copy(kernels.sources.pos), copy(kernels.sources.mom), kernels;
                stepsize, n_fi_steps,
            )
            @test actual.pos ≈ expected.pos atol=2e-15 rtol=2e-13
            @test actual.mom ≈ expected.mom atol=2e-15 rtol=2e-13
            @test actual.ham ≈ expected.ham atol=2e-15 rtol=2e-13
        end
    end
end

@testset "all six phase-point specs expose a generic integrator endpoint" begin
    euclidean = ReactiveHMCExamples.euclidean_examples()
    riemannian = ReactiveHMCExamples.riemannian_examples()
    softabs = ReactiveHMCExamples.softabs_examples()
    cases = (
        (euclidean.gaussian, (:pot_f, :grad_f)),
        (euclidean.relativistic, (:pot_f, :grad_f)),
        (riemannian.gaussian,
         (:pot_f, :grad_f, :metric_f, :metric_grad_f, :metric_inverse_f)),
        (riemannian.relativistic,
         (:pot_f, :grad_f, :metric_f, :metric_grad_f, :metric_inverse_f)),
        (softabs.gaussian,
         (:pot_f, :grad_f, :premetric_f, :premetric_grad_f,
          :softabs_geometry_f)),
        (softabs.relativistic,
         (:pot_f, :grad_f, :premetric_f, :premetric_grad_f,
          :softabs_geometry_f)),
    )
    registration = ReactiveKernels.kernel_registration(
        ReactiveHMCIntegratorFixture.generalized_leapfrog!,
    )
    required = Set((:pos, :mom, :dham_dpos, :dham_dmom, :ham))

    for (kernels, expected_authorities) in cases
        prepared = ReactiveKernels._prepare_factory(kernels.spec, registration)
        plan = ReactiveKernels.kernel_prepared_plan(prepared)
        slots = ReactiveKernels.kernel_plan_slots(plan)
        names = Set(slot.path[1] for slot in slots)
        external = Set(ReactiveKernels.kernel_prepared_external(prepared))
        authority_names = Set(
            slot.path[1] for slot in slots if slot.canon in external
        )

        @test required ⊆ names
        @test authority_names == Set(expected_authorities)
        @test ReactiveKernels.kernel_prepared_grad_recipe(prepared) == 0
        @test !isempty(ReactiveKernels.kernel_prepared_handles(prepared))
    end
end
