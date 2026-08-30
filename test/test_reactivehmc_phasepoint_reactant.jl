using LinearAlgebra
using ReactiveKernels
using Reactant
using Test
import Reactant: @compile

if !isdefined(@__MODULE__, :ReactiveHMCExamples)
    include(joinpath(@__DIR__, "..", "examples", "preexisting_reactivehmc.jl"))
end
if !isdefined(@__MODULE__, :ReactiveHMCIntegratorFixture)
    include(joinpath(@__DIR__, "..", "benchmark",
                     "reactivehmc_integrator_kernel_fixture.jl"))
end

_rhmc_potential(position) = sum(abs2, position) / 2
_rhmc_gradient(position) = (_rhmc_potential(position), position)
_rhmc_metric(position) =
    (_rhmc_gradient(position)..., Diagonal(1 .+ abs2.(position)))
function _rhmc_metric_gradient(position)
    pot, dpot, metric = _rhmc_metric(position)
    dimension = length(position)
    dimension == 2 || error("phase-point acceptance callback is pinned to dimension two")
    unit = [1.0 0.0; 0.0 1.0]
    derivative = reshape(unit, dimension, dimension, 1) .*
        reshape(unit, dimension, 1, dimension) .*
        reshape(2 .* position, 1, 1, dimension)
    (pot, dpot, metric, derivative)
end

_rhmc_trace_source(value::AbstractVector) = Reactant.to_rarray(value)
_rhmc_trace_source(value::Diagonal) =
    Diagonal(Reactant.to_rarray(value.diag))
_rhmc_trace_source(value) = value

@testset "all six ReactiveHMC phase-point kernels compile generically" begin
    position_host = [0.25, -0.5]
    momentum_host = [0.4, 0.1]
    metric_host = Diagonal(ones(2))
    constructors = (
        () -> ReactiveHMCExamples.euclidean_phasepoint_kernels(
            Val(:gaussian), _rhmc_potential, _rhmc_gradient,
            metric_host, position_host, momentum_host),
        () -> ReactiveHMCExamples.riemannian_phasepoint_kernels(
            Val(:gaussian), _rhmc_potential, _rhmc_gradient,
            _rhmc_metric, _rhmc_metric_gradient,
            position_host, momentum_host),
        () -> ReactiveHMCExamples.softabs_phasepoint_kernels(
            Val(:gaussian), _rhmc_potential, _rhmc_gradient,
            _rhmc_metric, _rhmc_metric_gradient,
            position_host, momentum_host),
        () -> ReactiveHMCExamples.euclidean_phasepoint_kernels(
            Val(:relativistic), _rhmc_potential, _rhmc_gradient,
            metric_host, position_host, momentum_host;
            speed=1.5, mass=0.8),
        () -> ReactiveHMCExamples.riemannian_phasepoint_kernels(
            Val(:relativistic), _rhmc_potential, _rhmc_gradient,
            _rhmc_metric, _rhmc_metric_gradient,
            position_host, momentum_host;
            speed=1.5, mass=0.8),
        () -> ReactiveHMCExamples.softabs_phasepoint_kernels(
            Val(:relativistic), _rhmc_potential, _rhmc_gradient,
            _rhmc_metric, _rhmc_metric_gradient,
            position_host, momentum_host;
            speed=1.5, mass=0.8),
    )
    position = Reactant.to_rarray(position_host)
    momentum = Reactant.to_rarray(momentum_host)
    for construct in constructors
        kernels = construct()
        geometry_f = kernels.geometry
        dpos_f = kernels.dham_dpos
        dmom_f = kernels.dham_dmom
        hamiltonian_f = kernels.hamiltonian
        expected_geometry = geometry_f(position_host)
        expected = (hamiltonian_f(expected_geometry, momentum_host),
                    dpos_f(expected_geometry, momentum_host),
                    dmom_f(expected_geometry, momentum_host))

        compiled_geometry = @compile geometry_f(position)
        actual_geometry = compiled_geometry(position)
        compiled_hamiltonian = @compile hamiltonian_f(actual_geometry, momentum)
        compiled_dpos = @compile dpos_f(actual_geometry, momentum)
        compiled_dmom = @compile dmom_f(actual_geometry, momentum)
        @test compiled_hamiltonian(actual_geometry, momentum) ≈ expected[1]
        @test Array(compiled_dpos(actual_geometry, momentum)) ≈ expected[2]
        @test Array(compiled_dmom(actual_geometry, momentum)) ≈ expected[3]

        wanted = (:pot, :dham_dpos, :dham_dmom, :ham)
        endpoint = prepare(
            kernels.spec; have = propertynames(kernels.sources), want = wanted,
        )
        host_sources = values(kernels.sources)
        traced_sources = map(_rhmc_trace_source, host_sources)
        expected_endpoint = endpoint(host_sources...)
        compiled_endpoint = @compile endpoint(traced_sources...)
        actual_endpoint = compiled_endpoint(traced_sources...)
        @test actual_endpoint[1] ≈ expected_endpoint[1]
        @test Array(actual_endpoint[2]) ≈ expected_endpoint[2]
        @test Array(actual_endpoint[3]) ≈ expected_endpoint[3]
        @test actual_endpoint[4] ≈ expected_endpoint[4]

        transition = compile_state_transition(
            kernels.spec,
            partial(ReactiveHMCIntegratorFixture.generalized_leapfrog!;
                    stepsize=0.06, n_fi_steps=2),
            host_sources,
        )
        host_state = initial_state(transition)
        traced_state = map(_rhmc_trace_source, host_state)
        expected_transition = transition(host_state)
        compiled_transition = @compile transition(traced_state)
        actual_transition = compiled_transition(traced_state)
        @test Array(actual_transition.pos) ≈ expected_transition.pos
        @test Array(actual_transition.mom) ≈ expected_transition.mom
        @test actual_transition.ham ≈ expected_transition.ham
    end
end
