using LinearAlgebra
using ReactiveKernels
using Reactant
using Test
import Reactant: @compile

if !isdefined(@__MODULE__, :ReactiveHMCExamples)
    include(joinpath(@__DIR__, "..", "examples", "preexisting_reactivehmc.jl"))
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
    end
end
