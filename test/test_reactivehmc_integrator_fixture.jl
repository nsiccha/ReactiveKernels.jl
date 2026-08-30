using LinearAlgebra
using ReactiveKernels
using Test
import TOML

if !isdefined(@__MODULE__, :ReactiveHMCExamples)
    include(joinpath(@__DIR__, "..", "examples", "preexisting_reactivehmc.jl"))
end
include(joinpath(@__DIR__, "..", "benchmark",
                 "reactivehmc_integrator_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "receipts",
                 "validate_reactivehmc_integrators.jl"))

const RHMC_INTEGRATORS = ReactiveHMCIntegratorFixture

function _integrator_observables(point, kernels)
    (; pos=point.pos, mom=point.mom, ham=point.ham,
       dham_dpos=kernels.dham_dpos(point.geometry, point.mom),
       dham_dmom=kernels.dham_dmom(point.geometry, point.mom))
end

@testset "ReactiveHMC integrator source and independent receipt" begin
    generalized = ReactiveKernels.kernel_registration(
        RHMC_INTEGRATORS.generalized_leapfrog!)
    midpoint = ReactiveKernels.kernel_registration(
        RHMC_INTEGRATORS.implicit_midpoint!)
    @test (generalized.kind, generalized.subject, generalized.write_roots) ==
          (:free_method, :phasepoint, (:mom, :pos))
    @test (midpoint.kind, midpoint.subject, midpoint.write_roots) ==
          (:free_method, :phasepoint, (:pos, :mom))
    @test ReactiveKernels.kernel_port_names(
        RHMC_INTEGRATORS.generalized_leapfrog!) ==
        (:phasepoint, :stepsize, :n_fi_steps)
    @test ReactiveKernels.kernel_port_names(
        RHMC_INTEGRATORS.implicit_midpoint!) ==
        (:phasepoint, :stepsize, :n_fi_steps)

    source_path = joinpath(@__DIR__, "..", "benchmark",
                           "reactivehmc_integrator_kernel_fixture.jl")
    source = read(source_path, String)
    @test occursin("pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))", source)
    @test occursin("(; dham_dmom, dham_dpos) = phasepoint", source)
    @test occursin("f(args...; stepsize=stepsize / n_steps, kwargs...)", source)

    receipt_path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                            "reactivehmc-integrators-ca9-v1.toml")
    @test isempty(validate_reactivehmc_integrator_receipt(receipt_path))
    receipt = TOML.parsefile(receipt_path)
    source_digests = Dict(ReactiveHMCAlgorithmCorpus.UPSTREAM.source_sha256)
    @test receipt["pins"]["reactivehmc_revision"] ==
          ReactiveHMCAlgorithmCorpus.UPSTREAM.revision
    @test receipt["pins"]["integrators_sha256"] ==
          source_digests["src/integrators.jl"]
    @test receipt["pins"]["phasepoints_sha256"] ==
          source_digests["src/phasepoints.jl"]

    euclidean = ReactiveHMCExamples.euclidean_examples()
    riemannian = ReactiveHMCExamples.riemannian_examples()
    multistep = ReactiveHMCExamples.multistep!(
        ReactiveHMCExamples.generalized_leapfrog!, [0.25, -0.5], [0.4, 0.1],
        riemannian.gaussian;
        stepsize=0.06, n_fi_steps=2, n_steps=3)
    actual = Dict(
        "leapfrog" => _integrator_observables(
            euclidean.euclidean_phasepoint, euclidean.gaussian),
        "generalized_leapfrog" =>
            _integrator_observables(
                riemannian.generalized_leapfrog, riemannian.gaussian),
        "implicit_midpoint" =>
            _integrator_observables(
                riemannian.implicit_midpoint, riemannian.gaussian),
        "multistep" => _integrator_observables(multistep, riemannian.gaussian),
    )
    for expected in receipt["cases"]
        observed = actual[expected["name"]]
        for field in (:pos, :mom, :ham, :dham_dpos, :dham_dmom)
            @test isapprox(getproperty(observed, field), expected[string(field)];
                           rtol=2e-13, atol=2e-15)
        end
    end
end
