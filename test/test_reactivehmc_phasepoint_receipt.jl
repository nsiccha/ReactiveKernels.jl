using ReactiveKernels
using ReactiveKernelsCompatibilityExamples: ReactiveHMCExamples
using Test
import TOML

include(joinpath(@__DIR__, "..", "benchmark", "receipts",
                 "validate_reactivehmc_phasepoints.jl"))

function _phasepoint_observables(kernels, position, momentum)
    geometry = kernels.geometry(position)
    pot = hasproperty(geometry, :pot) ? geometry.pot : sum(abs2, position) / 2
    (; pot,
       ham=kernels.hamiltonian(geometry, momentum),
       dham_dpos=kernels.dham_dpos(geometry, momentum),
       dham_dmom=kernels.dham_dmom(geometry, momentum))
end

function _phasepoint_spec_observables(kernels)
    wanted = (:pot, :dham_dpos, :dham_dmom, :ham)
    endpoint = prepare(
        kernels.spec; have = propertynames(kernels.sources), want = wanted,
    )
    NamedTuple{wanted}(endpoint(values(kernels.sources)...))
end

@testset "ReactiveHMC six phase-point variants against physical receipt" begin
    receipt_path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                            "reactivehmc-phasepoints-ca9-v1.toml")
    @test isempty(validate_reactivehmc_phasepoint_receipt(receipt_path))
    receipt = TOML.parsefile(receipt_path)
    source_digests = Dict(ReactiveHMCAlgorithmCorpus.UPSTREAM.source_sha256)
    @test receipt["pins"]["reactivehmc_revision"] ==
          ReactiveHMCAlgorithmCorpus.UPSTREAM.revision
    @test receipt["pins"]["phasepoints_sha256"] ==
          source_digests["src/phasepoints.jl"]

    position = receipt["inputs"]["position"]
    momentum = receipt["inputs"]["momentum"]
    euclidean = ReactiveHMCExamples.euclidean_examples()
    riemannian = ReactiveHMCExamples.riemannian_examples()
    softabs = ReactiveHMCExamples.softabs_examples()
    actual = Dict(
        "euclidean" => _phasepoint_observables(
            euclidean.gaussian, position, momentum),
        "riemannian" => _phasepoint_observables(
            riemannian.gaussian, position, momentum),
        "softabs" => _phasepoint_observables(
            softabs.gaussian, position, momentum),
        "relativistic_euclidean" => _phasepoint_observables(
            euclidean.relativistic, position, momentum),
        "relativistic_riemannian" => _phasepoint_observables(
            riemannian.relativistic, position, momentum),
        "relativistic_softabs" => _phasepoint_observables(
            softabs.relativistic, position, momentum),
    )
    direct = Dict(
        "euclidean" => _phasepoint_spec_observables(euclidean.gaussian),
        "riemannian" => _phasepoint_spec_observables(riemannian.gaussian),
        "softabs" => _phasepoint_spec_observables(softabs.gaussian),
        "relativistic_euclidean" =>
            _phasepoint_spec_observables(euclidean.relativistic),
        "relativistic_riemannian" =>
            _phasepoint_spec_observables(riemannian.relativistic),
        "relativistic_softabs" =>
            _phasepoint_spec_observables(softabs.relativistic),
    )
    for expected in receipt["cases"]
        observed = actual[expected["name"]]
        direct_observed = direct[expected["name"]]
        for field in (:pot, :ham, :dham_dpos, :dham_dmom)
            @test isapprox(getproperty(observed, field), expected[string(field)];
                           rtol=2e-13, atol=2e-15)
            @test isapprox(
                getproperty(direct_observed, field), expected[string(field)];
                rtol=2e-13, atol=2e-15,
            )
        end
    end
end
