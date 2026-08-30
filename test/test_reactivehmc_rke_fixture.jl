using ReactiveKernels
using Test
import TOML

include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_rke_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "receipts", "validate_reactivehmc_rke.jl"))

const RHMC_RKE = ReactiveHMCRKEFixture

@testset "ReactiveHMC RKE source and independent receipt" begin
    registration = ReactiveKernels.kernel_registration(RHMC_RKE.rke)
    @test registration.kind == :object_kernel
    @test isempty(registration.write_roots)
    @test ReactiveKernels.kernel_port_names(RHMC_RKE.rke) ==
          (:lambertw, :m, :c, :c1, :c2, :P0_sq)
    @test map(ir -> ir.id.name, ReactiveKernels.method_irs(RHMC_RKE.rke)) ==
          (:e_sq, :p_sq, :P_sq, :cdf_sq, :quantile_sq)

    source_path = joinpath(@__DIR__, "..", "benchmark", "reactivehmc_rke_kernel_fixture.jl")
    source = read(source_path, String)
    @test occursin("@kernel rke(lambertw; m=1.0, c=1.0)", source)
    @test occursin("p_sq(x_sq) = exp(-e_sq(__self__, x_sq))", source)
    @test occursin("cdf_sq(x_sq) = (P0_sq - P_sq(__self__, x_sq)) / P0_sq", source)
    @test count("lambertw(", source) == 2
    @test !occursin("Val(", source)

    receipt_path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                            "reactivehmc-rke-ca9-v1.toml")
    receipt = TOML.parsefile(receipt_path)
    @test isempty(validate_reactivehmc_rke_receipt(receipt_path))
    @test receipt["pins"]["reactivehmc_revision"] ==
          ReactiveHMCAlgorithmCorpus.UPSTREAM.revision
    source_digests = Dict(ReactiveHMCAlgorithmCorpus.UPSTREAM.source_sha256)
    @test receipt["pins"]["energies_sha256"] == source_digests["src/energies.jl"]
end
