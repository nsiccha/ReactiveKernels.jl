using ReactiveKernels
using Test
import TOML

include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_hmc_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "receipts", "validate_reactivehmc_hmc.jl"))

const RHMC_HMC = ReactiveHMCHMCFixture

@testset "ReactiveHMC fixed-step HMC source and independent receipt" begin
    registration = ReactiveKernels.kernel_registration(RHMC_HMC.hmc_state)
    @test registration.kind == :object_kernel
    @test isempty(registration.write_roots)
    @test ReactiveKernels.kernel_port_names(RHMC_HMC.hmc_state) ==
          (:init, :n_steps, :min_dham, :step_f, :stats_f,
           :gofwd, :fwd, :dham, :diverged)
    @test map(ir -> ir.id.name, ReactiveKernels.method_irs(RHMC_HMC.hmc_state)) ==
          (:randbernoullilog, :step!)

    source_path = joinpath(@__DIR__, "..", "benchmark",
                           "reactivehmc_hmc_kernel_fixture.jl")
    source = read(source_path, String)
    @test occursin("@kernel hmc_state(", source)
    @test occursin("init.mom = sqrt(fwd.metric) * Random.randn!(rng, init.mom)", source)
    @test occursin("stats_f(__self__)", source)
    @test findfirst("stats_f(__self__)", source) < findfirst("diverged && return", source)
    @test occursin("randbernoullilog(__self__, rng, dham) && copy!!(init, fwd)", source)
    @test !occursin(r"^\s*rcopy!\("m, source)

    receipt_path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                            "reactivehmc-hmc-ca9-v1.toml")
    receipt = TOML.parsefile(receipt_path)
    @test isempty(validate_reactivehmc_hmc_receipt(receipt_path))
    @test receipt["pins"]["reactivehmc_revision"] ==
          ReactiveHMCAlgorithmCorpus.UPSTREAM.revision
    source_digests = Dict(ReactiveHMCAlgorithmCorpus.UPSTREAM.source_sha256)
    @test receipt["pins"]["hmc_sha256"] == source_digests["src/hmc.jl"]
end
