# Provenance-freeze fixture for the batch-1 incremental run (performance review 2026-09-09).
# Proves: (1) freeze at process start captures harness + package source byte-hashes + git identity;
# (2) verify_provenance passes with no drift; (3) an ACTUAL mid-run source mutation makes verify
# REFUSE (the receipt must not misattribute mutated code); (4) write_phase carries a provenance block
# only when opted in — the frozen-82 default is unchanged; (5) aggregate(...; preserve_provenance=true)
# keeps BOTH the native and the reactant process-start block (the last phase never overwrites the first).
# Stdlib-only (TOML/SHA/git) — no RK env needed. Run: julia --startup-file=no benchmark/test_batch1_provenance_fixture.jl
include(joinpath(@__DIR__, "all80_receipt.jl"))
using Test
import TOML
using .All80Receipt: freeze_provenance!, verify_provenance, write_phase, aggregate

@testset "batch-1 process-start provenance: freeze / verify / mid-run-mutation refusal" begin
    tmp = mktempdir(); mkpath(joinpath(tmp, "src"))
    write(joinpath(tmp, "src", "x.jl"), "const A = 1\n")

    snap = freeze_provenance!(; packages = Dict("tmppkg" => tmp),
        upstream_hash = "a7ef985b", extra = Dict("query" => "diamonds-diamonds", "q" => [0.1, -0.2]))
    # captured the expected fields
    @test haskey(snap, "harness_file_sha256") && haskey(snap["harness_file_sha256"], "all80_receipt.jl")
    @test snap["packages"]["tmppkg"]["src_bytes_sha256"] isa AbstractString
    @test snap["upstream_posteriordb_models_sha256"] == "a7ef985b"
    @test snap["query"] == "diamonds-diamonds"             # caller `extra` merged at top level
    @test verify_provenance()["captured_at"] == snap["captured_at"]   # no drift ⇒ returns the snapshot

    # (3) ACTUAL mid-run mutation of a tracked source file ⇒ verify REFUSES.
    write(joinpath(tmp, "src", "x.jl"), "const A = 2   # mutated mid-run\n")
    @test_throws ErrorException verify_provenance()

    # (4) write_phase: provenance omitted by default (frozen-82 unchanged), written when opted in.
    write(joinpath(tmp, "src", "x.jl"), "const A = 1\n")   # restore so the snapshot is valid again
    snap2 = freeze_provenance!(; packages = Dict("tmppkg" => tmp), upstream_hash = "a7ef985b")
    nat = joinpath(tmp, "native.toml"); rea = joinpath(tmp, "reactant.toml")
    write_phase(nat, "native", Dict("m1" => Dict("primal_rk" => 1.0)))                       # default: no provenance
    write_phase(rea, "reactant", Dict("m1" => Dict("primal_rk_reactant" => 2.0)); provenance = snap2)
    @test !haskey(TOML.parsefile(nat), "provenance")
    @test haskey(TOML.parsefile(rea), "provenance")

    # (5) aggregate preserves per-phase provenance — but only for phases that HAVE it, keyed by phase.
    #     Give both phases their OWN block and confirm both survive (last does not overwrite first).
    write_phase(nat, "native", Dict("m1" => Dict("primal_rk" => 1.0)); provenance = snap2)
    out = joinpath(tmp, "batch1.toml")
    aggregate([nat, rea], out; meta = Dict("batch" => "batch1"), preserve_provenance = true)
    agg = TOML.parsefile(out)
    @test haskey(agg["meta"], "provenance")
    @test haskey(agg["meta"]["provenance"], "native") && haskey(agg["meta"]["provenance"], "reactant")
    @test agg["meta"]["batch"] == "batch1"                 # existing meta preserved
    # frozen-82 default path: no provenance in meta.
    aggregate([nat, rea], out; meta = Dict("batch" => "x"))   # preserve_provenance=false default → but nat/rea carry blocks
    @test !haskey(TOML.parsefile(out)["meta"], "provenance")  # not collected unless opted in
end
println("BATCH1_PROVENANCE_FIXTURE_OK")
