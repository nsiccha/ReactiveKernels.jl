# Focused producer/receipt contract for incremental all80 batches.
# Stdlib-only: no RK, Enzyme, Reactant, PosteriorDB, model execution, or measurement.
include(joinpath(@__DIR__, "all80_receipt.jl"))
using Test
import TOML
using .All80Receipt: freeze_provenance!, write_provenance_lock, load_provenance!,
    certify_loaded_modules!, verify_provenance, write_phase, aggregate,
    validate_batch, assert_batch_resume!, input_identity, ORDINARY_AD_BACKEND

function _snapshot(tmp; phase, run_id = "run-1", keys = ["m1"],
    backend = ORDINARY_AD_BACKEND, batch = "batch1")
    snap = freeze_provenance!(; packages = Dict("tmppkg" => tmp),
        upstream_hash = "a7ef985b", harness_files = [
            "all80_posteriordb.jl", "all80_posteriordb_body.jl",
            "all80_reactant_body.jl", "all80_reactant_evals.jl",
            "all80_registry.jl", "all80_receipt.jl", "all80_axes.jl",
            "all80_metadata.jl", "all80_parity.jl"],
        extra = Dict{String,Any}("batch" => batch, "phase" => phase,
            "requested_keys" => keys, "run_id" => run_id,
            "ad_backend" => backend))
    # A parent lock remains uncertified until the child certifies actual loaded roots.
    snap
end

function _identity(phase)
    query = Dict{String,Any}("points" => [[0.1, -0.2]],
        "selected_point_index" => 1, "ad_backend" => ORDINARY_AD_BACKEND)
    input_identity(; phase = phase, posterior = "m1", model = "model",
        stan_path = "model.stan", stan_sha256 = "a"^64,
        library = "model.so", library_sha256 = "b"^64,
        dataset_path = "data.json", dataset_json_sha256 = "c"^64,
        dataset_json_bytes = 2, query = query, stan_perm = nothing)
end

function _complete_row()
    Dict{String,Any}(
        "primal_turing" => 1.0, "primal_stan" => 1.0,
        "gradient_turing" => 1.0, "gradient_stan" => 1.0,
        "hmc_ahmc_turing" => 1.0, "primal_rk" => 1.0,
        "gradient_rk" => 1.0, "hmc_rk_native" => 1.0,
        "primal_rk_reactant" => 1.0, "gradient_rk_reactant" => 1.0,
        "hmc_rk_reactant" => 1.0, "primal_opt_stan" => "deferred",
        "primal_further_turing" => "deferred", "gradient_opt_stan" => "deferred",
        "gradient_further_turing" => "deferred", "parity_pass" => true,
        "input_identity_native" => _identity("native"),
        "input_identity_reactant" => _identity("reactant"))
end

@testset "incremental source lock, loaded-root gate, and exact receipt publication" begin
    tmp = mktempdir(); mkpath(joinpath(tmp, "src")); mkpath(joinpath(tmp, "ext"))
    write(joinpath(tmp, "Project.toml"), "name = \"tmppkg\"\n")
    write(joinpath(tmp, "src", "x.jl"), "const A = 1\n")
    write(joinpath(tmp, "ext", "x_ext.jl"), "const B = 2\n")

    parent = _snapshot(tmp; phase = "native")
    @test parent["packages"]["tmppkg"]["src_ext_project_sha256"] isa AbstractString
    @test parent["loaded_modules_certified"] == false
    lock_path = joinpath(tmp, "source-lock.toml")
    write_provenance_lock(lock_path, parent)
    loaded = load_provenance!(lock_path)
    @test loaded["source_id"] == parent["source_id"]
    @test verify_provenance()["source_id"] == parent["source_id"]

    # Loaded roots cannot be certified for a no-git package, even when its source path resolves.
    toy = joinpath(tmp, "src", "ToyPkg.jl")
    write(toy, "module ToyPkg\nend\n")
    include(toy)
    @test_throws ErrorException certify_loaded_modules!(
        Dict{String,Any}("tmppkg" => ToyPkg))

    # Mid-run byte drift is refused for src, ext, and Project through the frozen tree hash.
    write(joinpath(tmp, "ext", "x_ext.jl"), "const B = 3   # mutated\n")
    @test_throws ErrorException verify_provenance()
    write(joinpath(tmp, "ext", "x_ext.jl"), "const B = 2\n")

    # Build a complete two-phase, exact-key receipt from producer-like snapshots.
    native_snapshot = _snapshot(tmp; phase = "native")
    reactant_snapshot = _snapshot(tmp; phase = "reactant")
    # Simulate the child's post-import certification for aggregation/publication only.
    native_snapshot["loaded_modules_certified"] = true
    reactant_snapshot["loaded_modules_certified"] = true
    native = joinpath(tmp, "native.toml"); reactant = joinpath(tmp, "reactant.toml")
    write_phase(native, "native", Dict("m1" => _complete_row());
        provenance = native_snapshot)
    write_phase(reactant, "reactant", Dict("m1" => _complete_row());
        provenance = reactant_snapshot)
    parsed_native = TOML.parsefile(native)
    @test parsed_native["source_id"] == native_snapshot["source_id"]
    @test parsed_native["run_id"] == "run-1"

    out = joinpath(tmp, "batch1.toml")
    aggregate([native, reactant], out; meta = Dict{String,Any}(
        "batch" => "batch1", "args" => ["m1"]), preserve_provenance = true,
        require_phases = ["native", "reactant"], expected_keys = ["m1"],
        batch = "batch1")
    @test isempty(validate_batch(out; expected_keys = ["m1"],
        phases = ("native", "reactant"), batch = "batch1"))

    # The actual producer writes provenance on BOTH phase paths, including Reactant.
    @test Set(keys(TOML.parsefile(out)["meta"]["provenance"])) ==
        Set(["native", "reactant"])

    # Exact-key resume accepts its own checkpoint and refuses foreign/incompatible rows.
    assert_batch_resume!(parsed_native, native; phase = "native",
        targets = ["m1"], provenance = native_snapshot)
    foreign = _snapshot(tmp; phase = "native", run_id = "run-2")
    @test_throws ErrorException assert_batch_resume!(parsed_native, native;
        phase = "native", targets = ["m1"], provenance = foreign)
    @test_throws ErrorException assert_batch_resume!(parsed_native, native;
        phase = "native", targets = ["other"], provenance = native_snapshot)

    # Strict aggregation fails missing/duplicate/mismatched phases and incomplete key sets.
    @test_throws ErrorException aggregate([native], out;
        require_phases = ["native", "reactant"], expected_keys = ["m1"], batch = "batch1")
    @test_throws ErrorException aggregate([native, native], out;
        require_phases = ["native", "reactant"], expected_keys = ["m1"], batch = "batch1")
    @test_throws ErrorException aggregate([native, reactant], out;
        require_phases = ["native", "reactant"], expected_keys = ["other"], batch = "batch1")
    incomplete = joinpath(tmp, "incomplete.toml")
    write_phase(incomplete, "reactant", Dict{String,Any}();
        provenance = reactant_snapshot)
    @test_throws ErrorException aggregate([native, incomplete], out;
        require_phases = ["native", "reactant"], expected_keys = ["m1"], batch = "batch1")
    no_provenance = joinpath(tmp, "no-provenance.toml")
    write_phase(no_provenance, "reactant", Dict("m1" => _complete_row()))
    @test_throws ErrorException aggregate([native, no_provenance], out;
        require_phases = ["native", "reactant"], expected_keys = ["m1"], batch = "batch1")

    # Frozen-82 default remains opt-out: no provenance is collected or emitted.
    frozen_native = joinpath(tmp, "frozen-native.toml")
    frozen_reactant = joinpath(tmp, "frozen-reactant.toml")
    write_phase(frozen_native, "native", Dict("m1" => _complete_row()))
    write_phase(frozen_reactant, "reactant", Dict("m1" => _complete_row()))
    frozen_out = joinpath(tmp, "frozen.toml")
    aggregate([frozen_native, frozen_reactant], frozen_out;
        meta = Dict{String,Any}("args" => ["m1"]))
    @test !haskey(TOML.parsefile(frozen_out)["meta"], "provenance")
end
println("BATCH1_PROVENANCE_FIXTURE_OK")
