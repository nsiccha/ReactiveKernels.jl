# Focused producer/receipt contract for incremental all80 batches.
# Stdlib-only: no RK, Enzyme, Reactant, PosteriorDB, model execution, or measurement.
include(joinpath(@__DIR__, "all80_receipt.jl"))
using Random
using Test
import SHA
import TOML
module CertPkg
end

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
        "dim" => 2,
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

@testset "actual parent producer seams" begin
    source = read(joinpath(@__DIR__, "all80_posteriordb.jl"), String)
    entrypoint = "get(ENV, _INNER, \"\") == \"1\" ? include(_BODY) : _run()"
    @test occursin(entrypoint, source)
    producer = Module(:All80ParentProducer, true, true)
    Core.eval(producer, :(include(path::AbstractString) = Base.include($producer, path)))
    producer_source = tempname() * ".jl"
    write(producer_source, replace(source,
        entrypoint => "nothing", "@__DIR__" => repr(@__DIR__)))
    Base.include(producer, producer_source)
    for phase in ("native", "reactant")
        files = producer._phase_harness_files(phase)
        @test "all80_receipt.jl" in files
        @test all(f -> endswith(f, ".jl"), files)
        @test count(f -> occursin("eight_schools_density", f), files) == 1
    end
    @test length(rand(RandomDevice(), UInt8, 16)) == 16
end

@testset "native input identity helper seam" begin
    source = read(joinpath(@__DIR__, "all80_posteriordb_body.jl"), String)
    start = findfirst("function _input_identity(", source)
    stop = findfirst("\n# Scale-aware native parity floor", source)
    start !== nothing && stop !== nothing && first(stop) > first(start) ||
        error("could not extract native input identity helper")
    helper_source = source[first(start):first(stop)-1]
    harness = Module(:NativeIdentityHarness, true, true)
    Core.eval(harness, :(import Main.All80Receipt))
    Core.eval(harness, :(import Main.SHA))
    Base.include_string(harness, """
        module PosteriorDB
        dataset(_) = nothing
        model(_) = "model"
        implementation(_, _) = nothing
        path(_) = $(repr(joinpath(@__DIR__, "all80_receipt.jl")))
        load(_, ::Type{String}) = "{}"
        name(x) = x
        end
        """)
    Base.include_string(harness, helper_source)
    identity = harness._input_identity(nothing, "m1", [[0.1, -0.2]];
        phase = "native", seed = 1, scale = 0.2, draws = 3, stan_perm = nothing)
    @test identity["posterior"] == "m1"
    @test identity["phase"] == "native"
    @test identity["query"]["points"] == [[0.1, -0.2]]
end

@testset "dotdot package roots certify" begin
    # Frozen roots ARE normpath(joinpath(@__DIR__, ".."))-shaped, and normpath keeps a
    # trailing separator on ".."-resolved inputs (Julia 1.10) — the loaded-root check
    # must strip it rather than double the separator into a false mismatch (this exact
    # shape failed a live DISCOVER run while the path visibly sat inside the root).
    dotroot = mktempdir(); mkpath(joinpath(dotroot, "src"))
    write(joinpath(dotroot, "Project.toml"), "name = \"dotpkg\"\n")
    write(joinpath(dotroot, "src", "DotPkg.jl"), "const DOT = 1\n")
    run(`git -C $dotroot init -q`)
    run(`git -C $dotroot config user.email fixture@example.com`)
    run(`git -C $dotroot config user.name fixture`)
    run(`git -C $dotroot add Project.toml src/DotPkg.jl`)
    run(`git -C $dotroot commit -q -m fixture-package`)
    DotPkg = Module(:DotPkg)
    Base.include(DotPkg, joinpath(dotroot, "src", "DotPkg.jl"))
    snap = freeze_provenance!(; packages = Dict("dotpkg" => joinpath(dotroot, "src", "..")),
        upstream_hash = "a7ef985b", harness_files = ["all80_receipt.jl"],
        extra = Dict{String,Any}("phase" => "native", "batch" => "batch1",
            "requested_keys" => ["m1"], "run_id" => "run-dotdot",
            "ad_backend" => ORDINARY_AD_BACKEND))
    All80Receipt._PROV_START[] = snap
    certified = certify_loaded_modules!(Dict{String,Any}("dotpkg" => DotPkg))
    @test certified["loaded_modules_certified"] == true
end

@testset "Reactant numeric digest helper seam" begin
    source = read(joinpath(@__DIR__, "all80_reactant_body.jl"), String)
    start = findfirst("_numeric_vector_sha256(values) =", source)
    stop = findfirst("\nfunction reference_valid_probe", source)
    start !== nothing && stop !== nothing && first(stop) > first(start) ||
        error("could not extract numeric digest helper")
    helper = source[first(start):first(stop)-1]
    harness = Module(:ReactantDigestHarness, true, true)
    Core.eval(harness, :(import Main.SHA))
    Base.include_string(harness, helper)
    @test length(harness._numeric_vector_sha256([0.1, -0.2])) == 64
    @test harness._numeric_vector_sha256([0.1, -0.2]) ==
        harness._numeric_vector_sha256([0.1, -0.2])
end

@testset "incremental source lock, loaded-root gate, and exact receipt publication" begin
    tmp = mktempdir(); mkpath(joinpath(tmp, "src")); mkpath(joinpath(tmp, "ext"))
    write(joinpath(tmp, "Project.toml"), "name = \"tmppkg\"\n")
    write(joinpath(tmp, "src", "x.jl"), "const A = 1\n")
    write(joinpath(tmp, "ext", "x_ext.jl"), "const B = 2\n")

        gitroot = mktempdir(); mkpath(joinpath(gitroot, "src"))
    write(joinpath(gitroot, "Project.toml"), "name = \"certpkg\"\n")
    write(joinpath(gitroot, "src", "CertPkg.jl"), "const CERT = 1\n")
    run(`git -C $gitroot init -q`)
    run(`git -C $gitroot config user.email fixture@example.com`)
    run(`git -C $gitroot config user.name fixture`)
    run(`git -C $gitroot add Project.toml src/CertPkg.jl`)
    run(`git -C $gitroot commit -q -m fixture-package`)
    CertPkg = Module(:CertPkg)
    Base.include(CertPkg, joinpath(gitroot, "src", "CertPkg.jl"))

    parent = _snapshot(gitroot; phase = "native")
    @test parent["packages"]["tmppkg"]["src_ext_project_sha256"] isa AbstractString
    @test parent["loaded_modules_certified"] == false
    lock_path = joinpath(tmp, "source-lock.toml")
    write_provenance_lock(lock_path, parent)
    loaded = load_provenance!(lock_path)
    @test loaded["source_id"] == parent["source_id"]
    @test verify_provenance()["source_id"] == parent["source_id"]

    # Loaded roots cannot be certified for a no-git package, even when its source path resolves.
    no_git_snapshot = freeze_provenance!(; packages = Dict("tmppkg" => tmp),
        upstream_hash = "a7ef985b", harness_files = ["all80_receipt.jl"],
        extra = Dict{String,Any}("phase" => "native", "batch" => "batch1",
            "requested_keys" => ["m1"], "run_id" => "run-1",
            "ad_backend" => ORDINARY_AD_BACKEND))
    _PROV_START = All80Receipt._PROV_START
    _PROV_START[] = no_git_snapshot
    toy = joinpath(tmp, "src", "ToyPkg.jl")
    write(toy, "module ToyPkg\nend\n")
    include(toy)
    @test_throws ErrorException certify_loaded_modules!(Dict{String,Any}())
    @test_throws ErrorException certify_loaded_modules!(
        Dict{String,Any}("tmppkg" => ToyPkg))

    # Mid-run byte drift is refused for src, ext, and Project through the frozen tree hash.
    write(joinpath(tmp, "ext", "x_ext.jl"), "const B = 3   # mutated\n")
    @test_throws ErrorException verify_provenance()
    write(joinpath(tmp, "ext", "x_ext.jl"), "const B = 2\n")

    # Build a complete two-phase, exact-key receipt from producer-like snapshots.
    certified_snapshot = _snapshot(gitroot; phase = "native")
    _PROV_START[] = certified_snapshot
    certified_snapshot = certify_loaded_modules!(
        Dict{String,Any}("tmppkg" => CertPkg))
    native_snapshot = _snapshot(gitroot; phase = "native")
    reactant_snapshot = _snapshot(gitroot; phase = "reactant")
    for snapshot in (native_snapshot, reactant_snapshot)
        snapshot["loaded_module_paths"] = certified_snapshot["loaded_module_paths"]
        snapshot["module_versions"] = certified_snapshot["module_versions"]
        snapshot["loaded_modules_certified"] = true
    end
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
    bad_identity = TOML.parsefile(out)
    row = bad_identity["models"]["m1"]
    row["dim"] = 3
    row["input_identity_native"]["phase"] = "wrong"
    row["input_identity_native"]["posterior"] = "foreign"
    row["input_identity_native"]["dataset_json_sha256"] = "not-a-digest"
    row["input_identity_native"]["query"]["points"] = []
    bad_path = joinpath(tmp, "bad-identity.toml")
    open(bad_path, "w") do io; TOML.print(io, bad_identity; sorted = true); end
    @test !isempty(validate_batch(bad_path; expected_keys = ["m1"],
        phases = ("native", "reactant"), batch = "batch1"))
    extra_phase = TOML.parsefile(out)
    extra_phase["meta"]["provenance"]["unexpected"] =
        extra_phase["meta"]["provenance"]["native"]
    extra_path = joinpath(tmp, "extra-phase.toml")
    open(extra_path, "w") do io; TOML.print(io, extra_phase; sorted = true); end
    @test !isempty(validate_batch(extra_path; expected_keys = ["m1"],
        phases = ("native", "reactant"), batch = "batch1"))

    # The actual producer writes provenance on BOTH phase paths, including Reactant.
    @test Set(keys(TOML.parsefile(out)["meta"]["provenance"])) ==
        Set(["native", "reactant"])
    @test All80Receipt.selected_point_stan_parity_ok(9.0, 10.0, 1.0)
    @test !All80Receipt.selected_point_stan_parity_ok(9.0, 11.0, 1.0)
    reactant_evals = read(joinpath(@__DIR__, "all80_reactant_evals.jl"), String)
    reactant_body = read(joinpath(@__DIR__, "all80_reactant_body.jl"), String)
    @test occursin("All80Receipt.selected_point_stan_parity_ok", reactant_evals)
    @test occursin("stan_value = probe.reference_value", reactant_body)
    @test occursin("rk_offset = entry.off_rk", reactant_body)

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
