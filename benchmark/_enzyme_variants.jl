# Registered-versus-dogfood Enzyme runs for benchmark wrappers.
#
# A wrapper that times Enzyme gradients resolves its environment once per
# variant and runs the same body, arguments and Julia command in each:
#
#   registered  the wrapper's own registry Enzyme selection (its baseline)
#   dogfood     the same environment with Enzyme and its in-tree EnzymeCore
#               developed from the host's exact local dogfood commit
#
# Every requested variant is attempted independently. A setup, precompile or
# body failure is recorded with its error and log rather than dropped, and the
# other variant still runs. The matrix file records Julia, CPU and thread
# settings, both resolved dependency sets and their differences, the exact
# dogfood revision, the benchmark source state, and paired numeric leaves of
# the two receipts. Registered is the baseline; dogfood/registered ratios are
# within-implementation deltas, and a variant without a receipt has no ratio.
#
# Environment variables:
#   RK_ENZYME_VARIANTS            comma-separated subset of registered,dogfood
#                                 (default both)
#   RK_ENZYME_DOGFOOD_MIRROR      local Enzyme.jl Git mirror (default: this
#                                 host's KB mirror)
#   RK_ENZYME_DOGFOOD_REVISION    ref or full SHA in that mirror (default
#                                 refs/kb-fetched/local/dogfood)
#   RK_ENZYME_DOGFOOD_CHECKOUTS   directory of immutable per-SHA checkouts
#   RK_ENZYME_VARIANTS_DIR        run directory (default: a new directory
#                                 under ~/.local/state/reactivekernels-benchmarks)
module EnzymeVariants

import Pkg
import SHA
import TOML
using Dates

include(joinpath(@__DIR__, "..", "deps", "source_dependencies.jl"))

export run_enzyme_variants, select_enzyme!, requested_output, variant_arguments
export rk_source_dependencies

const VARIANTS = ("registered", "dogfood")
const SCHEMA = "reactivekernels.enzyme-variants/v1"
const ENZYME_PACKAGES = ("Enzyme", "EnzymeCore", "Enzyme_jll")
const _STATE = joinpath(homedir(), ".local", "state", "reactivekernels-benchmarks")

struct DogfoodSource
    path::String
    revision::String
    version::String
    core_path::String
    core_version::String
    mirror::String
    ref::String
end

"""
One variant's run: its name, work directory, receipt path (or `nothing`) and,
for dogfood, the resolved local source or the error that prevented it.
"""
struct Variant
    name::String
    dir::String
    receipt::Union{Nothing,String}
    dogfood::Union{Nothing,DogfoodSource,Exception}
end

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))
_text_sha256(bytes) = bytes2hex(SHA.sha256(bytes))
_error_text(e, bt = catch_backtrace()) = first(sprint(showerror, e, bt), 8000)

function requested_variants()
    raw = get(ENV, "RK_ENZYME_VARIANTS", join(VARIANTS, ","))
    names = unique(String.(strip.(split(raw, ','; keepempty = false))))
    isempty(names) && error("RK_ENZYME_VARIANTS selects no variant")
    unknown = setdiff(names, VARIANTS)
    isempty(unknown) || error("unknown Enzyme variant(s) $(join(unknown, ", ")); " *
                              "expected a subset of $(join(VARIANTS, ","))")
    [name for name in VARIANTS if name in names]
end

"The output path a wrapper's caller requested via `--output=` or an environment variable."
function requested_output(arguments = ARGS; env = nothing)
    for argument in arguments
        startswith(argument, "--output=") && return abspath(split(argument, '='; limit = 2)[2])
    end
    env === nothing && return nothing
    path = get(ENV, env, "")
    isempty(path) ? nothing : abspath(path)
end

"`arguments` with any `--output=` replaced by this variant's receipt path."
variant_arguments(arguments, variant::Variant) = String[
    startswith(argument, "--output=") ? "--output=$(variant.receipt)" : argument
    for argument in arguments]

"Materialize the exact local dogfood commit as an immutable clean checkout."
function dogfood_source()
    mirror = get(ENV, "RK_ENZYME_DOGFOOD_MIRROR",
        joinpath(homedir(), ".local", "state", "kb-agents", "git-mirrors", "Enzyme.jl.git"))
    ref = get(ENV, "RK_ENZYME_DOGFOOD_REVISION", "refs/kb-fetched/local/dogfood")
    isdir(mirror) || error("local Enzyme dogfood mirror is unavailable: $mirror " *
                           "(set RK_ENZYME_DOGFOOD_MIRROR)")
    revision = try
        _git(mirror, "rev-parse", "--verify", "$ref^{commit}")
    catch e
        error("dogfood revision $ref does not resolve to a commit in $mirror")
    end
    occursin(r"^[0-9a-f]{40}$", revision) ||
        error("dogfood revision $ref resolved to $revision, not a full commit SHA")
    checkouts = get(ENV, "RK_ENZYME_DOGFOOD_CHECKOUTS", joinpath(_STATE, "enzyme-dogfood"))
    path = joinpath(checkouts, "Enzyme.jl-" * revision)
    if !isdir(path)
        mkpath(checkouts)
        staged = mktempdir(checkouts; prefix = ".staging-", cleanup = false)
        try
            clone = joinpath(staged, "Enzyme.jl")
            run(`git clone --quiet --no-checkout $mirror $clone`)
            run(`git -C $clone checkout --quiet --detach $revision`)
            isdir(path) || mv(clone, path)
        finally
            rm(staged; recursive = true, force = true)
        end
    end
    head = _git(path, "rev-parse", "HEAD")
    head == revision || error("dogfood checkout $path is at $head, not $revision")
    status = _git(path, "status", "--porcelain")
    isempty(status) || error("dogfood checkout $path has local changes:\n$status")
    core = joinpath(path, "lib", "EnzymeCore")
    DogfoodSource(realpath(path), revision,
        TOML.parsefile(joinpath(path, "Project.toml"))["version"],
        realpath(core), TOML.parsefile(joinpath(core, "Project.toml"))["version"],
        mirror, ref)
end

function _manifest_path(project = Base.active_project())
    for name in ("JuliaManifest.toml", "Manifest.toml")
        path = joinpath(dirname(project), name)
        isfile(path) && return path
    end
    error("no manifest beside $project")
end

"Resolved packages as name => a description independent of the environment directory."
function _manifest_packages(manifest)
    root = dirname(manifest)
    packages = Dict{String,Dict{String,Any}}()
    for (name, entries) in TOML.parsefile(manifest)["deps"]
        entry = only(entries)
        description = Dict{String,Any}(
            "version" => get(entry, "version", ""), "uuid" => entry["uuid"])
        if haskey(entry, "path")
            path = normpath(joinpath(root, entry["path"]))
            description["path"] = startswith(path, root) ?
                "<environment>/" * relpath(path, root) : realpath(path)
        end
        for key in ("git-tree-sha1", "repo-url", "repo-rev")
            haskey(entry, key) && (description[key] = entry[key])
        end
        packages[name] = description
    end
    packages
end

function _manifest_entry(name)
    deps = TOML.parsefile(_manifest_path())["deps"]
    haskey(deps, name) || error("$name is absent from the resolved environment")
    only(deps[name])
end

function _manifest_source(name)
    entry = _manifest_entry(name)
    haskey(entry, "path") || return nothing
    realpath(normpath(joinpath(dirname(_manifest_path()), entry["path"])))
end

"""
    select_enzyme!(variant)

Put this variant's Enzyme into the active environment. Call it after the
wrapper's registry and local-source additions and before any precompile. The
registered variant keeps the wrapper's own registry selection. The dogfood
variant develops Enzyme and its in-tree EnzymeCore from the exact local
commit; an exact Enzyme/EnzymeCore compat entry in the active project moves to
that source's versions.
"""
function select_enzyme!(variant::Variant)
    variant.name == "registered" && return _verify_enzyme(variant)
    dogfood = variant.dogfood
    dogfood isa DogfoodSource || throw(dogfood)
    project_path = Base.active_project()
    project = TOML.parsefile(project_path)
    compat = get(project, "compat", Dict{String,Any}())
    pins = Dict("Enzyme" => dogfood.version, "EnzymeCore" => dogfood.core_version)
    if any(haskey(compat, name) for name in keys(pins))
        for (name, version) in pins
            haskey(compat, name) && (compat[name] = "=$version")
        end
        open(io -> TOML.print(io, project; sorted = true), project_path, "w")
    end
    Pkg.develop([Pkg.PackageSpec(path = dogfood.path),
                 Pkg.PackageSpec(path = dogfood.core_path)])
    _verify_enzyme(variant)
end

"Fail unless the active environment resolved this variant's Enzyme source."
function _verify_enzyme(variant::Variant)
    if variant.name == "registered"
        _manifest_source("Enzyme") === nothing ||
            error("registered environment resolved a local Enzyme source")
    else
        dogfood = variant.dogfood
        _manifest_source("Enzyme") == dogfood.path ||
            error("dogfood environment did not resolve Enzyme from $(dogfood.path); " *
                  "the wrapper must call select_enzyme!(variant)")
        _manifest_source("EnzymeCore") == dogfood.core_path ||
            error("dogfood environment did not resolve the in-tree EnzymeCore")
    end
    nothing
end

const _PROBE = raw"""
import LinearAlgebra, Pkg, TOML
deps = Dict(info.name => info for info in values(Pkg.dependencies()))
package(name) = haskey(deps, name) ? Dict(
    "version" => string(deps[name].version),
    "source" => something(deps[name].source, ""),
    "tracking_path" => deps[name].is_tracking_path,
    "tracking_registry" => deps[name].is_tracking_registry) : Dict("absent" => true)
cpu_target = Base.JLOptions().cpu_target == C_NULL ? "native" :
    unsafe_string(Base.JLOptions().cpu_target)
TOML.print(stdout, Dict(
    "julia_version" => string(VERSION),
    "cpu_target" => cpu_target,
    "cpu_name" => Sys.CPU_NAME,
    "cpu_model" => first(Sys.cpu_info()).model,
    "cpu_threads" => Sys.CPU_THREADS,
    "julia_threads" => Threads.nthreads(),
    "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
    "check_bounds" => Int(Base.JLOptions().check_bounds),
    "packages" => Dict(name => package(name) for name in split(ARGS[1], ','))); sorted = true)
"""

const _MATCHED_SETTINGS = ("julia_version", "cpu_target", "cpu_name", "cpu_model",
    "julia_threads", "blas_threads", "check_bounds")

function _probe(environment, packages)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$environment -e $_PROBE $(join(packages, ','))`
    TOML.parse(readchomp(command))
end

"Run `command` to completion, streaming its merged output to stdout and `log`."
function _run_logged(command, log)
    open(log, "w") do io
        pipe = Pipe()
        process = run(pipeline(ignorestatus(command); stdout = pipe, stderr = pipe); wait = false)
        close(pipe.in)
        for line in eachline(pipe; keep = true)
            write(io, line)
            flush(io)
            print(stdout, line)
        end
        wait(process)
        process
    end
end

function _log_tail(log; lines = 40)
    isfile(log) || return ""
    text = readlines(log)
    join(text[max(1, end - lines + 1):end], '\n')
end

# Variables whose values are benchmark configuration rather than host secrets.
const _RECORDED_ENV = r"^(JULIA_|RK_|REACTIVEKERNELS_|AS_|OPENBLAS_|OMP_|MKL_)"

"Command argv and environment with this variant's paths made symbolic."
function _command_shape(command::Cmd, environment, variant::Variant)
    symbolic(text) = replace(text, environment => "<environment>", variant.dir => "<variant>")
    env = command.env === nothing ? ["$k=$v" for (k, v) in ENV] : command.env
    env = sort!([symbolic(entry) for entry in env])
    Dict{String,Any}(
        "exec" => [symbolic(argument) for argument in command.exec],
        "dir" => symbolic(command.dir),
        "env_sha256" => _text_sha256(join(env, '\0')),
        "env" => filter(entry -> occursin(_RECORDED_ENV, entry), env))
end

function _source_state(root)
    Dict{String,Any}(
        "revision" => _git(root, "rev-parse", "HEAD"),
        "tracked_diff_sha256" => _text_sha256(read(`git -C $root diff HEAD`)),
        "tracked_dirty" => !isempty(_git(root, "status", "--porcelain", "--untracked-files=no")))
end

function _run_variant(prepare, variant::Variant, packages, precompile)
    record = Dict{String,Any}("status" => "environment_failed")
    environment = mktempdir(prefix = "reactivekernels-enzyme-$(variant.name)-")
    try
        # An unavailable dogfood source fails this variant before any setup.
        variant.name == "dogfood" && !(variant.dogfood isa DogfoodSource) && throw(variant.dogfood)
        Pkg.activate(environment)
        command = nothing
        record["setup_seconds"] = @elapsed begin
            command = prepare(environment, variant)
            command isa Cmd || error("prepare must return the body command, got $(typeof(command))")
        end
        _verify_enzyme(variant)
        precompile && (record["precompile_seconds"] = @elapsed Pkg.precompile())
        manifest = _manifest_path()
        for path in (Base.active_project(), manifest)
            cp(path, joinpath(variant.dir, basename(path)); force = true)
        end
        record["manifest_sha256"] = _text_sha256(read(manifest))
        record["dependencies"] = _manifest_packages(manifest)
        record["settings"] = _probe(environment, packages)
        record["command"] = _command_shape(command, environment, variant)
        log = joinpath(variant.dir, "body.log")
        record["log"] = log
        start = time()
        process = _run_logged(command, log)
        record["body_seconds"] = time() - start
        record["exit_code"] = process.exitcode
        record["term_signal"] = process.termsignal
        if success(process)
            record["status"] = "completed"
        else
            record["status"] = "failed"
            record["error"] = "body exited with code $(process.exitcode)" *
                (process.termsignal == 0 ? "" : " (signal $(process.termsignal))")
            record["log_tail"] = _log_tail(log)
        end
    catch e
        record["error"] = _error_text(e)
    finally
        rm(environment; recursive = true, force = true)
    end
    receipt = variant.receipt
    record["receipt"] = receipt !== nothing && isfile(receipt) ? receipt : nothing
    record
end

function _dependency_differences(registered, dogfood)
    differences = Dict{String,Any}[]
    for name in sort!(collect(union(keys(registered), keys(dogfood))))
        a = get(registered, name, nothing)
        b = get(dogfood, name, nothing)
        a == b && continue
        push!(differences, Dict{String,Any}("name" => name,
            "enzyme_package" => name in ENZYME_PACKAGES,
            "registered" => something(a, Dict("absent" => true)),
            "dogfood" => something(b, Dict("absent" => true))))
    end
    differences
end

_is_number(x) = x isa Real && !(x isa Bool)

function _numeric_leaves!(leaves, value, path)
    if value isa AbstractDict
        for (key, child) in value
            _numeric_leaves!(leaves, child, isempty(path) ? string(key) : "$path.$key")
        end
    elseif value isa AbstractVector
        for (index, child) in pairs(value)
            _numeric_leaves!(leaves, child, "$path[$index]")
        end
    elseif _is_number(value)
        leaves[path] = Float64(value)
    end
    leaves
end

# Paired measurement leaves of two receipts from the same body. Paths name
# the same timing call because both variants ran identical code and inputs.
const _MEASUREMENT_KEY = r"(time|seconds|_ns|_us|_ms|bytes|allocs|memory)"i

function _paired_measurements(registered, dogfood)
    a = _numeric_leaves!(Dict{String,Float64}(), TOML.parsefile(registered), "")
    b = _numeric_leaves!(Dict{String,Float64}(), TOML.parsefile(dogfood), "")
    rows = Dict{String,Any}[]
    for path in sort!(collect(intersect(keys(a), keys(b))))
        occursin(_MEASUREMENT_KEY, last(split(path, '.'))) || continue
        row = Dict{String,Any}("path" => path, "registered" => a[path], "dogfood" => b[path])
        a[path] != 0 && (row["dogfood_over_registered"] = b[path] / a[path])
        push!(rows, row)
    end
    rows
end

_sibling(path, suffix) = let (stem, ext) = splitext(path)
    stem * suffix * (isempty(ext) ? ".toml" : ext)
end

"""
    run_enzyme_variants(prepare; suite, root, output = nothing, packages = ())

Run one benchmark body under each requested Enzyme variant.

`prepare(environment, variant)` populates the already activated `environment`
exactly as the wrapper's registered setup does, including its own registry
Enzyme specification, calls `select_enzyme!(variant)` after those additions
and before any precompile, and returns the body `Cmd`. It tells the body to
write its receipt to `variant.receipt` when one was requested. The runner then
verifies the variant's Enzyme source, precompiles the environment unless
`precompile=false` (so a compile failure is an environment failure), records
settings and dependencies, and runs the command.

With an `output` path the registered receipt is published there, the dogfood
receipt beside it as `<stem>.enzyme-dogfood.toml`, and the matrix as
`<stem>.enzyme-variants.toml`; the run directory keeps logs, projects and
manifests. The process exits with an error when the baseline variant (registered,
or dogfood when run alone) did not complete; a dogfood failure beside a
completed registered baseline is recorded in the matrix and printed.
"""
function run_enzyme_variants(prepare::Function; suite::AbstractString, root::AbstractString,
        output = nothing, packages = (), precompile = true)
    names = requested_variants()
    root = realpath(root)
    stamp = Dates.format(now(UTC), "yyyymmddTHHMMSS")
    run_dir = abspath(get(ENV, "RK_ENZYME_VARIANTS_DIR",
        joinpath(_STATE, "enzyme-variants", suite, stamp * "Z-" * first(_git(root, "rev-parse", "HEAD"), 9))))
    mkpath(run_dir)
    receipt_name = output === nothing ? nothing : basename(output)
    packages = unique(String[ENZYME_PACKAGES..., "ReactiveKernels", packages...])
    source_before = _source_state(root)
    dogfood = if "dogfood" in names
        try
            dogfood_source()
        catch e
            ErrorException(sprint(showerror, e))
        end
    end
    variants = Dict(name => Variant(name, mkpath(joinpath(run_dir, name)),
        receipt_name === nothing ? nothing : joinpath(run_dir, name, receipt_name),
        name == "dogfood" ? dogfood : nothing) for name in names)
    records = Dict{String,Any}()
    for name in VARIANTS
        if !(name in names)
            records[name] = Dict{String,Any}("status" => "not_requested")
            continue
        end
        println("ENZYME_VARIANT ", name, " start")
        flush(stdout)
        records[name] = _run_variant(prepare, variants[name], packages, precompile)
        println("ENZYME_VARIANT ", name, " status=", records[name]["status"])
        flush(stdout)
    end
    source_after = _source_state(root)

    if dogfood isa DogfoodSource
        merge!(records["dogfood"], Dict("source" => dogfood.path, "source_revision" => dogfood.revision,
            "version" => dogfood.version, "enzymecore_version" => dogfood.core_version,
            "mirror" => dogfood.mirror, "ref" => dogfood.ref, "source_dirty" => false))
    elseif dogfood !== nothing
        records["dogfood"]["source_error"] = dogfood.msg
    end
    haskey(records["registered"], "dependencies") &&
        (records["registered"]["source"] = "registry")

    comparison = Dict{String,Any}("baseline" => "registered",
        "delta" => "dogfood/registered within one implementation; no variant is chosen by speed")
    invalid = String[]
    source_before == source_after || push!(invalid, "benchmark source changed during the run")
    both = all(name -> haskey(records[name], "settings"), VARIANTS)
    if both
        a, b = records["registered"], records["dogfood"]
        for field in _MATCHED_SETTINGS
            a["settings"][field] == b["settings"][field] ||
                push!(invalid, "$field differs: registered $(a["settings"][field]), dogfood $(b["settings"][field])")
        end
        a["command"] == b["command"] || push!(invalid, "body command differs between variants")
        comparison["dependency_differences"] =
            _dependency_differences(a["dependencies"], b["dependencies"])
    end
    comparison["status"] = !isempty(invalid) ? "invalid" :
        all(name -> records[name]["status"] == "completed", VARIANTS) ? "complete" : "incomplete"
    isempty(invalid) || (comparison["errors"] = invalid)
    receipts = [get(records[name], "receipt", nothing) for name in VARIANTS]
    if comparison["status"] == "complete" && all(!isnothing, receipts)
        comparison["measurements"] = _paired_measurements(receipts...)
    end

    matrix = Dict{String,Any}(
        "schema" => SCHEMA,
        "suite" => suite,
        "generated_at" => string(now(UTC), "Z"),
        "run_dir" => run_dir,
        "benchmark_source" => merge(source_before, Dict("root" => root)),
        "julia_command" => collect(Base.julia_cmd().exec),
        "variants" => records,
        "comparison" => comparison)
    _write_toml(joinpath(run_dir, "enzyme-variants.toml"), matrix)
    if output !== nothing
        published = Dict(output => get(records["registered"], "receipt", nothing),
            _sibling(output, ".enzyme-dogfood") => get(records["dogfood"], "receipt", nothing))
        for (destination, source) in published
            source === nothing && continue
            mkpath(dirname(destination))
            cp(source, destination; force = true)
        end
        cp(joinpath(run_dir, "enzyme-variants.toml"), _sibling(output, ".enzyme-variants"); force = true)
    end
    println("ENZYME_VARIANTS matrix=", joinpath(run_dir, "enzyme-variants.toml"),
        " comparison=", comparison["status"],
        join((" $name=$(records[name]["status"])" for name in VARIANTS)))
    baseline = "registered" in names ? "registered" : "dogfood"
    records[baseline]["status"] == "completed" ||
        error("$baseline Enzyme variant did not complete: $(get(records[baseline], "error", records[baseline]["status"]))")
    matrix
end

# TOML cannot represent `nothing`; omit those keys rather than invent values.
_tomlable(value::AbstractDict) = Dict{String,Any}(string(k) => _tomlable(v) for (k, v) in value if v !== nothing)
_tomlable(value::AbstractVector) = Any[_tomlable(v) for v in value if v !== nothing]
_tomlable(value) = value

_write_toml(path, value) = open(io -> TOML.print(io, _tomlable(value); sorted = true), path, "w")

end # module EnzymeVariants
