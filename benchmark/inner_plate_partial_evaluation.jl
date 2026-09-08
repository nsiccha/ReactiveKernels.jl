# Focused acceptance/measurement driver. Run in a consumer environment containing
# RK, its distribution package, BenchmarkTools, SpecialFunctions, and Enzyme;
# add Reactant for RK_INNER_PE_BACKEND=reactant. Output belongs in task scratch.
# Use this SAME detached driver checkout for both compiler revisions. Select
# the loaded compiler with RK_INNER_PE_COMPILER_ROOT/SHA and ROLE=baseline or
# candidate (all variables have the RK_INNER_PE_ prefix). Baseline measures ON;
# candidate measures alternating OFF/ON pairs. For cross-revision comparisons,
# alternate fresh baseline/candidate processes over several pairs, recording
# PAIR_ID and PROCESS_ORDER=1/2. The second process must set MATCH_RECEIPT to
# the first process's TOML output; mismatched dependencies, inputs or source
# hashes fail acceptance. Package-load/compilation time is never steady time.
const load_started = time()
using ReactiveKernels, BenchmarkTools, SpecialFunctions, TOML, SHA, Test, Pkg, Statistics
import ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: binomial
using DifferentiationInterface: AutoEnzyme
import Enzyme

const backend_name = get(ENV, "RK_INNER_PE_BACKEND", "native")
backend_name in ("native", "reactant") || error("unknown backend: $backend_name")
reactant_loaded() = any(id -> id.name == "Reactant", keys(Base.loaded_modules))
if backend_name == "reactant"
    @eval using Reactant
    @assert Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing
end
backend_name == "native" && reactant_loaded() && error(
    "native-only measurements require a fresh Julia process without Reactant loaded")
const package_load_s = time() - load_started
const ad_backend = AutoEnzyme(; mode = Enzyme.Reverse)
const root = dirname(@__DIR__)
include("_repro_guard.jl")
const driver_sha = _require_clean_detached_candidate(root)
const run_role = get(ENV, "RK_INNER_PE_ROLE", "candidate")
run_role in ("baseline", "candidate") || error("unknown role: $run_role")
const compiler_root = realpath(get(ENV, "RK_INNER_PE_COMPILER_ROOT", root))
const compiler_sha = _require_clean_detached_candidate(compiler_root)
const expected_compiler_sha = get(ENV, "RK_INNER_PE_COMPILER_SHA",
    run_role == "candidate" ? driver_sha : "")
occursin(r"^[0-9a-f]{40}$", expected_compiler_sha) ||
    error("an explicit full RK_INNER_PE_COMPILER_SHA is required for a baseline")
compiler_sha == expected_compiler_sha || error("loaded compiler revision differs from requested SHA")
realpath(dirname(dirname(pathof(ReactiveKernels)))) == compiler_root ||
    error("loaded ReactiveKernels must come from the selected exact compiler checkout")
realpath(dirname(dirname(pathof(ReactiveKernelsDistributionKernels)))) ==
    realpath(joinpath(compiler_root, "packages", "ReactiveKernelsDistributionKernels")) ||
    error("distribution source must come from the selected exact compiler checkout")
const paired_repeats = parse(Int, get(ENV, "RK_INNER_PE_REPEATS", "9"))
paired_repeats > 0 || error("RK_INNER_PE_REPEATS must be positive")
const process_pair_id = get(ENV, "RK_INNER_PE_PAIR_ID", "unpaired")
const process_order = parse(Int, get(ENV, "RK_INNER_PE_PROCESS_ORDER", "0"))
(process_pair_id == "unpaired" ? process_order == 0 : process_order in (1, 2)) ||
    error("a process pair needs a PAIR_ID and PROCESS_ORDER=1 or 2")
const match_receipt_path = get(ENV, "RK_INNER_PE_MATCH_RECEIPT", "")
(process_order == 2) == !isempty(match_receipt_path) ||
    error("only PROCESS_ORDER=2 must supply RK_INNER_PE_MATCH_RECEIPT")
const match_receipt_bytes = isempty(match_receipt_path) ? UInt8[] : read(match_receipt_path)
const match_receipt = isempty(match_receipt_bytes) ? nothing : TOML.parse(String(copy(match_receipt_bytes)))

function source_receipts()
    paths = ("benchmark/inner_plate_partial_evaluation.jl",
        "packages/ReactiveKernelsPPLExamples/src/_ppl_source_authority.jl",
        "packages/ReactiveKernelsPPLExamples/src/surgical.jl",
        "packages/ReactiveKernelsDistributionKernels/src/distribution_kernel_sources.jl")
    Dict(path => bytes2hex(sha256(read(joinpath(root, path)))) for path in paths)
end
const source_hashes = source_receipts()
const distribution_source = "packages/ReactiveKernelsDistributionKernels/src/distribution_kernel_sources.jl"
bytes2hex(sha256(read(joinpath(compiler_root, distribution_source)))) ==
    source_hashes[distribution_source] || error("distribution endpoint source differs from fixture checkout")

function dependency_receipts()
    Dict(string(uuid) => Dict(
        "name" => info.name, "version" => string(info.version),
        "tree_hash" => something(info.tree_hash, ""),
        "git_revision" => something(info.git_revision, ""),
        "source" => info.source,
        "path_sha" => info.is_tracking_path ?
            _require_clean_detached_candidate(info.source) : "")
        for (uuid, info) in Pkg.dependencies())
end

function comparable_dependencies(dependencies)
    # Only these two path dependencies intentionally change checkout/revision.
    # Their versions still match; the consumed endpoint source is hashed above.
    Dict(uuid => Dict(k => v for (k, v) in info if
        !(info["name"] in ("ReactiveKernels", "ReactiveKernelsDistributionKernels") &&
          k in ("source", "path_sha"))) for (uuid, info) in dependencies)
end

function verify_process_pair(output)
    match_receipt === nothing && return
    peer = match_receipt
    for key in ("driver_sha", "sources", "julia", "backend", "threads", "cpu", "host",
                "process_pair_id", "paired_repeats")
        output[key] == peer[key] || error("process-pair mismatch: $key")
    end
    peer["process_order"] == 1 || error("matched receipt must be the first process")
    peer["role"] != run_role || error("process pair must compare baseline with candidate")
    peer["compiler_sha"] != compiler_sha || error("process pair must compare different compiler revisions")
    comparable_dependencies(output["dependencies"]) == comparable_dependencies(peer["dependencies"]) ||
        error("process-pair dependency identities differ")
    current_on = Dict(row["case"] => row for row in output["rows"] if row["bound"])
    peer_on = Dict(row["case"] => row for row in peer["rows"] if row["bound"])
    Set(keys(current_on)) == Set(keys(peer_on)) || error("process-pair cases differ")
    for (label, row) in current_on
        previous = peer_on[label]
        row["input_sha256"] == previous["input_sha256"] || error("process-pair inputs/query differ: $label")
        @test row["value"] ≈ previous["value"] rtol=2e-12 atol=1e-10
        @test row["gradient"] ≈ previous["gradient"] rtol=2e-11 atol=1e-10
    end
    output["matched_process_receipt_sha256"] = bytes2hex(sha256(match_receipt_bytes))
end

@kernel normalizer(q::Vector{Float64}, observed::Vector{Int}, trials::Vector{Int}) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(observed, trials, parameter) do y, n, theta
        log_choose::Float64 = loggamma(n + 1.0) - loggamma(y + 1.0) - loggamma(n - y + 1.0)
        result::Float64 = log_choose * theta
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel binomial_plate(q::Vector{Float64}, observed::Vector{Int}, trials::Vector{Int}) = begin
    probability::Float64 = 1 / (1 + exp(-sum(q)))
    pointwise = plate(observed, trials, probability) do y, n, p
        binomial(n, p).logpdf(y)
    end
    total::Float64 = sum(pointwise)
end

@kernel cheap(q::Vector{Float64}, data::Vector{Float64}, a::Float64, b::Float64) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(data, a, b, parameter) do x, scale, offset, theta
        transformed::Float64 = scale * x + offset
        result::Float64 = transformed * theta
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel cheap_raw_live(q::Vector{Float64}, data::Vector{Float64}, a::Float64, b::Float64) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(data, a, b, parameter) do x, scale, offset, theta
        transformed::Float64 = scale * x + offset
        result::Float64 = transformed * theta + x
        result
    end
    total::Float64 = sum(pointwise)
end

# Consume the existing source authority and surgical example unchanged, without
# loading every unrelated PPL example as part of this bounded acceptance driver.
module ReactiveKernelsPPLExamples
using ReactiveKernels: KernelSpec, PreparedKernel
include("../packages/ReactiveKernelsPPLExamples/src/_ppl_source_authority.jl")
include("../packages/ReactiveKernelsPPLExamples/src/surgical.jl")
end

invoke_kernel(kernel, args) = kernel(args...)
invoke_gradient(prepared, buffer, args) = ad_value_and_gradient!(prepared, buffer, args...)
function load_observation()
    # Boundary samples and smoothed load averages cannot prove an interval was
    # quiet. /proc process visibility can also be scoped by the runner.
    observation = Dict{String,Any}("unix_s" => time(), "monotonic_ns" => time_ns())
    observation["proc_loadavg"] = Sys.islinux() ? strip(read("/proc/loadavg", String)) : "unavailable"
    observation
end

function measure(f, args...)
    before = load_observation()
    trial = @benchmark $f($args...) seconds=0.2 samples=1000 evals=1
    after = load_observation()
    # BenchmarkTools retains minimum memory/allocations across all samples;
    # even median(trial) carries those minima, not median allocation counts.
    Dict("minimum_ns" => minimum(trial).time, "median_ns" => median(trial).time,
        "minimum_bytes" => trial.memory, "minimum_allocations" => trial.allocs,
        "sample_count" => length(trial.times), "evals_per_sample" => trial.params.evals,
        "sample_ns" => copy(trial.times), "sample_gc_ns" => copy(trial.gctimes),
        "load_before" => before, "load_after" => after)
end

# Exact logical input/query evidence shared across both compiler revisions.
function input_receipt(q, data, active, want)
    io = IOBuffer()
    TOML.print(io, Dict("q" => q, "data" => Dict(String(k) => v for (k, v) in pairs(data)),
        "active" => String(active), "want" => String(want)); sorted=true)
    bytes2hex(sha256(take!(io)))
end

function dump_inner(label, p)
    println("INNER_PLAN ", label)
    for r in p.recipes
        r.op isa ReactiveKernels._AuthoredPlateOp || continue
        println("plate ", only(r.outputs).name, " inputs=", map(v -> v.name, r.inputs))
        for cell in plate_body(r).recipes
            println("  ", map(v -> v.name, cell.outputs), " <- ",
                    map(v -> v.name, cell.inputs), " source=", cell.source)
        end
    end
end

# These are operand-byte footprints, not hardware DRAM counters. Axes-only
# operands retain storage but cause no scalar loads in the generated loop.
function used_inputs(p)
    used = Set{Int}()
    for r in p.recipes
        if r.op isa ReactiveKernels._AuthoredPlateOp
            inner = plate_body(r)
            needed = Set(canon_id(inner.graph, v.id)
                         for cell in inner.recipes for v in cell.inputs)
            union!(needed, (canon_id(inner.graph, v.id) for v in inner.want))
            for (v, outer) in zip(inner.have, r.inputs)
                canon_id(inner.graph, v.id) in needed &&
                    push!(used, canon_id(p.graph, outer.id))
            end
        else
            union!(used, (canon_id(p.graph, v.id) for v in r.inputs))
        end
    end
    used
end

function run_case(label, spec, q, data; active = :q, want = :total)
    backend_name == "native" && reactant_loaded() && error(
        "Reactant was loaded into the native-only measurement process")
    reference_value = nothing
    reference_gradient = nothing
    results = Dict[]
    states = NamedTuple[]
    original = plan(spec; want)
    original_used = used_inputs(original)
    dump_inner(label * " original", original)
    for bound in (false, true)
        args = bound ? (q,) : (q, values(data)...)
        prep = @timed prepare(spec; want, bound = bound ? data : NamedTuple())
        kernel = prep.value
        first_call = @timed kernel(args...)
        ad_prep = @timed prepare_ad(kernel, ad_backend, args...; active)
        prepared = ad_prep.value
        gradient = similar(q)
        first_ad = @timed ad_value_and_gradient!(prepared, gradient, args...)
        value = first_call.value
        if !bound
            reference_value, reference_gradient = value, copy(gradient)
        else
            @test value ≈ reference_value rtol=2e-12 atol=1e-10
            @test gradient ≈ reference_gradient rtol=2e-11 atol=1e-10
            dump_inner(label * " residual", kernel.plan)
        end
        # The baseline's OFF path establishes identical value/gradient controls,
        # but only baseline ON belongs in the cross-revision timing comparison.
        !bound && run_role == "baseline" && continue
        constants = [r for r in kernel.plan.recipes if r.op isa ReactiveKernels._BoundConstant]
        caches = [r for r in constants if startswith(String(only(r.outputs).name), "bound_plate_")]
        cache_bytes = sum(sizeof(r.op.value) for r in caches if r.op.value isa AbstractArray; init=0)
        residual_used = used_inputs(kernel.plan)
        eliminated = sum(sizeof(value) for (name, value) in pairs(data)
            if value isa AbstractArray && canon_id(original.graph, port(spec, name).id) in original_used &&
               !(canon_id(original.graph, port(spec, name).id) in residual_used); init=0)
        row = Dict{String,Any}(
            "case" => label, "bound" => bound, "backend" => backend_name,
            "role" => run_role, "input_sha256" => input_receipt(q, data, active, want),
            "host_context" => reactant_loaded() ? "reactant_loaded" : "native_only",
            "prepare_s" => prep.time, "prepare_bytes" => prep.bytes,
            "native_first_call_s" => first_call.time,
            "ad_prepare_s" => ad_prep.time, "ad_first_call_s" => first_ad.time,
            "value" => value, "gradient" => copy(gradient),
            "cache_array_bytes" => cache_bytes, "raw_array_bytes_eliminated" => eliminated,
            "net_array_operand_bytes" => cache_bytes - eliminated,
            "retained_constant_bytes" => Base.summarysize(Tuple(r.op.value for r in constants)),
            "repeats" => Dict[])
        if backend_name == "native"
            primal = (invoke_kernel, kernel, args)
            derivative = (invoke_gradient, prepared, gradient, args)
        else
            traced = map(Reactant.to_rarray, args)
            compilation = @timed Reactant.compile(kernel, traced; sync=true)
            ad_compilation = @timed compile_ad_value_and_gradient(prepared, traced...; sync=true)
            compiled, compiled_ad = compilation.value, ad_compilation.value
            @test Float64(compiled(traced...)) ≈ value rtol=2e-11 atol=1e-9
            rv, rg = compiled_ad(traced...)
            @test Float64(rv) ≈ value rtol=2e-11 atol=1e-9
            @test Array(rg) ≈ gradient rtol=2e-10 atol=1e-9
            row["reactant_compile_s"] = compilation.time
            row["reactant_ad_compile_s"] = ad_compilation.time
            primal = (invoke_kernel, compiled, traced)
            derivative = (invoke_kernel, compiled_ad, traced)
        end
        push!(states, (; row, primal, derivative))
        push!(results, row)
    end
    # Warm both variants before timing either. Alternate variant and metric
    # order; preserve each trial, not just the best repeat or a ratio of minima.
    for repeat in 1:paired_repeats
        order = isodd(repeat + process_order) ? collect(eachindex(states)) : reverse(collect(eachindex(states)))
        for (position, index) in enumerate(order)
            state = states[index]
            sample = Dict{String,Any}("repeat" => repeat, "variant_position" => position,
                "metric_order" => isodd(repeat) ? ["primal", "value_gradient"] : ["value_gradient", "primal"])
            for metric in sample["metric_order"]
                sample[metric] = measure((metric == "primal" ? state.primal : state.derivative)...)
            end
            push!(state.row["repeats"], sample)
            println("MEASUREMENT case=", label, " role=", run_role,
                " bound=", state.row["bound"], " repeat=", repeat,
                " primal_median_ns=", sample["primal"]["median_ns"],
                " value_gradient_median_ns=", sample["value_gradient"]["median_ns"])
        end
    end
    results
end

function main()
    dependencies = dependency_receipts()
    if match_receipt !== nothing
        comparable_dependencies(dependencies) == comparable_dependencies(match_receipt["dependencies"]) ||
            error("process-pair dependency identities differ before measurement")
    end
    started = load_observation()
    rows = Dict[]
    sizes = parse.(Int, split(get(ENV, "RK_INNER_PE_SIZES", "12,4096,100000"), ','))
    for n in sizes
        observed = [mod(i, 7) for i in 1:n]
        trials = [20 + mod(i, 13) for i in 1:n]
        data = collect(range(0.1, 2.0; length=n))
        append!(rows, run_case("normalizer/$n", normalizer, [0.3], (; observed, trials)))
        append!(rows, run_case("binomial/$n", binomial_plate, [0.3], (; observed, trials)))
        append!(rows, run_case("affine/$n", cheap, [0.3], (; data, a=1.2, b=0.4)))
        append!(rows, run_case("affine-raw-live/$n", cheap_raw_live, [0.3], (; data, a=1.2, b=0.4)))
    end
    surgical = ReactiveKernelsPPLExamples.SurgicalExample.evaluate_surgical_source()
    @test surgical.source == strip(ReactiveKernelsPPLExamples.SurgicalExample.SURGICAL_SOURCE, '\n')
    append!(rows, Base.invokelatest(run_case, "surgical/12", surgical.model,
        surgical.inputs.q, (; successes=surgical.inputs.successes, totals=surgical.inputs.totals);
        active=:unconstrained, want=:posterior))
    dependency_receipts() == dependencies || error(
        "dependency identities changed during the measurement process")
    source_receipts() == source_hashes || error("fixture source changed during measurement")
    _require_clean_detached_candidate(root) == driver_sha || error("driver HEAD changed during measurement")
    _require_clean_detached_candidate(compiler_root) == compiler_sha || error("compiler HEAD changed during measurement")
    output = Dict("julia" => string(VERSION), "package_load_s" => package_load_s,
        "driver_sha" => driver_sha, "compiler_sha" => compiler_sha, "role" => run_role,
        "dependencies" => dependencies, "backend" => backend_name,
        "process_pair_id" => process_pair_id, "process_order" => process_order,
        "paired_repeats" => paired_repeats, "sources" => source_hashes,
        "threads" => Threads.nthreads(), "cpu" => Sys.CPU_NAME,
        "host" => get(ENV, "KB_HOST", "unknown"),
        "load_start" => started, "load_end" => load_observation(),
        "load_sampling_caveat" => "Boundary samples and smoothed /proc/loadavg cannot establish a quiet interval; process visibility may be scoped.",
        "rows" => rows)
    verify_process_pair(output)
    path = get(ENV, "RK_INNER_PE_OUTPUT", "")
    isempty(path) || open(io -> TOML.print(io, output), path; write=true)
    println("INNER_PLATE_PE_COMPLETE rows=", length(rows))
end

main()
