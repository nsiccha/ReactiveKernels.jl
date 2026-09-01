# Inner body for the pinned MNIST native-RK-AD / Reactant-compiled-AD
# comparison. This is the AD analog of mnist_reactant_comparison_body.jl
# (primal): it reuses the exact authored model source and the SAME derivative
# outcome/boundary protocol published by the AD-only receipt
# (benchmark/receipts/mnist-logistic-ad-v1.toml). It never copies a prior,
# likelihood, or AD evaluator: it selects RK graph boundaries with
# `prepare`/`prepare_ad` and consumes the first-class RK verbs
# `ad_value_and_gradient!` (native) and `compile_ad_value_and_gradient`
# (Reactant-compiled) — no hand-rolled AD-through-Reactant glue.

using BenchmarkTools
using Dates
using Pkg
using Random
using SHA
using Statistics
using TOML
using Reactant
import Enzyme
using DifferentiationInterface: AutoEnzyme
using ReactiveKernels
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    MNIST_LOGISTIC_SOURCE, NUM_CLASSES, build_mnist_logistic_graph
import MLDatasets

include(joinpath(@__DIR__, "_mnist_dataset_profiles.jl"))

const DEFAULT_MNIST_REACTANT_AD_ROUNDS = 10
# The published receipt fits the full MNIST training split; RK_MNIST_REACTANT_AD_N
# overrides it for a quicker local reproduction.
const DEFAULT_MNIST_REACTANT_AD_N = 60000
const MNIST_REACTANT_AD_BOUNDARIES =
    ("packed_unconstrained", "structured_parameters")
const MNIST_REACTANT_AD_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
# The differentiable scalar cells published by mnist-logistic-ad-v1 (native RK
# AD support). Reactant-compiled AD is a subset of these: it additionally needs
# the primal kernel to compile through Reactant, which is discovered by
# attempting the compile, never hard-coded.
const MNIST_REACTANT_AD_SCALAR_SUPPORTED = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
))

const AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))
_mnist_reactant_ad_generator_sha256(path) = bytes2hex(sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name absent from the benchmark environment")
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

_rounds() = parse(Int, get(
    ENV, "RK_MNIST_REACTANT_AD_ROUNDS", string(DEFAULT_MNIST_REACTANT_AD_ROUNDS)))
_observations(profile) = parse(Int, get(
    ENV, "RK_MNIST_REACTANT_AD_N", string(_mnist_default_observations(profile))))

# Same headline estimator as the matched native-AD receipt: the minimum of
# per-round BenchmarkTools minimums (uncontended cost; medians retained).
function _measurement(f; rounds::Int)
    benchmark = @benchmarkable $f()
    times_ns = Float64[]; bytes = Int[]; allocs = Int[]
    for _ in 1:rounds
        estimate = minimum(run(benchmark; samples = 200, seconds = 0.2))
        push!(times_ns, estimate.time)
        push!(bytes, estimate.memory)
        push!(allocs, estimate.allocs)
    end
    Dict(
        "times_ns" => times_ns, "min_ns" => minimum(times_ns),
        "median_ns" => median(times_ns),
        "bytes" => bytes, "median_bytes" => Int(median(bytes)),
        "allocs" => allocs, "median_allocs" => Int(median(allocs)),
    )
end

_trace_value(value::AbstractArray) = Reactant.to_rarray(value)
# The class count is a shape parameter of the compiled program; it stays a
# static compile-time constant rather than a traced number.
_trace_value(value::Integer) = value
_trace_value(value::Number) = Reactant.to_rarray(value; track_numbers = true)
_trace_args(args::Tuple) = map(_trace_value, args)

function _diagnostic(err)
    line = first(split(sprint(showerror, err), '\n'))
    length(line) <= 800 ? line : first(line, 797) * "..."
end

# The differentiable-scalar cell definitions, selecting RK graph boundaries
# exactly as the published mnist-logistic-ad-v1 receipt does; `nothing` marks a
# non-differentiable-scalar cell (vector pointwise, or the structured boundary
# whose two active ports have no public multi-active AD contract).
function _ad_definition(model, boundary, outcome, unconstrained, X, y)
    boundary == "packed_unconstrained" || return nothing
    if outcome == "joint"
        return (kernel = prepare(model;
                    have = (:unconstrained, :X, :y, :num_classes),
                    want = :density),
                args = (unconstrained, X, y, NUM_CLASSES),
                active = :unconstrained,
                description = "gradient of the full joint w.r.t. the packed coefficient vector")
    elseif outcome == "prior"
        return (kernel = prepare(model; have = :unconstrained, want = :prior),
                args = (unconstrained,), active = :unconstrained,
                description = "gradient of the standard-normal coefficient log prior")
    elseif outcome == "likelihood"
        return (kernel = prepare(model;
                    have = (:unconstrained, :X, :y, :num_classes),
                    want = :likelihood),
                args = (unconstrained, X, y, NUM_CLASSES),
                active = :unconstrained,
                description = "gradient of the summed softmax categorical log likelihood")
    end
    nothing
end

# Unsupported-cell diagnostics, mirroring mnist-logistic-ad-v1's reasons.
function _unsupported_reason(boundary, outcome)
    outcome == "pointwise" && return (
        "pointwise is vector-valued; no matched public Jacobian/VJP contract, " *
        "so it stays unsupported rather than inventing a fused case")
    boundary == "structured_parameters" && return (
        "the structured (W, b) boundary has two active ports; no public " *
        "multi-active AD contract is invented for it")
    "unsupported matrix cell"
end

function _assert_ad_matrix(ad_receipt)
    get(ad_receipt, "schema", "") == "mnist-logistic-ad-v1" ||
        error("unexpected MNIST AD receipt schema")
    protocol = ad_receipt["protocol"]
    Tuple(protocol["input_boundaries"]) == MNIST_REACTANT_AD_BOUNDARIES ||
        error("Reactant AD benchmark boundaries drifted from the AD receipt")
    Tuple(protocol["outcomes"]) == MNIST_REACTANT_AD_OUTCOMES ||
        error("Reactant AD benchmark outcomes drifted from the AD receipt")
    # The AD receipt's supported scalar inventory must equal ours, so this
    # benchmark differentiates exactly the cells the AD receipt does.
    supported = Set{Tuple{String,String}}()
    for row in ad_receipt["measurements"]
        get(row, "supported", false) &&
            push!(supported, (row["boundary"], row["outcome"]))
    end
    supported == MNIST_REACTANT_AD_SCALAR_SUPPORTED || error(
        "AD receipt supported inventory drifted from this benchmark's cells")
    nothing
end

function run_comparison()
    repo = normpath(joinpath(@__DIR__, ".."))
    rounds = _rounds()
    dataset_profile = _mnist_dataset_profile()
    n = _observations(dataset_profile)
    rounds >= 1 || error("round count must be positive")

    dataset_metadata = nothing
    data_load_seconds = @elapsed begin
        X, y, dataset_metadata = _load_mnist_dataset(
            dataset_profile, n; wren_reference = _mnist_wren_reference_path())
    end
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1

    # The exact coefficient point of the matched primal and AD receipts.
    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    unconstrained = vcat(vec(W), b)

    model = nothing
    preparation_seconds = @elapsed model = build_mnist_logistic_graph()

    ad_path = joinpath(@__DIR__, "receipts", "mnist-logistic-ad-v1.toml")
    isfile(ad_path) || error("the published AD receipt is required: $ad_path")
    ad_receipt = TOML.parsefile(ad_path)
    _assert_ad_matrix(ad_receipt)

    measurements = Dict{String,Any}[]
    for boundary in MNIST_REACTANT_AD_BOUNDARIES,
        outcome in MNIST_REACTANT_AD_OUTCOMES
        row = Dict{String,Any}("boundary" => boundary, "outcome" => outcome)
        definition = _ad_definition(model, boundary, outcome, unconstrained, X, y)
        if definition === nothing
            row["description"] = _unsupported_reason(boundary, outcome)
            row["rk_native_ad_supported"] = false
            row["rk_native_ad_error"] = _unsupported_reason(boundary, outcome)
            row["rk_reactant_ad_supported"] = false
            row["rk_reactant_ad_error"] = _unsupported_reason(boundary, outcome)
            push!(measurements, row)
            println("boundary=$boundary outcome=$outcome unsupported")
            continue
        end

        row["description"] = definition.description
        row["active_port"] = String(definition.active)
        row["rk_native_ad_supported"] = true

        # Native RK AD reference (prepare_ad + ad_value_and_gradient!). The
        # preparation is outside steady-state timing.
        prepare_seconds = @elapsed prepared = prepare_ad(
            definition.kernel, AD_BACKEND, definition.args...;
            active = definition.active)
        row["ad_preparation_seconds"] = prepare_seconds
        native_gradient = similar(definition.args[1])
        native_value, native_gradient = ad_value_and_gradient!(
            prepared, native_gradient, definition.args...)
        native_gradient_ref = collect(Float64, native_gradient)
        native_call = let prepared = prepared, native_gradient = native_gradient,
                          args = definition.args
            () -> ad_value_and_gradient!(prepared, native_gradient, args...)
        end
        row["rk_native_ad"] = _measurement(native_call; rounds)

        # Reactant-compiled AD (compile_ad_value_and_gradient). Tracing, compile,
        # and first execution are all timed separately, outside steady state.
        traced = nothing
        transfer_seconds = @elapsed traced = _trace_args(definition.args)
        row["reactant_transfer_seconds"] = transfer_seconds
        compile_started = time_ns()
        try
            compiled = compile_ad_value_and_gradient(prepared, traced...)
            row["reactant_ad_compile_seconds"] =
                Float64(time_ns() - compile_started) / 1e9
            compiled_value = compiled_gradient = nothing
            row["reactant_first_execution_seconds"] = @elapsed begin
                compiled_value, compiled_gradient = compiled(traced...)
            end
            reactant_gradient = collect(Float64, Array(compiled_gradient))
            reactant_value = Float64(compiled_value)
            length(reactant_gradient) == length(native_gradient_ref) ||
                error("Reactant AD gradient length parity failed")
            absolute = maximum(abs.(reactant_gradient .- native_gradient_ref))
            row["max_abs_error"] = absolute
            row["max_rel_error"] = maximum(
                abs.(reactant_gradient .- native_gradient_ref) ./
                max.(1.0, abs.(native_gradient_ref)))
            row["value_abs_error"] = abs(reactant_value - Float64(native_value))
            row["value_rel_error"] = row["value_abs_error"] /
                max(1.0, abs(Float64(native_value)))
            (row["max_rel_error"] <= 1e-9 && row["value_rel_error"] <= 1e-9) ||
                error("native/Reactant AD parity failed")
            reactant_call = let compiled = compiled, traced = traced
                () -> compiled(traced...)
            end
            row["rk_reactant_ad"] = _measurement(reactant_call; rounds)
            row["rk_reactant_ad_supported"] = true
        catch err
            row["reactant_ad_compile_seconds"] = get(
                row, "reactant_ad_compile_seconds",
                Float64(time_ns() - compile_started) / 1e9)
            row["rk_reactant_ad_supported"] = false
            row["rk_reactant_ad_error"] = _diagnostic(err)
        end
        push!(measurements, row)
        println("boundary=$boundary outcome=$outcome " *
                "reactant_ad=$(row["rk_reactant_ad_supported"])")
    end

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "mnist_logistic.jl")
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")
    receipt = Dict{String,Any}(
        "schema" => "mnist-reactant-ad-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => candidate_sha,
            "reactivekernels_dirty" => false,
            "reactivekernels_version" => _package_version("ReactiveKernels"),
            "reactivekernelspplexamples_version" =>
                _package_version("ReactiveKernelsPPLExamples"),
            "reactivekernelsdistributionkernels_version" =>
                _package_version("ReactiveKernelsDistributionKernels"),
            "reactant_version" => _package_version("Reactant"),
            "reactant_jll_version" => _package_version("Reactant_jll"),
            "enzyme_version" => _package_version("Enzyme"),
            "differentiationinterface_version" =>
                _package_version("DifferentiationInterface"),
            "benchmarktools_version" => _package_version("BenchmarkTools"),
            "mldatasets_version" => _package_version("MLDatasets"),
            "julia_version" => string(VERSION),
            "source_authority_path" => source_path,
            "source_authority_blob" =>
                _git(repo, "rev-parse", "$candidate_sha:$source_path"),
            "source_text_sha256" => bytes2hex(sha256(MNIST_LOGISTIC_SOURCE)),
            "ad_receipt_path" => "benchmark/receipts/mnist-logistic-ad-v1.toml",
            "ad_receipt_sha256" => _mnist_reactant_ad_generator_sha256(ad_path),
            "ad_receipt_reactivekernels_sha" =>
                ad_receipt["pins"]["reactivekernels_sha"],
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
            "reactant_backend" => "default CPU",
        ),
        "setup" => Dict(
            "environment_seconds" => parse(Float64, get(
                ENV, "RK_MNIST_REACTANT_AD_ENV_SETUP_SECONDS", "0")),
            "package_precompile_seconds" => parse(Float64, get(
                ENV, "RK_MNIST_REACTANT_AD_PRECOMPILE_SECONDS", "0")),
            "data_load_seconds" => data_load_seconds,
            "kernel_preparation_seconds" => preparation_seconds,
        ),
        "protocol" => merge(Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "num_classes" => NUM_CLASSES,
            "source_reused" => true,
            "matrix_source" => "benchmark/receipts/mnist-logistic-ad-v1.toml",
            "input_boundaries" => collect(MNIST_REACTANT_AD_BOUNDARIES),
            "outcomes" => collect(MNIST_REACTANT_AD_OUTCOMES),
            "gradient_operation" => "value and gradient",
            "rk_native_ad_surface" => "prepare_ad + ad_value_and_gradient!",
            "rk_reactant_ad_surface" =>
                "prepare_ad + compile_ad_value_and_gradient",
            "rk_ad_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
            "parity_reference" => "native RK reverse pass (ad_value_and_gradient!)",
            "partial_evaluation_enabled" => false,
            "runtime_data_ports" => ["X", "y", "num_classes"],
            "native_and_reactant_use_same_runtime_boundary" => true,
            "rounds" => rounds,
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "estimator" => "minimum of per-round BenchmarkTools minimum times (uncontended cost; medians and raw rounds retained)",
            "reactant_sync" => true,
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "ad_preparation_in_timed_region" => false,
            "reactant_compile_time_in_timed_region" => false,
            "reactant_transfers_in_timed_region" => false,
            "reactant_readback_in_timed_region" => false,
            "first_execution_in_steady_state_region" => false,
            "unsupported_cells_recorded" => true,
            "pointwise_jacobian_or_vjp_invented" => false,
            "structured_multi_active_boundary_invented" => false,
            "parity_rtol" => 1e-9,
            "parity_atol" => 1e-9,
        ), dataset_metadata),
        "measurements" => measurements,
    )

    output = _output_path()
    if output === nothing
        TOML.print(stdout, receipt; sorted = true)
    else
        mkpath(dirname(abspath(output)))
        open(output, "w") do io
            TOML.print(io, receipt; sorted = true)
        end
        println("receipt=$(abspath(output))")
    end
    receipt
end

run_comparison()
