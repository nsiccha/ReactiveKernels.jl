# Inner body for the pinned MNIST native-RK/Reactant comparison.

using BenchmarkTools
using Dates
using Pkg
using Random
using SHA
using Statistics
using TOML
using Reactant
using Reactant: @compile
using ReactiveKernels
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    MNIST_LOGISTIC_SOURCE, NUM_CLASSES, build_mnist_logistic_graph
import MLDatasets

include(joinpath(@__DIR__, "_mnist_dataset_profiles.jl"))

const DEFAULT_MNIST_REACTANT_ROUNDS = 10
# The published receipt fits the full MNIST training split; RK_MNIST_REACTANT_N
# overrides it for a quicker local reproduction.
const DEFAULT_MNIST_REACTANT_N = 60000
const MNIST_REACTANT_BOUNDARIES = ("packed_unconstrained", "structured_parameters")
const MNIST_REACTANT_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))
_mnist_reactant_generator_sha256(path) = bytes2hex(sha256(
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
    ENV, "RK_MNIST_REACTANT_ROUNDS", string(DEFAULT_MNIST_REACTANT_ROUNDS)))
_observations(profile) = parse(Int, get(
    ENV, "RK_MNIST_REACTANT_N", string(_mnist_default_observations(profile))))

_comparable(outcome, value) = outcome == "pointwise" ?
    collect(Float64, Array(value)) : Float64(value)

# Same headline estimator as the matched primal receipt: the minimum of
# per-round BenchmarkTools minimums estimates the uncontended cost of the
# ms-scale matmul cells on a shared host; medians and raw rounds are retained.
function _measurement(f, args...; rounds::Int)
    invocation = let f = f, args = args
        () -> f(args...)
    end
    benchmark = @benchmarkable $invocation()
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

_compile_call(kernel, args::Tuple{Any}) = @compile sync = true kernel(args[1])
_compile_call(kernel, args::Tuple{Any,Any}) =
    @compile sync = true kernel(args[1], args[2])
_compile_call(kernel, args::NTuple{4,Any}) =
    @compile sync = true kernel(args[1], args[2], args[3], args[4])
_compile_call(kernel, args::NTuple{5,Any}) =
    @compile sync = true kernel(args[1], args[2], args[3], args[4], args[5])

function _diagnostic(err)
    line = first(split(sprint(showerror, err), '\n'))
    length(line) <= 800 ? line : first(line, 797) * "..."
end

function _definitions(model, unconstrained, W, b, X, y)
    packed = (
        joint = prepare(model;
            have = (:unconstrained, :X, :y, :num_classes), want = :density),
        prior = prepare(model; have = :unconstrained, want = :prior),
        likelihood = prepare(model;
            have = (:unconstrained, :X, :y, :num_classes), want = :likelihood),
        pointwise = prepare(model;
            have = (:unconstrained, :X, :y, :num_classes), want = :pointwise),
    )
    structured = (
        joint = prepare(model;
            have = (:W, :b, :X, :y, :num_classes), want = :density),
        prior = prepare(model; have = (:W, :b), want = :prior),
        likelihood = prepare(model;
            have = (:W, :b, :X, :y, :num_classes), want = :likelihood),
        pointwise = prepare(model;
            have = (:W, :b, :X, :y, :num_classes), want = :pointwise),
    )
    packed_args = (unconstrained, X, y, NUM_CLASSES)
    structured_args = (W, b, X, y, NUM_CLASSES)
    (
        (boundary = "packed_unconstrained", outcome = "joint",
         description = "full joint over the flattened sampler vector",
         kernel = packed.joint, args = packed_args),
        (boundary = "packed_unconstrained", outcome = "prior",
         description = "standard-normal coefficient log prior",
         kernel = packed.prior, args = (unconstrained,)),
        (boundary = "packed_unconstrained", outcome = "likelihood",
         description = "summed softmax categorical log likelihood",
         kernel = packed.likelihood, args = packed_args),
        (boundary = "packed_unconstrained", outcome = "pointwise",
         description = "per-observation softmax categorical log likelihoods",
         kernel = packed.pointwise, args = packed_args),
        (boundary = "structured_parameters", outcome = "joint",
         description = "full joint over the (W, b) coefficients",
         kernel = structured.joint, args = structured_args),
        (boundary = "structured_parameters", outcome = "prior",
         description = "standard-normal coefficient log prior",
         kernel = structured.prior, args = (W, b)),
        (boundary = "structured_parameters", outcome = "likelihood",
         description = "summed softmax categorical log likelihood",
         kernel = structured.likelihood, args = structured_args),
        (boundary = "structured_parameters", outcome = "pointwise",
         description = "per-observation softmax categorical log likelihoods",
         kernel = structured.pointwise, args = structured_args),
    )
end

function _assert_primal_matrix(primal_receipt, definitions)
    get(primal_receipt, "schema", "") == "mnist-logistic-v1" ||
        error("unexpected MNIST primal receipt schema")
    protocol = primal_receipt["protocol"]
    Tuple(protocol["input_boundaries"]) == MNIST_REACTANT_BOUNDARIES ||
        error("Reactant benchmark input boundaries drifted from the primal receipt")
    Tuple(protocol["outcomes"]) == MNIST_REACTANT_OUTCOMES ||
        error("Reactant benchmark outcomes drifted from the primal receipt")
    rows = primal_receipt["measurements"]
    for definition in definitions
        matches = filter(rows) do row
            row["boundary"] == definition.boundary &&
                row["outcome"] == definition.outcome
        end
        length(matches) == 1 || error(
            "primal receipt does not contain exactly one " *
            "$(definition.boundary) / $(definition.outcome) row")
        haskey(only(matches), "rk_native") || error(
            "native support drifted for " *
            "$(definition.boundary) / $(definition.outcome)")
    end
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
        GC.gc()
    end
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1

    # The exact coefficient point of the matched primal receipt: small enough
    # that every backend agreed there before timing; unchanged here so the two
    # receipts describe the same evaluation.
    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    unconstrained = vcat(vec(W), b)

    model = definitions = nothing
    preparation_seconds = @elapsed begin
        model = build_mnist_logistic_graph()
        definitions = _definitions(model, unconstrained, W, b, X, y)
    end

    primal_path = joinpath(@__DIR__, "receipts", "mnist-logistic-v1.toml")
    isfile(primal_path) || error(
        "the published primal receipt is required: $primal_path")
    primal_receipt = TOML.parsefile(primal_path)
    _assert_primal_matrix(primal_receipt, definitions)

    measurements = Dict{String,Any}[]
    for definition in definitions
        row = Dict{String,Any}(
            "boundary" => definition.boundary,
            "outcome" => definition.outcome,
            "description" => definition.description,
            "rk_native_supported" => true,
        )
        native_value = _comparable(
            definition.outcome, definition.kernel(definition.args...))
        row["rk_native"] = _measurement(
            definition.kernel, definition.args...; rounds)

        traced_args = nothing
        transfer_seconds = @elapsed traced_args = _trace_args(definition.args)
        row["reactant_transfer_seconds"] = transfer_seconds
        compile_started = time_ns()
        try
            compiled = _compile_call(definition.kernel, traced_args)
            row["reactant_compile_seconds"] =
                Float64(time_ns() - compile_started) / 1e9
            compiled_value = nothing
            row["reactant_first_execution_seconds"] = @elapsed begin
                compiled_value = compiled(traced_args...)
            end
            comparable = _comparable(definition.outcome, compiled_value)
            isapprox(comparable, native_value; rtol = 1e-9, atol = 1e-9) ||
                error("native/Reactant value parity failed")
            absolute = comparable isa Number ?
                abs(comparable - native_value) :
                maximum(abs.(comparable .- native_value))
            row["max_abs_error"] = absolute
            row["max_rel_error"] = comparable isa Number ?
                absolute / max(1.0, abs(native_value)) :
                maximum(abs.(comparable .- native_value) ./
                        max.(1.0, abs.(native_value)))
            row["rk_reactant"] = _measurement(compiled, traced_args...; rounds)
            row["rk_reactant_supported"] = true
        catch err
            row["reactant_compile_seconds"] = get(
                row, "reactant_compile_seconds",
                Float64(time_ns() - compile_started) / 1e9)
            row["rk_reactant_supported"] = false
            row["rk_reactant_error"] = _diagnostic(err)
        end
        push!(measurements, row)
        reactant_supported = row["rk_reactant_supported"]
        println("boundary=$(definition.boundary) outcome=$(definition.outcome) " *
                "reactant=$reactant_supported")
    end

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "mnist_logistic.jl")
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")
    receipt = Dict{String,Any}(
        "schema" => "mnist-reactant-v1",
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
            "benchmarktools_version" => _package_version("BenchmarkTools"),
            "mldatasets_version" => _package_version("MLDatasets"),
            "julia_version" => string(VERSION),
            "source_authority_path" => source_path,
            "source_authority_blob" =>
                _git(repo, "rev-parse", "$candidate_sha:$source_path"),
            "source_text_sha256" => bytes2hex(sha256(MNIST_LOGISTIC_SOURCE)),
            "primal_receipt_sha256" =>
                _mnist_reactant_generator_sha256(primal_path),
            "primal_receipt_reactivekernels_sha" =>
                primal_receipt["pins"]["reactivekernels_sha"],
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
                ENV, "RK_MNIST_REACTANT_ENV_SETUP_SECONDS", "0")),
            "package_precompile_seconds" => parse(Float64, get(
                ENV, "RK_MNIST_REACTANT_PRECOMPILE_SECONDS", "0")),
            "data_load_seconds" => data_load_seconds,
            "kernel_preparation_seconds" => preparation_seconds,
        ),
        "protocol" => merge(Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "num_classes" => NUM_CLASSES,
            "source_reused" => true,
            "matrix_source" => "benchmark/receipts/mnist-logistic-v1.toml",
            "input_boundaries" => collect(MNIST_REACTANT_BOUNDARIES),
            "outcomes" => collect(MNIST_REACTANT_OUTCOMES),
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
            "reactant_compile_time_in_timed_region" => false,
            "reactant_transfers_in_timed_region" => false,
            "reactant_readback_in_timed_region" => false,
            "unsupported_cells_recorded" => true,
            "gradients_included" => false,
            "generated_predictions_included" => false,
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
