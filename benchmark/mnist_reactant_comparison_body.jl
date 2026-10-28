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
    MNIST_LOGISTIC_SOURCE, MNIST_LOGISTIC_OPTIMIZED_SOURCE, NUM_CLASSES,
    build_mnist_logistic_graph, build_mnist_logistic_optimized_graph
import MLDatasets

include(joinpath(@__DIR__, "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

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
_observations() = parse(Int, get(
    ENV, "RK_MNIST_REACTANT_N", string(DEFAULT_MNIST_REACTANT_N)))

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
_compile_call(kernel, args::NTuple{3,Any}) =
    @compile sync = true kernel(args[1], args[2], args[3])
_compile_call(kernel, args::NTuple{4,Any}) =
    @compile sync = true kernel(args[1], args[2], args[3], args[4])
_compile_call(kernel, args::NTuple{5,Any}) =
    @compile sync = true kernel(args[1], args[2], args[3], args[4], args[5])

function _diagnostic(err)
    line = first(split(sprint(showerror, err), '\n'))
    length(line) <= 800 ? line : first(line, 797) * "..."
end

function _load_mnist(n)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    train = MLDatasets.MNIST(split = :train)
    total = size(train.features, 3)
    n <= total || error("requested $n MNIST images but only $total are available")
    pixels = reshape(train.features[:, :, 1:n], 28 * 28, n)   # 784×n Float32 in [0,1]
    X = Matrix{Float64}(transpose(pixels))                    # n×784
    y = Int.(train.targets[1:n]) .+ 1                         # one-based classes
    X, y
end

_reactant_want(outcome) = outcome == "joint" ? :density : Symbol(outcome)

function _definition(model, configuration, boundary, outcome,
                     unconstrained, W, b, X, y)
    state, reason = matrix_support(configuration, boundary, outcome)
    descriptions = Dict(
        "joint" => "full joint over the selected parameter boundary",
        "prior" => "standard-normal coefficient log prior",
        "likelihood" => "summed softmax categorical log likelihood",
        "pointwise" => "per-observation softmax categorical log likelihoods",
    )
    state == "supported" && configuration.compiler == "reactant" || return (;
        boundary, outcome, description = descriptions[outcome], state,
        reason, kernel = nothing, args = (),
    )
    packed = boundary == "packed_unconstrained"
    data_dependent = outcome != "prior"
    have = packed ?
        (data_dependent ? (:unconstrained, :X, :y, :num_classes) :
                          (:unconstrained,)) :
        (data_dependent ? (:W, :b, :X, :y, :num_classes) : (:W, :b))
    bound = configuration.data == "bound" ?
        (; X, y, num_classes = NUM_CLASSES) : NamedTuple()
    kernel = prepare(model; have, want = _reactant_want(outcome), bound)
    args = if packed
        data_dependent && configuration.data == "unbound" ?
            (unconstrained, X, y, NUM_CLASSES) : (unconstrained,)
    else
        data_dependent && configuration.data == "unbound" ?
            (W, b, X, y, NUM_CLASSES) : (W, b)
    end
    (; boundary, outcome, description = descriptions[outcome], state, reason,
       kernel, args)
end

function _assert_primal_matrix(primal_receipt, model, configuration,
                               boundary, outcome, expected_state)
    get(primal_receipt, "schema", "") == "mnist-logistic-primal-v3" ||
        error("unexpected MNIST primal receipt schema")
    protocol = primal_receipt["protocol"]
    Tuple(protocol["input_boundaries"]) == MNIST_REACTANT_BOUNDARIES ||
        error("Reactant benchmark input boundaries drifted from the primal receipt")
    Tuple(protocol["outcomes"]) == MNIST_REACTANT_OUTCOMES ||
        error("Reactant benchmark outcomes drifted from the primal receipt")
    native_configuration = configuration.data == "bound" ?
        "primal_native_bound" : "primal_native"
    matches = filter(primal_receipt["measurements"]) do row
        row["provider"] == "rk" && row["model"] == model &&
            row["configuration"] == native_configuration &&
            row["boundary"] == boundary && row["outcome"] == outcome
    end
    length(matches) == 1 || error(
        "primal receipt does not contain exactly one $model / " *
        "$native_configuration / $boundary / $outcome row")
    get(only(matches), "state", "") == expected_state || error(
        "native support drifted for $model / $native_configuration / " *
        "$boundary / $outcome")
    nothing
end

function run_comparison()
    repo = normpath(joinpath(@__DIR__, ".."))
    rounds = _rounds()
    n = _observations()
    rounds >= 1 || error("round count must be positive")

    data_load_seconds = @elapsed ((X, y) = _load_mnist(n))
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1

    # The exact coefficient point of the matched primal receipt: small enough
    # that every backend agreed there before timing; unchanged here so the two
    # receipts describe the same evaluation.
    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    unconstrained = vcat(vec(W), b)

    rk_models = nothing
    preparation_seconds = @elapsed rk_models = (
        (name = "idiomatic", graph = build_mnist_logistic_graph()),
        (name = "vcat_free", graph = build_mnist_logistic_optimized_graph()),
    )
    reactant_configurations = filter(
        configuration -> configuration.differentiation == "primal" &&
            configuration.compiler == "reactant",
        MNIST_RK_CONFIGURATIONS,
    )

    primal_receipt_path =
        joinpath("benchmark", "receipts", "mnist-logistic-primal-v3.toml")
    primal_path = get(
        ENV, "RK_MNIST_PRIMAL_RECEIPT",
        joinpath(repo, primal_receipt_path))
    isfile(primal_path) || error(
        "the published primal receipt is required: $primal_path")
    primal_receipt = TOML.parsefile(primal_path)

    measurements = Dict{String,Any}[]
    for model_definition in rk_models,
        configuration in reactant_configurations,
        boundary in MNIST_REACTANT_BOUNDARIES,
        outcome in MNIST_REACTANT_OUTCOMES
        definition = nothing
        kernel_preparation_seconds = @elapsed definition = _definition(
            model_definition.graph, configuration, boundary, outcome,
            unconstrained, W, b, X, y)
        _assert_primal_matrix(
            primal_receipt, model_definition.name, configuration,
            boundary, outcome, definition.state)
        row = Dict{String,Any}(
            "provider" => "rk",
            "model" => model_definition.name,
            "configuration" => configuration.id,
            "boundary" => boundary,
            "outcome" => outcome,
            "description" => definition.description,
            "state" => definition.state,
            "kernel_preparation_seconds" => kernel_preparation_seconds,
        )
        if definition.state != "supported"
            row["reason"] = definition.reason
            push!(measurements, row)
            println("model=$(model_definition.name) configuration=$(configuration.id) " *
                    "boundary=$boundary outcome=$outcome state=$(definition.state)")
            continue
        end
        native_value = _comparable(
            definition.outcome, definition.kernel(definition.args...))
        row["native_control"] = _measurement(
            definition.kernel, definition.args...; rounds)

        # The public partially-evaluated kernel remains q-only (or W,b-only).
        # Large bound arrays cross the compiler ABI as trailing hidden device
        # operands so Reactant does not encode the full dataset as program
        # literals. Passing device handles is included in the compiled call;
        # host-to-device transfer remains setup outside steady-state timing.
        compiler_kernel, bound_arrays = configuration.data == "bound" ?
            ReactiveKernels._externalize_bound_arrays(definition.kernel) :
            (definition.kernel, ())
        compiler_args = (definition.args..., bound_arrays...)
        row["reactant_externalized_bound_array_count"] = length(bound_arrays)

        traced_args = nothing
        transfer_seconds = @elapsed traced_args = _trace_args(compiler_args)
        row["reactant_transfer_seconds"] = transfer_seconds
        compile_started = time_ns()
        try
            compiled = _compile_call(compiler_kernel, traced_args)
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
            row["result"] = _measurement(compiled, traced_args...; rounds)
        catch err
            row["reactant_compile_seconds"] = get(
                row, "reactant_compile_seconds",
                Float64(time_ns() - compile_started) / 1e9)
            row["state"] = "unsupported_runtime"
            row["reason"] = _diagnostic(err)
        end
        push!(measurements, row)
        println("model=$(model_definition.name) configuration=$(configuration.id) " *
                "boundary=$boundary outcome=$outcome state=$(row["state"])")
    end

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "mnist_logistic.jl")
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")
    receipt = Dict{String,Any}(
        "schema" => "mnist-reactant-v2",
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
            "enzyme_jll_version" => _package_version("Enzyme_jll"),
            "gpucompiler_version" => _package_version("GPUCompiler"),
            "llvm_version" => _package_version("LLVM"),
            "benchmarktools_version" => _package_version("BenchmarkTools"),
            "mldatasets_version" => _package_version("MLDatasets"),
            "julia_version" => string(VERSION),
            "source_authority_path" => source_path,
            "source_authority_blob" =>
                _git(repo, "rev-parse", "$candidate_sha:$source_path"),
            "idiomatic_source_text_sha256" =>
                bytes2hex(sha256(MNIST_LOGISTIC_SOURCE)),
            "vcat_free_source_text_sha256" =>
                bytes2hex(sha256(MNIST_LOGISTIC_OPTIMIZED_SOURCE)),
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
        "protocol" => Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "data" => "MLDatasets MNIST train split, first N images",
            "num_observations" => n,
            "num_features" => features,
            "num_classes" => NUM_CLASSES,
            "source_reused" => true,
            "matrix_source" => primal_receipt_path,
            "input_boundaries" => collect(MNIST_REACTANT_BOUNDARIES),
            "outcomes" => collect(MNIST_REACTANT_OUTCOMES),
            "models" => collect(MNIST_MODELS),
            "matrix_layout" =>
                "long-form provider/model/configuration/boundary/outcome rows",
            "rk_configurations" => [
                configuration.id for configuration in reactant_configurations],
            "bound_ports" => ["X", "y", "num_classes"],
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
            "bound_array_compiler_abi" =>
                "public residual kernel stays data-bound; array-valued bound constants are transferred once and passed as trailing hidden device operands to avoid dataset-sized compiler literals",
            "unsupported_cells_recorded" => true,
            "gradients_included" => false,
            "generated_predictions_included" => false,
            "parity_rtol" => 1e-9,
            "parity_atol" => 1e-9,
        ),
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

get(ENV, "RK_MNIST_REACTANT_DEFINITIONS_ONLY", "") == "1" || run_comparison()
