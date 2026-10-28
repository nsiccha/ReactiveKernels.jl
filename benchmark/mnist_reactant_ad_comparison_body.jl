# Inner body for the pinned MNIST native-RK-AD / Reactant-compiled-AD
# comparison. This is the AD analog of mnist_reactant_comparison_body.jl
# (primal): it reuses the exact authored model source and the SAME derivative
# outcome/boundary protocol published by the AD-only receipt
# (benchmark/receipts/mnist-logistic-ad-v2.toml). It never copies a prior,
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
    MNIST_LOGISTIC_SOURCE, MNIST_LOGISTIC_OPTIMIZED_SOURCE, NUM_CLASSES,
    build_mnist_logistic_graph, build_mnist_logistic_optimized_graph
import MLDatasets

include(joinpath(@__DIR__, "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

const DEFAULT_MNIST_REACTANT_AD_ROUNDS = 10
# The published receipt fits the full MNIST training split; RK_MNIST_REACTANT_AD_N
# overrides it for a quicker local reproduction.
const DEFAULT_MNIST_REACTANT_AD_N = 60000
const MNIST_REACTANT_AD_BOUNDARIES =
    ("packed_unconstrained", "structured_parameters")
const MNIST_REACTANT_AD_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
# Native and Reactant AD consume the same declared matrix contract. Compilation
# is still attempted rather than assumed; any runtime failure remains a receipt
# row and makes the strict suite validator fail.
const AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const BOUND_AD_BACKEND = AutoEnzyme(
    ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

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
_observations() = parse(Int, get(
    ENV, "RK_MNIST_REACTANT_AD_N", string(DEFAULT_MNIST_REACTANT_AD_N)))

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

# One Reactant-AD definition over either public graph and either data-binding
# mode. Unsupported/N-A cells remain rows with reasons.
function _ad_definition(model, configuration, boundary, outcome,
                        unconstrained, X, y)
    state, reason = matrix_support(configuration, boundary, outcome)
    descriptions = Dict(
        "joint" => "gradient of the full joint w.r.t. the packed coefficient vector",
        "prior" => "gradient of the standard-normal coefficient log prior",
        "likelihood" => "gradient of the summed softmax categorical log likelihood",
        "pointwise" => "pointwise Jacobian/VJP",
    )
    state == "supported" || return (;
        state, reason, description = descriptions[outcome],
            kernel = nothing, spec = nothing, want = nothing,
            bound = NamedTuple(), args = (), active = :unconstrained,
            data_binding = configuration.data,
    )
    want = outcome == "joint" ? :density : Symbol(outcome)
    if configuration.data == "bound"
        kernel = prepare(
            model; have = (:unconstrained, :X, :y, :num_classes),
            want, bound = (; X, y, num_classes = NUM_CLASSES))
        return (;
            state, reason, description = descriptions[outcome],
            kernel, spec = nothing, want, bound = NamedTuple(),
            args = (unconstrained,), active = :unconstrained,
            data_binding = configuration.data,
        )
    end
    kernel = outcome == "prior" ?
        prepare(model; have = :unconstrained, want) :
        prepare(model; have = (:unconstrained, :X, :y, :num_classes), want)
    args = outcome == "prior" ? (unconstrained,) :
        (unconstrained, X, y, NUM_CLASSES)
    (; state, reason, description = descriptions[outcome], kernel,
       spec = nothing, want, bound = NamedTuple(), args,
       active = :unconstrained, data_binding = configuration.data)
end

function _prepare_ad_definition(definition)
    backend = definition.data_binding == "bound" ?
        BOUND_AD_BACKEND : AD_BACKEND
    prepare_ad(
        definition.kernel, backend, definition.args...;
        active = definition.active)
end

function _assert_ad_matrix(ad_receipt, model, configuration,
                           boundary, outcome, expected_state)
    get(ad_receipt, "schema", "") == "mnist-logistic-ad-v2" ||
        error("unexpected MNIST AD receipt schema")
    protocol = ad_receipt["protocol"]
    Tuple(protocol["input_boundaries"]) == MNIST_REACTANT_AD_BOUNDARIES ||
        error("Reactant AD benchmark boundaries drifted from the AD receipt")
    Tuple(protocol["outcomes"]) == MNIST_REACTANT_AD_OUTCOMES ||
        error("Reactant AD benchmark outcomes drifted from the AD receipt")
    native_configuration = configuration.data == "bound" ?
        "ad_native_bound" : "ad_native"
    matches = filter(ad_receipt["measurements"]) do row
        row["provider"] == "rk" && row["model"] == model &&
            row["configuration"] == native_configuration &&
            row["boundary"] == boundary && row["outcome"] == outcome
    end
    length(matches) == 1 || error(
        "AD receipt does not contain exactly one $model / " *
        "$native_configuration / $boundary / $outcome row")
    get(only(matches), "state", "") == expected_state || error(
        "native AD support drifted for $model / $native_configuration / " *
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

    # The exact coefficient point of the matched primal and AD receipts.
    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    unconstrained = vcat(vec(W), b)

    rk_models = nothing
    preparation_seconds = @elapsed rk_models = (
        (name = "idiomatic", graph = build_mnist_logistic_graph()),
        (name = "vcat_free", graph = build_mnist_logistic_optimized_graph()),
    )
    reactant_ad_configurations = filter(
        configuration -> configuration.differentiation == "value_and_gradient" &&
            configuration.compiler == "reactant",
        MNIST_RK_CONFIGURATIONS,
    )

    ad_receipt_path =
        joinpath("benchmark", "receipts", "mnist-logistic-ad-v2.toml")
    ad_path = get(
        ENV, "RK_MNIST_AD_RECEIPT",
        joinpath(repo, ad_receipt_path))
    isfile(ad_path) || error("the published AD receipt is required: $ad_path")
    ad_receipt = TOML.parsefile(ad_path)

    measurements = Dict{String,Any}[]
    for model_definition in rk_models,
        configuration in reactant_ad_configurations,
        boundary in MNIST_REACTANT_AD_BOUNDARIES,
        outcome in MNIST_REACTANT_AD_OUTCOMES
        definition = _ad_definition(
            model_definition.graph, configuration, boundary, outcome,
            unconstrained, X, y)
        _assert_ad_matrix(
            ad_receipt, model_definition.name, configuration,
            boundary, outcome, definition.state)
        row = Dict{String,Any}(
            "provider" => "rk", "model" => model_definition.name,
            "configuration" => configuration.id,
            "boundary" => boundary, "outcome" => outcome,
            "description" => definition.description,
            "state" => definition.state,
        )
        if definition.state != "supported"
            row["reason"] = definition.reason
            push!(measurements, row)
            println("model=$(model_definition.name) configuration=$(configuration.id) " *
                    "boundary=$boundary outcome=$outcome state=$(definition.state)")
            continue
        end

        row["active_port"] = String(definition.active)

        # Native RK AD reference (prepare_ad + ad_value_and_gradient!). The
        # preparation is outside steady-state timing.
        prepare_seconds = @elapsed prepared = _prepare_ad_definition(definition)
        row["ad_preparation_seconds"] = prepare_seconds
        native_gradient = similar(definition.args[1])
        native_value, native_gradient = ad_value_and_gradient!(
            prepared, native_gradient, definition.args...)
        native_gradient_ref = collect(Float64, native_gradient)
        native_call = let prepared = prepared, native_gradient = native_gradient,
                          args = definition.args
            () -> ad_value_and_gradient!(prepared, native_gradient, args...)
        end
        row["native_control"] = _measurement(native_call; rounds)

        # Reactant-compiled AD (compile_ad_value_and_gradient). Tracing, compile,
        # and first execution are all timed separately, outside steady state.
        # For bound data, array-valued partial-evaluation constants become
        # hidden inactive device operands rather than dataset-sized compiler
        # literals. The public/native prepared kernel remains q-only.
        _, bound_arrays = definition.data_binding == "bound" ?
            ReactiveKernels._externalize_bound_arrays(definition.kernel) :
            (definition.kernel, ())
        row["reactant_externalized_bound_array_count"] = length(bound_arrays)
        compiler_args = (definition.args..., bound_arrays...)
        traced = nothing
        transfer_seconds = @elapsed traced = _trace_args(compiler_args)
        row["reactant_transfer_seconds"] = transfer_seconds
        compile_started = time_ns()
        try
            public_count = length(definition.args)
            public_traced = traced[1:public_count]
            bound_traced = traced[(public_count + 1):end]
            compiled = isempty(bound_arrays) ?
                compile_ad_value_and_gradient(prepared, public_traced...) :
                ReactiveKernels._reactant_compile_ad_externalized(
                    Val(:value_and_gradient), prepared,
                    public_traced, bound_traced; sync = true)
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
            row["result"] = _measurement(reactant_call; rounds)
        catch err
            row["reactant_ad_compile_seconds"] = get(
                row, "reactant_ad_compile_seconds",
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
        "schema" => "mnist-reactant-ad-v2",
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
            "enzyme_jll_version" => _package_version("Enzyme_jll"),
            "gpucompiler_version" => _package_version("GPUCompiler"),
            "llvm_version" => _package_version("LLVM"),
            "differentiationinterface_version" =>
                _package_version("DifferentiationInterface"),
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
            "ad_receipt_path" => ad_receipt_path,
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
        "protocol" => Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "data" => "MLDatasets MNIST train split, first N images",
            "num_observations" => n,
            "num_features" => features,
            "num_classes" => NUM_CLASSES,
            "source_reused" => true,
            "matrix_source" => ad_receipt_path,
            "input_boundaries" => collect(MNIST_REACTANT_AD_BOUNDARIES),
            "outcomes" => collect(MNIST_REACTANT_AD_OUTCOMES),
            "models" => collect(MNIST_MODELS),
            "matrix_layout" =>
                "long-form provider/model/configuration/boundary/outcome rows",
            "rk_configurations" => [
                configuration.id for configuration in reactant_ad_configurations],
            "bound_ports" => ["X", "y", "num_classes"],
            "gradient_operation" => "value and gradient",
            "rk_native_ad_surface" => "prepare_ad + ad_value_and_gradient!",
            "rk_reactant_ad_surface" =>
                "prepare_ad + compile_ad_value_and_gradient",
            "rk_ad_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
            "rk_bound_ad_backend" =>
                "AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)",
            "parity_reference" => "native RK reverse pass (ad_value_and_gradient!)",
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
            "bound_array_compiler_abi" =>
                "public residual AD kernel stays data-bound; array-valued bound constants are inactive trailing device operands for compilation and steady-state calls",
            "unsupported_cells_recorded" => true,
            "pointwise_jacobian_or_vjp_invented" => false,
            "structured_multi_active_boundary_invented" => false,
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

get(ENV, "RK_MNIST_REACTANT_AD_DEFINITIONS_ONLY", "") == "1" ||
    run_comparison()
