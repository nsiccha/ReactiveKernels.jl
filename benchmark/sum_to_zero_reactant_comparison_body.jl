# Inner body for the pinned sum-to-zero Reactant comparison. It imports the
# exact native comparator definitions rather than copying the manual control.

using Dates
using Pkg
using SHA
using Statistics
using TOML
using Reactant
using DifferentiationInterface
import Enzyme
using ReactiveKernels
using ReactiveKernelsPPLExamples.SumToZeroExample:
    SUM_TO_ZERO_SOURCE, build_sum_to_zero_graph
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA

module NativeSumToZeroControls
ENV["RK_SUM_TO_ZERO_DEFINITIONS_ONLY"] = "1"
include(joinpath(@__DIR__, "sum_to_zero_comparison_body.jl"))
end
delete!(ENV, "RK_SUM_TO_ZERO_DEFINITIONS_ONLY")
const Native = NativeSumToZeroControls

const DEFAULT_SUM_TO_ZERO_REACTANT_ROUNDS = 20
const DEFAULT_SUM_TO_ZERO_REACTANT_TARGET_SECONDS = 0.05
const AD_BACKEND = AutoEnzyme(
    ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
const OBSERVATIONS = Float64.(EIGHT_SCHOOLS_Y)
const OBSERVATION_SCALES = Float64.(EIGHT_SCHOOLS_SIGMA)
const ALPHA_PRIOR_SD = 5.0

_rounds() = parse(Int, get(
    ENV, "RK_SUM_TO_ZERO_REACTANT_ROUNDS",
    string(DEFAULT_SUM_TO_ZERO_REACTANT_ROUNDS)))
_target_seconds() = parse(Float64, get(
    ENV, "RK_SUM_TO_ZERO_REACTANT_TARGET_SECONDS",
    string(DEFAULT_SUM_TO_ZERO_REACTANT_TARGET_SECONDS)))

function _output_path()
    for argument in ARGS
        startswith(argument, "--output=") &&
            return split(argument, '='; limit = 2)[2]
    end
    nothing
end

function _elapsed_batch_ns(f, argument, repetitions)
    result = nothing
    started = time_ns()
    for _ in 1:repetitions
        result = f(argument)
    end
    elapsed = time_ns() - started
    result === nothing && error("timed function unexpectedly returned nothing")
    Float64(elapsed) / repetitions
end

function _measurement(f, argument; rounds, target_seconds)
    f(argument)
    repetitions = 1
    while repetitions < 1_048_576
        elapsed_ns = _elapsed_batch_ns(f, argument, repetitions) * repetitions
        elapsed_ns >= target_seconds * 1e9 && break
        repetitions *= 2
    end
    times_ns = [
        _elapsed_batch_ns(f, argument, repetitions) for _ in 1:rounds
    ]
    Dict(
        "times_ns" => times_ns,
        "min_ns" => minimum(times_ns),
        "median_ns" => median(times_ns),
        "calls_per_round" => repetitions,
    )
end

# Backend adapter only: the mathematical implementation remains the exact
# manual control loaded above. Constants are embedded like the RK bound cut.
function _manual_reactant_bound(unconstrained)
    Reactant.@allowscalar Native.manual_sum_to_zero_posterior(
        unconstrained, OBSERVATIONS, OBSERVATION_SCALES, ALPHA_PRIOR_SD)
end

function _measurement_row(provider, configuration, differentiation,
                          result; compile_seconds, first_execution_seconds,
                          value_abs_error, gradient_max_abs_error = nothing)
    row = Dict{String,Any}(
        "provider" => provider,
        "configuration" => configuration,
        "operation" => differentiation == "primal" ?
            "posterior" : "posterior value and gradient",
        "differentiation" => differentiation,
        "compiler" => "reactant",
        "state" => "supported",
        "reactant_compile_seconds" => compile_seconds,
        "reactant_first_execution_seconds" => first_execution_seconds,
        "value_abs_error" => value_abs_error,
        "result" => result,
    )
    gradient_max_abs_error === nothing ||
        (row["gradient_max_abs_error"] = gradient_max_abs_error)
    row
end

function _unsupported_turing_row(differentiation)
    Dict{String,Any}(
        "provider" => "turing",
        "configuration" => differentiation == "primal" ?
            "turing_reactant" : "turing_reactant_ad",
        "operation" => differentiation == "primal" ?
            "posterior" : "posterior value and gradient",
        "differentiation" => differentiation,
        "compiler" => "reactant",
        "state" => "unsupported",
        "reason" =>
            "DynamicPPL's model evaluator is not a public Reactant-traceable interface",
    )
end

function run_sum_to_zero_reactant_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    rounds = _rounds()
    target_seconds = _target_seconds()
    rounds >= 1 || error("round count must be positive")
    target_seconds > 0 || error("timing target must be positive")

    native_path = get(
        ENV, "RK_SUM_TO_ZERO_NATIVE_RECEIPT",
        joinpath(@__DIR__, "receipts", "sum-to-zero-native-v1.toml"))
    isabspath(native_path) || (native_path = normpath(joinpath(root, native_path)))
    native_receipt = TOML.parsefile(native_path)
    native_receipt["schema"] == "sum-to-zero-hotloop-v1" ||
        error("unexpected native sum-to-zero receipt schema")
    native_receipt["protocol"]["hot_loop_excludes_recovery"] == true ||
        error("native benchmark allowed recovery into the sampler hot loop")

    q = [0.5, log(2.0), (0.25 .* collect(1.0:7.0))...]
    model_setup = @timed build_sum_to_zero_graph()
    model = model_setup.value
    kernel_setup = @timed prepare(
        model;
        have = (
            :unconstrained, :observations, :observation_scales, :α_prior_sd),
        want = :posterior,
        bound = (;
            observations = OBSERVATIONS,
            observation_scales = OBSERVATION_SCALES,
            α_prior_sd = ALPHA_PRIOR_SD,
        ),
    )
    rk_kernel = kernel_setup.value

    q_traced = nothing
    transfer_seconds = @elapsed q_traced = Reactant.to_rarray(q)

    rk_primal_compile_seconds = @elapsed rk_primal =
        Reactant.compile(rk_kernel, (q_traced,); sync = true)
    manual_primal_compile_seconds = @elapsed manual_primal =
        Reactant.compile(_manual_reactant_bound, (q_traced,); sync = true)
    rk_primal_first_seconds = @elapsed rk_primal_value = rk_primal(q_traced)
    manual_primal_first_seconds = @elapsed manual_primal_value =
        manual_primal(q_traced)
    rk_primal_host = Float64(rk_primal_value)
    manual_primal_host = Float64(manual_primal_value)
    primal_error = abs(rk_primal_host - manual_primal_host)
    primal_error <= 1e-11 || error("Reactant primal parity failed")

    rk_prepared_ad = prepare_ad(
        rk_kernel, AD_BACKEND, q; active = :unconstrained)
    rk_ad_compile_seconds = @elapsed rk_ad =
        compile_ad_value_and_gradient(rk_prepared_ad, q_traced)
    manual_ad_call = let backend = AD_BACKEND
        unconstrained -> value_and_gradient(
            _manual_reactant_bound, backend, unconstrained)
    end
    manual_ad_compile_seconds = @elapsed manual_ad =
        Reactant.compile(manual_ad_call, (q_traced,); sync = true)
    rk_ad_first_seconds = @elapsed rk_ad_value, rk_ad_gradient = rk_ad(q_traced)
    manual_ad_first_seconds = @elapsed manual_ad_value, manual_ad_gradient =
        manual_ad(q_traced)
    rk_ad_host = Float64(rk_ad_value)
    manual_ad_host = Float64(manual_ad_value)
    rk_gradient_host = Array(rk_ad_gradient)
    manual_gradient_host = Array(manual_ad_gradient)
    ad_value_error = abs(rk_ad_host - manual_ad_host)
    gradient_error = maximum(abs.(rk_gradient_host .- manual_gradient_host))
    ad_value_error <= 1e-11 || error("Reactant AD value parity failed")
    gradient_error <= 1e-10 || error("Reactant AD gradient parity failed")

    native_manual_value = only(filter(
        row -> row["configuration"] == "manual_primal",
        native_receipt["measurements"],
    ))["value"]
    abs(manual_primal_host - native_manual_value) <= 1e-11 ||
        error("native/Reactant manual value parity failed")

    measurements = Dict{String,Any}[
        _measurement_row(
            "rk", "rk_primal_reactant", "primal",
            _measurement(rk_primal, q_traced; rounds, target_seconds);
            compile_seconds = rk_primal_compile_seconds,
            first_execution_seconds = rk_primal_first_seconds,
            value_abs_error = primal_error,
        ),
        _measurement_row(
            "manual_julia", "manual_primal_reactant", "primal",
            _measurement(manual_primal, q_traced; rounds, target_seconds);
            compile_seconds = manual_primal_compile_seconds,
            first_execution_seconds = manual_primal_first_seconds,
            value_abs_error = 0.0,
        ),
        _unsupported_turing_row("primal"),
        _measurement_row(
            "rk", "rk_ad_reactant", "value_and_gradient",
            _measurement(rk_ad, q_traced; rounds, target_seconds);
            compile_seconds = rk_ad_compile_seconds,
            first_execution_seconds = rk_ad_first_seconds,
            value_abs_error = ad_value_error,
            gradient_max_abs_error = gradient_error,
        ),
        _measurement_row(
            "manual_julia", "manual_ad_reactant", "value_and_gradient",
            _measurement(manual_ad, q_traced; rounds, target_seconds);
            compile_seconds = manual_ad_compile_seconds,
            first_execution_seconds = manual_ad_first_seconds,
            value_abs_error = 0.0,
            gradient_max_abs_error = 0.0,
        ),
        _unsupported_turing_row("value_and_gradient"),
    ]

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "sum_to_zero.jl")
    native_comparator_path =
        joinpath("benchmark", "sum_to_zero_comparison_body.jl")
    reactant_comparator_path =
        joinpath("benchmark", "sum_to_zero_reactant_comparison_body.jl")
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")
    receipt = Dict{String,Any}(
        "schema" => "sum-to-zero-reactant-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => candidate_sha,
            "reactivekernels_dirty" => false,
            "reactivekernels_version" => Native.package_version("ReactiveKernels"),
            "reactivekernelspplexamples_version" =>
                Native.package_version("ReactiveKernelsPPLExamples"),
            "reactivekernelsdistributionkernels_version" =>
                Native.package_version("ReactiveKernelsDistributionKernels"),
            "reactant_version" => Native.package_version("Reactant"),
            "reactant_jll_version" => Native.package_version("Reactant_jll"),
            "enzyme_version" => Native.package_version("Enzyme"),
            "differentiationinterface_version" =>
                Native.package_version("DifferentiationInterface"),
            "julia_version" => string(VERSION),
            "model_source" => Native.source_pin(root, source_path),
            "native_comparator_source" =>
                Native.source_pin(root, native_comparator_path),
            "reactant_comparator_source" =>
                Native.source_pin(root, reactant_comparator_path),
            "model_source_text_sha256" => bytes2hex(sha256(SUM_TO_ZERO_SOURCE)),
            "native_receipt_path" =>
                "benchmark/receipts/sum-to-zero-native-v1.toml",
            "native_receipt_sha256" => bytes2hex(sha256(read(native_path))),
            "native_receipt_reactivekernels_sha" =>
                native_receipt["pins"]["reactivekernels_sha"],
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
                ENV, "RK_SUM_TO_ZERO_REACTANT_ENV_SETUP_SECONDS", "0")),
            "package_precompile_seconds" => parse(Float64, get(
                ENV, "RK_SUM_TO_ZERO_REACTANT_PRECOMPILE_SECONDS", "0")),
            "model_seconds" => model_setup.time,
            "rk_kernel_seconds" => kernel_setup.time,
            "transfer_seconds" => transfer_seconds,
        ),
        "protocol" => Dict(
            "model" => "sum-to-zero hierarchical Eight Schools",
            "boundary" => "packed unconstrained parameters",
            "outcome" => "full unconstrained posterior",
            "hot_loop_excludes_recovery" => true,
            "source_reused" => true,
            "bound_ports" => [
                "observations", "observation_scales", "α_prior_sd"],
            "manual_control" =>
                "exact native receipt control compiled through Reactant",
            "turing_reactant_state" => "unsupported",
            "rounds" => rounds,
            "target_seconds_per_round" => target_seconds,
            "estimator" =>
                "median per-call time from calibrated elapsed batches",
            "reactant_sync" => true,
            "reactant_transfers_in_timed_region" => false,
            "reactant_compile_time_in_timed_region" => false,
            "reactant_readback_in_timed_region" => false,
            "first_execution_in_steady_state_region" => false,
            "parity_atol" => 1e-10,
            "parity_rtol" => 1e-11,
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

run_sum_to_zero_reactant_comparison()
