# Inner body for the pinned Eight Schools native-RK-AD / Reactant-compiled-AD
# comparison. This is the AD analog of eight_schools_reactant_comparison_body.jl
# (primal): it reuses the exact authored model source and the SAME derivative
# outcome/boundary protocol published by the AD-only receipt
# (benchmark/receipts/eight-schools-ad-v2.toml). It never copies a prior,
# likelihood, transform, or AD evaluator: it selects RK graph boundaries with
# `prepare`/`prepare_ad` and consumes the first-class RK verbs
# `ad_value_and_gradient!` (native) and `compile_ad_value_and_gradient`
# (Reactant-compiled) — no hand-rolled AD-through-Reactant glue.

using Dates
using Pkg
using SHA
using Statistics
using TOML
using Reactant
import Enzyme
using DifferentiationInterface: AutoEnzyme
using ReactiveKernels
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_SOURCE, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA,
    build_eight_schools_graph

include(joinpath(@__DIR__, "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

const DEFAULT_EIGHT_SCHOOLS_REACTANT_AD_ROUNDS = 20
const DEFAULT_EIGHT_SCHOOLS_REACTANT_AD_TARGET_SECONDS = 0.02
const EIGHT_SCHOOLS_AD_BOUNDARIES = EIGHT_SCHOOLS_BOUNDARIES
const EIGHT_SCHOOLS_AD_OUTCOMES = EIGHT_SCHOOLS_OUTCOMES
_eight_schools_reactant_ad_generator_sha256(path) = bytes2hex(sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))
const AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const BOUND_AD_BACKEND = AutoEnzyme(
    ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))

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
    ENV, "RK_EIGHT_SCHOOLS_REACTANT_AD_ROUNDS",
    string(DEFAULT_EIGHT_SCHOOLS_REACTANT_AD_ROUNDS)))
_target_seconds() = parse(Float64, get(
    ENV, "RK_EIGHT_SCHOOLS_REACTANT_AD_TARGET_SECONDS",
    string(DEFAULT_EIGHT_SCHOOLS_REACTANT_AD_TARGET_SECONDS)))

function _elapsed_batch_ns(f, repetitions)
    result = nothing
    started = time_ns()
    for _ in 1:repetitions
        result = f()
    end
    elapsed = time_ns() - started
    result === nothing && error("timed function unexpectedly returned nothing")
    Float64(elapsed) / repetitions
end

# Calibrated elapsed-batch timing, identical to the primal Reactant receipt so
# native and Reactant-compiled AD share one estimator.
function _measurement(f; rounds, target_seconds)
    f() # first execution / Julia compilation stays outside steady-state timing
    repetitions = 1
    while repetitions < 1_048_576
        elapsed_ns = _elapsed_batch_ns(f, repetitions) * repetitions
        elapsed_ns >= target_seconds * 1e9 && break
        repetitions *= 2
    end
    times_ns = [_elapsed_batch_ns(f, repetitions) for _ in 1:rounds]
    Dict(
        "times_ns" => times_ns,
        "median_ns" => median(times_ns),
        "calls_per_round" => repetitions,
    )
end

_trace_value(value::Number) = Reactant.to_rarray(value; track_numbers = true)
_trace_value(value::AbstractArray) = Reactant.to_rarray(value)
_trace_args(args::Tuple) = map(_trace_value, args)

function _diagnostic(err)
    line = first(split(sprint(showerror, err), '\n'))
    length(line) <= 800 ? line : first(line, 797) * "..."
end

# The differentiable-scalar cell definitions. These select RK graph boundaries
# exactly as the published AD receipt's _rk_definition does; `nothing` marks a
# non-differentiable-scalar cell (vector pointwise, NamedTuple constrained
# boundary, or the undefined minimal joint/prior).
function _ad_definition(model, configuration, boundary, outcome, q, theta,
                        observations, observation_scales)
    state, reason = matrix_support(configuration, boundary, outcome)
    state == "supported" || return (; state, reason, kernel = nothing,
                                      args = (), active = :none,
                                      description = reason)
    if boundary == "packed_unconstrained"
        want = outcome == "joint" ? :posterior :
               outcome == "prior" ? :unconstrained_prior : Symbol(outcome)
        if outcome == "joint"
            description = "gradient of the full joint w.r.t. the packed unconstrained vector"
        elseif outcome == "prior"
            description = "gradient of the unconstrained log prior"
        elseif outcome == "likelihood"
            description = "gradient of the summed log likelihood after unpacking"
        else
            return (; state = "unsupported",
                    reason = _unsupported_reason(boundary, outcome),
                    kernel = nothing, args = (), active = :none,
                    description = _unsupported_reason(boundary, outcome))
        end
        have = outcome == "prior" ? (:unconstrained,) :
            (:unconstrained, :observations, :observation_scales)
        bound = configuration.data == "bound" ?
            (; observations, observation_scales) : NamedTuple()
        kernel = prepare(model; have, want, bound)
        args = configuration.data == "bound" || outcome == "prior" ?
            (q,) : (q, observations, observation_scales)
        return (; state, reason, kernel, args, active = :unconstrained,
                description)
    elseif boundary == "minimal_likelihood" && outcome == "likelihood"
        bound = configuration.data == "bound" ?
            (; observations, observation_scales) : NamedTuple()
        kernel = prepare(model;
            have = (:θ, :observations, :observation_scales),
            want = :likelihood, bound)
        args = configuration.data == "bound" ? (theta,) :
            (theta, observations, observation_scales)
        return (; state, reason, kernel, args, active = :θ,
                description = "gradient of the summed likelihood from θ, observations, and scales")
    end
    (; state = "unsupported", reason = _unsupported_reason(boundary, outcome),
     kernel = nothing, args = (), active = :none,
     description = _unsupported_reason(boundary, outcome))
end

# Unsupported-cell diagnostics, mirroring eight-schools-ad-v2's matrix contract.
function _unsupported_reason(boundary, outcome)
    outcome == "pointwise" && return (
        "pointwise is vector-valued; no matched public Jacobian/VJP contract, " *
        "so it stays unsupported rather than inventing a fused case")
    boundary == "constrained_parameters" && return (
        "the public RK AD boundary accepts floating scalar/array/tuple storage, " *
        "not the primal matrix's constrained NamedTuple")
    boundary == "minimal_likelihood" && return (
        "joint and prior are unavailable from the minimal likelihood HAVE boundary")
    "unsupported matrix cell"
end

_comparable(value) = collect(Float64, Array(value))

function _assert_ad_matrix(ad_receipt)
    get(ad_receipt, "schema", "") == "eight-schools-ad-v2" ||
        error("unexpected Eight Schools AD receipt schema")
    protocol = ad_receipt["protocol"]
    Tuple(protocol["input_boundaries"]) == EIGHT_SCHOOLS_AD_BOUNDARIES ||
        error("Reactant AD benchmark boundaries drifted from the AD receipt")
    Tuple(protocol["outcomes"]) == EIGHT_SCHOOLS_AD_OUTCOMES ||
        error("Reactant AD benchmark outcomes drifted from the AD receipt")
    Set(protocol["rk_configurations"]) ==
        Set(("ad_native", "ad_native_bound")) ||
        error("unexpected Eight Schools native AD configuration inventory")
    nothing
end

function run_comparison()
    repo = normpath(joinpath(@__DIR__, ".."))
    rounds = _rounds()
    target_seconds = _target_seconds()
    rounds >= 1 || error("round count must be positive")
    target_seconds > 0 || error("timing target must be positive")

    observations = Float64.(EIGHT_SCHOOLS_Y)
    observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
    mu = 1.5
    log_tau = log(2.0)
    theta = 0.25 .* collect(1.0:8.0)
    unconstrained = [mu, log_tau, theta...]

    model = nothing
    preparation_seconds = @elapsed begin
        # Clone from the runtime template evaluated from EIGHT_SCHOOLS_SOURCE, as
        # in the primal Reactant benchmark, to avoid newer-world closures here.
        model = build_eight_schools_graph()
    end
    configurations = filter(
        configuration -> configuration.differentiation == "value_and_gradient" &&
            configuration.compiler == "reactant",
        EIGHT_SCHOOLS_RK_CONFIGURATIONS,
    )

    ad_path = get(
        ENV, "RK_EIGHT_SCHOOLS_AD_RECEIPT",
        joinpath(@__DIR__, "receipts", "eight-schools-ad-v2.toml"))
    isabspath(ad_path) || (ad_path = normpath(joinpath(repo, ad_path)))
    isfile(ad_path) || error(
        "the published AD receipt is required: $ad_path")
    ad_receipt = TOML.parsefile(ad_path)
    _assert_ad_matrix(ad_receipt)

    measurements = Dict{String,Any}[]
    for boundary in EIGHT_SCHOOLS_AD_BOUNDARIES,
        outcome in EIGHT_SCHOOLS_AD_OUTCOMES,
        configuration in configurations
        definition = _ad_definition(
            model, configuration, boundary, outcome, unconstrained, theta,
            observations, observation_scales)
        row = Dict{String,Any}(
            "provider" => "rk", "model" => "centered",
            "configuration" => configuration.id,
            "boundary" => boundary, "outcome" => outcome,
            "description" => definition.description,
            "state" => definition.state,
        )
        if definition.state != "supported"
            row["reason"] = definition.reason
            push!(measurements, row)
            println("configuration=$(configuration.id) boundary=$boundary " *
                    "outcome=$outcome state=$(definition.state)")
            continue
        end

        row["active_port"] = String(definition.active)

        # Native RK AD reference (prepare_ad + ad_value_and_gradient!). The
        # preparation is outside steady-state timing.
        backend = configuration.data == "bound" ?
            BOUND_AD_BACKEND : AD_BACKEND
        prepare_seconds = @elapsed prepared = prepare_ad(
            definition.kernel, backend, definition.args...;
            active = definition.active)
        row["ad_preparation_seconds"] = prepare_seconds
        # The active port is the first HAVE argument in every cell (see the AD
        # receipt's _rk_definition and test/test_ad_reactant.jl), so the native
        # gradient buffer is shaped like args[1].
        native_gradient = similar(definition.args[1])
        native_value, native_gradient = ad_value_and_gradient!(
            prepared, native_gradient, definition.args...)
        native_gradient_ref = collect(Float64, native_gradient)
        row["native_control"] = _measurement(
            () -> ad_value_and_gradient!(prepared, native_gradient, definition.args...);
            rounds, target_seconds)

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
            reactant_gradient = _comparable(compiled_gradient)
            reactant_value = Float64(compiled_value)
            length(reactant_gradient) == length(native_gradient_ref) ||
                error("Reactant AD gradient length parity failed")
            row["max_abs_error"] =
                maximum(abs.(reactant_gradient .- native_gradient_ref))
            row["value_abs_error"] = abs(reactant_value - Float64(native_value))
            (row["max_abs_error"] <= 1e-10 && row["value_abs_error"] <= 1e-10) ||
                error("native/Reactant AD parity failed")
            row["result"] = _measurement(
                () -> compiled(traced...); rounds, target_seconds)
        catch err
            row["state"] = "unsupported_runtime"
            row["reason"] = _diagnostic(err)
            row["reactant_ad_compile_seconds"] = get(
                row, "reactant_ad_compile_seconds",
                Float64(time_ns() - compile_started) / 1e9)
        end
        push!(measurements, row)
        println("configuration=$(configuration.id) boundary=$boundary " *
                "outcome=$outcome state=$(row["state"])")
    end

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "eight_schools.jl")
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")
    receipt = Dict{String,Any}(
        "schema" => "eight-schools-reactant-ad-v2",
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
            "julia_version" => string(VERSION),
            "source_authority_path" => source_path,
            "source_authority_blob" =>
                _git(repo, "rev-parse", "$candidate_sha:$source_path"),
            "source_text_sha256" => bytes2hex(sha256(EIGHT_SCHOOLS_SOURCE)),
            "ad_receipt_path" =>
                "benchmark/receipts/eight-schools-ad-v2.toml",
            "ad_receipt_sha256" =>
                _eight_schools_reactant_ad_generator_sha256(ad_path),
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
                ENV, "RK_EIGHT_SCHOOLS_AD_ENV_SETUP_SECONDS", "0")),
            "package_precompile_seconds" => parse(Float64, get(
                ENV, "RK_EIGHT_SCHOOLS_AD_PRECOMPILE_SECONDS", "0")),
            "kernel_preparation_seconds" => preparation_seconds,
        ),
        "protocol" => Dict(
            "model" => "centered Eight Schools",
            "source_reused" => true,
            "matrix_source" => "benchmark/receipts/eight-schools-ad-v2.toml",
            "input_boundaries" => collect(EIGHT_SCHOOLS_AD_BOUNDARIES),
            "outcomes" => collect(EIGHT_SCHOOLS_AD_OUTCOMES),
            "models" => collect(EIGHT_SCHOOLS_MODELS),
            "matrix_layout" =>
                "long-form provider/model/configuration/boundary/outcome rows",
            "rk_configurations" => [configuration.id for configuration in configurations],
            "bound_ports" => ["observations", "observation_scales"],
            "gradient_operation" => "value and gradient",
            "rk_native_ad_surface" => "prepare_ad + ad_value_and_gradient!",
            "rk_reactant_ad_surface" =>
                "prepare_ad + compile_ad_value_and_gradient",
            "rk_ad_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
            "rk_bound_ad_backend" =>
                "AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)",
            "parity_reference" => "native RK reverse pass (ad_value_and_gradient!)",
            "rounds" => rounds,
            "target_seconds_per_round" => target_seconds,
            "estimator" => "median per-call time from calibrated elapsed batches",
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
            "parity_atol" => 1.0e-10,
            "parity_rtol" => 1.0e-11,
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

get(ENV, "RK_EIGHT_SCHOOLS_REACTANT_AD_DEFINITIONS_ONLY", "") == "1" ||
    run_comparison()
