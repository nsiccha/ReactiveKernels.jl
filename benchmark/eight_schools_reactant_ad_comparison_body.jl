# Inner body for the pinned Eight Schools native-RK-AD / Reactant-compiled-AD
# comparison. This is the AD analog of eight_schools_reactant_comparison_body.jl
# (primal): it reuses the exact authored model source and the SAME derivative
# outcome/boundary protocol published by the AD-only receipt
# (benchmark/receipts/eight-schools-ad-v1.toml). It never copies a prior,
# likelihood, transform, or AD evaluator: it selects RK graph boundaries with
# `prepare` plus the public prepared gradient/pullback verbs. The Reactant subset
# consumes `compile_ad_value_and_gradient`; no compiled pullback or structured
# tracing surface is invented by this benchmark.

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

const DEFAULT_EIGHT_SCHOOLS_REACTANT_AD_ROUNDS = 20
const DEFAULT_EIGHT_SCHOOLS_REACTANT_AD_TARGET_SECONDS = 0.02
const EIGHT_SCHOOLS_AD_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood")
const EIGHT_SCHOOLS_AD_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
# The complete native reverse-AD inventory published by eight-schools-ad-v1.
const EIGHT_SCHOOLS_AD_SUPPORTED = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
    ("packed_unconstrained", "pointwise"),
    ("constrained_parameters", "joint"),
    ("constrained_parameters", "prior"),
    ("constrained_parameters", "likelihood"),
    ("minimal_likelihood", "likelihood"),
    ("minimal_likelihood", "pointwise"),
))
# Only these array-backed scalar-gradient cells use the public compiled-AD verb.
# The compile attempt remains evidence-bearing: packed joint/prior retain their
# real primal compiler rejection, while the two likelihood cells must compile.
const EIGHT_SCHOOLS_REACTANT_AD_COMPILE_CANDIDATES = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
    ("minimal_likelihood", "likelihood"),
))

_eight_schools_reactant_ad_generator_sha256(path) = bytes2hex(sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))
const AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

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

_sensitivity_values(value::Number) = (Float64(value),)
_sensitivity_values(value::AbstractArray) = Float64.(vec(value))
_sensitivity_values(value::Tuple) =
    Iterators.flatten(_sensitivity_values(child) for child in value)
_sensitivity_values(value::NamedTuple) =
    Iterators.flatten(_sensitivity_values(child) for child in values(value))
_flatten_sensitivity(value) = collect(_sensitivity_values(value))

function _diagnostic(err)
    line = first(split(sprint(showerror, err), '\n'))
    length(line) <= 800 ? line : first(line, 797) * "..."
end

# These definitions mirror the published native receipt's exact RK boundaries.
# `nothing` marks only its three honestly unsupported cells.
function _ad_definition(model, boundary, outcome, q, theta, parameters,
                        observations, observation_scales)
    if boundary == "packed_unconstrained"
        if outcome == "joint"
            return (kernel = prepare(model;
                        have = (:unconstrained, :observations, :observation_scales),
                        want = :posterior),
                    args = (q, observations, observation_scales),
                    active = :unconstrained,
                    description = "gradient of the full joint w.r.t. the packed unconstrained vector")
        elseif outcome == "prior"
            return (kernel = prepare(model;
                        have = :unconstrained, want = :unconstrained_prior),
                    args = (q,), active = :unconstrained,
                    description = "gradient of the unconstrained log prior")
        elseif outcome == "likelihood"
            return (kernel = prepare(model;
                        have = (:unconstrained, :observations, :observation_scales),
                        want = :likelihood),
                    args = (q, observations, observation_scales),
                    active = :unconstrained,
                    description = "gradient of the summed log likelihood after unpacking")
        elseif outcome == "pointwise"
            return (kernel = prepare(model;
                        have = (:unconstrained, :observations, :observation_scales),
                        want = :pointwise),
                    args = (q, observations, observation_scales),
                    active = :unconstrained,
                    description = "pointwise-likelihood VJP w.r.t. the packed unconstrained vector")
        end
    elseif boundary == "constrained_parameters"
        outcome == "pointwise" && return nothing
        want = outcome == "joint" ? :constrained_logdensity : Symbol(outcome)
        kernel = if outcome == "prior"
            prepare(model; have = :parameters, want)
        elseif outcome in ("joint", "likelihood")
            prepare(model;
                have = (:parameters, :observations, :observation_scales), want)
        else
            return nothing
        end
        arguments = outcome == "prior" ? (parameters,) :
            (parameters, observations, observation_scales)
        return (kernel, args = arguments, active = :parameters,
                description = "structured NamedTuple gradient of the constrained $outcome")
    elseif boundary == "minimal_likelihood" &&
           outcome in ("likelihood", "pointwise")
        return (kernel = prepare(model;
                    have = (:θ, :observations, :observation_scales),
                    want = Symbol(outcome)),
                args = (theta, observations, observation_scales),
                active = :θ,
                description = outcome == "likelihood" ?
                    "gradient of the summed likelihood from θ, observations, and scales" :
                    "pointwise-likelihood VJP from θ, observations, and scales")
    end
    nothing
end

# Unsupported-cell diagnostics, mirroring eight-schools-ad-v1's _unsupported_reason.
function _unsupported_reason(boundary, outcome)
    boundary == "constrained_parameters" && outcome == "pointwise" && return (
        "the public reverse-pullback surface supports the pointwise WANT, and " *
        "the public gradient surface supports the constrained NamedTuple, but " *
        "DifferentiationInterface/Enzyme cannot currently annotate their " *
        "MixedDuplicated cross-product")
    boundary == "minimal_likelihood" && return (
        "joint and prior are unavailable from the minimal likelihood HAVE boundary")
    "unsupported matrix cell"
end

function _reactant_unsupported_reason(boundary, outcome)
    outcome == "pointwise" && return (
        "native value-and-pullback is measured, but ReactiveKernels exposes no " *
        "compiled reverse-pullback verb; no compiled VJP is claimed")
    boundary == "constrained_parameters" && return (
        "native structured gradient is measured, but this receipt has no public " *
        "compiled structured active-argument/result contract; no structured " *
        "Reactant tracing is claimed")
    _unsupported_reason(boundary, outcome)
end

_comparable(value) = collect(Float64, Array(value))

function _assert_ad_matrix(ad_receipt)
    get(ad_receipt, "schema", "") == "eight-schools-ad-v1" ||
        error("unexpected Eight Schools AD receipt schema")
    protocol = ad_receipt["protocol"]
    Tuple(protocol["input_boundaries"]) == EIGHT_SCHOOLS_AD_BOUNDARIES ||
        error("Reactant AD benchmark boundaries drifted from the AD receipt")
    Tuple(protocol["outcomes"]) == EIGHT_SCHOOLS_AD_OUTCOMES ||
        error("Reactant AD benchmark outcomes drifted from the AD receipt")
    # The native support inventory must match exactly, including structured
    # gradients and the two fixed-cotangent pointwise pullbacks.
    supported = Set{Tuple{String,String}}()
    for row in ad_receipt["measurements"]
        get(row, "supported", false) &&
            push!(supported, (row["boundary"], row["outcome"]))
    end
    supported == EIGHT_SCHOOLS_AD_SUPPORTED || error(
        "AD receipt supported inventory drifted from this benchmark's cells")
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
    tau = exp(log_tau)
    theta = 0.25 .* collect(1.0:8.0)
    unconstrained = [mu, log_tau, theta...]
    parameters = (; μ = mu, τ = tau, θ = theta)

    model = nothing
    preparation_seconds = @elapsed begin
        # Clone from the runtime template evaluated from EIGHT_SCHOOLS_SOURCE, as
        # in the primal Reactant benchmark, to avoid newer-world closures here.
        model = build_eight_schools_graph()
    end

    ad_path = joinpath(@__DIR__, "receipts", "eight-schools-ad-v1.toml")
    isfile(ad_path) || error(
        "the published AD receipt is required: $ad_path")
    ad_receipt = TOML.parsefile(ad_path)
    _assert_ad_matrix(ad_receipt)

    measurements = Dict{String,Any}[]
    for boundary in EIGHT_SCHOOLS_AD_BOUNDARIES,
        outcome in EIGHT_SCHOOLS_AD_OUTCOMES
        row = Dict{String,Any}("boundary" => boundary, "outcome" => outcome)
        definition = _ad_definition(
            model, boundary, outcome, unconstrained, theta, parameters,
            observations, observation_scales)
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

        matched_row = only(filter(ad_receipt["measurements"]) do candidate
            candidate["boundary"] == boundary && candidate["outcome"] == outcome
        end)
        pointwise = outcome == "pointwise"
        structured = first(definition.args) isa NamedTuple
        row["operation"] = pointwise ? "value and pullback" : "value and gradient"
        row["operation"] == matched_row["operation"] || error(
            "native/Reactant receipt operation drifted for $boundary / $outcome")

        seed = pointwise ? Float64.(matched_row["output_cotangent"]) : 1.0
        pointwise && (row["output_cotangent"] = seed)
        prepared = nothing
        prepare_started = time_ns()
        if pointwise
            prepared = prepare_ad_pullback(
                definition.kernel, AD_BACKEND, seed, definition.args...;
                active = definition.active)
        else
            prepared = prepare_ad(
                definition.kernel, AD_BACKEND, definition.args...;
                active = definition.active)
        end
        row["ad_preparation_seconds"] =
            Float64(time_ns() - prepare_started) / 1e9

        native_call = if pointwise
            cotangent = similar(first(definition.args))
            () -> ad_value_and_pullback!(
                prepared, cotangent, seed, definition.args...)
        elseif structured
            () -> ad_value_and_gradient(prepared, definition.args...)
        else
            gradient = similar(first(definition.args))
            () -> ad_value_and_gradient!(
                prepared, gradient, definition.args...)
        end
        native_value, native_sensitivity = native_call()
        native_sensitivity_ref = _flatten_sensitivity(native_sensitivity)
        row["native_sensitivity_length"] = length(native_sensitivity_ref)
        row["rk_native_ad"] = _measurement(
            native_call; rounds, target_seconds)

        cell = (boundary, outcome)
        if !(cell in EIGHT_SCHOOLS_REACTANT_AD_COMPILE_CANDIDATES)
            row["rk_reactant_ad_supported"] = false
            row["rk_reactant_ad_error"] =
                _reactant_unsupported_reason(boundary, outcome)
            push!(measurements, row)
            println("boundary=$boundary outcome=$outcome reactant_ad=false (not exposed)")
            continue
        end

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
            length(reactant_gradient) == length(native_sensitivity_ref) ||
                error("Reactant AD gradient length parity failed")
            row["max_abs_error"] =
                maximum(abs.(reactant_gradient .- native_sensitivity_ref))
            row["value_abs_error"] = abs(reactant_value - Float64(native_value))
            (row["max_abs_error"] <= 1e-10 && row["value_abs_error"] <= 1e-10) ||
                error("native/Reactant AD parity failed")
            row["rk_reactant_ad"] = _measurement(
                () -> compiled(traced...); rounds, target_seconds)
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
        "packages", "ReactiveKernelsPPLExamples", "src", "eight_schools.jl")
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")
    receipt = Dict{String,Any}(
        "schema" => "eight-schools-reactant-ad-v1",
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
                "benchmark/receipts/eight-schools-ad-v1.toml",
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
            "matrix_source" => "benchmark/receipts/eight-schools-ad-v1.toml",
            "input_boundaries" => collect(EIGHT_SCHOOLS_AD_BOUNDARIES),
            "outcomes" => collect(EIGHT_SCHOOLS_AD_OUTCOMES),
            "gradient_operation" =>
                "value and gradient for scalar WANTs; value and reverse pullback for pointwise WANTs",
            "rk_native_ad_surface" =>
                "prepare_ad + ad_value_and_gradient[!]; prepare_ad_pullback + ad_value_and_pullback[!]",
            "rk_reactant_ad_surface" =>
                "prepare_ad + compile_ad_value_and_gradient",
            "compiled_pullback_exposed" => false,
            "compiled_structured_active_exposed" => false,
            "rk_ad_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
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
            "pointwise_vjp_contract" =>
                "one deterministic output cotangent through public prepared reverse pullbacks; no full Jacobian",
            "structured_cotangent_ownership" =>
                "NamedTuple sensitivities use the public nonmutating RK gradient; array sensitivities use caller-owned destinations",
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

run_comparison()
