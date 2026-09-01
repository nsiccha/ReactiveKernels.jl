# Inner body for the pinned Eight Schools native-RK/Reactant comparison.

using Dates
using Pkg
using SHA
using Statistics
using TOML
using Reactant
using Reactant: @compile
using ReactiveKernels
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_SOURCE, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA,
    build_eight_schools_graph

include(joinpath(@__DIR__, "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

const DEFAULT_EIGHT_SCHOOLS_REACTANT_ROUNDS = 20
const DEFAULT_EIGHT_SCHOOLS_REACTANT_TARGET_SECONDS = 0.02

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))
_eight_schools_reactant_generator_sha256(path) = bytes2hex(sha256(
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
    ENV, "RK_EIGHT_SCHOOLS_REACTANT_ROUNDS",
    string(DEFAULT_EIGHT_SCHOOLS_REACTANT_ROUNDS)))
_target_seconds() = parse(Float64, get(
    ENV, "RK_EIGHT_SCHOOLS_REACTANT_TARGET_SECONDS",
    string(DEFAULT_EIGHT_SCHOOLS_REACTANT_TARGET_SECONDS)))

_comparable(outcome, value) = outcome == "pointwise" ?
    collect(Float64, Array(value)) : Float64(value)

function _elapsed_batch_ns(f, args, repetitions)
    result = nothing
    started = time_ns()
    for _ in 1:repetitions
        result = f(args...)
    end
    elapsed = time_ns() - started
    result === nothing && error("timed function unexpectedly returned nothing")
    Float64(elapsed) / repetitions
end

function _measurement(f, args...; rounds, target_seconds)
    f(args...) # first execution and Julia compilation stay outside steady timing
    repetitions = 1
    while repetitions < 1_048_576
        elapsed_ns = _elapsed_batch_ns(f, args, repetitions) * repetitions
        elapsed_ns >= target_seconds * 1e9 && break
        repetitions *= 2
    end
    times_ns = [
        _elapsed_batch_ns(f, args, repetitions) for _ in 1:rounds
    ]
    Dict(
        "times_ns" => times_ns,
        "median_ns" => median(times_ns),
        "calls_per_round" => repetitions,
    )
end

_trace_value(value::Number) = Reactant.to_rarray(value; track_numbers = true)
_trace_value(value::AbstractArray) = Reactant.to_rarray(value)
_trace_value(value::NamedTuple) =
    NamedTuple{keys(value)}(map(_trace_value, values(value)))
_trace_args(args::Tuple) = map(_trace_value, args)

_compile_call(kernel, args::Tuple{Any}) = @compile sync = true kernel(args[1])
_compile_call(kernel, args::Tuple{Any,Any,Any}) =
    @compile sync = true kernel(args[1], args[2], args[3])

function _diagnostic(err)
    line = first(split(sprint(showerror, err), '\n'))
    length(line) <= 800 ? line : first(line, 797) * "..."
end

function _rk_definition(model, configuration, boundary, outcome,
                        unconstrained, parameters, theta,
                        observations, observation_scales)
    state, reason = matrix_support(configuration, boundary, outcome)
    state == "supported" || return (; state, reason, kernel = nothing, args = ())
    configuration.compiler == "reactant" || return (
        state = "unsupported", reason = "not a Reactant configuration",
        kernel = nothing, args = ())
    data_dependent = outcome != "prior"
    have, want, active = if boundary == "packed_unconstrained"
        (data_dependent ?
            (:unconstrained, :observations, :observation_scales) :
            (:unconstrained,),
         outcome == "joint" ? :posterior :
            outcome == "prior" ? :unconstrained_prior : Symbol(outcome),
         unconstrained)
    elseif boundary == "constrained_parameters"
        (data_dependent ?
            (:parameters, :observations, :observation_scales) : (:parameters,),
         outcome == "joint" ? :constrained_logdensity : Symbol(outcome),
         parameters)
    else
        ((:θ, :observations, :observation_scales), Symbol(outcome), theta)
    end
    bound = configuration.data == "bound" ?
        (; observations, observation_scales) : NamedTuple()
    kernel = prepare(model; have, want, bound)
    args = configuration.data == "bound" || outcome == "prior" ?
        (active,) : (active, observations, observation_scales)
    (; state, reason, kernel, args)
end

function _assert_primal_matrix(primal_receipt)
    get(primal_receipt, "schema", "") == "eight-schools-primal-v2" ||
        error("unexpected Eight Schools primal receipt schema")
    protocol = primal_receipt["protocol"]
    Tuple(protocol["input_boundaries"]) == EIGHT_SCHOOLS_BOUNDARIES ||
        error("Reactant benchmark input boundaries drifted from the primal receipt")
    Tuple(protocol["outcomes"]) == EIGHT_SCHOOLS_OUTCOMES ||
        error("Reactant benchmark outcomes drifted from the primal receipt")
    Set(protocol["rk_configurations"]) == Set((
        "primal_native", "primal_native_bound",
        "primal_nonallocating", "primal_nonallocating_bound")) ||
        error("unexpected Eight Schools primal configuration inventory")
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
    preparation_seconds = @elapsed model = build_eight_schools_graph()
    configurations = filter(
        configuration -> configuration.differentiation == "primal" &&
            configuration.compiler == "reactant",
        EIGHT_SCHOOLS_RK_CONFIGURATIONS,
    )

    primal_path = get(
        ENV, "RK_EIGHT_SCHOOLS_PRIMAL_RECEIPT",
        joinpath(@__DIR__, "receipts", "eight-schools-primal-v2.toml"))
    isabspath(primal_path) || (primal_path = normpath(joinpath(repo, primal_path)))
    isfile(primal_path) || error(
        "the published primal receipt is required: $primal_path")
    primal_receipt = TOML.parsefile(primal_path)
    _assert_primal_matrix(primal_receipt)

    measurements = Dict{String,Any}[]
    for boundary in EIGHT_SCHOOLS_BOUNDARIES, outcome in EIGHT_SCHOOLS_OUTCOMES,
        configuration in configurations
        definition = _rk_definition(
            model, configuration, boundary, outcome, unconstrained,
            parameters, theta, observations, observation_scales)
        row = Dict{String,Any}(
            "provider" => "rk", "model" => "centered",
            "configuration" => configuration.id,
            "boundary" => boundary, "outcome" => outcome,
            "state" => definition.state,
        )
        if definition.state != "supported"
            row["reason"] = definition.reason
            push!(measurements, row)
            println("configuration=$(configuration.id) boundary=$boundary " *
                    "outcome=$outcome state=$(definition.state)")
            continue
        end

        native_value = _comparable(
            outcome, definition.kernel(definition.args...))
        row["native_control"] = _measurement(
            definition.kernel, definition.args...; rounds, target_seconds)
        traced_args = nothing
        row["reactant_transfer_seconds"] = @elapsed begin
            traced_args = _trace_args(definition.args)
        end
        compile_started = time_ns()
        try
            compiled = _compile_call(definition.kernel, traced_args)
            row["reactant_compile_seconds"] =
                Float64(time_ns() - compile_started) / 1e9
            compiled_value = nothing
            row["reactant_first_execution_seconds"] = @elapsed begin
                compiled_value = compiled(traced_args...)
            end
            comparable = _comparable(outcome, compiled_value)
            isapprox(comparable, native_value; rtol = 1e-11, atol = 1e-12) ||
                error("native/Reactant value parity failed")
            row["max_abs_error"] = comparable isa Number ?
                abs(comparable - native_value) :
                maximum(abs.(comparable .- native_value))
            row["result"] = _measurement(
                compiled, traced_args...; rounds, target_seconds)
        catch err
            row["state"] = "unsupported_runtime"
            row["reason"] = _diagnostic(err)
            row["reactant_compile_seconds"] = get(
                row, "reactant_compile_seconds",
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
        "schema" => "eight-schools-reactant-v2",
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
            "julia_version" => string(VERSION),
            "source_authority_path" => source_path,
            "source_authority_blob" =>
                _git(repo, "rev-parse", "$candidate_sha:$source_path"),
            "source_text_sha256" => bytes2hex(sha256(EIGHT_SCHOOLS_SOURCE)),
            "primal_receipt_sha256" =>
                _eight_schools_reactant_generator_sha256(primal_path),
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
                ENV, "RK_EIGHT_SCHOOLS_ENV_SETUP_SECONDS", "0")),
            "package_precompile_seconds" => parse(Float64, get(
                ENV, "RK_EIGHT_SCHOOLS_PRECOMPILE_SECONDS", "0")),
            "kernel_preparation_seconds" => preparation_seconds,
        ),
        "protocol" => Dict(
            "model" => "centered Eight Schools",
            "source_reused" => true,
            "matrix_source" =>
                "benchmark/receipts/eight-schools-primal-v2.toml",
            "input_boundaries" => collect(EIGHT_SCHOOLS_BOUNDARIES),
            "outcomes" => collect(EIGHT_SCHOOLS_OUTCOMES),
            "models" => collect(EIGHT_SCHOOLS_MODELS),
            "matrix_layout" =>
                "long-form provider/model/configuration/boundary/outcome rows",
            "rk_configurations" => [configuration.id for configuration in configurations],
            "bound_ports" => ["observations", "observation_scales"],
            "rounds" => rounds,
            "target_seconds_per_round" => target_seconds,
            "estimator" => "median per-call time from calibrated elapsed batches",
            "reactant_sync" => true,
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "reactant_compile_time_in_timed_region" => false,
            "reactant_transfers_in_timed_region" => false,
            "reactant_readback_in_timed_region" => false,
            "unsupported_cells_recorded" => true,
            "gradients_included" => false,
            "generated_predictions_included" => false,
            "parity_rtol" => 1e-11,
            "parity_atol" => 1e-12,
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

get(ENV, "RK_EIGHT_SCHOOLS_REACTANT_DEFINITIONS_ONLY", "") == "1" ||
    run_comparison()
