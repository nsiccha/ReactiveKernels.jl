# Inner body for the pinned Eight Schools AD-only comparison.

using BenchmarkTools
using Dates
using Pkg
using SHA
using Statistics
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, build_eight_schools_graph
using DifferentiationInterface
import DynamicPPL
import Enzyme

include(joinpath(@__DIR__, "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

include(joinpath(@__DIR__, "_ad_comparison_support.jl"))
using .ADComparisonSupport

# Load, rather than copy, the exact Turing model and manual lower-bound density
# definitions used to produce the published primal matrix. The opt-out affects
# only the file's terminal run call; the ordinary primal script remains unchanged.
module PublishedEightSchoolsPrimal
ENV["RK_EIGHT_SCHOOLS_DEFINITIONS_ONLY"] = "1"
include(joinpath(@__DIR__, "eight_schools_primal_comparison_body.jl"))
end
delete!(ENV, "RK_EIGHT_SCHOOLS_DEFINITIONS_ONLY")

const Primal = PublishedEightSchoolsPrimal
const LDP = DynamicPPL.LogDensityProblems
const RK_AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const RK_BOUND_AD_BACKEND = AutoEnzyme(
    ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
const TURING_AD_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)
const DEFAULT_EIGHT_SCHOOLS_AD_ROUNDS = 10
const EIGHT_SCHOOLS_AD_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "DifferentiationInterface", "Enzyme", "BenchmarkTools",
)
const EIGHT_SCHOOLS_AD_BOUNDARIES = EIGHT_SCHOOLS_BOUNDARIES
const EIGHT_SCHOOLS_AD_OUTCOMES = EIGHT_SCHOOLS_OUTCOMES
const EIGHT_SCHOOLS_MODEL_PUBLISHED_SHA =
    "0412b756a068dc495c1352b2d3595d0eceee4af0"
const EIGHT_SCHOOLS_COMPARATOR_PUBLISHED_SHA =
    "7d9ba71fcafdc588c25c825c2a094a15320cedc5"

const _RKValueGradientCall = RKValueGradientCall
const _DIValueGradientCall = DIValueGradientCall
const _RKAllocatingValueGradientCall = RKAllocatingValueGradientCall
const _DIAllocatingValueGradientCall = DIAllocatingValueGradientCall
const _RKValuePullbackCall = RKValuePullbackCall
const _DIValuePullbackCall = DIValuePullbackCall
const _TuringValueGradientCall = TuringValueGradientCall
const _measurement = measurement
const _build_and_first_call = build_and_first_call
const _record_implementation = record_implementation
const _gradient_error = gradient_error
const _gradient_scale = gradient_scale
const _flatten_sensitivity = flatten_sensitivity
const _package_version = package_version
const _output_path = output_path
const _source_pin = source_pin
const _published_source_pin = published_source_pin
const _nothing_paths = nothing_paths

function _finite_difference_gradient(objective, point)
    gradient = similar(point, Float64)
    plus = copy(point)
    minus = copy(point)
    for index in eachindex(point)
        step = cbrt(eps(Float64)) * max(1.0, abs(point[index]))
        plus[index] = point[index] + step
        minus[index] = point[index] - step
        gradient[index] = (objective(plus) - objective(minus)) / (2step)
        plus[index] = point[index]
        minus[index] = point[index]
    end
    gradient
end

function _finite_difference_gradient(objective, point::NamedTuple)
    names = keys(point)
    gradients = map(names, values(point)) do name, value
        if value isa AbstractFloat
            step = cbrt(eps(Float64)) * max(1.0, abs(value))
            plus = merge(point, NamedTuple{(name,)}((value + step,)))
            minus = merge(point, NamedTuple{(name,)}((value - step,)))
            (objective(plus) - objective(minus)) / (2step)
        elseif value isa AbstractArray{<:AbstractFloat}
            gradient = similar(value, Float64)
            plus_value = copy(value)
            minus_value = copy(value)
            for index in eachindex(value)
                step = cbrt(eps(Float64)) * max(1.0, abs(value[index]))
                plus_value[index] = value[index] + step
                minus_value[index] = value[index] - step
                plus = merge(point, NamedTuple{(name,)}((plus_value,)))
                minus = merge(point, NamedTuple{(name,)}((minus_value,)))
                gradient[index] = (objective(plus) - objective(minus)) / (2step)
                plus_value[index] = value[index]
                minus_value[index] = value[index]
            end
            gradient
        else
            error("unsupported finite-difference field $name::$(typeof(value))")
        end
    end
    NamedTuple{names}(gradients)
end

_pointwise_seed(observations) =
    [isodd(index) ? 0.5 + 0.1index : -(0.5 + 0.1index)
     for index in eachindex(observations)]

function _manual_definition(boundary, outcome, q, θ, parameters, observations,
                            observation_scales)
    if boundary == "packed_unconstrained"
        if outcome == "joint"
            objective = Primal._manual_unconstrained_joint
            contexts = (
                DifferentiationInterface.Constant(observations),
                DifferentiationInterface.Constant(observation_scales),
            )
            raw = x -> objective(x, observations, observation_scales)
        elseif outcome == "prior"
            objective = Primal._manual_unconstrained_prior
            contexts = ()
            raw = objective
        elseif outcome in ("likelihood", "pointwise")
            objective = Primal._manual_unconstrained_likelihood
            outcome == "pointwise" &&
                (objective = Primal._manual_unconstrained_pointwise)
            contexts = (
                DifferentiationInterface.Constant(observations),
                DifferentiationInterface.Constant(observation_scales),
            )
            raw = x -> objective(x, observations, observation_scales)
        else
            return nothing
        end
        return (; objective, contexts, raw, point = q)
    elseif boundary == "constrained_parameters"
        outcome == "pointwise" && return nothing
        objective = if outcome == "joint"
            Primal._manual_constrained_joint
        elseif outcome == "prior"
            Primal._manual_constrained_prior
        elseif outcome == "likelihood"
            Primal._manual_constrained_likelihood
        else
            return nothing
        end
        contexts = outcome == "prior" ? () : (
            DifferentiationInterface.Constant(observations),
            DifferentiationInterface.Constant(observation_scales),
        )
        raw = outcome == "prior" ? objective :
            x -> objective(x, observations, observation_scales)
        return (; objective, contexts, raw, point = parameters)
    elseif boundary == "minimal_likelihood" &&
           outcome in ("likelihood", "pointwise")
        objective = outcome == "likelihood" ?
            Primal._manual_likelihood : Primal._manual_pointwise
        contexts = (
            DifferentiationInterface.Constant(observations),
            DifferentiationInterface.Constant(observation_scales),
        )
        raw = x -> objective(x, observations, observation_scales)
        return (; objective, contexts, raw, point = θ)
    end
    nothing
end

function _rk_definition(model, boundary, outcome, q, θ, parameters, observations,
                        observation_scales, data_binding)
    if boundary == "packed_unconstrained"
        want = outcome == "joint" ? :posterior :
               outcome == "prior" ? :unconstrained_prior : Symbol(outcome)
        if outcome == "prior"
            data_binding == "bound" && return nothing
        elseif !(outcome in ("joint", "likelihood", "pointwise"))
            return nothing
        end
        have = outcome == "prior" ? (:unconstrained,) :
            (:unconstrained, :observations, :observation_scales)
        bound = data_binding == "bound" ?
            (; observations, observation_scales) : NamedTuple()
        kernel = prepare(model; have, want, bound)
        arguments = data_binding == "bound" || outcome == "prior" ?
            (q,) : (q, observations, observation_scales)
        return (; kernel, arguments, active = :unconstrained)
    elseif boundary == "constrained_parameters"
        outcome == "pointwise" && return nothing
        want = outcome == "joint" ? :constrained_logdensity : Symbol(outcome)
        data_dependent = outcome != "prior"
        have = data_dependent ?
            (:parameters, :observations, :observation_scales) : (:parameters,)
        bound = data_binding == "bound" ?
            (; observations, observation_scales) : NamedTuple()
        kernel = prepare(model; have, want, bound)
        arguments = data_binding == "bound" || !data_dependent ?
            (parameters,) : (parameters, observations, observation_scales)
        return (; kernel, arguments, active = :parameters)
    elseif boundary == "minimal_likelihood" &&
           outcome in ("likelihood", "pointwise")
        bound = data_binding == "bound" ?
            (; observations, observation_scales) : NamedTuple()
        kernel = prepare(model;
            have = (:θ, :observations, :observation_scales),
            want = Symbol(outcome), bound)
        arguments = data_binding == "bound" ? (θ,) :
            (θ, observations, observation_scales)
        return (; kernel, arguments,
                active = :θ)
    end
    nothing
end

function _prepare_rk_ad(definition, data_binding)
    backend = data_binding == "bound" ? RK_BOUND_AD_BACKEND : RK_AD_BACKEND
    prepare_ad(
        definition.kernel, backend, definition.arguments...;
        active = definition.active)
end

function _turing_logdensity(model, outcome)
    evaluator = outcome == "joint" ? DynamicPPL.getlogjoint_internal :
                outcome == "prior" ? DynamicPPL.getlogprior_internal :
                outcome == "likelihood" ? DynamicPPL.getloglikelihood : nothing
    isnothing(evaluator) && return nothing
    DynamicPPL.LogDensityFunction(
        model, evaluator, DynamicPPL.LinkAll();
        fix_transforms = true, adtype = TURING_AD_BACKEND)
end

function _unsupported_reason(boundary, outcome)
    boundary == "constrained_parameters" && outcome == "pointwise" && return (
        "the public reverse-pullback surface supports pointwise and structured " *
        "gradients separately, but Enzyme cannot annotate their MixedDuplicated cross-product")
    boundary == "minimal_likelihood" && return (
        "joint and prior are unavailable from the minimal likelihood HAVE boundary")
    "unsupported matrix cell"
end

_rounds() = parse(Int, get(
    ENV, "RK_EIGHT_SCHOOLS_AD_ROUNDS", string(DEFAULT_EIGHT_SCHOOLS_AD_ROUNDS)))

const _eight_schools_ad_generator_text = normalized_text
const _eight_schools_ad_generator_read = normalized_read
const _eight_schools_ad_generator_sha256 = text_sha256

function _verified_model_source_pin(root, relative_path)
    published, text = _published_source_pin(
        root, EIGHT_SCHOOLS_MODEL_PUBLISHED_SHA, relative_path)
    current = _eight_schools_ad_generator_read(joinpath(root, relative_path))
    eight_schools_model_source_preserves_published_authority(current, text) || error(
        "Eight Schools model source no longer preserves published authority " *
        EIGHT_SCHOOLS_MODEL_PUBLISHED_SHA)
    merge(published, Dict(
        "current" => _source_pin(root, relative_path),
        "current_delta" => EIGHT_SCHOOLS_MODEL_SOURCE_CURRENT_DELTA,
    ))
end

function _verified_comparator_source_pin(root, relative_path)
    published, text = _published_source_pin(
        root, EIGHT_SCHOOLS_COMPARATOR_PUBLISHED_SHA, relative_path)
    guard = "get(ENV, \"RK_EIGHT_SCHOOLS_DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n"
    current = _eight_schools_ad_generator_read(joinpath(root, relative_path))
    comparator_source_matches_current_delta(current, text, guard) || error(
        "Eight Schools comparator differs from its published authority by more " *
        COMPARATOR_SOURCE_CURRENT_DELTA)
    merge(published, Dict(
        "current" => _source_pin(root, relative_path),
        "current_delta" => COMPARATOR_SOURCE_CURRENT_DELTA,
    ))
end

function run_eight_schools_ad_comparison()
    rounds = _rounds()
    root = normpath(joinpath(@__DIR__, ".."))
    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "eight_schools.jl")
    comparator_path = joinpath("benchmark", "eight_schools_primal_comparison_body.jl")
    primal_receipt_path =
        joinpath("benchmark", "receipts", "eight-schools-primal-v2.toml")
    primal_path = get(
        ENV, "RK_EIGHT_SCHOOLS_PRIMAL_RECEIPT", primal_receipt_path)
    primal_absolute = isabspath(primal_path) ? primal_path : joinpath(root, primal_path)

    # Fail authority/prerequisite gates before setup or any timed measurement.
    model_source_pin = _verified_model_source_pin(root, source_path)
    comparator_source_pin = _verified_comparator_source_pin(root, comparator_path)
    primal_receipt = TOML.parsefile(primal_absolute)
    observations = Float64.(EIGHT_SCHOOLS_Y)
    observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
    μ = 1.5
    log_τ = log(2.0)
    τ = exp(log_τ)
    θ = 0.25 .* collect(1.0:8.0)
    q = [μ, log_τ, θ...]
    parameters = (; μ, τ, θ)

    model_setup = @timed build_eight_schools_graph()
    model = model_setup.value
    turing_setup = @timed Primal.turing_eight_schools(
        observations, observation_scales)
    turing_model = turing_setup.value
    turing_joint = DynamicPPL.LogDensityFunction(
        turing_model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_q = Primal._turing_vector(turing_joint, parameters)
    isapprox(turing_q, q; rtol = 1e-12, atol = 1e-12) ||
        error("Turing's linked parameter vector does not match the RK boundary")

    native_ad_configurations = filter(
        configuration -> configuration.differentiation == "value_and_gradient" &&
            configuration.compiler == "native",
        EIGHT_SCHOOLS_RK_CONFIGURATIONS,
    )
    measurements = Dict{String,Any}[]
    for boundary in EIGHT_SCHOOLS_AD_BOUNDARIES,
        outcome in EIGHT_SCHOOLS_AD_OUTCOMES
        base_state, base_reason = matrix_support(
            first(native_ad_configurations), boundary, outcome)
        manual_definition = _manual_definition(
            boundary, outcome, q, θ, parameters,
            observations, observation_scales)
        pointwise = outcome == "pointwise"
        seed = pointwise ? _pointwise_seed(observations) : nothing
        reference_value = base_state == "supported" ?
            manual_definition.raw(manual_definition.point) : nothing
        scalar_reference = base_state != "supported" ? nothing : pointwise ?
            x -> sum(seed .* manual_definition.raw(x)) : manual_definition.raw
        reference_gradient = base_state == "supported" ?
            _finite_difference_gradient(scalar_reference, manual_definition.point) :
            nothing
        structured = base_state == "supported" &&
            manual_definition.point isa NamedTuple

        for configuration in native_ad_configurations
            state, reason = matrix_support(configuration, boundary, outcome)
            row = Dict{String,Any}(
                "provider" => "rk", "model" => "centered",
                "configuration" => configuration.id,
                "boundary" => boundary, "outcome" => outcome,
                "state" => state,
            )
            if state == "supported"
                definition = _rk_definition(
                    model, boundary, outcome, q, θ, parameters, observations,
                    observation_scales, configuration.data)
                rk_call, rk_result, rk_setup = _build_and_first_call() do
                    backend = configuration.data == "bound" ?
                        RK_BOUND_AD_BACKEND : RK_AD_BACKEND
                    if pointwise
                        prepared = prepare_ad_pullback(
                            definition.kernel, backend, seed,
                            definition.arguments...; active = definition.active)
                        _RKValuePullbackCall(
                            prepared, similar(manual_definition.point), seed,
                            definition.arguments)
                    elseif structured
                        prepared = _prepare_rk_ad(definition, configuration.data)
                        _RKAllocatingValueGradientCall(
                            prepared, definition.arguments)
                    else
                        prepared = _prepare_rk_ad(definition, configuration.data)
                        _RKValueGradientCall(
                            prepared, similar(manual_definition.point),
                            definition.arguments)
                    end
                end
                row["active_port"] = String(definition.active)
                row["operation"] = pointwise ?
                    "value and pullback" : "value and gradient"
                pointwise && (row["output_cotangent"] = seed)
                row["finite_difference_gradient"] =
                    _flatten_sensitivity(reference_gradient)
                row["result"] = _record_implementation(
                    rk_call, rk_result, rk_setup,
                    reference_value, reference_gradient;
                    rounds, caller_owned = !structured)
            else
                row["reason"] = reason
            end
            push!(measurements, row)
        end

        manual_row = Dict{String,Any}(
            "provider" => "manual_julia", "model" => "centered",
            "configuration" => "manual_ad", "boundary" => boundary,
            "outcome" => outcome, "state" => base_state,
        )
        if base_state == "supported"
            manual_call, manual_result, manual_setup = _build_and_first_call() do
                if pointwise
                    preparation = DifferentiationInterface.prepare_pullback(
                        manual_definition.objective, RK_AD_BACKEND,
                        manual_definition.point, (seed,),
                        manual_definition.contexts...)
                    _DIValuePullbackCall(
                        manual_definition.objective,
                        similar(manual_definition.point), seed, preparation,
                        RK_AD_BACKEND, manual_definition.point,
                        manual_definition.contexts)
                elseif structured
                    preparation = DifferentiationInterface.prepare_gradient(
                        manual_definition.objective, RK_AD_BACKEND,
                        manual_definition.point, manual_definition.contexts...)
                    _DIAllocatingValueGradientCall(
                        manual_definition.objective, preparation, RK_AD_BACKEND,
                        manual_definition.point, manual_definition.contexts)
                else
                    preparation = DifferentiationInterface.prepare_gradient(
                        manual_definition.objective, RK_AD_BACKEND,
                        manual_definition.point, manual_definition.contexts...)
                    _DIValueGradientCall(
                        manual_definition.objective,
                        similar(manual_definition.point), preparation,
                        RK_AD_BACKEND, manual_definition.point,
                        manual_definition.contexts)
                end
            end
            manual_row["active_port"] = String(boundary == "minimal_likelihood" ?
                :θ : boundary == "constrained_parameters" ?
                :parameters : :unconstrained)
            manual_row["operation"] = pointwise ?
                "value and pullback" : "value and gradient"
            pointwise && (manual_row["output_cotangent"] = seed)
            manual_row["finite_difference_gradient"] =
                _flatten_sensitivity(reference_gradient)
            manual_row["result"] = _record_implementation(
                manual_call, manual_result, manual_setup,
                reference_value, reference_gradient;
                rounds, caller_owned = !structured)
        else
            manual_row["reason"] = base_reason
        end
        push!(measurements, manual_row)

        turing_state = base_state == "supported" && !pointwise &&
            boundary == "packed_unconstrained" ? "supported" : "unsupported"
        turing_row = Dict{String,Any}(
            "provider" => "turing", "model" => "centered",
            "configuration" => "turing_ad", "boundary" => boundary,
            "outcome" => outcome, "state" => turing_state,
        )
        if turing_state == "supported"
            turing_call, turing_result, turing_ad_setup = _build_and_first_call() do
                _TuringValueGradientCall(
                    _turing_logdensity(turing_model, outcome), turing_q)
            end
            turing_row["active_port"] = "unconstrained"
            turing_row["finite_difference_gradient"] =
                _flatten_sensitivity(reference_gradient)
            turing_row["result"] = _record_implementation(
                turing_call, turing_result, turing_ad_setup,
                reference_value, reference_gradient;
                rounds, caller_owned = false)
        else
            turing_row["reason"] = base_state != "supported" ? base_reason :
                pointwise ?
                    "the compared Turing density has no matched public pointwise VJP surface" :
                    "Turing has no public constrained or θ-only AD boundary"
        end
        push!(measurements, turing_row)
        println("boundary=$boundary outcome=$outcome complete")
    end

    receipt = Dict{String,Any}(
        "schema" => "eight-schools-ad-v2",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            "julia_version" => string(VERSION),
            "model_source" => model_source_pin,
            "primal_comparator_source" => comparator_source_pin,
            "primal_receipt_path" => primal_receipt_path,
            "primal_receipt_sha256" => _eight_schools_ad_generator_sha256(
                primal_absolute),
            "primal_receipt_reactivekernels_sha" =>
                primal_receipt["pins"]["reactivekernels_sha"],
            (string(lowercase(name), "_version") => _package_version(name)
             for name in EIGHT_SCHOOLS_AD_PACKAGES)...,
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
        ),
        "setup" => Dict(
            "rk_model_build_seconds" => model_setup.time,
            "rk_model_build_bytes" => model_setup.bytes,
            "turing_model_build_seconds" => turing_setup.time,
            "turing_model_build_bytes" => turing_setup.bytes,
        ),
        "protocol" => Dict(
            "model" => "centered Eight Schools",
            "input_boundaries" => collect(EIGHT_SCHOOLS_AD_BOUNDARIES),
            "outcomes" => collect(EIGHT_SCHOOLS_AD_OUTCOMES),
            "models" => collect(EIGHT_SCHOOLS_MODELS),
            "matrix_layout" =>
                "long-form provider/model/configuration/boundary/outcome rows",
            "rk_configurations" => [
                configuration.id for configuration in native_ad_configurations],
            "bound_ports" => ["observations", "observation_scales"],
            "source_reused" => true,
            "matrix_source" => primal_receipt_path,
            "gradient_operation" =>
                "value and gradient for scalar WANTs; value and reverse pullback for pointwise WANTs",
            "rk_surface" =>
                "prepare_ad + ad_value_and_gradient[!]; prepare_ad_pullback + ad_value_and_pullback[!]",
            "rk_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
            "rk_bound_backend" =>
                "AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)",
            "manual_control" =>
                "the primal receipt's manual Julia density differentiated through the same prepared DI+Enzyme boundary",
            "turing_surface" =>
                "LogDensityProblems.logdensity_and_gradient on DynamicPPL.LogDensityFunction",
            "turing_backend" =>
                "AutoEnzyme(runtime activity, Const function annotation)",
            "parity_oracle" => "central finite differences of the manual Julia density",
            "pointwise_jacobian_or_vjp_invented" => false,
            "pointwise_vjp_contract" =>
                "one deterministic output cotangent through public prepared reverse pullbacks; no full Jacobian",
            "structured_cotangent_ownership" =>
                "NamedTuple sensitivities use public allocating DI/RK gradients; array sensitivities use caller-owned destinations",
            "preparation_in_timed_region" => false,
            "first_execution_in_steady_state_region" => false,
            "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "parity_relative_tolerance" => 5e-6,
            "parity_absolute_tolerance" => 5e-8,
        ),
        "measurements" => measurements,
    )

    nothing_paths = _nothing_paths(receipt)
    isempty(nothing_paths) || error(
        "receipt contains non-TOML nothing values at " * join(nothing_paths, ", "))

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

get(ENV, "RK_EIGHT_SCHOOLS_AD_DEFINITIONS_ONLY", "") == "1" ||
    run_eight_schools_ad_comparison()
