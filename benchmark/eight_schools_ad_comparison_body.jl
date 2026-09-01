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
const EIGHT_SCHOOLS_AD_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood")
const EIGHT_SCHOOLS_AD_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const EIGHT_SCHOOLS_MODEL_PUBLISHED_SHA =
    "0412b756a068dc495c1352b2d3595d0eceee4af0"
const EIGHT_SCHOOLS_COMPARATOR_PUBLISHED_SHA =
    "7d9ba71fcafdc588c25c825c2a094a15320cedc5"

const _RKValueGradientCall = RKValueGradientCall
const _DIValueGradientCall = DIValueGradientCall
const _TuringValueGradientCall = TuringValueGradientCall
const _measurement = measurement
const _build_and_first_call = build_and_first_call
const _record_implementation = record_implementation
const _gradient_error = gradient_error
const _gradient_scale = gradient_scale
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

function _manual_definition(boundary, outcome, q, θ, observations,
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
        elseif outcome == "likelihood"
            objective = Primal._manual_unconstrained_likelihood
            contexts = (
                DifferentiationInterface.Constant(observations),
                DifferentiationInterface.Constant(observation_scales),
            )
            raw = x -> objective(x, observations, observation_scales)
        else
            return nothing
        end
        return (; objective, contexts, raw, point = q)
    elseif boundary == "minimal_likelihood" && outcome == "likelihood"
        objective = Primal._manual_likelihood
        contexts = (
            DifferentiationInterface.Constant(observations),
            DifferentiationInterface.Constant(observation_scales),
        )
        raw = x -> objective(x, observations, observation_scales)
        return (; objective, contexts, raw, point = θ)
    end
    nothing
end

function _rk_definition(model, boundary, outcome, q, θ, observations,
                        observation_scales)
    if boundary == "packed_unconstrained"
        if outcome == "joint"
            kernel = prepare(model;
                have = (:unconstrained, :observations, :observation_scales),
                want = :posterior)
            arguments = (q, observations, observation_scales)
        elseif outcome == "prior"
            kernel = prepare(model;
                have = :unconstrained, want = :unconstrained_prior)
            arguments = (q,)
        elseif outcome == "likelihood"
            kernel = prepare(model;
                have = (:unconstrained, :observations, :observation_scales),
                want = :likelihood)
            arguments = (q, observations, observation_scales)
        else
            return nothing
        end
        return (; kernel, arguments, active = :unconstrained)
    elseif boundary == "minimal_likelihood" && outcome == "likelihood"
        kernel = prepare(model;
            have = (:θ, :observations, :observation_scales), want = :likelihood)
        return (; kernel, arguments = (θ, observations, observation_scales),
                active = :θ)
    end
    nothing
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
    outcome == "pointwise" && return (
        "pointwise is vector-valued and neither compared public surface exposes " *
        "a useful matched Jacobian/VJP contract")
    boundary == "constrained_parameters" && return (
        "the public RK AD boundary accepts floating scalar/array/tuple storage, " *
        "not the primal matrix's constrained NamedTuple")
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
    _eight_schools_ad_generator_read(joinpath(root, relative_path)) == text || error(
        "Eight Schools model source drifted from published authority " *
        EIGHT_SCHOOLS_MODEL_PUBLISHED_SHA)
    merge(published, Dict("current" => _source_pin(root, relative_path)))
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

    measurements = Dict{String,Any}[]
    for boundary in EIGHT_SCHOOLS_AD_BOUNDARIES,
        outcome in EIGHT_SCHOOLS_AD_OUTCOMES
        manual_definition = _manual_definition(
            boundary, outcome, q, θ, observations, observation_scales)
        rk_definition = _rk_definition(
            model, boundary, outcome, q, θ, observations, observation_scales)
        row = Dict{String,Any}(
            "boundary" => boundary,
            "outcome" => outcome,
        )
        if isnothing(rk_definition) || isnothing(manual_definition)
            row["supported"] = false
            row["unsupported_reason"] = _unsupported_reason(boundary, outcome)
            push!(measurements, row)
            println("boundary=$boundary outcome=$outcome unsupported")
            continue
        end

        row["supported"] = true
        row["active_port"] = String(rk_definition.active)
        reference_value = Float64(manual_definition.raw(manual_definition.point))
        reference_gradient = _finite_difference_gradient(
            manual_definition.raw, manual_definition.point)
        row["finite_difference_gradient"] = collect(reference_gradient)

        rk_call, rk_result, rk_setup = _build_and_first_call() do
            prepared = prepare_ad(
                rk_definition.kernel, RK_AD_BACKEND,
                rk_definition.arguments...; active = rk_definition.active)
            _RKValueGradientCall(
                prepared, similar(manual_definition.point), rk_definition.arguments)
        end
        row["rk_native"] = _record_implementation(
            rk_call, rk_result, rk_setup, reference_value, reference_gradient;
            rounds, caller_owned = true)

        manual_call, manual_result, manual_setup = _build_and_first_call() do
            preparation = DifferentiationInterface.prepare_gradient(
                manual_definition.objective, RK_AD_BACKEND,
                manual_definition.point, manual_definition.contexts...)
            _DIValueGradientCall(
                manual_definition.objective, similar(manual_definition.point),
                preparation, RK_AD_BACKEND, manual_definition.point,
                manual_definition.contexts)
        end
        row["manual_enzyme"] = _record_implementation(
            manual_call, manual_result, manual_setup,
            reference_value, reference_gradient;
            rounds, caller_owned = true)

        if boundary == "packed_unconstrained"
            turing_call, turing_result, turing_ad_setup = _build_and_first_call() do
                _TuringValueGradientCall(
                    _turing_logdensity(turing_model, outcome), turing_q)
            end
            row["turing_enzyme"] = _record_implementation(
                turing_call, turing_result, turing_ad_setup,
                reference_value, reference_gradient;
                rounds, caller_owned = false)
        else
            row["turing_supported"] = false
            row["turing_unsupported_reason"] =
                "the published primal matrix has no Turing minimal-likelihood boundary"
        end

        push!(measurements, row)
        println("boundary=$boundary outcome=$outcome complete")
    end

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "eight_schools.jl")
    primal_path = joinpath("benchmark", "receipts", "eight-schools-primal-v1.toml")
    comparator_path = joinpath("benchmark", "eight_schools_primal_comparison_body.jl")
    primal_receipt = TOML.parsefile(joinpath(root, primal_path))
    receipt = Dict{String,Any}(
        "schema" => "eight-schools-ad-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            "julia_version" => string(VERSION),
            "model_source" => _verified_model_source_pin(root, source_path),
            "primal_comparator_source" =>
                _verified_comparator_source_pin(root, comparator_path),
            "primal_receipt_path" => primal_path,
            "primal_receipt_sha256" => _eight_schools_ad_generator_sha256(
                joinpath(root, primal_path)),
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
            "source_reused" => true,
            "matrix_source" => primal_path,
            "gradient_operation" => "value and gradient",
            "rk_surface" => "prepare_ad + ad_value_and_gradient!",
            "rk_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
            "manual_control" =>
                "the primal receipt's manual Julia density differentiated through the same prepared DI+Enzyme boundary",
            "turing_surface" =>
                "LogDensityProblems.logdensity_and_gradient on DynamicPPL.LogDensityFunction",
            "turing_backend" =>
                "AutoEnzyme(runtime activity, Const function annotation)",
            "parity_oracle" => "central finite differences of the manual Julia density",
            "pointwise_jacobian_or_vjp_invented" => false,
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

run_eight_schools_ad_comparison()
