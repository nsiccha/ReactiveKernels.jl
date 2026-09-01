# Inner body for the pinned Eight Schools primal comparison.

using BenchmarkTools
using Dates
using Pkg
using Statistics
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, build_eight_schools_graph
import DynamicPPL
import Turing
using Turing: filldist
using Distributions: Cauchy, Normal, truncated

include(joinpath(@__DIR__, "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

const LDP = DynamicPPL.LogDensityProblems
const DEFAULT_EIGHT_SCHOOLS_ROUNDS = 10
const COMPARISON_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "BenchmarkTools", "MutatingFunctions",
)

if get(ENV, "RK_EIGHT_SCHOOLS_DEFINITIONS_ONLY", "") != "1"
    using MutatingFunctions
    Base.get_extension(ReactiveKernels, :ReactiveKernelsMutatingFunctionsExt) === nothing &&
        error("the non-allocating column requires ReactiveKernelsMutatingFunctionsExt to load")
end

# DOCS-BASELINE-BEGIN: turing
Turing.@model function turing_eight_schools(observations, observation_scales)
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5); lower = 0)
    # θⱼ ~ Normal(μ, τ) as an i.i.d. product. This is the exact centered
    # Eight Schools prior that `MvNormal(fill(μ, J), τ^2 * I)` encodes, but the
    # `filldist` form avoids the mean-vector allocation and the dense MvNormal
    # evaluation path, so it is the allocation-lighter public Turing baseline
    # while keeping the identical density and linked parameter order.
    θ ~ filldist(Normal(μ, τ), length(observations))
    for j in eachindex(observations)
        observations[j] ~ Normal(θ[j], observation_scales[j])
    end
    return nothing
end
# DOCS-BASELINE-END: turing

# DOCS-BASELINE-BEGIN: manual
_normal_logpdf(x, location, scale, log_scale = log(scale)) =
    -0.5 * log(2π) - log_scale - 0.5 * ((x - location) / scale)^2

function _manual_prior(μ, τ, θ)
    log_τ = log(τ)
    total = _normal_logpdf(μ, 0.0, 5.0)
    total += log(2.0) - log(π) - log(5.0) - log1p((τ / 5.0)^2)
    @inbounds for θj in θ
        total += _normal_logpdf(θj, μ, τ, log_τ)
    end
    total
end

function _manual_likelihood(θ, observations, observation_scales)
    total = 0.0
    @inbounds for j in eachindex(observations, θ, observation_scales)
        total += _normal_logpdf(observations[j], θ[j], observation_scales[j])
    end
    total
end

function _manual_pointwise(θ, observations, observation_scales)
    output = similar(observations, Float64)
    @inbounds for j in eachindex(observations, θ, observation_scales)
        output[j] = _normal_logpdf(
            observations[j], θ[j], observation_scales[j])
    end
    output
end

_manual_constrained(μ, τ, θ, observations, observation_scales) =
    _manual_prior(μ, τ, θ) +
    _manual_likelihood(θ, observations, observation_scales)

_manual_unconstrained_joint(q, observations, observation_scales) =
    _manual_constrained(q[1], exp(q[2]), @view(q[3:end]),
                        observations, observation_scales) + q[2]
_manual_unconstrained_prior(q) =
    _manual_prior(q[1], exp(q[2]), @view(q[3:end])) + q[2]
_manual_unconstrained_likelihood(q, observations, observation_scales) =
    _manual_likelihood(@view(q[3:end]), observations, observation_scales)
_manual_unconstrained_pointwise(q, observations, observation_scales) =
    _manual_pointwise(@view(q[3:end]), observations, observation_scales)

_manual_constrained_joint(parameters, observations, observation_scales) =
    _manual_constrained(parameters.μ, parameters.τ, parameters.θ,
                        observations, observation_scales)
_manual_constrained_prior(parameters) =
    _manual_prior(parameters.μ, parameters.τ, parameters.θ)
_manual_constrained_likelihood(parameters, observations, observation_scales) =
    _manual_likelihood(parameters.θ, observations, observation_scales)
_manual_constrained_pointwise(parameters, observations, observation_scales) =
    _manual_pointwise(parameters.θ, observations, observation_scales)
# DOCS-BASELINE-END: manual

function _turing_vector(ldf, parameters)
    accumulator = DynamicPPL.OnlyAccsVarInfo(
        DynamicPPL.VectorParamAccumulator(ldf))
    _, accumulator = DynamicPPL.init!!(
        ldf.model,
        accumulator,
        DynamicPPL.InitFromParams(parameters),
        ldf.transform_strategy,
    )
    collect(DynamicPPL.get_vector_params(accumulator))
end

_turing_density(ldf, parameters) = LDP.logdensity(ldf, parameters)
_turing_pointwise(model, init) =
    DynamicPPL.pointwise_loglikelihoods(model, init)

_turing_constrained_joint(model, parameters) =
    DynamicPPL.logjoint(model, parameters)
_turing_constrained_prior(model, parameters) =
    DynamicPPL.logprior(model, parameters)
_turing_constrained_likelihood(model, parameters) =
    DynamicPPL.loglikelihood(model, parameters)

_pointwise_vector(pointwise::AbstractVector) = collect(Float64, pointwise)
function _pointwise_vector(pointwise)
    output = Float64[]
    for value in values(pointwise)
        value isa AbstractArray ? append!(output, value) : push!(output, value)
    end
    output
end

_evaluate(spec) = first(spec)(last(spec)...)
_comparable(outcome, value) =
    outcome == "pointwise" ? _pointwise_vector(value) : Float64(value)

function _measurement(f, args...; rounds::Int)
    # Capture the already-prepared call once. BenchmarkTools then invokes a
    # concrete zero-argument closure; Julia inlines the tuple splat, while the
    # timed region contains neither setup nor dynamic argument construction.
    invocation = let f = f, args = args
        () -> f(args...)
    end
    benchmark = @benchmarkable $invocation()
    times_ns = Float64[]
    bytes = Int[]
    allocs = Int[]
    for _ in 1:rounds
        estimate = minimum(run(benchmark; samples = 200, seconds = 0.2))
        push!(times_ns, estimate.time)
        push!(bytes, estimate.memory)
        push!(allocs, estimate.allocs)
    end
    Dict(
        "times_ns" => times_ns,
        "median_ns" => median(times_ns),
        "bytes" => bytes,
        "median_bytes" => Int(median(bytes)),
        "allocs" => allocs,
        "median_allocs" => Int(median(allocs)),
    )
end

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name absent from the benchmark environment")
end

function _package_git_revision(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(something(info.git_revision, "unknown"))
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
    ENV, "RK_EIGHT_SCHOOLS_ROUNDS", string(DEFAULT_EIGHT_SCHOOLS_ROUNDS)))

_eight_schools_want(outcome) =
    outcome == "joint" ? nothing : Symbol(outcome)

function _rk_primal_definition(model, configuration, boundary, outcome,
                               q, parameters, theta, observations,
                               observation_scales)
    state, reason = matrix_support(configuration, boundary, outcome)
    state == "supported" || return (; state, reason, spec = nothing)
    configuration.differentiation == "primal" || return (
        state = "unsupported", reason = "not a primal execution configuration",
        spec = nothing)
    configuration.compiler == "native" || return (
        state = "unsupported",
        reason = "Reactant configurations are measured by the matched compiler receipt",
        spec = nothing)

    data_dependent = outcome != "prior"
    have, want, arguments = if boundary == "packed_unconstrained"
        h = data_dependent ?
            (:unconstrained, :observations, :observation_scales) :
            (:unconstrained,)
        w = outcome == "joint" ? :posterior : Symbol(outcome == "prior" ?
            "unconstrained_prior" : outcome)
        a = data_dependent && configuration.data == "unbound" ?
            (q, observations, observation_scales) : (q,)
        (h, w, a)
    elseif boundary == "constrained_parameters"
        h = data_dependent ?
            (:parameters, :observations, :observation_scales) : (:parameters,)
        w = outcome == "joint" ? :constrained_logdensity : Symbol(outcome)
        a = data_dependent && configuration.data == "unbound" ?
            (parameters, observations, observation_scales) : (parameters,)
        (h, w, a)
    else
        h = (:θ, :observations, :observation_scales)
        a = configuration.data == "unbound" ?
            (theta, observations, observation_scales) : (theta,)
        (h, Symbol(outcome), a)
    end
    bound = configuration.data == "bound" ?
        (; observations, observation_scales) : NamedTuple()
    kernel = configuration.allocation == "ordinary" ?
        prepare(model; have, want, bound) :
        prepare_nonallocating(model; have, want, bound)
    (; state, reason, spec = (kernel, arguments))
end

function _manual_primal_definition(boundary, outcome, q, parameters, theta,
                                   observations, observation_scales)
    if boundary == "packed_unconstrained"
        outcome == "joint" && return (_manual_unconstrained_joint,
            (q, observations, observation_scales))
        outcome == "prior" && return (_manual_unconstrained_prior, (q,))
        outcome == "likelihood" && return (_manual_unconstrained_likelihood,
            (q, observations, observation_scales))
        return (_manual_unconstrained_pointwise,
                (q, observations, observation_scales))
    elseif boundary == "constrained_parameters"
        outcome == "joint" && return (_manual_constrained_joint,
            (parameters, observations, observation_scales))
        outcome == "prior" && return (_manual_constrained_prior, (parameters,))
        outcome == "likelihood" && return (_manual_constrained_likelihood,
            (parameters, observations, observation_scales))
        return (_manual_constrained_pointwise,
                (parameters, observations, observation_scales))
    end
    outcome == "joint" || outcome == "prior" ? nothing :
        outcome == "likelihood" ?
            (_manual_likelihood, (theta, observations, observation_scales)) :
            (_manual_pointwise, (theta, observations, observation_scales))
end

function _turing_primal_definition(turing, boundary, outcome, parameters)
    boundary == "minimal_likelihood" && return nothing
    if boundary == "packed_unconstrained"
        outcome == "pointwise" && return (
            _turing_pointwise, (turing.model, turing.unconstrained_pointwise_init))
        view = outcome == "joint" ? turing.joint :
               outcome == "prior" ? turing.prior : turing.likelihood
        return (_turing_density, (view, turing.vector))
    end
    outcome == "pointwise" && return (
        _turing_pointwise, (turing.model, turing.constrained_pointwise_init))
    evaluator = outcome == "joint" ? _turing_constrained_joint :
                outcome == "prior" ? _turing_constrained_prior :
                _turing_constrained_likelihood
    (evaluator, (turing.model, parameters))
end

function _measurement_row(; provider, configuration, boundary, outcome,
                          description, state = "supported", reason = "",
                          result = nothing)
    row = Dict{String,Any}(
        "provider" => provider,
        "model" => "centered",
        "configuration" => configuration,
        "boundary" => boundary,
        "outcome" => outcome,
        "description" => description,
        "state" => state,
    )
    isempty(reason) || (row["reason"] = reason)
    result === nothing || (row["result"] = result)
    row
end

function run_comparison()
    rounds = _rounds()
    observations = Float64.(EIGHT_SCHOOLS_Y)
    observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
    μ = 1.5
    log_τ = log(2.0)
    τ = exp(log_τ)
    θ = 0.25 .* collect(1.0:8.0)
    unconstrained = [μ, log_τ, θ...]
    parameters = (; μ, τ, θ)

    rk_model = build_eight_schools_graph()
    native_primal_configurations = filter(
        configuration -> configuration.differentiation == "primal" &&
            configuration.compiler == "native",
        EIGHT_SCHOOLS_RK_CONFIGURATIONS,
    )

    turing_model = turing_eight_schools(observations, observation_scales)
    turing_unconstrained_joint = DynamicPPL.LogDensityFunction(
        turing_model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
        fix_transforms = true,
    )
    turing_unconstrained_prior = DynamicPPL.LogDensityFunction(
        turing_model, DynamicPPL.getlogprior_internal, DynamicPPL.LinkAll();
        fix_transforms = true,
    )
    turing_unconstrained_likelihood = DynamicPPL.LogDensityFunction(
        turing_model, DynamicPPL.getloglikelihood, DynamicPPL.LinkAll();
        fix_transforms = true,
    )
    turing_unconstrained_parameters = _turing_vector(
        turing_unconstrained_joint, parameters)
    isapprox(turing_unconstrained_parameters, unconstrained;
             rtol = 1e-12, atol = 1e-12) ||
        error("Turing's linked parameter vector does not match the RK boundary")
    turing_unconstrained_pointwise_init = DynamicPPL.InitFromVector(
        turing_unconstrained_parameters, turing_unconstrained_joint)
    turing_constrained_pointwise_init = DynamicPPL.InitFromParams(
        parameters, nothing)
    turing = (
        model = turing_model,
        joint = turing_unconstrained_joint,
        prior = turing_unconstrained_prior,
        likelihood = turing_unconstrained_likelihood,
        vector = turing_unconstrained_parameters,
        unconstrained_pointwise_init = turing_unconstrained_pointwise_init,
        constrained_pointwise_init = turing_constrained_pointwise_init,
    )
    descriptions = Dict(
        "joint" => "full joint over the selected parameter boundary",
        "prior" => "log prior over the selected parameter boundary",
        "likelihood" => "summed observation log likelihood",
        "pointwise" => "eight pointwise observation log likelihoods",
    )

    measurements = Dict{String,Any}[]
    for boundary in EIGHT_SCHOOLS_BOUNDARIES, outcome in EIGHT_SCHOOLS_OUTCOMES
        description = descriptions[outcome]
        manual = _manual_primal_definition(
            boundary, outcome, unconstrained, parameters, θ,
            observations, observation_scales)
        reference = manual === nothing ? nothing :
            _comparable(outcome, _evaluate(manual))

        for configuration in native_primal_configurations
            definition = _rk_primal_definition(
                rk_model, configuration, boundary, outcome, unconstrained,
                parameters, θ, observations, observation_scales)
            result = nothing
            if definition.state == "supported"
                actual = _comparable(outcome, _evaluate(definition.spec))
                isapprox(actual, reference; rtol = 1e-11, atol = 1e-12) ||
                    error("RK parity failed for $(configuration.id) / " *
                          "$boundary / $outcome")
                function_, arguments = definition.spec
                result = _measurement(function_, arguments...; rounds)
            end
            push!(measurements, _measurement_row(;
                provider = "rk", configuration = configuration.id,
                boundary, outcome, description, state = definition.state,
                reason = definition.reason, result))
        end

        if manual === nothing
            push!(measurements, _measurement_row(;
                provider = "manual_julia", configuration = "manual_primal",
                boundary, outcome, description, state = "unsupported",
                reason = "joint and prior are unavailable from the minimal likelihood boundary"))
        else
            function_, arguments = manual
            push!(measurements, _measurement_row(;
                provider = "manual_julia", configuration = "manual_primal",
                boundary, outcome, description,
                result = _measurement(function_, arguments...; rounds)))
        end

        turing_spec = _turing_primal_definition(
            turing, boundary, outcome, parameters)
        if turing_spec === nothing
            push!(measurements, _measurement_row(;
                provider = "turing", configuration = "turing_primal",
                boundary, outcome, description, state = "unsupported",
                reason = "Turing has no public θ-only minimal-likelihood boundary"))
        else
            actual = _comparable(outcome, _evaluate(turing_spec))
            isapprox(actual, reference; rtol = 1e-11, atol = 1e-12) ||
                error("Turing parity failed for $boundary / $outcome")
            function_, arguments = turing_spec
            push!(measurements, _measurement_row(;
                provider = "turing", configuration = "turing_primal",
                boundary, outcome, description,
                result = _measurement(function_, arguments...; rounds)))
        end
        println("boundary=$boundary outcome=$outcome complete")
    end

    receipt = Dict{String,Any}(
        "schema" => "eight-schools-primal-v2",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            (string(lowercase(name), "_version") => _package_version(name)
             for name in COMPARISON_PACKAGES)...,
            "mutatingfunctions_rev" => _package_git_revision("MutatingFunctions"),
            "julia_version" => string(VERSION),
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
        ),
        "protocol" => Dict(
            "model" => "centered Eight Schools",
            "models" => collect(EIGHT_SCHOOLS_MODELS),
            "input_boundaries" => collect(EIGHT_SCHOOLS_BOUNDARIES),
            "outcomes" => collect(EIGHT_SCHOOLS_OUTCOMES),
            "rk_configurations" => [
                configuration.id for configuration in native_primal_configurations],
            "matrix_layout" =>
                "long-form provider/model/configuration/boundary/outcome rows",
            "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "turing_transform_strategy" => "fixed transforms",
            "turing_pointwise_api" => "DynamicPPL.pointwise_loglikelihoods",
            "unsupported_cells_recorded" => true,
            "rk_nonallocating_pass" =>
                "prepare_nonallocating over the public graph (MutatingFunctions extension; caches seeded before timing)",
            "bound_ports" => ["observations", "observation_scales"],
            "bound_prior_state" =>
                "not_applicable: minimal prior cut has no data ports",
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

get(ENV, "RK_EIGHT_SCHOOLS_DEFINITIONS_ONLY", "") == "1" || run_comparison()
