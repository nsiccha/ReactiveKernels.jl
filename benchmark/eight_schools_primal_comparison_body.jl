# Inner body for the pinned Eight Schools primal comparison.

using BenchmarkTools
using Dates
using LinearAlgebra
using Pkg
using Statistics
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, build_eight_schools_graph
import DynamicPPL
import Turing
using Distributions: Cauchy, MvNormal, Normal, truncated

const LDP = DynamicPPL.LogDensityProblems
const DEFAULT_EIGHT_SCHOOLS_ROUNDS = 10
const COMPARISON_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "BenchmarkTools",
)

Turing.@model function turing_eight_schools(observations, observation_scales)
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5); lower = 0)
    θ ~ MvNormal(fill(μ, length(observations)), τ^2 * I)
    for j in eachindex(observations)
        observations[j] ~ Normal(θ[j], observation_scales[j])
    end
    return nothing
end

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

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

_rounds() = parse(Int, get(
    ENV, "RK_EIGHT_SCHOOLS_ROUNDS", string(DEFAULT_EIGHT_SCHOOLS_ROUNDS)))

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

    # This clones the graph evaluated from the exact public Eight Schools
    # source at module load, so all timed kernels share that source authority
    # without crossing a fresh Core.eval world-age boundary here.
    rk_model = build_eight_schools_graph()
    rk_unconstrained_joint = prepare(rk_model;
        have = (:unconstrained, :observations, :observation_scales),
        want = :posterior)
    rk_unconstrained_prior = prepare(rk_model;
        have = :unconstrained, want = :unconstrained_prior)
    rk_unconstrained_likelihood = prepare(rk_model;
        have = (:unconstrained, :observations, :observation_scales),
        want = :likelihood)
    rk_unconstrained_pointwise = prepare(rk_model;
        have = (:unconstrained, :observations, :observation_scales),
        want = :pointwise)

    rk_constrained_joint = prepare(rk_model;
        have = (:parameters, :observations, :observation_scales),
        want = :constrained_logdensity)
    rk_constrained_prior = prepare(rk_model;
        have = :parameters, want = :prior)
    rk_constrained_likelihood = prepare(rk_model;
        have = (:parameters, :observations, :observation_scales),
        want = :likelihood)
    rk_constrained_pointwise = prepare(rk_model;
        have = (:parameters, :observations, :observation_scales),
        want = :pointwise)

    rk_minimal_likelihood = prepare(rk_model;
        have = (:θ, :observations, :observation_scales),
        want = :likelihood)
    rk_minimal_pointwise = prepare(rk_model;
        have = (:θ, :observations, :observation_scales),
        want = :pointwise)

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

    definitions = (
        (
            boundary = "packed_unconstrained", outcome = "joint",
            description = "full joint including the transform Jacobian",
            rk = (rk_unconstrained_joint,
                  (unconstrained, observations, observation_scales)),
            manual = (_manual_unconstrained_joint,
                      (unconstrained, observations, observation_scales)),
            turing = (_turing_density,
                      (turing_unconstrained_joint,
                       turing_unconstrained_parameters)),
        ),
        (
            boundary = "packed_unconstrained", outcome = "prior",
            description = "log prior including the transform Jacobian",
            rk = (rk_unconstrained_prior, (unconstrained,)),
            manual = (_manual_unconstrained_prior, (unconstrained,)),
            turing = (_turing_density,
                      (turing_unconstrained_prior,
                       turing_unconstrained_parameters)),
        ),
        (
            boundary = "packed_unconstrained", outcome = "likelihood",
            description = "summed log likelihood after unpacking the latent vector",
            rk = (rk_unconstrained_likelihood,
                  (unconstrained, observations, observation_scales)),
            manual = (_manual_unconstrained_likelihood,
                      (unconstrained, observations, observation_scales)),
            turing = (_turing_density,
                      (turing_unconstrained_likelihood,
                       turing_unconstrained_parameters)),
        ),
        (
            boundary = "packed_unconstrained", outcome = "pointwise",
            description = "eight pointwise log likelihoods after unpacking",
            rk = (rk_unconstrained_pointwise,
                  (unconstrained, observations, observation_scales)),
            manual = (_manual_unconstrained_pointwise,
                      (unconstrained, observations, observation_scales)),
            turing = (_turing_pointwise,
                      (turing_model, turing_unconstrained_pointwise_init)),
        ),
        (
            boundary = "constrained_parameters", outcome = "joint",
            description = "full joint excluding the transform Jacobian",
            rk = (rk_constrained_joint,
                  (parameters, observations, observation_scales)),
            manual = (_manual_constrained_joint,
                      (parameters, observations, observation_scales)),
            turing = (_turing_constrained_joint, (turing_model, parameters)),
        ),
        (
            boundary = "constrained_parameters", outcome = "prior",
            description = "constrained-space log prior",
            rk = (rk_constrained_prior, (parameters,)),
            manual = (_manual_constrained_prior, (parameters,)),
            turing = (_turing_constrained_prior, (turing_model, parameters)),
        ),
        (
            boundary = "constrained_parameters", outcome = "likelihood",
            description = "summed log likelihood",
            rk = (rk_constrained_likelihood,
                  (parameters, observations, observation_scales)),
            manual = (_manual_constrained_likelihood,
                      (parameters, observations, observation_scales)),
            turing = (_turing_constrained_likelihood,
                      (turing_model, parameters)),
        ),
        (
            boundary = "constrained_parameters", outcome = "pointwise",
            description = "eight pointwise log likelihoods",
            rk = (rk_constrained_pointwise,
                  (parameters, observations, observation_scales)),
            manual = (_manual_constrained_pointwise,
                      (parameters, observations, observation_scales)),
            turing = (_turing_pointwise,
                      (turing_model, turing_constrained_pointwise_init)),
        ),
        (
            boundary = "minimal_likelihood", outcome = "joint",
            description = "unsupported without prior parameters",
            rk = nothing, manual = nothing, turing = nothing,
        ),
        (
            boundary = "minimal_likelihood", outcome = "prior",
            description = "unsupported without prior parameters",
            rk = nothing, manual = nothing, turing = nothing,
        ),
        (
            boundary = "minimal_likelihood", outcome = "likelihood",
            description = "summed likelihood from θ, observations, and scales only",
            rk = (rk_minimal_likelihood,
                  (θ, observations, observation_scales)),
            manual = (_manual_likelihood,
                      (θ, observations, observation_scales)),
            turing = nothing,
        ),
        (
            boundary = "minimal_likelihood", outcome = "pointwise",
            description = "pointwise likelihoods from θ, observations, and scales only",
            rk = (rk_minimal_pointwise,
                  (θ, observations, observation_scales)),
            manual = (_manual_pointwise,
                      (θ, observations, observation_scales)),
            turing = nothing,
        ),
    )

    measurements = Dict{String,Any}[]
    for definition in definitions
        row = Dict{String,Any}(
            "boundary" => definition.boundary,
            "outcome" => definition.outcome,
            "description" => definition.description,
        )
        reference = definition.manual === nothing ? nothing :
            _comparable(definition.outcome, _evaluate(definition.manual))
        for (key, spec) in (
            "rk_native" => definition.rk,
            "manual_julia" => definition.manual,
            "turing_native" => definition.turing,
        )
            spec === nothing && continue
            actual = _comparable(definition.outcome, _evaluate(spec))
            isapprox(actual, reference; rtol = 1e-11, atol = 1e-12) || error(
                "parity failed for $(definition.boundary) / " *
                "$(definition.outcome) / $key",
            )
            function_, arguments = spec
            row[key] = _measurement(function_, arguments...; rounds)
        end
        push!(measurements, row)
        println("boundary=$(definition.boundary) outcome=$(definition.outcome) complete")
    end

    receipt = Dict{String,Any}(
        "schema" => "eight-schools-primal-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            (string(lowercase(name), "_version") => _package_version(name)
             for name in COMPARISON_PACKAGES)...,
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
            "input_boundaries" => [
                "packed_unconstrained", "constrained_parameters",
                "minimal_likelihood",
            ],
            "outcomes" => ["joint", "prior", "likelihood", "pointwise"],
            "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "turing_transform_strategy" => "fixed transforms",
            "turing_pointwise_api" => "DynamicPPL.pointwise_loglikelihoods",
            "unsupported_cells_omitted" => true,
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

run_comparison()
