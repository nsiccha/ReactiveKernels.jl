# Inner body for the pinned sum-to-zero hot-loop comparison. The benchmark
# deliberately measures only the packed unconstrained posterior used by a
# sampler. The requested-only common-shift recovery is a separate operation and
# never enters these timings.

using BenchmarkTools
using Dates
using Pkg
using SHA
using Statistics
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.SumToZeroExample:
    SUM_TO_ZERO_SOURCE, build_sum_to_zero_graph
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
using DifferentiationInterface
import DynamicPPL
import Enzyme
import Turing
using Turing: filldist
using Distributions: Cauchy, Normal, truncated

include(joinpath(@__DIR__, "_ad_comparison_support.jl"))
using .ADComparisonSupport:
    RKValueGradientCall, DIValueGradientCall, TuringValueGradientCall,
    build_and_first_call, record_implementation, package_version,
    source_pin

const LDP = DynamicPPL.LogDensityProblems
const DEFAULT_SUM_TO_ZERO_ROUNDS = 10
const RK_AD_BACKEND = AutoEnzyme(
    ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
const TURING_AD_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)
const COMPARISON_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "DifferentiationInterface", "Enzyme", "BenchmarkTools",
)

# DOCS-BASELINE-BEGIN: turing
function _turing_sum_to_zero!(effects_s2z, effects_free)
    running_sum = zero(eltype(effects_free))
    nfree = length(effects_free)
    @inbounds for offset in 1:nfree
        i = nfree - offset + 1
        w = effects_free[i] / sqrt(i * (i + 1))
        running_sum += w
        effects_s2z[i] = running_sum
        effects_s2z[i + 1] = running_sum - (i + 1) * w
    end
    effects_s2z
end

Turing.@model function turing_sum_to_zero(
        observations, observation_scales, α_prior_sd)
    K = length(observations)
    τ ~ truncated(Cauchy(0, 5); lower = 0)
    α_s2z ~ Normal(0, sqrt(α_prior_sd^2 + τ^2 / K))
    effects_free ~ filldist(Normal(0, τ), K - 1)

    effects_s2z = Vector{eltype(effects_free)}(undef, K)
    _turing_sum_to_zero!(effects_s2z, effects_free)

    # RK writes the subspace density as K scalar Normal terms plus log(τ).
    # The K - 1 free-coordinate Normal density above differs only by this
    # fixed base-measure constant, so retain it for exact value parity.
    Turing.@addlogprob! -0.5 * log(2π)
    for j in eachindex(observations)
        observations[j] ~ Normal(
            α_s2z + effects_s2z[j], observation_scales[j])
    end
    return nothing
end
# DOCS-BASELINE-END: turing

# DOCS-BASELINE-BEGIN: manual
_manual_normal_logpdf(x, location, scale, log_scale = log(scale)) =
    -0.5 * log(2π) - log_scale - 0.5 * ((x - location) / scale)^2

function manual_sum_to_zero_posterior(
        unconstrained, observations, observation_scales, α_prior_sd)
    α_s2z = unconstrained[1]
    log_τ = unconstrained[2]
    τ = exp(log_τ)
    effects_free = @view unconstrained[3:length(unconstrained)]
    K = length(observations)

    α_scale = sqrt(α_prior_sd^2 + τ^2 / K)
    total = _manual_normal_logpdf(α_s2z, 0.0, α_scale)
    total += log(2.0) - log(π) - log(5.0) - log1p((τ / 5.0)^2)

    # Each `i + 1` output is final when it is formed, so the optimized manual
    # lower bound can consume the transform without materializing its vector.
    running_sum = zero(eltype(unconstrained))
    @inbounds for offset in 1:length(effects_free)
        i = length(effects_free) - offset + 1
        w = effects_free[i] / sqrt(i * (i + 1))
        running_sum += w
        effect = running_sum - (i + 1) * w
        total += _manual_normal_logpdf(effect, 0.0, τ, log_τ)
        total += _manual_normal_logpdf(
            observations[i + 1], α_s2z + effect,
            observation_scales[i + 1])
    end
    total += _manual_normal_logpdf(running_sum, 0.0, τ, log_τ)
    total += _manual_normal_logpdf(
        observations[1], α_s2z + running_sum, observation_scales[1])

    # One log(τ) is the constrained-subspace normalization and the other is
    # the positive-scale transform Jacobian.
    total + 2log_τ
end
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

function _measurement(call; rounds::Int)
    benchmark = @benchmarkable $call()
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
        "min_ns" => minimum(times_ns),
        "median_ns" => median(times_ns),
        "bytes" => bytes,
        "median_bytes" => Int(median(bytes)),
        "allocs" => allocs,
        "median_allocs" => Int(median(allocs)),
    )
end

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

_rounds() = parse(Int, get(
    ENV, "RK_SUM_TO_ZERO_ROUNDS", string(DEFAULT_SUM_TO_ZERO_ROUNDS)))

function _output_path()
    for argument in ARGS
        startswith(argument, "--output=") &&
            return split(argument, '='; limit = 2)[2]
    end
    nothing
end

function _package_version(name)
    name == "ReactiveKernels" && return package_version(name)
    package_version(name)
end

_source_sha256(path) = bytes2hex(sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function _primal_row(provider, configuration, call, value; rounds)
    Dict{String,Any}(
        "provider" => provider,
        "configuration" => configuration,
        "operation" => "posterior",
        "differentiation" => "primal",
        "compiler" => "native",
        "state" => "supported",
        "value" => Float64(value),
        "result" => _measurement(call; rounds),
    )
end

function _ad_row(provider, configuration, call, result, setup,
                 reference_value, reference_gradient; rounds,
                 caller_owned_gradient)
    Dict{String,Any}(
        "provider" => provider,
        "configuration" => configuration,
        "operation" => "posterior value and gradient",
        "differentiation" => "value_and_gradient",
        "compiler" => "native",
        "state" => "supported",
        "result" => record_implementation(
            call, result, setup, reference_value, reference_gradient;
            rounds, caller_owned = caller_owned_gradient),
    )
end

function run_sum_to_zero_comparison()
    rounds = _rounds()
    rounds >= 1 || error("round count must be positive")
    root = normpath(joinpath(@__DIR__, ".."))
    observations = Float64.(EIGHT_SCHOOLS_Y)
    observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
    α_prior_sd = 5.0
    α_s2z = 0.5
    log_τ = log(2.0)
    effects_free = 0.25 .* collect(1.0:7.0)
    unconstrained = [α_s2z, log_τ, effects_free...]

    model_setup = @timed build_sum_to_zero_graph()
    model = model_setup.value
    rk_kernel_setup = @timed prepare(
        model;
        have = (
            :unconstrained, :observations, :observation_scales, :α_prior_sd),
        want = :posterior,
        bound = (; observations, observation_scales, α_prior_sd),
    )
    rk_kernel = rk_kernel_setup.value

    manual_call = let q = unconstrained, observations = observations,
                      observation_scales = observation_scales,
                      α_prior_sd = α_prior_sd
        () -> manual_sum_to_zero_posterior(
            q, observations, observation_scales, α_prior_sd)
    end
    rk_call = let kernel = rk_kernel, q = unconstrained
        () -> kernel(q)
    end
    reference_value = manual_call()
    rk_value = rk_call()
    isapprox(rk_value, reference_value; rtol = 1e-12, atol = 1e-12) ||
        error("RK/manual posterior parity failed")

    turing_setup = @timed turing_sum_to_zero(
        observations, observation_scales, α_prior_sd)
    turing_model = turing_setup.value
    turing_joint = DynamicPPL.LogDensityFunction(
        turing_model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
        fix_transforms = true,
    )
    τ = exp(log_τ)
    turing_parameters = (; τ, α_s2z, effects_free)
    turing_unconstrained = _turing_vector(turing_joint, turing_parameters)
    expected_turing_unconstrained = [log_τ, α_s2z, effects_free...]
    isapprox(turing_unconstrained, expected_turing_unconstrained;
             rtol = 1e-12, atol = 1e-12) ||
        error("Turing linked parameter order differs from its declared contract")
    turing_call = let density = turing_joint, q = turing_unconstrained
        () -> LDP.logdensity(density, q)
    end
    turing_value = turing_call()
    isapprox(turing_value, reference_value; rtol = 1e-11, atol = 1e-12) ||
        error("Turing/manual posterior parity failed: $turing_value != $reference_value")

    measurements = Dict{String,Any}[
        _primal_row("rk", "rk_primal_native", rk_call, rk_value; rounds),
        _primal_row("manual_julia", "manual_primal", manual_call,
                    reference_value; rounds),
        _primal_row("turing", "turing_primal", turing_call,
                    turing_value; rounds),
    ]

    constants = (
        Constant(observations), Constant(observation_scales),
        Constant(α_prior_sd),
    )
    manual_ad_call, manual_ad_result, manual_ad_setup =
        build_and_first_call() do
            preparation = prepare_gradient(
                manual_sum_to_zero_posterior, RK_AD_BACKEND,
                unconstrained, constants...)
            DIValueGradientCall(
                manual_sum_to_zero_posterior, similar(unconstrained),
                preparation, RK_AD_BACKEND, unconstrained, constants)
        end
    reference_value_ad, manual_gradient = manual_ad_result
    isapprox(reference_value_ad, reference_value;
             rtol = 1e-12, atol = 1e-12) ||
        error("manual primal/AD value parity failed")
    finite_difference_gradient = _finite_difference_gradient(
        q -> manual_sum_to_zero_posterior(
            q, observations, observation_scales, α_prior_sd),
        unconstrained,
    )
    maximum(abs.(manual_gradient .- finite_difference_gradient)) <= 1e-8 ||
        error("manual Enzyme gradient failed finite-difference parity")
    push!(measurements, _ad_row(
        "manual_julia", "manual_ad", manual_ad_call, manual_ad_result,
        manual_ad_setup, reference_value, finite_difference_gradient;
        rounds, caller_owned_gradient = true))

    rk_ad_call, rk_ad_result, rk_ad_setup = build_and_first_call() do
        prepared = prepare_ad(
            rk_kernel, RK_AD_BACKEND, unconstrained; active = :unconstrained)
        RKValueGradientCall(prepared, similar(unconstrained), (unconstrained,))
    end
    push!(measurements, _ad_row(
        "rk", "rk_ad_native", rk_ad_call, rk_ad_result, rk_ad_setup,
        reference_value, finite_difference_gradient;
        rounds, caller_owned_gradient = true))

    turing_ad_call, turing_ad_result, turing_ad_setup =
        build_and_first_call() do
            density = DynamicPPL.LogDensityFunction(
                turing_model, DynamicPPL.getlogjoint_internal,
                DynamicPPL.LinkAll();
                fix_transforms = true, adtype = TURING_AD_BACKEND)
            TuringValueGradientCall(density, turing_unconstrained)
        end
    # Turing's model order is (log(τ), α_s2z, effects_free), while the RK and
    # manual order is (α_s2z, log(τ), effects_free). Compare in Turing order;
    # the timed operation itself performs no benchmark-only permutation.
    turing_reference_gradient = [
        finite_difference_gradient[2], finite_difference_gradient[1],
        finite_difference_gradient[3:end]...,
    ]
    push!(measurements, _ad_row(
        "turing", "turing_ad", turing_ad_call, turing_ad_result,
        turing_ad_setup, reference_value, turing_reference_gradient;
        rounds, caller_owned_gradient = false))

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "sum_to_zero.jl")
    comparator_path = joinpath("benchmark", "sum_to_zero_comparison_body.jl")
    receipt = Dict{String,Any}(
        "schema" => "sum-to-zero-hotloop-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            "model_source" => source_pin(root, source_path),
            "comparator_source" => source_pin(root, comparator_path),
            "model_source_text_sha256" => bytes2hex(sha256(SUM_TO_ZERO_SOURCE)),
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
        "setup" => Dict(
            "model_seconds" => model_setup.time,
            "rk_kernel_seconds" => rk_kernel_setup.time,
            "turing_model_seconds" => turing_setup.time,
        ),
        "protocol" => Dict(
            "model" => "sum-to-zero hierarchical Eight Schools",
            "boundary" => "packed unconstrained parameters",
            "outcome" => "full unconstrained posterior",
            "hot_loop_excludes_recovery" => true,
            "finite_difference_gradient" => finite_difference_gradient,
            "bound_ports" => [
                "observations", "observation_scales", "α_prior_sd"],
            "manual_control" =>
                "allocation-free fused O(K) transform, prior, and likelihood",
            "turing_interface" =>
                "DynamicPPL.LogDensityFunction with fixed transforms",
            "rk_native_ad_surface" =>
                "prepare_ad + ad_value_and_gradient!",
            "manual_ad_surface" =>
                "DifferentiationInterface + Enzyme reverse mode",
            "turing_ad_surface" =>
                "LogDensityProblems.logdensity_and_gradient + Enzyme reverse mode",
            "rounds" => rounds,
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "estimator" =>
                "median of per-round BenchmarkTools minimum times",
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "first_execution_in_steady_state_region" => false,
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

get(ENV, "RK_SUM_TO_ZERO_DEFINITIONS_ONLY", "") == "1" ||
    run_sum_to_zero_comparison()
