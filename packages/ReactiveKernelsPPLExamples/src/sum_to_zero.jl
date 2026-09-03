module SumToZeroExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source
using ..EightSchoolsExample: EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA

export SUM_TO_ZERO_SOURCE, build_sum_to_zero_graph
export evaluate_sum_to_zero_source, demo

const SUM_TO_ZERO_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@kernel model(unconstrained::Vector{Float64},
              observations::Vector{Float64},
              observation_scales::Vector{Float64},
              α_prior_sd::Float64,
              reconstruction_innovation::Float64) = begin
    # q = (α_s2z, log(τ), K - 1 free effect coordinates). Match the packed
    # scalar extraction used by Eight Schools so this boundary is traceable as
    # a tensor program without changing its ordinary Julia meaning.
    α_s2z::Float64 = sum(view(unconstrained, 1:1))
    log_τ::Float64 = sum(view(unconstrained, 2:2))
    effects_free::AbstractVector{Float64} =
        view(unconstrained, 3:length(unconstrained))

    # Positive support for the population effects scale. Either τ or log_τ may
    # be an authoritative HAVE value, as in the Eight Schools graph.
    log_τ::Float64 = log(τ)
    τ::Float64 = exp(log_τ)

    # Stan's O(K) pivot-coordinate transform, inlined because it is new to this
    # example. It maps K - 1 free coordinates to K effects with zero sum.
    effects_s2z::AbstractVector{Float64} = let
        nfree = length(effects_free)
        constrained = Vector{Float64}(undef, nfree + 1)
        running_sum = 0.0
        for offset in 1:nfree
            i = nfree - offset + 1
            w = effects_free[i] / sqrt(i * (i + 1))
            running_sum += w
            constrained[i] = running_sum
            constrained[i + 1] = running_sum - (i + 1) * w
        end
        constrained
    end

    # Orthonormality makes the intrinsic volume adjustment exactly zero. Keep
    # this distinct from the lower-dimensional Normal normalization below.
    sum_to_zero_log_jacobian::Float64 = 0.0
    log_jacobian::Float64 = log_τ + sum_to_zero_log_jacobian

    # Parameters are also an authoritative constrained HAVE boundary. The
    # joint producer shares the constrained values when both they and the total
    # sampler-space Jacobian are requested.
    parameters = (; α_s2z, τ, effects_s2z)
    (parameters, log_jacobian::Float64) =
        ((; α_s2z, τ, effects_s2z),
         log_τ + sum_to_zero_log_jacobian)
    (α_s2z::Float64,
     τ::Float64,
     effects_s2z::AbstractVector{Float64}) =
        (parameters.α_s2z, parameters.τ, parameters.effects_s2z)

    K::Int = length(effects_s2z)

    # α_s2z = α_bayes + mean(a_bayes), with
    # Var(mean(a_bayes) | τ) = τ² / K.
    α_s2z_prior_scale::Float64 = sqrt(α_prior_sd^2 + τ^2 / K)
    α_s2z_prior::Float64 =
        normal(0.0, α_s2z_prior_scale).logpdf(α_s2z)

    # τ ~ HalfCauchy(0, 5). Keep the reusable endpoint call on its own
    # assignment before applying the truncation constant.
    τ_cauchy::Float64 = cauchy(0.0, 5.0).logpdf(τ)
    τ_prior::Float64 = log(2.0) + τ_cauchy

    effects_pointwise = plate(effects_s2z, τ, log_τ) do effect, effect_τ, effect_log_τ
        normal(;
            location = 0.0,
            scale = effect_τ,
            log_scale = effect_log_τ,
        ).logpdf(effect)
    end
    effects_prior_sum::Float64 = sum(effects_pointwise)

    # The vectorized density supplies K scale normalizers on a K - 1
    # dimensional subspace. This +log(τ) repairs that prior normalization; it
    # is not a transform Jacobian.
    subspace_normalization::Float64 = log_τ
    effects_prior::Float64 = effects_prior_sum + subspace_normalization
    prior::Float64 = α_s2z_prior + τ_prior + effects_prior

    pointwise = plate(observations, effects_s2z, observation_scales, α_s2z) do observed, effect, observation_scale, intercept
        normal(intercept + effect, observation_scale).logpdf(observed)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Stochastically restore the common-shift direction removed by the
    # constraint. The caller supplies a standard-Normal innovation so the graph
    # stays pure and the reconstruction is replayable.
    mean_effect_variance::Float64 = τ^2 / K
    intercept_variance::Float64 = α_prior_sd^2
    reconstruction_weight::Float64 =
        mean_effect_variance / (intercept_variance + mean_effect_variance)
    reconstruction_sd::Float64 = sqrt(
        intercept_variance * mean_effect_variance /
        (intercept_variance + mean_effect_variance),
    )
    mean_effect_bayes::Float64 =
        reconstruction_weight * α_s2z +
        reconstruction_sd * reconstruction_innovation
    α_bayes::Float64 = α_s2z - mean_effect_bayes
    effects_bayes::AbstractVector{Float64} =
        effects_s2z .+ mean_effect_bayes
    superpopulation = (;
        α_bayes,
        effects_bayes,
        realized_effect_mean = mean_effect_bayes,
    )

    return posterior
end

unconstrained = [0.5, log(2.0), 0.25 .* collect(1:7)...]
observations = EIGHT_SCHOOLS_Y
observation_scales = EIGHT_SCHOOLS_SIGMA
α_prior_sd = 5.0
reconstruction_innovation = -0.25

requested_nodes = (:parameters, :prior, :likelihood, :superpopulation)
evaluation_kernel = prepare(model;
    have = (:unconstrained, :observations, :observation_scales,
            :α_prior_sd, :reconstruction_innovation),
    want = requested_nodes)
inputs = (;
    unconstrained,
    observations,
    observation_scales,
    α_prior_sd,
    reconstruction_innovation,
)
output = evaluation_kernel(Tuple(inputs)...)

docs_example = (;
    name = :sum_to_zero_recovery,
    origin = "Inline sum-to-zero model and super-population recovery",
    inputs,
    model,
    kernel = evaluation_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_sum_to_zero_source()
    _evaluate_ppl_source(SUM_TO_ZERO_SOURCE, @__MODULE__; bindings = (
        :EIGHT_SCHOOLS_Y, :EIGHT_SCHOOLS_SIGMA,
    ))
end

const _SUM_TO_ZERO_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SUM_TO_ZERO_GRAPH_TEMPLATE[] = evaluate_sum_to_zero_source().model
    nothing
end

"""
    build_sum_to_zero_graph()

Build the Eight-Schools-shaped sum-to-zero model and stochastic
super-population reconstruction as one declarative `KernelSpec`. The Stan
pivot transform and recovery equations are authored inline; Normal and Cauchy
are reused from `ReactiveKernelsDistributionKernels`.
"""
function build_sum_to_zero_graph()
    compose(_SUM_TO_ZERO_GRAPH_TEMPLATE[])
end

function demo()
    artifact = evaluate_sum_to_zero_source()
    parameters, prior, likelihood, superpopulation = artifact.output
    println("sum-to-zero parameters = ", parameters)
    println("log prior = ", prior, ", log likelihood = ", likelihood)
    println("recovered super-population draw = ", superpopulation)
    nothing
end

end # module SumToZeroExample

if abspath(PROGRAM_FILE) == @__FILE__
    SumToZeroExample.demo()
end
