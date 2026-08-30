module PathfinderJLKernelAuthoringFixture

using LinearAlgebra
using ReactiveKernels

export PATHFINDER_JL_CANDIDATE, PATHFINDER_JL_OUTPUTS
export pathfinder_jl_compact_candidate, pathfinder_jl_fixture_inputs
export run_pathfinder_jl_fixture

"""
A fixed-history compact L-BFGS Pathfinder candidate.

This is a compiler-oriented transcription of Pathfinder.jl 0.10.7 revision
`dba8c9acc25f2905078d428ddd50b5d9276c3847`, especially
`lbfgs_inverse_hessian`, `fit_mvnormals`, and `rand_and_logpdf`. The caller
supplies a nonempty fixed-capacity set of accepted history pairs in
chronological columns; the acceptance fixture uses two. Fixing that capacity
for each compiled executable makes the mathematical source stateless while
retaining the production compact representation:

```
B = [H₀Y  S]
R = triu(S'Y)
D = [0  -R⁻¹; -R⁻ᵀ  R⁻ᵀ(E + Y'H₀Y)R⁻¹]
H = H₀ + BDB'
```

Path management, curvature rejection, and `gilbert_init` recovery remain
ordinary Julia; the existing one-pair fixture exercises those operations.
"""
@kernel pathfinder_jl_compact_candidate(
        logdensity::Function,
        position::Vector{Float64},
        gradient::Vector{Float64},
        alpha::Vector{Float64},
        history_steps::Matrix{Float64},
        history_gradient_deltas::Matrix{Float64},
        parameter_identity::Matrix{Float64},
        history_identity::Matrix{Float64},
        elbo_standard_draws::Matrix{Float64},
        output_standard_draws::Matrix{Float64}) = begin
    diagonal_covariance::Matrix{Float64} =
        parameter_identity .* transpose(alpha)
    scaled_gradient_deltas::Matrix{Float64} =
        alpha .* history_gradient_deltas

    # Pathfinder.jl's compact inverse-Hessian representation, written without
    # mutable WoodburyPDMat caches so it remains a pure have→want graph.
    history_cross::Matrix{Float64} =
        transpose(history_steps) * history_gradient_deltas
    history_r_inverse = inv(UpperTriangular(history_cross))
    history_diagonal::Matrix{Float64} =
        history_identity .* transpose(diag(history_cross))
    compact_middle::Matrix{Float64} =
        history_diagonal .+
        transpose(history_gradient_deltas) * scaled_gradient_deltas
    scaled_cross_projection::Matrix{Float64} =
        scaled_gradient_deltas * history_r_inverse
    step_projection::Matrix{Float64} =
        history_steps * transpose(history_r_inverse)
    covariance::Matrix{Float64} =
        diagonal_covariance .-
        scaled_cross_projection * transpose(history_steps) .-
        history_steps * transpose(scaled_cross_projection) .+
        step_projection * compact_middle * transpose(step_projection)

    mean::Vector{Float64} = position .+ covariance * gradient
    covariance_factorization = cholesky(Symmetric(covariance))
    covariance_factor = UpperTriangular(covariance_factorization.factors)
    elbo_draws::Matrix{Float64} =
        mean .+ transpose(covariance_factor) * elbo_standard_draws
    output_draws::Matrix{Float64} =
        mean .+ transpose(covariance_factor) * output_standard_draws

    half_logdet_covariance::Float64 =
        sum(log, diag(covariance_factorization.factors))
    standard_squared_norms::Vector{Float64} =
        vec(sum(abs2, elbo_standard_draws; dims = 1))
    log_q::Vector{Float64} =
        -0.5 * length(position) * log(2pi) .-
        half_logdet_covariance .- 0.5 .* standard_squared_norms
    target_logdensity::Vector{Float64} = logdensity(elbo_draws)
    elbo::Float64 =
        sum(target_logdensity .- log_q) / size(elbo_standard_draws, 2)
end

const PATHFINDER_JL_OUTPUTS = (
    :history_cross,
    :compact_middle,
    :covariance,
    :mean,
    :elbo_draws,
    :log_q,
    :elbo,
    :output_draws,
)

const PATHFINDER_JL_CANDIDATE = prepare(
    pathfinder_jl_compact_candidate;
    have = (
        :logdensity,
        :position,
        :gradient,
        :alpha,
        :history_steps,
        :history_gradient_deltas,
        :parameter_identity,
        :history_identity,
        :elbo_standard_draws,
        :output_standard_draws,
    ),
    want = PATHFINDER_JL_OUTPUTS,
)

function _gilbert_recover(alpha, step, gradient_delta)
    a = dot(gradient_delta, alpha .* gradient_delta)
    b = dot(gradient_delta, step)
    c = dot(step, step ./ alpha)
    1.0 ./ (
        (a / b) .* alpha .+
        gradient_delta .^ 2 ./ b .-
        (a / (b * c)) .* step .^ 2 ./ alpha .^ 2)
end

"""Deterministic two-history input matching the pinned Pathfinder.jl source."""
function pathfinder_jl_fixture_inputs()
    precision = [1.5 0.25; 0.25 0.8]
    positions = [
        -1.0 -0.55 -0.22;
         0.8  0.40  0.12
    ]
    gradients = -precision * positions
    first_step = positions[:, 2] .- positions[:, 1]
    first_delta = gradients[:, 1] .- gradients[:, 2]
    second_step = positions[:, 3] .- positions[:, 2]
    second_delta = gradients[:, 2] .- gradients[:, 3]
    alpha_first = _gilbert_recover([0.7, 1.2], first_step, first_delta)
    alpha = _gilbert_recover(alpha_first, second_step, second_delta)
    history_steps = hcat(first_step, second_step)
    history_gradient_deltas = hcat(first_delta, second_delta)
    parameter_identity = Matrix{Float64}(I, 2, 2)
    history_identity = Matrix{Float64}(I, 2, 2)
    elbo_standard_draws = [
         0.7 -0.6 0.1  1.3;
        -0.9  0.4 1.0 -0.2
    ]
    output_standard_draws = [
        -1.1  0.55 0.25;
         0.65 -0.15 1.35
    ]
    logdensity_normalizer =
        -0.5 * size(precision, 1) * log(2pi) + 0.5 * logdet(precision)
    logdensity = draws -> vec(
        logdensity_normalizer .-
        0.5 .* sum(draws .* (precision * draws); dims = 1))
    return (;
        logdensity,
        position = positions[:, 3],
        gradient = gradients[:, 3],
        alpha,
        history_steps,
        history_gradient_deltas,
        parameter_identity,
        history_identity,
        elbo_standard_draws,
        output_standard_draws,
    )
end

function run_pathfinder_jl_fixture(kernel = PATHFINDER_JL_CANDIDATE)
    inputs = pathfinder_jl_fixture_inputs()
    values = kernel(Tuple(inputs)...)
    NamedTuple{PATHFINDER_JL_OUTPUTS}(values)
end

end # module PathfinderJLKernelAuthoringFixture
