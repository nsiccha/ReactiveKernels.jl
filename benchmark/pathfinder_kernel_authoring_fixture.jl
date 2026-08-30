module PathfinderKernelAuthoringFixture

using LinearAlgebra
using ReactiveKernels

export PATHFINDER_CANDIDATE, PATHFINDER_OUTPUTS, pathfinder_candidate
export pathfinder_fixture_inputs, run_pathfinder_fixture, run_single_path

"""
One deterministic Pathfinder local-approximation candidate.

The optimizer path and standard-normal draws are explicit HAVE ports.  The
kernel applies the curvature safeguard and diagonal recovery from Algorithm 3
of Zhang et al. (2022), then the inverse-BFGS update underlying Algorithm 4.
This fixture intentionally uses one retained `(step, gradient_delta)` pair:
limited-memory size one is the smallest complete instance of the published
quasi-Newton construction.

`logdensity` accepts a `dimension × draws` matrix and returns one target log
density per column.  Supplying both ELBO and output noise makes the kernel
deterministic and backend-neutral; RNG ownership stays with the caller.
"""
@kernel pathfinder_candidate(
        logdensity::Function,
        position::Vector{Float64},
        gradient::Vector{Float64},
        alpha::Vector{Float64},
        step::Vector{Float64},
        gradient_delta::Vector{Float64},
        identity::Matrix{Float64},
        elbo_standard_draws::Matrix{Float64},
        output_standard_draws::Matrix{Float64},
        curvature_tolerance::Float64) = begin
    curvature::Float64 = dot(step, gradient_delta)
    curvature_threshold::Float64 =
        curvature_tolerance * dot(gradient_delta, gradient_delta)
    curvature_accepted::Bool = curvature > curvature_threshold

    # Algorithm 3 (alpha-RECOVER). Safe denominators keep the rejected branch
    # finite under eager/select-style accelerator lowering.
    safe_curvature::Float64 = curvature_accepted ? curvature : 1.0
    alpha_a::Float64 = dot(gradient_delta, alpha .* gradient_delta)
    alpha_c::Float64 = dot(step, step ./ alpha)
    safe_alpha_c::Float64 = curvature_accepted ? alpha_c : 1.0
    recovered_alpha::Vector{Float64} = 1.0 ./ (
        (alpha_a / safe_curvature) .* alpha .+
        (gradient_delta .^ 2) ./ safe_curvature .-
        (alpha_a / (safe_curvature * safe_alpha_c)) .*
            (step .^ 2) ./ (alpha .^ 2))
    alpha_next::Vector{Float64} =
        curvature_accepted ? recovered_alpha : alpha

    # Inverse-BFGS covariance for one retained history pair. With a rejected
    # pair, rho=0 and this reduces exactly to diag(alpha).
    rho::Float64 = curvature_accepted ? inv(safe_curvature) : 0.0
    diagonal_covariance::Matrix{Float64} =
        identity .* transpose(alpha_next)
    bfgs_transform::Matrix{Float64} =
        identity .- rho .* (step * transpose(gradient_delta))
    covariance::Matrix{Float64} =
        bfgs_transform * diagonal_covariance * transpose(bfgs_transform) .+
        rho .* (step * transpose(step))

    # Algorithm 4 local Gaussian: mean = theta + Sigma * grad(log p).
    mean::Vector{Float64} = position .+ covariance * gradient
    covariance_factorization = cholesky(Symmetric(covariance))
    # Use the factorization's generic packed-factor surface. Native Cholesky
    # and Reactant's BatchedCholesky both store the authoritative upper factor
    # there, whereas only the native object provides the convenience `.L`.
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

const PATHFINDER_OUTPUTS = (
    :alpha_next,
    :curvature_accepted,
    :covariance,
    :mean,
    :elbo_draws,
    :log_q,
    :elbo,
    :output_draws,
)

const PATHFINDER_CANDIDATE = prepare(
    pathfinder_candidate;
    have = (
        :logdensity,
        :position,
        :gradient,
        :alpha,
        :step,
        :gradient_delta,
        :identity,
        :elbo_standard_draws,
        :output_standard_draws,
        :curvature_tolerance,
    ),
    want = PATHFINDER_OUTPUTS,
)

"""Run the same prepared candidate kernel along one supplied optimizer path."""
function run_single_path(
        kernel,
        logdensity,
        positions,
        gradients,
        initial_alpha,
        identity,
        elbo_standard_draws,
        output_standard_draws;
        curvature_tolerance = 1e-12)
    size(positions) == size(gradients) ||
        throw(DimensionMismatch("positions and gradients must have equal shape"))
    dimension, path_points = size(positions)
    path_points >= 2 || throw(ArgumentError("Pathfinder needs at least one path update"))
    size(identity) == (dimension, dimension) ||
        throw(DimensionMismatch("identity must be dimension × dimension"))
    candidates = path_points - 1
    size(elbo_standard_draws, 3) == candidates ||
        throw(DimensionMismatch("one ELBO-noise slab is required per candidate"))
    size(output_standard_draws, 3) == candidates ||
        throw(DimensionMismatch("one output-noise slab is required per candidate"))

    alpha = copy(initial_alpha)
    results = NamedTuple[]
    for candidate_index in 1:candidates
        path_index = candidate_index + 1
        step = positions[:, path_index] .- positions[:, path_index - 1]
        gradient_delta =
            gradients[:, path_index - 1] .- gradients[:, path_index]
        values = kernel(
            logdensity,
            positions[:, path_index],
            gradients[:, path_index],
            alpha,
            step,
            gradient_delta,
            identity,
            elbo_standard_draws[:, :, candidate_index],
            output_standard_draws[:, :, candidate_index],
            curvature_tolerance,
        )
        result = NamedTuple{PATHFINDER_OUTPUTS}(values)
        push!(results, result)
        alpha = result.alpha_next
    end

    elbos = map(result -> result.elbo, results)
    best_index = argmax(elbos)
    best = results[best_index]
    return (;
        candidates = Tuple(results),
        elbos,
        best_index,
        mean = best.mean,
        covariance = best.covariance,
        draws = best.output_draws,
    )
end

"""Deterministic correlated-Gaussian fixture shared by native and Reactant gates."""
function pathfinder_fixture_inputs()
    precision = [1.5 0.25; 0.25 0.8]
    positions = [
        -1.0 -0.55 -0.22 -0.05;
         0.8  0.40  0.12  0.02
    ]
    gradients = -precision * positions
    initial_alpha = [0.7, 1.2]
    identity = Matrix{Float64}(I, 2, 2)
    elbo_standard_draws = cat(
        [-1.0 0.2 1.1 -0.4; 0.5 -1.2 0.3 0.8],
        [0.7 -0.6 0.1 1.3; -0.9 0.4 1.0 -0.2],
        [-0.3 0.9 -1.1 0.6; 1.2 -0.5 0.2 -1.0];
        dims = 3,
    )
    output_standard_draws = cat(
        [0.15 -0.75 1.25; -0.35 0.95 0.45],
        [-1.1 0.55 0.25; 0.65 -0.15 1.35],
        [0.8 -0.2 -0.95; -0.6 1.1 0.05];
        dims = 3,
    )
    logdensity_normalizer =
        -0.5 * size(precision, 1) * log(2pi) + 0.5 * logdet(precision)
    logdensity = draws -> vec(
        logdensity_normalizer .-
        0.5 .* sum(draws .* (precision * draws); dims = 1))
    return (;
        precision,
        positions,
        gradients,
        initial_alpha,
        identity,
        elbo_standard_draws,
        output_standard_draws,
        curvature_tolerance = 1e-12,
        logdensity,
    )
end

function run_pathfinder_fixture(kernel = PATHFINDER_CANDIDATE)
    inputs = pathfinder_fixture_inputs()
    run_single_path(
        kernel,
        inputs.logdensity,
        inputs.positions,
        inputs.gradients,
        inputs.initial_alpha,
        inputs.identity,
        inputs.elbo_standard_draws,
        inputs.output_standard_draws;
        curvature_tolerance = inputs.curvature_tolerance,
    )
end

end # module PathfinderKernelAuthoringFixture
