module NutpieDiagonalAdaptationExample

using ReactiveKernels

export NUTS_RS_REVISION, NUTS_RS_SOURCE_DIGESTS
export INITIALIZE_INPUTS, INITIALIZE_OUTPUTS, ADAPTATION_INPUTS, ADAPTATION_OUTPUTS
export ORACLE_INPUTS
export nutpie_diagonal_initialize, nutpie_diagonal_adaptation
export initialize_kernel, adaptation_kernel, initial_state, advance, run

# This is an external compiler example, not a nutpie sampler API.  The mathematical
# source below follows the default draw+gradient diagonal adaptation in nuts-rs at
# one pinned revision.  The independent receipt in `benchmark/nutpie_diag_oracle.toml`
# is produced by the standalone Rust program beside it, never by this Julia path.
const NUTS_RS_REVISION = "97be9ab88cfaadfafd9e5f4409a3b1d5af62805a"
const NUTS_RS_SOURCE_DIGESTS = (
    "src/transform/adapt/diagonal.rs" =>
        "7b6c0ed9c914eb26e492c77e38c9536d73b211cf90ca05f315e487d081c72960",
    "src/transform/diagonal.rs" =>
        "3a8eb25fa91da6343a4fe36800b1b4e11abd8a80393dba19e6954104252946b7",
    "src/math/cpu_math.rs" =>
        "84e4ae66e629fcbbb9259da115ca7d3ec265dc16a50e9ba611942e4ce6aafeb7",
    "src/adapt_strategy.rs" =>
        "93c0ddd5ec7ddfa653e1e7382887944d728e12e861af85efd84b3cc4e6f0b837",
)

const LOWER_LIMIT = 1e-20
const UPPER_LIMIT = 1e20

# nuts-rs `Strategy::init` seeds both estimators from the raw point and initializes
# the transformation from the raw gradient.  This kernel is the mathematical
# transformation part of that operation.  There is no randomness in adaptation.
@kernel nutpie_diagonal_initialize(
        position::Vector{Float64}, gradient::Vector{Float64}) = begin
    gradient_scale::Vector{Float64} =
        inv.(clamp.(abs.(gradient), LOWER_LIMIT, UPPER_LIMIT))
    finite_gradient_scale::Vector{Bool} =
        (gradient_scale .- gradient_scale) .== zero.(gradient_scale)
    transform_variance::Vector{Float64} =
        ifelse.(finite_gradient_scale, gradient_scale, one.(gradient_scale))
    stds::Vector{Float64} = sqrt.(transform_variance)
    inv_stds::Vector{Float64} = sqrt.(inv.(transform_variance))
    transformation_mean::Vector{Float64} =
        position .+ stds .* stds .* gradient
    logdet::Float64 = sum(log.(inv_stds))
    return stds, inv_stds, transformation_mean, logdet
end

# One functional adaptation step.  All mutable nuts-rs ownership is represented
# explicitly as HAVE/WANT values, leaving RK to compile only transparent array
# mathematics.  The ordering is the upstream `GlobalStrategy::adapt` order:
#
#   1. update current draw, current gradient, background draw, background gradient;
#   2. optionally move the just-updated background estimators into current ownership;
#   3. reset background ownership after the move;
#   4. adapt the transform from current when it owns at least three samples.
#
# `variance` is nuts-rs's sum of squared OLD-mean differences, intentionally not
# Welford M2.  The common `(count - 1)^-1` scale cancels in draw_var/grad_var.
@kernel nutpie_diagonal_adaptation(
        draw_mean::Vector{Float64}, draw_variance::Vector{Float64},
        grad_mean::Vector{Float64}, grad_variance::Vector{Float64},
        count::Float64,
        background_draw_mean::Vector{Float64},
        background_draw_variance::Vector{Float64},
        background_grad_mean::Vector{Float64},
        background_grad_variance::Vector{Float64},
        background_count::Float64,
        stds::Vector{Float64}, inv_stds::Vector{Float64},
        transformation_mean::Vector{Float64}, logdet::Float64,
        transformation_id::Float64,
        position::Vector{Float64}, gradient::Vector{Float64},
        is_good::Bool, switch_now::Bool, adapt_now::Bool) = begin
    # Current estimator: draw then gradient, exactly as `update_estimators`.
    sampled_count::Float64 = count + 1.0
    draw_diff::Vector{Float64} = position .- draw_mean
    sampled_draw_mean::Vector{Float64} = ifelse.(
        sampled_count == 1.0,
        position,
        draw_mean .+ draw_diff ./ sampled_count,
    )
    sampled_draw_variance::Vector{Float64} = ifelse.(
        sampled_count == 1.0,
        zero.(draw_variance),
        draw_variance .+ draw_diff .* draw_diff,
    )
    grad_diff::Vector{Float64} = gradient .- grad_mean
    sampled_grad_mean::Vector{Float64} = ifelse.(
        sampled_count == 1.0,
        gradient,
        grad_mean .+ grad_diff ./ sampled_count,
    )
    sampled_grad_variance::Vector{Float64} = ifelse.(
        sampled_count == 1.0,
        zero.(grad_variance),
        grad_variance .+ grad_diff .* grad_diff,
    )
    updated_count::Float64 = ifelse.(is_good, sampled_count, count)
    updated_draw_mean::Vector{Float64} =
        ifelse.(is_good, sampled_draw_mean, draw_mean)
    updated_draw_variance::Vector{Float64} =
        ifelse.(is_good, sampled_draw_variance, draw_variance)
    updated_grad_mean::Vector{Float64} =
        ifelse.(is_good, sampled_grad_mean, grad_mean)
    updated_grad_variance::Vector{Float64} =
        ifelse.(is_good, sampled_grad_variance, grad_variance)

    # Background estimator: draw then gradient.  It is updated before a switch.
    sampled_background_count::Float64 = background_count + 1.0
    background_draw_diff::Vector{Float64} = position .- background_draw_mean
    sampled_background_draw_mean::Vector{Float64} = ifelse.(
        sampled_background_count == 1.0,
        position,
        background_draw_mean .+ background_draw_diff ./ sampled_background_count,
    )
    sampled_background_draw_variance::Vector{Float64} = ifelse.(
        sampled_background_count == 1.0,
        zero.(background_draw_variance),
        background_draw_variance .+ background_draw_diff .* background_draw_diff,
    )
    background_grad_diff::Vector{Float64} = gradient .- background_grad_mean
    sampled_background_grad_mean::Vector{Float64} = ifelse.(
        sampled_background_count == 1.0,
        gradient,
        background_grad_mean .+ background_grad_diff ./ sampled_background_count,
    )
    sampled_background_grad_variance::Vector{Float64} = ifelse.(
        sampled_background_count == 1.0,
        zero.(background_grad_variance),
        background_grad_variance .+ background_grad_diff .* background_grad_diff,
    )
    updated_background_count::Float64 =
        ifelse.(is_good, sampled_background_count, background_count)
    updated_background_draw_mean::Vector{Float64} =
        ifelse.(is_good, sampled_background_draw_mean, background_draw_mean)
    updated_background_draw_variance::Vector{Float64} = ifelse.(
        is_good, sampled_background_draw_variance, background_draw_variance)
    updated_background_grad_mean::Vector{Float64} =
        ifelse.(is_good, sampled_background_grad_mean, background_grad_mean)
    updated_background_grad_variance::Vector{Float64} = ifelse.(
        is_good, sampled_background_grad_variance, background_grad_variance)

    # `switch` is a move: current takes the just-updated background state, then
    # background becomes a fresh zero-count estimator.
    next_count::Float64 =
        ifelse.(switch_now, updated_background_count, updated_count)
    next_draw_mean::Vector{Float64} =
        ifelse.(switch_now, updated_background_draw_mean, updated_draw_mean)
    next_draw_variance::Vector{Float64} = ifelse.(
        switch_now, updated_background_draw_variance, updated_draw_variance)
    next_grad_mean::Vector{Float64} =
        ifelse.(switch_now, updated_background_grad_mean, updated_grad_mean)
    next_grad_variance::Vector{Float64} = ifelse.(
        switch_now, updated_background_grad_variance, updated_grad_variance)
    next_background_count::Float64 =
        ifelse.(switch_now, 0.0, updated_background_count)
    next_background_draw_mean::Vector{Float64} = ifelse.(
        switch_now, zero.(updated_background_draw_mean), updated_background_draw_mean)
    next_background_draw_variance::Vector{Float64} = ifelse.(
        switch_now, zero.(updated_background_draw_variance),
        updated_background_draw_variance)
    next_background_grad_mean::Vector{Float64} = ifelse.(
        switch_now, zero.(updated_background_grad_mean), updated_background_grad_mean)
    next_background_grad_variance::Vector{Float64} = ifelse.(
        switch_now, zero.(updated_background_grad_variance),
        updated_background_grad_variance)

    # Invalid ratio results (zero, infinity, NaN) preserve the previous scale,
    # exactly matching `fill_invalid = None`.  A successful adaptation still
    # recomputes the center/logdet and increments the transformation id.
    can_adapt::Bool = adapt_now & (next_count >= 3.0)
    variance_ratio_root::Vector{Float64} =
        sqrt.(next_draw_variance ./ next_grad_variance)
    finite_ratio::Vector{Bool} =
        (variance_ratio_root .- variance_ratio_root) .== zero.(variance_ratio_root)
    valid_ratio::Vector{Bool} =
        finite_ratio .& (variance_ratio_root .!= zero.(variance_ratio_root))
    clamped_variance::Vector{Float64} =
        clamp.(variance_ratio_root, LOWER_LIMIT, UPPER_LIMIT)
    candidate_stds::Vector{Float64} = sqrt.(clamped_variance)
    candidate_inv_stds::Vector{Float64} = sqrt.(inv.(clamped_variance))
    use_candidate::Vector{Bool} = valid_ratio .& can_adapt
    next_stds::Vector{Float64} = ifelse.(use_candidate, candidate_stds, stds)
    next_inv_stds::Vector{Float64} =
        ifelse.(use_candidate, candidate_inv_stds, inv_stds)
    candidate_transformation_mean::Vector{Float64} =
        next_draw_mean .+ next_stds .* next_stds .* next_grad_mean
    next_transformation_mean::Vector{Float64} = ifelse.(
        can_adapt, candidate_transformation_mean, transformation_mean)
    candidate_logdet::Float64 = sum(log.(next_inv_stds))
    next_logdet::Float64 = ifelse.(can_adapt, candidate_logdet, logdet)
    next_transformation_id::Float64 =
        ifelse.(can_adapt, transformation_id + 1.0, transformation_id)

    return next_draw_mean, next_draw_variance,
           next_grad_mean, next_grad_variance, next_count,
           next_background_draw_mean, next_background_draw_variance,
           next_background_grad_mean, next_background_grad_variance,
           next_background_count, next_stds, next_inv_stds,
           next_transformation_mean, next_logdet, next_transformation_id
end

const INITIALIZE_INPUTS = (:position, :gradient)
const INITIALIZE_OUTPUTS = (:stds, :inv_stds, :transformation_mean, :logdet)
const ADAPTATION_INPUTS = (
    :draw_mean, :draw_variance, :grad_mean, :grad_variance, :count,
    :background_draw_mean, :background_draw_variance,
    :background_grad_mean, :background_grad_variance, :background_count,
    :stds, :inv_stds, :transformation_mean, :logdet, :transformation_id,
    :position, :gradient, :is_good, :switch_now, :adapt_now,
)
const ADAPTATION_OUTPUTS = (
    :next_draw_mean, :next_draw_variance,
    :next_grad_mean, :next_grad_variance, :next_count,
    :next_background_draw_mean, :next_background_draw_variance,
    :next_background_grad_mean, :next_background_grad_variance,
    :next_background_count, :next_stds, :next_inv_stds,
    :next_transformation_mean, :next_logdet, :next_transformation_id,
)

const initialize_kernel = prepare(nutpie_diagonal_initialize;
    have = INITIALIZE_INPUTS, want = INITIALIZE_OUTPUTS)
const adaptation_kernel = prepare(nutpie_diagonal_adaptation;
    have = ADAPTATION_INPUTS, want = ADAPTATION_OUTPUTS)

const ORACLE_INPUTS = (
    init_position = [1.0, 10.0, -2.0, 3.0],
    init_gradient = [4.0, 2.0, -0.5, 1.0],
    steps = (
        (position = [2.0, 10.0, -1.0, 3.0],
         gradient = [3.0, 3.0, -0.5, 1.0],
         is_good = true, switch_now = false, adapt_now = true),
        (position = [4.0, 10.0, 1.0, 3.0],
         gradient = [1.0, 5.0, -0.5, 1.0],
         is_good = true, switch_now = false, adapt_now = true),
        (position = [99.0, 99.0, 99.0, 99.0],
         gradient = [99.0, 99.0, 99.0, 99.0],
         is_good = false, switch_now = false, adapt_now = true),
        (position = [7.0, 10.0, 4.0, 3.0],
         gradient = [-2.0, 8.0, -0.5, 1.0],
         is_good = true, switch_now = true, adapt_now = true),
        (position = [11.0, 10.0, 8.0, 3.0],
         gradient = [-6.0, 13.0, -0.5, 1.0],
         is_good = true, switch_now = false, adapt_now = true),
    ),
)

function initial_state(position::Vector{Float64}, gradient::Vector{Float64})
    stds, inv_stds, transformation_mean, logdet =
        initialize_kernel(position, gradient)
    zeros_like = zero(position)
    (
        draw_mean = copy(position), draw_variance = copy(zeros_like),
        grad_mean = copy(gradient), grad_variance = copy(zeros_like), count = 1.0,
        background_draw_mean = copy(position),
        background_draw_variance = copy(zeros_like),
        background_grad_mean = copy(gradient),
        background_grad_variance = copy(zeros_like),
        background_count = 1.0,
        stds, inv_stds, transformation_mean, logdet, transformation_id = 0.0,
    )
end

function advance(state, position::Vector{Float64}, gradient::Vector{Float64};
                 is_good::Bool, switch_now::Bool, adapt_now::Bool)
    values = adaptation_kernel(
        state.draw_mean, state.draw_variance,
        state.grad_mean, state.grad_variance, state.count,
        state.background_draw_mean, state.background_draw_variance,
        state.background_grad_mean, state.background_grad_variance,
        state.background_count, state.stds, state.inv_stds,
        state.transformation_mean, state.logdet, state.transformation_id,
        position, gradient, is_good, switch_now, adapt_now,
    )
    NamedTuple{(
        :draw_mean, :draw_variance, :grad_mean, :grad_variance, :count,
        :background_draw_mean, :background_draw_variance,
        :background_grad_mean, :background_grad_variance, :background_count,
        :stds, :inv_stds, :transformation_mean, :logdet, :transformation_id,
    )}(values)
end

function run()
    state = initial_state(ORACLE_INPUTS.init_position, ORACLE_INPUTS.init_gradient)
    for step in ORACLE_INPUTS.steps
        state = advance(state, step.position, step.gradient;
            is_good = step.is_good,
            switch_now = step.switch_now,
            adapt_now = step.adapt_now)
    end
    state
end

end # module
