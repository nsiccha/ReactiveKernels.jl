module ReactiveKernelsPPLGibbsExt

import ReactiveKernelsPPL as PPL
import WarmupHMC as WH
using Random: AbstractRNG, Xoshiro
using LinearAlgebra: Diagonal

mutable struct GibbsState{P,R,D,N,W}
    const problem::P
    rng::R
    position::Vector{Float64}
    discrete::D
    nuts::N
    const warmup::W
end

# One exact draw of every finite integer block given the position.
function _draw_integers(problem, rng, position, values)
    for block in problem.integers
        conditional = Base.invokelatest(block.kernel, position, values)
        weights = PPL._gibbs_weights(conditional.logweights)
        draw = conditional.states[PPL._integer_draw_index(rng, weights)]
        for (path, value) in zip(block.paths, draw)
            values = PPL._gibbs_set(values, path, value)
        end
    end
    return values
end

function PPL.initialize_gibbs(rng::AbstractRNG, problem::PPL.GibbsProblem,
        u0::AbstractVector; discrete, metric=nothing, stepsize=nothing, max_tree_depth=10,
        warmup::Integer=1000)
    PPL._check_integer_parameters(problem.layout, discrete)
    position, values = Vector{Float64}(u0), PPL._gibbs_state(discrete)
    length(position) == problem.layout.total || throw(DimensionMismatch(
        "Gibbs position has the wrong dimension"))
    PPL._check_integer_parameters(problem.layout, values)
    isfinite(Base.invokelatest(problem.joint, position, values)) || throw(ArgumentError(
        "Gibbs initial position and integer state must have finite joint density"))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative"))
    owned_rng = copy(rng)
    report = (; sweeps=0, divergent=0, evaluations=0)
    nuts = if problem.continuous === nothing
        nothing
    elseif stepsize !== nothing
        scale = metric === nothing ? Diagonal(ones(length(position))) : deepcopy(metric)
        WH.FixedKernelNUTSState(Xoshiro(rand(owned_rng, UInt64)), position;
            metric=scale, stepsize, max_tree_depth)
    else
        metric === nothing || throw(ArgumentError(
            "a metric needs a stepsize; omit both to adapt the kernel during warm-up"))
        # Warm-up sweeps: exact integer draws, then a transition of WarmupHMC's
        # continuously adapting kernel, which follows the continuous target as
        # the integers change. Its final diagonal scale and step size become the
        # fixed kernel of every later sweep.
        target = problem.continuous
        target.discrete = values
        adapting = WH.ContinuousNUTSState(Xoshiro(rand(owned_rng, UInt64)), target; init=position)
        divergent = 0
        for _ in 1:warmup
            values = _draw_integers(problem, owned_rng, adapting.position, values)
            target.discrete = values
            divergent += WH.continuous_nuts_step(adapting, target).divergent
        end
        position = copy(adapting.position)
        report = (; sweeps=warmup, divergent, evaluations=adapting.total_evaluation_counter)
        WH.FixedKernelNUTSState(Xoshiro(rand(owned_rng, UInt64)), position;
            metric=copy(adapting.scale), stepsize=adapting.stepsize, max_tree_depth)
    end
    GibbsState(problem, owned_rng, position, values, nuts, report)
end

function PPL.gibbs_step!(state::GibbsState)
    problem = state.problem
    rng = copy(state.rng)
    values = _draw_integers(problem, rng, state.position, state.discrete)
    nuts, position, divergent, stats = if state.nuts === nothing
        nothing, state.position, false, nothing
    else
        problem.continuous.discrete = values
        transition = WH.fixed_kernel_nuts_step(state.nuts, problem.continuous; blas_threads=1)
        transition.state, copy(transition.state.position), transition.divergent, transition.stats
    end
    state.rng, state.discrete, state.nuts, state.position = rng, values, nuts, position
    return (; state, divergent, stats)
end

function PPL.gibbs_draws(state::GibbsState, n::Integer; discard::Integer=0)
    n >= 0 && discard >= 0 || throw(ArgumentError("draw and discard counts must be nonnegative"))
    for _ in 1:discard
        PPL.gibbs_step!(state)
    end
    positions = Matrix{Float64}(undef, length(state.position), n)
    discrete = Vector{typeof(state.discrete)}(undef, n)
    divergent = falses(n)
    for i in 1:n
        transition = PPL.gibbs_step!(state)
        positions[:, i] = state.position
        discrete[i] = deepcopy(state.discrete)
        divergent[i] = transition.divergent
    end
    parameters = PPL.restore_draws(state.problem.layout, positions; discrete)
    return (; positions, discrete, parameters, divergent)
end

end
