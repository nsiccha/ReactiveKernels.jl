module NutsSourceOracle

using LinearAlgebra
using LogExpFunctions
using Random

# An eager, ordinary-Julia execution model of the locked authoring fixture.  It deliberately does not
# inspect MethodIR, a prepared plan, either native emitter, the Reactant control machine, canonical masks,
# or compiler storage.  Pure derived fields are recomputed eagerly after each authored HAVE mutation;
# this is observationally equivalent to the fixture's demand-driven currentness rules.

mutable struct PhasePoint{T,V<:AbstractVector{T}}
    pos::V
    mom::V
    dpot::V
    dkin::V
    pot::T
    kin::T
    ham::T
end

mutable struct MomentumView{V}
    mom::V
    dham_dmom::V
end

mutable struct MomentumSum{V}
    bwd::V
    fwd::V
end

mutable struct Tree{T,V}
    log_weight::Vector{T}
    bwd::MomentumView{V}
    bwd_fwd::MomentumView{V}
    summed_mom::MomentumSum{V}
end

mutable struct State{T,M,C,V}
    init::PhasePoint{T,V}
    fwd::PhasePoint{T,V}
    bwd::PhasePoint{T,V}
    trees::Vector{Tree{T,V}}
    proposals::Vector{PhasePoint{T,V}}
    metric::M
    chol_metric::C
    stepsize::T
    max_depth::Int
    min_dham::T
    gofwd::Bool
    may_sample::Bool
    may_continue::Bool
    dham::T
    diverged::Bool
    n_steps::Int
    reached_depth::Int
    acceptance_rate::T
end

function _phasepoint(pos::V, mom::V, metric, chol_metric) where {T,V<:AbstractVector{T}}
    dpot = similar(pos)
    dkin = similar(mom)
    phase = PhasePoint(copy(pos), copy(mom), dpot, dkin,
        zero(T), zero(T), zero(T))
    _recompute_potential!(phase)
    _recompute_kinetic!(phase, metric, chol_metric)
end

function _recompute_potential!(phase::PhasePoint)
    @. phase.dpot = 2 * phase.pos
    phase.pot = sum(abs2, phase.pos)
    phase.ham = phase.pot + phase.kin
    phase
end

function _recompute_kinetic!(phase::PhasePoint, metric, chol_metric)
    phase.dkin .= chol_metric \ phase.mom
    phase.kin = oftype(phase.pot, 0.5) *
        (logdet(chol_metric) + dot(phase.mom, phase.dkin))
    phase.ham = phase.pot + phase.kin
    phase
end


function _copy_phase!(destination::PhasePoint, source::PhasePoint)
    copyto!(destination.pos, source.pos)
    copyto!(destination.mom, source.mom)
    copyto!(destination.dpot, source.dpot)
    copyto!(destination.dkin, source.dkin)
    destination.pot = source.pot
    destination.kin = source.kin
    destination.ham = source.ham
    destination
end

function _tree(template::PhasePoint{T,V}) where {T,V}
    z1 = zero(template.pos)
    z2 = zero(template.pos)
    z3 = zero(template.pos)
    z4 = zero(template.pos)
    Tree(T[oftype(template.ham, -Inf), oftype(template.ham, -Inf)],
        MomentumView(z1, z2), MomentumView(zero(template.pos), zero(template.pos)),
        MomentumSum(z3, z4))
end

function State(; pos=Float64[1, 2], mom=Float64[3, 4],
        metric=Float64[2 0; 0 2], stepsize=0.1, max_depth=6,
        min_dham=-1000.0)
    T = eltype(pos)
    metric_t = Matrix{T}(metric)
    chol = cholesky(metric_t)
    init = _phasepoint(Vector{T}(pos), Vector{T}(mom), metric_t, chol)
    fwd = _phasepoint(Vector{T}(pos), Vector{T}(mom), metric_t, chol)
    bwd = _phasepoint(Vector{T}(pos), Vector{T}(mom), metric_t, chol)
    trees = [_tree(init) for _ in 1:(max_depth + 1)]
    proposals = [_phasepoint(Vector{T}(pos), Vector{T}(mom), metric_t, chol)
                 for _ in 1:(max_depth + 2)]
    State(init, fwd, bwd, trees, proposals, metric_t, chol, T(stepsize),
        max_depth, T(min_dham), true, true, true, zero(T),
        !(zero(T) >= T(min_dham)), 0, 0, zero(T))
end

function _reset!(state::State)
    state.gofwd = true
    state.may_sample = true
    state.may_continue = true
    state.dham = zero(state.init.ham)
    state.diverged = !(state.dham >= state.min_dham)
    state.n_steps = 0
    state.reached_depth = 0
    state.acceptance_rate = zero(state.init.ham)
    _copy_phase!(state.fwd, state.init)
    _copy_phase!(state.bwd, state.init)
    _copy_phase!(state.proposals[1], state.init)
    _copy_phase!(state.proposals[end], state.init)
    state
end

function _leapfrog!(state::State, phase::PhasePoint)
    halfstep = oftype(state.stepsize, 0.5) * state.stepsize
    @. phase.mom -= halfstep * phase.dpot
    _recompute_kinetic!(phase, state.metric, state.chol_metric)
    @. phase.pos += state.stepsize * phase.dkin
    _recompute_potential!(phase)
    @. phase.mom -= halfstep * phase.dpot
    _recompute_kinetic!(phase, state.metric, state.chol_metric)
    phase
end

function _collectstats!(state::State)
    state.n_steps += 1
    invn = one(state.dham) / state.n_steps
    acceptance = state.dham >= zero(state.dham) ? one(state.dham) : exp(state.dham)
    state.acceptance_rate = (one(state.dham) - invn) * state.acceptance_rate +
        invn * acceptance
end

function _swap_proposal!(state::State, i::Int, j::Int=length(state.proposals))
    state.proposals[i], state.proposals[j] = state.proposals[j], state.proposals[i]
end

function _finite_or_neginf(x)
    x - x == zero(x) ? x : -(one(x) / zero(x))
end

function _start!(state::State, phase::PhasePoint, depth::Int, rng)
    if depth == 1
        _leapfrog!(state, phase)
        state.dham = _finite_or_neginf(state.init.ham - phase.ham)
        _collectstats!(state)
        state.diverged = !(state.dham >= state.min_dham)
        if state.diverged
            state.may_continue = false
            return
        end
        state.trees[1].log_weight[1] = state.dham
        _copy_phase!(state.proposals[1], phase)
        return
    end

    _start!(state, phase, depth - 1, rng)
    if !state.may_continue
        state.may_sample = false
        return
    end
    _swap_proposal!(state, depth - 1, depth)
    _finish!(state, phase, depth - 1, rng)
    if state.may_sample
        difference = state.trees[depth - 1].log_weight[1] -
            state.trees[depth].log_weight[1]
        if difference > zero(difference) || -Random.randexp(rng) < difference
            _swap_proposal!(state, depth - 1, depth)
        end
    end
end

function _finish!(state::State, phase::PhasePoint, depth::Int, rng)
    tree = state.trees[depth]
    supertree = state.trees[depth + 1]
    tree.log_weight[2] = tree.log_weight[1]
    if depth == 1
        copyto!(supertree.bwd.mom, phase.mom)
        copyto!(supertree.bwd.dham_dmom, phase.dkin)
    else
        copyto!(supertree.bwd.mom, tree.bwd.mom)
        copyto!(supertree.bwd.dham_dmom, tree.bwd.dham_dmom)
        copyto!(tree.bwd_fwd.mom, phase.mom)
        copyto!(tree.bwd_fwd.dham_dmom, phase.dkin)
        copyto!(tree.summed_mom.bwd, tree.summed_mom.fwd)
    end

    _start!(state, phase, depth, rng)
    if !state.may_continue
        state.may_sample = false
        return
    end
    supertree.log_weight[1] = LogExpFunctions.logaddexp(
        tree.log_weight[1], tree.log_weight[2])
    if depth == 1
        @. supertree.summed_mom.fwd = supertree.bwd.mom + phase.mom
        backward_dot = zero(state.dham)
        forward_dot = zero(state.dham)
        for i in eachindex(supertree.summed_mom.fwd)
            backward_dot += supertree.summed_mom.fwd[i] * supertree.bwd.dham_dmom[i]
            forward_dot += supertree.summed_mom.fwd[i] * phase.dkin[i]
        end
        state.may_continue = backward_dot > zero(backward_dot) &&
            forward_dot > zero(forward_dot)
        return
    end

    @. supertree.summed_mom.fwd = tree.summed_mom.bwd + tree.summed_mom.fwd
    base_backward_dot = zero(state.dham)
    base_forward_dot = zero(state.dham)
    sum1_backward_dot = zero(state.dham)
    sum1_forward_dot = zero(state.dham)
    sum2_backward_dot = zero(state.dham)
    sum2_forward_dot = zero(state.dham)
    for i in eachindex(supertree.summed_mom.fwd)
        base_backward_dot += supertree.summed_mom.fwd[i] * supertree.bwd.dham_dmom[i]
        base_forward_dot += supertree.summed_mom.fwd[i] * phase.dkin[i]
        sum1_backward_dot += (tree.summed_mom.bwd[i] + tree.bwd.mom[i]) *
            supertree.bwd.dham_dmom[i]
        sum1_forward_dot += (tree.summed_mom.bwd[i] + tree.bwd.mom[i]) *
            tree.bwd.dham_dmom[i]
        sum2_backward_dot += (tree.bwd_fwd.mom[i] + tree.summed_mom.fwd[i]) *
            tree.bwd_fwd.dham_dmom[i]
        sum2_forward_dot += (tree.bwd_fwd.mom[i] + tree.summed_mom.fwd[i]) * phase.dkin[i]
    end
    state.may_continue =
        base_backward_dot > zero(base_backward_dot) &&
        base_forward_dot > zero(base_forward_dot) &&
        sum1_backward_dot > zero(sum1_backward_dot) &&
        sum1_forward_dot > zero(sum1_forward_dot) &&
        sum2_backward_dot > zero(sum2_backward_dot) &&
        sum2_forward_dot > zero(sum2_forward_dot)
end

function _flip_neg!(state::State, phase::PhasePoint, depth::Int)
    tree = state.trees[depth]
    @. tree.bwd.mom = -phase.mom
    @. tree.bwd.dham_dmom = -phase.dkin
    @. tree.summed_mom.fwd *= -1
end

function _flip!(state::State, depth::Int)
    depth > 1 || return
    state.gofwd = !state.gofwd
    _flip_neg!(state, state.gofwd ? state.bwd : state.fwd, depth)
end

function transition!(state::State, rng)
    Random.randn!(rng, state.init.mom)
    LinearAlgebra.lmul!(state.chol_metric.L, state.init.mom)
    _recompute_kinetic!(state.init, state.metric, state.chol_metric)
    _reset!(state)

    # reset establishes gofwd=true, so this is the authored backward-endpoint momentum mutation.
    @. state.bwd.mom *= -1
    _recompute_kinetic!(state.bwd, state.metric, state.chol_metric)
    state.trees[1].log_weight[1] = zero(state.init.ham)
    for depth in 1:state.max_depth
        state.reached_depth = depth
        rand(rng, Bool) && _flip!(state, depth)
        _finish!(state, state.gofwd ? state.fwd : state.bwd, depth, rng)
        state.may_sample || break
        difference = state.trees[depth].log_weight[1] -
            state.trees[depth].log_weight[2]
        if difference > zero(difference) || -Random.randexp(rng) < difference
            _swap_proposal!(state, depth)
        end
        state.may_continue || break
    end
    _copy_phase!(state.init, state.proposals[end])
    state
end

function snapshot(state::State)
    (pos=copy(state.init.pos), mom=copy(state.init.mom),
     dpot=copy(state.init.dpot), dkin=copy(state.init.dkin),
     pot=state.init.pot, kin=state.init.kin, ham=state.init.ham,
     gofwd=state.gofwd, may_sample=state.may_sample,
     may_continue=state.may_continue, diverged=state.diverged,
     n_steps=state.n_steps, reached_depth=state.reached_depth,
     acceptance_rate=state.acceptance_rate, dham=state.dham)
end

end # module NutsSourceOracle
