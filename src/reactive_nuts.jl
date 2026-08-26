# Flat compiled-reactive NUTS phase-point group — increments 1-2 of the ca9
# ReactiveHMC port. Following poc's approved "(b) FLATTENED" verdict, the sampler
# state is ONE wide `ReactiveProgram` spanning the `init`, `fwd`, and `bwd`
# endpoints, with a `gofwd` HAVE source. The active-endpoint selection, the energy
# error `dham`, and the `diverged` flag are ordinary reactive recipes over this
# wider HAVE — no nested compiled states, no HMC-specific manual invalidation.
#
# ca9's `nuts_state.dham` is refreshed by hand (`@invalidatedependants!`) because
# ReactiveObjects leaves cross-object reactivity unsolved. Here `dham` is a genuine
# compiled reactive node: mutating any active endpoint's `mom`/`pos` invalidates
# `dham` and a later read recomputes minimally. Selection branches at compute time
# over static edges to BOTH endpoints, so flipping `gofwd` reselects with no graph
# rebuild.
#
# Increment 2 — near-zero reactive overhead. Each endpoint's potential+gradient and
# kinetic work is ONE owned mutable single-output bundle recipe plus pure 0-B
# projections. The bundle slots are routed through the MutatingFunctions-agnostic
# in-place getter cache hook (a hand-written `cache_apply`, no MF dependency), so
# each `CompiledReactiveState` SLOT owns its buffer and reuses it in place across
# invalidate→recompute. Because ownership is per slot (per instance), `copy` /
# `copyto!` deep-copy the buffers and distinct proposals never alias — the property
# a caller-shared or program-level buffer would violate.
#
# The group reuses the tested `ReactivePhasePoint{S,H}` runtime wrapper, so every
# node is reachable as a property. It is AD-agnostic: `potential_gradient!(gradient,
# position) -> potential` is the caller-supplied scalar-potential boundary that
# fills the passed (slot-owned) gradient buffer in place — the DI+Enzyme
# `value_and_gradient!` is wired in at the call site with no shared caller buffer.

const _REACTIVE_NUTS_ENDPOINTS = (:init, :fwd, :bwd)
const _REACTIVE_NUTS_DEFAULT_MIN_DHAM = -1000.0

# --- Owned per-endpoint bundles: single-output mutable slots reused in place. ---

"""
    _ValueGradient(value, gradient)

Owned value/gradient bundle for one endpoint's potential evaluation. The `gradient`
buffer is filled in place on every recompute; `value` is the scalar potential.
"""
mutable struct _ValueGradient{T,V<:AbstractVector{T}}
    value::T
    gradient::V
end

"""
    _Kinetic(kinetic, velocity)

Owned kinetic-energy bundle for one endpoint: `velocity = M^-1 mom` filled in place,
`kinetic = 0.5(logdet M + mom·velocity)`.
"""
mutable struct _Kinetic{T,V<:AbstractVector{T}}
    kinetic::T
    velocity::V
end

# Recipe op wrapping the caller's in-place scalar-potential boundary
# `f(gradient, position) -> potential`. A plain (pure) call — used on the first
# evaluation and in the pure program — allocates a fresh bundle; the in-place
# `cache_apply` method below reuses the slot bundle's buffer.
struct _GradientBundleOp{F}
    f::F
end
@inline function (op::_GradientBundleOp)(position)
    gradient = similar(position)
    value = op.f(gradient, position)
    _ValueGradient(value, gradient)
end

_kinetic_energy(chol, momentum, velocity) =
    convert(eltype(velocity),
            (logdet(chol) + dot(momentum, velocity)) / 2)

struct _KineticBundleOp end
@inline function (::_KineticBundleOp)(chol, momentum)
    velocity = chol \ momentum
    _Kinetic(_kinetic_energy(chol, momentum, velocity), velocity)
end

# Hand-written, MutatingFunctions-agnostic in-place hook. Contract: reuse and
# return the (possibly new) cache; treat an immutable/isbits or unregistered cache
# as a passthrough recompute. Only the two owned bundle types reuse in place.
@inline function _nuts_cache_apply(cache::_ValueGradient, op::_GradientBundleOp,
                                   position)
    cache.value = op.f(cache.gradient, position)
    cache
end
@inline function _nuts_cache_apply(cache::_Kinetic, ::_KineticBundleOp,
                                   chol, momentum)
    copyto!(cache.velocity, momentum)
    ldiv!(chol, cache.velocity)
    cache.kinetic = _kinetic_energy(chol, momentum, cache.velocity)
    cache
end
@inline _nuts_cache_apply(cache, op, args...) = op(args...)

# Route ONLY the two owned bundle recipes through the in-place hook; the scalar and
# borrowed-vector projections stay on the pure (allocation-free) branch.
@inline _nuts_is_mutating(recipe::Recipe) =
    valtype(only(recipe.outputs)) <: Union{_ValueGradient,_Kinetic}

# The bundles are mutable, buffer-owning slot values reused in place — exactly like
# arrays, they MUST be deep-copied when a CompiledReactiveState is copied, or a
# proposal clone's in-place recompute would corrupt the source's buffer. Extend the
# per-slot copy hooks so `copy` clones the owned buffer and `copyto!` fills the
# destination's buffer in place (preserving per-instance ownership).
_copy_slot_value(bundle::_ValueGradient) =
    _ValueGradient(bundle.value, copy(bundle.gradient))
_copy_slot_value(bundle::_Kinetic) =
    _Kinetic(bundle.kinetic, copy(bundle.velocity))

function _copy_slot_value!(destination::_ValueGradient, source::_ValueGradient)
    destination.value = source.value
    destination.gradient = _copy_slot_value!(destination.gradient, source.gradient)
    destination
end
function _copy_slot_value!(destination::_Kinetic, source::_Kinetic)
    destination.kinetic = source.kinetic
    destination.velocity = _copy_slot_value!(destination.velocity, source.velocity)
    destination
end

"""
    reactive_nuts_group(potential_gradient!, metric, position, momentum;
                        gofwd = true, min_dham = -1000.0)

Build the flat compiled-reactive NUTS phase-point group. All three endpoints
(`init`, `fwd`, `bwd`) are initialized from `(position, momentum)` and share the
Cholesky factor of the fixed Euclidean `metric`. `potential_gradient!(gradient, q)`
must fill `gradient` in place and return the scalar potential `U(q)`.

The returned [`ReactivePhasePoint`](@ref) exposes, per endpoint `e` in
`(:init, :fwd, :bwd)`, the HAVE sources `e_pos`, `e_mom` and the reactive nodes
`e_pot`, `e_dpot_dpos`, `e_kin`, `e_dham_dmom`, `e_ham`; plus the shared HAVE
sources `metric`, `gofwd`, and the reactive selection/diagnostic nodes
`active_ham` (`gofwd ? fwd_ham : bwd_ham`), `dham` (`init_ham - active_ham`), and
`diverged` (`!(dham >= min_dham)`).

Each endpoint's potential+gradient and kinetic work is an owned single-output
bundle reused in place through the per-slot in-place getter hook, so a warmed
invalidate→recompute is allocation-free apart from the caller boundary's own work.
"""
function reactive_nuts_group(potential_gradient!, metric, position, momentum;
                             gofwd::Bool = true,
                             min_dham::Real = _REACTIVE_NUTS_DEFAULT_MIN_DHAM)
    gradient_op = _GradientBundleOp(potential_gradient!)
    kinetic_op = _KineticBundleOp()
    gradient_bundle0 = gradient_op(position)
    chol0 = cholesky(metric)
    kinetic_bundle0 = kinetic_op(chol0, momentum)
    potential0 = gradient_bundle0.value
    ham0 = potential0 + kinetic_bundle0.kinetic

    graph = Graph()
    metric_value = value!(graph, :metric, typeof(metric))
    gofwd_value = value!(graph, :gofwd, Bool)
    chol = value!(graph, :chol_metric, typeof(chol0))
    add!(graph, metric_value => chol, cholesky)

    # Per-endpoint Hamiltonian sub-graph: an owned value/gradient bundle and an
    # owned kinetic bundle (both reused in place), then pure projections into the
    # exposed scalar/vector nodes, joined into ham. Endpoints differ only by their
    # HAVE sources.
    ports = Dict{Symbol,Any}()
    for endpoint in _REACTIVE_NUTS_ENDPOINTS
        pos = ports[Symbol(endpoint, :_pos)] =
            value!(graph, Symbol(endpoint, :_pos), typeof(position))
        mom = ports[Symbol(endpoint, :_mom)] =
            value!(graph, Symbol(endpoint, :_mom), typeof(momentum))
        gradient_bundle =
            value!(graph, Symbol(endpoint, :_valgrad), typeof(gradient_bundle0))
        pot = ports[Symbol(endpoint, :_pot)] =
            value!(graph, Symbol(endpoint, :_pot), typeof(potential0))
        dpot = ports[Symbol(endpoint, :_dpot_dpos)] =
            value!(graph, Symbol(endpoint, :_dpot_dpos),
                   typeof(gradient_bundle0.gradient))
        kinetic_bundle =
            value!(graph, Symbol(endpoint, :_kinetic), typeof(kinetic_bundle0))
        kinetic = ports[Symbol(endpoint, :_kin)] =
            value!(graph, Symbol(endpoint, :_kin), typeof(kinetic_bundle0.kinetic))
        dham_dmom = ports[Symbol(endpoint, :_dham_dmom)] =
            value!(graph, Symbol(endpoint, :_dham_dmom),
                   typeof(kinetic_bundle0.velocity))
        ham = ports[Symbol(endpoint, :_ham)] =
            value!(graph, Symbol(endpoint, :_ham), typeof(ham0))
        add!(graph, pos => gradient_bundle, gradient_op)
        add!(graph, gradient_bundle => pot, bundle -> bundle.value)
        add!(graph, gradient_bundle => dpot, bundle -> bundle.gradient)
        add!(graph, (chol, mom) => kinetic_bundle, kinetic_op)
        add!(graph, kinetic_bundle => kinetic, bundle -> bundle.kinetic)
        add!(graph, kinetic_bundle => dham_dmom, bundle -> bundle.velocity)
        add!(graph, (pot, kinetic) => ham, +)
    end

    # Active-endpoint selection + energy error + divergence — reactive recipes over
    # the wider HAVE. `gofwd` is a HAVE source; `active_ham` branches over the
    # static edges to both endpoint hamiltonians at compute time.
    active_ham = value!(graph, :active_ham, typeof(ham0))
    add!(graph, (gofwd_value, ports[:fwd_ham], ports[:bwd_ham]) => active_ham,
         (forward, forward_ham, backward_ham) ->
             forward ? forward_ham : backward_ham)
    # `dham` is the finite-guarded energy error init_ham - active_ham (matching the
    # oracle's _finite_or_neginf(init.ham - forward.ham) exactly), and `min_dham` is
    # a reactive HAVE source so the divergence threshold can be changed and
    # `diverged` recomputes reactively.
    dham = value!(graph, :dham, typeof(ham0))
    add!(graph, (ports[:init_ham], active_ham) => dham,
         (init_ham, selected_ham) -> _finite_or_neginf(init_ham - selected_ham))
    min_dham_value = value!(graph, :min_dham, typeof(ham0))
    diverged = value!(graph, :diverged, Bool)
    add!(graph, (dham, min_dham_value) => diverged,
         (energy_error, threshold) -> !(energy_error >= threshold))

    haves = (metric_value, gofwd_value, min_dham_value,
             ports[:init_pos], ports[:init_mom],
             ports[:fwd_pos], ports[:fwd_mom],
             ports[:bwd_pos], ports[:bwd_mom])
    endpoint_wants = Tuple(Iterators.flatten(
        (ports[Symbol(endpoint, suffix)]
         for suffix in (:_pot, :_dpot_dpos, :_kin, :_dham_dmom, :_ham))
        for endpoint in _REACTIVE_NUTS_ENDPOINTS))
    wants = (chol, active_ham, dham, diverged, endpoint_wants...)

    program = _prepare_reactive(graph; have = haves, want = wants,
                                cache_apply = _nuts_cache_apply,
                                is_mutating = _nuts_is_mutating)
    state = program(metric, gofwd, convert(typeof(ham0), min_dham),
                    copy(position), copy(momentum),
                    copy(position), copy(momentum),
                    copy(position), copy(momentum))

    handle_names = (:metric, :gofwd, :min_dham, :chol_metric,
                    :active_ham, :dham, :diverged,
                    (Symbol(endpoint, suffix)
                     for endpoint in _REACTIVE_NUTS_ENDPOINTS
                     for suffix in (:_pos, :_mom, :_pot, :_dpot_dpos,
                                    :_kin, :_dham_dmom, :_ham))...)
    handle_ports = (metric_value, gofwd_value, min_dham_value, chol,
                    active_ham, dham, diverged,
                    (ports[Symbol(endpoint, suffix)]
                     for endpoint in _REACTIVE_NUTS_ENDPOINTS
                     for suffix in (:_pos, :_mom, :_pot, :_dpot_dpos,
                                    :_kin, :_dham_dmom, :_ham))...)
    values = NamedTuple{handle_names}(handle_ports)
    _phasepoint(program, state, values)
end

# ---------------------------------------------------------------------------
# CompiledNUTSState — the ca9 multinomial NUTS transition running on the flat
# compiled-reactive group. The tree-growth / U-turn / proposal / statistics
# orchestration mirrors the ordinary-Julia `NUTSState` oracle
# (src/hmc.jl:424-735) line-for-line, but every phase-point read/write is routed
# through the ONE flat group's handles: `init`, and the two moving endpoints
# `fwd`/`bwd` selected by the reactive `gofwd` source. The energy error `dham` and
# `diverged` are read from the group's reactive nodes (init_ham - active_ham),
# never hand-computed. Recursion, loops, RNG draws and proposal swaps stay
# ordinary inferred Julia, exactly as in ca9. `NUTSState` is retained unchanged as
# the parity oracle.
# ---------------------------------------------------------------------------

# One proposal is an ordinary owned (pos, mom) snapshot; swaps are plain vector
# element swaps and the accepted sample is `proposals[end].pos`.
mutable struct _NUTSProposal{V<:AbstractVector}
    pos::V
    mom::V
end
_nuts_proposal(pos, mom) = _NUTSProposal(copy(pos), copy(mom))

mutable struct CompiledNUTSState{R,G,F,S,T,TR,PR} <: AbstractNUTSState
    rng::R
    group::G                 # the flat compiled-reactive phase-point group
    step_f::F
    stats_f::S
    max_depth::Int
    min_energy_error::T
    trees::TR
    proposals::PR
    go_forward::Bool
    may_sample::Bool
    may_continue::Bool
    energy_error::T
    diverged::Bool
    depth::Int
    n_steps::Int
    acceptance_sum::T
end

# A concrete, typed read-only view of the group's `init` endpoint that mirrors the
# ordinary phase-point interface (`.pos/.mom/.ham/.metric/.dham_dpos/.dham_dmom/
# .pot/.chol_metric`) used by adaptation, statistics, and user callbacks. It maps
# each name to the corresponding `init_*` (or shared) reactive node, so consumers
# see the init endpoint — never the whole group.
struct _CompiledInitView{G}
    group::G
end
@inline function Base.getproperty(view::_CompiledInitView, name::Symbol)
    group = getfield(view, :group)
    name === :pos && return group.init_pos
    name === :mom && return group.init_mom
    name === :ham && return group.init_ham
    name === :pot && return group.init_pot
    name === :dham_dpos && return group.init_dpot_dpos
    name === :dham_dmom && return group.init_dham_dmom
    name === :metric && return group.metric
    name === :chol_metric && return group.chol_metric
    name === :group && return group
    return getproperty(group, name)
end

# Expose ca9-style property names over the group so the compiled state reads like
# the reference object (`state.dham`, `state.gofwd`, `state.init.pos`, ...). `init`
# is a typed init-endpoint view, NOT the whole group.
function Base.getproperty(state::CompiledNUTSState, name::Symbol)
    name === :dham && return getfield(state, :energy_error)
    name === :gofwd && return getfield(state, :go_forward)
    name === :init && return _CompiledInitView(getfield(state, :group))
    getfield(state, name)
end

reactive_program(state::CompiledNUTSState) = state.group.state.program

# --- State-access interface for the shared adaptation/statistics helpers. ---
_nuts_position(state::CompiledNUTSState) = state.group.init_pos
_nuts_metric(state::CompiledNUTSState) = state.group.metric
function _set_nuts_metric!(state::CompiledNUTSState, metric)
    set!(state.group.state, state.group.handles.metric, metric)
    metric
end
function _nuts_metric_is_source(state::CompiledNUTSState)
    handles = state.group.handles
    hasproperty(handles, :metric) &&
        state.group.state.program.sources[_slot_index(handles.metric)]
end

"""
    compiled_nuts_state(init::ReactivePhasePoint; rng, step_f, stats_f=nothing,
                        max_depth=10, min_dham=-1000.0)

Build the compiled-reactive multinomial NUTS transition from a flat phase-point
group produced by [`reactive_nuts_group`](@ref). Same public surface as the
[`NUTSState`](@ref) oracle — [`step!`](@ref), [`sample!`](@ref),
[`refresh_momentum!`](@ref), [`diagnostics`](@ref) — with the sampler state living
in one compiled `ReactiveProgram`.
"""
const _REACTIVE_NUTS_REQUIRED_HANDLES = (
    :gofwd, :chol_metric, :dham, :diverged, :active_ham,
    :init_pos, :init_mom, :init_ham, :init_dpot_dpos, :init_dham_dmom,
    :fwd_pos, :fwd_mom, :fwd_ham, :fwd_dpot_dpos, :fwd_dham_dmom,
    :bwd_pos, :bwd_mom, :bwd_ham, :bwd_dpot_dpos, :bwd_dham_dmom,
)

function compiled_nuts_state(group::ReactivePhasePoint; rng, step_f,
                             stats_f = nothing, max_depth::Integer = 10,
                             min_dham = -1000.0)
    max_depth >= 1 || throw(ArgumentError("max_depth must be positive"))
    handle_names = keys(getfield(group, :handles))
    all(name -> name in handle_names, (_REACTIVE_NUTS_REQUIRED_HANDLES...,
                                       :min_dham)) ||
        throw(ArgumentError(
            "compiled_nuts_state requires a flat phase-point group from " *
            "reactive_nuts_group; got a ReactivePhasePoint without the " *
            "init/fwd/bwd + gofwd/dham/min_dham handles"))
    # The compiled transition drives one in-place leapfrog on the active endpoint.
    # Only `partial(leapfrog!; stepsize=...)` is supported; reject any other
    # integrator rather than silently ignoring its func/args.
    (step_f isa PartialFunction && step_f.func === leapfrog! &&
     isempty(step_f.largs) && isempty(step_f.rargs) &&
     hasproperty(step_f, :stepsize)) || throw(ArgumentError(
        "compiled_nuts_state supports step_f = partial(leapfrog!; stepsize=...); " *
        "other integrators are not yet supported on the compiled group"))
    prototype = group.init_mom
    scalar = typeof(float(group.init_ham - group.init_ham))
    # Sync the reactive divergence threshold so group.diverged uses this min_dham.
    # Convert to the group's potential scalar type (which may differ from the
    # coordinate eltype), read off the existing min_dham source value.
    set!(group.state, group.handles.min_dham,
         convert(typeof(group.min_dham), min_dham))
    trees = [_nuts_tree((; mom = prototype)) for _ in 1:(max_depth + 1)]
    proposals = [_nuts_proposal(group.init_pos, group.init_mom)
                 for _ in 1:(max_depth + 2)]
    CompiledNUTSState(
        rng, group, step_f, stats_f, Int(max_depth),
        convert(scalar, min_dham), trees, proposals,
        true, true, true, zero(scalar), false, 0, 0, zero(scalar),
    )
end

# --- Type-stable endpoint access over the flat group (branch on go_forward). ---
# fwd/bwd are the two moving endpoints; `go_forward` selects which is "forward".
@inline _cn_fwd_mom(state) = state.go_forward ? state.group.fwd_mom : state.group.bwd_mom
@inline _cn_fwd_vel(state) = state.go_forward ? state.group.fwd_dham_dmom : state.group.bwd_dham_dmom
@inline _cn_fwd_ham(state) = state.go_forward ? state.group.fwd_ham : state.group.bwd_ham
@inline _cn_fwd_pos(state) = state.go_forward ? state.group.fwd_pos : state.group.bwd_pos
@inline _cn_bwd_mom(state) = state.go_forward ? state.group.bwd_mom : state.group.fwd_mom
@inline _cn_bwd_vel(state) = state.go_forward ? state.group.bwd_dham_dmom : state.group.fwd_dham_dmom

# One in-place leapfrog on the ACTIVE (forward) endpoint, matching ca9's
# integrator: mom half-kick (reads gradient), pos drift (reads velocity), mom
# half-kick (reads recomputed gradient). Uses the group's per-slot in-place
# bundles, so the reactive recompute is allocation-free.
@inline function _group_leapfrog!(group::ReactivePhasePoint, ::Val{P},
                                  stepsize) where {P}
    gs = group.state
    handles = group.handles
    mom_h = getfield(handles, Symbol(P, :_mom))
    pos_h = getfield(handles, Symbol(P, :_pos))
    grad_h = getfield(handles, Symbol(P, :_dpot_dpos))
    vel_h = getfield(handles, Symbol(P, :_dham_dmom))
    gradient = get!(gs, grad_h)
    mutate!(gs, mom_h) do momentum
        @. momentum -= 0.5 * stepsize * gradient
        momentum
    end
    velocity = get!(gs, vel_h)
    mutate!(gs, pos_h) do position
        @. position += stepsize * velocity
        position
    end
    gradient2 = get!(gs, grad_h)
    mutate!(gs, mom_h) do momentum
        @. momentum -= 0.5 * stepsize * gradient2
        momentum
    end
    group
end

@inline function _cn_step_forward!(state::CompiledNUTSState)
    stepsize = state.step_f.stepsize
    state.go_forward ? _group_leapfrog!(state.group, Val(:fwd), stepsize) :
                       _group_leapfrog!(state.group, Val(:bwd), stepsize)
end

# Keep the reactive selection source in sync with go_forward, so the group's
# `active_ham`/`dham`/`diverged` nodes track the current forward endpoint.
@inline function _cn_sync_gofwd!(state::CompiledNUTSState)
    set!(state.group.state, state.group.handles.gofwd, state.go_forward)
    state
end

@inline function _cn_negate_backward_mom!(state::CompiledNUTSState)
    handles = state.group.handles
    gs = state.group.state
    if state.go_forward
        mutate!(gs, handles.bwd_mom) do m; @. m *= -1; m; end
    else
        mutate!(gs, handles.fwd_mom) do m; @. m *= -1; m; end
    end
    state
end

# Reset both moving endpoints to `init` in one 0-alloc HAVE-boundary group copy,
# then reset proposals/trees/flags — the ca9 per-transition restore.
function _cn_reset_transition!(state::CompiledNUTSState)
    handles = state.group.handles
    gs = state.group.state
    copy_group!(gs,
        (handles.fwd_pos, handles.fwd_mom, handles.bwd_pos, handles.bwd_mom),
        (handles.init_pos, handles.init_mom, handles.init_pos, handles.init_mom))
    init_pos = state.group.init_pos
    init_mom = state.group.init_mom
    for proposal in state.proposals
        copyto!(proposal.pos, init_pos)
        copyto!(proposal.mom, init_mom)
    end
    foreach(_reset_tree!, state.trees)
    state.go_forward = true
    _cn_sync_gofwd!(state)
    state.may_sample = true
    state.may_continue = true
    state.energy_error = zero(state.energy_error)
    state.diverged = false
    state.depth = 0
    state.n_steps = 0
    state.acceptance_sum = zero(state.acceptance_sum)
    state
end

function _cn_swap_proposal!(state::CompiledNUTSState, first_index::Int,
                            second_index::Int = length(state.proposals))
    state.proposals[first_index], state.proposals[second_index] =
        state.proposals[second_index], state.proposals[first_index]
    state
end

function _cn_snapshot_forward!(proposal::_NUTSProposal, state::CompiledNUTSState)
    copyto!(proposal.pos, _cn_fwd_pos(state))
    copyto!(proposal.mom, _cn_fwd_mom(state))
    proposal
end

function _cn_flip!(state::CompiledNUTSState, depth::Int)
    depth > 1 || return state
    state.go_forward = !state.go_forward
    _cn_sync_gofwd!(state)
    tree = state.trees[depth]
    backward_mom = _cn_bwd_mom(state)
    backward_vel = _cn_bwd_vel(state)
    @. tree.backward.momentum = -backward_mom
    @. tree.backward.velocity = -backward_vel
    @. tree.summed_momentum.forward *= -1
    state
end

@inline function _cn_collect_stats!(state::CompiledNUTSState)
    state.stats_f === nothing || state.stats_f(state)
    state
end

function _cn_start_tree!(state::CompiledNUTSState, depth::Int)
    if depth == 1
        _cn_step_forward!(state)
        state.n_steps += 1
        # Reactive energy error + divergence read straight from the group's compiled
        # nodes: dham = _finite_or_neginf(init_ham - active_ham), selected by gofwd;
        # diverged = !(dham >= min_dham) over the reactive min_dham threshold source.
        state.energy_error = state.group.dham
        state.acceptance_sum += _min1exp(state.energy_error)
        state.diverged = state.group.diverged
        _cn_collect_stats!(state)
        if state.diverged
            state.may_continue = false
            return state
        end
        state.trees[1].log_weight[1] = state.energy_error
        _cn_snapshot_forward!(state.proposals[1], state)
        return state
    end

    _cn_start_tree!(state, depth - 1)
    if !state.may_continue
        state.may_sample = false
        return state
    end
    _cn_swap_proposal!(state, depth - 1, depth)
    _cn_finish_tree!(state, depth - 1)
    if state.may_sample && _rand_bernoulli_log(
            state.rng,
            state.trees[depth - 1].log_weight[1] -
                state.trees[depth].log_weight[1],
        )
        _cn_swap_proposal!(state, depth - 1, depth)
    end
    state
end

function _cn_finish_tree!(state::CompiledNUTSState, depth::Int)
    tree = state.trees[depth]
    supertree = state.trees[depth + 1]
    tree.log_weight[2] = tree.log_weight[1]
    forward_mom = _cn_fwd_mom(state)
    forward_vel = _cn_fwd_vel(state)

    if depth == 1
        copyto!(supertree.backward.momentum, forward_mom)
        copyto!(supertree.backward.velocity, forward_vel)
    else
        copyto!(supertree.backward.momentum, tree.backward.momentum)
        copyto!(supertree.backward.velocity, tree.backward.velocity)
        copyto!(tree.backward_forward.momentum, forward_mom)
        copyto!(tree.backward_forward.velocity, forward_vel)
        copyto!(tree.summed_momentum.backward, tree.summed_momentum.forward)
    end

    _cn_start_tree!(state, depth)
    if !state.may_continue
        state.may_sample = false
        return state
    end

    # forward endpoint may have moved during the recursive start_tree!; re-read.
    forward_mom = _cn_fwd_mom(state)
    forward_vel = _cn_fwd_vel(state)
    supertree.log_weight[1] = logaddexp(tree.log_weight[1], tree.log_weight[2])
    if depth == 1
        @. supertree.summed_momentum.forward =
            supertree.backward.momentum + forward_mom
        state.may_continue = _compute_criterion(
            supertree.summed_momentum.forward,
            supertree.backward.velocity,
            forward_vel,
        )
    else
        @. supertree.summed_momentum.forward =
            tree.summed_momentum.backward + tree.summed_momentum.forward
        state.may_continue =
            _compute_criterion(
                supertree.summed_momentum.forward,
                supertree.backward.velocity,
                forward_vel,
            ) &&
            _compute_criterion_sum(
                tree.summed_momentum.backward,
                tree.backward.momentum,
                supertree.backward.velocity,
                tree.backward.velocity,
            ) &&
            _compute_criterion_sum(
                tree.backward_forward.momentum,
                tree.summed_momentum.forward,
                tree.backward_forward.velocity,
                forward_vel,
            )
    end
    state
end

function _cn_restore_init!(state::CompiledNUTSState)
    proposal = state.proposals[end]
    handles = state.group.handles
    gs = state.group.state
    mutate!(gs, handles.init_pos) do position
        copyto!(position, proposal.pos)
        position
    end
    mutate!(gs, handles.init_mom) do momentum
        copyto!(momentum, proposal.mom)
        momentum
    end
    state
end

"Advance one multinomial NUTS transition on the compiled-reactive group."
function step!(state::CompiledNUTSState)
    _cn_reset_transition!(state)
    _cn_negate_backward_mom!(state)
    state.trees[1].log_weight[1] = 0

    for depth in 1:state.max_depth
        rand(state.rng, Bool) && _cn_flip!(state, depth)
        _cn_finish_tree!(state, depth)
        state.depth = depth
        state.may_sample || break
        if _rand_bernoulli_log(
                state.rng,
                state.trees[depth].log_weight[1] -
                    state.trees[depth].log_weight[2],
            )
            _cn_swap_proposal!(state, depth)
        end
        state.may_continue || break
    end
    _cn_restore_init!(state)
    diagnostics(state)
end

function diagnostics(state::CompiledNUTSState)
    acceptance = state.n_steps == 0 ? zero(state.acceptance_sum) :
                 state.acceptance_sum / state.n_steps
    NUTSDiagnostics(state.depth, state.n_steps, acceptance,
                    state.diverged, state.energy_error)
end

function refresh_momentum!(state::CompiledNUTSState)
    factor = state.group.chol_metric
    mutate!(state.group.state, state.group.handles.init_mom) do momentum
        randn!(state.rng, momentum)
        lmul!(factor.L, momentum)
        momentum
    end
    state.group
end

function sample!(state::CompiledNUTSState)
    state.stats_f isa TrajectoryStats && reset!(state.stats_f, state.init)
    refresh_momentum!(state)
    step!(state)
end
# sample!(state, draws) is the shared AbstractNUTSState method in hmc.jl.

# --- Adaptation probe: one-step acceptance at a candidate stepsize. Mirrors the
# oracle's _probe_acceptance (same RNG draw order) so find_initial_stepsize! is
# transition-for-transition identical to the oracle. ---
function _probe_acceptance(state::CompiledNUTSState, stepsize)
    refresh_momentum!(state)
    probe = copy(state.group)
    handles = probe.handles
    gs = probe.state
    copy_group!(gs, (handles.fwd_pos, handles.fwd_mom),
                (handles.init_pos, handles.init_mom))
    set!(gs, handles.gofwd, true)
    _group_leapfrog!(probe, Val(:fwd), stepsize)
    # probe.dham = _finite_or_neginf(init_ham - fwd_ham) — the one-step energy error.
    _min1exp(probe.dham)
end

# --- Forward-endpoint readers for the optional TrajectoryStats recorder. ---
@inline _cn_fwd_dpos(state) =
    state.go_forward ? state.group.fwd_dpot_dpos : state.group.bwd_dpot_dpos
@inline _cn_fwd_pot(state) =
    state.go_forward ? state.group.fwd_pot : state.group.bwd_pot

function (stats::TrajectoryStats)(state::CompiledNUTSState)
    prepend = !state.go_forward
    column = _reserve_trajectory_column!(stats, prepend)
    stats.position_storage[:, column] .= _cn_fwd_pos(state)
    stats.gradient_storage[:, column] .= -_cn_fwd_dpos(state)
    if prepend
        pushfirst!(stats.dhams, state.energy_error)
        pushfirst!(stats.pots, _cn_fwd_pot(state))
        pushfirst!(stats.idxs, length(stats.idxs))
    else
        push!(stats.dhams, state.energy_error)
        push!(stats.pots, _cn_fwd_pot(state))
        push!(stats.idxs, length(stats.idxs))
    end
    stats
end

"""
    nuts_state(point::ReactivePhasePoint; rng, step_f, ...)

Construct a NUTS sampler. When `point` is a flat phase-point group from
[`reactive_nuts_group`](@ref) it builds the compiled-reactive
[`CompiledNUTSState`](@ref) (the default public path); a plain single-endpoint
phase point builds the ordinary-Julia reference oracle (`_oracle_nuts_state`,
retained for parity tests).
"""
function nuts_state(point::ReactivePhasePoint; kwargs...)
    _is_reactive_nuts_group(point) || throw(ArgumentError(
        "nuts_state builds the compiled-reactive sampler and requires a flat " *
        "phase-point group from reactive_nuts_group(potential_gradient!, metric, " *
        "position, momentum); got a plain phase point. Build a group first, or " *
        "use the internal reference oracle _oracle_nuts_state for parity checks."))
    compiled_nuts_state(point; kwargs...)
end

_is_reactive_nuts_group(point::ReactivePhasePoint) =
    all(name -> name in keys(getfield(point, :handles)),
        (:gofwd, :min_dham, :init_pos, :fwd_pos, :bwd_pos, :dham))
