# Stateful Hamiltonian phase points and the multinomial NUTS transition from
# ReactiveHMC.jl `src/nuts.jl` at main@ca9ea4ca41924bb0e1fadc01c717e1333916aba6.
# The file is byte-identical on dev@a8a33f958ab0dffb5696ce7da7fcdcdd6983c208.
# The sampler algorithm is intentionally kept separate from graph construction:
# every Hamiltonian field is supplied by a `ReactiveProgram`, while the
# integrators mutate only declared HAVE slots through the generic invalidation
# API.

"""
    ReactivePhasePoint

A mutable Hamiltonian phase point backed by a compiled [`ReactiveProgram`](@ref).
Reading a property such as `point.ham` lazily recomputes invalid dependencies.
Assignments and dotted in-place updates of declared source properties
automatically invalidate their downstream fields.
"""
struct ReactivePhasePoint{S,H}
    state::S
    handles::H
end

@inline Base.getproperty(point::ReactivePhasePoint, name::Symbol) =
    getproperty(point, Val(name))

@inline Base.getproperty(point::ReactivePhasePoint, ::Val{:state}) =
    getfield(point, :state)
@inline Base.getproperty(point::ReactivePhasePoint, ::Val{:handles}) =
    getfield(point, :handles)
@inline Base.getproperty(point::ReactivePhasePoint, ::Val{:program}) =
    getfield(point, :state).program

@generated function Base.getproperty(point::ReactivePhasePoint{S,H},
                                     ::Val{name}) where {S,H,name}
    name in fieldnames(H) || return :(getfield(point, $(QuoteNode(name))))
    quote
        handles = getfield(point, :handles)
        handle = getfield(handles, $(QuoteNode(name)))
        get!(getfield(point, :state), handle)
    end
end

@inline Base.setproperty!(point::ReactivePhasePoint, name::Symbol, value) =
    setproperty!(point, Val(name), value)

@generated function Base.setproperty!(point::ReactivePhasePoint{S,H},
                                      ::Val{name}, value) where {S,H,name}
    name in fieldnames(H) || return :(throw(ArgumentError(
        "ReactivePhasePoint has no mutable state property $name",
    )))
    quote
        handles = getfield(point, :handles)
        set!(getfield(point, :state),
             getfield(handles, $(QuoteNode(name))), value)
    end
end

Base.propertynames(point::ReactivePhasePoint, private::Bool = false) =
    private ? (:state, :handles, :program,
               propertynames(getfield(point, :handles))...) :
              (:program, propertynames(getfield(point, :handles))...)

struct _ReactivePhaseProperty{P,H}
    point::P
    handle::H
end

@inline Base.dotgetproperty(point::ReactivePhasePoint, name::Symbol) =
    Base.dotgetproperty(point, Val(name))

@generated function Base.dotgetproperty(point::ReactivePhasePoint{S,H},
                                        ::Val{name}) where {S,H,name}
    name in fieldnames(H) || return :(getproperty(point, Val(name)))
    :(_ReactivePhaseProperty(
        point,
        getfield(getfield(point, :handles), $(QuoteNode(name))),
    ))
end

function Base.materialize!(destination::_ReactivePhaseProperty,
                           broadcasted::Base.Broadcast.Broadcasted)
    point = destination.point
    mutate!(point.state, destination.handle) do value
        Base.materialize!(value, broadcasted)
    end
end

@inline mutate!(f, point::ReactivePhasePoint, name::Symbol) =
    mutate!(f, point, Val(name))

@generated function mutate!(f, point::ReactivePhasePoint{S,H},
                            ::Val{name}) where {S,H,name}
    name in fieldnames(H) || return :(throw(ArgumentError(
        "ReactivePhasePoint has no state property $name",
    )))
    quote
        handles = getfield(point, :handles)
        mutate!(f, getfield(point, :state),
                getfield(handles, $(QuoteNode(name))))
    end
end

reactive_program(point::ReactivePhasePoint) = point.state.program
plan(point::ReactivePhasePoint) = point.state.program.plan
code_expr(point::ReactivePhasePoint, name::Symbol) =
    code_expr(point.state.program, getproperty(point.handles, name))

Base.copy(point::ReactivePhasePoint) =
    ReactivePhasePoint(copy(point.state), point.handles)

function Base.copyto!(destination::ReactivePhasePoint,
                      source::ReactivePhasePoint)
    destination.handles === source.handles || throw(ArgumentError(
        "phase points belong to different ReactivePrograms",
    ))
    copyto!(destination.state, source.state)
    destination
end

_tr_prod(a::AbstractMatrix, b::AbstractMatrix) = sum(a' .* b)

function _phasepoint(program, state, values)
    handles = map(value -> statevalue(program, value), values)
    ReactivePhasePoint(state, handles)
end

"""
    euclidean_phasepoint(potential, potential_gradient, metric, position, momentum)

Prepare a Gaussian-kinetic Euclidean phase point. `potential_gradient(q)` must
return `(potential(q), gradient(q))`. The returned point exposes `pos`, `mom`,
`metric`, `pot`, `dpot_dpos`, `chol_metric`, `kin`, `ham`, `dham_dpos`, and
`dham_dmom`; its `program`, selected `plan`, and generated getter expressions
remain inspectable.

This is the ReactiveKernels port of ReactiveHMC.jl's public phase-point
contract. The combined oracle is selected whenever a gradient is required, so
its potential output is shared rather than recomputed.
"""
function euclidean_phasepoint(potential, potential_gradient, metric,
                              position, momentum)
    potential0, gradient0 = potential_gradient(position)
    chol0 = cholesky(metric)
    velocity0 = chol0 \ momentum
    kinetic0 = 0.5 * (logdet(chol0) + dot(momentum, velocity0))

    graph = Graph()
    pos = value!(graph, :pos, typeof(position))
    mom = value!(graph, :mom, typeof(momentum))
    metric_value = value!(graph, :metric, typeof(metric))
    pot = value!(graph, :pot, typeof(potential0))
    dpot = value!(graph, :dpot_dpos, typeof(gradient0))
    chol = value!(graph, :chol_metric, typeof(chol0))
    kinetic = value!(graph, :kin, typeof(kinetic0))
    dham_dmom = value!(graph, :dham_dmom, typeof(velocity0))
    ham = value!(graph, :ham, typeof(potential0 + kinetic0))

    add!(graph, pos => pot, potential)
    add!(graph, pos => (pot, dpot), potential_gradient)
    add!(graph, metric_value => chol, cholesky)
    add!(graph, (chol, mom) => (kinetic, dham_dmom), (factor, p) -> begin
        velocity = factor \ p
        (0.5 * (logdet(factor) + dot(p, velocity)), velocity)
    end)
    add!(graph, (pot, kinetic) => ham, +)

    program = prepare_reactive(
        graph;
        have = (pos, mom, metric_value),
        want = (pot, dpot, chol, kinetic, ham, dham_dmom),
    )
    state = program(position, momentum, metric)
    values = (;
        pos,
        mom,
        metric = metric_value,
        pot,
        dpot_dpos = dpot,
        chol_metric = chol,
        kin = kinetic,
        ham,
        dham_dpos = dpot,
        dham_dmom,
    )
    _phasepoint(program, state, values)
end

function _riemannian_dpos(momentum, chol, inverse_metric,
                          metric_gradient, potential_gradient)
    velocity = chol \ momentum
    kinetic_gradient = map(eachslice(metric_gradient; dims = 3)) do partial_metric
        0.5 * _tr_prod(inverse_metric, partial_metric) -
            0.5 * dot(velocity, partial_metric, velocity)
    end
    kinetic_gradient + potential_gradient
end

"""
    riemannian_phasepoint(potential, potential_gradient, metric,
                          metric_gradient, position, momentum)

Prepare the position-dependent Gaussian-kinetic phase point from ReactiveHMC.
`metric(q)` returns `(potential, gradient, metric)` and
`metric_gradient(q)` returns those values plus the three-dimensional metric
gradient tensor. Use [`generalized_leapfrog!`](@ref) or
[`implicit_midpoint!`](@ref).
"""
function riemannian_phasepoint(potential, potential_gradient, metric,
                               metric_gradient, position, momentum)
    potential0, gradient0, metric0, metric_gradient0 = metric_gradient(position)
    chol0 = cholesky(metric0)
    inverse0 = Symmetric(inv(chol0))
    velocity0 = chol0 \ momentum
    kinetic0 = 0.5 * (logdet(chol0) + dot(momentum, velocity0))
    dpos0 = _riemannian_dpos(
        momentum, chol0, inverse0, metric_gradient0, gradient0,
    )

    graph = Graph()
    pos = value!(graph, :pos, typeof(position))
    mom = value!(graph, :mom, typeof(momentum))
    pot = value!(graph, :pot, typeof(potential0))
    dpot = value!(graph, :dpot_dpos, typeof(gradient0))
    metric_value = value!(graph, :metric, typeof(metric0))
    metric_grad = value!(graph, :metric_grad, typeof(metric_gradient0))
    chol = value!(graph, :chol_metric, typeof(chol0))
    inverse_metric = value!(graph, :inv_metric, typeof(inverse0))
    kinetic = value!(graph, :kin, typeof(kinetic0))
    dham_dpos = value!(graph, :dham_dpos, typeof(dpos0))
    dham_dmom = value!(graph, :dham_dmom, typeof(velocity0))
    ham = value!(graph, :ham, typeof(potential0 + kinetic0))

    add!(graph, pos => pot, potential)
    add!(graph, pos => (pot, dpot), potential_gradient)
    add!(graph, pos => (pot, dpot, metric_value), metric)
    add!(graph, pos => (pot, dpot, metric_value, metric_grad), metric_gradient)
    add!(graph, metric_value => chol, cholesky)
    add!(graph, chol => inverse_metric, factor -> Symmetric(inv(factor)))
    add!(graph, (chol, mom) => (kinetic, dham_dmom), (factor, p) -> begin
        velocity = factor \ p
        (0.5 * (logdet(factor) + dot(p, velocity)), velocity)
    end)
    add!(graph,
         (mom, chol, inverse_metric, metric_grad, dpot) => dham_dpos,
         _riemannian_dpos)
    add!(graph, (pot, kinetic) => ham, +)

    program = prepare_reactive(
        graph;
        have = (pos, mom),
        want = (
            pot, dpot, metric_value, metric_grad, chol, inverse_metric,
            kinetic, ham, dham_dpos, dham_dmom,
        ),
    )
    state = program(position, momentum)
    values = (;
        pos,
        mom,
        pot,
        dpot_dpos = dpot,
        metric = metric_value,
        metric_grad,
        chol_metric = chol,
        inv_metric = inverse_metric,
        kin = kinetic,
        ham,
        dham_dpos,
        dham_dmom,
    )
    _phasepoint(program, state, values)
end

"One in-place Störmer-Verlet step with automatic dependency invalidation."
function leapfrog!(phasepoint; stepsize)
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
    @. phasepoint.pos += stepsize * phasepoint.dham_dmom
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
    phasepoint
end

"Generalized leapfrog for position-dependent metrics."
function generalized_leapfrog!(phasepoint; stepsize, n_fi_steps)
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps
        @. phasepoint.mom = mom0 - 0.5 * stepsize * phasepoint.dham_dpos
    end
    dham_dmom0 = copy(phasepoint.dham_dmom)
    for _ in 1:n_fi_steps
        @. phasepoint.pos = pos0 +
            0.5 * stepsize * (dham_dmom0 + phasepoint.dham_dmom)
    end
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
    phasepoint
end

"Implicit midpoint integrator for non-separable Hamiltonians."
function implicit_midpoint!(phasepoint; stepsize, n_fi_steps)
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps
        dham_dmom, dham_dpos = phasepoint.dham_dmom, phasepoint.dham_dpos
        @. phasepoint.pos = pos0 + 0.5 * stepsize * dham_dmom
        @. phasepoint.mom = mom0 - 0.5 * stepsize * dham_dpos
    end
    @. phasepoint.pos = 2 * phasepoint.pos - pos0
    @. phasepoint.mom = 2 * phasepoint.mom - mom0
    phasepoint
end

function multistep(f, args...; n_steps, stepsize, kwargs...)
    for _ in 1:n_steps
        f(args...; stepsize = stepsize / n_steps, kwargs...)
    end
    first(args)
end

multistep(f; n_steps) =
    (args...; kwargs...) -> multistep(f, args...; n_steps, kwargs...)

"Partially applied callable used for sampler integrators and adaptation."
struct PartialFunction{F,L<:Tuple,R<:Tuple,K<:NamedTuple} <: Function
    func::F
    largs::L
    rargs::R
    kwargs::K
end

(f::PartialFunction)(args...; kwargs...) =
    f.func(f.largs..., args..., f.rargs...; f.kwargs..., kwargs...)

function Base.getproperty(f::PartialFunction, name::Symbol)
    name in fieldnames(typeof(f)) && return getfield(f, name)
    getproperty(getfield(f, :kwargs), name)
end

partial(f, args...; kwargs...) = PartialFunction(f, args, (), (; kwargs...))
partial(f, ::Colon, args...; kwargs...) =
    PartialFunction(f, (), args, (; kwargs...))
partial(f, left, ::Colon, args...; kwargs...) =
    PartialFunction(f, (left,), args, (; kwargs...))

_finite_or_neginf(x) = isfinite(x) ? x : typeof(x)(-Inf)
_min1exp(x) = x >= 0 ? one(x) : exp(x)
_rand_bernoulli_log(rng, log_probability) =
    log_probability > 0 ? true : -randexp(rng) < log_probability

_momentum_velocity(momentum, velocity) = (; momentum, velocity)
_trajectory(backward, forward) = (; backward, forward)

function _nuts_tree(point)
    prototype = point.mom
    scalar = eltype(prototype)
    zeros_like() = zeros(scalar, size(prototype))
    (
        log_weight = fill(scalar(-Inf), 2),
        backward = _momentum_velocity(zeros_like(), zeros_like()),
        backward_forward = _momentum_velocity(zeros_like(), zeros_like()),
        summed_momentum = _trajectory(zeros_like(), zeros_like()),
    )
end

"Diagnostics for one completed [`NUTSState`](@ref) transition."
struct NUTSDiagnostics{T}
    depth::Int
    n_steps::Int
    acceptance_rate::T
    diverged::Bool
    energy_error::T
end

"""
    NUTSState

Mutable multinomial No-U-Turn sampler state faithfully ported from
[`ReactiveHMC.jl/src/nuts.jl`](https://github.com/nsiccha/ReactiveHMC.jl/blob/ca9ea4ca41924bb0e1fadc01c717e1333916aba6/src/nuts.jl)
at `main@ca9ea4ca41924bb0e1fadc01c717e1333916aba6`; that source is
byte-identical at `dev@a8a33f958ab0dffb5696ce7da7fcdcdd6983c208`.
The transition uses the generalized endpoint-momentum U-turn criterion from
that implementation. Hamiltonian values and derivatives are supplied lazily
by the phase point's compiled ReactiveKernels graph.
"""
mutable struct NUTSState{R,P,F,S,T,TR,PR}
    rng::R
    init::P
    step_f::F
    stats_f::S
    max_depth::Int
    min_energy_error::T
    endpoints::NTuple{2,P}
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

function Base.getproperty(state::NUTSState, name::Symbol)
    name === :fwd && return _forward(state)
    name === :bwd && return _backward(state)
    name === :dham && return getfield(state, :energy_error)
    name === :gofwd && return getfield(state, :go_forward)
    getfield(state, name)
end

_forward(state::NUTSState) =
    state.endpoints[state.go_forward ? 1 : 2]
_backward(state::NUTSState) =
    state.endpoints[state.go_forward ? 2 : 1]

"""
    nuts_state(init; rng, step_f, stats_f=nothing,
               max_depth=10, min_dham=-1000)

Create the ReactiveHMC-compatible multinomial NUTS transition. `step_f` is an
integrator callable such as `partial(leapfrog!; stepsize=0.25)`. A low-level
[`step!`](@ref) uses the momentum currently stored in `init`; [`sample!`](@ref)
refreshes momentum before each transition and is the convenient sampling API.
"""
function nuts_state(init::ReactivePhasePoint; rng,
                    step_f,
                    stats_f = nothing,
                    max_depth::Integer = 10,
                    min_dham = -1000.0)
    max_depth >= 1 || throw(ArgumentError("max_depth must be positive"))
    endpoints = (copy(init), copy(init))
    trees = [_nuts_tree(init) for _ in 1:(max_depth + 1)]
    proposals = [copy(init) for _ in 1:(max_depth + 2)]
    scalar = typeof(float(init.ham - init.ham))
    NUTSState(
        rng,
        init,
        step_f,
        stats_f,
        Int(max_depth),
        convert(scalar, min_dham),
        endpoints,
        trees,
        proposals,
        true,
        true,
        true,
        zero(scalar),
        false,
        0,
        0,
        zero(scalar),
    )
end

function _reset_tree!(tree)
    fill!(tree.log_weight, -Inf)
    fill!(tree.backward.momentum, 0)
    fill!(tree.backward.velocity, 0)
    fill!(tree.backward_forward.momentum, 0)
    fill!(tree.backward_forward.velocity, 0)
    fill!(tree.summed_momentum.backward, 0)
    fill!(tree.summed_momentum.forward, 0)
    tree
end

function _reset_transition!(state::NUTSState)
    copyto!(state.endpoints[1], state.init)
    copyto!(state.endpoints[2], state.init)
    for proposal in state.proposals
        copyto!(proposal, state.init)
    end
    foreach(_reset_tree!, state.trees)
    state.go_forward = true
    state.may_sample = true
    state.may_continue = true
    state.energy_error = zero(state.energy_error)
    state.diverged = false
    state.depth = 0
    state.n_steps = 0
    state.acceptance_sum = zero(state.acceptance_sum)
    state
end

function _compute_criterion(momentum, backward_velocity, forward_velocity)
    dot(momentum, backward_velocity) > 0 &&
        dot(momentum, forward_velocity) > 0
end

_compute_criterion_sum(left_momentum, right_momentum,
                       backward_velocity, forward_velocity) =
    _compute_criterion(
        Base.broadcasted(+, left_momentum, right_momentum),
        backward_velocity,
        forward_velocity,
    )

function _swap_proposal!(state::NUTSState, first_index::Int,
                         second_index::Int = length(state.proposals))
    state.proposals[first_index], state.proposals[second_index] =
        state.proposals[second_index], state.proposals[first_index]
    state
end

function _flip!(state::NUTSState, depth::Int)
    depth > 1 || return state
    state.go_forward = !state.go_forward
    tree = state.trees[depth]
    backward = _backward(state)
    @. tree.backward.momentum = -backward.mom
    @. tree.backward.velocity = -backward.dham_dmom
    @. tree.summed_momentum.forward *= -1
    state
end

function _collect_stats!(state::NUTSState)
    state.stats_f === nothing || state.stats_f(state)
    state
end

function _start_tree!(state::NUTSState, depth::Int)
    if depth == 1
        state.step_f(_forward(state))
        state.n_steps += 1
        state.energy_error = _finite_or_neginf(state.init.ham - _forward(state).ham)
        state.acceptance_sum += _min1exp(state.energy_error)
        state.diverged = !(state.energy_error >= state.min_energy_error)
        _collect_stats!(state)
        if state.diverged
            state.may_continue = false
            return state
        end
        state.trees[1].log_weight[1] = state.energy_error
        copyto!(state.proposals[1], _forward(state))
        return state
    end

    _start_tree!(state, depth - 1)
    if !state.may_continue
        state.may_sample = false
        return state
    end
    _swap_proposal!(state, depth - 1, depth)
    _finish_tree!(state, depth - 1)
    if state.may_sample && _rand_bernoulli_log(
            state.rng,
            state.trees[depth - 1].log_weight[1] -
                state.trees[depth].log_weight[1],
        )
        _swap_proposal!(state, depth - 1, depth)
    end
    state
end

function _finish_tree!(state::NUTSState, depth::Int)
    tree = state.trees[depth]
    supertree = state.trees[depth + 1]
    tree.log_weight[2] = tree.log_weight[1]
    forward = _forward(state)

    if depth == 1
        copyto!(supertree.backward.momentum, forward.mom)
        copyto!(supertree.backward.velocity, forward.dham_dmom)
    else
        copyto!(supertree.backward.momentum, tree.backward.momentum)
        copyto!(supertree.backward.velocity, tree.backward.velocity)
        copyto!(tree.backward_forward.momentum, forward.mom)
        copyto!(tree.backward_forward.velocity, forward.dham_dmom)
        copyto!(tree.summed_momentum.backward, tree.summed_momentum.forward)
    end

    _start_tree!(state, depth)
    if !state.may_continue
        state.may_sample = false
        return state
    end

    supertree.log_weight[1] = logaddexp(
        tree.log_weight[1], tree.log_weight[2],
    )
    if depth == 1
        @. supertree.summed_momentum.forward =
            supertree.backward.momentum + forward.mom
        state.may_continue = _compute_criterion(
            supertree.summed_momentum.forward,
            supertree.backward.velocity,
            forward.dham_dmom,
        )
    else
        @. supertree.summed_momentum.forward =
            tree.summed_momentum.backward + tree.summed_momentum.forward
        state.may_continue =
            _compute_criterion(
                supertree.summed_momentum.forward,
                supertree.backward.velocity,
                forward.dham_dmom,
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
                forward.dham_dmom,
            )
    end
    state
end

"Advance one multinomial NUTS transition using the momentum already in `state.init`."
function step!(state::NUTSState)
    _reset_transition!(state)
    backward = _backward(state)
    @. backward.mom *= -1
    state.trees[1].log_weight[1] = 0

    for depth in 1:state.max_depth
        rand(state.rng, Bool) && _flip!(state, depth)
        _finish_tree!(state, depth)
        state.depth = depth
        state.may_sample || break
        if _rand_bernoulli_log(
                state.rng,
                state.trees[depth].log_weight[1] -
                    state.trees[depth].log_weight[2],
            )
            _swap_proposal!(state, depth)
        end
        state.may_continue || break
    end
    copyto!(state.init, state.proposals[end])
    diagnostics(state)
end

function diagnostics(state::NUTSState)
    acceptance = state.n_steps == 0 ? zero(state.acceptance_sum) :
                 state.acceptance_sum / state.n_steps
    NUTSDiagnostics(
        state.depth,
        state.n_steps,
        acceptance,
        state.diverged,
        state.energy_error,
    )
end

"Draw a fresh Gaussian momentum appropriate for the phase point metric."
function refresh_momentum!(state::NUTSState)
    point = state.init
    factor = point.chol_metric
    mutate!(point, :mom) do momentum
        randn!(state.rng, momentum)
        lmul!(factor.L, momentum)
    end
    point
end

"Refresh momentum and run one NUTS transition."
function sample!(state::NUTSState)
    refresh_momentum!(state)
    step!(state)
end

"""
    sample!(state, draws; discard_initial=0)

Run a usable NUTS chain, returning a named tuple with a dense `samples` matrix
(parameters × draws) and one [`NUTSDiagnostics`](@ref) per retained draw.
"""
function sample!(state::NUTSState, draws::Integer; discard_initial::Integer = 0)
    draws >= 0 || throw(ArgumentError("draws must be non-negative"))
    discard_initial >= 0 || throw(ArgumentError(
        "discard_initial must be non-negative",
    ))
    for _ in 1:discard_initial
        sample!(state)
    end
    position = state.init.pos
    samples = Matrix{eltype(position)}(undef, length(position), draws)
    stats = Vector{NUTSDiagnostics{typeof(state.energy_error)}}(undef, draws)
    for draw in 1:draws
        stats[draw] = sample!(state)
        samples[:, draw] .= state.init.pos
    end
    (; samples, diagnostics = stats)
end

"Mutable Nesterov dual-averaging state matching ReactiveHMC's adaptation rule."
mutable struct DualAveragingState{T}
    target::T
    regularization_scale::T
    relaxation_exponent::T
    offset::T
    iteration::T
    error::T
    center::T
    log_current::T
    log_final::T
    current::T
    final::T
end

function dual_averaging_state(initial;
                              target = 0.8,
                              regularization_scale = 0.05,
                              relaxation_exponent = 0.75,
                              offset = 10)
    T = typeof(float(initial))
    iteration = one(T)
    error = zero(T)
    center = log(T(10)) + log(T(initial))
    log_current = center
    log_final = zero(T)
    DualAveragingState(
        T(target), T(regularization_scale), T(relaxation_exponent), T(offset),
        iteration, error, center, log_current, log_final,
        exp(log_current), exp(log_final),
    )
end

function fit!(state::DualAveragingState, acceptance_rate)
    state.iteration += 1
    state.error += (state.target - acceptance_rate - state.error) /
                   (state.iteration + state.offset)
    state.log_current = state.center -
        sqrt(state.iteration) / state.regularization_scale * state.error
    weight = state.iteration^(-state.relaxation_exponent)
    state.log_final += weight * (state.log_current - state.log_final)
    state.current = exp(state.log_current)
    state.final = exp(state.log_final)
    state
end

"Online componentwise variance estimate using ReactiveHMC's Welford update."
mutable struct WelfordVariance{T,V}
    n::T
    mean::V
    var::V
end

welford_var(dimension::Integer, ::Type{T} = Float64) where {T} =
    WelfordVariance(zero(T), zeros(T, dimension), zeros(T, dimension))

_smooth(previous, new, weight) = (1 - weight) * previous + weight * new

function step!(state::WelfordVariance, value::AbstractVector; weight = 1)
    state.n += weight
    fraction = weight / state.n
    @. state.var = _smooth(
        state.var,
        (value - _smooth(state.mean, value, fraction)) * (value - state.mean),
        fraction,
    )
    @. state.mean = _smooth(state.mean, value, fraction)
    state
end

function step!(state::WelfordVariance, values::AbstractMatrix; kwargs...)
    for value in eachcol(values)
        step!(state, value; kwargs...)
    end
    state
end
