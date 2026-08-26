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

struct _KineticBundleOp end
@inline function (::_KineticBundleOp)(chol, momentum)
    velocity = chol \ momentum
    _Kinetic(0.5 * (logdet(chol) + dot(momentum, velocity)), velocity)
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
    cache.kinetic = 0.5 * (logdet(chol) + dot(momentum, cache.velocity))
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
    dham = value!(graph, :dham, typeof(ham0))
    add!(graph, (ports[:init_ham], active_ham) => dham,
         (init_ham, selected_ham) -> init_ham - selected_ham)
    diverged = value!(graph, :diverged, Bool)
    minimum_dham = Float64(min_dham)
    add!(graph, dham => diverged,
         energy_error -> !(energy_error >= minimum_dham))

    haves = (metric_value, gofwd_value,
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
    state = program(metric, gofwd,
                    copy(position), copy(momentum),
                    copy(position), copy(momentum),
                    copy(position), copy(momentum))

    handle_names = (:metric, :gofwd, :chol_metric,
                    :active_ham, :dham, :diverged,
                    (Symbol(endpoint, suffix)
                     for endpoint in _REACTIVE_NUTS_ENDPOINTS
                     for suffix in (:_pos, :_mom, :_pot, :_dpot_dpos,
                                    :_kin, :_dham_dmom, :_ham))...)
    handle_ports = (metric_value, gofwd_value, chol,
                    active_ham, dham, diverged,
                    (ports[Symbol(endpoint, suffix)]
                     for endpoint in _REACTIVE_NUTS_ENDPOINTS
                     for suffix in (:_pos, :_mom, :_pot, :_dpot_dpos,
                                    :_kin, :_dham_dmom, :_ham))...)
    values = NamedTuple{handle_names}(handle_ports)
    _phasepoint(program, state, values)
end
