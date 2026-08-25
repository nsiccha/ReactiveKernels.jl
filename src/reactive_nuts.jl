# Flat compiled-reactive NUTS phase-point group — increment 1 of the ca9
# ReactiveHMC port. Following poc's approved "(b) FLATTENED" verdict, the sampler
# state is ONE wide `ReactiveProgram` spanning the `init`, `fwd`, and `bwd`
# endpoints, with a `gofwd` HAVE source. The active-endpoint selection, the energy
# error `dham`, and the `diverged` flag are ordinary reactive recipes over this
# wider HAVE — no nested compiled states, no HMC-specific manual invalidation.
#
# ca9's `nuts_state.dham` is refreshed by hand (`@invalidatedependants!`) because
# ReactiveObjects leaves cross-object reactivity unsolved. Here `dham` is a
# genuine compiled reactive node: mutating any active endpoint's `mom`/`pos`
# invalidates `dham` and a later read recomputes minimally. Selection is a branch
# at compute time over static edges to BOTH endpoints, so flipping `gofwd`
# reselects with no graph rebuild.
#
# The group reuses the tested `ReactivePhasePoint{S,H}` runtime wrapper, so every
# node is reachable as a property: reactive outputs (`group.dham`, `group.fwd_ham`)
# read through `get!`, HAVE sources (`group.gofwd`, `group.fwd_mom`) write through
# `set!`, and the whole state stays `copy`/`copyto!`-able for proposal swaps.
#
# This constructor builds the program directly from the `Graph` API (the dev
# scaffold the design note permits); a later increment authors the same shape
# through the public `@reactive` facade. It is AD-agnostic: `potential_gradient(q)`
# is the caller-supplied scalar-potential boundary returning `(potential, gradient)`
# — the DI+Enzyme owned value/gradient bundle is wired in at the call site.

const _REACTIVE_NUTS_ENDPOINTS = (:init, :fwd, :bwd)
const _REACTIVE_NUTS_DEFAULT_MIN_DHAM = -1000.0

"""
    reactive_nuts_group(potential_gradient, metric, position, momentum;
                        gofwd = true, min_dham = -1000.0)

Build the flat compiled-reactive NUTS phase-point group. All three endpoints
(`init`, `fwd`, `bwd`) are initialized from `(position, momentum)` and share the
Cholesky factor of the fixed Euclidean `metric`. `potential_gradient(q)` must
return `(potential(q), gradient(q))`.

The returned [`ReactivePhasePoint`](@ref) exposes, per endpoint `e` in
`(:init, :fwd, :bwd)`, the HAVE sources `e_pos`, `e_mom` and the reactive nodes
`e_pot`, `e_dpot_dpos`, `e_kin`, `e_dham_dmom`, `e_ham`; plus the shared HAVE
sources `metric`, `gofwd`, and the reactive selection/diagnostic nodes
`active_ham` (`gofwd ? fwd_ham : bwd_ham`), `dham` (`init_ham - active_ham`), and
`diverged` (`!(dham >= min_dham)`).
"""
function reactive_nuts_group(potential_gradient, metric, position, momentum;
                             gofwd::Bool = true,
                             min_dham::Real = _REACTIVE_NUTS_DEFAULT_MIN_DHAM)
    potential0, gradient0 = potential_gradient(position)
    chol0 = cholesky(metric)
    velocity0 = chol0 \ momentum
    kinetic0 = 0.5 * (logdet(chol0) + dot(momentum, velocity0))
    ham0 = potential0 + kinetic0

    graph = Graph()
    metric_value = value!(graph, :metric, typeof(metric))
    gofwd_value = value!(graph, :gofwd, Bool)
    chol = value!(graph, :chol_metric, typeof(chol0))
    add!(graph, metric_value => chol, cholesky)

    # Per-endpoint Hamiltonian sub-graph: (pos -> pot, gradient) and
    # (chol, mom -> kinetic, velocity), joined into ham. Static edges; the
    # endpoints differ only by their HAVE sources.
    ports = Dict{Symbol,Any}()
    kinetic_recipe = (factor, p) -> begin
        velocity = factor \ p
        (0.5 * (logdet(factor) + dot(p, velocity)), velocity)
    end
    for endpoint in _REACTIVE_NUTS_ENDPOINTS
        pos = ports[Symbol(endpoint, :_pos)] =
            value!(graph, Symbol(endpoint, :_pos), typeof(position))
        mom = ports[Symbol(endpoint, :_mom)] =
            value!(graph, Symbol(endpoint, :_mom), typeof(momentum))
        pot = ports[Symbol(endpoint, :_pot)] =
            value!(graph, Symbol(endpoint, :_pot), typeof(potential0))
        dpot = ports[Symbol(endpoint, :_dpot_dpos)] =
            value!(graph, Symbol(endpoint, :_dpot_dpos), typeof(gradient0))
        kinetic = ports[Symbol(endpoint, :_kin)] =
            value!(graph, Symbol(endpoint, :_kin), typeof(kinetic0))
        dham_dmom = ports[Symbol(endpoint, :_dham_dmom)] =
            value!(graph, Symbol(endpoint, :_dham_dmom), typeof(velocity0))
        ham = ports[Symbol(endpoint, :_ham)] =
            value!(graph, Symbol(endpoint, :_ham), typeof(ham0))
        add!(graph, pos => (pot, dpot), potential_gradient)
        add!(graph, (chol, mom) => (kinetic, dham_dmom), kinetic_recipe)
        add!(graph, (pot, kinetic) => ham, +)
    end

    # Active-endpoint selection + energy error + divergence, all reactive recipes
    # over the wider HAVE. `gofwd` is a HAVE source; `active_ham` branches over the
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

    program = prepare_reactive(graph; have = haves, want = wants)
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
