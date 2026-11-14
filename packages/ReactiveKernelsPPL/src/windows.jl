# Windows of one recurrence, fitted apart and joined through their boundary
# state (multiple shooting, Bock & Plitt 1984). Each window is a binding of one
# program to its own rows. A window's recurrence starts from a boundary input,
# a declared parameter that its `@scan` seeds from, and ends at a boundary
# output, an authored value. Joining substitutes each window's output for the
# next window's input and counts every shared coordinate's prior once, so the
# joined windows are the model of the whole series.

"""
    JoinedWindows

Windows of one recurrence joined through their boundary state, from
[`prepare_joined_windows`](@ref). Evaluate with
[`joined_value_and_gradient!`](@ref) or call it on a joined point; its
coordinates are [`joined_coordinates`](@ref). Not thread-safe: prepare one per
thread, as for `SamplerQuery`.
"""
struct JoinedWindows{S,D,O,V}
    built::Any
    plans::Vector{StructuralPlan}
    layouts::Vector{LayoutTable}
    boundary::Vector{Pair{Symbol,Symbol}}
    samplers::Vector{S}
    # Per window: the prior and Jacobian terms the joined model counts in an
    # earlier window or drops with a fed input, or `nothing`.
    dropped::Vector{D}
    # Per window but the last: its boundary outputs' value kernel and pullback,
    # or `nothing` when the next window takes nothing from it, and the shape of
    # each output (`nothing` for a scalar).
    values::Vector{V}
    pullbacks::Vector{O}
    shapes::Vector{Vector{Any}}
    # Per window: the joined position of each packed coordinate, 0 for a fed one.
    slots::Vector{Vector{Int}}
    # Per window: (packed position, boundary pair, output entry) of each fed
    # input coordinate, and the layout entry whose transform it takes.
    fed::Vector{Vector{NTuple{3,Int}}}
    fed_entries::Vector{Vector{LayoutEntry}}
    coordinates::Vector{@NamedTuple{window::Int, position::Int,
        path::Tuple{Vararg{Symbol}}, index::Tuple}}
    # Workspaces.
    points::Vector{Vector{Float64}}
    gradients::Vector{Vector{Float64}}
    work::Vector{Vector{Float64}}
    scale::Vector{Vector{Float64}}
end

"""
    prepare_joined_windows(built, plans, u0s; boundary, backend,
                           locals = (), on_error = nothing) -> JoinedWindows

Join windows of one recurrence through their boundary state. `plans` are
bindings of the program `built` was made from, one per window in time order
(each window's rows, one build for every window); `u0s` are exemplar packed
points of each. `boundary` is one pair `output => input` or a collection of
them:

- `output` names an authored value of a window, its state at the window's end
  (`h_out = h[T]`, or a per-index output of a `@plate` of recurrences);
- `input` names a declared parameter that the window's recurrence starts from
  (`h_in ~ Normal(0, 2)` read by the seed `h[1] = h_in`), a scalar or an array
  with an elementwise support.

Entry `j` of a window's output is the end of the trajectory that starts at
entry `j` of the same window's input, so an array output has the input's length
and order. The joined model feeds window `k`'s output entry to the entry of
window `k + 1`'s input with the same [`coordinate_identities`](@ref) (a level
label of `levels(g)`, or the position): a fed input entry is no longer a
coordinate, and its prior and Jacobian are dropped. An input entry no earlier
trajectory reaches (a unit that starts in this window) keeps its coordinate and
its prior.

The joined log density is the sum of the windows' log densities with

- every fed input entry set to the previous window's output, mapped through its
  support's unconstraining transform, and its prior and Jacobian terms dropped;
- every coordinate that an earlier window holds, by identity, being that
  window's coordinate, with its prior and Jacobian counted only there (a global
  such as a step scale, or a unit's static matched by its level label).

Coordinates of a `@scan`'s sampled steps, the boundary inputs and the
parameters named in `locals` belong to their window only: position `j` of a
scan is another time step in another window. Name in `locals` any other
parameter whose positional entries are per-window (innovations declared
outside the scan, a plate of recurrences' draws).

Windows that cover a series one after another, each starting from the previous
window's end state, then have the series' log density and gradient: with
`h_in` the state before the series and `h_out` the state at its last step, two
windows reproduce one window over all rows. Gradients flow back through each
output into the window that produced it. A prior that a later window computes
from its own rows differently (a scale from all its rows) is counted as the
first window holding the coordinate computes it.

Use [`joined_point`](@ref) to pack the window points and
[`joined_value_and_gradient!`](@ref) to evaluate their joint density.
[`window_points`](@ref) recovers each window point with its boundary inputs;
[`joined_coordinates`](@ref) identifies the retained coordinates.
"""
function prepare_joined_windows(built, plans::AbstractVector{<:StructuralPlan},
        u0s::AbstractVector{<:AbstractVector{<:Real}}; boundary, backend, locals = (),
        on_error = nothing)
    K = length(plans)
    K >= 1 || throw(ContractValidationError("[windows] at least one window is needed"))
    length(u0s) == K || throw(ContractValidationError(
        "[windows] $(length(u0s)) exemplar points for $K windows"))
    pairs = _window_boundary(boundary)
    local_names = Set{Symbol}(Symbol[Symbol(n) for n in locals])
    builds = [_binding_build(built, plan) for plan in plans]
    layouts = LayoutTable[b.layout for b in builds]
    exemplars = Vector{Float64}[Vector{Float64}(u) for u in u0s]
    for k in 1:K
        length(exemplars[k]) == layouts[k].total || throw(ContractValidationError(
            "[windows] window $k: exemplar length $(length(exemplars[k])) ≠ layout " *
            "total $(layouts[k].total)"))
    end
    inputs = [[_window_input_entry(layouts[k], input, k) for (_, input) in pairs] for k in 1:K]
    ids = [coordinate_identities(layout) for layout in layouts]
    samplers = [prepare_sampler(builds[k], plans[k], exemplars[k]; backend, on_error)
        for k in 1:K]

    # Each window's output kernels, first evaluated to check their lengths.
    outputs = Tuple(first.(pairs))
    values = Any[nothing for _ in 1:K]
    shapes = [Any[] for _ in 1:K]
    for k in 1:(K - 1)
        values[k] = _window_output_kernel(builds[k], plans[k], outputs, k; on_error)
        out = _window_outputs(Base.invokelatest(values[k], exemplars[k]), length(pairs))
        for (p, (output, input)) in enumerate(pairs)
            o = out[p]
            (o isa Real || (o isa AbstractArray && eltype(o) <: Real)) ||
                throw(ContractValidationError(
                    "[windows] window $k: boundary output $output must be a number " *
                    "or an array of numbers; got $(summary(o))"))
            n = length(o)
            n == inputs[k][p].size || throw(ContractValidationError(
                "[windows] window $k: boundary output $output has $n entries, but " *
                "its input $input has $(inputs[k][p].size); entry j of the output " *
                "is the end of the trajectory starting at entry j of the input"))
            push!(shapes[k], o isa Real ? nothing : size(o))
        end
    end

    # Joined positions: an input entry fed by the previous window's output is
    # no coordinate; a shared coordinate is the first holder's.
    slots = [zeros(Int, layout.total) for layout in layouts]
    fed = [NTuple{3,Int}[] for _ in 1:K]
    fed_entries = [LayoutEntry[] for _ in 1:K]
    coordinates = @NamedTuple{window::Int, position::Int, path::Tuple{Vararg{Symbol}},
        index::Tuple}[]
    seen = Dict{Any,Int}()
    for k in 1:K
        layout = layouts[k]
        input_positions = Set{Int}()
        for (p, e) in enumerate(inputs[k])
            union!(input_positions, e.offset:(e.offset + e.size - 1))
            k == 1 && continue
            previous = inputs[k - 1][p]
            index = Dict(ids[k - 1][previous.offset + j - 1] => j for j in 1:previous.size)
            for i in e.offset:(e.offset + e.size - 1)
                j = get(index, ids[k][i], 0)
                j == 0 && continue
                push!(fed[k], (i, p, j))
                push!(fed_entries[k], e)
            end
        end
        fed_positions = Set(first.(fed[k]))
        windowed = _windowed_positions(layout, local_names)
        for i in 1:layout.total
            i in fed_positions && continue
            id = ids[k][i]
            shared = !(i in input_positions) && !(i in windowed)
            if shared && haskey(seen, id)
                slots[k][i] = seen[id]
                continue
            end
            push!(coordinates, (; window = k, position = i, id.path, id.index))
            slots[k][i] = length(coordinates)
            shared && (seen[id] = length(coordinates))
        end
    end

    # A coordinate's prior and Jacobian count in the window that holds it first.
    dropped = Any[nothing for _ in 1:K]
    for k in 2:K
        own = Set(slots[k][i] for i in 1:layouts[k].total if slots[k][i] != 0 &&
            coordinates[slots[k][i]].window == k)
        events = _window_dropped_events(builds[k], plans[k], exemplars[k], slots[k], own, k)
        isempty(events.factors) && continue
        dropped[k] = only(Base.invokelatest(prepare_query_ad, prepare_ad, builds[k], plans[k],
            [events], backend, exemplars[k]; on_error))
    end

    # The pullback of each output the next window reads.
    pullbacks = Any[nothing for _ in 1:K]
    for k in 1:(K - 1)
        isempty(fed[k + 1]) && continue
        prepared = Base.invokelatest(prepare_ad_pullback, values[k], backend,
            _window_cotangent(_window_seed(shapes[k])), exemplars[k]; active = :unconstrained)
        pullbacks[k] = QueryAD(prepared, layouts[k])
    end
    for k in 1:(K - 1)
        isempty(fed[k + 1]) && (values[k] = nothing)
    end

    JoinedWindows(built, collect(StructuralPlan, plans), layouts, pairs, samplers, dropped,
        values, pullbacks, shapes, slots, fed, fed_entries, coordinates,
        [copy(u) for u in exemplars], [zeros(layout.total) for layout in layouts],
        [zeros(layout.total) for layout in layouts],
        [zeros(layout.total) for layout in layouts])
end

_window_boundary(p::Pair) = [Symbol(p.first) => Symbol(p.second)]
function _window_boundary(pairs)
    result = Pair{Symbol,Symbol}[]
    for p in pairs
        p isa Pair || throw(ContractValidationError(
            "[windows] boundary must be `output => input` pairs; got $(repr(p))"))
        push!(result, Symbol(p.first) => Symbol(p.second))
    end
    isempty(result) && throw(ContractValidationError("[windows] no boundary pair given"))
    allunique(last.(result)) || throw(ContractValidationError(
        "[windows] each boundary input takes one output"))
    result
end

# The layout entry of a boundary input: a declared scalar or array parameter
# whose coordinates are its entries, one by one.
function _window_input_entry(layout::LayoutTable, input::Symbol, k::Int)
    i = findfirst(e -> e.name === input, layout.entries)
    i === nothing && throw(ContractValidationError(
        "[windows] window $k: the boundary input $input is not a declared parameter " *
        "of this binding; its parameters are " *
        join(unique(e.name for e in layout.entries), ", ")))
    e = layout.entries[i]
    elementwise = (e.kind === :sampled && e.size == 1) || (e.kind === :array &&
        !_is_slice_transform(e.transform) && !_is_lkj_stack(e.transform) &&
        prod(e.dims; init = 1) == e.size && !haskey(layout.cell_latents, e.name) &&
        !haskey(layout.cell_geometries, e.name))
    elementwise && e.transform in (:identity, :exp, :logistic, :interval, :floored, :upper, :bounded) ||
        throw(ContractValidationError(
            "[windows] window $k: the boundary input $input must be a scalar or an array " *
            "of independent entries (support transform $(e.transform), kind $(e.kind))"))
    _dynamic_bounds(e) && throw(ContractValidationError(
        "[windows] window $k: the boundary input $input has bounds that read the " *
        "model's values; a fed entry needs fixed bounds"))
    e
end

# The boundary outputs of a window as one kernel returning their tuple.
function _window_output_kernel(built, plan, outputs, k; on_error)
    try
        _prepare_wants(built, plan, outputs; on_error)
    catch err
        err isa ContractValidationError && rethrow()
        throw(ContractValidationError("[windows] window $k: the boundary outputs " *
            "$(join(outputs, ", ")) must be authored values of the model: " *
            sprint(showerror, err)))
    end
end

# A cotangent per boundary output: a number for a scalar, else an array.
_window_seed(shapes) = Tuple(s === nothing ? 0.0 : zeros(s) for s in shapes)
# One WANT is its value itself, several a tuple (`ReactiveKernels.prepare`), and
# a pullback's cotangent takes the same shape.
_window_outputs(out, n) = n == 1 ? (out,) : out
_window_cotangent(seed::Tuple) = length(seed) == 1 ? only(seed) : seed

# Coordinates that belong to their window alone: a scan's sampled seeds and
# steps (at top level a `:scan` block, in a `@plate` the draw arrays its cell
# scan allocates, `_rkppl_scan_<state>_…`) and the parameters named `locals`.
function _windowed_positions(layout::LayoutTable, names::Set{Symbol})
    positions = Set{Int}()
    for e in layout.entries
        (e.kind === :scan || startswith(string(e.name), "_rkppl_scan_") ||
            e.name in names || get(layout.name_paths, e.name, (e.name,))[1] in names) ||
            continue
        union!(positions, e.offset:(e.offset + e.size - 1))
    end
    positions
end

# The prior and Jacobian events of window `k` that the joined model counts
# elsewhere: those owning coordinates an earlier window holds or a fed input,
# and those owning none that read none of this window's own coordinates.
function _window_dropped_events(built, plan, u0, slots, own, k)
    catalogue = density_factors(built, plan)
    dropped = DensityFactor[]
    for event in catalogue
        owned = factor_coordinates(catalogue, event)
        if isempty(owned)
            # A density of the earlier windows' or the fed inputs' values alone.
            support = factor_support(catalogue, event; u0)
            isempty(support) || any(i -> slots[i] in own, support) ||
                push!(dropped, event)
            continue
        end
        kept = count(i -> slots[i] in own, owned)
        kept == length(owned) && continue
        kept == 0 || throw(ContractValidationError(
            "[windows] window $k: the authored density $event owns coordinates of this " *
            "window and of an earlier one (or a fed boundary input), so it cannot be " *
            "counted once; keep a joint law over shared and per-window coordinates " *
            "out of the boundary"))
        push!(dropped, event)
    end
    select_factors(catalogue, dropped)
end

"""
    joined_coordinates(joined) -> Vector{NamedTuple}

The joined coordinates in order: per coordinate the window that holds it first,
its packed position there, and its [`coordinate_identities`](@ref) `path` and
`index`. A joined point and gradient follow this order.
"""
joined_coordinates(w::JoinedWindows) = copy(w.coordinates)

"""
    joined_point(joined, us) -> Vector{Float64}

The joined point of the windows' packed points `us`: each joined coordinate
from the window that holds it first. Fed inputs and later copies of shared
coordinates in `us` are not read.
"""
function joined_point(w::JoinedWindows, us::AbstractVector{<:AbstractVector{<:Real}})
    length(us) == length(w.layouts) || throw(ContractValidationError(
        "[windows] $(length(us)) points for $(length(w.layouts)) windows"))
    Float64[us[c.window][c.position] for c in w.coordinates]
end

"""
    window_points(joined, x) -> Vector{Vector{Float64}}

Each window's packed point at the joined point `x`: its own and shared
coordinates from `x`, each fed input entry from the previous window's boundary
output, unconstrained by the input's support.
"""
function window_points(w::JoinedWindows, x::AbstractVector{<:Real})
    _window_forward!(w, x)
    [copy(u) for u in w.points]
end

(w::JoinedWindows)(x::AbstractVector{<:Real}) =
    first(joined_value_and_gradient!(w, zeros(length(w.coordinates)), x))

function _window_forward!(w::JoinedWindows, x::AbstractVector{<:Real})
    length(x) == length(w.coordinates) || throw(ContractValidationError(
        "[windows] joined point length $(length(x)) ≠ $(length(w.coordinates)) coordinates"))
    K = length(w.layouts)
    for k in 1:K
        u = w.points[k]
        for (i, slot) in enumerate(w.slots[k])
            slot == 0 || (u[i] = x[slot])
        end
        if !isempty(w.fed[k])
            out = _window_outputs(Base.invokelatest(w.values[k - 1], w.points[k - 1]),
                length(w.boundary))
            for (m, (i, p, j)) in enumerate(w.fed[k])
                e = w.fed_entries[k][m]
                v = out[p]
                # The coordinate's own element of the input, for a bound per element.
                at = i - e.offset + 1
                u[i] = _unconstrain_elt(e, Float64(v isa Number ? v : vec(v)[j]), at)
                # d(unconstrained)/d(output) of the input's support.
                w.scale[k][i] = exp(-_logjac_elt(e, u[i], at))
            end
        end
    end
    nothing
end

"""
    joined_value_and_gradient!(joined, g, x) -> (value, g)

The joined windows' log density at the joined point `x` and its gradient over
[`joined_coordinates`](@ref), written into `g`. A fed input's derivative flows
back through the previous window's boundary output.
"""
function joined_value_and_gradient!(w::JoinedWindows, g::AbstractVector, x::AbstractVector{<:Real})
    length(g) == length(w.coordinates) || throw(ContractValidationError(
        "[windows] gradient length $(length(g)) ≠ $(length(w.coordinates)) coordinates"))
    _window_forward!(w, x)
    K = length(w.layouts)
    value = 0.0
    for k in 1:K
        u, gk = w.points[k], w.gradients[k]
        v, _ = sampler_value_and_gradient!(w.samplers[k], gk, u)
        value += v
        if w.dropped[k] !== nothing
            d, _ = ad_value_and_gradient!(w.dropped[k], w.work[k], u)
            value -= d
            gk .-= w.work[k]
        end
    end
    fill!(g, 0.0)
    for k in K:-1:1
        gk = w.gradients[k]
        if k < K && !isempty(w.fed[k + 1])
            seed = _window_seed(w.shapes[k])
            for (i, p, j) in w.fed[k + 1]
                d = w.gradients[k + 1][i] * w.scale[k + 1][i]
                seed = seed[p] isa AbstractArray ? (seed[p][j] += d; seed) :
                    Base.setindex(seed, seed[p] + d, p)
            end
            _, back = ad_value_and_pullback(w.pullbacks[k], _window_cotangent(seed),
                w.points[k])
            gk .+= back
        end
        for (i, slot) in enumerate(w.slots[k])
            slot == 0 || (g[slot] += gk[i])
        end
    end
    value, g
end

"""
    JoinedWindowsProblem

[`JoinedWindows`](@ref) as a log density over its joined coordinates. With
`LogDensityProblems` loaded it implements `dimension`, `capabilities` (first
order), `logdensity` and `logdensity_and_gradient`. Not thread-safe.
"""
struct JoinedWindowsProblem{W<:JoinedWindows}
    joined::W
    gradient::Vector{Float64}
end

"""
    joined_problem(joined) -> JoinedWindowsProblem

The joined windows as a log density of the joined point, for an optimizer,
Pathfinder, a sampler or a variational fit.
"""
joined_problem(w::JoinedWindows) = JoinedWindowsProblem(w, zeros(length(w.coordinates)))

function _joined_problem_value_and_gradient(p::JoinedWindowsProblem, x::AbstractVector{<:Real})
    value, _ = joined_value_and_gradient!(p.joined, p.gradient, x)
    value, copy(p.gradient)
end
