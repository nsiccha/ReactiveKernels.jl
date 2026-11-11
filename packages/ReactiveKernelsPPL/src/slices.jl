# Slices of one bound model. A set of partition labels selects observation
# cells (RK's `plate_partition`, through `prepare_partition_sampler`); the
# slice's density adds every authored prior and Jacobian term that owns a
# coordinate those cells reach, closed over what those terms read, each term
# once. The binding is prepared once: another label set is another plan of the
# same kernels, and its prior terms are a factor selection whose rows are data.

"""
    SliceSampler

Slices of one bound model, from [`prepare_slices`](@ref). Evaluate one with
[`slice_value_and_gradient!`](@ref); its coordinates are
[`slice_coordinates`](@ref) and its labels [`slice_labels`](@ref). A label set's
coordinates and prior terms are worked out on its first use and kept. Not
thread-safe: prepare one per thread, as for `PartitionSampler`.
"""
mutable struct SliceSampler{S,C,B}
    const likelihood::S
    const catalogue::C
    const built::Any
    const plan::StructuralPlan
    const backend::B
    const locality::Union{Nothing,Symbol}
    const on_error::Any
    const exemplar::Vector{Float64}
    # The label order of `slice_labels`, which a label set is normalized to.
    const order::Dict{Any,Int}
    # Every authored density event, the packed coordinates it owns, the events
    # owning each coordinate, and the events that own none.
    const events::Vector{DensityFactor}
    const owned::Vector{Vector{Int}}
    const owners::Vector{Vector{Int}}
    const ownerless::Vector{Int}
    # Structural read support per event, computed on first need.
    const supports::Dict{Int,Vector{Int}}
    # Prepared slices by normalized label tuple.
    const slices::Dict{Tuple,Any}
    # A packed-length gradient for a batch of labels.
    const dense::Vector{Float64}
end

struct _Slice{P}
    labels::Vector{Any}
    coordinates::Vector{Int}
    events::Vector{DensityFactor}
    prior::P
    # Per label: its likelihood gradient's buffer (footprint order) and the
    # position of each footprint coordinate in `coordinates`.
    buffers::Vector{Vector{Float64}}
    slots::Vector{Vector{Int}}
    active::Vector{Float64}
    prior_gradient::Vector{Float64}
    gradient::Vector{Float64}
end

"""
    prepare_slices(built, plan, observations, u0; by, backend,
                   partitioned = (), locality = :error, on_error = nothing) -> SliceSampler

Prepare the slices of one bound model: the model restricted to the cells whose
label is in a chosen set, with its prior. `observations` and `by` are those of
[`prepare_partition_sampler`](@ref): `by` gives each observation's cells a
label (a column name, or `observation => labels`), so a slice can be a region of
a national binding, a subject group, or a set of observations (an observation
not named belongs to no slice). `partitioned`, `locality` and `on_error` pass
through to it; `u0` is an exemplar packed point.

A slice of the label set `S` has the log density

- the sum of the cells labelled in `S`, plus
- every authored prior and coordinate-Jacobian term ([`density_factors`](@ref))
  that owns a packed coordinate those cells reach, with the coordinates those
  terms read added in turn until nothing new is reached. Each term counts once,
  however many labels reach it.

Its coordinates, [`slice_coordinates`](@ref), are exactly the coordinates that
density reads. Nothing outside them is read or differentiated, and cells of
other labels are not evaluated. Hyperparameters such as a group scale join the
slice through the priors that read them, and a joint law or coupled transform
stays whole: if a slice reaches one of its coordinates, all of them join.

For a model whose units nest in the labels, the slice of a label equals the
model bound to that label's data alone, in value and gradient, coordinates
matched by [`coordinate_identities`](@ref). It differs where the whole binding
changes a slice's terms: a joint law over every unit, or a data-derived value
computed from all rows (for example a frame centred on all sites).

An authored density that owns no coordinate and reads coordinates both inside
and outside a slice cannot be assigned to it, and that slice is refused with a
`ContractValidationError` naming the density.

Pass a label or collection of labels to [`slice_coordinates`](@ref) and
[`slice_value_and_gradient!`](@ref). The default gradient storage contains
only the slice coordinates; `compact=false` writes into full packed storage.

The binding is prepared once. Another label set uses the same compiled
kernels: its cells are another plan of the partition kernels, and its prior
terms a factor selection whose rows are data, prepared on the set's first use.
"""
function prepare_slices(built, plan::StructuralPlan, observations,
        u0::AbstractVector{<:Real}; by, backend, partitioned = (), locality = :error,
        on_error = nothing)
    built = _binding_build(built, plan)
    layout = built.layout::LayoutTable
    exemplar = Vector{Float64}(u0)
    likelihood = prepare_partition_sampler(built, plan, observations, exemplar; by, backend,
        compact = true, partitioned, locality, on_error)
    catalogue = density_factors(built, plan)
    events = collect(DensityFactor, catalogue)
    owned = Vector{Int}[factor_coordinates(catalogue, e) for e in events]
    owners = [Int[] for _ in 1:layout.total]
    for (k, coordinates) in enumerate(owned), c in coordinates
        push!(owners[c], k)
    end
    labels = partition_labels(likelihood)
    order = Dict{Any,Int}(label => i for (i, label) in enumerate(labels))
    SliceSampler(likelihood, catalogue, built, plan, backend, locality, on_error, exemplar,
        order, events, owned, owners, findall(isempty, owned), Dict{Int,Vector{Int}}(),
        Dict{Tuple,Any}(), zeros(layout.total))
end

"""
    slice_labels(slices) -> Vector

The labels of `slices`, in first-seen order. A label set is any collection of
them; a single label stands for itself.
"""
slice_labels(s::SliceSampler) = partition_labels(s.likelihood)

"""
    slice_coordinates(slices, labels) -> Vector{Int}

The sorted packed positions, in the whole binding's layout, of the slice of
`labels` (one label, or a collection of them). A compact gradient follows this
order. `coordinate_identities(built.layout)[slice_coordinates(slices, labels)]`
names them by meaning, for matching against another binding.
"""
slice_coordinates(s::SliceSampler, labels) = copy(_slice(s, labels).coordinates)

"""
    slice_value_and_gradient!(slices, g, u, labels; compact = true) -> (value, g)

The log density of the slice of `labels` (one label, or a collection of them)
at the packed point `u`, and its gradient written into `g`: over
[`slice_coordinates`](@ref) in that order when `compact`, otherwise over the
whole packed vector with zeros outside the slice. `u` is read only; only its
slice coordinates matter. Several labels share one pullback through shared
values when the slices were prepared with `partitioned`.
"""
function slice_value_and_gradient!(s::SliceSampler, g::AbstractVector, u::AbstractVector,
        labels; compact::Bool = true)
    point = u isa Vector{Float64} ? u : Vector{Float64}(u)
    total = length(s.dense)
    length(point) == total || throw(ContractValidationError(
        "[slices] packed length $(length(point)) ≠ layout total $total"))
    slice = _slice(s, labels)
    n = length(slice.coordinates)
    expected = compact ? n : total
    length(g) == expected || throw(ContractValidationError(
        "[slices] gradient length $(length(g)) ≠ expected length $expected"))
    work = slice.gradient
    fill!(work, 0.0)
    value = 0.0
    if length(slice.labels) > 1 && s.likelihood.prefix !== nothing
        v, _ = partition_batch_value_and_gradient!(s.likelihood, s.dense, point, slice.labels)
        value += v
        for (i, c) in enumerate(slice.coordinates)
            work[i] += s.dense[c]
        end
    else
        for (k, label) in enumerate(slice.labels)
            buffer = slice.buffers[k]
            v, _ = partition_value_and_gradient!(s.likelihood, buffer, point, label)
            value += v
            for (i, slot) in enumerate(slice.slots[k])
                work[slot] += buffer[i]
            end
        end
    end
    if slice.prior !== nothing
        for (i, c) in enumerate(slice.coordinates)
            slice.active[i] = point[c]
        end
        v, _ = ad_value_and_gradient!(slice.prior, slice.prior_gradient, point, slice.active)
        value += v
        work .+= slice.prior_gradient
    end
    if compact
        copyto!(g, work)
    else
        fill!(g, 0.0)
        for (i, c) in enumerate(slice.coordinates)
            g[c] = work[i]
        end
    end
    return value, g
end

"""
    SliceProblem

One slice of a [`SliceSampler`](@ref) as a log density over its compact
coordinates, with the binding's other coordinates held at a base point it owns.
With `LogDensityProblems` loaded it implements `dimension`, `capabilities`
(first order), `logdensity` and `logdensity_and_gradient`. Not thread-safe.
"""
struct SliceProblem{S<:SliceSampler}
    slices::S
    labels::Tuple
    coordinates::Vector{Int}
    point::Vector{Float64}
    gradient::Vector{Float64}
end

"""
    slice_problem(slices, labels; base = u0) -> SliceProblem

The slice of `labels` as a log density of its compact coordinates
`x = u[slice_coordinates(slices, labels)]`, for an optimizer, Pathfinder or a
variational fit. `base` (default: the exemplar `slices` was prepared with) is
copied. The slice reads none of its other coordinates, so it only completes a
packed point. `slice_problem(...)(x)` is the log density.
"""
function slice_problem(s::SliceSampler, labels; base::AbstractVector{<:Real} = s.exemplar)
    length(base) == length(s.dense) || throw(ContractValidationError(
        "[slices] base length $(length(base)) ≠ layout total $(length(s.dense))"))
    key = _slice_key(s, labels)
    slice = _slice(s, key)
    SliceProblem(s, key, copy(slice.coordinates), Vector{Float64}(base),
        zeros(length(slice.coordinates)))
end

(p::SliceProblem)(x::AbstractVector{<:Real}) = _slice_problem_value(p, x)
_slice_problem_value(p::SliceProblem, x) = first(_slice_problem_value_and_gradient(p, x))
function _slice_problem_value_and_gradient(p::SliceProblem, x::AbstractVector{<:Real})
    length(x) == length(p.coordinates) || throw(DimensionMismatch(
        "slice point length $(length(x)) ≠ slice dimension $(length(p.coordinates))"))
    for (i, c) in enumerate(p.coordinates)
        p.point[c] = x[i]
    end
    value, _ = slice_value_and_gradient!(p.slices, p.gradient, p.point, p.labels)
    value, copy(p.gradient)
end

# ── label sets and their prepared slices ─────────────────────────────

function _slice_key(s::SliceSampler, labels)
    haskey(s.order, labels) && return (labels,)
    labels isa Tuple && haskey(s.slices, labels) && return labels
    list = if labels isa Union{AbstractVector,Tuple,AbstractSet}
        collect(Any, labels)
    else
        throw(ContractValidationError("[slices] $(repr(labels)) is not a label of " *
            "these slices; the labels are $(join(repr.(slice_labels(s)), ", "))"))
    end
    isempty(list) && throw(ContractValidationError("[slices] a slice needs at least one label"))
    for label in list
        haskey(s.order, label) || throw(ContractValidationError(
            "[slices] $(repr(label)) is not a label of these slices; the labels are " *
            join(repr.(slice_labels(s)), ", ")))
    end
    Tuple(sort!(unique(list); by = label -> s.order[label]))
end

function _slice(s::SliceSampler, labels)
    key = _slice_key(s, labels)
    get!(() -> _prepare_slice(s, key), s.slices, key)
end

function _slice_support(s::SliceSampler, k::Int)
    get!(s.supports, k) do
        factor_support(s.catalogue, s.events[k]; u0 = s.exemplar)
    end
end

# The coordinates the labels' cells reach, closed over the priors and
# Jacobians that own them and what those read; returns the selected events.
function _slice_closure!(s::SliceSampler, reached::Set{Int})
    selected = falses(length(s.events))
    queue = collect(reached)
    while !isempty(queue)
        c = pop!(queue)
        for k in s.owners[c]
            selected[k] && continue
            selected[k] = true
            for q in Iterators.flatten((s.owned[k], _slice_support(s, k)))
                q in reached || (push!(reached, q); push!(queue, q))
            end
        end
    end
    for k in s.ownerless
        support = _slice_support(s, k)
        inside = count(in(reached), support)
        iszero(inside) && continue
        inside == length(support) || throw(ContractValidationError(
            "[slices] the authored density $(s.events[k]) owns no coordinate and reads " *
            "coordinates both inside and outside this slice, so it belongs to no slice " *
            "alone; it reads $(length(support) - inside) coordinates outside it"))
        selected[k] = true
    end
    findall(selected)
end

function _prepare_slice(s::SliceSampler, key::Tuple)
    labels = collect(Any, key)
    reached = Set{Int}()
    for label in labels
        union!(reached, partition_footprint(s.likelihood, label))
    end
    selected = _slice_closure!(s, reached)
    coordinates = sort!(collect(reached))
    events = s.events[selected]
    prior = isempty(events) ? nothing : Base.invokelatest(prepare_query_ad, prepare_ad,
        s.built, s.plan, select_factors(s.catalogue, events), s.backend, s.exemplar;
        coordinates, locality = s.locality === nothing ? :error : s.locality,
        on_error = s.on_error)
    footprints = [partition_footprint(s.likelihood, label) for label in labels]
    slots = [Int[searchsortedfirst(coordinates, c) for c in footprint]
        for footprint in footprints]
    n = length(coordinates)
    _Slice(labels, coordinates, events, prior, [zeros(length(f)) for f in footprints], slots,
        zeros(n), zeros(n), zeros(n))
end
