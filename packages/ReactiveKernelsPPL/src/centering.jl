"""
    adaptive_centering_problem(built, plan, problem, backend; cells, u0 = nothing,
                               on_error = nothing)

Wrap a packed-coordinate LogDensityProblems target for WarmupHMC's adaptive
partial centering. Load `WarmupHMC` and `LogDensityProblems` to enable this
extension. `built` and the bound `plan` describe the same model as `problem`.

This function is public but not exported, because BayesianRegressionModels
exports a different `adaptive_centering_problem`. Import it with
`using ReactiveKernelsPPL: adaptive_centering_problem`, or call it qualified.

Select whole real-valued latent parameters by their graph names, for example
`cells = (z = (location = :z_location, log_scale = :z_log_scale),)`.

Each location and log-scale is either a `Real` constant shared by the block
or a symbol naming a value in the model graph. For example,
`cells = (z = (location = 0.0, log_scale = :z_log_scale),)` needs no named zero
in the model. A named value is a scalar (shared by the selected coordinates)
or an array with the latent's shape, indexed in Julia's linear order. Named
values' numerical work and derivatives stay in the graph. One joint kernel
computes all selected endpoints, sharing their common graph work across cells
and both affine roles. WarmupHMC 0.2.21
or later evaluates it once per vector and level (below) on a map or inverse,
and shares it across scoring candidates. Identity maps do not evaluate the
endpoints.

Each cell optionally supplies `target` and `source` in `[0, 1]`: zero is
non-centered, one is centered. Both default to `0.0`. `target` describes the
model's fixed coordinates; WarmupHMC adapts `source`. Unselected coordinates
pass through, and sampling returns the original model coordinates.

The selected latent must have an identity support transform. This condition
concerns coordinates, not prior families (a real ICAR block, for example, can
be selected). Locations and log-scales may read unselected latents and the
latents of other selected cells, but never their own cell's latent.

A cell whose location or log-scale reads another selected cell's latent is
centred about that latent's model-frame value, as in a nested hierarchy:

    cells = (z_g = (location = :mu, log_scale = :log_tau_g),
             z_u = (location = :unit_location, log_scale = :log_tau_u))

with `unit_location = mu .+ tau_g .* z_g[group]`. The reads order the cells
into levels: a cell that reads no selected latent is on level 1, and a cell
that reads latents of levels up to `k` is on level `k + 1`. WarmupHMC maps the
levels in order, so re-centering a parent level keeps the meaning of the
coordinates centred on it. The joint kernel runs once per level, after the
levels before it are mapped. Cells that read each other's latents, directly
or through other cells, form a cycle and are refused. Nested levels need
WarmupHMC 0.3.0 or later, whose accessors read earlier pairs at their model
values; with WarmupHMC 0.2 they are refused.

Construction evaluates each named value once to check its shape, using `u0`
(default zeros) as a model-coordinate exemplar. The joint execution kernel
is independent of the number of latent coordinates.
"""
function adaptive_centering_problem end

# Every pair shares one callable, including across cells and affine roles.
# Keep the prepared kernel in one typed holder instead of copying it into
# every pair. The holder is read-only throughout evaluation and ordinary AD.
struct _CenteringAccessor{K}
    kernel::Base.RefValue{K}
end
@inline (a::_CenteringAccessor)(x) = a.kernel[](x)

# A Markov chain's per-state accessor: one family of the chains' frame kernel.
# Its first entry is the first state's location (`_ChainInitial`).
struct _ChainAccessor{K}
    kernel::_CenteringAccessor{K}
    family::Int
end
@inline (a::_ChainAccessor)(x) = a.kernel(x).values[a.family]
struct _ChainInitial{K}
    kernel::_CenteringAccessor{K}
    family::Int
end
@inline (a::_ChainInitial)(x) = a.kernel(x).values[a.family][1]

# Runtime family/element selectors keep heterogeneous endpoint shapes concrete
# without concatenating or copying their arrays into a flat buffer.
struct _CenteringFrame{N}
    values::NTuple{N,Vector{Float64}}
end
@inline Base.getindex(frame::_CenteringFrame, index::CartesianIndex{2}) =
    frame.values[index[1]][index[2]]

const _CENTERING_VALUE = :_ppl_centering_value

# Every family returns one concrete container type. Scalars occupy one slot;
# arrays retain Julia's linear coordinate order. This conversion is a graph
# recipe, so the ordinary backend differentiates through it with the model.
_centering_values(value::Real) = Float64[value]
# Accessors only read the graph's outputs. A dense Float64 array already has
# the required storage, so flatten it without copying its elements. This also
# leaves bound, cached arrays read-only during maps and ordinary reverse AD.
_centering_values(value::Array{Float64}) = vec(value)
_centering_values(value::AbstractArray{<:Real}) = collect(Float64, vec(value))
# A recurrence per `@plate` index records one vector per index; its chain reads
# them concatenated in plate order.
_centering_values(value::AbstractVector{<:AbstractVector{<:Real}}) =
    Float64[v for entry in value for v in entry]
_centering_frame(values...) =
    _CenteringFrame(map(_centering_values, values))

function _centering_kernel(built, plan, endpoints::Tuple; on_error)
    RK = ReactiveKernels
    spec = RK._copy_cell_factor_spec(built.spec)
    haskey(spec.ports, _CENTERING_VALUE) && throw(ContractValidationError(
        "[centering] graph port `$_CENTERING_VALUE` collides with an adapter port"))
    # The endpoint count is graph structure; each endpoint's size stays runtime
    # data. A tuple avoids allocating an outer container on every evaluation.
    output = value!(spec.graph, _CENTERING_VALUE, _CenteringFrame{length(endpoints)})
    spec.ports[_CENTERING_VALUE] = output
    push!(spec.port_order, _CENTERING_VALUE)
    constant_names = Symbol[]
    constants = Float64[]
    inputs = map(enumerate(endpoints)) do (i, endpoint)
        endpoint isa Symbol && return spec[endpoint]
        # Numeric configuration is a bound graph input, never source data.
        name = Symbol("_ppl_centering_constant_", i)
        while haskey(spec.ports, name)
            name = Symbol(name, "_")
        end
        input = value!(spec.graph, name, Float64)
        spec.ports[name] = input
        push!(spec.port_order, name)
        push!(constant_names, name)
        push!(constants, endpoint)
        input
    end
    add!(spec.graph, Tuple(inputs) => output, _centering_frame)
    have = (_query_have(plan)..., constant_names...)
    bound = merge(_query_bound(plan), NamedTuple{Tuple(constant_names)}(Tuple(constants)))
    _absent_refusal(plan, spec, have, output) do
        # Samplers also evaluate column views of their position matrices.
        # Keep the graph's AbstractVector input contract instead of narrowing
        # this callable to the construction exemplar's concrete storage type.
        Base.invokelatest(_prepare_cached, plan, spec; have, want = output,
            bound, on_error)
    end
end

function _centering_cells(built, plan::StructuralPlan, cells::NamedTuple;
        u0, on_error)
    isbound(plan) || throw(ContractValidationError(
        "[centering] adaptive_centering_problem requires a bound plan"))
    built = _binding_build(built, plan)
    _check_built_program(built, plan)
    layout = built.layout
    exemplar = _query_exemplar(layout, u0)
    entries = Dict(e.name => e for e in layout.entries)
    selected = Symbol[]
    for name in keys(cells)
        haskey(entries, name) || throw(ContractValidationError(
            "[centering] `$name` is not a packed latent parameter"))
        entry = entries[name]
        entry.transform === :identity || throw(ContractValidationError(
            "[centering] `$name` has a $(entry.transform) support transform; " *
            "an affine cell must name identity-transformed real coordinates"))
        push!(selected, name)
    end
    RK = ReactiveKernels
    graph = built.spec.graph
    selected_ids = [n => RK.canon_id(graph, built.spec[n].id) for n in selected]
    # The selected latents each cell's endpoints read, in cell keyword order.
    reads = Dict(n => Symbol[] for n in selected)
    kernel_names = Union{Float64,Symbol}[]
    families_by_name = Dict{Symbol,Int}()
    values = Dict{Symbol,Any}()
    records = NamedTuple[]
    for name in selected
        entry = entries[name]
        cell = cells[name]
        cell isa NamedTuple && haskey(cell, :location) && haskey(cell, :log_scale) ||
            throw(ContractValidationError("[centering] `$name` needs " *
                "(location = :graph_name, log_scale = :graph_name)"))
        all(k -> k in (:location, :log_scale, :target, :source), keys(cell)) ||
            throw(ContractValidationError("[centering] unknown field in cell `$name`"))
        target = _centering_endpoint(get(cell, :target, 0.0), name, :target)
        source = _centering_endpoint(get(cell, :source, 0.0), name, :source)
        shape, element_shapes = _parameter_shape(layout, entry, entry.size)
        element_shapes === nothing && prod(shape) == entry.size ||
            throw(ContractValidationError("[centering] `$name` needs a scalar or " *
                "rectangular array of identity-transformed coordinates"))
        families = Int[]
        scalars = Bool[]
        for field in (:location, :log_scale)
            node = cell[field]
            if node isa Real
                # Store configuration as runtime data behind the same accessor
                # type as graph queries. Give each constant its own slot: merging
                # equal constants would make the tuple type depend on their values.
                push!(kernel_names, Float64(node))
                push!(families, length(kernel_names))
                push!(scalars, true)
                continue
            end
            node isa Symbol && haskey(built.spec.ports, node) ||
                throw(ContractValidationError("[centering] `$name.$field` must " *
                    "be a real constant or name an existing graph value; got $(repr(node))"))
            for (other, id) in selected_ids
                RK._cell_factor_reads(built.spec, built.spec[node], (id,)) || continue
                other === name && throw(ContractValidationError("[centering] " *
                    "`$name.$field` (`$node`) reads `$name` itself; a cell's " *
                    "location and log-scale must not depend on its own coordinates"))
                other in reads[name] || push!(reads[name], other)
            end
            value = get!(values, node) do
                full = prepare_query(built, plan, (node,); on_error, exemplars = (exemplar,))
                Base.invokelatest(full, exemplar)
            end
            (value isa Real || value isa AbstractArray{<:Real} && size(value) == shape) ||
                throw(ContractValidationError("[centering] `$node` must be a real " *
                    "scalar or an array of shape $shape for `$name`; got $(typeof(value))"))
            family = get!(families_by_name, node) do
                push!(kernel_names, node)
                length(kernel_names)
            end
            push!(families, family)
            push!(scalars, value isa Real)
        end
        push!(records, (; name, offset = entry.offset, size = entry.size,
            location = families[1], log_scale = families[2],
            scalar_location = scalars[1], scalar_log_scale = scalars[2], target, source))
    end
    levels = _centering_levels(selected, reads)
    records = [merge(cell, (; stage = levels[cell.name])) for cell in records]
    # Levels map in order, each after the cells it reads. Cell keyword order
    # need not match packed coordinate order: within a level, increasing
    # indices let WarmupHMC share one kernel evaluation over the level's cells.
    sort!(records; by = cell -> (cell.stage, cell.offset))
    kernel = isempty(kernel_names) ? nothing :
        _centering_kernel(built, plan, Tuple(kernel_names); on_error)
    chains, chain_kernel = _centering_chains(built, plan, layout, entries, exemplar,
        selected, selected_ids, cells; on_error)
    (; kernel, records, chains, chain_kernel, dimension = layout.total)
end

# The recurrences `markov_chains` marked: those whose recorded frame
# (`_rkppl_chain_<x>_coef`) the graph holds.
function _centering_chain_names(spec)
    out = Symbol[]
    prefix, suffix = _CHAIN_PREFIX, "_coef"
    for name in keys(spec.ports)
        str = String(name)
        startswith(str, prefix) && endswith(str, suffix) && !occursin("_frame_", str) ||
            continue
        push!(out, Symbol(str[length(prefix)+1:end-length(suffix)]))
    end
    sort!(out)
end

# One record per marked recurrence: its states in time order (each `@plate`
# index's in plate order) and the kernel families of its coefficient, drift
# and log step SD. The frame reads only coordinates no chain or cell
# transforms, and no cell's endpoint reads a chain's states.
function _centering_chains(built, plan, layout, entries, exemplar, selected,
        selected_ids, cells; on_error)
    RK = ReactiveKernels
    graph = built.spec.graph
    names = _centering_chain_names(built.spec)
    isempty(names) && return NamedTuple[], nothing
    records = NamedTuple[]
    frame_names = Symbol[]
    state_entries = Symbol[]
    for x in names
        frames = Tuple(_chain_name(x, r) for r in _CHAIN_ROLES)
        coef = Base.invokelatest(prepare_query(built, plan, (frames[1],); on_error,
            exemplars = (exemplar,)), exemplar)
        lengths = coef isa AbstractVector{<:Real} ? [length(coef)] : length.(coef)
        seed = Symbol(:_rkppl_scan_, x, :_seed_, x, :_1)
        draw = Symbol(:_rkppl_scan_, x, :_draw_, x)
        states = Int[]
        used = Symbol[]
        if coef isa AbstractVector{<:Real}
            entry = haskey(entries, x) ? entries[x] : get(entries, Symbol(:_ppl_scan_z_, x), nothing)
            entry === nothing || entry.size == only(lengths) || (entry = nothing)
            entry === nothing && throw(ContractValidationError("[centering] the " *
                "states of Markov chain `$x` are not one packed block of its length"))
            append!(states, entry.offset .+ (0:entry.size-1))
            push!(used, entry.name)
        else
            sampled = haskey(entries, seed)
            haskey(entries, draw) && (!sampled || entries[seed].size == length(lengths)) ||
                throw(ContractValidationError("[centering] the states of Markov chain " *
                    "`$x` are not its cells' seed and step blocks"))
            steps = entries[draw]
            sum(lengths; init = 0) - (sampled ? length(lengths) : 0) == steps.size ||
                throw(ContractValidationError("[centering] Markov chain `$x` records " *
                    "$(sum(lengths; init = 0)) states but packs $(steps.size) steps"))
            run = 0
            for (g, n) in enumerate(lengths)
                sampled && push!(states, entries[seed].offset + g - 1)
                k = sampled ? n - 1 : n
                append!(states, steps.offset + run .+ (0:k-1))
                run += k
            end
            sampled && push!(used, seed)
            push!(used, draw)
        end
        for name in used
            name in selected && throw(ContractValidationError("[centering] `$name` " *
                "holds the states of Markov chain `$x`; a chain's states take no cell"))
        end
        append!(state_entries, used)
        start = length(frame_names)
        append!(frame_names, frames)
        push!(records, (; name = x, states, coef = start + 1, drift = start + 2,
            log_scale = start + 3))
    end
    # Accessors read the source vector. A chain's frame reading transformed
    # coordinates, or a cell's endpoint reading a chain's states, would read
    # values whose meaning changes as warm-up adapts.
    ids = [n => RK.canon_id(graph, built.spec[n].id) for n in state_entries
        if haskey(built.spec.ports, n)]
    for name in frame_names, (n, id) in [ids; selected_ids]
        RK._cell_factor_reads(built.spec, built.spec[name], (id,)) &&
            throw(ContractValidationError("[centering] the transition of a Markov " *
                "chain (`$name`) reads `$n`; a chain's coefficients, drift and step " *
                "size may read only coordinates no chain or cell transforms"))
    end
    for (name, cell) in pairs(cells), field in (:location, :log_scale)
        node = cell[field]
        node isa Symbol || continue
        for (n, id) in ids
            RK._cell_factor_reads(built.spec, built.spec[node], (id,)) &&
                throw(ContractValidationError("[centering] `$name.$field` (`$node`) " *
                    "reads `$n`, the states of a Markov chain; a cell's endpoints " *
                    "may not read a chain's states"))
        end
    end
    kernel = _centering_kernel(built, plan, Tuple(frame_names); on_error)
    records, kernel
end

# A cell reading no selected latent is on level 1; a cell reading cells of
# levels up to `k` is on level `k + 1`. Reads that return to a cell are a cycle,
# which no order of the affine maps can evaluate on model-frame values.
function _centering_levels(selected, reads)
    levels = Dict{Symbol,Int}()
    path = Symbol[]
    function visit(name)
        haskey(levels, name) && return levels[name]
        at = findfirst(==(name), path)
        if at !== nothing
            cycle = join(("`$n`" for n in (path[at:end]..., name)), " → ")
            throw(ContractValidationError("[centering] cells $cycle read each " *
                "other's latents through their locations or log-scales; nested " *
                "cells must form levels, each read only by later ones"))
        end
        push!(path, name)
        level = 1 + maximum(visit, reads[name]; init = 0)
        pop!(path)
        levels[name] = level
    end
    foreach(visit, selected)
    levels
end

function _centering_endpoint(value, name, field)
    value isa Real && isfinite(value) && 0 <= value <= 1 ||
        throw(ContractValidationError("[centering] `$name.$field` must be in [0, 1]"))
    Float64(value)
end
