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
values' numerical work and derivatives stay in the graph. WarmupHMC 0.2.19
or later evaluates each named quantity once
per contiguous latent block and shares its result across the coordinates,
including when the quantity contains a reduction over unselected latents.

Each cell optionally supplies `target` and `source` in `[0, 1]`: zero is
non-centered, one is centered. Both default to `0.0`. `target` describes the
model's fixed coordinates; WarmupHMC adapts `source`. Unselected coordinates
pass through, and sampling returns the original model coordinates.

The selected latent must have an identity support transform. Locations and
log-scales may read unselected latents, but must not read any selected latent:
WarmupHMC's indexed affine maps evaluate accessors on the source vector, and
their inverse and log-Jacobian require these quantities to stay unchanged by
the map. This condition concerns coordinates, not prior families (a real
ICAR block, for example, can be selected).

Construction evaluates each named value once to check its shape, using `u0`
(default zeros) as a model-coordinate exemplar. It prepares one graph kernel
per named value, independent of the number of latent coordinates.
"""
function adaptive_centering_problem end

# One concrete accessor type for every pair, even across different authored
# cell families. Tuple recursion dispatches statically on the prepared kernels;
# the family index is runtime data, never a type parameter.
# Share the tuple through one typed holder: storing every prepared kernel
# inline in every accessor makes each indexed map large and costly to copy.
struct _CenteringAccessor{K}
    kernels::Base.RefValue{K}
    family::Int
end
@inline (a::_CenteringAccessor)(x) = _centering_value(a.kernels[], a.family, x)
@inline _centering_value(kernels::Tuple, family::Int, x) =
    family == 1 ? first(kernels)(x) :
        _centering_value(Base.tail(kernels), family - 1, x)
_centering_value(::Tuple{}, family::Int, x) =
    throw(BoundsError((), family))

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

function _centering_kernel(built, plan, name::Symbol; on_error)
    RK = ReactiveKernels
    spec = RK._copy_cell_factor_spec(built.spec)
    haskey(spec.ports, _CENTERING_VALUE) && throw(ContractValidationError(
        "[centering] graph port `$_CENTERING_VALUE` collides with an adapter port"))
    output = value!(spec.graph, _CENTERING_VALUE, Vector{Float64})
    spec.ports[_CENTERING_VALUE] = output
    push!(spec.port_order, _CENTERING_VALUE)
    add!(spec.graph, (spec[name],) => output, _centering_values)
    have = _query_have(plan)
    _absent_refusal(plan, spec, have, output) do
        # Samplers also evaluate column views of their position matrices.
        # Keep the graph's AbstractVector input contract instead of narrowing
        # this callable to the construction exemplar's concrete storage type.
        Base.invokelatest(_prepare_cached, plan, spec; have, want = output,
            bound = _query_bound(plan), on_error)
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
    selected_ids = Set(RK.canon_id(graph, built.spec[n].id) for n in selected)
    kernels = Any[]
    kernel_names = Union{Nothing,Symbol}[]
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
                push!(kernels, Base.Returns(_centering_values(node)))
                push!(kernel_names, nothing)
                push!(families, length(kernels))
                push!(scalars, true)
                continue
            end
            node isa Symbol && haskey(built.spec.ports, node) ||
                throw(ContractValidationError("[centering] `$name.$field` must " *
                    "be a real constant or name an existing graph value; got $(repr(node))"))
            RK._cell_factor_reads(built.spec, built.spec[node], selected_ids) &&
                throw(ContractValidationError("[centering] `$node` reads a selected " *
                    "latent; indexed affine accessors must be invariant under " *
                    "the selected coordinate transforms"))
            value = get!(values, node) do
                full = prepare_query(built, plan, (node,); on_error, exemplars = (exemplar,))
                Base.invokelatest(full, exemplar)
            end
            (value isa Real || value isa AbstractArray{<:Real} && size(value) == shape) ||
                throw(ContractValidationError("[centering] `$node` must be a real " *
                    "scalar or an array of shape $shape for `$name`; got $(typeof(value))"))
            family = findfirst(==(node), kernel_names)
            if family === nothing
                push!(kernels, _centering_kernel(built, plan, node; on_error))
                push!(kernel_names, node)
                family = length(kernels)
            end
            push!(families, family)
            push!(scalars, value isa Real)
        end
        push!(records, (; offset = entry.offset, size = entry.size,
            location = families[1], log_scale = families[2],
            scalar_location = scalars[1], scalar_log_scale = scalars[2], target, source))
    end
    (; kernels = Tuple(kernels), records, dimension = layout.total)
end

function _centering_endpoint(value, name, field)
    value isa Real && isfinite(value) && 0 <= value <= 1 ||
        throw(ContractValidationError("[centering] `$name.$field` must be in [0, 1]"))
    Float64(value)
end
