"""
    adaptive_centering_problem(built, plan, problem, backend; cells, u0 = nothing,
                               on_error = nothing, locality = nothing)

Wrap a packed-coordinate LogDensityProblems target for WarmupHMC's adaptive
partial centering. Load `WarmupHMC` and `LogDensityProblems` to enable this
extension. `built` and the bound `plan` describe the same model as `problem`.

Select whole real-valued latent parameters by their graph names:

```julia
rp = adaptive_centering_problem(built, plan, problem, AutoEnzyme();
    cells = (z = (location = :z_location, log_scale = :z_log_scale),))
```

The location and log-scale are named values in the model graph. Each is a
scalar (shared by the selected coordinates) or an array with the latent's
shape, indexed in Julia's linear order. Their numerical work and derivatives
stay in the graph. Array queries use the graph's scalar cell lowering;
`locality` has the same meaning as in [`prepare_cell_query`](@ref).

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
(default zeros) as a model-coordinate exemplar. It prepares one scalar kernel
per named value, independent of the number of latent coordinates. Keep shared
scalar calculations small: WarmupHMC calls the accessors per coordinate.
"""
function adaptive_centering_problem end

# One concrete accessor type for every pair, even across different authored
# cell families. Tuple recursion dispatches statically on the prepared kernels;
# the family and element indices are runtime data, never type parameters.
struct _CenteringAccessor{K}
    kernels::K
    family::Int
    element::Int
end
(a::_CenteringAccessor)(x) = _centering_value(a.kernels, a.family, x, a.element)
@inline _centering_value(kernels::Tuple, family::Int, x, element::Int) =
    family == 1 ? first(kernels)(x, element) :
        _centering_value(Base.tail(kernels), family - 1, x, element)
_centering_value(::Tuple{}, family::Int, x, element::Int) =
    throw(BoundsError((), family))

const _CENTERING_INDEX = :_ppl_centering_index
const _CENTERING_VALUE = :_ppl_centering_value

function _centering_kernel(built, plan, name::Symbol, value, exemplar;
        on_error, locality)
    RK = ReactiveKernels
    spec = RK._copy_cell_factor_spec(built.spec)
    for port in (_CENTERING_INDEX, _CENTERING_VALUE)
        haskey(spec.ports, port) && throw(ContractValidationError(
            "[centering] graph port `$port` collides with an adapter port"))
    end
    index = value!(spec.graph, _CENTERING_INDEX, Int)
    spec.ports[_CENTERING_INDEX] = index
    push!(spec.port_order, _CENTERING_INDEX)
    push!(spec.have_names, _CENTERING_INDEX)
    term = value isa Real ? spec[name] :
        RK._cell_factor_term!(spec, spec[name], index, locality)
    output = value!(spec.graph, _CENTERING_VALUE, Float64)
    spec.ports[_CENTERING_VALUE] = output
    push!(spec.port_order, _CENTERING_VALUE)
    add!(spec.graph, (term,) => output, Float64)
    have = (_query_have(plan)..., _CENTERING_INDEX)
    _absent_refusal(plan, spec, have, output) do
        Base.invokelatest(_prepare_cached, plan, spec; have, want = output,
            bound = _query_bound(plan), on_error, exemplars = (exemplar, 1))
    end
end

function _centering_cells(built, plan::StructuralPlan, cells::NamedTuple;
        u0, on_error, locality)
    isbound(plan) || throw(ContractValidationError(
        "[centering] adaptive_centering_problem requires a bound plan"))
    locality in (nothing, :error, :warn, :ignore) || throw(ContractValidationError(
        "[centering] locality must be :error, :warn, :ignore or nothing"))
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
    kernel_names = Symbol[]
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
        for field in (:location, :log_scale)
            node = cell[field]
            node isa Symbol && haskey(built.spec.ports, node) ||
                throw(ContractValidationError("[centering] `$name.$field` must " *
                    "name an existing graph value; got $(repr(node))"))
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
                push!(kernels, _centering_kernel(built, plan, node, value, exemplar;
                    on_error, locality))
                push!(kernel_names, node)
                family = length(kernels)
            end
            push!(families, family)
        end
        push!(records, (; offset = entry.offset, size = entry.size,
            location = families[1], log_scale = families[2], target, source))
    end
    (; kernels = Tuple(kernels), records, dimension = layout.total)
end

function _centering_endpoint(value, name, field)
    value isa Real && isfinite(value) && 0 <= value <= 1 ||
        throw(ContractValidationError("[centering] `$name.$field` must be in [0, 1]"))
    Float64(value)
end
