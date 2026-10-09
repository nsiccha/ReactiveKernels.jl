# One cell of an authored plate as its own graph value (`plate_cell`). The cell
# recipe and its lowering live with the plate lowering (`_AuthoredPlateCellOp`,
# codegen.jl); this file is the public graph transform.

"""
    plate_cell(spec::KernelSpec, plate::Symbol; index = :cell, name = Symbol(plate, :_cell)) -> KernelSpec

A new kernel in which the value `name` is ONE cell of the authored plate that
produces `plate`, at the position held by the HAVE port `index`: a linear
position or a `CartesianIndex` in the plate's broadcast domain. For a plate
over `x` (or an `@plate for i in eachindex(x)` loop), cell `i` is the plate's
value at `x[i]`.

Planning a WANT of `name` runs only that cell, through the plate's own scalar
body and the same native plate lowering, so the value is the plate's element at
that position. An authored plate whose only consumer is this one is composed
into the cell, as plate chains are composed (see [`prepare`](@ref)), and so
also runs at that cell only. So is a plate read only by cells at the same
position (several `plate_cell` values sharing `index`): each of those cells
runs it at that position. Every other value the cell reads is planned as
usual. A cell outside the plate's domain throws `BoundsError`; the plate's
broadcast-domain checks are kept.

The default boundary of the result is the original HAVE ports plus `index`,
and the WANT `name`. `spec` is unchanged: the result has its own graph, sharing
the original values and recipes. `index` may name an existing HAVE port of
`spec` (or of an earlier `plate_cell` result) to evaluate several plates at the
same cell.
"""
function plate_cell(spec::KernelSpec, plate::Symbol; index::Symbol = :cell,
                    name::Symbol = Symbol(plate, :_cell))
    graph = spec.graph
    target = spec[plate]
    producers = Recipe[graph.recipes[id] for id in producers_of(graph, target.id)
                       if graph.recipes[id].op isa _AuthoredPlateOp]
    isempty(producers) && throw(ArgumentError(
        "plate_cell: :$plate is not produced by an authored plate"))
    length(producers) == 1 || throw(ArgumentError(
        "plate_cell: :$plate has $(length(producers)) authored plate producers"))
    recipe = only(producers)
    haskey(spec.ports, name) && throw(ArgumentError(
        "plate_cell: the kernel already has a port :$name"))
    copied = Graph(copy(graph.values), copy(graph.recipes),
        Dict{Int,Vector{Int}}(id => copy(ids) for (id, ids) in graph.producers),
        copy(graph.aliases), graph.version, nothing, graph.value_id_floor)
    ports = copy(spec.ports)
    order = copy(spec.port_order)
    have_names = copy(spec.have_names)
    position = if haskey(ports, index)
        existing = ports[index]
        isempty(producers_of(graph, existing.id)) || throw(ArgumentError(
            "plate_cell: the index port :$index is computed by the kernel"))
        existing
    else
        _kernel_declare!(copied, ports, order, index, Any)
    end
    _kernel_push_unique!(have_names, index)
    T = valtype(only(recipe.op.kernel.plan.want))
    cell = _kernel_declare!(copied, ports, order, name, T)
    _add_recipe!(copied, (position, recipe.inputs...), (cell,),
                 _AuthoredPlateCellOp(recipe.op), recipe.cost, nothing, false,
                 recipe.source)
    KernelSpec(copied, ports, order, have_names, Symbol[name], nothing)
end
