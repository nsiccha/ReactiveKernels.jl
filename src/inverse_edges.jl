# Automatic inverse edges for `@kernel`-authored graphs.
#
# A hand-written bidirectional pair (`log_scale = log(scale)` next to
# `scale = exp(log_scale)`) is boilerplate: the reverse direction is already
# determined by the forward call. After `_kernel_expand` emits the authored
# recipes, `_kernel_synthesize_inverse_edges!` provisions the missing reverse
# edges so the planner can traverse them in either direction:
#
# * a single-input, single-output recipe whose operation has an
#   `InverseFunctions.inverse` gains the reversed recipe (`x = exp(y)` for
#   `y = log(x)`). Operations without an inverse (anything whose `inverse`
#   is `NoInverse`, or whose `inverse` throws) gain nothing; and
# * a tuple/`NamedTuple` pack recipe (`params = (; x, y)`) gains one unpack
#   recipe per field that is a bare port (`x = params.x`), so a packed value
#   supplied as HAVE unpacks to its fields on demand.
#
# Synthesis is authoring-time only: graphs built by hand through `Graph()` /
# `add!` keep exactly the recipes they are given. A reverse edge is added only
# when no equivalent recipe (same canonical input, same canonical output,
# `===` operation) already exists, so hand-written pairs are unchanged, and the
# pass runs once over a snapshot — a synthesized edge's own reverse is the
# original recipe, so no fixpoint iteration is needed.

"""
    _kernel_synthesize_inverse_edges!(graph) -> Graph

Provision automatic reverse edges for every eligible authored recipe in
`graph` (see the file header). Runs once per `@kernel` construction, after
all authored recipes (including nested-spec splices) are in place.
"""
function _kernel_synthesize_inverse_edges!(graph::Graph)
    snapshot = copy(graph.recipes)
    for recipe in snapshot
        _kernel_synthesize_recipe_edges!(graph, recipe)
    end
    graph
end

function _kernel_synthesize_recipe_edges!(graph::Graph, recipe::Recipe)
    recipe.effectful && return nothing
    _kernel_synthesize_pack_edges!(graph, recipe) && return nothing
    length(recipe.inputs) == 1 || return nothing
    length(recipe.outputs) == 1 || return nothing
    forward_input = only(recipe.inputs)
    forward_output = only(recipe.outputs)
    canon_id(graph, forward_input.id) == canon_id(graph, forward_output.id) &&
        return nothing
    reversed_op = try
        inverse(recipe.op)
    catch
        return nothing
    end
    reversed_op isa NoInverse && return nothing
    in_id = canon_id(graph, forward_output.id)
    out_id = canon_id(graph, forward_input.id)
    _kernel_has_equivalent_recipe(graph, in_id, out_id, reversed_op) &&
        return nothing
    add!(graph; inputs = (forward_output,), outputs = (forward_input,),
         op = reversed_op, cost = recipe.cost,
         source = _kernel_inverse_source(reversed_op, forward_output.name))
    nothing
end

function _kernel_has_equivalent_recipe(
        graph::Graph, input_id::Int, output_id::Int, op)
    for candidate in graph.recipes
        length(candidate.inputs) == 1 || continue
        canon_id(graph, only(candidate.inputs).id) == input_id || continue
        any(o -> canon_id(graph, o.id) == output_id, candidate.outputs) ||
            continue
        candidate.op === op && return true
    end
    false
end

function _kernel_inverse_source(op, input_name::Symbol)
    op isa Function || return _NO_KERNEL_SOURCE
    name = try
        nameof(op)
    catch
        return _NO_KERNEL_SOURCE
    end
    (name isa Symbol && !startswith(string(name), "#")) ||
        return _NO_KERNEL_SOURCE
    mod = try
        parentmodule(op)
    catch
        return _NO_KERNEL_SOURCE
    end
    mod isa Module || return _NO_KERNEL_SOURCE
    # Fabricate a readable source only when the name resolves back to the
    # identical object; anything else renders through the generic op path.
    resolved = try
        isdefined(mod, name) && getfield(mod, name) === op
    catch
        false
    end
    resolved || return _NO_KERNEL_SOURCE
    Expr(:call, GlobalRef(mod, name), input_name)
end

# --- tuple pack/unpack -------------------------------------------------------
#
# A pack recipe lowers to an opaque fused closure, so unpack edges come from
# the authored pack shape in `recipe.source`, not from the operation. Each
# field whose entry is exactly a bare port gains one accessor recipe;
# computed entries occupy their position but unpack to no port, and a splat
# makes every later positional index unknowable so positional unpack stops
# there. Named fields are unaffected by splats.

function _kernel_named_pack_entry(arg)
    arg isa Symbol && return (arg, arg)
    arg isa Expr && (arg.head === :(=) || arg.head === :kw) &&
        length(arg.args) == 2 && arg.args[1] isa Symbol &&
        arg.args[2] isa Symbol || return nothing
    (arg.args[1], arg.args[2])
end

function _kernel_pack_fields(source)
    source isa Expr && source.head === :tuple || return nothing
    positioned = Any[]
    parameters = Any[]
    for arg in source.args
        if arg isa Expr && arg.head === :parameters
            append!(parameters, arg.args)
        else
            push!(positioned, arg)
        end
    end
    fields = Tuple{Symbol,Any,Symbol}[]
    for arg in parameters
        entry = _kernel_named_pack_entry(arg)
        entry === nothing || push!(fields, (:named, entry...))
    end
    mixed = !isempty(parameters)
    for (index, arg) in enumerate(positioned)
        if arg isa Symbol
            # A mixed positional/named shape is conservatively named-only:
            # the named part is sound regardless of the positional layout.
            mixed && continue
            push!(fields, (:positional, index, arg))
        elseif arg isa Expr && (arg.head === :(=) || arg.head === :kw) &&
               length(arg.args) == 2 && arg.args[1] isa Symbol
            entry = _kernel_named_pack_entry(arg)
            entry === nothing || push!(fields, (:named, entry...))
        elseif arg isa Expr && arg.head === :(...)
            break
        end
    end
    fields
end

function _kernel_synthesize_pack_edges!(graph::Graph, recipe::Recipe)
    length(recipe.outputs) == 1 || return false
    fields = _kernel_pack_fields(recipe.source)
    fields === nothing && return false
    packed = only(recipe.outputs)
    packed_id = canon_id(graph, packed.id)
    by_name = Dict{Symbol,Value}(v.name => v for v in recipe.inputs)
    for (kind, key, port) in fields
        target = get(by_name, port, nothing)
        target === nothing && continue
        canon_id(graph, target.id) == packed_id && continue
        op = kind === :named ? Base.Fix2(getproperty, key) :
             Base.Fix2(getindex, key)
        _kernel_has_equivalent_recipe(
            graph, packed_id, canon_id(graph, target.id), op) && continue
        source = kind === :named ? Expr(:., packed.name, QuoteNode(key)) :
                 Expr(:ref, packed.name, key)
        add!(graph; inputs = (packed,), outputs = (target,),
             op = op, cost = recipe.cost, source = source)
    end
    true
end

# --- display -----------------------------------------------------------------

_opname(f::Base.Fix1) = "Fix1($(_opname(f.f)), $(repr(f.x)))"
_opname(f::Base.Fix2) = "Fix2($(_opname(f.f)), $(repr(f.x)))"

function _readable_callee(f::Union{Base.Fix1,Base.Fix2})
    tag = f isa Base.Fix1 ? :Fix1 : :Fix2
    Expr(:call, GlobalRef(Base, tag), _readable_callee(f.f), f.x)
end
