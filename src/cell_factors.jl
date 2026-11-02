# Factor a composed scalar cell at its sums. All pieces remain ordinary graph
# recipes, so preparation, row pushdown and AD have one implementation.

# Typed aliases can separate an earlier reduction plate from the scalar cell
# that consumes it. Julia's typed local skips conversion when the value already
# satisfies the declaration. Only remove aliases proved to meet that condition;
# a declaration that can convert the value must remain part of the graph.
function _cell_factor_aliases(p::Plan, argument_types)
    candidates = filter(p.recipes) do r
        length(r.outputs) == 1 && !r.effectful && _cell_identity_form(r) !== nothing
    end
    isempty(candidates) && return p
    types = _cell_value_types(p; argument_types)
    redundant = Set(r.id for r in candidates if
        get(types, canon_id(p.graph, only(r.inputs).id), Any) <: valtype(only(r.outputs)))
    isempty(redundant) && return p
    old = p.graph
    g = Graph(copy(old.values), Recipe[], Dict{ValueId,Vector{Int}}(),
              copy(old.aliases), old.version, nothing)
    for r in p.recipes
        if r.id in redundant
            g.aliases[canon_id(g, only(r.outputs).id)] = canon_id(g, only(r.inputs).id)
        else
            _add_recipe!(g, r.inputs, r.outputs, r.op, r.cost, r.cse_key, r.effectful, r.source)
        end
    end
    plan(g; have = p.have, want = p.want)
end

function _cell_factor_graph(spec::KernelSpec, want; have = spec.have_names,
                            bound = NamedTuple(), on_error = nothing, locality = nothing)
    p = plan(spec; have, want)
    # Prove aliases with the declared live input types, not dense stand-ins:
    # factor value queries also accept other arrays with those declarations.
    p = _cell_factor_aliases(p, Any[haskey(bound, v.name) ? typeof(bound[v.name]) :
                                  valtype(v) for v in p.have])
    argument_types = Any[haskey(bound, v.name) ? typeof(bound[v.name]) :
                         _cell_standin_type(valtype(v)) for v in p.have]
    p = _fuse_authored_plate_chains(p)
    types = _uninferred_values!(_UninferredValue[], p, argument_types, Symbol[], true)
    known = Dict{ValueId,Any}(canon_id(p.graph, v.id) => bound[v.name]
                             for v in p.have if haskey(bound, v.name))
    prefix, residual, owned = _partial_split(p, Set(keys(known)))
    frontier = _partial_constants(p, owned, residual)
    if !isempty(frontier)
        data_have = Value[v for v in p.have if haskey(known, canon_id(p.graph, v.id))]
        data_plan = _partial_subplan(p, data_have, frontier, prefix)
        data_kernel = prepare(data_plan; on_error)
        result = Base.invokelatest(data_kernel,
            (known[canon_id(p.graph, v.id)] for v in data_have)...)
        for (v, x) in zip(frontier, _partial_prefix_values(frontier, result))
            known[canon_id(p.graph, v.id)] = x
        end
    end
    shapes = Dict{ValueId,Any}()
    function data_shape(v)
        id = canon_id(p.graph, v.id)
        get!(shapes, id) do
            haskey(known, id) && return axes(Base.broadcastable(known[id]))
            T = get(types, id, Any)
            T isa Type && T <: Number && return ()
            r = get(p.producer, id, nothing)
            r === nothing && return nothing
            form = _cell_form(r)
            form === nothing && return nothing
            # A form whose axes come from more than its leaves (a transposed
            # vector, a gather keeping axes of its source) is not shaped here.
            if form.kind in (:pointwise, :identity) && isempty(_cell_extra_parts(form))
                leaves = [data_shape(r.inputs[j]) for j in form.leaves]
                any(isnothing, leaves) && return nothing
                return _plate_broadcast_shape(Tuple(leaves))
            elseif form.kind === :range_view && all(x -> x isa Int, form.literal)
                return _cell_range_shape(Tuple(form.literal))
            end
            nothing
        end
    end
    original = p.graph
    g = Graph(copy(original.values), Recipe[], Dict{ValueId,Vector{Int}}(),
              copy(original.aliases), original.version, nothing)
    ports, order = copy(spec.ports), copy(spec.port_order)
    serial = Ref(0)
    function fresh(label, type = Any)
        name = Symbol(:_rk_factor_, label, :_, (serial[] += 1))
        while haskey(ports, name)
            name = Symbol(:_rk_factor_, label, :_, (serial[] += 1))
        end
        v = value!(g, name, type)
        ports[name] = v
        push!(order, name)
        v
    end
    sums = NamedTuple[]
    cell_outputs = Set{ValueId}()
    tuple_fields = Dict{ValueId,Tuple}()
    function emit(r, mapping = nothing, readable = r)
        mapvalue(v) = mapping === nothing ? v : mapping[canon_id(mapping_graph[], v.id)]
        ins = Tuple(mapvalue(v) for v in r.inputs)
        outs = Tuple(mapvalue(v) for v in r.outputs)
        # Composition can leave a cell's tuple constructor beside its field
        # reads. Preserve each field's own dependencies: a tuple of arrays is
        # not an elementwise array operation, but reading its known field is
        # an alias. A declared tuple may convert its fields, so leave it alone.
        entry = _constant_tuple_entry(readable)
        fields = entry === nothing ? nothing : get(tuple_fields, canon_id(g, only(ins).id), nothing)
        if fields !== nothing && 1 <= entry <= length(fields) &&
                (valtype(only(outs)) === Any || valtype(only(outs)) == valtype(fields[entry]))
            _kernel_alias!(g, only(outs), fields[entry], identity, 0.0)
            return
        end
        if r.op isa _KernelSourceOp && length(outs) == 1 && !r.effectful &&
                valtype(only(outs)) === Any && Meta.isexpr(r.source, :tuple, length(ins)) &&
                r.source.args == _recipe_source_names(readable)
            tuple_fields[canon_id(g, only(outs).id)] = ins
        end
        parts = _partial_split_parts(r; factors = true)
        if parts !== nothing
            # Asking the shared splitter for every leaf exposes every
            # boundary, retaining the same functions and error-policy twins.
            mapped = Recipe(r.id, ins, outs, r.op, r.cost, r.cse_key, r.effectful, r.source)
            expanded = Recipe[]
            leaves = Bool[isempty(n.kids) for n in parts.split.nodes]
            _partial_emit_split!(expanded, Dict{Int,Any}(), g, mapped, parts, leaves,
                                 () -> (serial[] += 1))
            for part in expanded
                q = _add_recipe!(g, part.inputs, part.outputs, part.op, part.cost,
                                 nothing, false, part.source)
                mapping === nothing || union!(cell_outputs, (v.id for v in q.outputs))
                record(q, part === last(expanded) ? only(r.outputs).name : only(part.outputs).name)
            end
        else
            q = _add_recipe!(g, ins, outs, r.op, r.cost, r.cse_key, r.effectful, r.source)
            mapping === nothing || union!(cell_outputs, (v.id for v in q.outputs))
            record(q, first(r.outputs).name)
        end
    end
    function record(r, name)
        length(r.outputs) == 1 || return
        argument = if r.op === Base.sum && length(r.inputs) == 1
            only(r.inputs)
        else
            analyzed = _cell_native(r; outlined = true)
            analyzed === nothing && return
            native, params, mod = analyzed
            _cell_sum_source(native) || return
            _native_callee_shadowed(native.args[1], params) && return
            _cell_binding(native.args[1], mod) === Base.sum || return
            input = findfirst(==(native.args[2]), params)
            input === nothing && return
            r.inputs[input]
        end
        push!(sums, (; name, value = only(r.outputs), argument))
    end
    mapping_graph = Ref{Graph}(g)
    for r in p.recipes
        if !(r.op isa _AuthoredPlateCellOp)
            emit(r)
            continue
        end
        r.op.rows && throw(ArgumentError("cell factors require a single cell"))
        body = r.op.plate.kernel.plan
        atomic = _plate_atomic(r.op.plate)
        batched = [i for i in eachindex(body.have) if !(i in atomic)]
        domain_inputs = [known[canon_id(p.graph, r.inputs[j + 1].id)]
                         for j in batched if haskey(known, canon_id(p.graph, r.inputs[j + 1].id))]
        data_axes = Base.Broadcast.combine_axes(domain_inputs...)
        mapping = Dict{ValueId,Value}()
        for (i, v) in enumerate(body.have)
            mapping[canon_id(body.graph, v.id)] = if i in atomic
                r.inputs[i + 1]
            else
                # Project a scalar input through a cell with the complete
                # original broadcast domain, including absorbed axis checks.
                projected = filter(batched) do j
                    id = canon_id(p.graph, r.inputs[j + 1].id)
                    T = get(types, id, Any)
                    # Infer axes through row forms without evaluating a live
                    # value. An operand whose axes add nothing is not needed
                    # to project another operand; its own factor still checks
                    # it. This keeps row metadata independent of elementwise
                    # scale arrays, including on singleton data domains.
                    shape = data_shape(r.inputs[j + 1])
                    j == i || haskey(known, id) || !(T isa Type &&
                        (T <: Number || shape !== nothing &&
                         _plate_broadcast_shape((data_axes, shape)) == data_axes))
                end
                small = Graph()
                roots = [value!(small, Symbol(:arg_, j), valtype(body.have[j])) for j in projected]
                scalar = prepare(small; have = roots, want = roots[findfirst(==(i), projected)])
                checks = Tuple(Tuple(findfirst(==(j), projected) for j in check if j in projected)
                               for check in r.op.plate.axis_checks if any(j -> j in projected, check))
                plate = _AuthoredPlateOp{typeof(scalar),()}(scalar, checks)
                output = fresh(:input, valtype(v))
                _add_recipe!(g, (first(r.inputs), Tuple(r.inputs[j + 1] for j in projected)...),
                    (output,), _AuthoredPlateCellOp(plate, r.op.locality), r.cost,
                    nothing, false, nothing)
                output
            end
        end
        for q in body.recipes, v in q.outputs
            mapping[canon_id(body.graph, v.id)] = fresh(v.name, valtype(v))
        end
        mapping_graph[] = body.graph
        originals = r.op.plate.kernel.lowered_recipes
        aligned = length(originals) == length(body.recipes)
        for (j, q) in enumerate(body.recipes)
            # Composed recipes retain their original source argument names.
            # Use that metadata only when it still describes this operation;
            # a differently lowered body must never drop recipes or alias a
            # field based on another operation's argument ordering.
            original = aligned ? originals[j] : q
            readable = original.op === q.op && isequal(original.source, q.source) &&
                length(original.inputs) == length(q.inputs) &&
                length(original.outputs) == length(q.outputs) ? original : q
            emit(q, mapping, readable)
        end
        output = mapping[canon_id(body.graph, only(body.want).id)]
        _add_recipe!(g, (output,), r.outputs, identity, 0.0, nothing, false, output.name)
    end
    result = KernelSpec(g, ports, order, Symbol[v.name for v in p.have],
                        Symbol[v.name for v in p.want], nothing)
    result = _cell_factor_scalar_reads(result, cell_outputs; bound, locality)
    (; spec = result, sums)
end

# Flattening a plate leaves scalar reads of shared arrays as ordinary recipes.
# Recover their row demand after splitting: otherwise a direct `eta[index]`
# next to a sum materializes all of eta, even when the sum is an independent
# input to the outer factor. Use the same identity cell as a summand, so value
# lowering, footprints and AD all follow the selected row. Type inference
# distinguishes a scalar vector read from a gathered vector or matrix slice.
function _cell_factor_scalar_reads(spec::KernelSpec, cell_outputs; bound, locality)
    p = plan(spec)
    argument_types = Any[haskey(bound, v.name) ? typeof(bound[v.name]) :
                         _cell_standin_type(valtype(v)) for v in p.have]
    # A cell's index port may be untyped (linear or Cartesian). Its result
    # type comes from the scalar body, independently of that index type.
    types = _plan_value_types(p, argument_types) do r, inputs
        T = _checked_recipe_type!(_UninferredValue[], r.op, inputs, Symbol[])
        T === nothing ? map(valtype, r.outputs) : _recipe_output_types(r, T)
    end
    old = spec.graph
    g = Graph(copy(old.values), Recipe[], Dict{ValueId,Vector{Int}}(),
              copy(old.aliases), old.version, nothing)
    for r in p.recipes
        inputs, op = r.inputs, r.op
        # Top-level recipes retain their original context. Only reads
        # exposed by flattening a cell need their lost row demand restored.
        analyzed = any(v -> v.id in cell_outputs, r.outputs) ?
            _cell_native(r; outlined = true) : nothing
        if length(r.outputs) == 1 && !r.effectful && analyzed !== nothing
            read = _cell_scalar_index(analyzed..., r.inputs,
                v -> get(types, canon_id(old, v.id), Any))
            if read !== nothing
                i = read.index
                if !(i isa Value)
                    i = value!(g, Symbol(only(r.outputs).name, :_index), read.type)
                    _add_recipe!(g, (), (i,), _BoundConstant(read.index), 0.0, nothing,
                                 false, read.index)
                end
                inputs, op = (i, read.array), _cell_index_op(locality)
            end
        end
        _add_recipe!(g, inputs, r.outputs, op, r.cost, r.cse_key, r.effectful, r.source)
    end
    KernelSpec(g, copy(spec.ports), copy(spec.port_order), copy(spec.have_names),
               copy(spec.want_names), nothing)
end

function _copy_cell_factor_spec(spec::KernelSpec; omit = Set{ValueId}())
    old = spec.graph
    g = Graph(copy(old.values), Recipe[], Dict{ValueId,Vector{Int}}(),
              copy(old.aliases), old.version, nothing)
    for r in old.recipes
        any(v -> v.id in omit, r.outputs) && continue
        _add_recipe!(g, r.inputs, r.outputs, r.op, r.cost, r.cse_key, r.effectful, r.source)
    end
    KernelSpec(g, copy(spec.ports), copy(spec.port_order), copy(spec.have_names),
               copy(spec.want_names), nothing)
end

# The first gather reached from the term supplies its row labels. Several
# operands may gather different arrays: their positions correspond by the
# authored broadcast, and the term query retains every operand.
function _cell_factor_rows(spec::KernelSpec, argument::Value)
    g = spec.graph
    seen = Set{ValueId}()
    function walk(v)
        id = canon_id(g, v.id)
        id in seen && return nothing
        push!(seen, id)
        ids = producers_of(g, id)
        length(ids) == 1 || return nothing
        r = g.recipes[only(ids)]
        form = _cell_form(r)
        form === nothing && return nothing
        index = findfirst(==(:index), form.roles)
        index === nothing || return r.inputs[index]
        for (i, input) in enumerate(r.inputs)
            _cell_row_role(form.roles[i]) || continue
            rows = walk(input)
            rows === nothing || return rows
        end
        nothing
    end
    walk(argument)
end

# Make one term an ordinary identity cell over the reduction's operand. Row
# demand then propagates through its elementwise recipes and gathers.
function _cell_factor_term!(spec::KernelSpec, argument::Value, index::Value, locality)
    output = value!(spec.graph, Symbol(argument.name, :_term), Any)
    spec.ports[output.name] = output
    push!(spec.port_order, output.name)
    _add_recipe!(spec.graph, (index, argument), (output,),
        _cell_index_op(locality), 1.0, nothing, false, nothing)
    output
end
