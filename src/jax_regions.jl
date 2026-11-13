# Source regions are outlined from selected recipes, not from printed Python.
# All cuts retain the original graph and producer assignments. Only the JAX
# execution boundary changes; native planning continues to see every recipe.

# Distribution expansion introduces values after the PPL ports were named.
# Derive their lexical spelling from the same provenance as the authored
# locals, before emitting bindings, so equal instances share executable code.
function _jax_region_names(g, region)
    substitutions = sort!([_jax_identifier(Set{String}(), from) =>
                            _jax_identifier(Set{String}(), to) for (from, to) in region.names];
                          by = p -> (-length(first(p)), first(p)))
    names = copy(region.names)
    for v in values(g.values)
        haskey(names, v.name) && continue
        label = _jax_identifier(Set{String}(), v.name)
        for (from, to) in substitutions
            label = replace(label, Regex("(?<![A-Za-z0-9])" * from * "(?![A-Za-z0-9])") => to)
        end
        names[v.name] = Symbol(label)
    end
    names
end

function _jax_region_members(p, region)
    g = p.graph
    boundary = Set(canon_id(g, v.id) for v in p.have)
    union!(boundary, (canon_id(g, v.id) for v in values(g.values) if v.name in region.inputs))
    selected = Set(r.id for r in p.recipes)
    members = Set{Int}()
    function visit(v)
        id = canon_id(g, v.id)
        id in boundary && return
        r = get(p.producer, id, nothing)
        r === nothing && return
        r.id in selected && !(r.id in members) || return
        push!(members, r.id)
        foreach(visit, r.inputs)
    end
    foreach(visit, (v for v in values(g.values) if v.name in region.outputs))
    members
end

function _jax_region_boundary(p, members)
    g = p.graph
    produced = Set(id for (id, r) in p.producer if r.id in members)
    have = Value[]
    for r in p.recipes
        r.id in members || continue
        for v in r.inputs
            canon_id(g, v.id) in produced || push!(have, v)
        end
    end
    # A stable rewrite across the boundary (log(exp(x)), for example) reads
    # the boundary producer's operands. Pass those captures explicitly too.
    for v in copy(have)
        r = get(p.producer, canon_id(g, v.id), nothing)
        r === nothing && continue
        length(r.outputs) == 1 && _jax_stable_view(r) !== nothing || continue
        append!(have, (a for a in r.inputs if !(canon_id(g, a.id) in produced)))
    end
    unique!(v -> canon_id(g, v.id), have)
    wanted = Set(canon_id(g, v.id) for v in p.want)
    for r in p.recipes
        r.id in members && continue
        union!(wanted, (canon_id(g, v.id) for v in r.inputs))
        for v in r.inputs
            parent = get(p.producer, canon_id(g, v.id), nothing)
            parent === nothing && continue
            parent.id in members && length(parent.outputs) == 1 &&
                _jax_stable_view(parent) !== nothing || continue
            union!(wanted, (canon_id(g, a.id) for a in parent.inputs))
        end
    end
    for c in p.checks, a in c.args
        a isa Value && push!(wanted, canon_id(g, a.id))
    end
    want = Value[v for r in p.recipes if r.id in members for v in r.outputs
                  if canon_id(g, v.id) in wanted && canon_id(g, v.id) in produced]
    unique!(v -> canon_id(g, v.id), want)
    have, want
end

# Contract each lexical region to a scheduling unit. Contraction may expose a
# cycle when a caller consumes an early local and feeds a later local back
# through an argument. Keep the ordinary recipe order in that case; the graph
# remains executable rather than acquiring a new authoring restriction.
function _jax_region_schedule(p, regions)
    ordinary() = [(region = nothing, members = Set([r.id]), have = collect(Value, r.inputs),
                   want = collect(Value, r.outputs)) for r in p.recipes]
    isempty(regions) && return ordinary()
    claimed = Set{Int}()
    units = Any[]
    for region in regions
        members = _jax_region_members(p, region)
        isempty(members) && continue
        isempty(intersect(members, claimed)) || continue
        have, want = _jax_region_boundary(p, members)
        isempty(want) && continue
        union!(claimed, members)
        push!(units, (region = region, members = members, have = have, want = want))
    end
    isempty(claimed) && return ordinary()
    for r in p.recipes
        r.id in claimed || push!(units,
            (region = nothing, members = Set([r.id]), have = collect(Value, r.inputs),
             want = collect(Value, r.outputs)))
    end
    available = Set(canon_id(p.graph, v.id) for v in p.have)
    ordered = Any[]
    while !isempty(units)
        index = findfirst(u -> all(canon_id(p.graph, v.id) in available for v in u.have), units)
        if index === nothing
            return ordinary()
        end
        unit = splice!(units, index)
        push!(ordered, unit)
        union!(available, (canon_id(p.graph, v.id) for v in unit.want))
    end
    ordered
end

_jax_entry_condition(selector, needed, ::Val{false}) =
    selector * " in " * _jax_py_literal(Tuple(String.(needed)))
_jax_entry_condition(selector, needed, ::Val{true}) =
    "any(name in " * selector * " for name in " * _jax_py_literal(Tuple(String.(needed))) * ")"

function _jax_demanded_recipes!(ctx, p, demand, labels, selector, available;
                                 outputs = Val(false), regions = ctx.mod.regions)
    for unit in _jax_region_schedule(p, regions)
        needed = [label for label in labels if any(label in demand[id] for id in unit.members)]
        guarded = needed != labels || outputs === Val(true)
        guarded && _jax_emit!(ctx, "if " * _jax_entry_condition(selector, needed, outputs) * ":")
        guarded && (ctx.scope.indent += 1)
        if unit.region === nothing
            r = p.graph.recipes[only(unit.members)]
            saved = ctx.scope.values
            ctx.scope.values = Dict(id => value for (id, value) in saved
                if all(in(available[id]), needed))
            try
                _jax_plan_recipe!(ctx, p, r)
                for v in r.outputs
                    id = canon_id(p.graph, v.id)
                    haskey(saved, id) && continue
                    saved[id] = ctx.scope.values[id]
                    available[id] = needed
                end
            finally
                ctx.scope.values = saved
            end
        else
            _jax_region_call!(ctx, p, unit, demand, labels, selector, available, outputs)
        end
        guarded && (ctx.scope.indent -= 1)
    end
end

function _jax_region_call!(ctx, p, unit, demand, labels, selector, available, mode)
    g, region = p.graph, unit.region
    names = _jax_region_names(g, region)
    # Graph aliases can span several call sites (a shared log(scale), for
    # example). Prefer this region's lexical alias over another instance's
    # name; a global last-alias choice would duplicate identical functions.
    rank(v) = (haskey(region.names, v.name) ? 0 :
               names[v.name] != Symbol(_jax_identifier(Set{String}(), v.name)) ? 1 : 2,
               length(String(names[v.name])), String(names[v.name]), String(v.name))
    hints = Dict{ValueId,Symbol}()
    for v in sort!(collect(values(g.values)); by = rank)
        get!(hints, canon_id(g, v.id), names[v.name])
    end
    hint(v) = get(hints, canon_id(g, v.id), v.name)
    have = sort(unit.have; by = v -> String(hint(v)))
    want = sort(unit.want; by = v -> String(hint(v)))
    recipes = [r for r in p.recipes if r.id in unit.members]
    q = Plan(g, have, want, recipes, p.producer, 0.0, Recipe[], InputCheck[])
    inputs = [_jax_lookup(ctx, g, v) for v in have]
    scope = _JaxScope()
    scope.named_hints = names
    push!(scope.used, "outputs", "results")
    params, passed = String[], String[]
    argument_slots = UnitRange{Int}[]
    for (v, a) in zip(have, inputs)
        start = length(passed) + 1
        value = _jax_function_input!(ctx.mod, scope, params, passed, hint(v), a)
        scope.values[canon_id(g, v.id)] = value
        push!(argument_slots, start:length(passed))
    end
    guard = _jax_function_guard!(ctx, scope, params, passed)
    for v in values(g.values)
        scope.hints[canon_id(g, v.id)] = hint(v)
    end
    keys = String[]
    used = Set{String}()
    for v in want
        push!(keys, _jax_identifier(used, hint(v)))
    end
    # Each requested output induces its own backward cut of this region.
    # Its dependencies guard the body and the caller's optional arguments.
    local_demand = Dict(r.id => String[] for r in recipes)
    input_demand = Dict(canon_id(g, v.id) => String[] for v in have)
    for (v, key) in zip(want, keys)
        seen = Set{Int}()
        function visit(value)
            id = canon_id(g, value.id)
            if haskey(input_demand, id)
                key in input_demand[id] || push!(input_demand[id], key)
                r = get(p.producer, id, nothing)
                if r !== nothing && length(r.outputs) == 1 && _jax_stable_view(r) !== nothing
                    for a in r.inputs
                        aid = canon_id(g, a.id)
                        haskey(input_demand, aid) && !(key in input_demand[aid]) &&
                            push!(input_demand[aid], key)
                    end
                end
                return
            end
            r = get(p.producer, id, nothing)
            (r === nothing || !(r.id in unit.members) || r.id in seen) && return
            push!(seen, r.id)
            push!(local_demand[r.id], key)
            foreach(visit, r.inputs)
        end
        visit(v)
    end
    inner = _JaxContext(ctx.mod, scope, ctx.recipe, ctx.source_module, Dict{Symbol,_PyVal}(), nothing)
    inner.traced_branches = ctx.traced_branches
    _jax_emit!(inner, "results = {}")
    inner_available = copy(input_demand)
    nested = [r for r in ctx.mod.regions if r !== region &&
              !isempty(_jax_region_members(q, r)) &&
              _jax_region_members(q, r) != unit.members]
    _jax_with_function_guard(scope, guard) do
        _jax_demanded_recipes!(inner, q, local_demand, keys, "outputs", inner_available;
                                outputs = Val(true), regions = nested)
    end
    results = _PyVal[]
    for (v, key) in zip(want, keys)
        _jax_emit!(inner, "if " * repr(key) * " in outputs:")
        scope.indent += 1
        result = _jax_returned(inner, _jax_lookup(inner, g, v))
        push!(results, result)
        pieces = [result.code]
        isempty(result.extent) || push!(pieces, result.extent)
        isempty(result.limit) || push!(pieces, result.limit)
        returned = length(pieces) == 1 ? only(pieces) : "(" * join(pieces, ", ") * ")"
        _jax_emit!(inner, "results[" * repr(key) * "] = " * returned)
        scope.indent -= 1
    end
    _jax_emit!(inner, "return results")
    text = join(scope.lines, "\n")
    executable = replace(text, _PY_STRING => "\"\"")
    used_params = findall(n -> occursin(Regex("\\b" * n * "\\b"), executable), params)
    bounds = [bn for (bn, _, _) in _jax_prepared_bounds(ctx.mod)
              if occursin(Regex("\\b" * bn * "\\b"), text)]
    signature = [n * ": ArrayLike" for n in params[used_params]]
    append!(signature, [n * ": int" for n in bounds])
    append!(signature, ["*", "outputs: tuple[str, ...]"])
    cache_key = (:source_region, region.name, Tuple(signature), text, region.doc)
    name = get(ctx.mod.helper_names, cache_key, nothing)
    if name === nothing
        has_value = any(v -> get(names, v.name, v.name) === :value, want)
        hint = has_value ? region.name : Symbol(region.name, :_density)
        name = _jax_identifier(ctx.mod.used, hint)
        ctx.mod.helper_names[cache_key] = name
        push!(ctx.mod.helpers, join([String["def " * name * "(" * join(signature, ", ") * ") -> dict:",
            _jax_docstring(region.doc, "    ")]; scope.lines], "\n"))
    end
    output_demand = Dict(key => demand[p.producer[canon_id(g, v.id)].id]
                        for (v, key) in zip(want, keys))
    for (v, slots) in zip(have, argument_slots)
        needed = [label for label in labels if any(label in output_demand[key]
                    for key in input_demand[canon_id(g, v.id)])]
        if needed != labels || mode === Val(true)
            condition = _jax_entry_condition(selector, needed, mode)
            for slot in slots
                passed[slot] = "(" * passed[slot] * " if " * condition * " else None)"
            end
        end
    end
    selected = "tuple(name for name, needed in " * _jax_py_literal(Tuple(
        (key, Tuple(String.(output_demand[key]))) for key in keys)) *
        " if " * (mode === Val(true) ? "any(n in " * selector * " for n in needed)" : selector * " in needed") * ")"
    call = name * "(" * join([passed[used_params]; bounds; "outputs=" * selected], ", ") * ")"
    result = _jax_bind!(ctx, first(region.outputs), _PyVal(call, Any))
    for (v, key, value) in zip(want, keys, results)
        id = canon_id(g, v.id)
        code = result.code * "[" * repr(key) * "]"
        extent = isempty(value.extent) ? "" : code * "[1]"
        limit = isempty(value.limit) ? "" : code * (isempty(extent) ? "[1]" : "[2]")
        isempty(extent) && isempty(limit) || (code *= "[0]")
        ctx.scope.values[id] = _PyVal(code, value.T, value.static, extent, limit, value.padded, value.origin)
        available[id] = output_demand[key]
    end
end
