# Native forward sampling runs at a query boundary. Its program retains ordinary
# Julia broadcasts and calls the same author bindings that the density resolved.
_is_generate(st) = Meta.isexpr(st, :macrocall) && first(st.args) === Symbol("@generate")
function _generated_statements(st)
    length(st.args) == 3 || _sfail("@generate takes a declaration or a begin block")
    body = last(st.args)
    return Meta.isexpr(body, :block) ? body.args : (body,)
end

function _draw_source(ast, mod, inputs, observed, scopes)
    generated = Expr(:block)
    fitted = ast
    fitted_scopes = scopes
    if any(_is_generate, ast.args)
        fitted = Expr(:block)
        for st in ast.args
            if _is_generate(st)
                append!(generated.args, _generated_statements(st))
            else
                push!(fitted.args, st)
            end
        end
        # Only the indexed columns escape a plate. Its iteration variable
        # and bare cell locals can have the same names as other model locals.
        gen_names = Set{Symbol}()
        for st in generated.args
            loop = _plate_loop(st)
            if loop === nothing
                _collect_binders!(gen_names, st)
            else
                union!(gen_names, _draw_plate_columns(loop))
            end
        end
        overlap = intersect(gen_names, union(inputs, _collect_binders!(Set{Symbol}(), fitted)))
        isempty(overlap) || _sfail("@generate redefines a fitted or input name: $(first(overlap))")
        reads = intersect(gen_names, _statement_value_reads!(Set{Symbol}(), fitted))
        isempty(reads) || _sfail("the fitted program reads generated value `$(first(reads))`")
        fitted_scopes = filter(s -> s.binding ∉ gen_names, scopes)
    end
    source = _DrawSource(fitted, generated, mod, copy(inputs), copy(observed), scopes)
    return fitted, source, fitted_scopes
end

"""
    sampling_draw(rng, rhs)

Draw one constrained event from the constructed sampling RHS. Implement this
method on a caller-owned law alongside `sampling_logdensity`. Only the supplied
random stream may be advanced. Conditional arguments belong to the RHS object.
Loading Distributions supplies this method for its distributions.
"""
sampling_draw(rng, rhs) = throw(ArgumentError("sampling RHS $(typeof(rhs)) needs a sampling_draw method"))

function _generation_constructor(::Val{name}) where {name}
    throw(ArgumentError("no native draw constructor for `$name`; the Distributions extension supplies the supported built-in laws"))
end

# Substitute only the distribution constructor. Parameters, keywords and
# caller-owned constructors keep their Julia meaning in the defining module.
function _generation_law(rhs, mod, names, site)
    dotted = _is_dotted_call(rhs)
    if Meta.isexpr(rhs, :call) || dotted
        head = first(rhs.args)
        if head isa Symbol && _builtin_rhs_head(head)
            head in (:flat, :Flat) && _sfail("draw at `$site`: an improper flat density has no normalized conditional generator")
            head === :restricted && _sfail("draw at `$site`: a restricted density needs a normalized joint generation law")
            # Name the public package in the executable expression. Extensions
            # can provide a constructor without binding their module in Main.
            _generation_constructor(Val(head))
            ctor = :($(GlobalRef(@__MODULE__, :_generation_constructor))(Val($(QuoteNode(head)))))
            args = dotted ? rhs.args[2].args : rhs.args[2:end]
            vals = if head in (:truncated, :Ordered)
                Any[_generation_law(first(args), mod, names, site),
                    (_resolve_module_calls(a, mod, names, site) for a in args[2:end])...]
            else
                Any[_resolve_module_calls(a, mod, names, site) for a in args]
            end
            return dotted ? Expr(:., ctor, Expr(:tuple, vals...)) : Expr(:call, ctor, vals...)
        end
    end
    return _resolve_module_calls(rhs, mod, names, site)
end

# The axes on a declaration describe the output domain. Broadcast a shared
# law across the domain without constructing a copy per cell. One whole-event
# law is sampled once and checked against its declared shape.
_draw_axis(axis) = length(axis)
_draw_axis(axis::Integer) = axis
function _draw_event(rng, law, shape::Tuple)
    value = sampling_draw(rng, law)
    isempty(shape) || (value isa AbstractArray && size(value) == shape) ||
        throw(DimensionMismatch("generated event has size $(size(value)), expected $shape"))
    return value
end

function _draw_elements(rng, laws, shape::Tuple)
    if isempty(shape)
        return laws isa AbstractArray ? map(law -> sampling_draw(rng, law), laws) : sampling_draw(rng, laws)
    end
    domain = CartesianIndices(shape)
    laws isa AbstractArray || return map(_ -> sampling_draw(rng, laws), domain)
    Base.Broadcast.combine_axes(domain, laws) == axes(domain) ||
        throw(DimensionMismatch("broadcast law does not fit declared shape $shape"))
    return broadcast((_, law) -> sampling_draw(rng, law), domain, laws)
end

# Each slice is one whole vector event. The distribution arguments can be
# shared or broadcast over the slices, just as they are in the density.
function _draw_slices(rng, laws, shape::Tuple{Int,Int}, dim)
    draws = _draw_elements(rng, laws, (shape[dim],))
    width = shape[3 - dim]
    all(x -> x isa AbstractVector && length(x) == width, draws) ||
        throw(DimensionMismatch("generated vector does not match its declared slice width $width"))
    isempty(draws) && return Matrix{eltype(eltype(draws))}(undef, shape)
    return stack(draws; dims=dim)
end

# A grouping axis is the same ordered set of levels used by array layout.
function _draw_axis_expr(ex)
    ex isa Expr || return ex
    if Meta.isexpr(ex, :call, 2) && first(ex.args) === :levels
        return Expr(:call, GlobalRef(@__MODULE__, :_ppl_levels), ex.args[2])
    end
    return Expr(ex.head, map(_draw_axis_expr, ex.args)...)
end

function _draw_site(st, name, mod, names, rng)
    lhs, rhs = st.args[2], st.args[3]
    slice = Meta.isexpr(lhs, :call, 2) && lhs.args[1] in (:eachrow, :eachcol)
    target = slice ? lhs.args[2] : lhs
    target isa Symbol || Meta.isexpr(target, :ref) ||
        _sfail("generation at `$name` needs a whole event or declared array axes")
    axes = target isa Symbol ? () : target.args[2:end]
    shape = Expr(:tuple, (Expr(:call, GlobalRef(@__MODULE__, :_draw_axis),
        _resolve_module_calls(_draw_axis_expr(a), mod, names, name)) for a in axes)...)
    law = _generation_law(rhs, mod, names, name)
    if slice
        length(axes) == 2 || _sfail("generation at `$name` needs two declared matrix axes")
        dim = lhs.args[1] === :eachrow ? 1 : 2
        return Expr(:call, GlobalRef(@__MODULE__, :_draw_slices), rng, law, shape, dim)
    end
    draw = _is_broadcast_sample(st) ? :_draw_elements : :_draw_event
    return Expr(:call, GlobalRef(@__MODULE__, draw), rng, law, shape)
end

struct _NativeDrawQuery{Role,F,D,L}
    program::F
    inputs::D
    layout::L
end

# Read one physical site or an axis from the current binding, including the
# input spelling of pinned/conditioned values and bound scalar definitions.
function _draw_input(plan, name)
    for key in (_conditioned_input(name), _bound_value_input(name), name)
        haskey(plan.columns, key) || continue
        v = plan.columns[key]
        _supplied(v) || _sfail("generation reads absent input `$name`")
        return v
    end
    _sfail("generation needs a value for `$name`")
end

function _draw_declarations(stmts, observed; generated = false, factors = Set{Symbol}())
    defs = Dict{Symbol,Expr}()
    order = Symbol[]
    function declare(name, st)
        haskey(defs, name) && _sfail("joint generation needs one conditional law for `$name`; a second factor does not specify a draw")
        defs[name] = st
        push!(order, name)
    end
    for st in stmts
        st isa LineNumberNode && continue
        loop = _plate_loop(st)
        if generated && loop !== nothing
            foreach(n -> declare(n, st), _draw_plate_columns(loop))
            continue
        elseif _is_sample(st) || _is_broadcast_sample(st) || Meta.isexpr(st, :(=), 2)
            name = _merge_stem(_stmt_lhs(st))
            name in observed && continue
            name in factors && !Meta.isexpr(st, :(=)) && continue
            if name isa Symbol
                declare(name, st)
                continue
            end
        end
        generated && _sfail("unsupported generated declaration `$st`")
    end
    return defs, order
end

# The LHS visible outside a plate loses one cell index. A nested plate
# writing x[i][j] first produces x[i] and the outer plate then produces x.
function _draw_plate_columns(loop)
    index = loop.args[1].args[1]
    columns = Any[]
    for st in loop.args[2].args
        st isa LineNumberNode && continue
        inner = _plate_loop(st)
        targets = if inner !== nothing
            _draw_plate_columns(inner)
        elseif _is_sample(st) || _is_broadcast_sample(st) || Meta.isexpr(st, :(=), 2)
            (_stmt_lhs(st),)
        else
            _sfail("unsupported generated plate declaration `$st`")
        end
        for lhs in targets
            lhs isa Symbol && continue
            target = Meta.isexpr(lhs, :call, 2) && first(lhs.args) in (:eachrow, :eachcol) ?
                lhs.args[2] : lhs
            while Meta.isexpr(target, :ref) && !(length(target.args) == 2 && target.args[2] === index)
                target = target.args[1]
            end
            Meta.isexpr(target, :ref, 2) || _sfail("generated plate output `$lhs` needs its cell index `$index`")
            push!(columns, target.args[1])
        end
    end
    return unique!(columns)
end

_draw_cell_refs(ex, index, locals) = ex
function _draw_cell_refs(ex::Expr, index, locals)
    Meta.isexpr(ex, :ref, 2) && ex.args[2] === index && haskey(locals, ex.args[1]) &&
        return locals[ex.args[1]]
    ex.head === :quote && return ex
    return Expr(ex.head, (_draw_cell_refs(a, index, locals) for a in ex.args)...)
end

_draw_rhs(st, name, mod, names, rng) = Meta.isexpr(st, :(=), 2) ?
    _resolve_module_calls(last(st.args), mod, names, name) : _draw_site(st, name, mod, names, rng)

# Construct the needed definitions in dependency order. The same traversal
# orders top-level draws and the declarations within one repeated call.
function _draw_lines(valueof, outputs, names; order=outputs)
    needed = Dict{Symbol,Tuple{Any,Vector{Symbol}}}()
    function visit(n)
        haskey(needed, n) && return
        value = valueof(n)
        deps = sort!(collect(intersect(_value_symbols(value), names)); by=string)
        needed[n] = (value, deps)
        foreach(visit, deps)
    end
    foreach(visit, outputs)
    lines = Expr[]
    active = Set{Symbol}()
    function emit(n)
        n in active && _sfail("cyclic generation dependency at `$n`")
        haskey(needed, n) || return
        value, deps = pop!(needed, n)
        push!(active, n)
        foreach(emit, deps)
        push!(lines, Expr(:(=), n, value))
        delete!(active, n)
    end
    foreach(emit, order)
    return lines
end

function _draw_records(cell, captures, indices)
    T = ReactiveKernels._promote_op(cell, typeof(captures), eltype(indices))
    collect(T, Base.Generator(ReactiveKernels._PackedComprehensionElement(cell, captures), indices))
end

# Draw a record in one invocation of the cell. Only columns read outside
# this plate leave the cell. Selecting another local projects that same row.
function _draw_plate_values(defs, declaration_order, outputs, mod, names, rng, cells)
    expressions = Dict{Symbol,Expr}()
    for plate in unique(defs[n] for n in declaration_order if _plate_loop(defs[n]) !== nothing)
        loop = _plate_loop(plate)
        index, domain = loop.args[1].args
        columns = _draw_plate_columns(loop)
        outside = Set(outputs)
        for st in values(defs)
            st === plate || union!(outside, _value_symbols(st))
        end
        exports = filter(in(outside), columns)
        isempty(exports) && continue
        locals = Dict(n => gensym(n) for n in columns)
        cell = _draw_cell_refs(loop.args[2], index, locals)
        cell_defs, order = _draw_declarations(cell.args, (); generated=true)
        projected = [locals[n] for n in exports]
        cell_names = union(names, keys(cell_defs), (index,))
        nested = _draw_plate_values(cell_defs, order, projected, mod, cell_names, rng, cells)
        lines = _draw_lines(projected, union(keys(cell_defs), keys(nested)); order) do n
            haskey(nested, n) && return nested[n]
            _draw_rhs(cell_defs[n], n, mod, cell_names, rng)
        end
        body = Expr(:block, lines..., Expr(:tuple, projected...))
        reads = _closure_reads(Expr(:->, index, body))
        captures = sort!(collect(intersect(reads, union(names, (rng,), first.(cells)))); by=string)
        # The ordinary RK comprehension cell gives empty domains their
        # inferred element type. Pack the captures just as wide RK cells do.
        env = gensym(:captures)
        bindings = [Expr(:(=), n, Expr(:ref, env, k)) for (k, n) in enumerate(captures)]
        program = Expr(:->, Expr(:tuple, env, index), Expr(:block, bindings..., body))
        handle = gensym(:cell)
        push!(cells, handle => ReactiveKernels._ArrayFunction(ReactiveKernels.compile(program)))
        indices = _resolve_module_calls(_draw_axis_expr(domain), mod, names, :plate)
        rows = gensym(:records)
        expressions[rows] = Expr(:call, GlobalRef(@__MODULE__, :_draw_records),
            handle, Expr(:tuple, captures...), indices)
        for (k, n) in enumerate(exports)
            field = Expr(:call, Expr(:curly, GlobalRef(ReactiveKernels, :_PackIndex), k))
            expressions[n] = Expr(:call, GlobalRef(Base, :map), field, rows)
        end
    end
    return expressions
end

function _prepare_draw(plan::StructuralPlan, role::Val{R}) where {R}
    isbound(plan) || _sfail("bind the model inputs before preparing a draw")
    src = plan.draw_source
    src === nothing && _sfail("native generation needs the sampling declarations of an authored RKPPL program")
    layout = assign_layout(plan)
    prior = R === :prior
    observed = union(src.observed, plan.conditioned, (r.response for r in plan.responses))
    # A factor on a computed value changes the joint law, even when every
    # individual declaration can be sampled.
    prior && any(p -> p.name ∉ observed, plan.factors) &&
        _sfail("the prior has a density factor on a computed value; supply a normalized joint generation law")
    fitted, order = _draw_declarations(src.model.args, observed;
        factors = Set(p.name for p in plan.factors))
    gen, generated_order = _draw_declarations(src.generated.args, (); generated=true)
    sites = Set(e.name for e in layout.entries)
    paths = _scope_name_paths(src.scopes)
    names = _collect_binders!(union(keys(fitted), keys(gen), src.inputs, sites), src.model)
    # A repeated submodel returning a local directly can use the same
    # identifier for its public return and for that private local.
    roots = Set(s.binding for s in src.scopes if length(s.path) == 1)
    outputs = prior ? [n for n in order if n in sites] :
        [n for n in generated_order if !haskey(paths, n) || n in roots]
    if prior
        for n in sites
            haskey(fitted, n) || _sfail("prior site `$n` needs an explicit sampling declaration with a generation law")
        end
    end
    rng = gensym(:rng)
    data = gensym(:inputs)
    params = gensym(:parameters)
    inputs = Symbol[]
    cells = Pair{Symbol,Any}[]
    plates = prior ? Dict{Symbol,Expr}() :
        _draw_plate_values(gen, generated_order, outputs, src.mod, names, rng, cells)
    lines = _draw_lines(outputs, union(names, keys(plates))) do n
        if haskey(plates, n)
            plates[n]
        elseif !prior && n in sites
            Expr(:call, GlobalRef(Base, :getproperty), params, QuoteNode(n))
        elseif n in observed || (!haskey(gen, n) && !haskey(fitted, n))
            push!(inputs, n)
            Expr(:call, GlobalRef(Base, :getproperty), data, QuoteNode(n))
        else
            st = haskey(gen, n) ? gen[n] : fitted[n]
            if !Meta.isexpr(st, :(=)) && !prior && !haskey(gen, n)
                _sfail("generated quantity reads fitted site `$n` with no value in the supplied parameter draw")
            end
            _draw_rhs(st, n, src.mod, names, rng)
        end
    end
    result = Expr(:tuple, Expr(:parameters, (Expr(:kw, n, n) for n in outputs)...))
    values = NamedTuple{Tuple(inputs)}(Tuple(_draw_input(plan, n) for n in inputs))
    bundle = data
    if !isempty(cells)
        bundle = gensym(:bindings)
        values = (; data=values, cells=Tuple(last(cell) for cell in cells))
        pushfirst!(lines,
            Expr(:(=), data, Expr(:., bundle, QuoteNode(:data))),
            (Expr(:(=), first(cell), Expr(:ref, Expr(:., bundle, QuoteNode(:cells)), k))
             for (k, cell) in enumerate(cells))...)
    end
    expr = Expr(:->, Expr(:tuple, rng, bundle, params), Expr(:block, lines..., result))
    compiled = ReactiveKernels.compile(expr)
    program = isempty(cells) ? compiled : ReactiveKernels._ArrayFunction(compiled)
    return _NativeDrawQuery{R,typeof(program),typeof(values),typeof(layout)}(program, values, layout)
end

"""
    prepare_prior(bound)

Prepare native joint prior generation from a bound authored model. Calling
`q(rng)` returns one constrained parameter NamedTuple. `q(rng, count)` returns
a vector of independent joint realizations. Both use a private copy of the
explicit stream. Pass a draw to `unconstrain(target_layout, draw)` for replay
on a compatible design. Each stochastic dependency is evaluated once.
"""
prepare_prior(plan::StructuralPlan) = Base.invokelatest(_prepare_draw, plan, Val(:prior))
prepare_prior(bound::RKPPLBoundModel) = prepare_prior(bind_data(bound))

"""
    prepare_generated(bound)

Prepare the model's `@generate` declarations as a native query. Call
`q(u, rng)` with packed sampler coordinates, or `q(draw, rng)` with their
constrained NamedTuple. It returns the generated bindings, using a copy of the
RNG. Generated declarations contribute no sampler coordinate or density.
"""
prepare_generated(plan::StructuralPlan) = Base.invokelatest(_prepare_draw, plan, Val(:generated))
prepare_generated(bound::RKPPLBoundModel) = prepare_generated(bind_data(bound))

function _prior_draw(q, rng)
    flat = q.program(rng, q.inputs, (;))
    return _scoped_draw_values(q.layout, collect(pairs(flat)))
end
(q::_NativeDrawQuery{:prior})(rng) = Base.invokelatest(_prior_draw, q, copy(rng))
function (q::_NativeDrawQuery{:prior})(rng, count::Integer)
    count >= 0 || throw(ArgumentError("draw count must be nonnegative"))
    stream = copy(rng)
    return Base.invokelatest(map, _ -> _prior_draw(q, stream), 1:count)
end
(q::_NativeDrawQuery{:generated})(u::AbstractVector, rng) = q(constrain(q.layout, u), rng)
(q::_NativeDrawQuery{:generated})(draw::NamedTuple, rng) =
    Base.invokelatest(q.program, copy(rng), q.inputs, _flat_draw_values(q.layout, draw))

function ReactiveKernels.code_expr(q::_NativeDrawQuery)
    f = q.program
    return ReactiveKernels._program_expression(f isa ReactiveKernels._ArrayFunction ? f.native : f)
end
