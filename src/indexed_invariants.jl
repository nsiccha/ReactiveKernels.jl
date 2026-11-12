# Reuse elementary scalar expressions across enclosing native plate cells.
# The source array is invariant and each index comes from an inner coordinate.
# Domain guards preserve lazy errors and ordinary derivatives; unsupported
# storage, singular live inputs and effectful regions retain their source loops.

# An array's element reads may mutate another input. Only ordinary storage
# recursively containing ordinary data can use values cached across those
# reads. Unknown objects keep their authored execution. Type proofs avoid
# traversing a dense vector of ordinary dense rows on every call.
_indexed_passive_type(::Type) = false
_indexed_passive_type(::Type{T}) where {T<:Union{Nothing,Bool,Int8,Int16,Int32,Int64,
    Int128,UInt8,UInt16,UInt32,UInt64,UInt128,Float16,Float32,Float64,Char}} = true
_indexed_passive_type(::Type{Array{T,N}}) where {T,N} = _indexed_passive_type(T)
_indexed_passive_type(::Type{BitArray{N}}) where N = true
_indexed_passive_type(::Type{CartesianIndex{N}}) where N = true
_indexed_passive_type(::Type{UnitRange{T}}) where T = _indexed_passive_type(T)
_indexed_passive_type(::Type{Base.OneTo{T}}) where T = _indexed_passive_type(T)
_indexed_passive_type(::Type{Base.RefValue{T}}) where T = _indexed_passive_type(T)
_indexed_data_safe(x) = _indexed_passive_type(typeof(x))
@generated function _indexed_data_safe(x::Tuple)
    # Constants and input tuples describe structural arguments, not data
    # extents. Keep their checks concrete even beyond Base's splat limit.
    foldr((i,rest) -> :(_indexed_data_safe(getfield(x,$i)) && $rest),
          1:fieldcount(x); init=true)
end
_indexed_data_safe(x::NamedTuple) = _indexed_data_safe(values(x))

function _indexed_formula_expr(formula, x)
    formula === :input && return x
    name, child = formula
    f = name === :log1pexp ? GlobalRef(LogExpFunctions, name) : GlobalRef(Base, name)
    Expr(:call, f, _indexed_formula_expr(child, x))
end

@generated function _indexed_apply(::Val{F}, x) where F
    _indexed_formula_expr(F, :x)
end

function _indexed_formula_guard(formula, x, strict)
    formula === :input && return strict ? :(isfinite($x)) : true
    name, child = formula
    value = _indexed_formula_expr(child, x)
    guard = _indexed_formula_guard(child, x, strict)
    domain = if name === :log
        strict ? :($value > 0 && isfinite(inv($value))) : :($value >= 0 || isnan($value))
    elseif name === :log1p
        strict ? :($value > -1 && isfinite(inv(1 + $value))) : :($value >= -1 || isnan($value))
    else
        true
    end
    result = _indexed_formula_expr(formula, x)
    finite = strict ? :(isfinite($result)) : true
    :($guard && $domain && $finite)
end

@generated function _indexed_scalar_safe(::Val{F}, x::Union{Float32,Float64},
                                         ::Val{Bound}) where {F,Bound}
    _indexed_formula_guard(F, :x, !Bound)
end
_indexed_scalar_safe(formula, x, bound) = false

_indexed_safe(formula, array, bound) = false
function _indexed_safe(formula, array::Union{Array{Float32},Array{Float64},_CellArray},
                       ::Val{Bound}) where Bound
    Bound isa Bool || return false
    _cell_array_operand(array) || return false
    for x in array
        _indexed_scalar_safe(formula, x, Val(Bound)) || return false
    end
    true
end
@generated function _indexed_safe(formula, array::_CellHcat, ::Val{Bound}) where Bound
    Bound isa Tuple || return false
    tests = [:( _indexed_safe(formula, getfield(array.arguments, $i), Val($(QuoteNode(b)))) )
             for (i, b) in enumerate(Bound)]
    checks = isempty(tests) ? true : foldl((a,b) -> :($a && $b), tests)
    :(length(array.arguments) == $(length(Bound)) && $checks)
end

function _indexed_map(formula, array)
    result = Array{eltype(array)}(undef, size(array))
    @inbounds for i in eachindex(array)
        result[i] = _indexed_apply(formula, array[i])
    end
    result
end

function _indexed_formula(source, argument)
    source === argument && return :input
    source isa Expr && source.head === :call && length(source.args) == 2 || return nothing
    f = _cell_array_function(source.args[1])
    name = f === (-) ? :- : f === log ? :log : f === log1p ? :log1p :
        f === exp ? :exp : f === LogExpFunctions.log1pexp ? :log1pexp : nothing
    name === nothing && return nothing
    child = _indexed_formula(source.args[2], argument)
    child === nothing ? nothing : (name, child)
end
_indexed_worthwhile(f) = f isa Tuple &&
    (first(f) !== :- || _indexed_worthwhile(last(f)))

# Pull a scalar unary function through an already-authored selection. This is
# used only to recognize two formulas; no code runs until the domain guard has
# proved that both resulting tables can be evaluated safely.
function _indexed_selection(source)
    if source isa Expr && source.head === :block
        statements = filter(x -> !(x isa LineNumberNode), source.args)
        length(statements) == 1 || return nothing
        return _indexed_selection(only(statements))
    end
    if source isa Expr && source.head === :if && length(source.args) == 3
        condition = Expr(:if, source.args[1], true, false)
        return (condition, source.args[2], source.args[3])
    elseif source isa Expr && source.head === :call
        fn = _cell_array_function(source.args[1])
        fn === ifelse && length(source.args) == 4 && return Tuple(source.args[2:4])
        length(source.args) == 2 || return nothing
        selected = _indexed_selection(source.args[2])
        selected === nothing && return nothing
        condition, yes, no = selected
        return (condition, Expr(:call, source.args[1], yes), Expr(:call, source.args[1], no))
    end
    nothing
end

# Flat native branches call their condition and their two arms separately.
# Recover their original complete source through the same checked source reader
# used by array composition, retaining the exact argument expressions.
function _indexed_source(rhs, recipes)
    expanded = _cell_array_expand_call(rhs, recipes)
    expanded === rhs || return expanded
    if rhs isa Expr && rhs.head === :call
        slot = _callable_slot(first(rhs.args))
        if slot !== nothing
            op = recipes[slot].op
            op in (axes, eachindex, length, size) &&
                return Expr(:call, GlobalRef(Base, nameof(op)), rhs.args[2:end]...)
        end
    end
    rhs isa Expr && rhs.head === :if && length(rhs.args) == 3 || return rhs
    condition = rhs.args[1]
    condition isa Expr && condition.head === :call || return rhs
    reference = condition.args[1]
    reference isa Expr && reference.head === :call && length(reference.args) == 3 || return rhs
    reference.args[1] in (:getfield, GlobalRef(Core, :getfield), GlobalRef(Base, :getfield)) || return rhs
    reference.args[3] == QuoteNode(:condition) || return rhs
    slot = _callable_slot(reference.args[2])
    slot === nothing && return rhs
    recipe = recipes[slot]
    recipe.op isa _KernelSourceOp || return rhs
    branch = getfield(recipe.op, :f)
    branch isa _KernelBranch || return rhs
    indices = typeof(branch).parameters[1:3]
    arguments = Any[nothing for _ in recipe.inputs]
    assigned = falses(length(arguments))
    for (positions, call) in zip(indices, rhs.args)
        call isa Expr && call.head === :call && length(call.args) == length(positions)+1 || return rhs
        for (position, argument) in zip(positions, call.args[2:end])
            assigned[position] && arguments[position] != argument && return rhs
            arguments[position] = argument
            assigned[position] = true
        end
    end
    all(assigned) || return rhs
    source = _cell_array_source(recipe, arguments)
    source === nothing ? rhs : source
end

function _indexed_alias(x)
    x isa Symbol && return x
    x isa Expr || return nothing
    if x.head === :block
        isempty(x.args) && return nothing
        return _indexed_alias(last(x.args))
    elseif x.head === :if
        arms = Any[a for a in x.args[2:end] if !_cell_array_throws(a)]
        isempty(arms) && return nothing
        aliases = map(_indexed_alias, arms)
        first(aliases) !== nothing && all(==(first(aliases)), aliases) && return first(aliases)
    elseif x.head === :call && length(x.args) == 2 &&
            _cell_array_function(first(x.args)) === identity
        return _indexed_alias(x.args[2])
    end
    nothing
end

function _indexed_parent(name, definitions, available, seen=Set{Symbol}())
    name isa Symbol || return nothing
    name in available && return name
    name in seen && return nothing
    push!(seen, name)
    rhs = get(definitions, name, nothing)
    rhs === nothing && return nothing
    _indexed_parent(_indexed_alias(rhs), definitions, available, seen)
end

function _indexed_bound(x, definitions, recipes, seen=Set{Symbol}())
    x isa Union{Number,QuoteNode,GlobalRef,Nothing} && return true
    if x isa Symbol
        x in seen && return false
        push!(seen, x)
        return haskey(definitions, x) && _indexed_bound(definitions[x], definitions, recipes, seen)
    end
    x isa Expr || return false
    slot = _constant_slot(x)
    slot === nothing || return recipes[slot].op isa _BoundConstant
    if x.head === :call
        return _cell_array_function(first(x.args)) !== nothing &&
            all(a -> _indexed_bound(a, definitions, recipes, copy(seen)), x.args[2:end])
    end
    false
end

function _indexed_flags(name, definitions, recipes, seen=Set{Symbol}())
    _indexed_bound(name, definitions, recipes) && return true
    name isa Symbol && !(name in seen) || return false
    push!(seen, name)
    source = get(definitions, name, nothing)
    alias = _indexed_alias(source)
    alias === nothing || return _indexed_flags(alias, definitions, recipes, seen)
    if source isa Expr && source.head === :call &&
            _cell_array_function(first(source.args)) === _cell_array_hcat
        return Tuple(_indexed_flags(a, definitions, recipes, copy(seen)) for a in source.args[2:end])
    end
    false
end

function _indexed_assigned!(names, ex)
    _index_run_visit(ex) do node
        node.head === :(=) && length(node.args) == 2 && node.args[1] isa Symbol &&
            push!(names, node.args[1])
    end
    names
end

_indexed_scalar_indices(indices...) = all(i -> i isa Int || i isa CartesianIndex, indices)

function _indexed_mentions(x, name)
    x === name && return true
    x isa Expr && !(x.head in (:quote, :inert)) &&
        any(a -> _indexed_mentions(a, name), x.args)
end

# A position in an inner ordinal domain may be reused even when that domain's
# length varies between outer cells. An index computed from the outer cell's
# value is different: moving its table above that loop need not save any work.
# This is an optimization eligibility check, not a restriction on authored code.
function _indexed_coordinate_varies(x, outer, inner, definitions, inside,
                                   seen=Set{Symbol}())
    if x isa Symbol
        x === outer && return true
        x in inner && return false
        x in seen && return true
        haskey(definitions, x) || return x in inside
        push!(seen, x)
        return _indexed_coordinate_varies(definitions[x], outer, inner,
                                          definitions, inside, seen)
    end
    x isa Expr || return false
    x.head in (:quote, :inert) && return false
    if x.head === :block
        # Inbounds coordinate readers carry a local temporary. The same block
        # may appear at several uses, so the global single-definition map
        # deliberately excludes it; resolve its local binding here instead.
        localdefs = copy(definitions)
        _index_run_visit(x) do node
            node.head === :(=) && length(node.args) == 2 && node.args[1] isa Symbol &&
                (localdefs[node.args[1]] = node.args[2])
        end
        alias = _indexed_alias(x)
        alias === nothing || return _indexed_coordinate_varies(alias, outer, inner,
                                                               localdefs, inside, seen)
    end
    alias = _indexed_alias(x)
    alias === nothing || return _indexed_coordinate_varies(alias, outer, inner,
                                                           definitions, inside, seen)
    if x.head === :call && _cell_array_function(first(x.args)) in
            (axes, eachindex, length, size)
        return false
    end
    any(a -> _indexed_coordinate_varies(a, outer, inner, definitions, inside,
                                        copy(seen)), x.args)
end

function _indexed_loop(loop, available, definitions, statements, recipes, data_guard)
    loop.head === :for && length(loop.args) == 2 || return nothing
    binding = loop.args[1]
    binding isa Expr && binding.head === :(=) && length(binding.args) == 2 || return nothing
    iteration = binding.args[2]
    iteration isa Expr && iteration.head === :call &&
        _cell_array_function(first(iteration.args)) === _plate_cells || return nothing
    inner = Set{Symbol}()
    _index_run_visit(loop.args[2]) do x
        x.head === :for && Meta.isexpr(first(x.args), :(=), 2) &&
            first(x.args).args[1] isa Symbol && push!(inner, first(x.args).args[1])
    end
    isempty(inner) && return nothing
    inside = _indexed_assigned!(Set{Symbol}(), loop)
    available = setdiff(available, inside)
    candidates = Tuple{Symbol,Expr,Symbol}[]
    for (argument, gather) in definitions
        argument in inside || continue
        gather isa Expr && gather.head === :ref && length(gather.args) >= 2 || continue
        any(i -> Meta.isexpr(i, :...), gather.args[2:end]) && continue
        root = _indexed_parent(gather.args[1], definitions, available)
        root === nothing && continue
        any(i -> _indexed_coordinate_varies(i, binding.args[1], inner,
            definitions, inside), gather.args[2:end]) && continue
        push!(candidates, (argument, gather, root))
    end
    isempty(candidates) && return nothing
    replacements = IdDict{Any,Any}()
    transformed = IdDict{Any,Any}()
    gathers = Dict{Symbol,Any}()
    caches = Dict{Any,Symbol}()
    prefix = Expr(:block)
    tests = Any[]
    function cached_read(argument, gather, root, formula)
        keys = get!(gathers, argument) do
            (gather, Any[gensym(:indexed_coordinate) for _ in gather.args[2:end]])
        end
        table = get!(caches, (root, formula)) do
            table = gensym(:indexed_invariant)
            tag = Expr(:call, GlobalRef(Base, :Val), QuoteNode(formula))
            flags = Expr(:call, GlobalRef(Base, :Val),
                QuoteNode(_indexed_flags(root, definitions, recipes)))
            push!(tests, Expr(:call, GlobalRef(@__MODULE__, :_indexed_safe), tag, root, flags))
            push!(prefix.args, Expr(:(=), table,
                Expr(:call, GlobalRef(@__MODULE__, :_indexed_map), tag, root)))
            table
        end
        # The original scalar gather proves the corresponding table access
        # is valid. Bind its coordinates once; retain that read and its checks.
        _inbounds_value(Expr(:ref, table, last(keys)...))
    end
    _index_run_visit(loop.args[2]) do statement
        statement.head === :(=) && length(statement.args) == 2 && statement.args[1] isa Symbol || return
        rhs = _indexed_source(statement.args[2], recipes)
        rewritten = rhs
        arguments = Symbol[]
        for (argument, gather, root) in candidates
            used = Ref(false)
            replacement = _index_run_replace(rewritten) do ex
                # Whole selections can use the two already-valid table
                # entries. Event-dependent surrounding arithmetic retains its
                # source control flow and only replaces eligible subexpressions.
                selection = _indexed_selection(ex)
                if selection !== nothing
                    condition, yes, no = selection
                    fy, fn = _indexed_formula(yes, argument), _indexed_formula(no, argument)
                    if fy !== nothing && fn !== nothing &&
                            (_indexed_worthwhile(fy) || _indexed_worthwhile(fn))
                        used[] = true
                        return Expr(:call, GlobalRef(Core, :ifelse), condition,
                            cached_read(argument, gather, root, fy),
                            cached_read(argument, gather, root, fn))
                    end
                end
                formula = _indexed_formula(ex, argument)
                if formula !== nothing && _indexed_worthwhile(formula)
                    used[] = true
                    return cached_read(argument, gather, root, formula)
                end
                ex
            end
            if used[]
                push!(arguments, argument)
                rewritten = replacement
            end
        end
        isempty(arguments) && return
        guards = [Expr(:call, GlobalRef(Core, :isa), argument, :(Union{Float32,Float64}))
                  for argument in arguments]
        scalar = foldl((a,b) -> Expr(:&&, a,b), guards)
        replacements[statement] = Expr(:(=), statement.args[1],
            Expr(:if, scalar, rewritten, statement.args[2]))
        transformed[statement] = rewritten
    end
    isempty(replacements) && return nothing
    for (argument, (gather, keys)) in gathers
        original = statements[argument]
        bindings = Any[Expr(:(=), key, index) for (key, index) in zip(keys, gather.args[2:end])]
        function still_read(ex)
            ex === original && return false
            ex isa Expr && ex.head === :local && all(a -> a isa Symbol, ex.args) && return false
            haskey(transformed, ex) && return _indexed_mentions(transformed[ex], argument)
            ex === argument && return true
            ex isa Expr && !(ex.head in (:quote, :inert)) && any(still_read, ex.args)
        end
        read = Expr(:(=), argument, Expr(:ref, gather.args[1], keys...))
        if !still_read(loop.args[2])
            scalar = Expr(:call, GlobalRef(@__MODULE__, :_indexed_scalar_indices), keys...)
            check = Expr(:call, GlobalRef(Base, :checkbounds), gather.args[1], keys...)
            push!(bindings, Expr(:if, scalar, check, read))
            # All value uses became table reads. On the proven scalar path,
            # retain the original bounds check but avoid loading it again.
            for (statement, rewritten) in transformed
                replacement = replacements[statement]
                guard = replacement.args[2].args[1]
                replacement.args[2].args[1] = _index_run_replace(guard) do ex
                    ex.head === :call && length(ex.args) == 3 &&
                        ex.args[1] == GlobalRef(Core, :isa) && ex.args[2] === argument ? scalar : ex
                end
            end
        else
            push!(bindings, read)
        end
        replacements[original] = Expr(:block, bindings...)
    end
    rewritten = _index_run_replace(x -> get(replacements, x, x), loop)
    push!(prefix.args, rewritten)
    guard = foldl((a,b) -> Expr(:&&, a,b), tests)
    # At least two outer iterations can reuse a table. The loop and storage
    # sizes remain runtime values; no input length enters emitted syntax.
    domain = loop.args[1].args[2]
    guard = :($(GlobalRef(Base, :length))($domain) > 1 && $data_guard && $guard)
    Expr(:if, guard, prefix, loop)
end

function _lower_indexed_invariants(ast::Expr, recipes)
    any(r -> r.effectful, recipes) && return ast
    definitions = Dict{Symbol,Any}()
    statements = Dict{Symbol,Expr}()
    repeated = Set{Symbol}()
    _index_run_visit(ast) do ex
        ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol || return
        lhs = ex.args[1]
        haskey(definitions, lhs) && push!(repeated, lhs)
        definitions[lhs] = _indexed_source(ex.args[2], recipes)
        statements[lhs] = ex
    end
    for name in repeated
        delete!(definitions, name)
        delete!(statements, name)
    end
    function rewrite(ex, available)
        ex isa Expr || return ex
        ex.head in (:quote, :inert) && return ex
        if ex.head === :for
            replacement = _indexed_loop(ex, available, definitions, statements, recipes, data_guard)
            replacement === nothing || return replacement
        end
        locals = copy(available)
        args = Any[]
        for child in ex.args
            push!(args, rewrite(child, locals))
            if child isa Expr && child.head === :(=) && child.args[1] isa Symbol
                push!(locals, child.args[1])
            end
        end
        Expr(ex.head, args...)
    end
    available = Set{Symbol}()
    if ast.head === :function && ast.args[1] isa Expr && ast.args[1].head === :tuple
        for argument in ast.args[1].args
            argument isa Symbol && push!(available, argument)
            argument isa Expr && argument.head === :(::) && argument.args[1] isa Symbol &&
                push!(available, argument.args[1])
        end
    end
    data = sort!(collect(setdiff(available, Set((:__callables__,)))); by=string)
    data_guard = Expr(:call, GlobalRef(@__MODULE__, :_indexed_data_safe), Expr(:tuple,data...))
    rewrite(ast, available)
end
