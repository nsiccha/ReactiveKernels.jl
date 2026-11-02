# Source-visible finite recurrences. This analysis belongs to RK: callers
# provide the current arguments, clock and carry storage, not a second model.
_stream_error(message) = throw(ArgumentError("[stream projection] " * message))
_stream_statements(ex) = Meta.isexpr(ex, :block) ?
    filter(x -> !(x isa LineNumberNode), ex.args) : Any[ex]

function _stream_current_index(index, time)
    index isa Integer && index == time || _stream_error(
        "a projected trajectory can only be gathered at the current clock ($time); got $index")
    return nothing
end

function _stream_float_carry(value)
    value isa Float64 || _stream_error("helper carries currently require Float64 scalars; got $(typeof(value))")
    return value
end

function _stream_float_input(value)
    value isa Float64 || _stream_error(
        "a helper temporal port currently requires one Float64 value per clock; got $(typeof(value))")
    return value
end

function _stream_assignment_names!(names, ex)
    ex isa Expr || return names
    if ex.head === :(=) || _kernel_assignment_head(ex.head)
        lhs = ex.args[1]
        # Indexed writes do not rebind their array.
        Meta.isexpr(lhs, :ref) || _lhs_symbols!(names, lhs)
    end
    foreach(a -> _stream_assignment_names!(names, a), ex.args)
    return names
end

# Recognize a trajectory allocation, retaining its element conversion. The
# allocation itself disappears; its data-dependent length is never unrolled.
function _stream_output_type(ex, series)
    Meta.isexpr(ex, :call) || return nothing
    head, args = ex.args[1], ex.args[2:end]
    if Meta.isexpr(head, :curly, 2) && head.args[1] in (:Vector, GlobalRef(Base,:Vector)) &&
            length(args) == 2 && args[1] in (:undef, GlobalRef(Core,:undef)) &&
            _stream_call_is(args[2], :length) && only(args[2].args[2:end]) in series
        return head.args[2]
    elseif _stream_call_is(ex, :similar) && length(args) == 1 && only(args) in series
        return Expr(:call, :eltype, only(args))
    end
    return nothing
end

function _stream_source_value(ex, series, output, element_type, clock; prelude=false)
    ex isa Symbol && ex in series && _stream_error(
        "whole trajectory $ex is used outside a current-element read")
    ex === output && _stream_error("helper reads its whole output trajectory")
    ex isa Expr || return ex
    if _stream_call_is(ex,:eltype) && length(ex.args) == 2 && ex.args[2] in series
        return Expr(:call, GlobalRef(Core,:typeof), ex.args[2])
    elseif Meta.isexpr(ex,:ref) && ex.args[1] in series
        !prelude && length(ex.args) == 2 && ex.args[2] == clock || _stream_error(
            "helper trajectory $(ex.args[1]) must be read at its current loop index")
        return ex.args[1]
    elseif ex.head === :(=) && Meta.isexpr(ex.args[1],:ref)
        lhs = ex.args[1]
        !prelude && lhs == Expr(:ref,output,clock) || _stream_error(
            "helper mutation must write its current output element")
        value = _stream_source_value(ex.args[2],series,output,element_type,clock)
        return Expr(:(=),output,Expr(:call,GlobalRef(Base,:convert),element_type,value))
    elseif Meta.isexpr(ex,:ref) && ex.args[1] == output
        _stream_error("helper output history requires an explicitly bounded carry")
    elseif ex.head in (:break,:continue)
        _stream_error("helper loop control $(ex.head) has no finite helper projection")
    elseif ex.head in (:for,:while,:generator,:comprehension,:macrocall,:function,:->,:return,:global)
        _stream_error("nested iteration or scope $(ex.head) has no finite helper projection")
    end
    return Expr(ex.head, (_stream_source_value(a,series,output,element_type,clock;prelude)
        for a in ex.args)...)
end

"""Project a recorded helper's vector loop to one exact transition.

`carry!(name, initial)` returns `(load_expression, store_expression_function)`.
Only the trajectory dimension is removed; all scalar math and branch order are
copied from the method source. No helper is executed during preparation.
"""
function _stream_helper_expression(source, arguments, series_arguments, carry!, time)
    names, rawbody, mod = source
    length(names) == length(arguments) || _stream_error("helper argument count differs from its recorded source")
    series = Set(n for (n,flag) in zip(names,series_arguments) if flag)
    body = _traceable_tail_body(rawbody, :stream_helper)
    statements = _stream_statements(body)
    loops = findall(st -> Meta.isexpr(st,:for), statements)
    length(loops) == 1 || _stream_error("helper needs one source-visible finite recurrence loop")
    loop_index = only(loops)
    loop_index == length(statements)-1 || _stream_error("helper must return its trajectory after the recurrence")
    output = last(statements)
    output isa Symbol || _stream_error("helper must return a named trajectory")
    loop = statements[loop_index]
    binding = loop.args[1]
    Meta.isexpr(binding,:(=),2) && binding.args[1] isa Symbol || _stream_error("helper needs one loop index")
    clock, domain = binding.args
    _stream_call_is(domain,:eachindex) && length(domain.args) > 1 &&
        all(a -> a in series, domain.args[2:end]) || _stream_error(
            "helper loop must iterate eachindex of declared temporal arguments")
    prelude = statements[1:loop_index-1]
    allocations = findall(st -> Meta.isexpr(st,:(=),2) && st.args[1] == output, prelude)
    length(allocations) == 1 || _stream_error("helper output must have one trajectory allocation")
    allocation = only(allocations)
    element_type = _stream_output_type(prelude[allocation].args[2],series)
    element_type === nothing && _stream_error("helper output allocation must preserve the temporal axis")
    deleteat!(prelude,allocation)
    initial_names = Set{Symbol}()
    for st in prelude
        Meta.isexpr(st,:(=),2) || _stream_error("helper initialization must use ordinary assignments")
        _stream_assignment_names!(initial_names,st)
    end
    modified = _stream_assignment_names!(Set{Symbol}(),loop.args[2])
    carries = sort!(collect(intersect(initial_names,modified)); by=string)
    isempty(intersect(Set(names),modified)) || _stream_error("helper must not rebind its arguments")
    # Initialized variables modified by the loop are its finite carry. The
    # remaining prelude definitions are invariant (types may use eltype of a
    # stream); current values cannot silently become new seeds each tick.
    projected = Any[_stream_source_value(st,series,output,element_type,clock;prelude=true)
        for st in prelude]
    locals = union(initial_names,modified,Set(names),Set((clock,output)))
    renamed = Dict(n => gensym(n) for n in locals)
    qualify(ex) = begin
        value = _inline_qualify(ex,mod,renamed)
        value === nothing && _stream_error("helper source contains an unresolved or mutable global")
        value
    end
    result = Any[Expr(:(=),renamed[n],flag ?
        Expr(:call,GlobalRef(@__MODULE__,:_stream_float_input),a) : a)
        for (n,a,flag) in zip(names,arguments,series_arguments)]
    push!(result,Expr(:(=),renamed[clock],time))
    append!(result,qualify.(projected))
    stores = Any[]
    for name in carries
        localname = renamed[name]
        initial = Expr(:call,GlobalRef(@__MODULE__,:_stream_float_carry),localname)
        load, store = carry!(name,initial)
        push!(result,Expr(:(=),localname,load))
        push!(stores,store(Expr(:call,GlobalRef(@__MODULE__,:_stream_float_carry),localname)))
    end
    transition = _stream_source_value(loop.args[2],series,output,element_type,clock)
    push!(result,qualify(transition))
    append!(result,stores)
    push!(result,renamed[output])
    return Expr(:let,Expr(:block),Expr(:block,result...))
end
