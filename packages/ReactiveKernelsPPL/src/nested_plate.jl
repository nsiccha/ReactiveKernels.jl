# A repeated submodel has an outer substitution for its arguments and values.
# Give each inner loop its own binders first, so a subject index can shadow a
# model argument (including the argument used to compute the loop's domain).
function _scope_cell_plate(st, taken, scoped)
    loop = _plate_loop(st)
    loop === nothing && return st
    index, domain = loop.args[1].args
    names = Set{Symbol}([index])
    for cell in loop.args[2].args
        Meta.isexpr(cell, :(=), 2) && cell.args[1] isa Symbol && push!(names, cell.args[1])
    end
    rename = Dict(name => _cell_scan_fresh!(taken, name) for name in sort!(collect(names)))
    union!(scoped, values(rename))
    return Expr(:macrocall, st.args[1:2]..., Expr(:for,
        Expr(:(=), rename[index], domain), _hsubst(loop.args[2], rename)))
end

# Preserve the inner iterator when a plate cell builds an indexed column.
# Each such column holds the results of a native RK plate for that outer
# index. A dotted observation consumes the resulting columns at both levels.
function _expand_nested_plates(ast, data)
    nested = Set{Symbol}()
    taken = nothing
    statements = Any[]
    for st in ast.args
        loop = _plate_loop(st)
        if loop === nothing || !any(c -> _plate_loop(c) !== nothing, loop.args[2].args)
            push!(statements, st)
            continue
        end
        isnothing(taken) && (taken = union(data, _expr_names(ast)))
        outer = loop.args[1].args[1]
        cells = Any[]
        for cell in loop.args[2].args
            inner = _plate_loop(cell)
            if inner === nothing
                push!(cells, cell)
            else
                _expand_nested_cells!(cells, nested, inner, outer, data, taken)
            end
        end
        push!(statements, Expr(:macrocall, st.args[1:2]...,
            Expr(:for, loop.args[1], Expr(:block, cells...))))
    end
    return Expr(:block, statements...), nested
end

function _expand_nested_cells!(cells, nested, loop, outer, data, taken)
    index, domain = loop.args[1].args
    locals = Pair{Symbol,Any}[]
    outputs = Set{Symbol}()
    function column(rhs)
        needed = _needed_cell_locals(locals, rhs)
        body = Expr(:block, LineNumberNode(0, :rkppl_plate),
            (Expr(:(=), n, v) for (n, v) in needed)..., rhs)
        return Expr(:do, Expr(:call, GlobalRef(ReactiveKernels, :plate), domain), Expr(:->, Expr(:tuple, index), body))
    end
    for st in loop.args[2].args
        st isa LineNumberNode && continue
        if Meta.isexpr(st, :(=), 2)
            lhs, rhs = st.args
            if lhs isa Symbol
                push!(locals, lhs => rhs)
            elseif Meta.isexpr(lhs, :ref, 2) && lhs.args[2] === index &&
                    Meta.isexpr(lhs.args[1], :ref, 2) && lhs.args[1].args[2] === outer
                push!(cells, Expr(:(=), lhs.args[1], column(rhs)))
                push!(outputs, lhs.args[1].args[1])
            else
                _sfail("nested plate output must designate both iterations, got $(repr(lhs))")
            end
        elseif _is_sample(st) || _is_broadcast_sample(st)
            lhs, law = st.args[2:3]
            (Meta.isexpr(lhs, :ref, 2) && lhs.args[2] === index &&
                Meta.isexpr(lhs.args[1], :ref, 2) && lhs.args[1].args[2] === outer &&
                lhs.args[1].args[1] in data) || _sfail(
                    "nested plate draw $(repr(lhs)) has no packed parameter layout")
            group = lhs.args[1]
            (domain == Expr(:call, :eachindex, group) ||
                domain == Expr(:call, :axes, group, 1)) || _sfail(
                    "nested observation `$lhs` needs a plate covering the subject axis " *
                    "of `$group`; scoring selected subjects of this response is not implemented")
            response = group.args[1]
            arg(value) = begin
                value isa Union{Number,QuoteNode,GlobalRef} && return value
                if Meta.isexpr(value, :ref, 2) && value.args[2] === index &&
                        Meta.isexpr(value.args[1], :ref, 2) && value.args[1].args[2] === outer &&
                        value.args[1].args[1] in outputs
                    return value.args[1]
                end
                name = _cell_scan_fresh!(taken, Symbol(response, :_argument))
                ref = Expr(:ref, name, outer)
                push!(cells, Expr(:(=), ref, column(value)))
                return ref
            end
            lifted = if _external_rhs(law) && (_is_dotted_call(law) || Meta.isexpr(law, :call))
                args = _is_dotted_call(law) ? law.args[2].args : law.args[2:end]
                Expr(:., first(law.args), Expr(:tuple, map(arg, args)...))
            else
                _map_rhs_values(arg, law)
            end
            if _is_broadcast_sample(st)
                push!(nested, response)
            elseif Meta.isexpr(lifted, :call)
                lifted = Expr(:., lifted.args[1], Expr(:tuple, lifted.args[2:end]...))
            end
            push!(cells, Expr(:call, :.~, lhs.args[1], lifted))
        else
            _sfail("unsupported nested plate statement $(repr(st))")
        end
    end
end
