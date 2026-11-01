# Cold graph projection used by PPL's finite-state filtering boundary. The
# caller supplies storage expressions; numerical graph/scan traversal stays in
# RK. This is a one-time-slice projection, not execution of a plan at run time.
_stream_has_series(ex::Symbol, names) = ex in names
_stream_has_series(ex, names) = ex isa Expr && any(a -> _stream_has_series(a,names),ex.args)

# A temporal design matrix is projected to one row, keeping its feature axis.
# Its matrix/vector product then has one temporal output, hence one scalar.
# These are algebraic row projections, not calls on a truncated history.
_stream_call_is(ex, name) = Meta.isexpr(ex,:call) &&
    (ex.args[1] === name || ex.args[1] == GlobalRef(Base,name))
function _stream_is_row(ex, rows)
    ex isa Symbol && return ex in rows
    _stream_call_is(ex,:hcat) && return true
    _stream_call_is(ex,:identity) && return _stream_is_row(ex.args[2],rows)
    return false
end

function _stream_row_product(row::AbstractMatrix, coefficients::AbstractVector)
    size(row,1)==1 || throw(ArgumentError("streamed design input must contain one observation row"))
    return only(row * coefficients)
end
_stream_row_product(row, coefficients) = throw(ArgumentError(
    "streamed design multiplication requires a design row and a fixed coefficient vector"))

function _stream_design_expression(ex, isseries, isrow, project)
    Meta.isexpr(ex,:call) || return nothing
    args = ex.args[2:end]
    if _stream_call_is(ex,:hcat) && any(isseries,args)
        all(isseries,args) || throw(ArgumentError(
            "streamed hcat columns must share the temporal axis; broadcast fixed columns over a stream"))
        return Expr(:call,GlobalRef(Base,:hcat),map(project,args)...)
    elseif _stream_call_is(ex,:*) && length(args)==2 && isrow(args[1])
        !isseries(args[2]) || throw(ArgumentError("streamed design coefficients must be fixed"))
        return Expr(:call,GlobalRef(@__MODULE__,:_stream_row_product),map(project,args)...)
    end
    # Constant columns whose sole dimension is an authored temporal length.
    # Other constructors/reductions keep their ordinary unsupported boundary.
    isempty(args) && return nothing
    dim = last(args)
    _stream_call_is(dim,:length) && length(dim.args)==2 && isseries(dim.args[2]) || return nothing
    isrow(dim.args[2]) && return nothing # length(matrix) includes its feature axis
    if _stream_call_is(ex,:ones) || _stream_call_is(ex,:zeros)
        length(args) in (1,2) || return nothing
        f = _stream_call_is(ex,:ones) ? :one : :zero
        typ = length(args)==1 ? Float64 : project(args[1])
        return Expr(:call,GlobalRef(Base,f),typ)
    elseif _stream_call_is(ex,:fill) && length(args)==2 && !isseries(args[1])
        return project(args[1])
    end
    return nothing
end

# A pointwise projection cannot reinterpret a vector reduction, gather or an
# opaque whole-vector call as a call on its current element. Broadcast syntax
# makes the authored elementwise contract explicit.
function _stream_check_pointwise(ex, names)
    ex isa Expr || return
    if ex.head === :call && _stream_has_series(ex,names)
        head = ex.args[1]
        dotted = head isa Symbol && startswith(String(head),".")
        (dotted || head === :identity) || throw(ArgumentError(
            "whole-sequence call $head has no authored pointwise streaming projection"))
    elseif ex.head === :ref && _stream_has_series(ex.args[1],names)
        throw(ArgumentError("whole-sequence indexing has no authored pointwise streaming projection"))
    end
    foreach(a -> _stream_check_pointwise(a,names),ex.args)
    nothing
end

function _stream_point_expression(spec::KernelSpec, arguments, series_arguments, operation!, carry!)
    p = plan(spec)
    length(arguments) == length(p.have) || throw(ArgumentError(
        "incremental kernel arguments differ from its authored boundary"))
    values = Dict(canon_id(p.graph,v.id) => ex for (v,ex) in zip(p.have,arguments))
    series = Dict(canon_id(p.graph,v.id) => flag for (v,flag) in zip(p.have,series_arguments))
    body = Any[]
    scans = 0
    for recipe in p.recipes
        recipe.effectful && throw(ArgumentError("incremental kernels must be pure"))
        args = [values[canon_id(p.graph,v.id)] for v in recipe.inputs]
        flags = [series[canon_id(p.graph,v.id)] for v in recipe.inputs]
        outputs = [gensym(:stream_value) for _ in recipe.outputs]
        lhs = length(outputs) == 1 ? only(outputs) : Expr(:tuple,outputs...)
        if recipe.op isa _AuthoredScanOp
            op = recipe.op
            _scan_has_history(op) && throw(ArgumentError("incremental kernel scan reads arbitrary history"))
            _scan_includes_init(op) && throw(ArgumentError(
                "include_init kernel streams need a separate seed event; use an authored PPL scan"))
            length(outputs) == 1 || throw(ArgumentError("expected a single scan trajectory"))
            scans += 1
            carry, store = carry!(scans,args[1])
            atomic = typeof(op).parameters[2]
            all(i -> !flags[i],atomic) && all(i -> flags[i],setdiff(eachindex(args),atomic)) ||
                throw(ArgumentError("streamed scan needs invariant seed/shared operands and time-varying iterated operands"))
            step_args = Any[carry]
            append!(step_args,[args[i] for i in 2:length(args) if i ∉ atomic])
            append!(step_args,[args[i] for i in 2:length(args) if i in atomic])
            next = gensym(:stream_carry)
            push!(body, Expr(:(=),Expr(:tuple,next,only(outputs)),
                Expr(:call,operation!(op.kernel),step_args...)), store(next))
        else
            recipe.op isa _AuthoredPlateOp && throw(ArgumentError(
                "an incremental provider's outer graph must have pointwise operations and finite scans"))
            names = Set(n for (n,flag) in zip(_recipe_source_names(recipe),flags) if flag)
            _stream_check_pointwise(recipe.source,names)
            push!(body,Expr(:(=),lhs,Expr(:call,operation!(recipe.op),args...)))
        end
        for (v,ex) in zip(recipe.outputs,outputs)
            values[canon_id(p.graph,v.id)] = ex
            series[canon_id(p.graph,v.id)] = any(flags)
        end
    end
    scans > 0 || throw(ArgumentError("kernel provider has no finite scan"))
    result = [values[canon_id(p.graph,v.id)] for v in p.want]
    push!(body,length(result) == 1 ? only(result) : Expr(:tuple,result...))
    return Expr(:block,body...)
end
