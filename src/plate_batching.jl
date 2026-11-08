# A plate whose cells multiply a vector of their own by a matrix every cell
# shares runs as matrix products over tiles of its cells (snag
# `plate-of-per-row-d1b4ed43`, USER pick on decision `0tpji4i`: "RK batches
# per-row plate cells natively").
#
#     dens = plate(idx, coordinates) do i, c
#         phase = transpose(frequency ./ lengths) * c
#         raw = Zc * cos.(phase) .+ Zs * sin.(phase)
#         value = (correlation * raw) .* scale
#         -0.5 * sum(abs2, value .- y[i, :])
#     end
#
# The cell loop runs each statement once per cell: per cell, `W' * c` is a
# matrix-vector product and native Enzyme reverse differentiates it as one, so a
# field of a few hundred flops per cell costs 6-14 times what the same arithmetic
# costs as matrix products over the cells (measured on the snag's sizes, 1,444 x
# 38 and 15,796 x 2 random Fourier features). Here the cells of one tile run
# together: each cell's number is one entry of a vector over the tile, each
# cell's vector one row of a matrix over the tile, and every statement runs once
# per tile on those, the way JAX's `vmap` of the cell does in the JAX target
# (`jax.jl`). A product `A * v` of a shared matrix and a cell's vector is one
# product `V * transpose(A)` per tile.
#
# The batched statement is the authored statement rewritten by these rules
# (`_batch_expr`), then lowered as an inline source is (`_kernel_native_body`,
# `_inline_qualify`):
#
# - a dotted call or operator over the cell's numbers and vectors and shared
#   numbers and vectors is the same broadcast, each shared vector read as a row
#   (`transpose(u)`);
# - `A * v`, a shared matrix times a cell's vector, is `V * transpose(A)`; a
#   product, sum, difference or quotient of a cell's numbers and vectors with
#   numbers is the elementwise one (`.*`, `.+`, `.-`, `./`), as Julia defines
#   them for a vector and a number, and the operations Julia does not define (a
#   vector plus a number) have no batched form;
# - `sum(v)` and `sum(f, v)` of a cell's vector are the sums of the rows;
# - `y[i, :]`, `y[i]` and `y[i, j]` of a shared array at the cell's integer `i`
#   gather the tile's rows, and `v[k]` of a cell's vector at a shared integer
#   is a column;
# - any other call of a function of the cell's numbers (and shared values, each
#   read whole) is that call at every cell, as a broadcast.
#
# Anything else keeps the cell loop: a plate is batched only when every
# statement that varies by cell has a form, the cell's result is a number or a
# `Vector` of numbers, and at least one statement is a product of a shared matrix
# and a cell's vector (a plate of numbers vectorizes as a cell loop already).
#
# The plan is decided at preparation from the exemplars' concrete types, as the
# in-place slice proof is (`readonly_slices.jl`), and the batched tiles run only
# where the call's shared arguments are of those types and the elements of the
# arguments the cells read one by one are of theirs, every argument the cells read
# spans the plate's one axis, and every vector argument holds vectors of one
# length; otherwise the cell loop runs. Values agree with the cell loop to
# rounding, not bit for bit: the products and the row sums add in another order.
# `RK_BATCHED_PLATES=0` keeps the cell loop (the comparison baseline), and
# `JULIA_DEBUG=ReactiveKernels` logs each plate planned batched (group
# `:plate_batching`) with the statements that vary by cell. Not
# batched: the non-allocating step, the opt-in plate reverse, a plate's cell
# query of one cell (one row gains nothing), a cell read at the rows a cell
# window computes, branch arms, nested plates and scans, and a cell matrix
# selected by the cell's index (a coefficient row of a block, `coefficients[b, :]`
# reshaped); the last is a later increment. A partition (`plate_partition`)
# runs its rows batched: its arguments are views of the plate's, of the same
# element types.
_plate_batching_enabled() = get(ENV, "RK_BATCHED_PLATES", "1") != "0"

# Cells per tile: one product per tile and statement.
const _PLATE_BATCH_TILE = 256

# One batched statement: the cell plan's recipe at `recipe`, whose output
# `output` is `kind` (`:scalar`, a number per cell, or `:vector`, a vector per
# cell), computed by `body` over its inputs bound to `names`.
struct _PlateBatchStep
    recipe::Int
    output::ValueId
    kind::Symbol
    names::Vector{Symbol}
    body::Any
end

# The batched form of one plate: its cell plan (`inner`, the identity the
# lowering checks), each argument's role (`:scalar` and `:vector` vary by cell,
# `:shared` is read whole), the argument types it was planned for, the batched
# statements in plan order, and the cell result's kind.
struct _PlateBatchPlan
    inner::Plan
    inputs::Vector{Symbol}
    types::Vector{Any}
    steps::Vector{_PlateBatchStep}
    want::Symbol
end

# Whether the cell of plate recipe `r` has a product, so the plan is worth
# typing.
_plate_batch_candidate(r::Recipe) = _plate_batch_candidate(r.op)
_plate_batch_candidate(op::_AuthoredPlateOp) = _plate_batching_enabled() &&
    !_plate_reverse_enabled() && any(_batch_has_product, getfield(op.kernel, :plan).recipes)
_batch_has_product(q::Recipe) = q.op === Base.:* || _batch_source_has_product(q.source)
_batch_source_has_product(ex) = ex isa Expr && ((ex.head === :call && length(ex.args) >= 3 &&
    ex.args[1] === :*) || any(_batch_source_has_product, ex.args))

# A shared value's kind from its type.
Base.@nospecializeinfer function _batch_shared_kind(@nospecialize(T))
    T isa Type || return :inv
    T <: Number && return :number
    T <: AbstractVector{<:Number} && return :invvec
    T <: AbstractMatrix{<:Number} && return :invmat
    :inv
end
_batch_varies(kind) = kind === :scalar || kind === :vector

"""
    _plate_batch_plan(p, r, types) -> Union{_PlateBatchPlan,Nothing}
    _plate_batch_plan(p, op, inputs, types)

The batched form of plate recipe `r` of `p` (of plate `op` over the values
`inputs` of `p`, as a partition of it reads them), given the types of `p`'s
values (`_readonly_value_types`), or `nothing` when it keeps its cell loop. The
comment at the top of this file states the conditions.
"""
_plate_batch_plan(p::Plan, r::Recipe, types) = _plate_batch_plan(p, r.op, r.inputs, types)
function _plate_batch_plan(p::Plan, op::_AuthoredPlateOp, values, types)
    inner = getfield(op.kernel, :plan)
    g, cg = p.graph, inner.graph
    atomic = (typeof(op)::DataType).parameters[2]
    outer = Any[get(types, canon_id(g, v.id), Any) for v in values]
    length(outer) == length(inner.have) || return nothing
    all(T -> T isa Type && isconcretetype(T), outer) || return nothing
    arguments = Any[_plate_argument_type(T, position in atomic) for (position, T) in enumerate(outer)]
    cell_types = _uninferred_values!(_UninferredValue[], inner, arguments, Symbol[], false)
    kinds = Dict{ValueId,Tuple{Symbol,Any}}()
    inputs = Symbol[]
    for (position, v) in enumerate(inner.have)
        cid = canon_id(cg, v.id)
        A, E = outer[position], arguments[position]
        if position in atomic || A <: Number
            push!(inputs, :shared)
            kinds[cid] = (_batch_shared_kind(E), E)
        elseif A <: AbstractVector && E <: Number
            push!(inputs, :scalar)
            kinds[cid] = (:scalar, E)
        elseif A <: AbstractVector && isconcretetype(E) && E <: AbstractVector{<:Number}
            push!(inputs, :vector)
            kinds[cid] = (:vector, E)
        else
            return nothing
        end
    end
    any(_batch_varies, inputs) || return nothing
    steps = _PlateBatchStep[]
    products = Ref(0)
    for (index, q) in enumerate(inner.recipes)
        length(q.outputs) == 1 || return nothing
        output = canon_id(cg, only(q.outputs).id)
        ins = ValueId[canon_id(cg, v.id) for v in q.inputs]
        all(cid -> haskey(kinds, cid), ins) || return nothing
        T = get(cell_types, output, Any)
        if !any(cid -> _batch_varies(first(kinds[cid])), ins)
            # A plate invariant: computed once, above the cells.
            kinds[output] = (_batch_shared_kind(T), T)
            continue
        end
        q.effectful && return nothing
        step = _plate_batch_step(q, index, output, ins, kinds, products)
        step === nothing && return nothing
        T isa Type && isconcretetype(T) || return nothing
        fits = step.kind === :scalar ? T <: Number : T <: AbstractVector{<:Number}
        fits || return nothing
        kinds[output] = (step.kind, T)
        push!(steps, step)
    end
    products[] > 0 || return nothing
    want = canon_id(cg, only(inner.want).id)
    k = findfirst(step -> step.output == want, steps)
    k === nothing && return nothing
    kind = steps[k].kind
    # A vector result is stored as a copy of its row.
    kind === :vector && !(last(kinds[want]) <: Vector) && return nothing
    @debug "ReactiveKernels: a plate runs as matrix products over tiles of its cells" statements =
        Any[_lane_strip_lines(inner.recipes[step.recipe].source) for step in steps] _group = :plate_batching
    _PlateBatchPlan(inner, inputs, outer, steps, kind)
end

# The batched statement of cell recipe `q`, or `nothing`.
function _plate_batch_step(q::Recipe, index::Int, output::ValueId, ins, kinds, products)
    op = q.op
    if op isa _KernelSourceOp
        analyzed = _cell_source(q)
        analyzed === nothing && return nothing
        source, params, mod = analyzed
        length(params) == length(ins) && allunique(params) || return nothing
    elseif op isa Function && fieldcount(typeof(op)) == 0
        # An exact bare call keeps its function as its operation.
        ref = _batch_function_ref(op)
        ref === nothing && return nothing
        params = Symbol[Symbol("#batch_argument#", j) for j in eachindex(ins)]
        source = Expr(:call, ref, params...)
        mod = parentmodule(op)
    else
        return nothing
    end
    mod isa Module || return nothing
    context = _BatchContext(mod, Dict{Symbol,Tuple{Symbol,Any}}(
        params[j] => kinds[ins[j]] for j in eachindex(params)), 0)
    batched = _batch_expr(context, source)
    batched === nothing && return nothing
    kind, ex, _ = batched
    _batch_varies(kind) || return nothing
    native = _kernel_native_body(ex, mod, Set{Symbol}(params))
    body = try
        macroexpand(mod, native)
    catch
        return nothing
    end
    renamed = Dict{Symbol,Symbol}(name => gensym(Symbol(:batch_, name)) for name in params)
    qualified = _inline_qualify(body, mod, renamed)
    qualified === nothing && return nothing
    products[] += context.products
    _PlateBatchStep(index, output, kind, Symbol[renamed[name] for name in params], qualified)
end

# `GlobalRef` to a function's own binding, or `nothing`.
function _batch_function_ref(f::Function)
    m, name = parentmodule(f), nameof(f)
    isdefined(m, name) && getglobal(m, name) === f ? GlobalRef(m, name) : nothing
end

mutable struct _BatchContext
    mod::Module
    env::Dict{Symbol,Tuple{Symbol,Any}}
    products::Int
end

# A batched subexpression: `(kind, expression, type)`, with `type` the cell's
# value type where known (`nothing` otherwise), or `nothing` without a form.
function _batch_expr(c::_BatchContext, ex)
    if ex isa Symbol
        haskey(c.env, ex) && return (first(c.env[ex]), ex, last(c.env[ex]))
        value = _global_value(c.mod, ex)
        value === nothing && return nothing
        v = something(value)
        v isa Number && _global_isconst(c.mod, ex) && return (:number, ex, typeof(v))
        return nothing
    end
    ex isa Number && return (:number, ex, typeof(ex))
    ex isa Expr || return nothing
    if ex.head === :. && length(ex.args) == 2 && Meta.isexpr(ex.args[2], :tuple)
        f = _batch_callee(c, ex.args[1])
        f === nothing && return nothing
        return _batch_broadcast(c, f, ex.args[2].args,
            args -> Expr(:., ex.args[1], Expr(:tuple, args...)))
    end
    if ex.head === :call && !isempty(ex.args)
        callee, args = ex.args[1], ex.args[2:end]
        any(a -> Meta.isexpr(a, (:parameters, :kw, :...)), args) && return nothing
        if callee isa Symbol && _is_broadcast_operator(callee)
            f = _batch_callee(c, Symbol(String(callee)[2:end]))
            f === nothing && return nothing
            return _batch_broadcast(c, f, args, parts -> Expr(:call, callee, parts...))
        end
        f = _batch_callee(c, callee)
        f === nothing && return nothing
        return _batch_call(c, callee, f, args)
    end
    ex.head === :ref && length(ex.args) >= 2 && return _batch_ref(c, ex.args[1], ex.args[2:end])
    nothing
end

# The function a callee names: a constant binding in the source's module (or a
# `GlobalRef`), never a value of the cell.
function _batch_callee(c::_BatchContext, callee)
    if callee isa Symbol
        haskey(c.env, callee) && return nothing
        _global_isconst(c.mod, callee) || return nothing
        value = _global_value(c.mod, callee)
    elseif callee isa GlobalRef
        isdefined(callee.mod, callee.name) && isconst(callee.mod, callee.name) || return nothing
        value = Some(getglobal(callee.mod, callee.name))
    else
        return nothing
    end
    value === nothing && return nothing
    f = something(value)
    f isa Function ? f : nothing
end

_batch_parts(c, args) = (parts = Any[_batch_expr(c, a) for a in args];
                         any(isnothing, parts) ? nothing : parts)
_batch_eltype(kind, T) = T === nothing ? nothing :
    kind === :vector || kind === :invvec ? eltype(T) : T
_batch_vector_type(E) = E isa Type && isconcretetype(E) ? Vector{E} : nothing

# A shared value computed inside a cell's statement, typed from its operands.
function _batch_shared_call(f, ex, parts)
    types = Any[last(part) for part in parts]
    all(T -> T isa Type, types) || return nothing
    R = Core.Compiler.return_type(f, Tuple{types...})
    R isa Type && isconcretetype(R) || return nothing
    (_batch_shared_kind(R), ex, R)
end

# A dotted call: the same broadcast over the tile, a shared vector read as a row.
function _batch_broadcast(c::_BatchContext, f, args, rebuild)
    parts = _batch_parts(c, args)
    parts === nothing && return nothing
    any(part -> _batch_varies(first(part)), parts) || return nothing
    batched = Any[]
    for (kind, ex, _) in parts
        # A shared matrix, or any other shared value, in a cell's broadcast has
        # no row of its own to read.
        kind in (:invmat, :inv) && return nothing
        push!(batched, kind === :invvec ? _batch_transpose(ex) : ex)
    end
    vector = any(part -> first(part) in (:vector, :invvec), parts)
    E = _batch_result(f, Any[_batch_eltype(first(part), last(part)) for part in parts])
    (vector ? :vector : :scalar, rebuild(batched), vector ? _batch_vector_type(E) : E)
end

_batch_dot(op::Symbol, a, b) = Expr(:call, op, a, b)
_batch_transpose(ex) = Expr(:call, GlobalRef(Base, :transpose), ex)
# The concrete type of `f` over element types `types`, or `nothing`.
function _batch_result(f, types)
    all(T -> T isa Type, types) || return nothing
    R = Core.Compiler.return_type(f, Tuple{types...})
    R isa Type && isconcretetype(R) ? R : nothing
end

# An undotted call over the tile.
function _batch_call(c::_BatchContext, callee, f, args)
    # `sum(f, v)` names its function, which is no value of the cell.
    f === Base.sum && return _batch_sum(c, args)
    parts = _batch_parts(c, args)
    parts === nothing && return nothing
    if !any(part -> _batch_varies(first(part)), parts)
        return _batch_shared_call(f, Expr(:call, callee, (part[2] for part in parts)...), parts)
    end
    f === Base.:* && return foldl((a, b) -> a === nothing ? nothing : _batch_times(c, a, b),
                                  parts[2:end]; init = parts[1])
    if f === Base.:+ || f === Base.:-
        op = f === Base.:+ ? Symbol(".+") : Symbol(".-")
        length(parts) == 1 && return _batch_negate(only(parts), f)
        f === Base.:- && length(parts) != 2 && return nothing
        return foldl((a, b) -> a === nothing ? nothing : _batch_add(a, b, op),
                     parts[2:end]; init = parts[1])
    end
    f === Base.:/ && length(parts) == 2 && return _batch_divide(parts...)
    # Any other function of the cell's numbers: one call per cell, a broadcast,
    # each shared value read whole.
    any(part -> first(part) === :vector, parts) && return nothing
    R = _batch_result(f, Any[last(part) for part in parts])
    R !== nothing && R <: Number || return nothing
    batched = Any[first(part) in (:invvec, :invmat, :inv) ?
                  Expr(:call, GlobalRef(Base, :Ref), part[2]) : part[2] for part in parts]
    (:scalar, Expr(:., callee, Expr(:tuple, batched...)), R)
end

_batch_number(kind) = kind === :number || kind === :scalar

# The element type `f` gives over two batched operands' elements, or `nothing`.
_batch_promote(f, a, b) = _batch_result(f, Any[_batch_eltype(a[1], a[3]), _batch_eltype(b[1], b[3])])

function _batch_times(c::_BatchContext, a, b)
    ka, kb = first(a), first(b)
    if !_batch_varies(ka) && !_batch_varies(kb)
        return _batch_shared_call(Base.:*, Expr(:call, GlobalRef(Base, :*), a[2], b[2]), (a, b))
    end
    E = _batch_promote(Base.:*, a, b)
    if ka === :invmat && kb === :vector
        c.products += 1
        return (:vector, Expr(:call, GlobalRef(Base, :*), b[2], _batch_transpose(a[2])),
                _batch_vector_type(E))
    end
    if (_batch_number(ka) && kb === :vector) || (ka === :vector && _batch_number(kb))
        return (:vector, _batch_dot(Symbol(".*"), a[2], b[2]), _batch_vector_type(E))
    end
    _batch_number(ka) && _batch_number(kb) &&
        return (:scalar, _batch_dot(Symbol(".*"), a[2], b[2]), E)
    ka === :scalar && kb === :invvec &&
        return (:vector, _batch_dot(Symbol(".*"), a[2], _batch_transpose(b[2])), _batch_vector_type(E))
    ka === :invvec && kb === :scalar &&
        return (:vector, _batch_dot(Symbol(".*"), _batch_transpose(a[2]), b[2]), _batch_vector_type(E))
    nothing
end

function _batch_add(a, b, op::Symbol)
    ka, kb = first(a), first(b)
    f = op === Symbol(".+") ? Base.:+ : Base.:-
    if !_batch_varies(ka) && !_batch_varies(kb)
        return _batch_shared_call(f, Expr(:call, GlobalRef(Base, nameof(f)), a[2], b[2]), (a, b))
    end
    E = _batch_promote(f, a, b)
    ka === :vector && kb === :vector &&
        return (:vector, _batch_dot(op, a[2], b[2]), _batch_vector_type(E))
    ka === :vector && kb === :invvec &&
        return (:vector, _batch_dot(op, a[2], _batch_transpose(b[2])), _batch_vector_type(E))
    ka === :invvec && kb === :vector &&
        return (:vector, _batch_dot(op, _batch_transpose(a[2]), b[2]), _batch_vector_type(E))
    _batch_number(ka) && _batch_number(kb) &&
        return (:scalar, _batch_dot(op, a[2], b[2]), E)
    nothing
end

function _batch_negate(a, f)
    _batch_varies(first(a)) || return nothing
    f === Base.:+ && return a
    (first(a), Expr(:call, Symbol(".-"), a[2]), a[3])
end

function _batch_divide(a, b)
    ka, kb = first(a), first(b)
    _batch_number(kb) || return nothing
    E = _batch_promote(Base.:/, a, b)
    ka === :vector && return (:vector, _batch_dot(Symbol("./"), a[2], b[2]), _batch_vector_type(E))
    _batch_number(ka) && return (:scalar, _batch_dot(Symbol("./"), a[2], b[2]), E)
    nothing
end

# `sum(v)` and `sum(f, v)` of a cell's vector: the tile's row sums.
function _batch_sum(c::_BatchContext, args)
    rows(arguments...) = Expr(:call, GlobalRef(Base, :vec), Expr(:call, GlobalRef(Base, :sum),
        Expr(:parameters, Expr(:kw, :dims, 2)), arguments...))
    length(args) in (1, 2) || return nothing
    v = _batch_expr(c, last(args))
    v !== nothing && first(v) === :vector || return nothing
    E = _batch_eltype(:vector, v[3])
    length(args) == 1 && return (:scalar, rows(v[2]), E)
    g = _batch_callee(c, first(args))
    g === nothing && return nothing
    (:scalar, rows(first(args), v[2]), _batch_result(g, Any[E]))
end

# A read of a shared array at the cell's integer, or of a cell's vector at a
# shared integer.
function _batch_ref(c::_BatchContext, array, indices)
    a = _batch_expr(c, array)
    a === nothing && return nothing
    integer(part) = part !== nothing && last(part) isa Type && last(part) <: Integer
    colon(ex) = ex === :(:)
    if first(a) in (:invmat, :invvec)
        length(indices) in (1, 2) || return nothing
        i = indices[1] isa Symbol && haskey(c.env, indices[1]) ? _batch_expr(c, indices[1]) : nothing
        i !== nothing && first(i) === :scalar && integer(i) || return nothing
        E = a[3] isa Type ? eltype(a[3]) : nothing
        if length(indices) == 1
            first(a) === :invvec || return nothing
            return (:scalar, Expr(:ref, a[2], i[2]), E)
        end
        first(a) === :invmat || return nothing
        colon(indices[2]) && return (:vector, Expr(:ref, a[2], i[2], :(:)), _batch_vector_type(E))
        j = _batch_expr(c, indices[2])
        j !== nothing && first(j) === :number && integer(j) || return nothing
        return (:scalar, Expr(:ref, a[2], i[2], j[2]), E)
    end
    if first(a) === :vector && length(indices) == 1
        k = _batch_expr(c, only(indices))
        k !== nothing && first(k) === :number && integer(k) || return nothing
        return (:scalar, Expr(:ref, a[2], :(:), k[2]), _batch_eltype(:vector, a[3]))
    end
    nothing
end

# Runtime helpers of the batched tiles.

# Whether an argument is of the type its plan was made for (`T`): a shared one
# exactly, one the cells read one by one in its elements. Constant where the
# body is compiled.
@inline _plate_batch_fits(::Val{:shared}, ::Type{T}, x) where {T} = typeof(x) === T
@inline _plate_batch_fits(::Val, ::Type{T}, x) where {T} =
    x isa AbstractVector && eltype(x) === eltype(T)

# Whether every vector of `x` has the length of the first.
function _plate_batch_uniform(x::AbstractVector)
    isempty(x) && return true
    n = length(first(x))
    for v in x
        length(v) == n || return false
    end
    true
end
@inline _plate_batch_uniform() = true
@inline _plate_batch_uniform(x, rest...) = _plate_batch_uniform(x) && _plate_batch_uniform(rest...)

# The cells' vectors at `rows`, one row each.
function _plate_batch_rows(x::AbstractVector, rows::AbstractUnitRange{Int})
    v = x[first(rows)]
    n = length(v)
    out = similar(v, eltype(v), (length(rows), n))
    t = 0
    for k in rows
        t += 1
        @inbounds v = x[k]
        for j in 1:n
            @inbounds out[t, j] = v[j]
        end
    end
    out
end

# The batched tiles of plate `plan`, with `cell_loop` where they do not apply
# (the comment at the top of this file). `operand(cid)` is the expression of a
# value every cell shares: a plate argument read whole, or a plate invariant's
# local.
function _lower_plate_batched_native(plan::_PlateBatchPlan, inner::Plan, locals,
                                     callargs, raw_arguments, output_axes,
                                     pointwise_lhs, accumulator, cell_loop)
    graph = inner.graph
    have = Dict(canon_id(graph, v.id) => position for (position, v) in enumerate(inner.have))
    varying = [k for k in eachindex(plan.inputs) if plan.inputs[k] !== :shared]
    any(k -> raw_arguments[k] === nothing, varying) && return cell_loop
    # The types the plan was made for: a shared argument's own, and the
    # elements of an argument the cells read one by one, whatever vector
    # holds them (a partition's view of the plate's).
    fits = Any[Expr(:call, GlobalRef(@__MODULE__, :_plate_batch_fits),
                    Expr(:call, GlobalRef(Base, :Val), QuoteNode(plan.inputs[k])), plan.types[k],
                    plan.inputs[k] === :shared ? callargs[k] : raw_arguments[k])
               for k in eachindex(plan.inputs)]
    ready = Expr(:&&, foldl((a, b) -> Expr(:&&, a, b), fits),
        Expr(:&&, Expr(:call, GlobalRef(@__MODULE__, :_plate_one_axis), output_axes),
             Expr(:&&, Expr(:call, GlobalRef(@__MODULE__, :_plate_spans), output_axes,
                            (raw_arguments[k] for k in varying)...),
                  Expr(:call, GlobalRef(@__MODULE__, :_plate_batch_uniform),
                       (raw_arguments[k] for k in varying if plan.inputs[k] === :vector)...))))
    axis, lo, hi, rows = gensym.((:batch_axis, :batch_lo, :batch_hi, :batch_rows))
    tile = Expr(:block, :($hi = $(GlobalRef(Base, :min))($lo + $(GlobalRef(@__MODULE__, :_PLATE_BATCH_TILE)) - 1,
                                                          $(GlobalRef(Base, :last))($axis))),
                :($rows = $lo:$hi))
    batched = Dict{ValueId,Symbol}()
    for k in varying
        cid = canon_id(graph, inner.have[k].id)
        name = gensym(:batch_cells)
        source = plan.inputs[k] === :scalar ?
            Expr(:call, GlobalRef(Base, :view), raw_arguments[k], rows) :
            Expr(:call, GlobalRef(@__MODULE__, :_plate_batch_rows), raw_arguments[k], rows)
        push!(tile.args, :($name = $source))
        batched[cid] = name
    end
    operand(cid) = haskey(batched, cid) ? batched[cid] :
        haskey(have, cid) ? callargs[have[cid]] : locals[cid]
    for step in plan.steps
        recipe = inner.recipes[step.recipe]
        name = gensym(:batch_value)
        bindings = Any[Expr(:(=), local_name, operand(canon_id(graph, input.id)))
                       for (local_name, input) in zip(step.names, recipe.inputs)]
        push!(tile.args, :($name = $(Expr(:let, Expr(:block, bindings...), step.body))))
        batched[step.output] = name
    end
    result = batched[canon_id(graph, only(inner.want).id)]
    t, cell, value = gensym.((:batch_t, :batch_cell, :batch_value))
    store = Expr(:block, :($cell = $rows[$t]),
        :($value = $(plan.want === :scalar ? :($result[$t]) : :($result[$t, :]))))
    pointwise_lhs === nothing ||
        push!(store.args, _inbounds_expr(:($pointwise_lhs[$cell] = $value)))
    accumulator === nothing || push!(store.args, :($accumulator =
        $(GlobalRef(@__MODULE__, :_plate_total_add))($accumulator, $value)))
    push!(tile.args, Expr(:for, :($t = $(GlobalRef(Base, :eachindex))($rows)), store))
    tiles = quote
        $axis = $(GlobalRef(Base, :only))($output_axes)
        for $lo in $(GlobalRef(Base, :first))($axis):$(GlobalRef(@__MODULE__, :_PLATE_BATCH_TILE)):$(GlobalRef(Base, :last))($axis)
            $tile
        end
    end
    Expr(:if, ready, tiles, cell_loop)
end
