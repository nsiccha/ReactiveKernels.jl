# Destination forms of authored source recipes for the native position driver
# (`_lower_replicated_with_ops`).
#
# A position residual allocates every array intermediate afresh at each
# position. Authored plates and scans already fill a recycled buffer
# (`_lane_allocation`); this file gives the same destination protocol to the
# source shapes that produce a dense array of known shape: a top-level dotted
# call (including `@.`), a top-level array slice `x[i, j...]`, a top-level
# `reduce(vcat, cells)` of dense vectors, and a top-level lazy branch
# (`c ? a : b`, `if`/`elseif`/`else`) whose arms produce arrays. Each arm of
# such a branch that is one of these shapes, or `zeros`/`ones`/`fill` of
# integer dimensions, fills the recipe's buffer; the branch stays lazy, so
# only the taken arm runs. A slice arm counts only beside an arm that is an
# array by its syntax (a dotted call, `zeros`/`ones`/`fill`, a slice by a
# range or `:`): `i > 0 ? x[i] : 0.0` reads a scalar and keeps its call.
#
# A dotted call or a `reduce(vcat, cells)` nested inside such a source, as the
# argument of another call (`minimum(x .- y)`) or inside a branch arm, fills a
# lane slot of its own. The recipe's value releases every such slot it may
# hold (`_lane_release!`), so a value that keeps the buffer (an opaque function
# returned it, a tuple holds it) is never overwritten. Nested slots are free names the position driver binds
# (`_CELL_SLOTS`); outside a position residual a nested call keeps its own
# allocation.
#
# The recipe's native closure computes `_native_broadcast_materialize(bc)`,
# `x[...]` or `reduce(vcat, cells)` from its captured source. The destination
# form evaluates the same lazy broadcast, indices or cells from that source and
# writes the result into a matching recycled buffer (`_lane_broadcast`,
# `_lane_getindex`, `_lane_reduce_vcat`, `_lane_zeros`, `_lane_ones`,
# `_lane_fill`), returning exactly the value the closure would. The source is
# reinterpreted only when that is unambiguous:
#
# - the operation is an ordinary-mode `:fused` source closure without
#   captures (an `on_error = :ignore` twin keeps its own stripped body), or
#   the all-in-one closure of such a lazy branch (`_KernelBranch`);
# - the closure's parameters name the recipe inputs positionally, so a
#   composed (renamed) recipe maps by position, exactly as its call does;
# - the source, after the native rewrite the closure itself received
#   (`_kernel_native_body`), is a plain expression of calls, indexing, tuples,
#   literals, subtype tests (`a <: b`, `a >: b`), type applications (`A{B}`),
#   lazy branches and short-circuit operators; every free name is a recipe
#   input or a module binding defined at lowering, resolved as the closure
#   resolves it (`GlobalRef`), and a subtype test calls the `<:` or `>:` its
#   module resolves, as Julia's lowering does.
#
# Anything else keeps the ordinary call, so this never changes which value is
# computed; it changes only where a dense result is stored.
#
# An RK-owned callee whose mathematics is a graph has a destination form too:
# a derivative rule's primal cut, called as a recipe's operation or in a lazy
# arm by its bare or module-qualified name (`_lane_binding`), and a child
# kernel prepared for a lazy arm (`_KernelPreparedChild`).
# Its lane kernel (`_lane_kernel`) is its plan lowered as a position residual
# part: its WANT fills the call's recycled buffer, and its own lane values and
# cell values fill slots kept per call site (`_LaneCalleeState`, in a nested
# slot of the caller). So the callee's plates, scans, dotted calls and slices
# reuse their buffers as the same mathematics spliced into the caller would.
# Only position residuals reach these forms; every other call, AD included,
# runs the callee's ordinary body.

_lane_source_ref(name) = GlobalRef(@__MODULE__, name)

"""
    _LaneKernel{N}

A callee's plan lowered as a position residual part (`_lane_kernel`): the
compiled body `f(ops, recycled, slots, inputs...)` and the table it reads,
with `N` slots for the callee's own lane and cell values.
"""
struct _LaneKernel{N,F,C,O}
    f::F
    constants::C
    callables::O
end
_LaneKernel{N}(f::F, constants::C, callables::O) where {N,F,C,O} =
    _LaneKernel{N,F,C,O}(f, constants, callables)

# The callables table of a native body, with every entry the body does not
# read set to `nothing`. A plate or scan keeps its operation in an entry the
# native body does not read, and that operation holds its prepared body; the
# rest of the table is singleton source operations, so the callable holding
# it stays isbits. A body that reads the table other than by literal index
# keeps the whole table.
function _native_table(ast::Expr, ops::Tuple)
    reads = _callable_reads(ast.args[2])
    reads === nothing ? ops : ntuple(i -> i in reads ? ops[i] : nothing, length(ops))
end

# The lane kernel of a plan with one WANT that a destination form produces,
# or `nothing` (no buffer to reuse). The plan is lowered as
# `_lower_replicated_with_ops` lowers its residual, without per-cell tables:
# a callee may run several times per position, so a stored cell result of its
# own plates is not dead at its next call.
function _lane_kernel(p::Plan)
    length(p.want) == 1 || return nothing
    g = p.graph
    cid = canon_id(g, only(p.want).id)
    any(v -> canon_id(g, v.id) == cid, p.have) && return nothing
    producer = get(p.producer, cid, nothing)
    producer isa Recipe && _replicated_lane_producer(producer) || return nothing
    output = gensym(:lane_output)
    recycle = Pair{ValueId,Symbol}[cid => output]
    excluded = Set(canon_id(g, v.id) for v in vcat(p.want, p.have))
    scratch = Symbol[]
    for v in _lane_slot_values(p, excluded)
        slot = gensym(Symbol(v.name, :_lane_slot))
        push!(recycle, canon_id(g, v.id) => slot)
        push!(scratch, slot)
    end
    cell_slots = Symbol[]
    ast, constants, callables, _ = _replicated_part_ast(p; recycle, cell_slots)
    slots = gensym(:lane_slots)
    inputs = Any[gensym(Symbol(v.name, :_lane_input)) for v in p.have]
    body = Expr(:block)
    for (index, slot) in enumerate([scratch; cell_slots])
        push!(body.args, :($slot = $(GlobalRef(Base, :getindex))($slots, $index)))
    end
    result = gensym(:lane_result)
    append!(body.args, _embedded_statements(ast, Any[inputs..., output, scratch...],
                                            result, 0))
    push!(body.args, Expr(:return, result))
    wrapper = Expr(:function,
        Expr(:tuple, _CONSTANTS_ARG, _CALLABLES_ARG, output, slots, inputs...), body)
    _LaneKernel{length(scratch) + length(cell_slots)}(
        compile(wrapper), Tuple(constants), _native_table(wrapper, Tuple(callables)))
end

# The lane kernel of an RK-owned callee, or `nothing`: a rule's
# (derivative_rules.jl) or a lazy-arm child's, built with the callee.
_lane_callee(f) = nothing
_lane_callee(child::_KernelPreparedChild) = getfield(child, :lane)

# The destination form of a callee call: the lane kernel with the call site's
# slots, or the ordinary call for a callee without one. The lane kernel is its
# own call, as the ordinary callee is (`_KernelPreparedChild`): inlined into a
# large residual, the result-type queries of its scans widened to `Any` once
# the session had inferred the same scan step elsewhere, and every step then
# dispatched at run time.
@inline _lane_call(recycled, slot, f, args::Vararg{Any,N}) where {N} =
    _lane_call_with(recycled, slot, _lane_callee(f), f, args...)
@inline _lane_call_with(recycled, slot, ::Nothing, f, args::Vararg{Any,N}) where {N} =
    f(args...)
@inline _lane_call_with(recycled, slot, lane::_LaneKernel, f, args::Vararg{Any,N}) where {N} =
    _lane_call_kernel(recycled, slot, lane, args...)
@noinline _lane_call_kernel(recycled, slot::Base.RefValue{Any}, lane::_LaneKernel{M},
                            args::Vararg{Any,N}) where {M,N} =
    _lane_kernel_body(lane, recycled, getfield(_lane_callee_state!(slot, M), :slots),
                      args...)
@generated function _lane_kernel_body(lane::_LaneKernel{M,F}, recycled, slots,
                                      args::Vararg{Any,N}) where {M,F,N}
    forwarded = Any[:(getfield(args, $index)) for index in 1:N]
    quote
        $(Expr(:meta, :inline))
        $(_native_body_call_expr(F, :(getfield(lane, :f)),
            Any[:(getfield(lane, :constants)), :(getfield(lane, :callables)),
                :recycled, :slots, forwarded...]))
    end
end

# `(native closure, parameter names, module)` of an eligible source operation,
# or `nothing`. `branches`: whether a lazy branch's all-in-one closure counts
# (`_lane_source_expr` only; the slice and window analyses read other shapes).
# `outlined`: `r` is a part a cell lowering outlined from an authored
# operation (`_cell_outline`), which carries no twin of its own.
function _lane_source_closure(r::Recipe; branches::Bool = false, outlined::Bool = false)
    op = r.op
    op isa _KernelSourceOp || return nothing
    kernel_sourceop_form(op) === :fused || return nothing
    # Ordinary authored operations carry their ignore-mode twin; the twin
    # itself (selected by `on_error = :ignore`) does not, and its body differs.
    outlined || getfield(op, :ignored_throws) !== nothing || return nothing
    r.source isa Expr || return nothing
    branches || !(getfield(op, :f) isa _KernelBranch) || return nothing
    f = _kernel_native_source(_lane_source_callable(getfield(op, :f)))
    fieldcount(typeof(f)) == 0 || return nothing
    params = _lane_source_params(f, length(r.inputs))
    params === nothing && return nothing
    (f, params, parentmodule(typeof(f)))
end

# The parameter names of a source closure's one method, when it takes `n`
# distinct named parameters, or `nothing`. The source reads its inputs by these
# names, which need not be the input values' names: a composed child's recipe
# reads its formals (`A * w`) over the caller's values (`X`, `b`), or over
# values scoped under the caller's name (`pointwise.X`).
Base.@nospecializeinfer function _lane_source_params(@nospecialize(f), n)
    candidates = methods(f)
    length(candidates) == 1 || return nothing
    params = Base.method_argnames(only(candidates))[2:end]
    length(params) == n || return nothing
    all(name -> name isa Symbol, params) || return nothing
    allunique(params) || return nothing
    params
end

# The source's names of the closure's parameters. A method stores a gensym
# parameter without its `##` prefix (`var"##cell_invariant#7"`, the name a
# lifted cell invariant reads by, is the parameter `cell_invariant#7`), so the
# source may read a parameter by either spelling.
function _lane_source_arguments(params, callargs)
    arguments = Dict{Symbol,Any}(zip(params, callargs))
    for (param, callarg) in zip(params, callargs)
        name = string(param)
        occursin('#', name) && !startswith(name, "##") || continue
        get!(arguments, Symbol("##", name), callarg)
    end
    arguments
end

# A lazy branch runs its authored all-in-one closure (`_KernelBranch`); a branch
# that preparation rebuilt from its parts (`_KernelBranchCall`) is no closure.
Base.@nospecializeinfer _lane_source_callable(@nospecialize(f)) = f
Base.@nospecializeinfer _lane_source_callable(@nospecialize(branch::_KernelBranch)) =
    getfield(branch, :call)

function _lane_strip_lines(ex)
    ex isa Expr || return ex
    args = Any[_lane_strip_lines(arg) for arg in ex.args if !(arg isa LineNumberNode)]
    ex.head === :block && length(args) == 1 && return only(args)
    Expr(ex.head, args...)
end

# `@.`/`@__dot__` is Base's pure syntactic rewrite; expand it exactly as the
# closure's definition did. Any other macro keeps the ordinary call.
function _lane_expand_dot(ex, mod::Module)
    ex isa Expr && ex.head === :macrocall || return ex
    length(ex.args) == 2 || return nothing
    name = ex.args[1]
    name === Symbol("@__dot__") && isdefined(mod, name) &&
        getglobal(mod, name) === getglobal(Base.Broadcast, Symbol("@__dot__")) ||
        return nothing
    Base.Broadcast.__dot__(ex.args[2])
end

_lane_has_macro(ex) = ex isa Expr &&
    (ex.head === :macrocall || any(_lane_has_macro, ex.args))

struct _LaneSourceContext
    mod::Module
    arguments::Dict{Symbol,Any}
    # The lane slots of nested dotted calls (`_lane_broadcast`), or `nothing`
    # where they cannot be bound: a nested call then keeps its allocation.
    nested::Union{Nothing,Vector{Symbol}}
    # Whether lazy branches and short-circuit operators are in the grammar
    # (`_lane_source_expr`); other rewrites keep the plain grammar.
    lazy::Bool
end
_LaneSourceContext(mod, arguments) = _LaneSourceContext(mod, arguments, nothing, false)

# The callee a call node names, when it has a lane kernel: a quoted object
# (`_KernelPreparedChild`) or a module binding, read as the closure reads it.
function _lane_node_callee(ctx::_LaneSourceContext, node)
    value = node isa QuoteNode ? node.value : _lane_binding(ctx, node)
    value === nothing || _lane_callee(value) === nothing ? nothing : value
end

# `_lane_call(recycled, state, callee, args...)` for a call whose callee has a
# lane kernel, with a fresh nested slot for the call site's state, or
# `nothing` where nested slots cannot be bound.
function _lane_callee_destination(ctx::_LaneSourceContext, callee, args, recycled)
    ctx.nested === nothing && return nothing
    state = gensym(:lane_callee)
    push!(ctx.nested, state)
    Expr(:call, _lane_source_ref(:_lane_call), recycled, state, callee, args...)
end

_lane_materialize_call(node) = node isa Expr && node.head === :call &&
    length(node.args) == 2 &&
    node.args[1] == _lane_source_ref(:_native_broadcast_materialize)

# Rewrite one node of the native source: inputs become their call arguments,
# other names `GlobalRef`s. `ends` holds the expressions that `end`/`begin`
# mean at this position (inside an index of the enclosing slice), or `nothing`
# where they are not ours to resolve (inside a nested indexing expression,
# which Julia lowers itself). Returns `nothing` outside the grammar.
function _lane_source_node(ctx::_LaneSourceContext, node, ends)
    if node isa Symbol
        if node === :end || node === :begin
            ends === nothing && return node
            return node === :end ? ends[1] : ends[2]
        end
        haskey(ctx.arguments, node) && return ctx.arguments[node]
        isdefined(ctx.mod, node) || return nothing
        return GlobalRef(ctx.mod, node)
    end
    node isa Union{Number,String,Char,QuoteNode,GlobalRef} && return node
    node isa Expr || return nothing
    head = node.head
    if head === :kw
        length(node.args) == 2 && node.args[1] isa Symbol || return nothing
        value = _lane_source_node(ctx, node.args[2], ends)
        value === nothing && return nothing
        return Expr(:kw, node.args[1], value)
    elseif head === :.
        # Module/property access `a.b`; dotted calls were rewritten already.
        length(node.args) == 2 && node.args[2] isa QuoteNode || return nothing
        owner = _lane_source_node(ctx, node.args[1], ends)
        owner === nothing && return nothing
        return Expr(:., owner, node.args[2])
    elseif head === :ref
        # A nested slice keeps Julia's own lowering of its `end`/`begin`; its
        # array expression still sees ours.
        length(node.args) >= 1 || return nothing
        array = _lane_source_node(ctx, node.args[1], ends)
        array === nothing && return nothing
        indices = Any[]
        for index in node.args[2:end]
            rewritten = _lane_source_node(ctx, index, nothing)
            rewritten === nothing && return nothing
            push!(indices, rewritten)
        end
        return Expr(:ref, array, indices...)
    elseif head === :call && ctx.nested !== nothing && _lane_materialize_call(node)
        # A nested dotted call fills its own lane slot.
        lazy = _lane_source_node(ctx, node.args[2], ends)
        lazy === nothing && return nothing
        slot = gensym(:lane_nested)
        push!(ctx.nested, slot)
        return Expr(:call, _lane_source_ref(:_lane_broadcast), slot, lazy)
    elseif head === :call && ctx.nested !== nothing && _lane_reduce_vcat_call(ctx, node)
        # So does a nested `reduce(vcat, cells)`.
        cells = _lane_source_node(ctx, node.args[3], ends)
        cells === nothing && return nothing
        slot = gensym(:lane_nested)
        push!(ctx.nested, slot)
        return Expr(:call, _lane_source_ref(:_lane_reduce_vcat), slot, cells)
    elseif head in (:<:, :>:) && length(node.args) == 2
        # Julia lowers `a <: b` to a call of the `<:` its module resolves. The
        # one-operand form is a bound in a type application, outside the grammar.
        args = Any[_lane_source_node(ctx, arg, ends) for arg in (head, node.args...)]
        any(isnothing, args) && return nothing
        return Expr(:call, args...)
    elseif head === :curly
        # Julia lowers `A{B}` to `Core.apply_type` and `A{<:B}` to a `where`
        # type, in any module: keep the syntax over the rewritten operands.
        args = Any[]
        for arg in node.args
            bound = arg isa Expr && arg.head in (:<:, :>:)
            bound && length(arg.args) != 1 && return nothing
            rewritten = _lane_source_node(ctx, bound ? only(arg.args) : arg, ends)
            rewritten === nothing && return nothing
            push!(args, bound ? Expr(arg.head, rewritten) : rewritten)
        end
        return Expr(:curly, args...)
    elseif head in (:call, :tuple, :parameters, :...) ||
           ctx.lazy && head in (:if, :elseif, :&&, :||)
        # Lazy branches and short-circuit operators keep their heads, so only
        # the arm or operand Julia would evaluate runs.
        args = Any[]
        for arg in node.args
            rewritten = _lane_source_node(ctx, arg, ends)
            rewritten === nothing && return nothing
            push!(args, rewritten)
        end
        return Expr(head, args...)
    end
    nothing
end

# The destination form of recipe `r`, whose arguments are `callargs`, filling
# `recycled` (`_lower_with_ops`), or `nothing` for the ordinary call. `nested`
# collects the lane slots of nested dotted calls; it defaults to the cell slots
# of the position residual being lowered (`_CELL_SLOTS`), and is `nothing`
# outside one. The slots are added only when the form is returned.
function _lane_source_expr(r::Recipe, callargs, recycled;
        nested::Union{Nothing,Vector{Symbol}} =
            get(task_local_storage(), _CELL_SLOTS, nothing))
    length(r.outputs) == 1 || return nothing
    if _lane_callee(r.op) !== nothing
        # The recipe's operation is the callee. It is quoted, as a lazy arm's
        # child is: read from the operation table, a rule that is not isbits
        # was boxed at every call.
        ctx = _LaneSourceContext(@__MODULE__, Dict{Symbol,Any}(), nested, false)
        call = _lane_callee_destination(ctx, QuoteNode(r.op), callargs, recycled)
        call === nothing && return nothing
        value = gensym(:lane_value)
        return Expr(:block, Expr(:(=), value, call),
                    Expr(:call, _lane_source_ref(:_lane_release!), value, last(nested)))
    end
    closure = _lane_source_closure(r; branches = true)
    closure === nothing && return nothing
    _, params, mod = closure
    source = _lane_expand_dot(_lane_strip_lines(r.source), mod)
    source isa Expr && !_lane_has_macro(source) || return nothing
    arguments = _lane_source_arguments(params, callargs)
    native = _lane_strip_lines(_kernel_native_body(source, mod, Set{Symbol}(keys(arguments))))
    slots = nested === nothing ? nothing : Symbol[]
    ctx = _LaneSourceContext(mod, arguments, slots, true)
    form = _lane_destination(ctx, native, recycled, nothing)
    form === nothing && return nothing
    expr, filled = form
    held = slots === nothing ? Symbol[] : slots
    filled || !isempty(held) || return nothing
    nested === nothing || append!(nested, held)
    isempty(held) && return expr
    # The value releases every nested slot it may hold.
    value = gensym(:lane_value)
    Expr(:block, Expr(:(=), value, expr),
         Expr(:call, _lane_source_ref(:_lane_release!), value, held...))
end

# `(expression, filled)`: `node` rewritten as `_lane_source_node` rewrites it,
# where the shapes with a destination form fill `recycled` (`filled`), or
# `nothing` outside the grammar. `slices`: whether a slice fills `recycled`,
# `nothing` at the source's top level, where a slice always does. A branch
# admits its slice arms only beside an arm that is an array by its syntax
# (`_lane_array_syntax`), decided once for the whole branch.
function _lane_destination(ctx::_LaneSourceContext, node, recycled,
                           slices::Union{Nothing,Bool})
    plain() = (rewritten = _lane_source_node(ctx, node, nothing);
               rewritten === nothing ? nothing : (rewritten, false))
    node isa Expr || return plain()
    if _lane_materialize_call(node)
        lazy = _lane_source_node(ctx, node.args[2], nothing)
        lazy === nothing && return nothing
        return (Expr(:call, _lane_source_ref(:_lane_broadcast), recycled, lazy), true)
    elseif node.head === :ref && length(node.args) >= 2
        slices === false && return plain()
        array = _lane_source_node(ctx, node.args[1], nothing)
        array === nothing && return nothing
        any(index -> index isa Expr && index.head === :..., node.args[2:end]) &&
            return nothing
        # `end`/`begin` in the k-th of n indices are `lastindex(x, k)` and
        # `firstindex(x, k)`, or the one-argument forms for a single index.
        held = gensym(:lane_array)
        count = length(node.args) - 1
        indices = Any[]
        for (k, index) in enumerate(node.args[2:end])
            position = count == 1 ? () : (k,)
            ends = (Expr(:call, GlobalRef(Base, :lastindex), held, position...),
                    Expr(:call, GlobalRef(Base, :firstindex), held, position...))
            rewritten = _lane_source_node(ctx, index, ends)
            rewritten === nothing && return nothing
            push!(indices, rewritten)
        end
        return (Expr(:let, Expr(:(=), held, array),
            Expr(:call, _lane_source_ref(:_lane_getindex), recycled, held, indices...)),
            true)
    elseif _lane_reduce_vcat_call(ctx, node)
        cells = _lane_source_node(ctx, node.args[3], nothing)
        cells === nothing && return nothing
        return (Expr(:call, _lane_source_ref(:_lane_reduce_vcat), recycled, cells), true)
    elseif (fill = _lane_fill_call(ctx, node)) !== nothing
        args = Any[]
        for arg in node.args[2:end]
            rewritten = _lane_source_node(ctx, arg, nothing)
            rewritten === nothing && return nothing
            push!(args, rewritten)
        end
        return (Expr(:call, _lane_source_ref(fill), recycled, args...), true)
    elseif node.head in (:if, :elseif) && length(node.args) == 3
        condition = _lane_source_node(ctx, node.args[1], nothing)
        condition === nothing && return nothing
        # The arms decide the slices of the whole branch, at its root.
        arm_slices = slices === nothing ? _lane_array_syntax(ctx, node) : slices
        arms = Any[]
        filled = false
        for arm in node.args[2:3]
            form = _lane_destination(ctx, arm, recycled, arm_slices)
            form === nothing && return nothing
            push!(arms, first(form))
            filled |= last(form)
        end
        return (Expr(node.head, condition, arms...), filled)
    elseif node.head === :call && !isempty(node.args) &&
           _lane_node_callee(ctx, node.args[1]) !== nothing
        # A call of an RK-owned callee with a lane kernel (`_lane_call`).
        callee = node.args[1] isa QuoteNode ? node.args[1] :
            _lane_source_node(ctx, node.args[1], nothing)
        args = Any[]
        for arg in node.args[2:end]
            arg isa Expr && arg.head in (:parameters, :kw, :...) && return plain()
            rewritten = _lane_source_node(ctx, arg, nothing)
            rewritten === nothing && return nothing
            push!(args, rewritten)
        end
        call = _lane_callee_destination(ctx, callee, args, recycled)
        call === nothing && return plain()
        return (call, true)
    end
    plain()
end

# Whether `node` is a call with arguments, without keywords or splats.
_lane_plain_call(node) = node isa Expr && node.head === :call &&
    length(node.args) >= 2 &&
    !any(arg -> arg isa Expr && arg.head in (:parameters, :kw, :...), node.args)

# The module binding a name of the source resolves to, as the closure resolves
# it, or `nothing` for a recipe input, an unbound name or another node. A
# qualified name (`Mod.Sub.f`) resolves through its owners
# (`_kernel_resolve_binding`), unless a recipe input is its root.
function _lane_binding(ctx::_LaneSourceContext, name)
    root = name
    while root isa Expr && root.head === :. && length(root.args) == 2
        root = root.args[1]
    end
    root isa Symbol && haskey(ctx.arguments, root) && return nothing
    _kernel_resolve_binding(ctx.mod, name)
end

# The destination name of `zeros`, `ones` or `fill` of Base, called without
# keywords or splats, or `nothing`.
function _lane_fill_call(ctx::_LaneSourceContext, node)
    _lane_plain_call(node) || return nothing
    f = _lane_binding(ctx, node.args[1])
    f === Base.zeros ? :_lane_zeros : f === Base.ones ? :_lane_ones :
        f === Base.fill ? :_lane_fill : nothing
end

# Whether `node` is Base's `reduce(vcat, x)`, without keywords (an `init`
# folds pairwise, a different method) or splats.
_lane_reduce_vcat_call(ctx::_LaneSourceContext, node) =
    _lane_plain_call(node) && length(node.args) == 3 &&
    _lane_binding(ctx, node.args[1]) === Base.reduce &&
    _lane_binding(ctx, node.args[2]) === Base.vcat

# Whether a branch has a leaf arm that is an array by its syntax: a dotted call,
# `zeros`/`ones`/`fill`, or a slice by a range or `:`.
function _lane_array_syntax(ctx::_LaneSourceContext, node)
    node isa Expr || return false
    node.head in (:if, :elseif) && length(node.args) == 3 &&
        return any(arm -> _lane_array_syntax(ctx, arm), node.args[2:3])
    _lane_materialize_call(node) && return true
    _lane_fill_call(ctx, node) === nothing || return true
    node.head === :ref || return false
    colon(f) = f === :(:) || f isa GlobalRef && f.name === :(:)
    any(node.args[2:end]) do index
        colon(index) || index isa Expr && index.head === :call &&
            !isempty(index.args) && colon(index.args[1])
    end
end

# Whether a recipe has a destination form, independently of its call site.
_lane_source_eligible(r::Recipe) = _lane_source_expr(
    r, Any[gensym(:argument) for _ in r.inputs], :recycled; nested = Symbol[]) !== nothing
