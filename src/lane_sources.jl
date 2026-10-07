# Destination forms of authored source recipes for the native position driver
# (`_lower_replicated_with_ops`).
#
# A position residual allocates every array intermediate afresh at each
# position. Authored plates and scans already fill a recycled buffer
# (`_lane_allocation`); this file gives the same destination protocol to the
# two source shapes that produce a dense array of known shape: a top-level
# dotted call (including `@.`) and a top-level array slice `x[i, j...]`.
#
# The recipe's native closure computes `_native_broadcast_materialize(bc)` or
# `x[...]` from its captured source. The destination form evaluates the same
# lazy broadcast or the same indices from that source and writes the result
# into a matching recycled buffer (`_lane_broadcast`, `_lane_getindex`),
# returning exactly the value the closure would. The source is reinterpreted
# only when that is unambiguous:
#
# - the operation is an ordinary-mode `:fused` source closure without
#   captures (an `on_error = :ignore` twin keeps its own stripped body);
# - the closure's parameters name the recipe inputs positionally, so a
#   composed (renamed) recipe maps by position, exactly as its call does;
# - the source, after the native rewrite the closure itself received
#   (`_kernel_native_body`), is a plain expression of calls, indexing, tuples
#   and literals; every free name is a recipe input or a module binding
#   defined at lowering, resolved as the closure resolves it (`GlobalRef`).
#
# Anything else keeps the ordinary call, so this never changes which value is
# computed; it changes only where a dense result is stored.

_lane_source_ref(name) = GlobalRef(@__MODULE__, name)

# `(native closure, parameter names, module)` of an eligible source operation,
# or `nothing`.
function _lane_source_closure(r::Recipe)
    op = r.op
    op isa _KernelSourceOp || return nothing
    kernel_sourceop_form(op) === :fused || return nothing
    # Ordinary authored operations carry their ignore-mode twin; the twin
    # itself (selected by `on_error = :ignore`) does not, and its body differs.
    op.ignored_throws === nothing && return nothing
    r.source isa Expr || return nothing
    f = _kernel_native_source(op.f)
    fieldcount(typeof(f)) == 0 || return nothing
    candidates = methods(f)
    length(candidates) == 1 || return nothing
    params = Base.method_argnames(only(candidates))[2:end]
    length(params) == length(r.inputs) || return nothing
    all(name -> name isa Symbol, params) || return nothing
    allunique(params) || return nothing
    (f, params, parentmodule(typeof(f)))
end

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
end

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
    elseif head in (:call, :tuple, :parameters, :...)
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
# `recycled` (`_lower_with_ops`), or `nothing` for the ordinary call.
function _lane_source_expr(r::Recipe, callargs, recycled)
    length(r.outputs) == 1 || return nothing
    closure = _lane_source_closure(r)
    closure === nothing && return nothing
    _, params, mod = closure
    source = _lane_expand_dot(_lane_strip_lines(r.source), mod)
    source isa Expr && !_lane_has_macro(source) || return nothing
    native = _lane_strip_lines(_kernel_native_body(source, mod, Set{Symbol}(params)))
    ctx = _LaneSourceContext(mod, Dict{Symbol,Any}(zip(params, callargs)))
    if native isa Expr && native.head === :call && length(native.args) == 2 &&
       native.args[1] == _lane_source_ref(:_native_broadcast_materialize)
        lazy = _lane_source_node(ctx, native.args[2], nothing)
        lazy === nothing && return nothing
        return Expr(:call, _lane_source_ref(:_lane_broadcast), recycled, lazy)
    elseif native isa Expr && native.head === :ref && length(native.args) >= 2
        array = _lane_source_node(ctx, native.args[1], nothing)
        array === nothing && return nothing
        any(index -> index isa Expr && index.head === :..., native.args[2:end]) &&
            return nothing
        # `end`/`begin` in the k-th of n indices are `lastindex(x, k)` and
        # `firstindex(x, k)`, or the one-argument forms for a single index.
        held = gensym(:lane_array)
        count = length(native.args) - 1
        indices = Any[]
        for (k, index) in enumerate(native.args[2:end])
            position = count == 1 ? () : (k,)
            ends = (Expr(:call, GlobalRef(Base, :lastindex), held, position...),
                    Expr(:call, GlobalRef(Base, :firstindex), held, position...))
            rewritten = _lane_source_node(ctx, index, ends)
            rewritten === nothing && return nothing
            push!(indices, rewritten)
        end
        return Expr(:let, Expr(:(=), held, array),
            Expr(:call, _lane_source_ref(:_lane_getindex), recycled, held, indices...))
    end
    nothing
end

# Whether a recipe has a destination form, independently of its call site.
_lane_source_eligible(r::Recipe) =
    _lane_source_expr(r, Any[gensym(:argument) for _ in r.inputs], :recycled) !== nothing
