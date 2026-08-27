# Lowering a Plan to ordinary straight-line Julia, the optional AST-transform
# boundary, and final RGF compilation into a PreparedKernel.
#
# Operations are *not* referenced as globals in the generated code (that would
# be fragile under world age; gist §9). Instead every selected recipe's `op` is
# passed positionally in an `__ops__` tuple and called by literal index, so the
# generated body is closed over nothing and specializes on the concrete op
# types. The hot path therefore touches no graph object.

const _OPS_ARG = :__ops__
const _CACHES_ARG = :__caches__
const _CACHE_APPLY_ARG = :__cache_apply__

# Assign a globally unique source-variable Symbol to every canonical value in
# the plan. User names are diagnostic hints, not binding authority: they may
# collide with one another, with a generated disambiguation such as `a_12`, or
# with the hidden `__ops__` argument.
function _varnames(p::Plan)
    g = p.graph
    ids = Int[]
    for v in p.have; push!(ids, canon_id(g, v.id)); end
    for r in p.recipes, o in r.outputs; push!(ids, canon_id(g, o.id)); end
    for w in p.want; push!(ids, canon_id(g, w.id)); end
    unique!(ids)
    used = Set{Symbol}((_OPS_ARG, _CACHES_ARG, _CACHE_APPLY_ARG))
    out = Dict{Int,Symbol}()
    for id in ids
        base = p.graph.values[id].name
        candidate = base
        suffix = 0
        while candidate in used
            candidate = suffix == 0 ? Symbol(base, :_, id) :
                        Symbol(base, :_, id, :_, suffix)
            suffix += 1
        end
        out[id] = candidate
        push!(used, candidate)
    end
    out
end

"""
    lower(p::Plan) -> Expr

Lower a plan to an ordinary anonymous-function `Expr` of the form

    function (__ops__, x::T1, y::T2)
        a = __ops__[1](x, y)
        ...
        return out
    end

This `Expr` is a first-class artifact: it may be inspected (`code_expr`) and
rewritten (`transform`) before compilation (gist §9).
"""
function lower(p::Plan)
    g = p.graph
    names = _varnames(p)
    nm(v) = names[canon_id(g, v.id)]
    argexprs = Any[_OPS_ARG]
    for v in p.have
        push!(argexprs, :($(nm(v))::$(valtype(v))))
    end
    body = Expr(:block)
    # HAVE is authoritative, and the first selected producer of any other
    # logical value owns its binding. Later recipes may emit that value as a
    # collateral multi-output; execute the recipe but discard the duplicate so
    # neither authoritative inputs nor earlier logical values are overwritten.
    assigned = Set(canon_id(g, v.id) for v in p.have)
    for (k, r) in enumerate(p.recipes)
        callargs = Any[nm(inp) for inp in r.inputs]
        call = Expr(:call, Expr(:ref, _OPS_ARG, k), callargs...)
        lhsnames = Any[]
        for output in r.outputs
            cid = canon_id(g, output.id)
            if cid in assigned
                push!(lhsnames, gensym(Symbol(nm(output), :_discard)))
            else
                push!(assigned, cid)
                push!(lhsnames, nm(output))
            end
        end
        if length(lhsnames) == 1
            push!(body.args, Expr(:(=), only(lhsnames), call))
        else
            lhs = Expr(:tuple, lhsnames...)
            push!(body.args, Expr(:(=), lhs, call))
        end
    end
    retval = length(p.want) == 1 ? nm(p.want[1]) :
             Expr(:tuple, (nm(w) for w in p.want)...)
    push!(body.args, Expr(:return, retval))
    Expr(:function, Expr(:tuple, argexprs...), body)
end

"""
    lower_batched(p::Plan; batched, reduce = :+) -> Expr

Lower a plan to a batched, **loop-invariant-hoisting** kernel. The `batched` HAVE
ports are arrays iterated element-wise; every recipe that does NOT (transitively)
depend on a batched value is emitted ONCE above the loop (the hoist), and only
the batch-dependent recipes run per element, their single scalar `want`
accumulated by `reduce` (default `:+`, i.e. a sum). This is how a vectorized log
density avoids recomputing shared work: e.g. `σ = exp(logσ)` / `log(σ)` is
computed once, not per observation. The generated function has the same
`(__ops__, have...)` signature as [`lower`](@ref) — batched ports passed as
arrays, shared ports as scalars — and the whole batch is one straight-line pass
that materializes no per-element vector.

Restricted (this lowering) to single-output recipes and a single scalar `want`
that is itself batched (the per-element density). `reduce` is spliced as a bare
callee, so use a `Base` reducer symbol (`:+`).
"""
function lower_batched(p::Plan; batched, reduce = :+)
    g = p.graph
    length(p.want) == 1 || throw(ArgumentError(
        "lower_batched requires a single scalar want; got $(length(p.want))"))
    for r in p.recipes
        length(r.outputs) == 1 || throw(ArgumentError(
            "lower_batched requires single-output recipes; recipe $(r.id) has $(length(r.outputs)) outputs"))
    end
    batched_names = Set{Symbol}(batched isa Symbol ? (batched,) : batched)
    names = _varnames(p)
    nm(v) = names[canon_id(g, v.id)]

    # Canonical ids of the batched HAVE ports — the arrays we index per element.
    batched_input_ids = Set{Int}()
    iter_port = nothing
    for v in p.have
        if v.name in batched_names
            push!(batched_input_ids, canon_id(g, v.id))
            iter_port === nothing && (iter_port = v)
        end
    end
    iter_port === nothing && throw(ArgumentError(
        "lower_batched: none of the have ports are batched (batched = $(sort(collect(batched_names))))"))

    # Propagate batch-dependency over produced values in execution (topological)
    # order: a value is batched iff its producing recipe consumes a batched value.
    batched_vals = Set{Int}(batched_input_ids)
    recipe_is_batched = falses(length(p.recipes))
    for (k, r) in enumerate(p.recipes)
        b = any(canon_id(g, inp.id) in batched_vals for inp in r.inputs)
        recipe_is_batched[k] = b
        b && for o in r.outputs
            push!(batched_vals, canon_id(g, o.id))
        end
    end

    want = only(p.want)
    canon_id(g, want.id) in batched_vals || throw(ArgumentError(
        "lower_batched: want :$(want.name) is loop-invariant (does not depend on a batched port); nothing to vectorize"))

    idx = gensym(:i)
    # A batched HAVE port is indexed by the loop var; everything else (an
    # invariant value computed once, or a batched intermediate that is loop-local)
    # is referenced by name.
    argref(inp) = canon_id(g, inp.id) in batched_input_ids ?
                  Expr(:ref, nm(inp), idx) : nm(inp)
    callexpr(k, r) = Expr(:call, Expr(:ref, _OPS_ARG, k),
                          (argref(inp) for inp in r.inputs)...)

    argexprs = Any[_OPS_ARG]
    for v in p.have
        # A batched port arrives as an array, so it is left unannotated (Julia
        # still specializes on the concrete array type); shared ports keep their
        # declared scalar type.
        push!(argexprs, canon_id(g, v.id) in batched_input_ids ?
              nm(v) : :($(nm(v))::$(valtype(v))))
    end

    body = Expr(:block)
    assigned = Set(canon_id(g, v.id) for v in p.have)
    # Invariant recipes: emit ONCE, above the loop. This is the hoist.
    for (k, r) in enumerate(p.recipes)
        recipe_is_batched[k] && continue
        out = only(r.outputs)
        cid = canon_id(g, out.id)
        cid in assigned && continue
        push!(assigned, cid)
        push!(body.args, Expr(:(=), nm(out), callexpr(k, r)))
    end

    acc = gensym(:acc)
    push!(body.args, :($acc = zero($(valtype(want)))))

    # The fused per-element loop: batched recipes only, the want accumulated.
    loopbody = Expr(:block)
    loop_assigned = copy(assigned)
    for (k, r) in enumerate(p.recipes)
        recipe_is_batched[k] || continue
        out = only(r.outputs)
        cid = canon_id(g, out.id)
        cid in loop_assigned && continue
        push!(loop_assigned, cid)
        push!(loopbody.args, Expr(:(=), nm(out), callexpr(k, r)))
    end
    push!(loopbody.args, :($acc = $reduce($acc, $(nm(want)))))
    push!(body.args, Expr(:for, :($idx = eachindex($(nm(iter_port)))), loopbody))
    push!(body.args, Expr(:return, acc))
    Expr(:function, Expr(:tuple, argexprs...), body)
end

"""
    transform(ast, passes...) -> Expr

Apply zero or more AST passes (each an `Expr -> Expr` function) in order. This
is the extension point for simplification, mutation/bufferization, or
backend-specific rewrites; it must not change planning semantics (gist §9).
"""
transform(ast::Expr, passes...) = foldl((a, pass) -> pass(a), passes; init = ast)

"""
    compile(ast::Expr) -> callable

Compile a lowered `Expr` into a native Julia function via
`RuntimeGeneratedFunctions`. The returned callable takes `(__ops__, args...)`.
"""
compile(ast::Expr) = @RuntimeGeneratedFunction(ast)

"""
    PreparedKernel

A small callable object holding the RGF-generated function, the positional
`ops` tuple, and metadata (graph values in call/return order, the plan, and the
lowered AST). Runtime invocation does not consult any planning logic.
"""
struct PreparedKernel{F,O,IN,OUT}
    f::F
    ops::O
    inputs::IN
    outputs::OUT
    plan::Plan
    ast::Expr
end

@inline function (k::PreparedKernel)(args...)
    length(args) == length(k.inputs) || throw(MethodError(k, args))
    k.f(k.ops, args...)
end

function _prepare(p::Plan, ast::Expr)
    f = compile(ast)
    ops = ntuple(i -> p.recipes[i].op, length(p.recipes))
    PreparedKernel(f, ops, Tuple(p.have), Tuple(p.want), p, ast)
end

# The optional MutatingFunctions extension uses one typed cache cell per
# selected recipe. Its `nothing` value requests the allocating implementation
# on first use; later calls feed the stored value back to the extension helper.
_cache_slot(::Value{T}) where {T} = Ref{Union{Nothing,T}}(nothing)

function _rewrite_nonallocating_calls(node)
    node isa Expr || return node
    args = map(_rewrite_nonallocating_calls, node.args)
    rewritten = Expr(node.head, args...)
    if rewritten.head === :call && rewritten.args[1] isa Expr
        callee = rewritten.args[1]
        if callee.head === :ref && length(callee.args) == 2 &&
           callee.args[1] === _OPS_ARG && callee.args[2] isa Int
            cache = Expr(:ref, _CACHES_ARG, callee.args[2])
            return Expr(:call, _CACHE_APPLY_ARG, cache, callee,
                        rewritten.args[2:end]...)
        end
    end
    rewritten
end

function _nonallocating_ast(ast::Expr)
    ast.head === :function ||
        throw(ArgumentError("non-allocating preparation requires a function Expr"))
    signature = ast.args[1]
    signature isa Expr && signature.head === :tuple &&
        !isempty(signature.args) && first(signature.args) === _OPS_ARG ||
        throw(ArgumentError("non-allocating preparation requires the lowered __ops__ signature"))
    args = Expr(:tuple, _OPS_ARG, _CACHES_ARG, _CACHE_APPLY_ARG,
                signature.args[2:end]...)
    Expr(:function, args, _rewrite_nonallocating_calls(ast.args[2]))
end

"""
    NonAllocatingKernel

A stateful prepared kernel whose selected single-output recipes are invoked
through `MutatingFunctions.apply!!`. Each recipe owns a persistent typed cache:
the first call seeds it and later calls offer it back for in-place reuse.

Each cache slot retains whatever its operation returned on the first call.
Registered allocating operations such as `copy` normally seed fresh
kernel-retained storage, but aliasing operations may retain caller-owned
inputs, and a no-recipe plan returns its `have` value directly. Treat mutable
results as borrowed values that may alias inputs or be overwritten by later
calls. A kernel instance is therefore neither reentrant nor safe for concurrent
calls; prepare one instance per independent caller.
"""
struct NonAllocatingKernel{F,O,C,A,IN,OUT}
    f::F
    ops::O
    caches::C
    cache_apply::A
    inputs::IN
    outputs::OUT
    plan::Plan
    ast::Expr
end

@inline function (k::NonAllocatingKernel)(args...)
    length(args) == length(k.inputs) || throw(MethodError(k, args))
    k.f(k.ops, k.caches, k.cache_apply, args...)
end

function _prepare_nonallocating(p::Plan, ast::Expr, cache_apply)
    for r in p.recipes
        length(r.outputs) == 1 || throw(ArgumentError(
            "prepare_nonallocating requires single-output recipes; recipe $(r.id) has $(length(r.outputs)) outputs"))
    end
    f = compile(ast)
    ops = ntuple(i -> p.recipes[i].op, length(p.recipes))
    caches = ntuple(i -> _cache_slot(only(p.recipes[i].outputs)), length(p.recipes))
    NonAllocatingKernel(f, ops, caches, cache_apply, Tuple(p.have),
                        Tuple(p.want), p, ast)
end

"""
    prepare(p::Plan; passes=()) -> PreparedKernel
    prepare(g::Graph; have, want, passes=()) -> PreparedKernel

Ergonomic composition of `plan -> lower -> transform -> compile`. `passes` is a
tuple of AST passes applied before compilation.
"""
prepare(p::Plan; passes = ()) =
    _prepare(p, isempty(passes) ? lower(p) : transform(lower(p), passes...))

function prepare(g::Graph; have = (), want = (), passes = ())
    p = plan(g; have = have, want = want)
    prepare(p; passes = passes)
end

"""
    prepare_nonallocating(p::Plan; passes=()) -> NonAllocatingKernel
    prepare_nonallocating(g::Graph; have, want, passes=()) -> NonAllocatingKernel

Optional MutatingFunctions-backed preparation interface. Install and load
`MutatingFunctions` alongside `ReactiveKernels` to activate the package
extension that supplies these methods. The extension prepares the same
straight-line plan as [`prepare`](@ref), then applies a final AST transform
that routes every selected operation through `MutatingFunctions.apply!!` and a
persistent per-recipe cache. User `passes` run before this final transform.

The first invocation populates the caches and may allocate. Later invocations
reuse them when the selected operations provide allocation-free `apply!!`
methods for the runtime argument and cache types. MutatingFunctions' generic
fallback preserves semantics but may still allocate, so allocation freedom is
a property of the complete lowered operation set rather than a planner
guarantee.

Every selected recipe must have exactly one output. Each slot retains the first
object returned by its operation: that is fresh kernel-retained storage for
ordinary allocating/registered operations, but it may be a caller-owned input
for aliasing operations. A no-recipe plan returns its input directly. Treat
mutable results as borrowed values that may alias inputs or be overwritten by
the next call; a prepared instance is not reentrant or thread-safe.
"""
function prepare_nonallocating(args...; kwargs...)
    throw(ArgumentError(
        "prepare_nonallocating requires the optional MutatingFunctions extension; install MutatingFunctions and load it with `using MutatingFunctions`"))
end

"Graph values in positional call order."
inputs(k::PreparedKernel) = k.inputs
inputs(k::NonAllocatingKernel) = k.inputs
inputs(p::Plan) = Tuple(p.have)
"Graph values in return order."
outputs(k::PreparedKernel) = k.outputs
outputs(k::NonAllocatingKernel) = k.outputs
outputs(p::Plan) = Tuple(p.want)

"""
    code_expr(p) -> Expr

The generated Julia `Expr` before RGF compilation. Accepts a `Plan` or a
`PreparedKernel`. Useful for asserting that unused operations are literally
absent from the kernel (gist §20).
"""
code_expr(p::Plan) = lower(p)
code_expr(k::PreparedKernel) = k.ast
code_expr(k::NonAllocatingKernel) = k.ast

# --- explanation -----------------------------------------------------------

_opname(op) = try
    n = nameof(op)
    startswith(string(n), "#") ? string(op) : string(n)
catch
    string(op)
end

function _recipe_line(r::Recipe)
    ins = join([string(v.name) for v in r.inputs], ", ")
    outs = length(r.outputs) == 1 ? string(r.outputs[1].name) :
           "(" * join([string(v.name) for v in r.outputs], ", ") * ")"
    "$outs = $(_opname(r.op))($ins)"
end

"""
    explain(p::Plan) -> String

Human-readable account of the plan: the have/want boundary, the selected
recipes with costs, the total cost, and the backward-reachable alternatives
that were not selected (gist §16).
"""
function explain(p::Plan)
    io = IOBuffer()
    println(io, "Have:")
    println(io, "  ", isempty(p.have) ? "(none)" : join([string(v.name) for v in p.have], ", "))
    println(io, "Want:")
    println(io, "  ", join([string(v.name) for v in p.want], ", "))
    println(io, "Selected recipes:")
    if isempty(p.recipes)
        println(io, "  (none — all wanted values are already in HAVE)")
    else
        lines = [_recipe_line(r) for r in p.recipes]
        w = maximum(length, lines)
        for (r, l) in zip(p.recipes, lines)
            println(io, "  ", rpad(l, w + 2), "cost ", r.cost)
        end
    end
    selected = Set(r.id for r in p.recipes)
    unused = [r for r in p.candidates if !(r.id in selected)]
    if !isempty(unused)
        println(io, "Alternatives not selected:")
        for r in unused
            println(io, "  ", rpad(_recipe_line(r), 0), "  (cost ", r.cost, ")")
        end
    end
    print(io, "Total graph cost: ", p.cost)
    String(take!(io))
end

Base.show(io::IO, p::Plan) = print(io, explain(p))
function Base.show(io::IO, k::PreparedKernel)
    print(io, "PreparedKernel(", join([string(v.name) for v in k.inputs], ", "),
          " -> ", join([string(v.name) for v in k.outputs], ", "), ")")
end

function Base.show(io::IO, k::NonAllocatingKernel)
    print(io, "NonAllocatingKernel(", join([string(v.name) for v in k.inputs], ", "),
          " -> ", join([string(v.name) for v in k.outputs], ", "), ")")
end
