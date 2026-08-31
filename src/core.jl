# Core graph identities: Value, Recipe, Graph.
#
# These objects are *compile/planning-time metadata only*. None of them are
# consulted inside a prepared kernel (see codegen.jl); the hot path sees only
# ordinary Julia values.

# A process-global counter giving every `Value` a stable identity independent of
# its name or of any particular graph. Identity must not depend solely on the
# name (gist §5), so two values may share a name yet remain distinct.
const _VALUE_COUNTER = Ref(0)
_next_value_id() = (_VALUE_COUNTER[] += 1)

"""
    Value{T}

A stable graph identity for a runtime value of Julia type `T`. `id` is the
identity (globally unique); `name` exists only for diagnostics and for
generated-code readability. Values are immutable and cheap to hash/compare.
"""
struct Value{T}
    id::Int
    name::Symbol
end

Value(name::Symbol, ::Type{T}) where {T} = Value{T}(_next_value_id(), name)

"""
    value(name, T)

Construct a standalone `Value{T}` with a fresh global identity. Use `value!` to
also register it into a graph.
"""
value(name::Symbol, ::Type{T}) where {T} = Value(name, T)

"The declared Julia runtime type of a value."
valtype(::Value{T}) where {T} = T

Base.:(==)(a::Value, b::Value) = a.id == b.id
Base.hash(v::Value, h::UInt) = hash(v.id, hash(:ReactiveKernelsValue, h))
Base.show(io::IO, v::Value{T}) where {T} = print(io, v.name, "::", T)

"""
    Recipe

A pure computation mapping input graph values to one or more output graph
values via `op`. RK does not inspect `op` to prove purity: registering an
ordinary recipe asserts this contract. Set `effectful=true` when the operation
is known not to satisfy it; effectful operations are rejected by the stateless
planner and therefore cannot enter a prepared kernel or plate. `cost` is a
deterministic planning hint (not measured runtime). `cse_key`, when
non-`nothing`, opts the operation into structural CSE (gist §8).
`source` is optional authored-RHS metadata for cold-path readable rendering; it
is kept on the planning recipe rather than the executable operation so it never
enters prepared hot-state tuples.
"""
struct _NoKernelSource end
const _NO_KERNEL_SOURCE = _NoKernelSource()

struct Recipe
    id::Int
    inputs::Tuple{Vararg{Value}}
    outputs::Tuple{Vararg{Value}}
    op::Any
    cost::Float64
    cse_key::Any
    effectful::Bool
    source::Any
end
Recipe(id, inputs, outputs, op, cost, cse_key, effectful) =
    Recipe(id, inputs, outputs, op, cost, cse_key, effectful, _NO_KERNEL_SOURCE)

"""
    _KernelSourceOp{DefToken,Form,F,TF}

An immutable wrapper marking a recipe operation SYNTHESIZED from captured `@kernel` source as
COMPILER-OWNED provenance (RK 07:21). Authoring wraps ONLY the anonymous-closure path of
`_kernel_operation` in this; a bare exact identity (`cholesky`/`+`/…) stays raw and is identity/domain
validated. `DefToken` is a definition-unique gensym baked in at graph build — NOT a security boundary (an internal
constructor/type parameter cannot prevent deliberate internal misuse); it is trusted only because the
supported authoring path is the ONLY thing that wraps a closure, so an arbitrary public Graph closure is
never auto-wrapped. It makes each fused op a distinct concrete type (survives the prepared ops-tuple,
carries no mutable registry). `Form` (RK 07:24)
distinguishes a `:portcall` — a call THROUGH A PORT, `callable(args…)`, whose first input is the callable
source and the rest are ordered args — from a general `:fused` expression, so a prepared handle can
self-derive the DESTINATION contract (a port-call with one owned buffer + one owned scalar output →
`f(dest, args…)::scalar`) from source SHAPE + typed slot roles, never from a name/Recipe id/inspection.
The call forwards INLINE. A RAW anonymous closure inserted into a Graph carries no wrapper and is
rejected as opaque when captured into a prepared handle.
"""
struct _KernelSourceOp{DefToken,Form,F,TF}
    f::F
    tensor_f::TF
end
_KernelSourceOp(::Val{DefToken}, ::Val{Form}, f::F, tensor_f::TF) where
        {DefToken,Form,F,TF} =
    _KernelSourceOp{DefToken,Form,F,TF}(f, tensor_f)
# Preserve the established internal constructor for compiler fixtures and
# already-authored handles; without an alternate body it uses the same callable
# in both modes.
_KernelSourceOp(token::Val, form::Val, f) = _KernelSourceOp(token, form, f, f)

# Optional tracing extensions classify their scalar/array argument types as
# tensorized.  The tuple fold is ordinary Julia dispatch over argument types,
# so it is resolved while tracing rather than becoming data-dependent control
# flow in the compiled program.
@inline _kernel_source_arg_style(arg) = Val(:native)
@inline _kernel_source_merge(::Val{:tensorized}, style) = Val(:tensorized)
@inline _kernel_source_merge(::Val{:native}, style) = style
@inline _kernel_source_style(::Tuple{}) = Val(:native)
@inline function _kernel_source_style(args::Tuple)
    _kernel_source_merge(
        _kernel_source_arg_style(first(args)),
        _kernel_source_style(Base.tail(args)),
    )
end
@inline _kernel_source_call(::Val{:native}, op::_KernelSourceOp, args) =
    op.f(args...)
@inline _kernel_source_call(::Val{:tensorized}, op::_KernelSourceOp, args) =
    op.tensor_f(args...)
@inline (op::_KernelSourceOp)(args...) =
    _kernel_source_call(_kernel_source_style(args), op, args)
kernel_sourceop_token(::_KernelSourceOp{DefToken}) where {DefToken} = DefToken
kernel_sourceop_form(::_KernelSourceOp{DefToken,Form}) where {DefToken,Form} = Form

"""
    Graph()

A mutable builder collecting `Value`s and `Recipe`s plus the producer index the
planner needs. Building executes nothing.
"""
mutable struct Graph
    values::Dict{Int,Value}          # id => Value
    recipes::Vector{Recipe}
    producers::Dict{Int,Vector{Int}} # canonical value id => indices into `recipes`
    aliases::Dict{Int,Int}           # value id => structurally-equal canonical id
    version::Int
end
Graph() = Graph(Dict{Int,Value}(), Recipe[], Dict{Int,Vector{Int}}(),
                Dict{Int,Int}(), 0)

_register!(g::Graph, v::Value) = (g.values[v.id] = v; v)

"""
    canon_id(g, id) -> Int

Resolve a value id to its structural-CSE canonical representative (gist §8).
Absent any structural CSE this is the identity.
"""
canon_id(g::Graph, id::Int) = haskey(g.aliases, id) ? canon_id(g, g.aliases[id]) : id

"""
    value!(g, name, T) -> Value{T}

Create a `Value{T}` and register it into graph `g`.
"""
function value!(g::Graph, name::Symbol, ::Type{T}) where {T}
    v = Value(name, T)
    _register!(g, v)
    g.version += 1
    v
end

_astuple(v::Value) = (v,)
_astuple(t::Tuple) = t
_astuple(v::AbstractVector) = Tuple(v)

function _cse_alias_plan(g::Graph, new_outputs::Tuple, old_outputs::Tuple, cse_key)
    targets = Dict{Int,Int}()
    edges = Pair{Int,Int}[]

    for (position, (new_output, old_output)) in enumerate(zip(new_outputs, old_outputs))
        new_type = valtype(new_output)
        old_type = valtype(old_output)
        new_type === old_type || throw(ArgumentError(
            "structural CSE output type mismatch for key $(repr(cse_key)) at position " *
            "$position: existing output $(old_output.name) has type $old_type, " *
            "new output $(new_output.name) has type $new_type"))

        source = canon_id(g, new_output.id)
        target = canon_id(g, old_output.id)
        source == target && continue
        if haskey(targets, source)
            targets[source] == target || throw(ArgumentError(
                "conflicting structural CSE output mapping for key $(repr(cse_key)) " *
                "at position $position: canonical value $source would map to both " *
                "$(targets[source]) and $target"))
            continue
        end
        targets[source] = target
        push!(edges, source => target)
    end

    # Validate the complete mapping before mutating the graph. In particular,
    # crossed multi-output mappings such as (a, b) => (b, a) must not create a
    # recursive alias chain.
    for start in sort!(collect(keys(targets)))
        seen = Set{Int}()
        current = start
        while haskey(targets, current)
            current in seen && throw(ArgumentError(
                "cyclic structural CSE output mapping for key $(repr(cse_key))"))
            push!(seen, current)
            current = targets[current]
        end
    end
    edges
end

function _reindex_producers!(g::Graph)
    empty!(g.producers)
    for recipe in g.recipes
        indexed = Set{Int}()
        for output in recipe.outputs
            canonical = canon_id(g, output.id)
            canonical in indexed && continue
            push!(get!(g.producers, canonical, Int[]), recipe.id)
            push!(indexed, canonical)
        end
    end
    g
end

"""
    add!(g, inputs => outputs, op; cost=1.0, cse_key=nothing, effectful=false)
    add!(g; inputs, outputs, op, cost=1.0, cse_key=nothing, effectful=false)

Register a recipe `(inputs...) --op--> (outputs...)`. `inputs`/`outputs` may be
a single `Value` or a tuple of `Value`s. Referenced values are auto-registered.
Returns the `Recipe`.
"""
function add!(g::Graph; inputs, outputs, op,
              cost::Real = 1.0, cse_key = nothing, effectful::Bool = false,
              source = _NO_KERNEL_SOURCE)
    ins = _astuple(inputs)
    outs = _astuple(outputs)
    recipe_cost = Float64(cost)
    if !isfinite(recipe_cost) || recipe_cost < 0
        throw(ArgumentError("recipe cost must be finite and non-negative, got $cost"))
    end
    # Opt-in structural CSE (gist §8): if a prior recipe carries the same
    # non-`nothing` cse_key, the same canonical inputs, and the same output
    # arity, it computes the same thing. Alias the new outputs onto the existing
    # producer's outputs instead of adding a duplicate recipe.
    if cse_key !== nothing && !effectful
        canon_ins = Tuple(canon_id(g, v.id) for v in ins)
        for r in g.recipes
            r.effectful && continue
            r.cse_key === nothing && continue
            isequal(r.cse_key, cse_key) || continue
            length(r.outputs) == length(outs) || continue
            Tuple(canon_id(g, v.id) for v in r.inputs) == canon_ins || continue
            alias_plan = _cse_alias_plan(g, outs, r.outputs, cse_key)
            for v in ins; _register!(g, v); end
            for v in outs; _register!(g, v); end
            for (source, target) in alias_plan
                g.aliases[source] = target
            end
            isempty(alias_plan) || _reindex_producers!(g)
            g.version += 1
            return r
        end
    end

    for v in ins; _register!(g, v); end
    for v in outs; _register!(g, v); end
    r = Recipe(length(g.recipes) + 1, ins, outs, op, recipe_cost, cse_key,
               effectful, source)
    push!(g.recipes, r)
    for v in outs
        push!(get!(g.producers, canon_id(g, v.id), Int[]), r.id)
    end
    g.version += 1
    r
end

add!(g::Graph, pair::Pair, op; kwargs...) =
    add!(g; inputs = pair.first, outputs = pair.second, op = op, kwargs...)

"All recipes that can produce value id `vid` (resolved through structural CSE)."
producers_of(g::Graph, vid::Int) = get(g.producers, canon_id(g, vid), Int[])
