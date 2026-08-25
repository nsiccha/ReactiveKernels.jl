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
values via `op`. `cost` is a deterministic planning hint (not measured
runtime). `cse_key`, when non-`nothing`, opts the operation into structural CSE
(gist §8). `effectful` operations are rejected by the planner (gist §12).
"""
struct Recipe
    id::Int
    inputs::Tuple{Vararg{Value}}
    outputs::Tuple{Vararg{Value}}
    op::Any
    cost::Float64
    cse_key::Any
    effectful::Bool
end

"""
    Graph()

A mutable builder collecting `Value`s and `Recipe`s plus the producer index the
planner needs. Building executes nothing.
"""
mutable struct Graph
    values::Dict{Int,Value}          # id => Value
    recipes::Vector{Recipe}
    producers::Dict{Int,Vector{Int}} # value id => indices into `recipes`
    version::Int
end
Graph() = Graph(Dict{Int,Value}(), Recipe[], Dict{Int,Vector{Int}}(), 0)

_register!(g::Graph, v::Value) = (g.values[v.id] = v; v)

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

"""
    add!(g, inputs => outputs, op; cost=1.0, cse_key=nothing, effectful=false)
    add!(g; inputs, outputs, op, cost=1.0, cse_key=nothing, effectful=false)

Register a recipe `(inputs...) --op--> (outputs...)`. `inputs`/`outputs` may be
a single `Value` or a tuple of `Value`s. Referenced values are auto-registered.
Returns the `Recipe`.
"""
function add!(g::Graph; inputs, outputs, op,
              cost::Real = 1.0, cse_key = nothing, effectful::Bool = false)
    ins = _astuple(inputs)
    outs = _astuple(outputs)
    for v in ins; _register!(g, v); end
    for v in outs; _register!(g, v); end
    r = Recipe(length(g.recipes) + 1, ins, outs, op, Float64(cost), cse_key, effectful)
    push!(g.recipes, r)
    for v in outs
        push!(get!(g.producers, v.id, Int[]), r.id)
    end
    g.version += 1
    r
end

add!(g::Graph, pair::Pair, op; kwargs...) =
    add!(g; inputs = pair.first, outputs = pair.second, op = op, kwargs...)

"All recipes that can produce value id `vid`."
producers_of(g::Graph, vid::Int) = get(g.producers, vid, Int[])
