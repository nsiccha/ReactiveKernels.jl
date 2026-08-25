# A closed-world compiled reactive state. Unlike `ReactiveState`, which plans
# arbitrary demands dynamically, a `ReactiveProgram` fixes its HAVE/WANT
# boundary once and emits one lazy getter kernel per reachable state value.
# Runtime mutation therefore performs only validity propagation; runtime reads
# perform only generated validity checks and the selected recipe calls.

"""
    ReactiveValue

A state-local typed handle returned by [`statevalue`](@ref). Keep these handles in
hot code: their slot index is a type parameter, so `get!`, `set!`, and `touch!`
use literal tuple indices without dictionary lookup.
"""
struct ReactiveValue{I,T}
    token::Base.RefValue{Nothing}
    graph::Graph
    value::Value{T}
end

valtype(::ReactiveValue{I,T}) where {I,T} = T
_slot_index(::ReactiveValue{I}) where {I} = I
Base.show(io::IO, value::ReactiveValue) = print(io, "state(", value.value, ")")

struct _ReactiveGetter{F,O}
    f::F
    ops::O
    ast::Expr
end

@inline (getter::_ReactiveGetter)(slots, valid) =
    getter.f(getter.ops, slots, valid)

"""
    ReactiveProgram

A prepared, inspectable state program produced by [`prepare_reactive`](@ref).
It contains a single selected `Plan`, an initialization kernel, compiled lazy
getter kernels, and a precomputed dependency graph for invalidation.
"""
struct ReactiveProgram{H,G,V,IN,OUT}
    token::Base.RefValue{Nothing}
    graph::Graph
    graph_version::Int
    handles::H
    getters::G
    values::V
    inputs::IN
    outputs::OUT
    index::Dict{Int,Int}
    dependents::Vector{Vector{Int}}
    sources::BitVector
    plan::Plan
end

"""
    CompiledReactiveState

An instance of a [`ReactiveProgram`](@ref). Values live in typed `Ref` slots;
`valid` and `frozen` are runtime state, while all recipe selection and lowering
belong to the shared program. Instances are mutable and not thread-safe.
"""
mutable struct CompiledReactiveState{P,S}
    program::P
    slots::S
    valid::BitVector
    frozen::BitVector
    stack::Vector{Int}
end

function _owned_output(plan::Plan, recipe::Recipe, id::Int)
    any(value -> canon_id(plan.graph, value.id) == id, plan.have) && return false
    owner = get(plan.producer, id, nothing)
    owner !== nothing && owner.id == recipe.id
end

function _state_values(plan::Plan)
    graph = plan.graph
    result = Value[]
    seen = Set{Int}()
    for value in plan.have
        id = canon_id(graph, value.id)
        id in seen && continue
        push!(seen, id)
        push!(result, graph.values[id])
    end
    for recipe in plan.recipes, output in recipe.outputs
        id = canon_id(graph, output.id)
        _owned_output(plan, recipe, id) || continue
        id in seen && continue
        push!(seen, id)
        push!(result, graph.values[id])
    end
    result
end

function _state_handles(token, graph::Graph, values)
    Tuple(begin
        T = valtype(value)
        ReactiveValue{i,T}(token, graph, value)
    end for (i, value) in enumerate(values))
end

function _state_dependencies(plan::Plan, index)
    graph = plan.graph
    dependents = [Int[] for _ in 1:length(index)]
    for recipe in plan.recipes, output in recipe.outputs
        output_id = canon_id(graph, output.id)
        _owned_output(plan, recipe, output_id) || continue
        output_index = index[output_id]
        for input in recipe.inputs
            input_index = index[canon_id(graph, input.id)]
            output_index in dependents[input_index] ||
                push!(dependents[input_index], output_index)
        end
    end
    dependents
end

function _ensure_expr(plan::Plan, index, recipe_index, id::Int)
    graph = plan.graph
    slot_index = index[id]
    recipe = get(plan.producer, id, nothing)
    if recipe === nothing
        return quote
            __valid__[$slot_index] || error(
                "compiled reactive source slot $slot_index is invalid",
            )
        end
    end

    body = Expr(:block)
    for input in recipe.inputs
        input_id = canon_id(graph, input.id)
        push!(body.args, _ensure_expr(plan, index, recipe_index, input_id))
    end
    args = [:(__slots__[$(index[canon_id(graph, input.id)])][]) for input in recipe.inputs]
    call = :(__ops__[$(recipe_index[recipe.id])]($(args...)))

    multiple = length(recipe.outputs) != 1
    results = if multiple
        names = [gensym(:recipe_result) for _ in recipe.outputs]
        push!(body.args, Expr(:(=), Expr(:tuple, names...), call))
        names
    else
        name = gensym(:recipe_result)
        push!(body.args, :($name = $call))
        (name,)
    end
    for (position, output) in enumerate(recipe.outputs)
        output_id = canon_id(graph, output.id)
        _owned_output(plan, recipe, output_id) || continue
        output_index = index[output_id]
        value_expr = results[position]
        push!(body.args, quote
            if !__valid__[$output_index]
                __slots__[$output_index][] = $value_expr
                __valid__[$output_index] = true
            end
        end)
    end

    quote
        if !__valid__[$slot_index]
            $body
        end
    end
end

function _getter_ast(plan::Plan, index, recipe_index, id::Int)
    slot_index = index[id]
    body = Expr(:block)
    push!(body.args, _ensure_expr(plan, index, recipe_index, id))
    push!(body.args, :(return __slots__[$slot_index][]))
    Expr(
        :function,
        Expr(:tuple, :__ops__, :__slots__, :__valid__),
        body,
    )
end

function _prepare_getter(plan::Plan, index, recipe_index, id::Int)
    ast = _getter_ast(plan, index, recipe_index, id)
    ops = Tuple(recipe.op for recipe in plan.recipes)
    _ReactiveGetter(compile(ast), ops, ast)
end

"""
    prepare_reactive(graph; have, want) -> ReactiveProgram

Prepare a closed-world stateful program. `have` declares the mutable,
authoritative state boundary; `want` declares all downstream values the state
may demand. Planning happens once. Each reachable value receives a compiled
lazy getter that reuses valid slots and recomputes only invalid dependencies.

Instantiate with `state = program(have_values...)`, then obtain typed handles
once with `x = statevalue(program, graph_value)`. Mutate a source with `set!`,
or mutate its stored object in a [`mutate!`](@ref) transaction. Both invalidate
downstream state; `get!` performs the generated minimal recomputation.

The selected graph and runtime slot types are fixed. This is the efficient
state-machine counterpart to the open-ended, dictionary-backed
[`ReactiveState`](@ref).
"""
function prepare_reactive(graph::Graph; have = (), want = ())
    selected = plan(graph; have = have, want = want)
    values = _state_values(selected)
    index = Dict(canon_id(graph, value.id) => i for (i, value) in enumerate(values))
    token = Ref{Nothing}(nothing)
    handles = _state_handles(token, graph, values)
    recipe_index = Dict(recipe.id => i for (i, recipe) in enumerate(selected.recipes))
    getters = Tuple(
        _prepare_getter(selected, index, recipe_index, canon_id(graph, value.id))
        for value in values
    )

    dependents = _state_dependencies(selected, index)
    source_ids = Set(canon_id(graph, value.id) for value in selected.have)
    sources = BitVector(canon_id(graph, value.id) in source_ids for value in values)
    ReactiveProgram(
        token,
        graph,
        graph.version,
        handles,
        getters,
        Tuple(values),
        Tuple(selected.have),
        Tuple(selected.want),
        index,
        dependents,
        sources,
        selected,
    )
end

"""
    prepare_reactive(spec::KernelSpec; have, want) -> ReactiveProgram

Prepare an authoring [`KernelSpec`] through the same compiled reactive state
engine as a hand-built graph. Symbol and `Value` boundary overrides are
resolved against the spec before delegating, so the authoring surface does not
introduce a second runtime or a sampler-specific refresh path.
"""
function prepare_reactive(spec::KernelSpec;
                          have = _KERNEL_DEFAULT_BOUNDARY,
                          want = _KERNEL_DEFAULT_BOUNDARY,
                          kwargs...)
    resolved_have = _kernel_selection(spec, have, spec.have_names, :have)
    resolved_want = _kernel_selection(spec, want, spec.want_names, :want)
    prepare_reactive(spec.graph;
                     have = resolved_have,
                     want = resolved_want,
                     kwargs...)
end

@generated function _state_slots(handles::H, args::A) where {H<:Tuple,A<:Tuple}
    handle_types = H.parameters
    arg_count = length(A.parameters)
    body = Expr(:tuple)
    for (index, handle_type) in enumerate(handle_types)
        value_type = handle_type.parameters[2]
        if index <= arg_count
            push!(body.args, :(Ref{$value_type}(args[$index])))
        else
            push!(body.args, :(Ref{$value_type}()))
        end
    end
    body
end

function (program::ReactiveProgram)(args...; frozen = nothing)
    program.graph.version == program.graph_version || throw(ArgumentError(
        "graph changed after prepare_reactive; prepare a new ReactiveProgram",
    ))
    length(args) == length(program.inputs) || throw(MethodError(program, args))
    slots = _state_slots(program.handles, args)
    count = length(slots)
    source_count = length(program.inputs)
    state = CompiledReactiveState(
        program,
        slots,
        BitVector(index <= source_count for index in 1:count),
        falses(count),
        Vector{Int}(undef, count),
    )
    if frozen !== nothing
        for (id, value) in frozen
            handle = statevalue(program, program.graph.values[canon_id(program.graph, id)])
            freeze!(state, handle, value)
        end
    end
    state
end

"Return the typed, literal-index state handle for a graph `Value`."
function statevalue(program::ReactiveProgram, value::Value)
    program.graph.version == program.graph_version || throw(ArgumentError(
        "graph changed after prepare_reactive; prepare a new ReactiveProgram",
    ))
    id = canon_id(program.graph, value.id)
    index = get(program.index, id, 0)
    index == 0 && throw(ArgumentError(
        "value $(value.name) is outside this ReactiveProgram",
    ))
    program.handles[index]
end

statevalue(state::CompiledReactiveState, value::Value) =
    statevalue(state.program, value)

@inline _tuple_slot(tuple, ::Val{I}) where {I} = getfield(tuple, I)

@inline function _check_state_version(state::CompiledReactiveState)
    state.program.graph.version == state.program.graph_version || throw(ArgumentError(
        "graph changed after prepare_reactive; prepare a new ReactiveProgram",
    ))
    nothing
end

function _check_program_handle(program::ReactiveProgram,
                               handle::ReactiveValue{I}) where {I}
    program.graph.version == program.graph_version || throw(ArgumentError(
        "graph changed after prepare_reactive; prepare a new ReactiveProgram",
    ))
    handle.token === program.token || throw(ArgumentError(
        "ReactiveValue belongs to a different program",
    ))
    handle.graph === program.graph || throw(ArgumentError(
        "ReactiveValue belongs to a different graph",
    ))
    1 <= I <= length(program.handles) || throw(ArgumentError(
        "ReactiveValue slot is outside this program",
    ))
    expected = _tuple_slot(program.handles, Val(I))
    handle.value.id == expected.value.id || throw(ArgumentError(
        "ReactiveValue belongs to a different program",
    ))
    nothing
end

function _check_handle(state::CompiledReactiveState, handle::ReactiveValue)
    _check_program_handle(state.program, handle)
end

@inline function Base.get!(state::CompiledReactiveState,
                           handle::ReactiveValue{I,T}) where {I,T}
    _check_handle(state, handle)
    getter = _tuple_slot(state.program.getters, Val(I))
    getter(state.slots, state.valid)::T
end

Base.get!(state::CompiledReactiveState, value::Value) =
    get!(state, statevalue(state, value))

function _invalidate_dependents!(state::CompiledReactiveState, root::Int)
    top = 0
    for child in state.program.dependents[root]
        state.frozen[child] && continue
        state.valid[child] || continue
        state.valid[child] = false
        top += 1
        state.stack[top] = child
    end
    while top > 0
        current = state.stack[top]
        top -= 1
        for child in state.program.dependents[current]
            state.frozen[child] && continue
            state.valid[child] || continue
            state.valid[child] = false
            top += 1
            state.stack[top] = child
        end
    end
    state
end

"""Mark an in-place mutation of a declared HAVE value and invalidate downstream slots."""
function touch!(state::CompiledReactiveState, handle::ReactiveValue{I}) where {I}
    _check_handle(state, handle)
    state.program.sources[I] || throw(ArgumentError(
        "touch! is restricted to the ReactiveProgram HAVE boundary; use freeze! for a derived cut point",
    ))
    state.valid[I] = true
    _invalidate_dependents!(state, I)
end

touch!(state::CompiledReactiveState, value::Value) =
    touch!(state, statevalue(state, value))

"""
    mutate!(state, source) do value
        ...
    end

Mutate the stored object for a declared HAVE `source`, then automatically mark
it changed and invalidate its downstream slots. Invalidation also runs if the
mutation throws, because the object may already have been modified.
"""
function mutate!(f, state::CompiledReactiveState, handle::ReactiveValue)
    _check_handle(state, handle)
    state.program.sources[_slot_index(handle)] || throw(ArgumentError(
        "mutate! is restricted to the ReactiveProgram HAVE boundary; use freeze! for a derived cut point",
    ))
    value = get!(state, handle)
    try
        f(value)
    finally
        touch!(state, handle)
    end
end

mutate!(f, state::CompiledReactiveState, value::Value) =
    mutate!(f, state, statevalue(state, value))

"Set a declared HAVE value and invalidate its downstream slots."
function set!(state::CompiledReactiveState, handle::ReactiveValue{I}, value) where {I}
    _check_handle(state, handle)
    state.program.sources[I] || throw(ArgumentError(
        "set! is restricted to the ReactiveProgram HAVE boundary; use freeze! for a derived cut point",
    ))
    _tuple_slot(state.slots, Val(I))[] = value
    state.valid[I] = true
    _invalidate_dependents!(state, I)
end

set!(state::CompiledReactiveState, graph_value::Value, value) =
    set!(state, statevalue(state, graph_value), value)

"Freeze the current value of a derived slot as an authoritative cut point."
function freeze!(state::CompiledReactiveState, handle::ReactiveValue{I}) where {I}
    _check_handle(state, handle)
    state.program.sources[I] && throw(ArgumentError(
        "ReactiveProgram HAVE values are already authoritative; use set!",
    ))
    get!(state, handle)
    state.frozen[I] = true
    state
end

"Install and freeze a derived slot, invalidating only its downstream consumers."
function freeze!(state::CompiledReactiveState, handle::ReactiveValue{I}, value) where {I}
    _check_handle(state, handle)
    state.program.sources[I] && throw(ArgumentError(
        "ReactiveProgram HAVE values are already authoritative; use set!",
    ))
    _invalidate_dependents!(state, I)
    _tuple_slot(state.slots, Val(I))[] = value
    state.valid[I] = true
    state.frozen[I] = true
    state
end

freeze!(state::CompiledReactiveState, value::Value) =
    freeze!(state, statevalue(state, value))
freeze!(state::CompiledReactiveState, graph_value::Value, value) =
    freeze!(state, statevalue(state, graph_value), value)

"Remove a compiled-state frozen cut point and recompute it lazily on demand."
function unfreeze!(state::CompiledReactiveState, handle::ReactiveValue{I}) where {I}
    _check_handle(state, handle)
    state.program.sources[I] && throw(ArgumentError(
        "cannot unfreeze the fixed ReactiveProgram HAVE boundary",
    ))
    state.frozen[I] = false
    state.valid[I] = false
    _invalidate_dependents!(state, I)
end

unfreeze!(state::CompiledReactiveState, value::Value) =
    unfreeze!(state, statevalue(state, value))

function _handle_tuple(value::ReactiveValue)
    (value,)
end
_handle_tuple(values::Tuple) = values
_handle_tuple(values::AbstractVector) = Tuple(values)

"Checkpoint valid derived values for a later `program(...; frozen=cp)` instance."
function checkpoint(state::CompiledReactiveState, values)
    result = Dict{Int,Any}()
    for value in _handle_tuple(values)
        handle = value isa ReactiveValue ? value : statevalue(state, value)
        _check_handle(state, handle)
        state.program.sources[_slot_index(handle)] && throw(ArgumentError(
            "checkpoint stores derived cut points; pass HAVE values to the new program instance",
        ))
        result[canon_id(state.program.graph, handle.value.id)] = get!(state, handle)
    end
    result
end

_copy_slot_value(value::AbstractArray) = copy(value)
_copy_slot_value(value) = value

function _copy_slot_value!(destination::AbstractArray, source::AbstractArray)
    axes(destination) == axes(source) || return copy(source)
    copyto!(destination, source)
    destination
end
_copy_slot_value!(destination, source) = source

@generated function _copy_slots(slots::S, valid::BitVector) where {S<:Tuple}
    body = Expr(:tuple)
    for (index, slot_type) in enumerate(S.parameters)
        value_type = slot_type.parameters[1]
        push!(body.args, quote
            if valid[$index]
                Ref{$value_type}(_copy_slot_value(slots[$index][]))
            else
                Ref{$value_type}()
            end
        end)
    end
    body
end

@generated function _copy_slots!(destination::D, source::S,
                                 destination_valid::BitVector,
                                 source_valid::BitVector) where {D<:Tuple,S<:Tuple}
    length(D.parameters) == length(S.parameters) ||
        return :(throw(DimensionMismatch("compiled state slot counts differ")))
    body = Expr(:block)
    for index in 1:length(D.parameters)
        push!(body.args, quote
            if source_valid[$index]
                destination[$index][] = if destination_valid[$index]
                    _copy_slot_value!(destination[$index][], source[$index][])
                else
                    _copy_slot_value(source[$index][])
                end
            end
        end)
    end
    push!(body.args, :(destination))
    body
end

function Base.copy(state::CompiledReactiveState)
    _check_state_version(state)
    slots = _copy_slots(state.slots, state.valid)
    CompiledReactiveState(
        state.program,
        slots,
        copy(state.valid),
        copy(state.frozen),
        similar(state.stack),
    )
end

function Base.copyto!(destination::CompiledReactiveState,
                      source::CompiledReactiveState)
    _check_state_version(destination)
    _check_state_version(source)
    destination.program === source.program || throw(ArgumentError(
        "compiled states belong to different ReactivePrograms",
    ))
    _copy_slots!(destination.slots, source.slots,
                 destination.valid, source.valid)
    copyto!(destination.valid, source.valid)
    copyto!(destination.frozen, source.frozen)
    destination
end

inputs(program::ReactiveProgram) = program.inputs
outputs(program::ReactiveProgram) = program.outputs
function code_expr(program::ReactiveProgram, handle::ReactiveValue{I}) where {I}
    _check_program_handle(program, handle)
    _tuple_slot(program.getters, Val(I)).ast
end
code_expr(program::ReactiveProgram, value::Value) =
    code_expr(program, statevalue(program, value))

function Base.show(io::IO, program::ReactiveProgram)
    print(
        io,
        "ReactiveProgram(",
        join((string(value.name) for value in program.inputs), ", "),
        " -> ",
        join((string(value.name) for value in program.outputs), ", "),
        ")",
    )
end
