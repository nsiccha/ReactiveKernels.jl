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

# In-place variant produced by `prepare_reactive_nonallocating`. It carries an
# extra injected `cache_apply` callable, passed positionally to the generated
# getter alongside the pure `ops` tuple. The generated body routes selected
# single-output recipes through `cache_apply` so a stale slot buffer is reused
# in place instead of allocating a fresh recipe return. `get!` calls both getter
# variants through the same `(slots, valid)` interface.
struct _ReactiveGetterInPlace{F,O,A}
    f::F
    ops::O
    cache_apply::A
    ast::Expr
end

@inline (getter::_ReactiveGetterInPlace)(slots, valid) =
    getter.f(getter.ops, getter.cache_apply, slots, valid)

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

# `mutating_slots` holds the slot indices whose (single-output, owned) producer
# recipe was selected for in-place evaluation. It is always empty for the pure
# `prepare_reactive` program, so that program's AST is byte-identical to before
# this hook existed. For a slot in the set, the store site is routed through the
# injected `__cache_apply__` helper (see `_ReactiveGetterInPlace`).
function _ensure_expr(plan::Plan, index, recipe_index, id::Int,
                      mutating_slots::Set{Int} = Set{Int}())
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
        push!(body.args, _ensure_expr(plan, index, recipe_index, input_id, mutating_slots))
    end
    args = [:(__slots__[$(index[canon_id(graph, input.id)])][]) for input in recipe.inputs]
    k = recipe_index[recipe.id]

    if length(recipe.outputs) == 1 && slot_index in mutating_slots
        # In-place single-output store (`slot_index == output_index` here). The
        # slot IS the per-instance cache: on the first touch the slot Ref is
        # undefined (`_state_slots` seeds derived slots with `Ref{T}()`), so seed
        # it with the ordinary allocating op; on every later (post-invalidation)
        # evaluation reuse the surviving buffer through `__cache_apply__`.
        # `_invalidate_dependents!` only flips validity and never clears a slot,
        # so an assigned slot stays assigned across invalidate→recompute cycles.
        # The helper's RETURN is stored unconditionally: `apply!!`-style helpers
        # mutate-and-return the cache, or return a fresh object on resize / an
        # immutable / a shape change, and either must land back in the slot.
        push!(body.args, quote
            if !__valid__[$slot_index]
                __slots__[$slot_index][] = isassigned(__slots__[$slot_index]) ?
                    __cache_apply__(__slots__[$slot_index][], __ops__[$k], $(args...)) :
                    __ops__[$k]($(args...))
                __valid__[$slot_index] = true
            end
        end)
    else
        # Pure path: straight-line allocate-and-store with identical operation order
        # and behavior (the emitted AST is no longer byte-identical to the pre-fix
        # one — the result-local NAMES change deterministically, see below).
        # Multi-output recipes always take this branch (single-output-only hook).
        # Result locals are named DETERMINISTICALLY from the stable recipe index `k`
        # (and the output position), NOT a fresh gensym: a gensym differs on every
        # getter build, so two structurally identical programs (e.g. two
        # `specialize=true` constructions of the same signature) produced distinct
        # RuntimeGeneratedFunction ASTs -> distinct getter/program TYPES -> a full
        # recompile per construction. A stable name makes the ASTs identical, so the
        # RGF/type is reused. `k` is unique per recipe (recipe_index is a bijection),
        # and each recipe is emitted under its own `if !__valid__[slot]` guard, so
        # reusing the name across a diamond re-emission is a guarded reassignment of
        # the same value — semantically identical to the old distinct gensyms.
        call = :(__ops__[$k]($(args...)))
        multiple = length(recipe.outputs) != 1
        results = if multiple
            names = [Symbol("__recipe_result_", k, "_", position)
                     for position in 1:length(recipe.outputs)]
            push!(body.args, Expr(:(=), Expr(:tuple, names...), call))
            names
        else
            name = Symbol("__recipe_result_", k)
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
    end

    quote
        if !__valid__[$slot_index]
            $body
        end
    end
end

function _getter_ast(plan::Plan, index, recipe_index, id::Int;
                     in_place::Bool = false,
                     mutating_slots::Set{Int} = Set{Int}())
    slot_index = index[id]
    body = Expr(:block)
    push!(body.args, _ensure_expr(plan, index, recipe_index, id, mutating_slots))
    push!(body.args, :(return __slots__[$slot_index][]))
    signature = in_place ?
        Expr(:tuple, :__ops__, :__cache_apply__, :__slots__, :__valid__) :
        Expr(:tuple, :__ops__, :__slots__, :__valid__)
    Expr(:function, signature, body)
end

function _prepare_getter(plan::Plan, index, recipe_index, id::Int;
                         cache_apply = nothing,
                         mutating_slots::Set{Int} = Set{Int}())
    in_place = cache_apply !== nothing
    ast = _getter_ast(plan, index, recipe_index, id;
                      in_place = in_place, mutating_slots = mutating_slots)
    ops = Tuple(recipe.op for recipe in plan.recipes)
    in_place ?
        _ReactiveGetterInPlace(compile(ast), ops, cache_apply, ast) :
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
# A type is a candidate for in-place reuse when it is a mutable, non-isbits
# object (arrays, `MutableDiffResult`-style containers, any mutable struct):
# those own heap storage a mutating op can fill in place. Scalars/isbits and
# immutable types can only be replaced, so they take the pure branch and never
# rely on the garbage contents of an `isassigned`-but-uninitialized `Ref`.
_mutable_like(::Type{T}) where {T} = ismutabletype(T) && !isbitstype(T)

"""
    _default_is_mutating(recipe) -> Bool

Default per-recipe in-place selector used by [`prepare_reactive_nonallocating`](@ref):
route a single-output recipe through the in-place hook when its output type is
mutable/array-like. A consumer may pass a narrower `is_mutating` to opt only
specific recipes in-place while sharing the same getter codegen.
"""
_default_is_mutating(recipe::Recipe) = _mutable_like(valtype(only(recipe.outputs)))

# The slot indices whose single-output, owned producer recipe is selected for
# in-place evaluation. Empty (so the AST is byte-identical to the pure program)
# whenever no `cache_apply` was injected.
function _mutating_slots(plan::Plan, graph::Graph, index, cache_apply, is_mutating)
    slots = Set{Int}()
    cache_apply === nothing && return slots
    for recipe in plan.recipes
        length(recipe.outputs) == 1 || continue
        output_id = canon_id(graph, only(recipe.outputs).id)
        _owned_output(plan, recipe, output_id) || continue
        is_mutating(recipe) || continue
        push!(slots, index[output_id])
    end
    slots
end

function prepare_reactive(graph::Graph; have = (), want = ())
    _prepare_reactive(graph; have = have, want = want)
end

"""
    prepare_reactive_nonallocating(graph; have, want, is_mutating=_default_is_mutating)
    prepare_reactive_nonallocating(spec::KernelSpec; ...)

Optional MutatingFunctions-backed reactive preparation. Install and load
`MutatingFunctions` alongside `ReactiveKernels` to activate the package
extension that supplies these methods. It prepares the same closed-world state
program as [`prepare_reactive`](@ref), but selected single-output recipes reuse
their per-instance slot buffer in place through `MutatingFunctions.apply!!`
instead of allocating a fresh recipe return on every recomputation.

Selection is per recipe via `is_mutating(recipe)::Bool` (default: mutable/
array-like outputs). The injected core hook is a single MF-agnostic callable
`cache_apply(cache, op, args...) -> newcache`; `MutatingFunctions.apply!!`
supplies it, and any conforming hand-written mutating op (including a
non-MutatingFunctions one) fits the same codegen. **Contract on `cache_apply`:**
it must return the (possibly new) result object, and it must treat an
immutable/isbits `cache` as passthrough-recompute (ignore the cache bits and
return `op(args...)` fresh) — `apply!!` already does both.

Slots are per [`CompiledReactiveState`](@ref) instance, so caches are owned
per instance; `copy`/`copyto!` deep-copy buffers, so distinct instances never
alias mutable storage. Reused results are borrowed values that may be
overwritten by the next recomputation; a state is mutable and not thread-safe.
"""
function prepare_reactive_nonallocating(args...; kwargs...)
    throw(ArgumentError(
        "prepare_reactive_nonallocating requires the optional MutatingFunctions extension; install MutatingFunctions and load it with `using MutatingFunctions`"))
end

function _prepare_reactive(graph::Graph; have = (), want = (),
                           cache_apply = nothing,
                           is_mutating = _default_is_mutating)
    selected = plan(graph; have = have, want = want)
    values = _state_values(selected)
    index = Dict(canon_id(graph, value.id) => i for (i, value) in enumerate(values))
    token = Ref{Nothing}(nothing)
    handles = _state_handles(token, graph, values)
    recipe_index = Dict(recipe.id => i for (i, recipe) in enumerate(selected.recipes))
    mutating_slots = _mutating_slots(selected, graph, index, cache_apply, is_mutating)
    getters = Tuple(
        _prepare_getter(selected, index, recipe_index, canon_id(graph, value.id);
                        cache_apply = cache_apply, mutating_slots = mutating_slots)
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

# Resolve a KernelSpec's boundary the same way `prepare_reactive` does, then
# delegate to the in-place-capable builder. The MutatingFunctions extension
# calls this for `prepare_reactive_nonallocating(spec::KernelSpec; ...)`, so the
# authoring surface shares one runtime with the hand-built-graph path.
function _prepare_reactive(spec::KernelSpec;
                           have = _KERNEL_DEFAULT_BOUNDARY,
                           want = _KERNEL_DEFAULT_BOUNDARY,
                           kwargs...)
    resolved_have = _kernel_selection(spec, have, spec.have_names, :have)
    resolved_want = _kernel_selection(spec, want, spec.want_names, :want)
    _prepare_reactive(spec.graph;
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

# `assign!` is the INTERNAL direct-slot writer the @reactive facade lowers every
# object field write to (both HAVE and derived), so the facade needs no
# HAVE/derived branching. It writes the slot, marks it valid, and runs the
# existing dependent-invalidation worklist — it never changes the frozen bit and
# rejects a frozen slot. On a HAVE source it equals `set!`; on an unfrozen
# derived slot it is a TEMPORARY OVERRIDE: `get!` short-circuits on the valid bit
# and returns the assigned value without recomputing, and a later change to a
# true upstream dependency invalidates this slot again so `get!` recomputes from
# the recipe. `set!` stays HAVE-only and exported; `assign!` is not exported —
# arbitrary slot pokes are not part of the public surface (approved by
# ReactiveKernels:poc against the compiled-state invariants).
function assign!(state::CompiledReactiveState, handle::ReactiveValue{I,T}, value) where {I,T}
    _check_handle(state, handle)
    state.frozen[I] && throw(ArgumentError(
        "assign! cannot override frozen slot $I; unfreeze! first"))
    _tuple_slot(state.slots, Val(I))[] = value
    state.valid[I] = true
    _invalidate_dependents!(state, I)
    state
end

"""
In-place `assign!`: materialize the slot, mutate it via `f`, then mark valid and
invalidate dependents (even if `f` throws). Returns `f`'s result so a rooted
in-place assignment lowered onto this preserves Julia's assignment value (the new
scalar for an indexed compound, the destination for a dotted/broadcast assign);
`try return f(value) finally …` keeps the finally invalidation on both paths.
"""
function assign!(f, state::CompiledReactiveState, handle::ReactiveValue{I,T}) where {I,T}
    _check_handle(state, handle)
    state.frozen[I] && throw(ArgumentError(
        "assign! cannot override frozen slot $I; unfreeze! first"))
    value = get!(state, handle)
    try
        return f(value)
    finally
        state.valid[I] = true
        _invalidate_dependents!(state, I)
    end
end

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

# --- grouped HAVE-boundary copy ---------------------------------------------
#
# `copy_group!` copies a *selected group* of source (HAVE) slots — e.g. a phase
# point's `(pos, mom)` — from one set of handles into another within the SAME
# compiled state, reusing array buffers so an array group is 0-alloc, then
# invalidating each destination's downstream through the ordinary HAVE-boundary
# API. It is the one additive primitive the sampler needs for proposal-accept,
# endpoint copy, and swap-via-temp: unlike whole-state `copyto!` it touches only
# the named slots, and unlike a nested-subscription scheme it adds no
# per-instance state — it is pure composition over `get!`/`set!`/`mutate!`, so
# it cannot clone stale subscriptions and preserves every slot-index/validity
# invariant of the engine.

# Array destinations reuse their existing buffer (in-place copyto!); scalar
# destinations are set. Dispatch is on the handle value-type parameter, so the
# branch is resolved at compile time and stays type-stable.
@inline _group_assign!(state::CompiledReactiveState,
                       dest::ReactiveValue{I,T}, value) where {I,T<:AbstractArray} =
    mutate!(buffer -> copyto!(buffer, value), state, dest)
@inline _group_assign!(state::CompiledReactiveState,
                       dest::ReactiveValue{I,T}, value) where {I,T} =
    set!(state, dest, value)

@inline function _copy_group!(state::CompiledReactiveState,
                              dest::Tuple, src::Tuple)
    d = first(dest)
    _check_handle(state, d)
    state.program.sources[_slot_index(d)] || throw(ArgumentError(
        "copy_group! destinations must be ReactiveProgram HAVE sources",
    ))
    _group_assign!(state, d, get!(state, first(src)))
    _copy_group!(state, Base.tail(dest), Base.tail(src))
end
@inline _copy_group!(::CompiledReactiveState, ::Tuple{}, ::Tuple{}) = nothing

"""
    copy_group!(state, dest_handles, src_handles)

Copy each source slot in `src_handles` into the paired destination slot in
`dest_handles`, in place where the value is an array (buffer reused, 0-alloc),
then invalidate each destination's downstream. Every destination handle must be
a declared HAVE source; sources may be any readable slot. This is an additive
composition over the HAVE-boundary API — it changes no invalidation or copy
internals — and is the group-copy building block for phase-point
proposal/endpoint bookkeeping. `dest_handles` and `src_handles` are equal-length
tuples of [`ReactiveValue`](@ref) handles; the pairing is walked by type-stable
tuple recursion so a homogeneous array group copies with zero allocation.

Pairs are applied **sequentially, left to right**, and an array destination is
mutated in place. Overlapping destination/source groups are therefore
order-dependent and generally NOT a swap: `copy_group!(state, (a, b), (b, a))`
copies `b` into `a` first, so both end holding the original `b`. To exchange two
groups, route through a temporary group (`t ← a`, `a ← b`, `b ← t`) — the
allocation-free idiom the sampler uses for endpoint/proposal swaps. Sources are
read (`get!`) before their paired destination is written, but later pairs see
earlier destinations' new values.
"""
function copy_group!(state::CompiledReactiveState,
                     dest_handles::Tuple, src_handles::Tuple)
    length(dest_handles) == length(src_handles) || throw(DimensionMismatch(
        "copy_group! needs equal-length destination and source handle tuples",
    ))
    _copy_group!(state, dest_handles, src_handles)
    state
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
