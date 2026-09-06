# Experimental consumer boundary over the existing finite MethodIR compiler.
# State is passed explicitly; preparation metadata never belongs to the state.
struct TranspiledState{S,K}
    storage::S
    key::K
end

struct NativeTranspiledProgram{P,F,A,O,K,I}
    program::P
    project::F
    argument::A
    outputs::O
    key::K
    initial::I
    iterations::Int
end

"""
    transpiled_endpoint(source, method, args...)

Prepare a captured reactive child and its captured free method for use as a
construction input to [`prepare_transpiled`](@ref). Owned numerical fields are
copied when the parent is prepared; fixed callable authorities remain shared.
This interface is experimental.
"""
transpiled_endpoint(source, method, args...) =
    NativePoint(native_endpoint(source, method, args))

function _output_path(path)
    path isa Symbol && return (path,)
    path isa Tuple && 1 <= length(path) <= 2 && all(x -> x isa Symbol, path) &&
        return path
    throw(ArgumentError("an output path must be a field Symbol or a (child, field) tuple"))
end

function _output_locations(program, endpoints, outputs::NamedTuple)
    isempty(outputs) && throw(ArgumentError("request at least one named output"))
    map(values(outputs)) do requested
        path = _output_path(requested)
        index = if length(path) == 1
            1
        else
            child = findfirst(==(first(path)), keys(endpoints))
            child === nothing && throw(ArgumentError("unknown output child $(first(path))"))
            child + 1
        end
        context = program.contexts[index]
        name = last(path)
        haskey(slotfields(context), name) ||
            throw(ArgumentError("unknown output field $path"))
        canon = slotfields(context)[name]
        role, slot = RK.kernel_plan_field(slotplan(context), canon)
        storage = role === :owned ? context.owned : context.shared
        value = RK._canon_slot(storage, Val(slot))
        (RK._kernel_dom_num_scalar(typeof(value)) ||
         RK._kernel_dom_num_array(typeof(value))) ||
            throw(ArgumentError("output $path must be a builtin numeric scalar or array"))
        (; index, name, role, slot)
    end
end

function _qualify_projection(x)
    x isa Expr || return x
    if x.head === :. && x.args[1] === :RK
        return GlobalRef(RK, x.args[2].value)
    elseif x.head === :call && x.args[1] isa Symbol && isdefined(RK, x.args[1])
        return Expr(:call, GlobalRef(RK, x.args[1]),
                    (_qualify_projection(a) for a in x.args[2:end])...)
    end
    Expr(x.head, (_qualify_projection(a) for a in x.args)...)
end

function _projection_expression(program, locations, names)
    setup = Any[]
    for (i, context) in enumerate(program.contexts)
        push!(setup, :(local $(Symbol(context.prefix, :_owned)) = getfield(getfield(stores, $i), 1)))
        push!(setup, :(local $(Symbol(context.prefix, :_shared)) = getfield(getfield(stores, $i), 2)))
        push!(setup, :(local $(Symbol(context.prefix, :_handles)) = getfield(resources, $i)))
    end
    reads = map(locations) do location
        slot_read(program.contexts[location.index], location.name)
    end
    result = Expr(:call, NamedTuple{names}, Expr(:tuple, reads...))
    _qualify_projection(:((stores, resources, constants, counts, argument) -> begin
        $(setup...)
        $result
    end))
end

function _runtime_contract(template, argument)
    typeof(argument) === typeof(template) ||
        throw(ArgumentError("runtime argument type changed; prepare for the new type"))
    if template isa AbstractArray
        axes(argument) == axes(template) ||
            throw(ArgumentError("runtime argument axes changed; prepare for the new shape"))
    elseif !(template isa Number || template isa Random.AbstractRNG)
        throw(ArgumentError("runtime arguments currently support numeric scalars, arrays and RNGs"))
    end
    nothing
end

# Captured methods and callable authorities are preparation metadata. Copying
# their compiler/module graphs is neither necessary nor a valid state copy.
_copy_construction(value::Union{Function, RK._Mode2KernelSkeleton,
    RK._StatefulKernelSkeleton}, seen) = value
_copy_construction(value::Union{Tuple,NamedTuple}, seen) =
    map(x -> _copy_construction(x, seen), value)
_copy_construction(value, seen) = Base.deepcopy_internal(value, seen)

"""
    prepare_transpiled(source, args...; method, argument, outputs,
        iterations=1, backend=:native, kernel_kwargs=(;),
        compiler_options=(;), backend_options=(;))

Compile a captured `@kernel` method for fixed types, shapes, call graph and
compiler-known mutation. `argument` provides the runtime argument's type/shape;
its value can change between calls. `outputs` maps names to source fields, e.g.
`(value=:value, position=(:init, :pos))`. Derived outputs are refreshed once after
the batch. `kernel_kwargs` supplies construction keywords.

Call `prepared(state, argument)` to execute `iterations` method calls and return
`(; state, outputs, argument)`. Continue with the returned state and argument,
or reuse an earlier state. Inputs are preserved and named outputs are independent
snapshots. State belongs to one preparation; its storage is private.

The optional `:reactant` backend requires loading Reactant and using
`Reactant.ReactantRNG` for random operations. It returns device values, compiles
the whole batch and synchronizes by default. Native Julia accepts its ordinary
RNGs. Runtime arguments currently support numeric scalars, arrays and RNGs.
Reprepare when types, shapes, construction controls or fixed authorities change.
The interface is experimental; unsupported source forms reject at preparation.
"""
function prepare_transpiled(source, args...; method::Symbol, argument,
        outputs::NamedTuple, iterations::Integer=1, backend::Symbol=:native,
        kernel_kwargs::NamedTuple=NamedTuple(),
        compiler_options::NamedTuple=NamedTuple(),
        backend_options::NamedTuple=NamedTuple())
    0 <= iterations <= typemax(Int) ||
        throw(ArgumentError("iterations must be a nonnegative machine integer"))
    _runtime_contract(argument, argument)
    # Copy numerical construction inputs together so known aliases survive.
    # NativePoint's copy isolates its owned fields and retains fixed authorities.
    owned_args, owned_kwargs = _copy_construction((args, kernel_kwargs), IdDict())
    parent = fast_native_factory(source, owned_args...; owned_kwargs...)
    options = merge((; count_steps=false, peel_loops=backend === :reactant), compiler_options)
    program = compile_native_slots(parent.kernel, parent.state, method;
        endpoints=parent.endpoints, effects=parent.effects, options...)
    locations = _output_locations(program, parent.endpoints, outputs)
    projection = _projection_expression(program, locations, keys(outputs))
    initial = deepcopy(map(first, program.stores))
    prepared = NativeTranspiledProgram(program, RK.compile(projection),
        deepcopy(argument), outputs, Ref(nothing), initial, Int(iterations))
    _prepare_transpiled_backend(Val(backend), prepared, projection, backend_options)
end

function _prepare_transpiled_backend(::Val{:native}, prepared, projection, options)
    isempty(options) || throw(ArgumentError("native backend accepts no backend_options"))
    prepared
end
_prepare_transpiled_backend(::Val{:reactant}, prepared, projection, options) =
    throw(ArgumentError("load Reactant before preparing the Reactant backend"))
_prepare_transpiled_backend(::Val{B}, prepared, projection, options) where {B} =
    throw(ArgumentError("unknown transpiler backend $B"))

"""
    initial_transpiled_state(prepared)

Return an independent initial state for an experimental prepared transpiled
program. Pass it to `prepared(state, argument)`; use the returned `.outputs` to
observe named values, and the returned `.state` to continue execution.
"""
initial_transpiled_state(prepared::NativeTranspiledProgram) =
    TranspiledState(deepcopy(prepared.initial), prepared.key)

function (prepared::NativeTranspiledProgram)(state::TranspiledState, argument)
    state.key === prepared.key ||
        throw(ArgumentError("state belongs to a different prepared program"))
    _runtime_contract(prepared.argument, argument)
    storage = deepcopy(state.storage)
    stores = map((owned, original) -> (owned, last(original)), storage, prepared.program.stores)
    live_argument = deepcopy(argument)
    counts = [0, 0]
    # Reuse the existing specialized batch loop. Ownership/projection work
    # stays in this outer wrapper instead of enlarging the numerical loop.
    slot_chain(merge(prepared.program, (; stores, counts)), live_argument, prepared.iterations)
    outputs = RK.RuntimeGeneratedFunctions.generated_callfunc(prepared.project,
        stores, prepared.program.resources, prepared.program.constants, counts, live_argument)
    (; state=TranspiledState(storage, prepared.key), outputs=deepcopy(outputs), argument=live_argument)
end
