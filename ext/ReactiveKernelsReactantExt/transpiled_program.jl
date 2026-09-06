const NSC = RK.NativeSlotCompiler

struct TranspiledBatch{F,P,M}
    step::F
    project::P
    metadata::M
    iterations::Int
end
Reactant.make_tracer(seen, previous::TranspiledBatch, path, mode; kwargs...) = previous
Reactant.traced_type_inner(::Type{T}, seen, mode::Reactant.TraceMode,
    track_numbers::Type, ndevices, runtime) where {T<:TranspiledBatch} = T

function (program::TranspiledBatch)(input, input_argument)
    state = deepcopy(input)
    argument = deepcopy(input_argument)
    counts = (0, 0)
    Reactant.@trace for _ in 1:program.iterations
        result = program.step(state, argument, counts, program.metadata)
        state = result.state
        argument = result.argument
        counts = result.counts
    end
    result = program.project(state, argument, counts, program.metadata)
    (; state=result.state, argument=result.argument, outputs=result.outputs)
end

struct ReactantTranspiledProgram{C,T,A,K,I}
    compiled::C
    traced::T
    argument::A
    key::K
    initial::I
end

function _device_argument(argument)
    if argument isa Random.AbstractRNG && !(argument isa Reactant.ReactantRNG)
        throw(ArgumentError("the Reactant backend requires Reactant.ReactantRNG for random operations"))
    end
    # The compiled batch owns its working copy. Conversion itself does not
    # mutate the caller's value, so a second host/device copy is unnecessary.
    Reactant.to_rarray(argument; track_numbers=true)
end

function NSC._runtime_contract(template::Reactant.ReactantRNG, argument::Reactant.ReactantRNG)
    typeof(argument) === typeof(template) && axes(argument.seed) == axes(template.seed) &&
        argument.algorithm == template.algorithm ||
        throw(ArgumentError("RNG type, seed shape or algorithm changed; prepare for the new RNG"))
    nothing
end

function _compile_transpiled(prepared, traced, options)
    get(options, :donated_args, :none) === :none ||
        throw(ArgumentError("the transpiler's state-preserving interface requires donated_args=:none"))
    driver = TranspiledBatch(traced.f, traced.projector, traced.metadata, prepared.iterations)
    state = Reactant.to_rarray(deepcopy(traced.state); track_numbers=true)
    argument = _device_argument(prepared.argument)
    settings = merge(cpu_compile_options(), (; sync=true, donated_args=:none, serializable=true), options)
    compiled = Reactant.compile(driver, (state, argument); settings...)
    ReactantTranspiledProgram(compiled, traced, argument, Ref(nothing), state)
end

function NSC._prepare_transpiled_backend(::Val{:reactant},
        prepared::NSC.NativeTranspiledProgram, projection, options)
    traced = compile_traced_slots(prepared.program; projection)
    Base.invokelatest(_compile_transpiled, prepared, traced, options)
end

NSC.initial_transpiled_state(prepared::ReactantTranspiledProgram) =
    NSC.TranspiledState(deepcopy(prepared.initial), prepared.key)

function (prepared::ReactantTranspiledProgram)(state::NSC.TranspiledState, argument)
    state.key === prepared.key ||
        throw(ArgumentError("state belongs to a different prepared program"))
    converted = _device_argument(argument)
    NSC._runtime_contract(prepared.argument, converted)
    result = prepared.compiled(state.storage, converted)
    (; state=NSC.TranspiledState(result.state, prepared.key),
       outputs=deepcopy(result.outputs), argument=result.argument)
end
