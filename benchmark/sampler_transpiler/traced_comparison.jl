function traced_slot_inputs(template)
    (Reactant.to_rarray(deepcopy(template);track_numbers=true),
     Reactant.to_rarray(UInt64[91,77]),
     Reactant.to_rarray((0,0);track_numbers=true))
end

# Own working storage once for the whole chain. Only the requested logical
# endpoint crosses the result boundary; internal scratch/cache mutations do
# not become observable writes to the caller's initial state.
function traced_owned_batch(call,projection,state,seed,counts,n)
    output=traced_slot_batch(call,deepcopy(state),copy(seed),deepcopy(counts),n)
    (;values=map(i->getfield(output.state.values,i),projection.values),
      seed=output.seed,counts=output.counts)
end

function check_traced_inputs_unchanged(inputs,template)
    actual=(inputs[1].values...,inputs[1].current...,inputs[3]...)
    expected=(template.values...,template.current...,0,0)
    for (value,initial) in zip(actual,expected)
        observed=value isa Reactant.AbstractConcreteArray ? Array(value) : convert(typeof(initial),value)
        isequal(observed,initial) || error("owned batch changed its caller's initial state")
    end
    Array(inputs[2])==UInt64[91,77] || error("owned batch changed its caller's seed")
end

function run_traced_comparison(prototype; n=1000, compiler_options=NamedTuple(), phasepoint_output=true)
    reset_native_slots!(prototype.prepared)
    program=measured("lower_traced_slots") do
        compile_traced_slots(prototype.prepared.program;compiler_options...)
    end
    println("proven_currentness_facts=",length(program.entry_facts),
        " runtime_currentness_flags=",length(program.state.current),
        " metadata_entries=",length(program.metadata.values))
    # This experimental backend emits ordinary Julia functions containing
    # branch closures. Cross their construction world once, outside timing.
    prefix=Symbol(prototype.prepared.program.contexts[2].prefix,:_owned)
    indices=Tuple(i for (i,pair) in enumerate(program.ordered) if first(first(pair))===prefix)
    Base.invokelatest(run_traced_prepared,program,prototype.comparator,n,prototype.L,
        phasepoint_output ? SlotStatic(indices) : nothing)
end

function run_traced_prepared(program,comparator,n,L,projection)
    call=TracedSlotCall(program.f,program.metadata)
    driver=projection===nothing ?
        (state,seed,counts)->traced_slot_batch(call,state,seed,counts,n) :
        (state,seed,counts)->traced_owned_batch(call,projection,state,seed,counts,n)
    inputs=traced_slot_inputs(program.state)
    compiled=measured("compile_traced_slots") do
        Reactant.compile(driver,inputs;sync=true,donated_args=:none,serializable=true)
    end
    println("mlir_bytes=",sizeof(compiled.module_string),
        " while_regions=",count("stablehlo.while",compiled.module_string),
        " if_regions=",count("stablehlo.if",compiled.module_string))
    measured("traced_slots_first_chain") do
        compiled(inputs...)
    end
    projection===nothing || check_traced_inputs_unchanged(inputs,program.state)
    ahmc_chain(Xoshiro(91),comparator.hamiltonian,comparator.kernel,
        deepcopy(comparator.initial),2)
    for _ in 1:7
        inputs=traced_slot_inputs(program.state)
        result=measured("traced_slots_$(n)_transitions") do
            compiled(inputs...)
        end
        steps=Int(result.counts[1])
        println("integration_steps=",steps)
        steps==L*n || error("shortened traced workload")
        for value in (projection===nothing ? result.state.values : result.values)
            value isa Reactant.AbstractConcreteArray || continue
            all(isfinite,Array(value)) || error("nonfinite traced vector")
        end
        rng=Xoshiro(91);point=deepcopy(comparator.initial)
        measured("ahmc_$(n)_transitions") do
            ahmc_chain(rng,comparator.hamiltonian,comparator.kernel,point,n)
        end
    end
    (;program,compiled)
end
