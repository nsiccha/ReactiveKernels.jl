function traced_slot_inputs(template)
    (Reactant.to_rarray(deepcopy(template);track_numbers=true),
     Reactant.to_rarray(UInt64[91,77]),
     Reactant.to_rarray((0,0);track_numbers=true))
end

function run_traced_comparison(prototype; n=1000, compiler_options=NamedTuple())
    reset_native_slots!(prototype.prepared)
    program=measured("lower_traced_slots") do
        compile_traced_slots(prototype.prepared.program;compiler_options...)
    end
    println("proven_currentness_facts=",length(program.entry_facts))
    # This experimental backend emits ordinary Julia functions containing
    # branch closures. Cross their construction world once, outside timing.
    Base.invokelatest(run_traced_prepared,program,prototype.comparator,n,prototype.L)
end

function run_traced_prepared(program,comparator,n,L)
    call=TracedSlotCall(program.f,program.metadata)
    driver=(state,seed,counts)->traced_slot_batch(call,state,seed,counts,n)
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
        for value in result.state.values
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
