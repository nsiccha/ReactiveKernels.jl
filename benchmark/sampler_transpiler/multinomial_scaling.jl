include(joinpath(@__DIR__,"multinomial_eight_schools.jl"))

# One row per timed batch. Data work and compilation have separate phases;
# source count instrumentation is outside the timed gradient callback.
function scaling_row(io,backend,L,n,sample,timing;phase="execute",paired_with=backend)
    println(io,join((backend,paired_with,L,n,L*n,sample,phase,timing.time,timing.bytes),','))
    flush(io)
end

function native_scaling(io,prototype,batches)
    prepared=prototype.prepared; comparator=prototype.comparator; L=prototype.L
    rng=reset_native_slots!(prepared)
    slot_chain(prepared.program,rng,2)
    ahmc_chain(Xoshiro(91),comparator.hamiltonian,comparator.kernel,
        deepcopy(comparator.initial),2)
    for n in batches, sample in 1:7
        rng=reset_native_slots!(prepared)
        timing=@timed slot_chain(prepared.program,rng,n)
        prepared.program.counts[1]==L*n || error("shortened native workload")
        all(isfinite,native_position(prepared.program)) || error("nonfinite native position")
        scaling_row(io,"native",L,n,sample,timing)
        rng=Xoshiro(91);point=deepcopy(comparator.initial)
        timing=@timed ahmc_chain(rng,comparator.hamiltonian,comparator.kernel,point,n)
        all(isfinite,timing.value.θ) || error("nonfinite AHMC position")
        scaling_row(io,"AdvancedHMC",L,n,sample,timing;paired_with="native")
    end
end

function traced_scaling(io,program,projection,comparator,L,batches;
        reactant_compile_options=NamedTuple())
    call=TracedSlotCall(program.f,program.metadata)
    for n in batches
        driver=(state,seed,counts)->traced_owned_batch(call,projection,state,seed,counts,n)
        inputs=traced_slot_inputs(program.state)
        timing=@timed Reactant.compile(driver,inputs;sync=true,donated_args=:none,
            serializable=true,reactant_compile_options...)
        scaling_row(io,"Reactant",L,n,0,timing;phase="compile")
        compiled=timing.value
        result=compiled(inputs...)
        Int(result.counts[1])==L*n || error("shortened warmup workload")
        check_traced_inputs_unchanged(inputs,program.state)
        ahmc_chain(Xoshiro(91),comparator.hamiltonian,comparator.kernel,
            deepcopy(comparator.initial),2)
        for sample in 1:7
            inputs=traced_slot_inputs(program.state)
            timing=@timed compiled(inputs...)
            result=timing.value
            Int(result.counts[1])==L*n || error("shortened traced workload")
            for value in result.values
                value isa Reactant.AbstractConcreteArray || continue
                all(isfinite,Array(value)) || error("nonfinite traced vector")
            end
            scaling_row(io,"Reactant",L,n,sample,timing)
            rng=Xoshiro(91);point=deepcopy(comparator.initial)
            timing=@timed ahmc_chain(rng,comparator.hamiltonian,comparator.kernel,point,n)
            all(isfinite,timing.value.θ) || error("nonfinite AHMC position")
            scaling_row(io,"AdvancedHMC",L,n,sample,timing;paired_with="Reactant")
        end
        println("scaling_complete backend=Reactant L=",L," transitions=",n,
            " mlir_bytes=",sizeof(compiled.module_string))
    end
end

function multinomial_scaling(path;batches=(100,1000,10000),steps=(4,16),
        prototype_options=NamedTuple(),reactant_compile_options=NamedTuple())
    open(path,"w") do io
        println(io,"backend,paired_with,steps_per_transition,transitions,gradient_evaluations,sample,phase,seconds,allocated_bytes")
        for L in steps
            native=build_multinomial_prototype(;L,prototype_options...)
            native_scaling(io,native,batches)
            traced=build_multinomial_prototype(;L,compiler_options=(peel_loops=true,),
                prototype_options...)
            reset_native_slots!(traced.prepared)
            program=compile_traced_slots(traced.prepared.program)
            prefix=Symbol(traced.prepared.program.contexts[2].prefix,:_owned)
            projection=SlotStatic(Tuple(i for (i,pair) in enumerate(program.ordered)
                if first(first(pair))===prefix))
            Base.invokelatest(traced_scaling,io,program,projection,traced.comparator,L,batches;
                reactant_compile_options)
        end
    end
    println("scaling_csv=",abspath(path))
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: multinomial_scaling.jl /absolute/output.csv")
    multinomial_scaling(only(ARGS))
end
