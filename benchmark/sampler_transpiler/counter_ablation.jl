# Isolate compiler bookkeeping using the same captured mathematical source.
# The counted executable validates the fixed integration work separately.
include(joinpath(@__DIR__,"multinomial_eight_schools.jl"))

function native_counter_sample(prototype,n,label)
    program=prototype.prepared.program
    rng=reset_native_slots!(prototype.prepared)
    result=@timed slot_chain(program,rng,n)
    expected=program.count_steps ? prototype.L*n : 0
    program.counts[1]==expected || error("counter ablation instrumentation")
    all(isfinite,native_position(program)) || error("nonfinite native result")
    println("counter_sample backend=native variant=",label," transitions=",n,
        " seconds=",result.time," bytes=",result.bytes)
end

function build_traced_counter_variant(program,L,n,label)
    traced=compile_traced_slots(program)
    for ((object,index),symbol) in traced.valid_ordered
        context=only(filter(c->Symbol(c.prefix,:_owned)===object,program.contexts))
        names=sort!([name for (name,canon) in slotfields(context) if
            RK.kernel_plan_field(slotplan(context),canon)==(:owned,index)])
        println("runtime_validity object=",object," fields=",names," symbol=",symbol)
    end
    Base.invokelatest(compile_counter_variant,traced,L,n,label,program.count_steps,
        Symbol(program.contexts[2].prefix,:_owned))
end
function compile_counter_variant(program,L,n,label,counted,prefix)
    call=TracedSlotCall(program.f,program.metadata)
    projection=SlotStatic(Tuple(i for (i,pair) in enumerate(program.ordered)
        if first(first(pair))===prefix))
    driver=(state,seed,counts)->traced_owned_batch(call,projection,state,seed,counts,n)
    inputs=traced_slot_inputs(program.state)
    compiled=measured("compile_counter_$(label)_$(n)") do
        Reactant.compile(driver,inputs;sync=true,donated_args=:none,serializable=true)
    end
    result=compiled(inputs...)
    Int(result.counts[1])==(counted ? L*n : 0) || error("traced counter instrumentation")
    check_traced_inputs_unchanged(inputs,program.state)
    println("counter_mlir variant=",label," transitions=",n,
        " bytes=",sizeof(compiled.module_string))
    (;compiled,template=program.state,L,n,label,counted)
end
function traced_counter_sample(variant)
    inputs=traced_slot_inputs(variant.template)
    result=@timed variant.compiled(inputs...)
    Int(result.value.counts[1])==(variant.counted ? variant.L*variant.n : 0) ||
        error("traced counter changed")
    for value in result.value.values
        value isa Reactant.AbstractConcreteArray || continue
        all(isfinite,Array(value)) || error("nonfinite traced result")
    end
    println("counter_sample backend=reactant variant=",variant.label,
        " transitions=",variant.n," seconds=",result.time," bytes=",result.bytes)
end
function run_counter_ablation()
    native=map((true,false)) do counted
        build_multinomial_prototype(L=16,compiler_options=(count_steps=counted,))
    end
    for p in native
        slot_chain(p.prepared.program,reset_native_slots!(p.prepared),2)
    end
    for n in (1000,10000),sample in 1:7
        for i in (isodd(sample) ? (1,2) : (2,1))
            native_counter_sample(native[i],n,i==1 ? "counted" : "uncounted")
        end
    end
    traced=map((true,false)) do counted
        build_multinomial_prototype(L=16,
            compiler_options=(count_steps=counted,peel_loops=true))
    end
    for n in (1000,10000)
        variants=map(enumerate(traced)) do (i,p)
            reset_native_slots!(p.prepared)
            build_traced_counter_variant(p.prepared.program,16,n,i==1 ? "counted" : "uncounted")
        end
        for sample in 1:7, i in (isodd(sample) ? (1,2) : (2,1))
            traced_counter_sample(variants[i])
        end
    end
end
run_counter_ablation()
