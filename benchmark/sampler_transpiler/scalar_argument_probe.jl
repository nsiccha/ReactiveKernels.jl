# A nonsampler consumer of the same finite MethodIR compiler. Each backend
# checks the source's closed-form result; no cross-backend comparison is used.
using ReactiveKernels
using ReactiveKernels.NativeSlotCompiler:
    fast_native_factory,compile_native_slots,slot_chain,slotplan,slotfields
const RK=ReactiveKernels

@kernel accumulating_value(value; rounds=4) = begin
    advance!(increment)=begin
        for _ in 1:rounds
            value=value+increment
        end
    end
end

function build_scalar_program()
    parent=fast_native_factory(accumulating_value,1.0;rounds=4)
    compile_native_slots(parent.kernel,parent.state,:advance!;
        endpoints=parent.endpoints,effects=parent.effects,peel_loops=true)
end

function check_native_scalar()
    program=build_scalar_program()
    slot_chain(program,2.0,1)
    context=only(program.contexts)
    _,slot=RK.kernel_plan_field(slotplan(context),slotfields(context)[:value])
    value=RK._canon_slot(context.owned,Val(slot))
    value==9.0 || error("native source accumulation failed")
    println("native_scalar_value=",value)
end

# Execute native lowering before loading the optional backend.
check_native_scalar()
using Reactant
const TSC=Base.get_extension(RK,:ReactiveKernelsReactantExt).TracedSlotCompiler

function check_traced_scalar(program)
    f,metadata=program.f,program.metadata
    driver=(state,increment,counts)->f(deepcopy(state),increment,counts,metadata)
    initial=Reactant.to_rarray(deepcopy(program.state);track_numbers=true)
    increment=Reactant.to_rarray(2.0;track_numbers=true)
    counts=Reactant.to_rarray((0,0);track_numbers=true)
    compiled=Reactant.compile(driver,(initial,increment,counts);
        sync=true,donated_args=:none)
    # Reuse one executable with different runtime scalar values.
    for (amount,expected) in ((2.0,9.0),(3.0,13.0))
        input=Reactant.to_rarray(amount;track_numbers=true)
        result=compiled(initial,input,counts)
        value=Float64(only(result.state.values))
        value==expected || error("traced source accumulation failed")
        Float64(result.argument)==amount || error("traced argument changed")
        Float64(only(initial.values))==1.0 || error("traced caller state changed")
        println("traced_scalar_value=",value," preserved_increment=",Float64(result.argument))
    end
end

Base.invokelatest(check_traced_scalar,TSC.compile_traced_slots(build_scalar_program()))
