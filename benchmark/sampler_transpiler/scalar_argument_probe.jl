# A nonsampler consumer of the same finite MethodIR compiler. The native
# lowering checks the source's closed-form result.
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

check_native_scalar()
