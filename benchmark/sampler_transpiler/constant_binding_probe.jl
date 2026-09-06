using ReactiveKernels
using ReactiveKernels.NativeSlotCompiler: fast_native_factory,compile_native_slots,
    slot_chain,slotplan,slotfields
const RK=ReactiveKernels

module ConstantConsumers
using ReactiveKernels
const gain=2.0
changing_gain=2.0
const payload=[1.0,2.0]
@kernel fixed(value)=begin
    advance!(increment)=begin
        value=value+gain*increment
    end
end
@kernel changing(value)=begin
    advance!(increment)=begin
        value=value+changing_gain*increment
    end
end
@kernel mutable_global(value)=begin
    advance!(increment)=begin
        value .= payload
    end
end
end

function prepare_constant_consumer(source,initial)
    parent=fast_native_factory(source,initial)
    compile_native_slots(parent.kernel,parent.state,:advance!;
        endpoints=parent.endpoints,effects=parent.effects)
end

program=prepare_constant_consumer(ConstantConsumers.fixed,1.0)
slot_chain(program,3.0,1)
context=only(program.contexts)
_,slot=RK.kernel_plan_field(slotplan(context),slotfields(context)[:value])
RK._canon_slot(context.owned,Val(slot))==7.0 || error("fixed scalar result")
for (source,initial,reason) in (
    (ConstantConsumers.changing,1.0,"captured constant binding"),
    (ConstantConsumers.mutable_global,zeros(2),"builtin scalar"),
)
    rejected=false
    try
        prepare_constant_consumer(source,initial)
    catch err
        occursin(reason,sprint(showerror,err)) || rethrow()
        rejected=true
    end
    rejected || error("unsupported global was admitted")
end
println("fixed_scalar_source=true mutable_or_nonconstant_globals_rejected=true")
