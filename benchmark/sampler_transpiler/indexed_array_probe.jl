using ReactiveKernels,Random
using ReactiveKernels.NativeSlotCompiler: fast_native_factory,compile_native_slots,
    slot_chain,slotplan,slotfields
const RK=ReactiveKernels

@kernel array_sum(total;samples=[1.0,2.0,3.0,4.0],indices=1:4)=begin
    advance!(unused)=begin
        total=zero(total)
        for i in indices
            total=total+samples[i]
        end
    end
end

function prepare_array_sum(indices)
    parent=fast_native_factory(array_sum,0.0;indices)
    compile_native_slots(parent.kernel,parent.state,:advance!;
        endpoints=parent.endpoints,effects=parent.effects)
end
for (indices,expected) in ((1:4,10.0),(4:-1:1,10.0),(2:2:4,6.0),(1:0,0.0))
    program=prepare_array_sum(indices)
    slot_chain(program,0,1)
    context=only(program.contexts)
    _,slot=RK.kernel_plan_field(slotplan(context),slotfields(context)[:total])
    RK._canon_slot(context.owned,Val(slot))==expected || error("indexed source result")
end
for indices in (0:3,1:5,5:-1:1)
    rejected=false
    try
        prepare_array_sum(indices)
    catch err
        occursin("prepared array bounds",sprint(showerror,err)) || rethrow()
        rejected=true
    end
    rejected || error("out-of-bounds loop was admitted")
end
# An arbitrary range can lie about the bounds of the values it iterates.
# Only the builtin range representations justify removing an index check.
struct MisleadingRange <: AbstractUnitRange{Int} end
Base.first(::MisleadingRange)=1
Base.last(::MisleadingRange)=4
Base.length(::MisleadingRange)=4
Base.iterate(::MisleadingRange,state=1)=state>4 ? nothing : (state+4,state+1)
let rejected=false
    try
        prepare_array_sum(MisleadingRange())
    catch err
        occursin("fixed-loop indices",sprint(showerror,err)) || rethrow()
        rejected=true
    end
    rejected || error("custom range was trusted for bounds elimination")
end
println("native_fixed_index_reads=true invalid_bounds_rejected=true")

using Reactant
const TSC=Base.get_extension(RK,:ReactiveKernelsReactantExt).TracedSlotCompiler
function check_traced_index(program)
    call=TSC.TracedSlotCall(program.f,program.metadata)
    driver=(state,seed,counts)->call(deepcopy(state),copy(seed),deepcopy(counts))
    inputs=(Reactant.to_rarray(deepcopy(program.state);track_numbers=true),
        Reactant.to_rarray(UInt64[91,77]),Reactant.to_rarray((0,0);track_numbers=true))
    compiled=Reactant.compile(driver,inputs;sync=true,donated_args=:none)
    result=compiled(inputs...)
    Float64(only(result.state.values))==10.0 || error("traced indexed source result")
    Float64(only(inputs[1].values))==0.0 || error("caller state changed")
    println("traced_fixed_index_reads=true caller_state_preserved=true")
end
const PROGRAM=TSC.compile_traced_slots(prepare_array_sum(4:-1:1))
Base.invokelatest(check_traced_index,PROGRAM)
