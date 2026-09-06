# Check source-level control behavior independently in each backend. These
# cases do not compare random trajectories or floating-point calculations.
include(joinpath(@__DIR__,"hmc_eight_schools.jl"))

function check_traced_loop_control(program,position_slot,expected_steps,n)
    call=TracedSlotCall(program.f,program.metadata)
    driver=(state,seed,counts)->traced_slot_batch(call,state,seed,counts,n)
    initial=copy(getfield(program.state.values,position_slot))
    inputs=traced_slot_inputs(program.state)
    compiled=measured("compile_control_probe") do
        Reactant.compile(driver,inputs;sync=true,donated_args=:none,serializable=true)
    end
    result=compiled(inputs...)
    Int(result.counts[1])==expected_steps || error("incorrect traced loop trip count")
    Array(getfield(result.state.values,position_slot))==initial ||
        error("source control should leave the initial position unchanged")
    println("traced_control_steps=",Int(result.counts[1]))
end

function check_loop_control(prototype,expected_steps;n=3)
    prepared=prototype.prepared
    rng=reset_native_slots!(prepared)
    initial=copy(native_position(prepared.program))
    slot_chain(prepared.program,rng,n)
    prepared.program.counts[1]==expected_steps || error("incorrect native loop trip count")
    native_position(prepared.program)==initial ||
        error("source control should leave the initial position unchanged")
    println("native_control_steps=",prepared.program.counts[1])
    reset_native_slots!(prepared)
    program=compile_traced_slots(prepared.program)
    context=prepared.program.contexts[2]
    _,slot=RK.kernel_plan_field(slotplan(context),slotfields(context)[:pos])
    key=(Symbol(context.prefix,:_owned),slot)
    position_slot=findfirst(p->first(p)==key,program.ordered)
    Base.invokelatest(check_traced_loop_control,program,position_slot,expected_steps,n)
end

function run_loop_control_probes()
    # A zero-trip loop must not execute the integrator. An unreachable energy
    # threshold makes the authored divergence return fire after its first step.
    check_loop_control(build_fast_prototype(L=0),0)
    check_loop_control(build_fast_prototype(L=16,source_options=(min_dham=Inf,)),3)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_loop_control_probes()
end
