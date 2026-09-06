# Exercise the backend primitive independently, especially ranges whose
# rejection branch runs often and the signed/full-width conversion boundary.
using ReactiveKernels, Reactant, Random
using ReactiveKernels.NativeSlotCompiler: fast_native_factory,compile_native_slots
const TSC=Base.get_extension(ReactiveKernels,:ReactiveKernelsReactantExt).TracedSlotCompiler

@kernel bounded_draw(value;choices=-3:3)=begin
    draw!(rng)=begin
        value=Random.rand(rng,choices)
    end
end

function draw_batch(sampler,initial_seed,initial_values)
    seed=copy(initial_seed)
    values=copy(initial_values)
    rng=Reactant.ReactantRNG(seed)
    Reactant.@trace for i in 1:length(values)
        Reactant.@allowscalar values[i]=TSC.slot_rand_range(rng,sampler)
    end
    (;values,seed=rng.seed)
end

function check_range(range)
    sampler=TSC.SlotRangeSampler(range)
    seed=Reactant.to_rarray(UInt64[91,77])
    values=Reactant.to_rarray(zeros(eltype(range),128))
    driver=(seed,values)->draw_batch(sampler,seed,values)
    compiled=Reactant.compile(driver,(seed,values);sync=true,donated_args=:none)
    result=compiled(seed,values)
    draws=Array(result.values)
    all(in(range),draws) || error("draw outside requested range")
    Array(seed)==UInt64[91,77] || error("caller seed changed")
    all(iszero,Array(values)) || error("caller output buffer changed")
    Array(result.seed)!=UInt64[91,77] || error("RNG did not advance")
    println("range=",range," span=",sampler.span," threshold=",sampler.threshold,
        " distinct_values=",length(unique(draws))," caller_inputs_preserved=true")
end

for range in (0:4,-3:3,typemin(Int64):typemax(Int64),
    UInt64(0):typemax(UInt64),Int8(-128):Int8(127),-1:typemax(Int64))
    check_range(range)
end

input=Reactant.to_rarray(typemax(UInt64);track_numbers=true)
convert_signed=x->TSC.slot_range_integer(Int64,x)
compiled=Reactant.compile(convert_signed,(input,);sync=true,donated_args=:none)
Int64(compiled(input)) == -1 || error("signed integer bit conversion")
println("signed_integer_boundary=true")

parent=fast_native_factory(bounded_draw,0)
native=compile_native_slots(parent.kernel,parent.state,:draw!;
    endpoints=parent.endpoints,effects=parent.effects)
program=TSC.compile_traced_slots(native)
any(v->v isa TSC.SlotRangeSampler,program.metadata.values) ||
    error("shared range did not select the fixed-range lowering")
function check_bound_range(program)
    call=TSC.TracedSlotCall(program.f,program.metadata)
    driver=(state,seed,counts)->TSC.traced_slot_batch(call,deepcopy(state),copy(seed),deepcopy(counts),128)
    inputs=(Reactant.to_rarray(deepcopy(program.state);track_numbers=true),
        Reactant.to_rarray(UInt64[91,77]),Reactant.to_rarray((0,0);track_numbers=true))
    compiled=Reactant.compile(driver,inputs;sync=true,donated_args=:none)
    result=compiled(inputs...)
    Int(only(result.state.values)) in -3:3 || error("bound range source output")
    Int(only(inputs[1].values))==0 || error("caller source state changed")
    println("captured_bound_range=true")
end
Base.invokelatest(check_bound_range,program)
