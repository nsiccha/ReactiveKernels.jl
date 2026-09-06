# Attribute compilation cost separately from preparation and execution.
# Sampling a JIT compiler adds substantial overhead: use the ordinary
# reactant-slots command for fresh-process compilation-time measurements.
include(joinpath(@__DIR__,"hmc_eight_schools.jl"))
using Profile, Serialization

function profile_sampler_compile(program,projection,output_dir)
    mkpath(output_dir)
    call=TracedSlotCall(program.f,program.metadata)
    driver=(state,seed,counts)->traced_owned_batch(call,projection,state,seed,counts,1000)
    inputs=traced_slot_inputs(program.state)
    Profile.init(n=10_000_000,delay=0.005)
    Profile.clear()
    compiled=measured("profiled_compile") do
        Profile.@profile Reactant.compile(driver,inputs;
            sync=true,donated_args=:none,serializable=true)
    end
    data,lidict=Profile.retrieve()
    serialize(joinpath(output_dir,"sampler_compile.profile"),(data,lidict))
    open(joinpath(output_dir,"sampler_compile.flat.txt"),"w") do io
        Profile.print(io,data,lidict;format=:flat,sortedby=:count,mincount=20,C=true)
    end
    open(joinpath(output_dir,"sampler_compile.tree.txt"),"w") do io
        Profile.print(IOContext(io,:displaysize=>(100000,240)),data,lidict;
            format=:tree,maxdepth=85,mincount=100,C=false)
    end
    println("profile_entries=",length(data)," profile_directory=",abspath(output_dir))
    result=compiled(inputs...)
    println("integration_steps=",Int(result.counts[1]),
        " mlir_bytes=",sizeof(compiled.module_string))
    Int(result.counts[1])==4000 || error("unexpected profile workload")
    check_traced_inputs_unchanged(inputs,program.state)
end

function run_compile_profile(output_dir)
    prototype=build_fast_prototype(compiler_options=(peel_loops=true,))
    reset_native_slots!(prototype.prepared)
    program=compile_traced_slots(prototype.prepared.program)
    prefix=Symbol(prototype.prepared.program.contexts[2].prefix,:_owned)
    indices=Tuple(i for (i,pair) in enumerate(program.ordered) if first(first(pair))===prefix)
    Base.invokelatest(profile_sampler_compile,program,SlotStatic(indices),output_dir)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_compile_profile(isempty(ARGS) ? mktempdir(;cleanup=false) : only(ARGS))
end
