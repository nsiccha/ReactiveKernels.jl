include(joinpath(@__DIR__,"multinomial_eight_schools.jl"))
include(joinpath(@__DIR__,"position_multinomial_hmc_kernel.jl"))
using TOML
function compare_packing(program,projection,output_dir; thresholds=(65536,1024,1048576))
    mkpath(output_dir)
    call=TracedSlotCall(program.f,program.metadata)
    driver=(state,seed,counts)->traced_owned_batch(call,projection,state,seed,counts,1000)
    inputs=traced_slot_inputs(program.state)
    results=Dict{String,Any}()
    compiled=Any[]
    for threshold in thresholds
        options=TracedSlotCompiler.cpu_compile_options(;xla_debug_options=(
            xla_backend_extra_options=Dict(
                "xla_cpu_small_while_loop_byte_threshold"=>string(threshold)),))
        timing=@timed Reactant.compile(driver,inputs;sync=true,donated_args=:none,
            serializable=true,options...)
        push!(compiled,timing.value)
        hlo=sprint(show,only(Reactant.XLA.get_hlo_modules(timing.value.exec)))
        write(joinpath(output_dir,"threshold-$(threshold).hlo"),hlo)
        results[string(threshold)]=Dict("compile_seconds"=>timing.time,
            "compile_bytes"=>timing.bytes,"optimized_hlo_bytes"=>sizeof(hlo),
            "small_calls"=>count("xla_cpu_small_call",hlo),"samples_seconds"=>Float64[])
        println("packed_compile threshold=",threshold," seconds=",timing.time,
            " bytes=",timing.bytes," small_calls=",count("xla_cpu_small_call",hlo))
        output=timing.value(inputs...)
        Int(output.counts[1])==16000 || error("shortened warmup")
        check_traced_inputs_unchanged(inputs,program.state)
    end
    for sample in 1:7
        order=isodd(sample) ? (1,2,3) : (3,2,1)
        for i in order
            threshold=thresholds[i]
            inputs=traced_slot_inputs(program.state)
            timing=@timed compiled[i](inputs...)
            Int(timing.value.counts[1])==16000 || error("shortened execution")
            for v in timing.value.values
                v isa Reactant.AbstractConcreteArray || continue
                all(isfinite,Array(v)) || error("nonfinite position/state")
            end
            check_traced_inputs_unchanged(inputs,program.state)
            push!(results[string(threshold)]["samples_seconds"],timing.time)
            println("packed_sample threshold=",threshold," sample=",sample," seconds=",timing.time)
        end
    end
    open(joinpath(output_dir,"results.toml"),"w") do io
        TOML.print(io,results)
    end
end
function run_cpu_loop_packing(output_dir)
    prototype=build_multinomial_prototype(;L=16,
        source=PositionMultinomialHMCAuthoring.multinomial_hmc_state,
        source_controls=(step_f=F.leapfrog!,stepsize=0.03),
        compiler_options=(peel_loops=true,))
    reset_native_slots!(prototype.prepared)
    program=compile_traced_slots(prototype.prepared.program)
    prefix=Symbol(prototype.prepared.program.contexts[2].prefix,:_owned)
    projection=SlotStatic(Tuple(i for (i,pair) in enumerate(program.ordered)
        if first(first(pair))===prefix))
    Base.invokelatest(compare_packing,program,projection,output_dir)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: cpu_loop_packing.jl /absolute/output-directory")
    run_cpu_loop_packing(only(ARGS))
end
