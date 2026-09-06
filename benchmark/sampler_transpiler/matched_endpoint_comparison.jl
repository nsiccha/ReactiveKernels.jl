# Like-for-like fixed-length endpoint HMC. The legacy source with per-step
# divergence checks is retained separately in endpoint_ablation.jl.
include(joinpath(@__DIR__, "endpoint_ablation.jl"))
include(joinpath(@__DIR__, "endpoint_hmc_kernel.jl"))

function matched_native_case(prototype, n)
    prepared = prototype.prepared
    compiled = rng -> begin
        slot_chain(prepared.program, rng, n)
        native_position(prepared.program), rng
    end
    (; label="rk_native_endpoint", compiled,
       inputs=() -> (reset_native_slots!(prepared),), compile_seconds=nothing,
       count_basis="fixed_source_loop",
       check=result -> begin
           all(isfinite,result[1]) || error("nonfinite native position")
           prototype.L*n
       end)
end

function matched_ahmc_case(prototype, n)
    comparator = prototype.comparator
    compiled = (rng, point) -> begin
        for _ in 1:n
            point = AdvancedHMC.transition(rng, comparator.hamiltonian,
                comparator.kernel, point).z
        end
        point.θ, rng
    end
    (; label="ahmc_endpoint", compiled,
       inputs=() -> (Xoshiro(91),deepcopy(comparator.initial)), compile_seconds=nothing,
       count_basis="FixedNSteps",
       check=result -> begin
           all(isfinite,result[1]) || error("nonfinite AHMC position")
           prototype.L*n
       end)
end

function matched_traced_case(program, projection, L, n, directory)
    call = TracedSlotCall(program.f, program.metadata)
    driver = (state, seed, counts) -> begin
        result = traced_owned_batch(call,projection,state,seed,counts,n)
        only(result.values), result.seed
    end
    inputs = () -> traced_slot_inputs(program.state)
    timing = @timed Reactant.compile(driver,inputs(); sync=true,
        donated_args=:none, serializable=true, TracedSlotCompiler.cpu_compile_options()...)
    label="rk_reactant_endpoint"
    ablation_save(timing.value,label,directory)
    (; label, compiled=timing.value, inputs, compile_seconds=timing.time,
       count_basis="fixed_source_loop",
       check=result -> begin
           all(isfinite,Array(result[1])) || error("nonfinite generated position")
           L*n
       end)
end

function matched_endpoint_comparison(directory; batches=(10000,), steps=(4,16))
    mkpath(directory)
    open(joinpath(directory,"samples.csv"),"w") do io
        println(io,"kernel,steps_per_transition,transitions,integration_steps,sample,phase,seconds,allocated_bytes")
        for n in batches, L in steps
            output=joinpath(directory,"n$(n)-L$(L)"); mkpath(output)
            native=build_fast_prototype(; L,source=EndpointHMCAuthoring.hmc_state,
                compiler_options=(count_steps=false,))
            native_case=Base.invokelatest(matched_native_case,native,n)
            ahmc_case=matched_ahmc_case(native,n)
            traced_native=build_fast_prototype(; L,source=EndpointHMCAuthoring.hmc_state,
                compiler_options=(count_steps=false,peel_loops=true))
            program=compile_traced_slots(traced_native.prepared.program)
            context=traced_native.prepared.program.contexts[2]
            _,slot=RK.kernel_plan_field(slotplan(context),slotfields(context)[:pos])
            key=(Symbol(context.prefix,:_owned),slot)
            index=only(findall(pair->first(pair)==key,program.ordered))
            open(joinpath(output,"rk-native.jl"),"w") do source_io
                Base.show_unquoted(source_io,native.prepared.program.expression)
            end
            traced_case=Base.invokelatest(matched_traced_case,program,
                SlotStatic((index,)),L,n,output)
            probprog_case=ablation_probprog_case(L,n,output)
            Base.invokelatest(ablation_samples,
                (native_case,ahmc_case,traced_case,probprog_case),io,L,n)
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    1 <= length(ARGS) <= 2 || error("usage: matched_endpoint_comparison.jl /absolute/output-directory [scaling]")
    length(ARGS)==1 || ARGS[2]=="scaling" || error("expected scaling")
    matched_endpoint_comparison(ARGS[1]; batches=length(ARGS)==2 ? (100,1000,10000) : (10000,))
end
