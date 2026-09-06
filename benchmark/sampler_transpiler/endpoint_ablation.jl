# Compare existing captured endpoint and multinomial kernels with ProbProg HMC.
# All compiled calls return only the final position, RNG, and (RK only) counters.
include(joinpath(@__DIR__, "multinomial_eight_schools.jl"))
include(joinpath(@__DIR__, "position_multinomial_hmc_kernel.jl"))
include(joinpath(@__DIR__, "probprog_hmc_scaling.jl"))

function ablation_rk_case(prototype, label, n, directory)
    native = prototype.prepared.program
    reset_native_slots!(prototype.prepared)
    program = compile_traced_slots(native)
    context = native.contexts[2]
    _, slot = RK.kernel_plan_field(slotplan(context), slotfields(context)[:pos])
    key = (Symbol(context.prefix, :_owned), slot)
    index = only(findall(pair -> first(pair) == key, program.ordered))
    open(joinpath(directory, label * "-native.jl"), "w") do io
        Base.show_unquoted(io, native.expression)
    end
    Base.invokelatest(ablation_compile_rk, program, SlotStatic((index,)), label, n, directory)
end

function ablation_compile_rk(program, projection, label, n, directory)
    call = TracedSlotCall(program.f, program.metadata)
    driver = (state, seed, counts) -> traced_owned_batch(call, projection, state, seed, counts, n)
    inputs = () -> traced_slot_inputs(program.state)
    timing = @timed Reactant.compile(driver, inputs(); sync=true,
        donated_args=:none, serializable=true, TracedSlotCompiler.cpu_compile_options()...)
    ablation_save(timing.value, label, directory)
    (; label, compiled=timing.value, inputs, compile_seconds=timing.time,
       count_basis="runtime_counter",
       check=result -> begin
           all(isfinite, Array(only(result.values))) || error("nonfinite RK position")
           Int(result.counts[1])
       end)
end

function ablation_probprog_case(L, n, directory)
    density, ad, q = build_density()
    logdensity = q -> density(reshape(q, :))
    value, gradient = RK.ad_value_and_gradient(ad, q)
    inputs = () -> (ReactantRNG(Reactant.to_rarray(UInt64[91,92])),
        logdensity, Reactant.to_rarray(reshape(copy(q),1,:)),
        Reactant.to_rarray(reshape(-gradient,1,:)), Reactant.ConcreteRNumber(-value),
        Reactant.ConcreteRNumber(0.03), Reactant.to_rarray(ones(length(q))))
    driver = (args...) -> probprog_hmc_program(args..., L, n)
    timing = @timed Reactant.compile(driver, inputs(); optimize=:probprog,
        sync=true, donated_args=:none, serializable=true,
        TracedSlotCompiler.cpu_compile_options()...)
    label = "probprog_endpoint"
    ablation_save(timing.value, label, directory)
    (; label, compiled=timing.value, inputs, compile_seconds=timing.time,
       count_basis="emitted_fixed_loop",
       check=result -> begin
           all(isfinite, Array(result[1])) || error("nonfinite ProbProg position")
           L*n # ProbProg's fixed loop bound; no instrumented counter.
       end)
end

function ablation_save(compiled, label, directory)
    write(joinpath(directory, label * ".mlir"), compiled.module_string)
    write(joinpath(directory, label * ".hlo"),
        sprint(show, only(Reactant.XLA.get_hlo_modules(compiled.exec))))
end

function ablation_samples(cases, io, L, n)
    for case in cases
        result = case.compiled(case.inputs()...)
        actual_steps = case.check(result)
        0 < actual_steps <= L*n || error("invalid $(case.label) integration count")
        # Existing endpoint HMC can reject a trajectory early on divergence.
        # Preserve that behavior and report actual work, never label it n*L.
        if case.compile_seconds !== nothing
            println(io, join((case.label,L,n,actual_steps,0,"compile",case.compile_seconds,0), ',')); flush(io)
            println("ablation_compile L=",L," case=",case.label," seconds=",case.compile_seconds); flush(stdout)
        end
        println("ablation_steps L=",L," case=",case.label," steps=",actual_steps,
            " nominal=",L*n," basis=",case.count_basis); flush(stdout)
    end
    samples = Dict(case.label => Float64[] for case in cases)
    for sample in 1:7
        order = isodd(sample) ? eachindex(cases) : reverse(eachindex(cases))
        for i in order
            case = cases[i]
            inputs = case.inputs()
            timing = @timed case.compiled(inputs...)
            actual_steps = case.check(timing.value)
            0 < actual_steps <= L*n || error("invalid $(case.label) integration count")
            push!(samples[case.label], timing.time)
            println(io, join((case.label,L,n,actual_steps,sample,"execute",timing.time,timing.bytes), ',')); flush(io)
        end
    end
    for case in cases
        println("ablation_result L=",L," case=",case.label,
            " median_us=",median(samples[case.label])*1e6/n); flush(stdout)
    end
end

function endpoint_ablation(directory; n=10000, steps=(4,16))
    mkpath(directory)
    open(joinpath(directory,"samples.csv"),"w") do io
        println(io,"kernel,steps_per_transition,transitions,integration_steps,sample,phase,seconds,allocated_bytes")
        for L in steps
            output = joinpath(directory,"L$(L)"); mkpath(output)
            cases = []
            endpoint = build_fast_prototype(; L, compiler_options=(peel_loops=true,))
            push!(cases, ablation_rk_case(endpoint,"rk_endpoint",n,output))
            multinomial = build_multinomial_prototype(; L,
                source=PositionMultinomialHMCAuthoring.multinomial_hmc_state,
                source_controls=(step_f=F.leapfrog!,stepsize=0.03),
                compiler_options=(peel_loops=true,))
            push!(cases, ablation_rk_case(multinomial,"rk_multinomial",n,output))
            push!(cases, ablation_probprog_case(L,n,output))
            Base.invokelatest(ablation_samples,Tuple(cases),io,L,n)
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: endpoint_ablation.jl /absolute/output-directory")
    endpoint_ablation(only(ARGS))
end
