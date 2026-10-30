# Like-for-like fixed-length endpoint HMC: the RK endpoint source against
# AdvancedHMC `EndPointTS`. Every call returns only the final position and RNG.
include(joinpath(@__DIR__, "multinomial_eight_schools.jl"))
include(joinpath(@__DIR__, "endpoint_hmc_kernel.jl"))
using Statistics: median

function matched_native_case(prototype, n)
    prepared = prototype.prepared
    compiled = rng -> begin
        slot_chain(prepared.program, rng, n)
        native_position(prepared.program), rng
    end
    (; label="rk_native_endpoint", compiled,
       inputs=() -> (reset_native_slots!(prepared),),
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
       inputs=() -> (Xoshiro(91),deepcopy(comparator.initial)),
       count_basis="FixedNSteps",
       check=result -> begin
           all(isfinite,result[1]) || error("nonfinite AHMC position")
           prototype.L*n
       end)
end

function endpoint_samples(cases, io, L, n)
    for case in cases
        actual_steps = case.check(case.compiled(case.inputs()...))
        0 < actual_steps <= L*n || error("invalid $(case.label) integration count")
        println("endpoint_steps L=",L," case=",case.label," steps=",actual_steps,
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
        println("endpoint_result L=",L," case=",case.label,
            " median_us=",median(samples[case.label])*1e6/n); flush(stdout)
    end
end

function matched_endpoint_comparison(directory; batches=(10000,), steps=(4,16))
    mkpath(directory)
    open(joinpath(directory,"samples.csv"),"w") do io
        println(io,"kernel,steps_per_transition,transitions,integration_steps,sample,phase,seconds,allocated_bytes")
        for n in batches, L in steps
            output=joinpath(directory,"n$(n)-L$(L)"); mkpath(output)
            native=build_fast_prototype(; L,source=EndpointHMCAuthoring.hmc_state,
                compiler_options=(count_steps=false,))
            open(joinpath(output,"rk-native.jl"),"w") do source_io
                Base.show_unquoted(source_io,native.prepared.program.expression)
            end
            cases=(Base.invokelatest(matched_native_case,native,n),
                matched_ahmc_case(native,n))
            Base.invokelatest(endpoint_samples,cases,io,L,n)
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    1 <= length(ARGS) <= 2 || error("usage: matched_endpoint_comparison.jl /absolute/output-directory [scaling]")
    length(ARGS)==1 || ARGS[2]=="scaling" || error("expected scaling")
    matched_endpoint_comparison(ARGS[1]; batches=length(ARGS)==2 ? (100,1000,10000) : (10000,))
end
