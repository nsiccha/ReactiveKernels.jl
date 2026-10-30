include(joinpath(@__DIR__,"multinomial_eight_schools.jl"))

# One row per timed batch. Data work is outside timing; source count
# instrumentation is outside the timed gradient callback.
function scaling_row(io,backend,L,n,sample,timing;phase="execute",paired_with=backend)
    println(io,join((backend,paired_with,L,n,L*n,sample,phase,timing.time,timing.bytes),','))
    flush(io)
end

function native_scaling(io,prototype,batches)
    prepared=prototype.prepared; comparator=prototype.comparator; L=prototype.L
    rng=reset_native_slots!(prepared)
    slot_chain(prepared.program,rng,2)
    ahmc_chain(Xoshiro(91),comparator.hamiltonian,comparator.kernel,
        deepcopy(comparator.initial),2)
    for n in batches, sample in 1:7
        rng=reset_native_slots!(prepared)
        timing=@timed slot_chain(prepared.program,rng,n)
        prepared.program.counts[1]==L*n || error("shortened native workload")
        all(isfinite,native_position(prepared.program)) || error("nonfinite native position")
        scaling_row(io,"native",L,n,sample,timing)
        rng=Xoshiro(91);point=deepcopy(comparator.initial)
        timing=@timed ahmc_chain(rng,comparator.hamiltonian,comparator.kernel,point,n)
        all(isfinite,timing.value.θ) || error("nonfinite AHMC position")
        scaling_row(io,"AdvancedHMC",L,n,sample,timing;paired_with="native")
    end
end

function multinomial_scaling(path;batches=(100,1000,10000),steps=(4,16),
        prototype_options=NamedTuple())
    open(path,"w") do io
        println(io,"backend,paired_with,steps_per_transition,transitions,gradient_evaluations,sample,phase,seconds,allocated_bytes")
        for L in steps
            native_scaling(io,build_multinomial_prototype(;L,prototype_options...),batches)
        end
    end
    println("scaling_csv=",abspath(path))
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: multinomial_scaling.jl /absolute/output.csv")
    multinomial_scaling(only(ARGS))
end
