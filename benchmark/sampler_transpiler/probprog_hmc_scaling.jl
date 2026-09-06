# Fixed-length endpoint HMC comparator supplied by Reactant.ProbProg.
# The generated RK kernel uses multinomial selection; preserve that distinction.
include(joinpath(@__DIR__, "hmc_eight_schools.jl"))
using Statistics, TOML, SHA
using Reactant: ProbProg, ReactantRNG

function probprog_hmc_program(rng, logdensity, position, gradient, potential,
                             step_size, inverse_mass, L, n)
    _, _, _, _, state = ProbProg.mcmc_logpdf(rng, logdensity, position;
        algorithm=:HMC, step_size, inverse_mass_matrix=inverse_mass,
        initial_gradient=gradient, initial_potential_energy=potential,
        trajectory_length=0.03 * L, num_warmup=0, num_samples=n, thinning=n,
        adapt_step_size=false, adapt_mass_matrix=false)
    state.position, state.rng
end

function probprog_hmc_scaling(path; steps=(4,16), batches=(100,1000,10000))
    density, ad, q = measured(build_density, "prepare_model")
    logdensity = q -> density(reshape(q, :))
    value, gradient = RK.ad_value_and_gradient(ad, q)
    inputs() = (ReactantRNG(Reactant.to_rarray(UInt64[91,92])),
        logdensity, Reactant.to_rarray(reshape(copy(q),1,:)),
        Reactant.to_rarray(reshape(-gradient,1,:)), Reactant.ConcreteRNumber(-value),
        Reactant.ConcreteRNumber(0.03), Reactant.to_rarray(ones(length(q))))
    backend = Base.get_extension(RK, :ReactiveKernelsReactantExt).TracedSlotCompiler
    options = backend.cpu_compile_options()
    mkpath(dirname(abspath(path)))
    open(path,"w") do io
        println(io,"backend,paired_with,steps_per_transition,transitions,gradient_evaluations,sample,phase,seconds,allocated_bytes")
        for L in steps, n in batches
            driver = (args...) -> probprog_hmc_program(args...,L,n)
            original=inputs()
            println("BEGIN probprog_compile L=",L," transitions=",n); flush(stdout)
            timing=@timed Reactant.compile(driver, original; optimize=:probprog,
                sync=true, donated_args=:none, serializable=true, options...)
            compiled=timing.value
            compile_seconds=timing.time
            println(io,join(("ProbProg HMC","ProbProg",L,n,L*n,0,"compile",timing.time,timing.bytes),','));flush(io)
            stem=splitext(path)[1]*"-L$(L)-n$(n)"
            write(stem*".mlir", compiled.module_string)
            write(stem*".hlo", sprint(show,only(Reactant.XLA.get_hlo_modules(compiled.exec))))
            result=compiled(original...)
            all(isfinite,Array(result[1])) || error("nonfinite ProbProg warmup position")
            Array(original[3]) == reshape(q,1,:) || error("ProbProg mutated caller position")
            # ProbProg advances its mutable RNG argument. Reset it outside each
            # timed call, as for native Julia RNGs.
            samples=Float64[]
            for sample in 1:7
                fresh=inputs()
                timing=@timed compiled(fresh...)
                all(isfinite,Array(timing.value[1])) || error("nonfinite ProbProg position")
                push!(samples,timing.time)
                println(io,join(("ProbProg HMC","ProbProg",L,n,L*n,sample,"execute",timing.time,timing.bytes),','));flush(io)
            end
            println("probprog_complete L=",L," transitions=",n,
                " compile_seconds=",compile_seconds," median_us=",median(samples)*1e6/n,
                " mlir_bytes=",sizeof(compiled.module_string));flush(stdout)
        end
    end
    println("probprog_csv=",abspath(path))
end

if abspath(PROGRAM_FILE)==@__FILE__
    1 <= length(ARGS) <= 3 || error("usage: probprog_hmc_scaling.jl /absolute/output.csv [L] [n]")
    steps=length(ARGS)>=2 ? (parse(Int,ARGS[2]),) : (4,16)
    batches=length(ARGS)>=3 ? (parse(Int,ARGS[3]),) : (100,1000,10000)
    probprog_hmc_scaling(ARGS[1];steps,batches)
end
