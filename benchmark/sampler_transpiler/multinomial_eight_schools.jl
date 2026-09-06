include(joinpath(@__DIR__,"hmc_eight_schools.jl"))
include(joinpath(@__DIR__,"multinomial_hmc_kernel.jl"))

function build_multinomial_prototype(; L=4, compiler_options=NamedTuple(),
        source=MultinomialHMCAuthoring.multinomial_hmc_state,
        source_controls=(step_fwd=RK.partial(F.leapfrog!;stepsize=0.03),
                         step_bwd=RK.partial(F.leapfrog!;stepsize=-0.03)))
    density,ad,q=measured(build_density,"prepare_model")
    potential,gradient=CallbackHandle(Potential(density)),CallbackHandle(Gradient(ad))
    endpoint_inputs=(potential,gradient,Diagonal(ones(length(q))),q,zeros(length(q)))
    endpoint=measured("prepare_native_endpoint") do
        native_endpoint(F.euclidean_phasepoint,F.leapfrog!,endpoint_inputs)
    end
    parent=measured("prepare_multinomial_factory") do
        fast_native_factory(source,NativePoint(endpoint);
            n_steps=L,source_controls...)
    end
    program=measured("lower_multinomial_native") do
        compile_native_slots(parent.kernel,parent.state,:step!;
            endpoints=parent.endpoints,effects=parent.effects,compiler_options...)
    end
    println("children=",keys(parent.endpoints))
    first(keys(parent.endpoints))===:init || error("expected init endpoint first")
    seed=deepcopy(program.contexts[2].owned)
    endpoint_comparator=build_ahmc_comparator((;density,ad,q,L))
    kernel=AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
        AdvancedHMC.Leapfrog(0.03),AdvancedHMC.FixedNSteps(L)))
    comparator=merge(endpoint_comparator,(;kernel))
    Base.mightalias(native_position(program),comparator.initial.θ) &&
        error("native and comparator initial positions alias")
    (;prepared=(;program,seed),comparator,L)
end

function check_multinomial_native(prototype;n=1000)
    program=prototype.prepared.program
    rng=reset_native_slots!(prototype.prepared)
    slot_chain(program,rng,n)
    println("diagnostic_integration_steps=",program.counts[1],
        " diagnostic_gradient_calls=",program.counts[2])
    program.counts[1]==prototype.L*n || error("incorrect integration count")
    all(isfinite,native_position(program)) || error("nonfinite native position")
    if prototype.L==0
        original_context=program.contexts[2]
        _,slot=RK.kernel_plan_field(slotplan(original_context),slotfields(original_context)[:pos])
        initial=RK._canon_slot(prototype.prepared.seed,Val(slot))
        native_position(program)==initial || error("zero-step source changed position")
    end
end

function write_multinomial_emissions(directory,native,result)
    mkpath(directory)
    open(joinpath(directory,"native.jl"),"w") do io
        println(io,"# Inspection expression; requires the prepared stores/resources/constants.")
        Base.show_unquoted(io,native.expression)
        println(io)
    end
    if result !== nothing && hasproperty(result,:compiled)
        open(joinpath(directory,"traced.jl"),"w") do io
            println(io,"# Inspection expression; requires this preparation's numeric state/metadata.")
            Base.show_unquoted(io,result.program.expression)
            println(io)
        end
        write(joinpath(directory,"kernel.mlir"),result.compiled.module_string)
    end
    println("emissions_directory=",abspath(directory))
end

if abspath(PROGRAM_FILE)==@__FILE__
    mode=isempty(ARGS) ? "native" : ARGS[1]
    n=length(ARGS)>1 ? parse(Int,ARGS[2]) : 1000
    L=length(ARGS)>2 ? parse(Int,ARGS[3]) : 4
    n>0 && (L>0 || (mode=="diagnostics" && L==0)) ||
        error("positive batch/count required (diagnostics admits zero steps)")
    prototype=build_multinomial_prototype(;L,
        compiler_options=(peel_loops=mode=="reactant",count_gradients=mode=="diagnostics"))
    result=if mode=="native"
        run_prepared_comparison(prototype.prepared,prototype.comparator,L,n)
    elseif mode=="reactant"
        run_traced_comparison(prototype;n)
    elseif mode=="diagnostics"
        check_multinomial_native(prototype;n)
    else
        error("expected native, reactant, or diagnostics")
    end
    length(ARGS)>3 && write_multinomial_emissions(ARGS[4],prototype.prepared.program,result)
end
