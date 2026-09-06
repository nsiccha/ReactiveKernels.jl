# Drivers around compiler output. Sampler/integrator mathematics live only in
# the two captured @kernel sources included by hmc_eight_schools.jl.
function build_native_slots(prototype)
    init = native_endpoint(F.euclidean_phasepoint,F.leapfrog!,prototype.endpoint_inputs)
    child = native_endpoint(F.euclidean_phasepoint,F.leapfrog!,prototype.endpoint_inputs)
    # The authored structural child shares read-only endpoint authorities.
    fwd = (; child.pf,child.owned,shared=init.shared)
    state = prototype.kernel(prototype.snapshot.init;
        n_steps=prototype.L,step_f=prototype.snapshot.step_f,stats_f=nothing)
    program = measured("lower_native_slots") do
        compile_native_slots(prototype.kernel,state,:step!;
            endpoints=(;init,fwd),
            effects=(step_f=(source=F.leapfrog!,controls=(stepsize=0.03,)),))
    end
    seed = deepcopy(init.owned)
    (; program,seed)
end

function reset_native_slots!(prepared)
    RK._canon_copy_endpoint!(prepared.program.contexts[2].owned,prepared.seed)
    RK._canon_copy_endpoint!(prepared.program.contexts[3].owned,prepared.seed)
    fill!(prepared.program.counts,0)
    Xoshiro(91)
end

function build_ahmc_comparator(prototype; callback_handles=true)
    density,ad=prototype.density,prototype.ad
    potential=q->density(q)
    gradient=q->RK.ad_value_and_gradient(ad,q)
    if callback_handles
        potential,gradient=CallbackHandle(potential),CallbackHandle(gradient)
    end
    hamiltonian=AdvancedHMC.Hamiltonian(
        AdvancedHMC.DiagEuclideanMetric(length(prototype.q)),
        potential,gradient)
    kernel=AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.EndPointTS}(
        AdvancedHMC.Leapfrog(0.03),AdvancedHMC.FixedNSteps(prototype.L)))
    hamiltonian,initial=AdvancedHMC.sample_init(Xoshiro(91),hamiltonian,prototype.q)
    (; hamiltonian,kernel,initial=initial.z)
end

function native_position(program)
    context=program.contexts[2]
    _,slot=RK.kernel_plan_field(slotplan(context),slotfields(context)[:pos])
    RK._canon_slot(context.owned,Val(slot))
end

function run_native_comparison(prototype; n=1000,compare=true)
    prepared=build_native_slots(prototype)
    comparator=compare ? build_ahmc_comparator(prototype) : nothing
    run_prepared_comparison(prepared,comparator,prototype.L,n)
end

# Construction returns new generated function types. Specialize this measurement
# loop on the concrete resulting program instead of dispatching through the
# dynamically constructed metadata inside the warm interval.
function run_prepared_comparison(prepared,comparator,L,n)
    program=prepared.program
    compare=comparator !== nothing
    rng=reset_native_slots!(prepared)
    measured("native_slots_first_chain") do
        slot_chain(program,rng,2)
    end
    if compare
        measured("ahmc_first_chain") do
            ahmc_chain(Xoshiro(91),comparator.hamiltonian,comparator.kernel,
                       deepcopy(comparator.initial),2)
        end
    end
    for _ in 1:7
        rng=reset_native_slots!(prepared)
        measured("native_slots_$(n)_transitions") do
            slot_chain(program,rng,n)
        end
        println("integration_steps=",program.counts[1])
        program.counts[1]==L*n || error("shortened native workload")
        all(isfinite,native_position(program)) || error("nonfinite native position")
        if compare
            rng=Xoshiro(91);point=deepcopy(comparator.initial)
            result=measured("ahmc_$(n)_transitions") do
                ahmc_chain(rng,comparator.hamiltonian,comparator.kernel,point,n)
            end
            all(isfinite,result.θ) || error("nonfinite AHMC position")
        end
    end
    println("native_position=",native_position(program));flush(stdout)
    (; prepared,comparator)
end

function build_fast_prototype(; compiler_options=NamedTuple())
    density,ad,q=measured(build_density,"prepare_model")
    potential,gradient=CallbackHandle(Potential(density)),CallbackHandle(Gradient(ad))
    endpoint_inputs=(potential,gradient,Diagonal(ones(length(q))),q,zeros(length(q)))
    endpoint=measured("prepare_native_endpoint") do
        native_endpoint(F.euclidean_phasepoint,F.leapfrog!,endpoint_inputs)
    end
    parent=measured("prepare_native_factory") do
        fast_native_factory(H.hmc_state,NativePoint(endpoint);
            n_steps=4,step_f=RK.partial(F.leapfrog!;stepsize=0.03),stats_f=nothing)
    end
    program=measured("lower_fast_native") do
        compile_native_slots(parent.kernel,parent.state,:step!;
            endpoints=parent.endpoints,effects=parent.effects,compiler_options...)
    end
    names=keys(parent.endpoints)
    println("children=",names)
    names==(:init,:fwd) || error("benchmark expects init then fwd")
    seed=deepcopy(program.contexts[2].owned)
    comparator=build_ahmc_comparator((;density,ad,q,L=4))
    (; prepared=(;program,seed),comparator)
end
