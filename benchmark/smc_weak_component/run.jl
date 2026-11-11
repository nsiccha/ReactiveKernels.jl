# Synthetic weak-component mixture: single-pilot and composed-pilot SMC.
# julia --project=<consumer environment> run.jl [particles moves single|two-pilot [output-directory]]
# SMC_SEEDS=41,42,43 selects independent pilot/production replicates.
using ReactiveKernels: @kernel
using ReactiveKernelsPPL, SequentialMonteCarlo, WarmupHMC, LogDensityProblems
using Enzyme, ADTypes, Random, Statistics, Test, Serialization
using ReactiveKernels
println("LIBENZYME ", Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
println("JULIA_COMMIT ", Base.GIT_VERSION_INFO.commit_short)
@assert Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt) !== nothing

@kernel toy_vector_mixture(y, other, means, scales, observation_sd, logweights) = begin
    total_sd = sqrt.(scales .^ 2 .+ observation_sd .^ 2)
    residual = (vcat(y, other) .- means) ./ total_sd
    scores = vec(sum(-0.5 .* residual .^ 2 .- log.(total_sd) .- 0.5 * log(2 * pi); dims = 1)) .+ logweights
    peak = maximum(scores)
    return peak + log(sum(exp.(scores .- peak)))
end
const TOY_EMPTY_COMPONENT = @rkppl begin
    first_mean ~ Ordered(Normal(prior_location, first_prior_sd), K)
    other_mean[1:1, 1:K] .~ Normal(prior_location, other_prior_sd)
    scales[1:2, 1:K] .~ Exponential.(scale_prior)
    weights ~ Dirichlet(alpha)
    means = vcat(permutedims(first_mean), other_mean)
    logweights = log.(weights)
    @plate for i in eachindex(y)
        y[i] ~ LogDensity(toy_vector_mixture, other[i], means, scales, observation_sd, logweights)
    end
end

function main()
    particles = isempty(ARGS) ? 4096 : parse(Int, ARGS[1])
    moves = length(ARGS) < 2 ? 8 : parse(Int, ARGS[2])
    route = length(ARGS) < 3 ? "two-pilot" : ARGS[3]
    route in ("single", "two-pilot") || error("route must be single or two-pilot")
    particles > 0 && moves > 0 || error("particles and moves must be positive")
    route == "two-pilot" && !iseven(moves) && error("two-pilot needs an even total move count")
    output = length(ARGS) < 4 ? nothing : mkpath(ARGS[4])
    save_result(name, value) = isnothing(output) ? nothing :
        serialize(joinpath(output, route * "_" * name * ".jls"), value)
    println("EXPERIMENT route=", route)
    unit = [1.0, 0.02]
    rng = Xoshiro(17)
    half = 0.08 .* unit .* randn(rng, 2, 6)
    x = hcat(half, -half)
    inputs = (; K = 2, prior_location = 0.0, first_prior_sd = 4.0 * unit[1],
        other_prior_sd = 4.0 * unit[2], scale_prior = 0.3 .* unit .+ zeros(2, 2),
        alpha = fill(2.0, 2), observation_sd = 0.02 .* unit,
        other = [x[2:2, i] for i in axes(x, 2)])
    plan = TOY_EMPTY_COMPONENT(; inputs...) | (; y = vec(x[1, :]))
    built = build_kernel(plan)
    init = unconstrain(built.layout, (; first_mean = [-0.1, 0.1],
        other_mean = zeros(1, 2), scales = 0.1 .* unit .+ zeros(2, 2), weights = [0.5, 0.5]))
    println("LAYOUT ",[(e.name,e.offset,e.size,e.transform) for e in built.layout.entries])
    proposal_rng = Xoshiro(5101)
    U = reduce(hcat, [begin
        w = randexp(proposal_rng, 2); w ./= sum(w)
        unconstrain(built.layout, (; first_mean = sort(4 .* randn(proposal_rng, 2)),
            other_mean = 0.08 .* randn(proposal_rng, 1, 2),
            scales = 0.3 .* unit .* randexp(proposal_rng, 2, 2), weights = w))
    end for _ in 1:1024])
    ref = SMCReference(vec(mean(U; dims = 2)), 1.5 .* vec(std(U; dims = 2)))
    println("REFERENCE location=",ref.location," scale=",ref.scale)
    # Negating all observations/means and reversing component order leaves
    # this symmetric dataset and every prior invariant. It exchanges which
    # ordered component carries most observations, so that event has mass 1/2.
    q = prepare_query(built, plan, :sampler)
    @testset "Exact reflection symmetry" begin
        for j in 1:8
            c = constrain(built.layout, U[:, j])
            reflected = (; first_mean = -reverse(c.first_mean),
                other_mean = -reverse(c.other_mean; dims = 2),
                scales = reverse(c.scales; dims = 2), weights = reverse(c.weights))
            reflected_u = unconstrain(built.layout, reflected)
            @test Base.invokelatest(q, U[:, j]) ≈
                Base.invokelatest(q, reflected_u) rtol = 1e-10
            reference_score(u) = -0.5 * sum(abs2, (u .- ref.location) ./ ref.scale)
            println("SYMMETRY sample=", j, " target_delta=",
                Base.invokelatest(q,U[:,j])-Base.invokelatest(q,reflected_u),
                " reference_delta=",reference_score(U[:,j])-reference_score(reflected_u))
        end
    end
    target = prepare_smc(built, plan, init; backend = AutoEnzyme(; mode = Enzyme.Reverse),
        model_id = "symmetric-empty-component-v2", binding_id = "twelve-two-dimensional-v2")
    gradient_query = prepare_sampler(built, plan, init; backend = AutoEnzyme(; mode=Enzyme.Reverse))
    max_scaled_error = 0.0
    for u in [init, (U[:,j] for j in 1:8)...]
        _, g = LogDensityProblems.logdensity_and_gradient(gradient_query, u)
        for k in eachindex(u)
            h = cbrt(eps(Float64))*max(abs(u[k]),1.0)
            plus, minus = copy(u), copy(u)
            plus[k] += h; minus[k] -= h
            fd = (gradient_query(plus)-gradient_query(minus))/(2h)
            max_scaled_error=max(max_scaled_error,abs(g[k]-fd)/max(abs(g[k]),abs(fd),1.0))
        end
    end
    println("GRADIENT max_scaled_error=",max_scaled_error)
    @assert max_scaled_error < 1e-5
    temperatures = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, .01, .05, .1, .2, .4, .6, .8,
        .9, .95, .975, .99, .995, .999, 1.0]
    println("VERSIONS RKPPL=", pkgversion(ReactiveKernelsPPL), " WHMC=", pkgversion(WarmupHMC),
        " Enzyme=", pkgversion(Enzyme), " SMC=", pkgversion(SequentialMonteCarlo),
        " Julia=", VERSION, " particles=", particles, " moves=", moves)
    for seed in parse.(Int, split(get(ENV, "SMC_SEEDS", "41,42,43"), ","))
        # These starts and both pilot RNG streams are fixed before production.
        # Each pilot uses the same normalized reference and the same target.
        starts = [unconstrain(built.layout,(;
            first_mean=[0.0,4.0],other_mean=zeros(1,2),
            scales=hcat(0.08 .* unit,0.3 .* unit),weights=[0.8,0.2])),
            unconstrain(built.layout,(;
            first_mean=[-4.0,0.0],other_mean=zeros(1,2),
            scales=hcat(0.3 .* unit,0.08 .* unit),weights=[0.2,0.8]))]
        tuning_s = @elapsed schedules = if route == "two-pilot"
            [tune_smc(target;
                rng=Xoshiro(100000+10seed+j),init=start,reference=ref,
                pilot=:adaptive,pilot_steps=256,temperatures,grid_start=:independent,
                options=SMCOptions(; metric=1.5 .* vec(std(U;dims=2)),
                    moves=div(moves,2),max_tree_depth=7,nthreads=1))
                for (j,start) in enumerate(starts)]
        else
            [tune_smc(target; rng=Xoshiro(100000+seed),init,reference=ref,
                pilot=:adaptive,pilot_steps=256,temperatures,grid_start=:sequential,
                options=SMCOptions(; metric=1.5 .* vec(std(U;dims=2)),
                    moves,max_tree_depth=7,nthreads=1))]
        end
        tuning=schedules[1]
        save_result("schedules_$(seed)", schedules)
        for (j,schedule) in enumerate(schedules)
            for (beta,kernel) in zip(schedule.temperatures,schedule.kernels)
                println("KERNEL seed=",seed," pilot=",j," beta=",beta,
                    " stepsize=",kernel.stepsize," metric=",kernel.metric)
            end
        end
        sampling_s = @elapsed begin
            fit = initialize_smc(target; rng=Xoshiro(seed), particles, reference=ref,
                options=SMCOptions(; tuning, nthreads=1, history_limit=1024))
            summarize = function (state)
                mass_first = 0.0
                for j in axes(state.positions,2)
                    c = constrain(built.layout,state.positions[:,j])
                    mm = vcat(permutedims(c.first_mean),c.other_mean)
                    ss = sqrt.(c.scales.^2 .+ inputs.observation_sd.^2)
                    m1 = 0.0
                    for i in axes(x,2)
                        z = vec(sum(-0.5 .* ((x[:,i].-mm)./ss).^2 .- log.(ss);dims=1)).+log.(c.weights)
                        z = exp.(z.-maximum(z)); m1 += z[1]/sum(z)
                    end
                    mass_first += state.weights[j]*(m1 > size(x,2)/2)
                end
                return mass_first
            end
            println("PATH seed=",seed," beta=",fit.temperature," mode_mass=",summarize(fit))
            while fit.temperature < 1
                fit = smc_step(fit)
                if route == "two-pilot"
                    # Rejuvenation retains this schedule for the next tempering
                    # step: the order is A,B then B,A. Each stage uses both, with
                    # a total of `moves` transitions and no particle-based choice.
                    fit = rejuvenate_smc(fit;tuning=schedules[isodd(fit.stage) ? 2 : 1],
                        moves=div(moves,2))
                end
                stage_history = route == "two-pilot" ? fit.history[end-1:end] : fit.history[end:end]
                println("PATH seed=",seed," beta=",fit.temperature," mode_mass=",summarize(fit),
                    " divergences=",sum(h.divergences for h in stage_history))
                flush(stdout)
            end
        end
        save_result("fit_$(seed)", smc_checkpoint(fit))
        for h in fit.history
            println("STAGE seed=",seed," kind=",h.kind," beta=",h.temperature," divergences=",h.divergences," tuning_index=",h.tuning_index," ancestors=",h.surviving_ancestors," coordinate_sd=",h.coordinate_sd)
        end
        event = zeros(particles)
        weak = zeros(particles)
        weight_mean = 0.0
        midpoint_mean = 0.0
        other_mean_sum = 0.0
        scale_logdiff = zeros(2)
        for j in 1:particles
            c = constrain(built.layout, fit.positions[:, j])
            weight_mean += fit.weights[j]*c.weights[1]
            midpoint_mean += fit.weights[j]*sum(c.first_mean)/2
            other_mean_sum += fit.weights[j]*sum(c.other_mean)
            scale_logdiff .+= fit.weights[j].*(log.(c.scales[:,1]).-log.(c.scales[:,2]))
            means = vcat(permutedims(c.first_mean), c.other_mean)
            sd = sqrt.(c.scales .^ 2 .+ inputs.observation_sd .^ 2)
            mass = zeros(2)
            for i in axes(x, 2)
                score = vec(sum(-0.5 .* ((x[:, i] .- means) ./ sd) .^ 2 .- log.(sd); dims = 1)) .+ log.(c.weights)
                w = exp.(score .- maximum(score)); mass .+= w ./ sum(w)
            end
            event[j] = mass[1] > size(x, 2) / 2
            weak[j] = minimum(mass) <= 1
        end
        println("MOMENTS seed=",seed," weight_mean=",weight_mean,
            " midpoint_mean=",midpoint_mean," other_mean_sum=",other_mean_sum,
            " scale_logdiff=",scale_logdiff)
        println("RESULT seed=", seed, " mode_mass=", sum(fit.weights .* event),
            " exact=0.5 weak_mass=", sum(fit.weights .* weak), " temperature=", fit.temperature,
            " stages=", fit.stage, " ancestors=", last(fit.history).surviving_ancestors,
            " divergences=", sum(h.divergences for h in fit.history),
            " divergence_fraction=", sum(h.divergences for h in fit.history) / (particles * moves * fit.stage),
            " final_divergences=", sum(h.divergences for h in
                (route == "two-pilot" ? fit.history[end-1:end] : fit.history[end:end])),
            " tuning_s=", tuning_s, " production_s=", sampling_s)
        flush(stdout)
    end
end
main()
