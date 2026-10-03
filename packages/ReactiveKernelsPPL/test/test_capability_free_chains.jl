using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_free_chain(subjects)
    data = Dict{Symbol,Any}(
        :subj=>repeat(collect(1:subjects); inner=2),
        :time=>repeat([0.0,5.0]; outer=subjects),
        :dsubj=>collect(1:subjects), :dtime=>zeros(subjects),
        :damt=>fill(50.0,subjects), :age=>collect(range(0.2,0.7; length=subjects)),
        :dv=>fill(0.5,2subjects))
    ast = quote
        sigma ~ Exponential(1)
        b0_vc ~ Normal(0,1)
        b1_vc ~ Normal(0,1)
        b0_k10 ~ Normal(0,1)
        b0_k12 ~ Normal(0,1)
        b0_k21 ~ Normal(0,1)
        b0_ka ~ Normal(0,1)
        log_Vc = b0_vc .+ b1_vc .* age
        log_k10 = b0_k10
        log_k12 = b0_k12
        log_k21 = b0_k21
        log_ka = b0_ka
        pk_sched = linear_pk_schedule(obs=(:subj,:time),dose=(:dsubj,:dtime,:damt))
        read_locs = linear_pk_read_locs(pk_sched,log_Vc,log_k10,log_k12,log_k21,log_ka)
        conc = read_locs[pk_sched.obs_map]
    end
    bound = bind_data(lower_rkppl(ast,data; conditioned=keys(data)),data)
    built = build_kernel(bound)
    u = unconstrain(built.layout,(; sigma=0.8,b0_vc=0.5,b1_vc=0.2,
        b0_k10=-0.3,b0_k12=-0.2,b0_k21=-0.1,b0_ka=0.4))
    sampler = prepare_sampler(built,bound,u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    collected = Base.invokelatest(prepare,built.spec;
        have=ReactiveKernelsPPL._query_have(bound),want=:conc,
        bound=ReactiveKernelsPPL._query_bound(bound))
    function oracle(v)
        th = constrain(built.layout,v)
        return logpdf(Exponential(),th.sigma)+log(th.sigma)+
            sum(logpdf(Normal(),getproperty(th,k)) for k in keys(th) if k !== :sigma)
    end
    # The same ordinary chain with an observation exposes the existing PK
    # calculation. Removing that observation must preserve the collected value.
    observed_ast = deepcopy(ast)
    push!(observed_ast.args,:(dv .~ Normal.(conc,sigma)))
    observed_bound = bind_data(lower_rkppl(observed_ast,data; conditioned=keys(data)),data)
    observed = build_kernel(observed_bound)
    observed_collected = Base.invokelatest(prepare,observed.spec;
        have=ReactiveKernelsPPL._query_have(observed_bound),want=:conc,
        bound=ReactiveKernelsPPL._query_bound(observed_bound))
    return (; data,bound,built,u,sampler,collected,oracle,observed_collected)
end

@testset "observation-free schedule chains keep priors and collected values" begin
    fx = _cap_free_chain(3)
    @test isempty(fx.bound.responses)
    @test isempty(only(fx.bound.kernel_plates).obs)
    @test Base.invokelatest(prepare_query(fx.built,fx.bound,:likelihood),fx.u) == 0
    @test Base.invokelatest(prepare_query(fx.built,fx.bound,:pointwise),fx.u) == (;)
    @test Base.invokelatest(fx.sampler.kernel,fx.u) ≈ fx.oracle(fx.u)
    value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
    @test value ≈ fx.oracle(fx.u)
    expected_gradient = copy(-fx.u)
    sigma_offset = only(e.offset for e in fx.built.layout.entries if e.name === :sigma)
    expected_gradient[sigma_offset] = 1-exp(fx.u[sigma_offset])
    @test gradient ≈ expected_gradient rtol=1e-12
    @test Base.invokelatest(fx.collected,fx.u) ≈ Base.invokelatest(fx.observed_collected,fx.u)
    @test all(isfinite,Base.invokelatest(fx.collected,fx.u))
end
