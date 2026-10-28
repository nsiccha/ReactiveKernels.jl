using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# The legacy raw IR fixtures retain every authored prior, including an
# auxiliary parameter whose response slot is replaced by fixed data.
function _aux_data_fixture(family, n; surface=false)
    student = family === :StudentT
    plan = student ? _student_plan(n) : _zip_plan(n)
    student || (plan.columns[:x] = repeat([0.15, 0.35, 0.65], cld(n,3))[1:n])
    data = (; x=plan.columns[:x], y=plan.columns[:y])
    if surface
        ast = student ? quote
            a ~ Normal(0,1)
            b ~ Normal(0,1)
            sigma ~ Exponential(1)
            nu ~ Gamma(2,0.1)
            mu = a .+ b .* x
            y .~ StudentT.(x,mu,sigma)
        end : quote
            a ~ Normal(0,1)
            b ~ Normal(0,1)
            zi ~ Beta(2,2)
            eta = a .+ b .* x
            y .~ ZeroInflatedPoisson.(exp.(eta),x)
        end
        plan = bind_data(lower_rkppl(ast,data; conditioned=(:y,)),data)
    else
        r = only(plan.responses)
        args = student ? (; nu=:x) : (; zi=:x)
        plan.responses[1] = LikelihoodSpec(r.family,r.link,r.response,r.predictor,
            r.scale,r.weights,r.evidence,r.label,r.trials,r.range; args...)
    end
    validate_plan(plan)
    built = build_kernel(plan)
    values = student ? (surface ? (; a=0.2,b=-0.3,sigma=0.7,nu=2.3) :
        (; mu=[0.2,-0.3],sigma=0.7,nu=2.3)) :
        (surface ? (; a=0.2,b=-0.3,zi=0.4) : (; eta=[0.2,-0.3],zi=0.4))
    u = unconstrain(built.layout,values)
    function parts(w)
        q = constrain(built.layout,w)
        coefs = surface ? [q.a,q.b] : (student ? q.mu : q.eta)
        mu = coefs[1] .+ coefs[2] .* data.x
        prior = sum(logpdf.(Normal(),coefs))
        likelihood = if student
            prior += logpdf(Exponential(),q.sigma) + logpdf(Gamma(2,0.1),q.nu)
            sum(logpdf.(LocationScale.(mu,q.sigma,TDist.(data.x)),data.y))
        else
            prior += logpdf(Beta(2,2),q.zi)
            sum(eachindex(data.y)) do i
                p, rate, y = data.x[i], exp(mu[i]), data.y[i]
                y == 0 ? log(p + (1-p)*pdf(Poisson(rate),0)) :
                    log1p(-p)+logpdf(Poisson(rate),y)
            end
        end
        jacobian = student ? log(q.sigma)+log(q.nu) : log(q.zi)+log1p(-q.zi)
        (; likelihood, prior, jacobian)
    end
    oracle = w -> begin p=parts(w); p.likelihood+p.prior+p.jacobian end
    kernel = prepare_query(built,plan,:sampler)
    (; family,surface,plan,built,u,kernel,parts,oracle)
end

function _aux_cell_scale_fixture(n)
    # Preserve the original transform and the cell prior; theta is not a
    # coefficient, and its normal prior is conditional on mu and tau.
    ast = Expr(:block,:(mu ~ Normal(0,5)),:(tau ~ HalfNormal(5)),
        :(_t2 = theta .+ 1),
        Expr(:macrocall,Symbol("@plate"),LineNumberNode(1),
            Expr(:for,:(i = eachindex(y)),Expr(:block,
                :(theta[i] ~ Normal(mu,tau)),
                :(y[i] ~ Normal.(theta[i],_t2[i]))))))
    data = (; y=repeat([0.2,0.5,0.8],cld(n,3))[1:n])
    plan = bind_data(lower_rkppl(ast,data; conditioned=(:y,)),data)
    built = build_kernel(plan)
    u = unconstrain(built.layout,(; mu=0.2,tau=0.8,
        theta=repeat([-0.2,0.1,0.4],cld(n,3))[1:n]))
    function prior(w)
        q=constrain(built.layout,w)
        logpdf(Normal(0,5),q.mu)+logpdf(truncated(Normal(0,5),0,Inf),q.tau)+
            sum(logpdf.(Normal(q.mu,q.tau),q.theta))+log(q.tau)
    end
    function oracle(w)
        q=constrain(built.layout,w)
        prior(w)+sum(eachindex(data.y)) do i
            s=q.theta[i]+1
            s > 0 ? logpdf(Normal(q.theta[i],s),data.y[i]) : -Inf
        end
    end
    kernel = prepare_query(built,plan,:sampler)
    (; plan,built,u,kernel,prior,oracle)
end

@testset "raw auxiliary data slots preserve response and unused priors" begin
    for family in (:StudentT,:ZIP), surface in (false,true)
        f=_aux_data_fixture(family,9; surface)
        _distributional_check(f,f.oracle,f.u)
        p=f.parts(f.u)
        for (query,expected) in ((:likelihood,p.likelihood),(:prior,p.prior),
                (:log_jacobian,p.jacobian),(:sampler,p.likelihood+p.prior+p.jacobian))
            k=prepare_query(f.built,f.plan,query)
            @test Base.invokelatest(k,f.u) ≈ expected
        end
        @test f.built.layout.total == (family === :StudentT ? 4 : 3)
        if !surface
            r=only(f.plan.responses)
            @test (family === :StudentT ? r.nu : r.zi) === :x
            # Binding an unbound raw IR slot resolves the column directly.
            p=StructuralPlan(f.plan.responses,f.plan.predictors,f.plan.population_priors,
                f.plan.parameters,f.plan.assignments,
                Dict{Symbol,AbstractVector}(),0)
            @test bind_data(p,f.plan.columns).columns[:x] == f.plan.columns[:x]
        end
    end
end

@testset "fixed auxiliary data have ordinary support and axis diagnostics" begin
    for (family,values) in ((:StudentT,([0.0,1,2],[-1.0,1,2],[NaN,1,2],[Inf,1,2])),
            (:ZIP,([-0.1,0.2,0.5],[1.1,0.2,0.5],[NaN,0.2,0.5],[Inf,0.2,0.5])))
        for x in values
            f=_aux_data_fixture(family,3)
            f.plan.columns[:x]=x
            @test_throws ContractValidationError validate_plan(f.plan)
        end
        f=_aux_data_fixture(family,3)
        f.plan.columns[:x]=[0.2,0.3]
        @test_throws ContractValidationError validate_plan(f.plan)
        f=_aux_data_fixture(family,3)
        f.plan.columns[:x]=["a","b","c"]
        @test_throws ContractValidationError validate_plan(f.plan)
    end
end

@testset "transformed cell-latent scale retains density Jacobian and lazy reverse" begin
    f=_aux_cell_scale_fixture(3)
    @test f.built.layout.total == 5
    _distributional_check(f,f.oracle,f.u)
    for theta in ([-1.0,0.1,0.4],[-1.2,0.1,0.4],[-1.0,-1.2,-1.5])
        invalid=unconstrain(f.built.layout,(; mu=0.2,tau=0.8,theta))
        @test Base.invokelatest(f.kernel,invalid) == -Inf
        ad=Base.invokelatest(prepare_ad,f.kernel,AutoEnzyme(; mode=Enzyme.Reverse),invalid;
            active=:unconstrained)
        value,gradient=Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,ad,similar(invalid),invalid)
        @test value == -Inf
        # Valid cells still contribute. Invalid cells retain their full
        # conditional prior, with no derivative from their likelihood arm.
        oracle=w->begin
            q=constrain(f.built.layout,w)
            f.prior(w)+sum([i for i in eachindex(theta) if theta[i]>-1]; init=0.0) do i
                logpdf(Normal(q.theta[i],q.theta[i]+1),f.plan.columns[:y][i])
            end
        end
        @test gradient ≈ _distributional_findiff(oracle,invalid) rtol=2e-5 atol=2e-7
    end
end
