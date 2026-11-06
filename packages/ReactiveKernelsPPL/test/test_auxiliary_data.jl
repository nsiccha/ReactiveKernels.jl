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

# A value fixed by bound data validates at bind on every route the surface lowers
# it by: a bare column, a computed argument, a named definition and an offsets-only
# predictor. Refusal is the documented contract of `LikelihoodSpec`'s nu and zi.
_aux_message(ast,data) = try
    bind_data(lower_rkppl(ast,data; conditioned=(:y,)),data)
    nothing
catch e
    e isa ContractValidationError ? sprint(showerror,e) : rethrow()
end
function _aux_likelihood(ast,data,u)
    plan=bind_data(lower_rkppl(ast,data; conditioned=(:y,)),data)
    built=build_kernel(plan)
    Base.invokelatest(prepare_query(built,plan,:likelihood),fill(u,built.layout.total))
end
_aux_zip_ast(arg,defs...) = :(begin
    a ~ Normal(0,1)
    $(defs...)
    y .~ ZeroInflatedPoisson.(exp.(a .* x),$arg)
end)
_aux_student_ast(arg,defs...) = :(begin
    a ~ Normal(0,1)
    $(defs...)
    y .~ StudentT.($arg,a .* x,1.0)
end)
_aux_zip_density(x,y,z,u) = sum(eachindex(y)) do i
    rate=exp(u*x[i])
    y[i] == 0 ? log(z[i]+(1-z[i])*exp(-rate)) : log1p(-z[i])+logpdf(Poisson(rate),y[i])
end
_aux_student_density(x,y,nu,u) = sum(logpdf.(TDist.(nu),y .- u .* x))

@testset "a fixed nu or zi value validates at bind on every surface route" begin
    x=collect(range(-1.0,1.0; length=6))
    counts=[0,2,0,1,3,0]
    y=[-0.4,0.1,0.3,0.2,0.7,1.1]
    u=0.2
    first_row(v,rest) = [v; fill(rest,5)]
    # (spelling, its value given the bound column, a column that stays in
    # support, a column that leaves it, how the refusal names the value)
    zi_routes=(
        (_aux_zip_ast(:z),identity,fill(0.3,6),first_row(1.5,0.3),"z"),
        (_aux_zip_ast(:z),identity,fill(0.3,6),first_row(-0.2,0.3),"z"),
        (_aux_zip_ast(:(z ./ 2)),v->v./2,fill(0.3,6),first_row(4.0,0.3),"`z ./ 2`"),
        (_aux_zip_ast(:zz,:(zz=1.5 .* z)),v->1.5.*v,fill(0.3,6),first_row(4.0,0.3),"zz"),
        (_aux_zip_ast(:zz,:(zz=z)),identity,fill(0.3,6),first_row(1.5,0.3),"zz"),
        (_aux_zip_ast(:zz,:(zz=z .- 0.1)),v->v.-0.1,fill(0.3,6),first_row(0.05,0.3),"zz"),
        (_aux_zip_ast(:(exp.(zz)),:(zz=z)),v->exp.(v),fill(-1.0,6),first_row(1.0,-1.0),"zz"),
    )
    for (ast,value,good,bad,name) in zi_routes
        data=(; x,y=counts,z=good)
        @test _aux_likelihood(ast,data,u) ≈ _aux_zip_density(x,counts,value(good),u)
        @test occursin("[y_resp] zi data $name must lie in [0, 1]",
            something(_aux_message(ast,(; x,y=counts,z=bad)),""))
    end
    nu_routes=(
        (_aux_student_ast(:nuc),identity,fill(4.0,6),first_row(-1.0,4.0),"nuc"),
        (_aux_student_ast(:(nuc .- 5)),v->v.-5,fill(8.0,6),first_row(4.0,8.0),"`nuc .- 5`"),
        (_aux_student_ast(:nn,:(nn=2 .* nuc)),v->2 .*v,fill(4.0,6),first_row(-1.0,4.0),"nn"),
        (_aux_student_ast(:nn,:(nn=nuc .+ 1)),v->v.+1,fill(3.0,6),first_row(-2.0,3.0),"nn"),
    )
    for (ast,value,good,bad,name) in nu_routes
        data=(; x,y,nuc=good)
        @test _aux_likelihood(ast,data,u) ≈ _aux_student_density(x,y,value(good),u)
        @test occursin("[y_resp] nu data $name must be finite positive numerics",
            something(_aux_message(ast,(; x,y,nuc=bad)),""))
    end
    # A column reads as authored, in the response's plan and in integer form.
    ast=_aux_student_ast(:nuc)
    @test only(lower_rkppl(ast,(; x,y,nuc=fill(4,6)); conditioned=(:y,)).responses).nu === :nuc
    @test _aux_likelihood(ast,(; x,y,nuc=fill(4,6)),u) ≈ _aux_student_density(x,y,fill(4,6),u)
    # Fixed values must be finite, as a literal and a scalar column must.
    @test occursin("nu data nuc must be finite real numerics",
        something(_aux_message(ast,(; x,y,nuc=first_row(Inf,4.0))),""))
    ast=_aux_zip_ast(:z)
    @test occursin("zi data z must be finite real numerics",
        something(_aux_message(ast,(; x,y=counts,z=first_row(NaN,0.3))),""))
    # A Bool or integer probability column keeps its ordinary meaning.
    @test _aux_likelihood(ast,(; x,y=counts,z=Bool[1,0,1,0,0,1]),u) ≈
        _aux_zip_density(x,counts,Bool[1,0,1,0,0,1],u)
    # The scalar and per-row forms refuse alike.
    @test occursin("zi data z must lie in [0, 1]",
        something(_aux_message(ast,(; x,y=counts,z=2.0)),""))
    # ZIB shares the slot.
    @test occursin("[y_resp] zi data z must lie in [0, 1]",
        something(_aux_message(:(begin
            a ~ Normal(0,1)
            y .~ ZeroInflatedBinomial.(8,logistic.(a .* x),z)
        end),(; x,y=counts,z=first_row(1.5,0.3))),""))
    # A plate cell reads the same column.
    plate_zi=quote
        a ~ Normal(0,1)
        @plate for i in eachindex(y)
            y[i] ~ ZeroInflatedPoisson(exp(a*x[i]),z[i])
        end
    end
    @test _aux_likelihood(plate_zi,(; x,y=counts,z=fill(0.3,6)),u) ≈
        _aux_zip_density(x,counts,fill(0.3,6),u)
    @test occursin("[y_resp] zi data z must lie in [0, 1]",
        something(_aux_message(plate_zi,(; x,y=counts,z=first_row(1.5,0.3))),""))
    plate_nu=quote
        a ~ Normal(0,1)
        @plate for i in eachindex(y)
            y[i] ~ StudentT(nuc[i],a*x[i],1.0)
        end
    end
    @test occursin("[y_resp] nu data nuc must be finite positive numerics",
        something(_aux_message(plate_nu,(; x,y,nuc=first_row(-1.0,4.0))),""))
    # Only the rows a response observes must lie in support.
    gappy=Union{Missing,Int}[missing; counts[2:end]]
    bad=first_row(1.5,0.3)
    @test isfinite(_aux_likelihood(ast,(; x,y=gappy,z=bad),u))
    @test occursin("zi data z must lie in [0, 1]",
        something(_aux_message(ast,(; x,y=Union{Missing,Int}[counts[1:end-1]; missing],z=bad)),""))
    # The scale slot's own computed routes refuse by what the author wrote.
    normal(arg,defs...) = :(begin
        a ~ Normal(0,1)
        $(defs...)
        y .~ Normal.(a .* x,$arg)
    end)
    @test occursin("per-observation scale `w .- 3` must be finite positive numerics",
        something(_aux_message(normal(:(w .- 3)),(; x,y,w=fill(2.0,6))),""))
    @test occursin("per-observation scale sd must be finite positive numerics",
        something(_aux_message(normal(:sd,:(sd=w .- 3)),(; x,y,w=fill(2.0,6))),""))
    @test _aux_likelihood(normal(:sd,:(sd=w .- 3)),(; x,y,w=fill(5.0,6)),u) ≈
        sum(logpdf.(Normal.(u .* x,2.0),y))
    # A design that leaves the response absent still checks the values it
    # binds, and has no value to check for a slot that reads the response.
    fit=Dict{Symbol,Any}(:x=>x,:y=>counts,:z=>fill(0.3,6))
    rebound(ast,design) = bind_data(bind_data(lower_rkppl(ast,fit; conditioned=(:y,)),fit),design)
    @test ReactiveKernelsPPL._absent_observations(
        rebound(ast,Dict{Symbol,Any}(:x=>x,:z=>fill(0.3,6)))) == [:y]
    @test_throws ContractValidationError rebound(ast,Dict{Symbol,Any}(:x=>x,:z=>first_row(1.5,0.3)))
    for definition in (:(zz=0.1 .+ 0 .* y),:(zz=y .- y .+ 0.1),:(zz=0.1 .* y))
        @test ReactiveKernelsPPL._absent_observations(
            rebound(_aux_zip_ast(:zz,definition),Dict{Symbol,Any}(:x=>x,:z=>fill(0.3,6)))) == [:y]
    end
    # A value that reads a sampled name is only known per query: its lazy
    # density guard answers, at every coordinate that leaves the support.
    live=_aux_zip_ast(:(g .* z),:(g ~ Normal(0,1)))
    @test _aux_message(live,(; x,y=counts,z=first_row(4.0,0.3))) === nothing
    plan=bind_data(lower_rkppl(live,(; x,y=counts,z=first_row(4.0,0.3)); conditioned=(:y,)),
        (; x,y=counts,z=first_row(4.0,0.3)))
    built=build_kernel(plan)
    k=prepare_query(built,plan,:likelihood)
    @test Base.invokelatest(k,unconstrain(built.layout,(; a=u,g=1.0))) == -Inf
    @test isfinite(Base.invokelatest(k,unconstrain(built.layout,(; a=u,g=0.1))))
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
