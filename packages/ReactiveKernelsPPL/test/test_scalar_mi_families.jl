using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# Correct family declarations exercise the raw mi_jobs contract. The
# observation vector is packed; predictor, weights and trials keep full rows.
const _SCALAR_MI_KINDS=(:bernoulli,:poisson,:binomial,:nb2,:categorical_logit,
    :ordered_logistic,:ordinal,:categorical,:stopping)
function _scalar_mi_fixture(kind,n; selected_range=nothing, variant=:plain)
    jobs=collect(1:2:n)
    x=collect(range(-0.8,0.9; length=n))
    y=kind===:bernoulli ? mod.(collect(eachindex(jobs)),2) :
        kind in (:poisson,:binomial,:nb2) ? mod.(collect(eachindex(jobs)),3) :
        1 .+ mod.(collect(eachindex(jobs)),3)
    data=Dict{Symbol,Any}(:y=>y,:x=>x,:Jobs=>jobs,
        :wt=>[0.5+0.1mod(i,4) for i in 1:n],:trials=>[4+mod(i,3) for i in 1:n])
    ast=quote
        a ~ Normal(0,1)
        b ~ Normal(0,2)
        eta = a .+ b .* x
    end
    response = if kind===:bernoulli
        :(Bernoulli.(logistic.(eta)))
    elseif kind===:poisson
        :(Poisson.(exp.(eta)))
    elseif kind===:binomial
        :(Binomial.(trials,logistic.(eta)))
    elseif kind===:nb2
        :(NegativeBinomial2.(exp.(eta),2.2))
    elseif kind===:categorical_logit
        push!(ast.args,:(c ~ Normal(0,1)),:(d ~ Normal(0,2)),:(eta3 = c .+ d .* x))
        :(CategoricalLogit.(eta,eta3))
    elseif kind===:ordered_logistic
        push!(ast.args,:(cuts ~ Ordered(Normal(0,1),2)))
        :(OrderedLogistic.(eta,Ref(cuts)))
    elseif kind===:ordinal
        push!(ast.args,:(cuts ~ Ordered(Normal(0,1),2)))
        :(Ordinal.(Cumulative(),LogitLink(),eta,Ref(cuts)))
    elseif kind===:stopping
        push!(ast.args,:(cuts[1:2] .~ Normal.(0,1)))
        :(Ordinal.(StoppingRatio(),LogitLink(),eta,Ref(cuts)))
    elseif kind===:categorical
        # Preserve the explicitly stated unused a/b priors too.
        push!(ast.args,:(s ~ Dirichlet([1.2,2.0,1.4])))
        :(Categorical(s))
    else
        error("unknown scalar mi fixture $kind")
    end
    if variant !== :plain
        kind === :stopping || error("variants exercise stopping-ratio inputs")
        data[:disc]=[0.8+0.1mod(i,3) for i in 1:n]
        data[:E]=hcat(0.1 .* x,-0.2 .* x)
        response=:(Ordinal.(StoppingRatio(),LogitLink(),eta,Ref(cuts),disc,eachrow(E)))
        if variant === :censored
            data[:y]=min.(data[:y],2)
            data[:lo]=ones(n);data[:hi]=fill(2.0,n)
            response=:(censored.($response,lo,hi))
        end
    end
    push!(ast.args,:(y .~ weighted.($response,wt)))
    plan=lower_rkppl(ast,keys(data);conditioned=keys(data))
    r=ReactiveKernelsPPL._with(only(plan.responses);mi_jobs=:Jobs,range=selected_range)
    plan=ReactiveKernelsPPL._with(plan;responses=[r])
    bound=bind_data(plan,data)
    built=build_kernel(bound)
    values = kind===:categorical_logit ? (; a=0.2,b=-0.3,c=-0.1,d=0.15) :
        kind in (:ordered_logistic,:ordinal,:stopping) ? (; a=0.2,b=-0.3,cuts=[-0.4,0.7]) :
        kind===:categorical ? (; a=0.2,b=-0.3,s=[0.2,0.5,0.3]) : (; a=0.2,b=-0.3)
    u=unconstrain(built.layout,values)
    positions=selected_range===nothing ? eachindex(jobs) :
        findall(j->j in selected_range,jobs)
    function pointwise(w)
        q=constrain(built.layout,w)
        map(positions) do k
            j=jobs[k]; yy=data[:y][k]; eta=q.a+q.b*x[j]
            lp = if kind===:bernoulli
                logpdf(Bernoulli(cdf(Logistic(),eta)),yy)
            elseif kind===:poisson
                logpdf(Poisson(exp(eta)),yy)
            elseif kind===:binomial
                logpdf(Binomial(data[:trials][j],cdf(Logistic(),eta)),yy)
            elseif kind===:nb2
                logpdf(NegativeBinomial(2.2,2.2/(2.2+exp(eta))),yy)
            elseif kind===:categorical_logit
                rates=exp.([0.0,eta,q.c+q.d*x[j]])
                logpdf(Categorical(rates/sum(rates)),yy)
            elseif kind===:categorical
                logpdf(Categorical(q.s),yy)
            elseif kind===:stopping
                d=variant===:plain ? 1.0 : data[:disc][j]
                effects=variant===:plain ? zeros(2) : data[:E][j,:]
                p=cdf.(Logistic(),d .* (q.cuts .- eta .- effects))
                if variant===:censored
                    yy==1 ? log(p[1]) : log1p(-p[1])
                else
                    sum(log1p(-p[t]) for t in 1:yy-1;init=0.0)+
                        (yy==3 ? 0.0 : log(p[yy]))
                end
            else
                hi=yy==3 ? 1.0 : cdf(Logistic(),q.cuts[yy]-eta)
                lo=yy==1 ? 0.0 : cdf(Logistic(),q.cuts[yy-1]-eta)
                log(hi-lo)
            end
            data[:wt][j]*lp
        end
    end
    function prior_and_jacobian(w)
        q=constrain(built.layout,w)
        prior=logpdf(Normal(),q.a)+logpdf(Normal(0,2),q.b)
        jacobian=0.0
        if kind===:categorical_logit
            prior+=logpdf(Normal(),q.c)+logpdf(Normal(0,2),q.d)
        elseif kind in (:ordered_logistic,:ordinal,:stopping)
            prior+=sum(logpdf.(Normal(),q.cuts))
            kind===:stopping || (jacobian=log(q.cuts[2]-q.cuts[1]))
        elseif kind===:categorical
            prior+=logpdf(Dirichlet([1.2,2.0,1.4]),q.s)
            jacobian=sum(log.(q.s))
        end
        prior+jacobian
    end
    oracle=w->sum(pointwise(w);init=0.0)+prior_and_jacobian(w)
    kernel=prepare_query(built,bound,:sampler)
    (; kind,variant,plan,bound,built,data,saved=deepcopy(data),u,kernel,oracle,pointwise)
end

@testset "valid scalar Case-A families preserve gathered likelihoods and priors" begin
    for kind in _SCALAR_MI_KINDS
        f=_scalar_mi_fixture(kind,9)
        _distributional_check(f,f.oracle,f.u)
        pw=Base.invokelatest(prepare_query(f.built,f.bound,:pointwise),f.u)
        @test only(values(pw)) ≈ f.pointwise(f.u)
        @test length(only(values(pw))) == length(f.data[:Jobs])
        @test f.data == f.saved
        # Ranges select packed observations by their original row numbers.
        ranged=_scalar_mi_fixture(kind,9;selected_range=2:8)
        _distributional_check(ranged,ranged.oracle,ranged.u)
        @test ranged.data == ranged.saved
    end
end

@testset "packed stopping-ratio inputs gather before stage and evidence evaluation" begin
    for variant in (:gathered,:censored), selected_range in (nothing,2:8)
        f=_scalar_mi_fixture(:stopping,9;variant,selected_range)
        _distributional_check(f,f.oracle,f.u)
        pw=Base.invokelatest(prepare_query(f.built,f.bound,:pointwise),f.u)
        @test only(values(pw)) ≈ f.pointwise(f.u)
        @test f.data == f.saved
    end
end

@testset "empty packed row selections preserve priors and empty pointwise values" begin
    for kind in _SCALAR_MI_KINDS
        f=_scalar_mi_fixture(kind,9;selected_range=2:2)
        _distributional_check(f,f.oracle,f.u)
        pw=Base.invokelatest(prepare_query(f.built,f.bound,:pointwise),f.u)
        @test isempty(only(values(pw)))
    end
end

@testset "scalar Case-A inputs retain index trial and numeric evidence validation" begin
    for jobs in ([1,1,3,5,7],[1,3,5,7,10],[3,1,5,7,9])
        f=_scalar_mi_fixture(:poisson,9)
        data=merge(f.data,Dict(:Jobs=>jobs))
        @test_throws ContractValidationError bind_data(f.plan,data)
    end
    f=_scalar_mi_fixture(:binomial,9)
    bad=merge(f.data,Dict(:trials=>fill(0,9)))
    @test_throws ContractValidationError bind_data(f.plan,bad)
    ordinal=_scalar_mi_fixture(:stopping,9;variant=:gathered)
    bad=merge(ordinal.data,Dict(:E=>zeros(5,2)))
    @test_throws ContractValidationError bind_data(ordinal.plan,bad)
    bad=merge(ordinal.data,Dict(:E=>zeros(9,1)))
    @test_throws DimensionMismatch bind_data(ordinal.plan,bad)
    # Evidence stays strict numeric data, including on a packed likelihood.
    cols,n=_mi_columns()
    cols[:lo]=fill(-1.0,n);cols[:hi]=fill(1.0,n)
    plan=_mi_gaussian_plan(;cols)
    r=ReactiveKernelsPPL._with(only(plan.responses);evidence=ResponseEvidence(:truncated,:lo,:hi))
    plan=ReactiveKernelsPPL._with(plan;responses=[r])
    @test validate_plan(plan) === nothing
    for name in (:lo,:hi)
        bad=copy(plan.columns);bad[name]=fill("bad",n)
        @test_throws ContractValidationError bind_data(_mi_unbind(plan),bad)
    end
end
