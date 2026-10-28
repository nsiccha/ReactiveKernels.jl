using Test, ReactiveKernels, ReactiveKernelsPPL
import Distributions as D
import DataAPI

# Ordinary integer observations with a declared level pool. This uses the
# public DataAPI protocol, including a category absent from the observations.
struct _AuditLevelData <: AbstractVector{Int}
    values::Vector{Int}
    pool::Vector{Int}
end
Base.size(x::_AuditLevelData) = size(x.values)
Base.IndexStyle(::Type{_AuditLevelData}) = IndexLinear()
Base.getindex(x::_AuditLevelData,i::Int) = x.values[i]
DataAPI.levels(x::_AuditLevelData) = x.pool

function _audit_cumulative(eta,c,k)
    1 <= k <= length(c)+1 || return -Inf
    all(diff(c) .> 0) || return -Inf
    hi = k == length(c)+1 ? 1.0 : D.cdf(D.Logistic(),c[k]-eta)
    lo = k == 1 ? 0.0 : D.cdf(D.Logistic(),c[k-1]-eta)
    return log(hi-lo)
end
function _audit_stopping(eta,c,k)
    lp = sum((D.logccdf(D.Logistic(),c[j]-eta) for j in 1:k-1);init=0.0)
    return k == length(c)+1 ? lp : lp + D.logcdf(D.Logistic(),c[k]-eta)
end

function _audit_cases(n)
    x=collect(range(-0.4,0.5;length=n))
    y=[mod(i-1,3)+1 for i in 1:n]
    z=fill(0.2,n)
    cases=NamedTuple[]
    for form in (:normal_cumulative,:ordered_stopping,:cauchy_stopping)
        ordered = form === :ordered_stopping
        prior = form === :cauchy_stopping ? :(c[1:2] .~ Cauchy.(0,1)) :
            ordered ? :(c ~ Ordered(Normal(0,1),2)) : :(c[1:2] .~ Normal.(0,1))
        response = form === :normal_cumulative ? :(y .~ OrderedLogistic.(b .* x,Ref(c))) :
            :(y .~ Ordinal.(StoppingRatio(),LogitLink(),b .* x,Ref(c)))
        expr=quote b ~ Normal(0,1); $prior; $response end
        law=form === :cauchy_stopping ? D.Cauchy() : D.Normal()
        density=form === :normal_cumulative ? _audit_cumulative : _audit_stopping
        push!(cases,(;label=String(form),expr,data=Dict{Symbol,Any}(:x=>x,:y=>y),
            q=(b=0.25,c=[-0.7,0.8]),jac=q->ordered ? sum(log,diff(q.c)) : 0.0,
            oracle=q->D.logpdf(D.Normal(),q.b)+sum(D.logpdf.(law,q.c))+
                sum(density(q.b*x[i],q.c,y[i]) for i in eachindex(y))))
    end
    for (label,size,width,observations) in (
            ("levels of predictor",:(length(levels(x))-1),n-1,y),
            ("levels of response minus two",:(length(levels(y))-2),2,
                _AuditLevelData(y,[1,2,3,4])),
            ("unique response",:(length(unique(y))-1),2,y))
        expr=quote
            b ~ Normal(0,1); c ~ Ordered(Normal(0,1),$size)
            y .~ OrderedLogistic.(b .* x,Ref(c))
        end
        push!(cases,(;label,expr,data=Dict{Symbol,Any}(:x=>x,:y=>observations),
            q=(b=0.25,c=collect(range(-0.7,1.0;length=width))),jac=q->sum(log,diff(q.c)),
            oracle=q->D.logpdf(D.Normal(),q.b)+sum(D.logpdf.(D.Normal(),q.c))+
                sum(_audit_cumulative(q.b*x[i],q.c,y[i]) for i in eachindex(y))))
    end
    expr=quote
        a ~ Normal(0,1); c ~ Ordered(Normal(0,1),length(levels(y))-1)
        mu=a .+ c[1] .* x; z .~ Normal.(mu,1)
    end
    push!(cases,(;label="data-sized ordered Gaussian reuse",expr,
        data=Dict{Symbol,Any}(:x=>x,:y=>[1,2,3],:z=>z),q=(a=0.25,c=[-0.7,0.8]),
        jac=q->sum(log,diff(q.c)),oracle=q->D.logpdf(D.Normal(),q.a)+
            sum(D.logpdf.(D.Normal(),q.c))+sum(D.logpdf.(D.Normal.(q.a .+ q.c[1].*x,1),z))))
    expr=quote
        @plate for i in eachindex(z)
            theta[i] ~ Flat()
        end
        z .~ Normal.(theta,1)
    end
    push!(cases,(;label="flat per-cell density",expr,data=Dict{Symbol,Any}(:z=>z),
        q=(theta=fill(0.3,n),),jac=q->0.0,
        oracle=q->sum(D.logpdf.(D.Normal.(q.theta,1),z))))
    return cases
end

function _audit_check(case)
    before=deepcopy(case.data)
    bound=bind_data(lower_rkppl(case.expr,case.data;conditioned=keys(case.data)),case.data)
    built=build_kernel(bound)
    u=unconstrain(built.layout,case.q)
    independent_jac=case.jac(case.q)
    @test logjac(built.layout,u) ≈ independent_jac atol=1e-12
    kernel=prepare_query(built,bound,:sampler)
    @test Base.invokelatest(kernel,u) ≈ case.oracle(case.q)+independent_jac atol=1e-11
    @test case.data == before
    gradient=_check_gradient(built.spec,bound,u)
    oracle_u=w->begin
        q=constrain(built.layout,w)
        case.oracle(q)+case.jac(q)
    end
    @test gradient ≈ _findiff_grad(oracle_u,u) atol=1e-7 rtol=1e-5
    return bound,built,kernel,u
end

@testset "later prior and ordinal audit: independent density, Jacobian and AD" begin
    for n in (6,12), case in _audit_cases(n)
        @testset "$(case.label) n=$n" begin
            _audit_check(case)
        end
    end
end

@testset "ordinary cumulative thresholds retain prior support and lazy density" begin
    case=first(_audit_cases(6))
    bound,built,kernel,_=_audit_check(case)
    for c in ([0.8,-0.7],[0.8,0.8])
        u=unconstrain(built.layout,(b=0.25,c=c))
        @test Base.invokelatest(kernel,u) == -Inf
        sampler=prepare_sampler(built,bound,u;backend=_GEN_BACKEND)
        value,gradient=sampler_value_and_gradient!(sampler,similar(u),u)
        @test value == -Inf
        @test gradient ≈ -u atol=1e-12
    end
end

const _AUDIT_BINOMIAL_AST = quote theta ~ Beta(1,1); k ~ Binomial(n,theta) end
const _audit_binomial_model = @rkppl begin theta ~ Beta(1,1); k ~ Binomial(n,theta) end

function _audit_scalar_binomial(door,n,k)
    data=Dict{Symbol,Any}(:n=>n,:k=>k)
    door === :public && return _audit_binomial_model(;n) | (;k)
    return bind_data(lower_rkppl(_AUDIT_BINOMIAL_AST,
        door === :names ? (:n,:k) : data;conditioned=(:k,)),data)
end

@testset "scalar Binomial observation doors: density, Jacobian and AD" begin
    for door in (:names,:values,:public), (n,k) in ((5,2),(9,4),(0,0))
        bound=_audit_scalar_binomial(door,n,k)
        built=build_kernel(bound)
        @test built.layout.total == 1
        @test :k in bound.conditioned
        q=(theta=0.37,)
        u=unconstrain(built.layout,q)
        prior=prepare_query(built,bound,:prior)
        likelihood=prepare_query(built,bound,:likelihood)
        @test Base.invokelatest(prior,u) ≈ D.logpdf(D.Beta(1,1),q.theta) atol=1e-12
        @test Base.invokelatest(likelihood,u) ≈ D.logpdf(D.Binomial(n,q.theta),k) atol=1e-12
        oracle=w->begin
            theta=1/(1+exp(-w[1]))
            D.logpdf(D.Beta(1,1),theta)+D.logpdf(D.Binomial(n,theta),k)+
                log(theta)+log1p(-theta)
        end
        _check_model_math(built,bound,u,oracle)
    end
end

@testset "audit sizes and scalar observations: Julia diagnostics and controls" begin
    cases=_audit_cases(6)
    sized=only(filter(c->c.label=="levels of response minus two",cases))
    plain=merge(sized.data,Dict(:y=>[1,2,3,1,2,3]))
    # refused: the actual one-threshold extent has support 1:2, but y contains 3.
    @test_throws ContractValidationError bind_data(lower_rkppl(sized.expr,plain;
        conditioned=keys(plain)),plain)
    _audit_check(sized)
    for size in (:(length(unique(y))-5),:(length(unique(y))/2))
        expr=quote
            a ~ Normal(0,1); c ~ Ordered(Normal(0,1),$size)
            mu=a .+ c[1] .* z; z .~ Normal.(mu,1)
        end
        data=Dict{Symbol,Any}(:y=>[1,2,3],:z=>[0.1,0.2,0.3])
        # refused: negative or noninteger Julia extent cannot allocate a vector.
        @test_throws ContractValidationError bind_data(lower_rkppl(expr,data;conditioned=keys(data)),data)
    end
    for door in (:names,:values,:public)
        # refused: Binomial's trial count cannot be negative, fractional or vector-valued.
        for n in (-1,2.5,[5,5])
            @test_throws ContractValidationError _audit_scalar_binomial(door,n,2)
        end
        # refused: scalar ~ observes one scalar, never an observation vector.
        @test_throws ContractValidationError _audit_scalar_binomial(door,5,[2])
        @test build_kernel(_audit_scalar_binomial(door,5,2)).layout.total == 1
        for k in (-1,6,2.5)
            bound=_audit_scalar_binomial(door,5,k)
            built=build_kernel(bound)
            u=unconstrain(built.layout,(theta=0.37,))
            sampler=prepare_sampler(built,bound,u;backend=_GEN_BACKEND)
            value,gradient=sampler_value_and_gradient!(sampler,similar(u),u)
            @test value == -Inf
            @test gradient ≈ [1-2*0.37] atol=1e-12
        end
    end
    # Refused: Binomial has no three-positional-argument constructor.
    @test_throws SurfaceLoweringError lower_rkppl(quote k ~ Binomial(n,0.5,3) end,
        (:n,:k);conditioned=(:k,))
end
