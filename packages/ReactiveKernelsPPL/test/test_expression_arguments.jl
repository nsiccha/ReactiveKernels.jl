using Test, ReactiveKernelsPPL
import ReactiveKernels
import Distributions as D

function _expression_compare(actual::NamedTuple, expected::NamedTuple)
    for name in keys(expected)
        _expression_compare(actual[name], expected[name])
    end
end
_expression_compare(actual, expected) = @test actual ≈ expected

# Explicit variants of the shipped library bodies, with a stated scale
# argument (user decisions 1cmodra and 10ldrvz).
@rkppl _expression_smooth(X, Z, scale) = begin
    b[axes(X, 2)] .~ Flat.()
    sd ~ HalfNormal(scale)
    z[axes(Z, 2)] .~ Normal.(0, 1)
    return X * b .+ Z * (sd .* z)
end
@rkppl _expression_horseshoe(X, scale) = begin
    tau ~ HalfCauchy(scale)
    lambda[axes(X, 2)] .~ HalfCauchy.(1)
    z[axes(X, 2)] .~ Normal.(0, 1)
    return z .* lambda .* tau
end

function _expression_check(expr, data, q, oracle)
    before = deepcopy(data)
    bound = bind_data(lower_rkppl(expr, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, q)
    kernel = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(kernel, u) ≈ oracle(q) + logjac(built.layout, u) atol=1e-11 rtol=1e-11
    @test data == before
    restored = constrain(built.layout, u)
    _expression_compare(restored, q)
    # Shared Enzyme/central-difference acceptance from test_generator.jl.
    _check_gradient(built.spec, bound, u)
    return bound, built, kernel, u
end

# Independent Distributions densities for the P8 argument positions. The
# same fixtures are also used by the compiled value/gradient checks.
_expression_sigmoid(x) = 1 / (1 + exp(-x))
function _expression_cases(n)
    x = collect(range(-0.4, 0.5; length=n))
    y = fill(0.2, n)
    cases = NamedTuple[]
    for inline in (false,true)
        expr = quote
            b ~ Normal(0,1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(y)
                theta[i] ~ Normal(0,1)
            end
        end
        if inline
            push!(expr.args, :(y .~ Normal.(theta .+ b .* x,sigma)))
        else
            push!(expr.args, :(mu = theta .+ b .* x), :(y .~ Normal.(mu,sigma)))
        end
        push!(cases, (;label="latent location inline=$inline",expr,data=Dict(:x=>x,:y=>y),
            q=(b=0.3,sigma=0.8,theta=fill(0.1,n)),oracle=q ->
                D.logpdf(D.Normal(),q.b)+D.logpdf(D.Exponential(),q.sigma)+
                sum(D.logpdf.(D.Normal(),q.theta))+
                sum(D.logpdf.(D.Normal.(q.theta .+ q.b.*x,q.sigma),y))))
    end
    for form in (:alias,:scalar_expression,:vector_inline,:vector_named,:data)
        expr = quote
            a ~ Normal(0,1)
            b ~ Normal(0,1)
            sigma ~ Exponential(1)
        end
        loc = form === :alias ? :m : form === :scalar_expression ? :(exp.(a)) :
            form === :vector_inline ? :(exp.(a .+ b .* x)) :
            form === :vector_named ? :m : :x
        form === :alias && push!(expr.args, :(m = a))
        form === :vector_named && push!(expr.args, :(m = exp.(a .+ b .* x)))
        push!(expr.args, :(y .~ MixtureModel.(vcat.(Normal.($loc,sigma),Normal.(0,1)),Ref([0.4,0.6]))))
        push!(cases, (;label="mixture location $form",expr,data=Dict(:x=>x,:y=>y),
            q=(a=0.2,b=-0.3,sigma=0.8),oracle=q -> begin
                means = form === :alias ? fill(q.a,n) : form === :scalar_expression ?
                    fill(exp(q.a),n) : form === :data ? x : exp.(q.a .+ q.b.*x)
                D.logpdf(D.Normal(),q.a)+D.logpdf(D.Normal(),q.b)+
                    D.logpdf(D.Exponential(),q.sigma)+
                    sum(log(0.4D.pdf(D.Normal(means[i],q.sigma),y[i])+
                        0.6D.pdf(D.Normal(),y[i])) for i in 1:n)
            end))
    end
    expr = quote
        lam ~ Normal(0,1)
        y .~ MixtureModel.(vcat.(Poisson.(exp.(lam)),Poisson.(4)),Ref([0.5,0.5]))
    end
    counts = [mod(i-1,3) for i in 1:n]
    push!(cases, (;label="mixture linked sampled rate",expr,data=Dict(:y=>counts),
        q=(lam=0.2,),oracle=q -> D.logpdf(D.Normal(),q.lam)+
            sum(log(0.5D.pdf(D.Poisson(exp(q.lam)),v)+0.5D.pdf(D.Poisson(4),v)) for v in counts)))
    for scale in (:(exp.(1.5)), :(exp.(x)), :(exp(s)), :(sqrt.(exp.(x))))
        sampled = scale == :(exp(s))
        expr = quote
            a ~ Normal(0, 1)
            y .~ Normal.(a, $scale)
        end
        sampled && pushfirst!(expr.args, :(s ~ Normal(0, 1)))
        data = sampled || scale == :(exp.(1.5)) ? Dict(:y=>y) : Dict(:y=>y, :x=>x)
        q = sampled ? (a=0.3, s=-0.2) : (a=0.3,)
        scales = sampled ? exp(q.s) : scale == :(exp.(1.5)) ? exp(1.5) :
            scale == :(exp.(x)) ? exp.(x) : sqrt.(exp.(x))
        push!(cases, (;label="scale $scale", expr, data, q, oracle=q ->
            D.logpdf(D.Normal(), q.a) + (sampled ? D.logpdf(D.Normal(), q.s) : 0) +
            sum(D.logpdf.(D.Normal.(q.a, scales), y))))
    end
    for named in (false, true)
        expr = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
        end
        if named
            push!(expr.args, :(mu = exp.(a .+ b .* x)), :(y .~ Normal.(mu, 1)))
        else
            push!(expr.args, :(y .~ Normal.(exp.(a .+ b .* x), 1)))
        end
        push!(cases, (;label="nonlinear location named=$named", expr,
            data=Dict(:x=>x,:y=>y), q=(a=0.1,b=0.3), oracle=q ->
                D.logpdf(D.Normal(),q.a)+D.logpdf(D.Normal(),q.b)+
                sum(D.logpdf.(D.Normal.(exp.(q.a .+ q.b .* x),1),y))))
    end
    for sign in (1,-1)
        expr = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            off = 1.0
            mu = a .+ b .* x .+ $sign .* off
            y .~ Normal.(mu,1)
        end
        push!(cases, (;label="named scalar offset sign=$sign",expr,data=Dict(:x=>x,:y=>y),
            q=(a=0.2,b=0.3),oracle=q -> D.logpdf(D.Normal(),q.a)+
                D.logpdf(D.Normal(),q.b)+sum(D.logpdf.(D.Normal.(
                    q.a .+ q.b.*x .+ sign,1),y))))
    end
    for nu in (:(2+2), :(2 .+ exp.(x)), :(2 + exp(s)))
        sampled = nu == :(2 + exp(s))
        expr = quote
            a ~ Normal(0, 1)
            y .~ StudentT.($nu, a, 1)
        end
        sampled && pushfirst!(expr.args, :(s ~ Normal(0, 1)))
        data = nu == :(2 .+ exp.(x)) ? Dict(:x=>x,:y=>y) : Dict(:y=>y)
        q = sampled ? (a=0.2,s=0.3) : (a=0.2,)
        nus = sampled ? 2+exp(q.s) : nu == :(2+2) ? 4 : 2 .+ exp.(x)
        push!(cases, (;label="nu $nu", expr, data, q, oracle=q ->
            D.logpdf(D.Normal(),q.a)+(sampled ? D.logpdf(D.Normal(),q.s) : 0)+
            sum(D.logpdf.(D.TDist.(nus), y .- q.a))))
    end
    counts = [mod(i-1,3) for i in 1:n]
    for zi in (:(0.1+0.1), :(logistic.(x)), :(logistic.(g)))
        sampled = zi == :(logistic.(g))
        expr = quote
            a ~ Normal(0, 1)
            y .~ ZeroInflatedPoisson.(exp.(a), $zi)
        end
        sampled && pushfirst!(expr.args, :(g ~ Normal(0, 1)))
        data = zi == :(logistic.(x)) ? Dict(:x=>x,:y=>counts) : Dict(:y=>counts)
        q = sampled ? (a=0.3,g=-0.4) : (a=0.3,)
        zs = sampled ? fill(_expression_sigmoid(q.g),n) : zi == :(0.1+0.1) ?
            fill(0.2,n) : _expression_sigmoid.(x)
        push!(cases, (;label="zi $zi", expr, data, q, oracle=q ->
            D.logpdf(D.Normal(),q.a)+(sampled ? D.logpdf(D.Normal(),q.g) : 0)+
            sum(log((counts[i] == 0 ? zs[i] : 0) +
                (1-zs[i])*D.pdf(D.Poisson(exp(q.a)),counts[i])) for i in 1:n)))
    end
    for mismatch in (:kappa, :mu)
        expr = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            k ~ Gamma(2, 1)
            k2 ~ Gamma(2, 1)
            mu = a .+ b .* x
        end
        second = mismatch === :kappa ? :((1 .- logistic.(mu)) .* k2) :
            :((1 .- logistic.(mu .+ 0.2)) .* k)
        push!(expr.args, :(y .~ Beta.(logistic.(mu) .* k, $second)))
        push!(cases, (;label="Beta $mismatch", expr, data=Dict(:x=>x,:y=>y),
            q=(a=0.2,b=-0.1,k=2.3,k2=1.7), oracle=q ->
                D.logpdf(D.Normal(),q.a)+D.logpdf(D.Normal(),q.b)+
                D.logpdf(D.Gamma(2,1),q.k)+D.logpdf(D.Gamma(2,1),q.k2)+
                sum(D.logpdf.(D.Beta.(_expression_sigmoid.(q.a .+ q.b .* x).*q.k,
                    (1 .- _expression_sigmoid.(q.a .+ q.b .* x .+
                        (mismatch === :mu ? 0.2 : 0))) .*
                    (mismatch === :kappa ? q.k2 : q.k)), y))))
    end
    for endpoints in (:named, :data, :sampled)
        expr = quote a ~ Normal(0, 1) end
        data = Dict{Symbol,Any}(:y=>y)
        if endpoints === :named
            push!(expr.args, :(lo = -pi), :(hi = pi))
        elseif endpoints === :data
            data[:lo] = -pi; data[:hi] = pi
        else
            push!(expr.args, :(lo ~ Normal(-pi, 0.2)), :(hi = lo + 2pi))
        end
        push!(expr.args, :(y .~ CircularVonMises.(a, 1.7, lo, hi)))
        q = endpoints === :sampled ? (a=0.1,lo=-pi+0.1) : (a=0.1,)
        push!(cases, (;label="interval $endpoints", expr, data, q, oracle=q ->
            D.logpdf(D.Normal(),q.a)+(endpoints === :sampled ?
                D.logpdf(D.Normal(-pi,0.2),q.lo) : 0)+
            sum(D.logpdf.(D.VonMises(q.a,1.7),y))))
    end
    group = [mod(i-1,2)+1 for i in 1:n]
    expr = quote
        sg ~ Exponential(1)
        z = x .+ 1
        c[levels(z)] .~ Normal.(0, sg)
        y .~ Normal.(c[z], 1)
    end
    push!(cases, (;label="computed grouping", expr, data=Dict(:x=>group,:y=>y),
        q=(sg=0.8,c=[-0.1,0.4]), oracle=q -> D.logpdf(D.Exponential(),q.sg)+
            sum(D.logpdf.(D.Normal(0,q.sg),q.c))+
            sum(D.logpdf.(D.Normal.(q.c[group],1),y))))
    expr = quote
        sg ~ Exponential(1)
        z = g .+ 0
        c[levels(g)] .~ Normal.(0,sg)
        y .~ Normal.(c[z],1)
    end
    push!(cases, (;label="computed gather index",expr,data=Dict(:g=>group,:y=>y),
        q=(sg=0.8,c=[-0.1,0.4]),oracle=q -> D.logpdf(D.Exponential(),q.sg)+
            sum(D.logpdf.(D.Normal(0,q.sg),q.c))+
            sum(D.logpdf.(D.Normal.(q.c[group],1),y))))
    levels = [mod(i-1,3)+1 for i in 1:n]
    for family in (:Normal, :Cauchy, :Laplace, :Logistic, :StudentT)
        element = family === :StudentT ? :(StudentT(2 + s, m, 2s)) :
            Expr(:call,family,:m,:(2s))
        expr = quote
            m ~ Normal(0, 1)
            s ~ Exponential(1)
            b ~ Normal(0, 1)
            c ~ Ordered($element, 2)
            y .~ OrderedLogistic.(b .* x, Ref(c))
        end
        push!(cases, (;label="ordered $family", expr, data=Dict(:x=>x,:y=>levels),
            q=(m=0.1,s=0.8,b=0.3,c=[-0.3,0.7]), oracle=q -> begin
                d = family === :Normal ? D.Normal(q.m,2q.s) :
                    family === :Cauchy ? D.Cauchy(q.m,2q.s) :
                    family === :Laplace ? D.Laplace(q.m,2q.s) :
                    family === :Logistic ? D.Logistic(q.m,2q.s) :
                    D.LocationScale(q.m,2q.s,D.TDist(2+q.s))
                prior = D.logpdf(D.Normal(),q.m)+D.logpdf(D.Exponential(),q.s)+
                    D.logpdf(D.Normal(),q.b)+sum(D.logpdf.(d,q.c))
                prior + sum(log((levels[i] == 3 ? 1.0 :
                    _expression_sigmoid(q.c[levels[i]]-q.b*x[i])) -
                    (levels[i] == 1 ? 0.0 :
                    _expression_sigmoid(q.c[levels[i]-1]-q.b*x[i]))) for i in 1:n)
            end))
    end
    expr = quote
        s ~ Normal(0, 1)
        p ~ Dirichlet(exp(s) .* alpha)
        y .~ Categorical(p)
    end
    alpha = [1.2,1.7,2.1]
    push!(cases, (;label="live concentrations", expr, data=Dict(:alpha=>alpha,:y=>levels),
        q=(s=0.2,p=[0.2,0.3,0.5]), oracle=q -> D.logpdf(D.Normal(),q.s)+
            D.logpdf(D.Dirichlet(exp(q.s).*alpha),q.p)+
            sum(log(q.p[i]) for i in levels)))
    expr = quote
        s ~ Exponential(1)
        b ~ Gamma(s + 1, 0.5)
        y .~ Normal.(b .* x, 1)
    end
    push!(cases, (;label="Gamma coefficient argument", expr, data=Dict(:x=>x,:y=>y),
        q=(s=0.7,b=0.6), oracle=q -> D.logpdf(D.Exponential(),q.s)+
            D.logpdf(D.Gamma(q.s+1,0.5),q.b)+
            sum(D.logpdf.(D.Normal.(q.b.*x,1),y))))
    expr = quote
        b ~ Normal(0, 1)
        s ~ Dirichlet(2, 1)
        m = cumsum(vcat(0.0, s))[c]
        y .~ Normal.(b .* (m .+ x), 1)
    end
    push!(cases, (;label="nested prefix/gather value", expr, data=Dict(:c=>levels,:x=>x,:y=>y),
        q=(b=0.3,s=[0.4,0.6]), oracle=q -> D.logpdf(D.Normal(),q.b)+
            D.logpdf(D.Dirichlet(ones(2)),q.s)+
            sum(D.logpdf.(D.Normal.(q.b.*([0.0,q.s[1],1.0][levels].+x),1),y))))
    expr = quote
        lo ~ Normal(0, 1)
        c[levels(g)] .~ Uniform.(lo, lo + 3)
        y .~ Normal.(c[g], 1)
    end
    push!(cases, (;label="factor Uniform sampled bounds", expr,
        data=Dict(:g=>group,:y=>y),q=(lo=0.2,c=[0.6,1.2]),oracle=q ->
            D.logpdf(D.Normal(),q.lo)+
            sum(D.logpdf.(D.Uniform(q.lo,q.lo+3),q.c))+
            sum(D.logpdf.(D.Normal.(q.c[group],1),y))))
    X = reshape(x, :, 1)
    Z = reshape(x .^ 2 .+ 0.1, :, 1)
    expr = quote
        s ~ Exponential(1)
        f ~ _expression_smooth(X, Z, s + 1)
        y .~ Normal.(f, 1)
    end
    push!(cases, (;label="smooth sampled scale", expr, data=Dict(:X=>X,:Z=>Z,:y=>y),
        q=(s=0.8,f=(b=[0.1],sd=0.6,z=[0.3])), oracle=q ->
            D.logpdf(D.Exponential(),q.s)+
            D.logpdf(D.truncated(D.Normal(0,q.s+1),0,Inf),q.f.sd)+
            sum(D.logpdf.(D.Normal(),q.f.z))+
            sum(D.logpdf.(D.Normal.(X*q.f.b+Z*(q.f.sd.*q.f.z),1),y))))
    expr = quote
        s ~ Exponential(1)
        b ~ _expression_horseshoe(X, s + 1)
        y .~ Normal.(X * b, 1)
    end
    push!(cases, (;label="horseshoe sampled scale", expr, data=Dict(:X=>X,:y=>y),
        q=(s=0.8,b=(tau=0.6,lambda=[0.9],z=[0.3])), oracle=q ->
            D.logpdf(D.Exponential(),q.s)+
            D.logpdf(D.truncated(D.Cauchy(0,q.s+1),0,Inf),q.b.tau)+
            sum(D.logpdf.(D.truncated(D.Cauchy(),0,Inf),q.b.lambda))+
            sum(D.logpdf.(D.Normal(),q.b.z))+
            sum(D.logpdf.(D.Normal.(X*(q.b.z.*q.b.lambda.*q.b.tau),1),y))))
    expr = quote
        alpha ~ Normal(0, 1)
        y .~ weighted.(Beta.(alpha, 2), w)
    end
    w = [0.5 + mod(i,3)/2 for i in 1:n]
    push!(cases, (;label="weighted Beta shape", expr,
        data=Dict(:y=>y,:w=>w), q=(alpha=1.7,), oracle=q ->
            D.logpdf(D.Normal(),q.alpha)+
            sum(w .* D.logpdf.(D.Beta(q.alpha,2),y))))
    expr = quote
        m ~ Normal(0, 1)
        k ~ Ordered(Normal(m, 1), 2)
        y .~ Normal.(0, 1)
    end
    push!(cases, (;label="unused ordered live prior", expr, data=Dict(:y=>y),
        q=(m=0.2,k=[-0.3,0.7]), oracle=q ->
            D.logpdf(D.Normal(),q.m)+sum(D.logpdf.(D.Normal(q.m,1),q.k))+
            sum(D.logpdf.(D.Normal(),y))))
    return cases
end

@testset "distribution and construct arguments are values" begin
    for n in (3,7), case in _expression_cases(n)
        @testset "$(case.label) n=$n" begin
            println("EXPRESSION_BEGIN ",case.label," n=",n); flush(stdout)
            _expression_check(case.expr,case.data,case.q,case.oracle)
        end
    end
end

_expression_empty_cases() = filter(c -> c.label in
    ("Beta kappa", "Beta mu", "scale exp.(x)", "weighted Beta shape"), _expression_cases(0))

@testset "empty response argument values preserve active priors" begin
    for case in _expression_empty_cases()
        _expression_check(case.expr,case.data,case.q,case.oracle)
    end
end

@testset "cell expressions and computed counts preserve loop semantics" begin
    expr = quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in 1:(1 + 1)
            y[i] ~ Normal(a + x[i] + 1, sigma)
        end
    end
    data=Dict(:x=>[-0.1,0.3],:y=>[0.2,0.4])
    _expression_check(expr,data,(a=0.2,sigma=0.8),q ->
        D.logpdf(D.Normal(),q.a)+D.logpdf(D.Exponential(),q.sigma)+
        sum(D.logpdf.(D.Normal.(q.a .+ data[:x] .+ 1,q.sigma),data[:y])))
end

# All live coordinates have standard Normal priors. The invalid response
# contributes -Inf and no gradient; the surviving prior gradient is -u.
function _expression_inactive_cases()
    return [
        (quote
            alpha ~ Normal(0,1)
            y .~ Beta.(alpha,2)
        end, Dict(:y=>[0.2,0.4]), (alpha=-0.3,)),
        (quote
            alpha ~ Normal(0,1)
            y .~ weighted.(Beta.(alpha,2),w)
        end, Dict(:y=>[0.2,0.4],:w=>[0.5,1.5]), (alpha=-0.3,)),
        (quote
            lo ~ Normal(0,1)
            hi ~ Normal(0,1)
            y .~ CircularVonMises.(0,1,lo,hi)
        end, Dict(:y=>[0.2,0.4]), (lo=0.3,hi=-0.2,)),
    ]
end
function _expression_inactive_native(expr,data,q)
    bound = bind_data(lower_rkppl(expr,data;conditioned=keys(data)),data)
    built = build_kernel(bound)
    u = unconstrain(built.layout,q)
    sampler = prepare_sampler(built,bound,u;backend=_GEN_BACKEND)
    value,gradient = sampler_value_and_gradient!(sampler,similar(u),u)
    @test value == -Inf
    @test gradient ≈ -u
    return sampler,u
end
@testset "argument domain checks keep inactive densities lazy" begin
    for (expr,data,q) in _expression_inactive_cases()
        _expression_inactive_native(expr,data,q)
    end
end


function _expression_beta_evidence_case(n, kind)
    lo, hi = 0.15, 0.85
    y = kind === :censored ? [mod1(i,3) == 1 ? lo :
        mod1(i,3) == 2 ? 0.4 : hi for i in 1:n] :
        collect(range(0.2,0.7; length=n))
    ctor = :(Beta.(1 + exp(a), 2 + exp(b)))
    wrapper = kind === :interval_censored ?
        Expr(:., kind, Expr(:tuple, ctor, hi)) :
        Expr(:., kind, Expr(:tuple, ctor, lo, hi))
    expr = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        y .~ $wrapper
    end
    q = (a=0.2,b=-0.1)
    oracle = function (q)
        dist = D.Beta(1 + exp(q.a), 2 + exp(q.b))
        likelihood = if kind === :interval_censored
            sum((log(D.cdf(dist,hi)-D.cdf(dist,v)) for v in y); init=0.0)
        else
            wrapped = kind === :truncated ? D.truncated(dist,lo,hi) :
                D.censored(dist,lo,hi)
            sum(D.logpdf.(Ref(wrapped),y))
        end
        return D.logpdf(D.Normal(),q.a)+D.logpdf(D.Normal(),q.b)+likelihood
    end
    return (; expr, data=Dict(:y=>y), q, oracle)
end

@testset "ordinary Beta argument values retain response evidence" begin
    for n in (0,3,7), kind in (:truncated,:censored,:interval_censored)
        case = _expression_beta_evidence_case(n,kind)
        bound, built, kernel, u =
            _expression_check(case.expr,case.data,case.q,case.oracle)
        @test only(bound.responses).evidence.kind === kind
        @test only(bound.responses).family === ReactiveKernelsPPL.BetaShapeFam
    end
end
