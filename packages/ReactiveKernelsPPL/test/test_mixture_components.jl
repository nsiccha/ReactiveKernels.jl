module MixtureComponentTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
import Distributions as D
import DifferentiationInterface as DI
import Enzyme
include(joinpath(pkgdir(ReactiveKernels), "test", "type_stability_audit.jl"))

struct ShiftedLaw end
struct KeywordLaw end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{ShiftedLaw}, args::Tuple)
    # A fresh local graph tests replay without a global graph registry. Its
    # four arguments intentionally differ from the neighboring two-arg laws.
    @kernel shifted(value, mu, logscale, shift, extra) = begin
        scale = sqrt(exp(2logscale) + extra^2)
        center = mu + sin(shift)
        return normal(center, scale).logpdf(value)
    end
    shifted
end
@kernel keyword_score(value, mu; scale=1.0) = begin
    return normal(mu, scale).logpdf(value)
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{KeywordLaw}, args::Tuple) = keyword_score

# Diagnostic instrumentation of one graph recipe, used only by value checks.
const NORMALIZERS = Ref(0)
counted_normalizer(s) = (NORMALIZERS[] += 1; s + log(2pi)/2)
struct CountedLaw end
@kernel counted_score(value, mu, logscale) = begin
    normalizer = counted_normalizer(logscale)
    z = (value-mu)*exp(-logscale)
    return -normalizer-z^2/2
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{CountedLaw}, args::Tuple) = counted_score

function fixture(n; kind=:builtin, shared=false, weights=[0.8,0.2], scale=0.7,
        rowweights=false)
    x = [0.2sin(i) for i in 1:n]
    y = [0.4cos(i) for i in 1:n]
    data = rowweights ? (;x,y,scale) : (;x,y,scale,weights)
    weight_expression = rowweights ?
        :(vcat.(1 .- logistic.(b .+ x), logistic.(b .+ x))) : :(Ref(weights))
    location = shared ? :a : :(a .+ b .* curved)
    other = kind === :builtin ? :(Cauchy.(mu, exp(s))) :
        kind === :student ? :(D.TDist.(exp(s))) :
        kind === :custom ? :(ShiftedLaw.(mu, s, b, 0.3)) :
        :(KeywordLaw.(mu; scale=exp(s)))
    ast = quote
        a ~ Normal(0,1)
        b ~ Normal(0,1)
        s ~ Normal(0,1)
        curved = sin.(x)
        mu = $location
        y .~ MixtureModel.(vcat.(Normal.(mu, scale), $other), $weight_expression)
    end
    plan = lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,))
    bound = bind_data(plan,data)
    built = build_kernel(bound)
    (;data,plan,bound,built,u=[0.2,-0.3,-0.1],kind,shared,rowweights)
end
function oracle(fx,u)
    a,b,s = u
    points = map(eachindex(fx.data.y)) do i
        mu = fx.shared ? a : a+b*sin(fx.data.x[i])
        law = fx.kind === :builtin ? D.Cauchy(mu,exp(s)) :
            fx.kind === :student ? D.TDist(exp(s)) :
            fx.kind === :custom ? D.Normal(mu+sin(b),sqrt(exp(2s)+0.3^2)) :
            D.Normal(mu,exp(s))
        # An independent probability sum suffices for these moderate toy values.
        w = fx.rowweights ? inv(1+exp(-b-fx.data.x[i])) : fx.data.weights[2]
        first_weight = fx.rowweights ? 1-w : fx.data.weights[1]
        log(first_weight*D.pdf(D.Normal(mu,fx.data.scale),fx.data.y[i]) +
            w*D.pdf(law,fx.data.y[i]))
    end
    prior = sum(D.logpdf(D.Normal(),v) for v in u)
    (;points,prior,total=prior+sum(points;init=0.0))
end
function finite_gradient(f,u)
    map(eachindex(u)) do i
        p,m = copy(u),copy(u)
        p[i]+=1e-5; m[i]-=1e-5
        (f(p)-f(m))/2e-5
    end
end
query(fx,cut) = prepare_query(fx.built,fx.bound,cut)
recipe_count(spec::KernelSpec) = recipe_count(spec.graph)
recipe_count(graph) = sum(graph.recipes;init=0) do recipe
    1 + (recipe_kind(recipe) === :plate ? recipe_count(plate_body(recipe)) : 0)
end
function check(fx; reverse=false)
    saved = deepcopy((fx.data,fx.u))
    sampler,pointwise,prior = query(fx,:sampler),query(fx,:pointwise),query(fx,:prior)
    source = string(Base.remove_linenums!(deepcopy(kernel_expr(fx.bound,fx.built.layout))))
    replayed = ReactiveKernelsPPL._eval_kernel_def(Meta.parse(source))
    replay = prepare_query((;spec=replayed,layout=fx.built.layout),fx.bound,:sampler)
    native_source = string(Base.remove_linenums!(deepcopy(code_expr(sampler))))
    native_replay = Core.eval(@__MODULE__,code_expr(sampler))
    @test !occursin("sampling_logdensity",source)
    @test !occursin("MixtureModel",string(code_expr(sampler)))
    @test !occursin("sampling_logdensity",string(code_expr(sampler)))
    if fx.kind === :builtin
        @test !occursin("sin(",native_source) # data-only curved column is bound once
    end
    for shift in (0.0,0.09)
        u = fx.u .+ shift
        expected = oracle(fx,u)
        @test Base.invokelatest(sampler,u) ≈ expected.total rtol=2e-12
        @test Base.invokelatest(replay,u) ≈ expected.total rtol=2e-12
        @test Base.invokelatest(native_replay,sampler.constants,sampler.callables,u) ≈ expected.total rtol=2e-12
        @test Base.invokelatest(pointwise,u).y ≈ expected.points rtol=2e-12
        @test Base.invokelatest(prior,u) ≈ expected.prior rtol=2e-12
    end
    if reverse
        prepared = prepare_sampler(fx.built,fx.bound,fx.u;
            backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        @test isempty(TypeStabilityAudit.findings(prepared.kernel, (Vector{Float64},)))
        @test isempty(TypeStabilityAudit.ad_call_findings(prepared.ad, fx.u))
        for shift in (0.0,0.09)
            u = fx.u .+ shift
            value,gradient = sampler_value_and_gradient!(prepared,similar(u),u)
            @test value ≈ oracle(fx,u).total
            @test gradient ≈ finite_gradient(v->oracle(fx,v).total,u) atol=2e-8 rtol=2e-7
        end
    end
    @test (fx.data,fx.u) == saved
    fx
end

@testset "native heterogeneous component graphs" begin
    @test Base.get_extension(ReactiveKernels,:ReactiveKernelsEnzymeExt) !== nothing
    @test Base.get_extension(ReactiveKernelsPPL,:ReactiveKernelsPPLDistributionsExt) !== nothing
    for kind in (:builtin,:student,:custom,:keyword), shared in (false,true)
        for n in (0,1,5)
            check(fixture(n;kind,shared);reverse=n==5)
        end
    end
    for weights in ([0.0,1.0],[1.0,0.0])
        check(fixture(3;kind=:custom,weights);reverse=true)
    end
    for kind in (:builtin,:custom)
        check(fixture(5;kind,rowweights=true);reverse=true)
    end
    small,large = fixture(3;kind=:custom),fixture(19;kind=:custom)
    @test recipe_count(small.built.spec) == recipe_count(large.built.spec)
    check(large)
    changed = merge(small.data,(;x=small.data.x .+ 0.4,y=small.data.y .- 0.2,
        scale=0.9,weights=[0.3,0.7]))
    rebound = bind_data(small.plan,changed)
    check(merge(small,(;data=changed,bound=rebound,built=build_kernel(rebound)));
        reverse=true)
    invalid = fixture(3;scale=-0.1)
    @test Base.invokelatest(query(invalid,:likelihood),invalid.u) == -Inf
end

@testset "heterogeneous mixture evidence" begin
    data = (;y=[-0.3,0.2,0.8])
    bound = bind_data(lower_rkppl(quote
        mu ~ Normal(0,1)
        y .~ truncated.(MixtureModel.(vcat.(Normal.(mu,0.7),Cauchy.(mu,1.2)),
            Ref([0.8,0.2])), -1.0, 1.0)
    end,data;mod=@__MODULE__,conditioned=(:y,)),data)
    built = build_kernel(bound)
    prepared = prepare_sampler(built,bound,[0.15];backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
    function expected(u)
        normal,cauchy = D.Normal(u[1],0.7),D.Cauchy(u[1],1.2)
        mass = 0.8*(D.cdf(normal,1)-D.cdf(normal,-1))+
            0.2*(D.cdf(cauchy,1)-D.cdf(cauchy,-1))
        D.logpdf(D.Normal(),u[1])+sum(log(0.8*D.pdf(normal,y)+0.2*D.pdf(cauchy,y))-log(mass)
            for y in data.y)
    end
    for u in ([0.15],[-0.2])
        value,gradient = sampler_value_and_gradient!(prepared,similar(u),u)
        @test value ≈ expected(u)
        @test gradient ≈ finite_gradient(expected,u) atol=2e-8 rtol=2e-7
    end
end

@testset "component graph work follows its dependencies" begin
    for n in (3,11), scope in (:bound,:shared,:row)
        data = (;x=[0.1i for i in 1:n], y=[0.2sin(i) for i in 1:n])
        scale = scope === :bound ? 0.2 : scope === :shared ? :s : :(s .+ x)
        bound = bind_data(lower_rkppl(quote
            s ~ Normal(0,1)
            y .~ MixtureModel.(vcat.(Normal.(0.0,1.0),
                CountedLaw.(0.0,$scale)), Ref([0.0,1.0]))
        end,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(bound)
        prepared = prepare_query(built,bound,:sampler)
        for s in (0.3,-0.2,800.0)
            NORMALIZERS[] = 0
            value = Base.invokelatest(prepared,[s])
            @test isfinite(value)
            @test NORMALIZERS[] == (scope === :bound ? 0 : scope === :shared ? 1 : n)
            if s == 800 && scope !== :bound
                # The component consumes log-scale directly; no exp(800)
                # physical scale may be introduced by mixture lowering.
                expected = D.logpdf(D.Normal(),s)-n*(s+log(2pi)/2)-
                    (scope === :row ? sum(data.x) : 0.0)
                @test value ≈ expected
            end
        end
    end
end
end
