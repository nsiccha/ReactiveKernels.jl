module OwnedDensityGraphTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import DifferentiationInterface as DI
import Enzyme

# A normalized triangular Gaussian law, owned entirely by this caller. Identity
# coordinates do not make its density independent across adjacent coordinates.
struct ChainLaw{T,A,B,C,D}
    location::A
    previous::B
    rho::C
    scale::D
end
ChainLaw{T}(a,b,c,d) where T = ChainLaw{T,typeof(a),typeof(b),typeof(c),typeof(d)}(a,b,c,d)
ChainLaw(a,b,c,d) = ChainLaw{Float64}(a,b,c,d)
const AliasLaw = ChainLaw

struct LocalLaw end
LocalLaw(args...) = ChainLaw(args...)
struct KeywordLaw end
KeywordLaw(a,b; rho=0.3,scale=1.0) = ChainLaw(a,b,rho,scale)
struct ObjectLaw end
ObjectLaw(args...) = ChainLaw(args...)
module Imported
import ..OwnedDensityGraphTests: ChainLaw
end

@kernel chain_density(value, location, previous, rho, scale) = begin
    center = sin.(location)
    centered = value .- center
    padded = vcat(0.0, centered)
    lagged = padded[previous]
    residual = centered .- rho .* lagged
    normalizer = length(location)*(log(2pi)/2 + log(scale))
    return -normalizer - sum(abs2, residual)/(2*scale^2)
end
@kernel coordinates(u, shape, location, previous, rho, scale) = begin
    return u
end
@kernel jacobian(u, shape, location, previous, rho, scale) = begin
    return 0.0
end
function geometry(args, shape; constrain=coordinates, inverse=coordinates, jac=jacobian)
    positional = filter(a->!Meta.isexpr(a,:parameters), args)
    dims = isempty(shape) ? (:(length($(positional[1]))),) : shape
    ParameterGeometry(dims, :(length($(positional[1]))); support=:real,
        constrain, unconstrain=inverse, logjac=jac)
end
ReactiveKernelsPPL.sampling_geometry(::Type{ChainLaw}, args, shape) = geometry(args,shape)
ReactiveKernelsPPL.sampling_geometry(::Type{ChainLaw{T}}, args, shape) where T = geometry(args,shape)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{ChainLaw}, args::Tuple) = chain_density
ReactiveKernelsPPL.sampling_distribution_graph(::Type{ChainLaw{T}}, args::Tuple) where T = chain_density
ReactiveKernelsPPL.sampling_geometry(::Type{LocalLaw}, args, shape) = geometry(args,shape)
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{LocalLaw}, args::Tuple)
    # No global binding for this spec: source replay must reconstruct it.
    @kernel local_density(value,location,previous,rho,scale) = begin
        return chain_density(value,location,previous,rho,scale)
    end
    local_density
end
@kernel keyword_density(value,location,previous; rho=0.3,scale=1.0) = begin
    return chain_density(value,location,previous,rho,scale)
end
@kernel keyword_coordinates(u,shape,location,previous; rho=0.3,scale=1.0) = begin
    return u
end
@kernel keyword_jacobian(u,shape,location,previous; rho=0.3,scale=1.0) = begin
    return 0.0
end
ReactiveKernelsPPL.sampling_geometry(::Type{KeywordLaw},args,shape) =
    geometry(args,shape;constrain=keyword_coordinates,inverse=keyword_coordinates,jac=keyword_jacobian)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{KeywordLaw},args::Tuple) = keyword_density
@kernel chain_object(location,previous,rho,scale) = begin
    logpdf(value) = chain_density(value,location,previous,rho,scale)
end
ReactiveKernelsPPL.sampling_geometry(::Type{ObjectLaw},args,shape) = geometry(args,shape)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{ObjectLaw},args::Tuple) =
    Expr(:call,GlobalRef(@__MODULE__,:chain_object),args...)

# Independent loop oracle, including the normalization. No call to this
# function should remain in the composed model graph.
function score(value, location, previous, rho, scale)
    total = -length(value)*(log(2pi)/2 + log(scale))
    for i in eachindex(value)
        j = previous[i]-1
        lag = j == 0 ? 0.0 : value[j]-sin(location[j])
        residual = value[i]-sin(location[i])-rho*lag
        total -= residual^2/(2scale^2)
    end
    total
end
ReactiveKernelsPPL.sampling_logdensity(d::ChainLaw, value) =
    score(value,d.location,d.previous,d.rho,d.scale)

function fixture(n; head=:ChainLaw, cells=false, keyword=false)
    location = cells ? [[0.1i for i in 1:m] for m in n] : [0.1i for i in 1:n]
    previous = cells ? [collect(1:m) for m in n] : collect(1:n)
    y = cells ? [[0.2cos(i) for i in 1:m] for m in n] : [0.2cos(i) for i in 1:n]
    data = (;location,previous,y)
    rhs = keyword ? :(KeywordLaw(location,previous;scale=exp(a),rho=rho)) :
        :($head(location,previous,rho,exp(a)))
    cellrhs = keyword ? :(KeywordLaw(location[i],previous[i];scale=exp(a),rho=rho)) :
        :($head(location[i],previous[i],rho,exp(a)))
    ast = cells ? quote
        a ~ Normal(0,1)
        rho ~ Normal(0,1)
        @plate for i in eachindex(location)
            w[i] ~ $cellrhs
            y[i] .~ Normal.(w[i],1)
        end
    end : quote
        a ~ Normal(0,1)
        rho ~ Normal(0,1)
        w[1:length(location)] ~ $rhs
        y .~ Normal.(w,1)
    end
    bound = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
    built = build_kernel(bound)
    w = cells ? [[0.04i for i in 1:m] for m in n] : [0.04i for i in 1:n]
    u = unconstrain(built.layout,(;a=0.15,rho=0.3,w))
    (;data,bound,built,u,cells)
end
function oracle(fx,u)
    nt = constrain(fx.built.layout,u)
    prior = -log(2pi)-(nt.a^2+nt.rho^2)/2
    if fx.cells
        for i in eachindex(nt.w)
            prior += score(nt.w[i],fx.data.location[i],fx.data.previous[i],nt.rho,exp(nt.a))
        end
        likelihood = sum((sum(-log(2pi)/2 .- (fx.data.y[i].-nt.w[i]).^2 ./ 2)
            for i in eachindex(nt.w));init=0.0)
    else
        prior += score(nt.w,fx.data.location,fx.data.previous,nt.rho,exp(nt.a))
        likelihood = sum(-log(2pi)/2 .- (fx.data.y.-nt.w).^2 ./ 2)
    end
    (;prior,likelihood,total=prior+likelihood)
end
function finite_gradient(f,u)
    map(eachindex(u)) do i
        p,m = copy(u),copy(u)
        p[i] += 1e-5; m[i] -= 1e-5
        (f(p)-f(m))/2e-5
    end
end
recipe_count(spec::KernelSpec) = recipe_count(spec.graph)
function recipe_count(graph)
    sum(graph.recipes;init=0) do recipe
        1 + (recipe_kind(recipe) === :plate ? recipe_count(plate_body(recipe)) : 0)
    end
end
function check(fx; reverse=false, direct=true)
    saved = deepcopy((fx.data,fx.u))
    source = kernel_expr(fx.bound,fx.built.layout)
    @test !occursin("sampling_logdensity",string(source))
    direct && @test occursin("_sampling_distribution_endpoint",string(source))
    replay = ReactiveKernelsPPL._eval_kernel_def(
        Meta.parse(string(Base.remove_linenums!(deepcopy(source)))))
    replay_query = prepare_query((;spec=replay,layout=fx.built.layout),fx.bound,:sampler)
    queries = map(cut->prepare_query(fx.built,fx.bound,cut),(:prior,:likelihood,:sampler))
    points = Base.invokelatest(prepare_query(fx.built,fx.bound,:pointwise),fx.u)
    @test keys(points) == (:y,)
    # The normalized joint law belongs only to the prior cut. It must never
    # become synthetic observations or per-coordinate likelihood factors.
    point_total = fx.cells ? sum(sum,points.y;init=0.0) : sum(points.y)
    @test point_total ≈ oracle(fx,fx.u).likelihood
    native = string(code_expr(queries[3]))
    @test !occursin("sampling_logdensity",native)
    @test !occursin("chain_density",native)
    # This data-only broadcast belongs to preparation, not each evaluation.
    @test !occursin("sin(",native)
    for shift in (0.0,0.07)
        u = fx.u .+ shift
        expected = oracle(fx,u)
        @test unconstrain(fx.built.layout,constrain(fx.built.layout,u)) ≈ u
        @test logjac(fx.built.layout,u) == 0.0
        @test Base.invokelatest(queries[1],u) ≈ expected.prior
        @test Base.invokelatest(queries[2],u) ≈ expected.likelihood
        @test Base.invokelatest(queries[3],u) ≈ expected.total
        @test Base.invokelatest(replay_query,u) ≈ expected.total
    end
    if reverse
        @test Base.get_extension(ReactiveKernels,:ReactiveKernelsEnzymeExt) !== nothing
        sampler = prepare_sampler(fx.built,fx.bound,fx.u;
            backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler,similar(fx.u),fx.u)
        @test value ≈ oracle(fx,fx.u).total
        @test gradient ≈ finite_gradient(u->oracle(fx,u).total,fx.u) atol=2e-8 rtol=2e-7
    end
    @test (fx.data,fx.u) == saved
    fx
end

@testset "caller-owned normalized joint density graphs" begin
    sizes = map((0,3,19)) do n
        fx = check(fixture(n); reverse=n==3)
        recipe_count(fx.built.spec)
    end
    @test sizes[2] == sizes[3]
    for head in (:AliasLaw, :LocalLaw, GlobalRef(@__MODULE__,:ChainLaw),
            :(OwnedDensityGraphTests.ChainLaw), :(Imported.ChainLaw), :(ChainLaw{Float64}))
        check(fixture(3;head))
    end
    for ns in ([2,1,3], [0,2])
        check(fixture(ns;cells=true); reverse=ns==[2,1,3])
    end
    check(fixture(3;keyword=true);reverse=true)
    check(fixture([2,1];keyword=true,cells=true);reverse=true)
    for head in (:LocalLaw, :(ChainLaw{Float64}))
        check(fixture([2,1];head,cells=true))
    end
    check(fixture(3;head=:ObjectLaw);direct=false,reverse=true)
    check(fixture([2,1];head=:ObjectLaw,cells=true);direct=false)
    small = fixture([2,1];cells=true)
    large = fixture(fill(3,19);cells=true)
    @test recipe_count(small.built.spec) == recipe_count(large.built.spec)
end

@testset "caller-owned density graphs for whole-event observations" begin
    for cells in (false,true)
        data = cells ? (;location=[[0.1,0.2],[0.3]],previous=[[1,2],[1]],y=[[0.3,0.4],[0.1]]) :
            (;location=[0.1,0.2],previous=[1,2],y=[0.3,0.4])
        observation = cells ? :(@plate for i in eachindex(location)
            y[i] ~ LocalLaw(location[i],previous[i],rho,exp(a))
        end) : :(y ~ LocalLaw(location,previous,rho,exp(a)))
        bound = bind_data(lower_rkppl(quote
            a ~ Normal(0,1)
            rho ~ Normal(0,1)
            $observation
        end,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(bound)
        source = string(Base.remove_linenums!(kernel_expr(bound,built.layout)))
        @test !occursin("sampling_logdensity",source)
        replay = ReactiveKernelsPPL._eval_kernel_def(Meta.parse(source))
        replay_query = prepare_query((;spec=replay,layout=built.layout),bound,:sampler)
        function expected(u)
            prior = -log(2pi)-(u[1]^2+u[2]^2)/2
            likelihood = cells ? sum(score(data.y[i],data.location[i],data.previous[i],u[2],exp(u[1]))
                for i in eachindex(data.y)) : score(data.y,data.location,data.previous,u[2],exp(u[1]))
            prior+likelihood
        end
        u = [0.15,0.3]
        sampler = prepare_sampler(built,bound,u;backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler,similar(u),u)
        @test value ≈ expected(u)
        @test Base.invokelatest(replay_query,u) ≈ expected(u)
        @test gradient ≈ finite_gradient(expected,u) atol=2e-8 rtol=2e-7
        @test coordinate_names(built.layout) == [:a,:rho]
    end
end

@testset "omitted density keywords preserve ordinary constructor defaults" begin
    data = (;location=[0.1,0.2],previous=[1,2],y=[0.3,0.4])
    bound = bind_data(lower_rkppl(quote
        a ~ Normal(0,1)
        y ~ KeywordLaw(location,previous;scale=exp(a))
    end,data;mod=@__MODULE__,conditioned=(:y,)),data)
    built = build_kernel(bound)
    @test occursin("sampling_logdensity",string(kernel_expr(bound,built.layout)))
    q = prepare_query(built,bound,:sampler)
    @test Base.invokelatest(q,[0.2]) ≈ -log(2pi)/2-0.2^2/2+
        score(data.y,data.location,data.previous,0.3,exp(0.2))
end
end
