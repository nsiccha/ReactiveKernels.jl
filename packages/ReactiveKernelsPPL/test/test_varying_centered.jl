using DifferentiationInterface: AutoEnzyme, Constant, gradient
using Distributions: MvNormal, Normal, Exponential, logpdf
using LinearAlgebra: Diagonal
using InteractiveUtils: code_llvm
using ReactiveKernels, ReactiveKernelsPPL, Test
import Enzyme

function _centered_prior_point(k,g)
    lower = [i==j ? 1.0 + .03i : .04sin(i+j) for i in 1:k for j in 1:i]
    return vcat(.1sin.(collect(1:k*g)),log.(1 .+ .05collect(1:k)),lower)
end

function _centered_prior_value(q,k,g)
    return ReactiveKernelsPPL._centered_correlated_logpdf(q[1:k*g],
        exp.(q[(k*g+1):(k*g+k)]),q[(k*g+k+1):end])
end

@testset "centered prior keeps runtime group and margin loops" begin
    ir = String[]
    for groups in (3,30)
        q = _centered_prior_point(13,groups)
        io = IOBuffer()
        code_llvm(io,_centered_prior_value,Tuple{typeof(q),Int,Int};debuginfo=:none)
        push!(ir,String(take!(io)))
    end
    normalized = [replace(v,r"(?<=_)\d+(?=\"?\()"=>"JIT") for v in ir]
    @test normalized[1] == normalized[2]
    @test occursin(" phi i64 ",ir[1]) && occursin("br i1",ir[1])
end

function _centered_prior_oracle(q,k,g)
    L = zeros(k,k)
    lower = q[(k*g+k+1):end]
    for i in 1:k, j in 1:i
        L[i,j] = lower[i*(i-1)÷2+j]
    end
    A = Diagonal(exp.(q[(k*g+1):(k*g+k)]))*L
    density = MvNormal(A*A')
    return sum(logpdf(density,q[((s-1)*k+1):(s*k)]) for s in 1:g)
end

@testset "centered correlated density and ordinary native reverse" begin
    backend = AutoEnzyme(;mode=Enzyme.Reverse)
    for (k,g) in ((1,3),(2,3),(13,3))
        q = _centered_prior_point(k,g)
        objective(p,k,g) = _centered_prior_value(p,k,g)
        @test objective(q,k,g) ≈ _centered_prior_oracle(q,k,g) rtol=4e-14
        @test gradient(objective,backend,q,Constant(k),Constant(g)) ≈
            _transit_fd_gradient(p -> _centered_prior_oracle(p,k,g),q) rtol=2e-6 atol=3e-8
    end
    @test_throws DimensionMismatch ReactiveKernelsPPL._centered_correlated_logpdf(ones(3),ones(2),ones(3))
    @test_throws ArgumentError ReactiveKernelsPPL._centered_correlated_logpdf(ones(2),[-1.,1.],ones(3))
end

function _centered_test_plan()
    return lower_rkppl(quote
        a ~ Normal(0.,1.)
        d ~ varying_draws(group,[1,x];centered=true,eta=1.5,sd=Exponential(2.))
        r ~ varying_slice(d,1:2)
        mu = a .+ r
        y .~ Normal.(mu,1.5)
    end,(:y,:group,:x))
end

function _centered_plan_oracle(u,layout,columns)
    nt = constrain(layout,u)
    a,L,tau,b = nt.mu[1],nt.L_group,nt.tau_group,nt.b_group
    groups = sort(unique(columns[:group]))
    means = [a+b[findfirst(==(group),groups),1]+
        b[findfirst(==(group),groups),2]*x for (group,x) in zip(columns[:group],columns[:x])]
    factor = Diagonal(tau)*L
    prior = sum(logpdf(Exponential(2.),s) for s in tau)+
        sum(logpdf(MvNormal(factor*factor'),vec(b[s,:])) for s in axes(b,1))+
        logpdf(Normal(0.,1.),a)+lkj_logconst(2,1.5)+(2*1.5-2)*log(L[2,2])
    return prior+sum(logpdf.(Normal.(means,1.5),columns[:y]))+logjac(layout,u)
end

@testset "centered correlated IR, layout, emitted prior and likelihood" begin
    plan = _centered_test_plan()
    @test only(plan.varying_draws).kind === :centered_correlated
    @test ReactiveKernelsPPL._varying_corr_names(only(plan.varying_draws)) ==
        (:L_group,:tau_group,:b_flat_group)
    @test validate_structure(plan) === nothing
    columns = Dict{Symbol,AbstractVector}(:group=>[3,1,2,3,1],
        :x=>[.2,1.,-.5,.3,.1],:y=>[.5,.2,-.1,.4,.3])
    bound = bind_data(plan,columns)
    built = build_kernel(bound)
    u = [.2,.35,log(.7),log(.9),.1,-.15,.2,.25,-.1,.3]
    @test length(u) == built.layout.total
    nt = constrain(built.layout,u)
    @test nt.b_group == [.1 -.15;.2 .25;-.1 .3]
    @test unconstrain(built.layout,nt) ≈ u
    query = prepare_query(built,bound,:sampler)
    @test query(u) ≈ _centered_plan_oracle(u,built.layout,columns) rtol=4e-13
    sampler = prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
    g = zeros(length(u))
    value,_ = sampler_value_and_gradient!(sampler,g,u)
    @test value ≈ query(u) rtol=2e-13
    @test g ≈ _transit_fd_gradient(p -> _centered_plan_oracle(p,built.layout,columns),u) rtol=2e-6 atol=3e-8
    @test_throws "Bool literal" lower_rkppl(quote
        d ~ varying_draws(group,[1,x];centered=1)
        r ~ varying_slice(d,1:2)
        mu = r
        y .~ Normal.(mu,1.)
    end,(:y,:group,:x))
end
