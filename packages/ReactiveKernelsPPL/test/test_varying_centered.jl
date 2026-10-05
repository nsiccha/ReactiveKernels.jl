using DifferentiationInterface: AutoEnzyme, gradient
using Distributions: MvNormal, Normal, Exponential, logpdf
using LinearAlgebra: Diagonal, Symmetric
using InteractiveUtils: code_llvm
using ReactiveKernels, ReactiveKernelsPPL, Test
import Enzyme

# Centered varying coefficients use ordinary multivariate row priors.
function _rows_prior_point(k, g)
    B = [.1sin(i + 3j) for i in 1:g, j in 1:k]
    mu = [.05j - .1 for j in 1:k]
    F = [i == j ? .8 + .05i : i > j ? .07cos(i - j) : 0. for i in 1:k, j in 1:k]
    return B, mu, F
end

_rows_prior_value(B, mu, F) =
    ReactiveKernelsPPL._mvnormal_cholesky_slices_logpdf(
        ReactiveKernelsPPL._SliceRows(), B, mu, F)

@testset "row-wise MvNormalCholesky keeps a runtime margin loop" begin
    io = IOBuffer()
    code_llvm(io, ReactiveKernelsPPL._lower_solve_rows_logpdf,
        Tuple{Matrix{Float64},Matrix{Float64},Float64}; debuginfo=:none)
    ir = String(take!(io))
    @test occursin(" phi i64 ", ir) && occursin("br i1", ir)
end

@testset "row-wise MvNormalCholesky density and ordinary native reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for (k, g) in ((1, 0), (1, 3), (2, 3), (13, 4), (13, 30))
        B, mu, F = _rows_prior_point(k, g)
        oracle(B, mu, F) =
            sum((logpdf(MvNormal(mu, F * F'), B[j, :]) for j in 1:g); init=0.0)
        @test _rows_prior_value(B, mu, F) ≈ oracle(B, mu, F) rtol=4e-14
        q = vcat(vec(B), mu, vec(F))
        unpack(q) = (reshape(q[1:g*k], g, k), q[g*k+1:g*k+k],
            reshape(q[g*k+k+1:end], k, k))
        objective(q) = _rows_prior_value(unpack(q)...)
        free = [i for i in eachindex(q) if i <= g*k + k ||
            (r = (i - g*k - k - 1) % k + 1; c = (i - g*k - k - 1) ÷ k + 1; r >= c)]
        @test gradient(objective, backend, q)[free] ≈
            _findiff_grad(p -> oracle(unpack(p)...), q)[free] rtol=2e-6 atol=3e-8
    end
    # The same rows as columns, and through the full covariance.
    for (k, g) in ((1, 3), (13, 4))
        B, mu, F = _rows_prior_point(k, g)
        @test ReactiveKernelsPPL._mvnormal_cholesky_slices_logpdf(
            ReactiveKernelsPPL._SliceCols(), permutedims(B), mu, F) ≈
            _rows_prior_value(B, mu, F) rtol=1e-14
        @test ReactiveKernelsPPL._mvnormal_slices_logpdf(
            ReactiveKernelsPPL._SliceRows(), B, mu, F * F') ≈
            _rows_prior_value(B, mu, F) rtol=1e-10
    end
    B, mu, F = _rows_prior_point(2, 3)
    # refused: row, mean and covariance-factor dimensions must agree (distribution domain, P3)
    @test_throws DimensionMismatch _rows_prior_value(B, [0.], F)
    # refused: row, mean and covariance-factor dimensions must agree (distribution domain, P3)
    @test_throws DimensionMismatch _rows_prior_value(B, mu, ones(3, 3))
    # refused: a Cholesky factor must be lower triangular with positive diagonal (distribution domain, P3)
    @test_throws ArgumentError _rows_prior_value(B, mu, [1. 0.; .5 -1.])
    # refused: a Cholesky factor must be lower triangular with positive diagonal (distribution domain, P3)
    @test_throws ArgumentError _rows_prior_value(B, mu, [1. .2; .5 1.])
    @test_throws ArgumentError ReactiveKernelsPPL._mvnormal_slices_logpdf(
        ReactiveKernelsPPL._SliceRows(), B, mu, [1. .2; .5 1.])
    @test_throws ArgumentError ReactiveKernelsPPL._mvnormal_slices_logpdf(
        ReactiveKernelsPPL._SliceRows(), B, mu, [1. 2.; 2. 1.])
    Sigma = F * F'
    @test ReactiveKernelsPPL._mvnormal_slices_logpdf(
        ReactiveKernelsPPL._SliceRows(), B, mu, Symmetric(Sigma)) ≈
        _rows_prior_value(B, mu, F) rtol = 1e-12
end

function _centered_test_plan()
    return lower_rkppl(quote
        a ~ Normal(0.,1.)
        sd[1:2] .~ Exponential.(2.)
        L ~ LKJCholesky(2,1.5)
        F = sd .* L
        eachrow(B[levels(group),1:2]) .~ MvNormalCholesky(zeros(2),F)
        mu = a .+ B[group,1] .+ x .* B[group,2]
        y .~ Normal.(mu,1.5)
    end,(:y,:group,:x); conditioned = (:y,:group,:x))
end

function _centered_plan_oracle(u,layout,columns)
    nt = constrain(layout,u)
    a,L,tau,b = nt.a,nt.L,nt.sd,nt.B
    groups = sort(unique(columns[:group]))
    means = [a+b[findfirst(==(group),groups),1]+
        b[findfirst(==(group),groups),2]*x for (group,x) in zip(columns[:group],columns[:x])]
    factor = Diagonal(tau)*L
    prior = sum(logpdf(Exponential(2.),s) for s in tau)+
        sum(logpdf(MvNormal(factor*factor'),vec(b[s,:])) for s in axes(b,1))+
        logpdf(Normal(0.,1.),a)+lkj_logconst(2,1.5)+(2*1.5-2)*log(L[2,2])
    return prior+sum(logpdf.(Normal.(means,1.5),columns[:y]))+logjac(layout,u)
end

@testset "centered row prior IR, layout, emitted prior and likelihood" begin
    plan = _centered_test_plan()
    @test isempty(plan.varying_draws)
    @test validate_structure(plan) === nothing
    columns = Dict{Symbol,AbstractVector}(:group=>[3,1,2,3,1],
        :x=>[.2,1.,-.5,.3,.1],:y=>[.5,.2,-.1,.4,.3])
    bound = bind_data(plan,columns)
    built = build_kernel(bound)
    base = constrain(built.layout,zeros(built.layout.total))
    B = [.1 -.15;.2 .25;-.1 .3]
    u = unconstrain(built.layout,merge(base,(a=.2,sd=[.7,.9],
        L=lkj_chol_constrain([.35],2),B=B)))
    @test length(u) == built.layout.total == 10
    nt = constrain(built.layout,u)
    @test nt.B == B
    @test unconstrain(built.layout,nt) ≈ u
    query = prepare_query(built,bound,:sampler)
    @test query(u) ≈ _centered_plan_oracle(u,built.layout,columns) rtol=4e-13
    sampler = prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
    g = zeros(length(u))
    value,_ = sampler_value_and_gradient!(sampler,g,u)
    @test value ≈ query(u) rtol=2e-13
    @test g ≈ _findiff_grad(p -> _centered_plan_oracle(p,built.layout,columns),u) rtol=2e-6 atol=3e-8
end
