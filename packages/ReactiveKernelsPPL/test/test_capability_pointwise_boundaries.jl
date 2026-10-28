using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_pointwise_boundary(kind, K=2)
    if kind === :observation
        ast = quote a ~ Normal(0,1); y .~ Normal.(a,1) end
        data = (; y=Float64[])
        expected = (; y=Float64[])
    elseif kind === :elementwise
        ast = quote a ~ Normal(0,1); Y[1:2,1:0] .~ Normal.(a,1) end
        data = (; Y=zeros(2,0))
        expected = data
    else
        L = lkj_chol_constrain(fill(0.2,K*(K-1)÷2),K)
        ast = quote a ~ Normal(0,1); L ~ LKJCholesky($K,2.3) end
        data = (; L)
        expected = (; L=K == 1 ? 0.0 : logpdf(LKJCholesky(K,2.3),Cholesky(LowerTriangular(L))))
    end
    plan = bind_data(lower_rkppl(ast,data; conditioned=keys(data)),Dict{Symbol,Any}(pairs(data)))
    built = build_kernel(plan)
    u = [0.3]
    pointwise = prepare_query(built,plan,:pointwise)
    sampler = prepare_sampler(built,plan,u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    return (; plan,built,u,pointwise,sampler,expected)
end

@testset "pointwise empty domains and conditioned LKJ factors" begin
    for (kind,K) in ((:observation,2),(:elementwise,2),(:lkj,1),(:lkj,2),(:lkj,5))
        fx = _cap_pointwise_boundary(kind,K)
        values = Base.invokelatest(fx.pointwise,fx.u)
        @test keys(values) == keys(fx.expected)
        @test all(size(values[k]) == size(fx.expected[k]) for k in keys(values))
        @test all(values[k] ≈ fx.expected[k] for k in keys(values))
        likelihood = sum((sum(v) for v in Base.values(fx.expected)); init=0.0)
        @test Base.invokelatest(prepare_query(fx.built,fx.plan,:likelihood),fx.u) ≈ likelihood
        value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
        @test value ≈ logpdf(Normal(),fx.u[1])+likelihood
        @test gradient ≈ -fx.u
    end
end
