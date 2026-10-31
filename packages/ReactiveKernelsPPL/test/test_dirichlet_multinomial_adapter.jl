using Test, ReactiveKernels, ReactiveKernelsPPL
import Distributions, SpecialFunctions, Enzyme
import DifferentiationInterface as DI

function _dm_adapter_density(alpha, n, y)
    sampling_logdensity(Distributions.DirichletMultinomial(n, alpha), y)
end
function _dm_adapter_reference(alpha, n, y)
    a0 = sum(alpha)
    SpecialFunctions.loggamma(n+1) + SpecialFunctions.loggamma(a0) -
        SpecialFunctions.loggamma(n+a0) + sum(eachindex(y)) do i
            SpecialFunctions.loggamma(y[i]+alpha[i]) -
                SpecialFunctions.loggamma(y[i]+1) - SpecialFunctions.loggamma(alpha[i])
        end
end
function _dm_adapter_gradient(alpha, n, y)
    a0 = sum(alpha)
    [SpecialFunctions.digamma(a0)-SpecialFunctions.digamma(n+a0) +
        SpecialFunctions.digamma(y[i]+alpha[i])-SpecialFunctions.digamma(alpha[i])
        for i in eachindex(y)]
end

@testset "DirichletMultinomial adapter support and native gradients" begin
    # Independent gamma/digamma oracles; the adapter calls the upstream law.
    for alpha in ([0.2,0.5,0.3], [1.2,2.5,0.7]), y in ([2,3,1], [0,6,0])
        n = sum(y)
        gradient = zeros(length(alpha))
        before_alpha, before_y = copy(alpha), copy(y)
        Enzyme.autodiff(Enzyme.Reverse, _dm_adapter_density, Enzyme.Active,
            Enzyme.Duplicated(alpha, gradient), Enzyme.Const(n), Enzyme.Const(y))
        oracle = _dm_adapter_gradient(alpha, n, y)
        @test _dm_adapter_density(alpha,n,y) ≈ _dm_adapter_reference(alpha,n,y)
        @test gradient ≈ oracle atol=1e-10 rtol=1e-10
        @test alpha == before_alpha
        @test y == before_y
        for bad_n in (n-1,n+1)
            @test _dm_adapter_density(alpha,bad_n,y) == -Inf
        end
    end
    for y in ([1.5,3.5,1.0], [-1,6,1], [NaN,3,3], [Inf,3,3])
        @test _dm_adapter_density([0.2,0.5,0.3],6,y) == -Inf
    end
    for y in ([2,3], [2,3,1,0])
        @test_throws DimensionMismatch _dm_adapter_density([0.2,0.5,0.3],6,y)
    end
    for n in (5,6,7)
        y = [2,3,1]
        data = Dict{Symbol,Any}(:y=>y,:n=>n)
        before = deepcopy(data)
        ast = quote
            alpha[1:3] .~ Exponential(1)
            y ~ Distributions.DirichletMultinomial(n, alpha)
        end
        plan = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=keys(data)), data)
        built = build_kernel(plan)
        u = log.([0.2,0.5,0.3])
        query = prepare_query(built,plan,:sampler)
        value = Base.invokelatest(query,u)
        if n == 6
            sampler = prepare_sampler(built,plan,u;backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
            value, g = sampler_value_and_gradient!(sampler,zeros(3),u)
            alpha = exp.(u)
            ref = _dm_adapter_reference(alpha,n,y)-sum(alpha)+sum(u)
            oracle = alpha .* _dm_adapter_gradient(alpha, n, y) .- alpha .+ 1
            @test value ≈ ref
            @test g ≈ oracle atol=1e-10 rtol=1e-10
        else
            @test value == -Inf
        end
        @test data == before
        @test u == log.([0.2,0.5,0.3])
    end
end
