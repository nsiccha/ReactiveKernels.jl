using DifferentiationInterface: AutoEnzyme, SecondOrder
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

module ExternalGeometryDerivativeFixtures
using Distributions
using ReactiveKernelsPPL

struct LowerLaw{L,M,S}
    lower::L
    mu::M
    scale::S
end

ReactiveKernelsPPL.sampling_geometry(::Type{<:LowerLaw}, args, shape) =
    ParameterGeometry(shape,
        isempty(shape) ? 1 : Expr(:call, :*, (:(length($axis)) for axis in shape)...);
        support=:real,
        constrain=(u, shape, lo, rest...) -> isempty(shape) ?
            lo + exp(only(u)) : reshape(lo .+ exp.(u), shape),
        unconstrain=(x, shape, lo, rest...) -> isempty(shape) ?
            [log(x-lo)] : vec(log.(x .- lo)),
        logjac=(u, shape, rest...) -> sum(u))

@inline function ReactiveKernelsPPL.sampling_logdensity(d::LowerLaw, x::AbstractArray)
    result = zero(eltype(x))
    for j in eachindex(x)
        result += logpdf(Normal(d.mu[j], d.scale), x[j])
    end
    return result
end

const source = quote
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    location = mu .* weights
    x[axes(lo, 1)] ~ LowerLaw(lo, location[rows], sigma)
    y ~ Normal(mu + sum(x), 1)
end
end

@testset "custom geometry HVP with live vector and scalar parents" begin
    second_order = SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse),
        AutoEnzyme(; mode=Enzyme.Forward))
    for n in (0, 2, 7)
        data = (; lo=fill(1.0, n), y=1.0, weights=ones(max(2, n)), rows=collect(1:n))
        plan = bind_data(lower_rkppl(ExternalGeometryDerivativeFixtures.source, data;
            conditioned=(:y,), mod=ExternalGeometryDerivativeFixtures), data)
        built = build_kernel(plan)
        u = zeros(n + 2)
        v = ones(n + 2)
        # Prepare the HVP first: no previous gradient or inference probe should
        # be required to differentiate the caller-owned geometry endpoints.
        query = prepare_query_ad(prepare_ad_hvp, built, plan, :sampler,
            second_order, (v,), u)
        gradient, (hv,) = ad_gradient_and_hvp(query, (v,), u)
        # At u=0, x[j]=2, sigma=1 and y-mu-sum(x)=1-2n. Differentiating
        # the Normal priors, observation and transform Jacobians gives these
        # vectors directly, without constructing a dense Hessian.
        @test gradient ≈ vcat(1.0, 3n, fill(-2.0n, n))
        @test hv ≈ vcat(-2.0-5n, -1.0-8n, fill(2.0-3n, n))
        @test u == zeros(n + 2)
        @test v == ones(n + 2)
    end
end
