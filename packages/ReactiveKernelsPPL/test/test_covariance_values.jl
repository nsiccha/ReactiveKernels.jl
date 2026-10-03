using Distributions
using LinearAlgebra
using ReactiveKernelsPPL
using Test

function _cv_joint_build(n; sampled = true)
    ast = quote
        $(sampled ? :(eta ~ Exponential(1)) : :(eta = 2.3))
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sd[1:2] .~ Gamma.(2, 1)
        C ~ LKJCholesky(2, eta)
        F = sd .* C
        mu1 = a .+ b .* x
        mu2 = b .- a .* x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], F)
    end
    data = Dict(:x => [0.1i for i in 1:n],
        :y1 => [0.2 * sin(i) for i in 1:n], :y2 => [0.3 * cos(i) for i in 1:n])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y1, :y2)), data)
    return bound, build_kernel(bound)
end

function _cv_joint_oracle(bound, nt; sampled = true)
    eta = sampled ? nt.eta : 2.3
    prior = logpdf(Normal(), nt.a) + logpdf(Normal(), nt.b) +
        sum(logpdf.(Gamma(2, 1), nt.sd)) +
        logpdf(LKJCholesky(2, eta), Cholesky(LowerTriangular(nt.C)))
    sampled && (prior += logpdf(Exponential(1), nt.eta))
    F = Diagonal(nt.sd) * nt.C
    c = bound.columns
    ll = sum(logpdf(MvNormal([nt.a + nt.b * c[:x][i],
        nt.b - nt.a * c[:x][i]], F * F'), [c[:y1][i], c[:y2][i]])
        for i in eachindex(c[:x]))
    return prior, ll
end

@testset "Covariance factors are explicit prior values" begin
    for sampled in (false, true), n in (7, 19)
        bound, built = _cv_joint_build(n; sampled)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        prior, ll = _cv_joint_oracle(bound, nt; sampled)
        @test _query(built.spec, bound, :prior, u) ≈ prior
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test unconstrain(built.layout, nt) ≈ u
        _check_gradient(built.spec, bound, u)
    end
    for declaration in (:(sd[1:3] .~ Gamma.(2, 1)),
            :(sd[1:2] .~ Normal.(0, 1)))
        ast = quote
            $declaration
            C ~ LKJCholesky(2, 2.3)
            F = sd .* C
            [y1, y2] ~ MvNormalCholesky([0.0, 0.0], F)
        end
        @test_throws ContractValidationError lower_rkppl(ast, (:y1, :y2);
            conditioned = (:y1, :y2))
    end
end
