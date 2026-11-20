module StudentKernelTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
import ReactiveKernelsDistributionKernels.DistributionKernelSources as DKS
import DifferentiationInterface as DI
import Distributions, Enzyme, SpecialFunctions

# `log1p(z^2 / nu)` in extended precision.
_reference(nu, z) = setprecision(BigFloat, 512) do
    Float64(log1p(big(z)^2 / big(nu)))
end

@kernel student_events(xs::Vector{Float64}, q::Vector{Float64}) = begin
    scores = plate(xs, Ref(q)) do x, q
        DKS.student_t(exp(q[1]), q[2], exp(q[3])).logpdf(x)
    end
    return sum(scores)
end
function _student_events(xs, q)
    nu, location, scale = exp(q[1]), q[2], exp(q[3])
    sum(Distributions.logpdf(Distributions.TDist(nu), (x - location) / scale) - q[3] for x in xs)
end

@testset "Student-t kernel is a shared factor times one logarithm" begin
    # The logarithm, on both sides of sqrt(nu) and where z^2 overflows.
    for nu in (0.3, 1.0, 4.0, 30.0, 1e6),
            z in (0.0, 1e-200, 1e-8, 0.1, 1.0, 3.0, -2.5, 50.0, 1e8, 1e150, 1e154, 1e160, -1e200, 1e300)
        @test DKS._student_log1p(nu, z) ≈ _reference(nu, z) rtol = 4eps()
    end
    @test DKS._student_log1p(3.0, Inf) == Inf
    @test DKS._student_log1p(3.0, -Inf) == Inf
    @test isnan(DKS._student_log1p(3.0, NaN))
    @test isnan(DKS._student_log1p(NaN, 0.7))
    # Infinite degrees of freedom pair `z^2` with the factor -1/2.
    @test DKS._student_log1p(Inf, 1.7) == 1.7^2
    @test_throws DomainError DKS._student_log1p(-1.0, 0.0)
    @test_throws DomainError DKS._student_log1p(-1.0, 2.0)

    density = prepare(DKS.standard_student_t.logpdf; have = (:z, :nu), want = :logpdf)
    normal = prepare(DKS.standard_normal.logpdf)
    for z in (-3.0, 0.0, 0.4, 12.0)
        @test density(z, Inf) ≈ normal(z)
    end
    # The density stays finite where z^2 overflows.
    for nu in (1.0, 4.0)
        expected = setprecision(BigFloat, 512) do
            b = big(nu)
            Float64(-log(b) / 2 - SpecialFunctions.logbeta(nu / 2, 0.5) -
                    (b + 1) / 2 * log1p(big(1e200)^2 / b))
        end
        @test density(1e200, nu) ≈ expected
    end

    # A summed plate over bound observations: values and ordinary reverse
    # gradients in the degrees of freedom, location and scale.
    xs = [-4.0, -0.3, 0.0, 0.2, 1.1, 2.5, 30.0, 1e3]
    f = prepare(student_events; bound = (; xs))
    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
    reference = q -> _student_events(xs, q)
    for q in ([log(1.0), 0.1, log(1.0)], [log(4.0), -0.5, log(0.5)], [log(30.0), 0.3, log(2.0)])
        @test f(q) ≈ reference(q)
        h = 1e-6
        numeric = [(reference(q .+ h .* (1:3 .== k)) - reference(q .- h .* (1:3 .== k))) / 2h
                   for k in 1:3]
        @test DI.gradient(f, backend, q) ≈ numeric rtol = 1e-6
    end
end
end
