# Owned `rk_beta_inc` (regularized incomplete beta over the Student-t cdf
# slice) vs the `SpecialFunctions.beta_inc` oracle, plus transparent-math
# Enzyme gradients (no rule: the continued-fraction body differentiates
# ordinarily) and the student-t cdf rewire pin.
using DifferentiationInterface: AutoEnzyme, gradient
using Distributions: TDist, cdf
import Enzyme
using ReactiveKernels: prepare
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    LOCATION_SCALE_SOURCE, rk_beta_inc, standard_student_t
import SpecialFunctions
using Test

@testset "rk_beta_inc matches beta_inc on the Student-t slice" begin
    nus = (0.5, 1.0, 2.0, 3.0, 4.0, 10.0, 30.0, 100.0)
    xs = vcat(0.0, 1.0, 10.0 .^ (-300:10:-10), 1.0 .- 10.0 .^ (-15:1:-1),
        collect(range(1e-4, 1 - 1e-4, length = 25)))
    for nu in nus, x in xs
        a = nu / 2
        got = rk_beta_inc(a, 0.5, x)
        @test isfinite(got)
        @test got ≈ first(SpecialFunctions.beta_inc(a, 0.5, x)) atol = 1e-12
    end
    # Endpoints are exact (the guards), never a 0·Inf limit artifact.
    for nu in nus
        @test rk_beta_inc(nu / 2, 0.5, 0.0) === 0.0
        @test rk_beta_inc(nu / 2, 0.5, 1.0) === 1.0
    end
end

@testset "rk_beta_inc differentiates as transparent math (no rule)" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    # x-partial is the beta density in closed form.
    for (a, b, x) in ((1.5, 0.5, 0.3), (1.5, 0.5, 0.8), (5.0, 0.5, 0.6))
        pdf = x^(a - 1) * (1 - x)^(b - 1) / exp(SpecialFunctions.logbeta(a, b))
        @test gradient(v -> rk_beta_inc(a, b, v), backend, x) ≈ pdf rtol = 1e-8
    end
    # a-partial against central finite differences (no closed form).
    for (a, b, x) in ((1.5, 0.5, 0.3), (2.0, 0.5, 0.7))
        h = 1e-6
        fd = (rk_beta_inc(a + h, b, x) - rk_beta_inc(a - h, b, x)) / 2h
        @test gradient(v -> rk_beta_inc(v, b, x), backend, a) ≈ fd rtol = 1e-6
    end
end

@testset "standard_student_t cdf rides rk_beta_inc" begin
    @test occursin("rk_beta_inc(nu / 2, 0.5", LOCATION_SCALE_SOURCE)
    @test !occursin("first(beta_inc(", LOCATION_SCALE_SOURCE)
    student_cdf = prepare(standard_student_t.cdf; have = (:z, :nu), want = :cdf)
    for nu in (1.0, 3.0, 4.0, 30.0)
        for z in (-37.0, -4.0, -0.4, 0.0, 0.7, 5.0, 41.0)
            @test student_cdf(z, nu) ≈ cdf(TDist(nu), z) atol = 1e-12
        end
    end
end
