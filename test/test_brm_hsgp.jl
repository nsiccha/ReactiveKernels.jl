using Test
using ReactiveKernels
using Distributions
include(joinpath(@__DIR__, "..", "examples", "brm_hsgp.jl"))
using .BRMHSGPExample

# Independent scalar distribution evaluation, with no graph/AD machinery.
function motorcycle_reference(q, c, data)
    lp = sum(logpdf(Normal(0, 4), q[i]) for i in (1, 2, 23, 24))
    curves = [zeros(length(data.y)), zeros(length(data.y))]
    for (offset, coffset, curve) in ((0, 0, curves[1]), (22, 20, curves[2]))
        rho, sd = exp(q[offset + 1]), exp(q[offset + 2])
        for j in 1:20
            omega = j * pi / 3
            # The SE spectral density is sd^2 * sqrt(2pi) * rho * exp(-rho^2*omega^2/2).
            logs = log(sd) + log(sqrt(2pi) * rho) / 2 - rho^2 * omega^2 / 4
            v = q[offset + 2 + j]
            prior_sd = exp(c[coffset + j] * logs)
            lp += logpdf(Normal(0, prior_sd), v)
            weight = v * exp((1 - c[coffset + j]) * logs)
            for i in eachindex(curve)
                curve[i] += sin(omega * (data.x[i] + 1.5)) / sqrt(1.5) * weight
            end
        end
    end
    lp + sum(logpdf(Normal(curves[1][i], exp(curves[2][i])), data.y[i])
             for i in eachindex(data.y))
end

@testset "BRM motorcycle dual HSGP" begin
    data = BRMHSGPExample.motorcycle_data(joinpath(@__DIR__, "..", "examples", "data", "mcycle.csv"))
    @test length(data.x) == length(data.y) == 133
    @test extrema(data.x) == (-1.0, 1.0)
    kernel = BRMHSGPExample.prepare_model(data)
    @test Tuple(p.name for p in inputs(kernel)) == (:q, :c)
    q = 0.04sin.(collect(1.0:44.0))
    q[[1, 23]] .= -2.0
    for c in (zeros(40), ones(40), collect(range(0, 1; length=40)),
              repeat([0.0, 0.25, 0.6, 1.0], 10))
        @test kernel(q, c) ≈ motorcycle_reference(q, c, data) atol=2e-10 rtol=2e-12
    end
    # Binding must fold the complete data-only design, but preserve spectrum,
    # coordinates, priors, and likelihood in the parameter-dependent program.
    basis_kernel = BRMHSGPExample.prepare_model(data; want=:basis)
    @test !occursin("sin", string(code_expr(basis_kernel)))
    @test size(basis_kernel(q, zeros(40))) == (133, 20)
    @test basis_kernel(q, zeros(40))[71, 13] ≈ sin(13pi / 3 * (data.x[71] + 1.5)) / sqrt(1.5)
    @test_throws DimensionMismatch BRMHSGPExample.prepare_model((; x=data.x[1:4], y=data.y))
end
