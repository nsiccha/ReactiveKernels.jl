# Run in an RKPPL consumer environment with Distributions and Enzyme.
# This is the sampler-level counterpart to repro_enzyme_dense_solve_view.jl.
# Enzyme dogfood f94b9a92 supports a strided view on the right of a dense
# solve. Earlier builds can compute both densities and fail the reverse call.
using Distributions, Enzyme, LinearAlgebra, ReactiveKernelsPPL, Test
using DifferentiationInterface: AutoEnzyme

@testset "Native reverse of a dense solve on packed coordinates" begin
    model = @rkppl begin
        b[1:K] .~ Normal.(0, 1)
        eta = A \ b
        y .~ Normal.(eta, 1)
    end
    bindings = ((; K=2, A=[2.0 -0.4; 0.3 1.5], y=[0.6, -0.7]),
                (; K=2, A=[1.4 0.1; -0.6 1.2], y=[-0.2, 0.8]))
    original = deepcopy(bindings)
    lowered = lower_rkppl(model, first(bindings); conditioned = (:y,))
    built = build_kernel(bind_data(lowered, first(bindings)))
    @test coordinate_names(built.layout) == [Symbol("b.1"), Symbol("b.2")]

    u = [0.35, -0.2]
    saved = copy(u)
    for data in bindings
        sampler = prepare_sampler(built, bind_data(lowered, data), u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        residual = data.y - data.A \ u
        expected_value = -length(u) * log(2π) - (dot(u, u) + dot(residual, residual)) / 2
        expected_gradient = transpose(data.A) \ residual - u
        @test sampler(u) ≈ expected_value
        gradient = fill(NaN, length(u))
        value, _ = sampler_value_and_gradient!(sampler, gradient, u)
        @test value ≈ expected_value
        @test gradient ≈ expected_gradient
        @test isequal(u, saved)
        @test isequal(bindings, original)
    end
end
