using ReactiveKernels, DifferentiationInterface, Test
import Enzyme
isdefined(@__MODULE__, :AuthoredPlateChains) ||
    include("fixtures/authored_plate_chains.jl")

_chain_gradient_allocated(ad::A, gradient, q, x, y) where {A} =
    @allocated ad_value_and_gradient!(ad, gradient, q, x, y)

@testset "Authored plate chains under plain reverse Enzyme" begin
    C = AuthoredPlateChains
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = [0.7]
    for n in (32, 4096)
        x = collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        kernel = prepare(C.chain)
        reference = prepare(C.flat)
        ad = prepare_ad(kernel, backend, q, x, y; active = :q)
        flat_ad = prepare_ad(reference, backend, q, x, y; active = :q)
        gradient, flat_gradient = zeros(1), zeros(1)
        v, _ = ad_value_and_gradient!(ad, gradient, q, x, y)
        flat_v, _ = ad_value_and_gradient!(flat_ad, flat_gradient, q, x, y)
        @test v == flat_v
        @test gradient == flat_gradient
        @test only(gradient) ≈ -sum((only(q) .* x .- y) .* x)
        _chain_gradient_allocated(ad, gradient, q, x, y)
        _chain_gradient_allocated(flat_ad, flat_gradient, q, x, y)
        @test _chain_gradient_allocated(ad, gradient, q, x, y) <=
              _chain_gradient_allocated(flat_ad, flat_gradient, q, x, y) + 64
        bound = prepare(C.chain; bound = (; x, y))
        bound_ad = prepare_ad(bound, backend, q; active = :q)
        @test ad_gradient(bound_ad, q) == gradient
    end
end
