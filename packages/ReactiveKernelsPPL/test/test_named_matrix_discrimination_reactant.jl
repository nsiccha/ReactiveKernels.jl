isdefined(@__MODULE__, :PPLNamedMatrixDiscriminationTests) ||
    include("test_named_matrix_discrimination.jl")
module PPLNamedMatrixDiscriminationReactantTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme, Test
using ..PPLNamedMatrixDiscriminationTests: BACKEND, build_case, reference,
    reference_gradient

@testset "submodel matrix discrimination compiles with ordinary reverse" begin
    bound, built, data = build_case()
    u = [0.1 * cos(i) for i in 1:built.layout.total]
    sampler = prepare_sampler(built, bound, u; backend=BACKEND)
    kernel = sampler.kernel
    ru = Reactant.to_rarray(u)
    primal = Reactant.@compile kernel(ru)
    reverse = compile_ad_value_and_gradient(sampler.ad, ru)
    original = deepcopy(data)
    for w in (u, -u)
        rw = Reactant.to_rarray(w)
        @test Float64(primal(rw)) ≈ reference(built.layout, data, w) rtol=1e-12
        value, grad = reverse(rw)
        @test Float64(value) ≈ reference(built.layout, data, w) rtol=1e-12
        @test Array(grad) ≈ reference_gradient(built.layout, data, w) rtol=1e-5 atol=1e-7
        @test Array(rw) == w
        @test data == original
    end
end

end
