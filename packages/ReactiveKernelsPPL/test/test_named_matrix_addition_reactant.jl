isdefined(@__MODULE__, :PPLNamedMatrixAdditionTests) || include("test_named_matrix_addition.jl")
module PPLNamedMatrixAdditionReactantTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme, Test
using ..PPLNamedMatrixAdditionTests: BACKEND, SPELLINGS, build_case, reference, gradient
using ..PPLNamedMatrixProductReactantTests: mlir_inventory, executable_inventory

@testset "matrix and graph components compile with bounded structure" begin
    for (label, beta, body) in SPELLINGS
        @testset "$label" begin
            inventories = Dict{Int,Any}()
            for n in (0, 1, 8, 40, 200)
                bound, built, data = build_case(body, n)
                names = coordinate_names(built.layout)
                u = [0.1 * cos(i) for i in eachindex(names)]
                sampler = prepare_sampler(built, bound, u; backend=BACKEND)
                kernel, ad = sampler.kernel, sampler.ad
                both(w) = ad_value_and_gradient(ad, w)
                ru = Reactant.to_rarray(u)
                primal = Reactant.@compile kernel(ru)
                reverse = compile_ad_value_and_gradient(ad, ru)
                original = deepcopy(data)
                for w in (u, -u)
                    rw = Reactant.to_rarray(w)
                    @test Float64(primal(rw)) ≈ reference(names, beta, data, w) rtol=1e-12
                    value, grad = reverse(rw)
                    @test Float64(value) ≈ reference(names, beta, data, w) rtol=1e-12
                    @test Array(grad) ≈ gradient(names, beta, data, w) rtol=1e-10 atol=1e-12
                    @test Array(rw) == w
                    @test data == original
                end
                inventories[n] = (
                    mlir_inventory(repr(Reactant.@code_hlo kernel(ru))),
                    mlir_inventory(repr(Reactant.@code_hlo both(ru))),
                    executable_inventory(primal), executable_inventory(reverse))
            end
            @test all(n -> inventories[n][1:2] == inventories[8][1:2], (8, 40, 200))
            @test inventories[40] == inventories[200]
        end
    end
end

end
