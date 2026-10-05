isdefined(@__MODULE__, :PPLNamedMatrixProductTests) || include("test_named_matrix_products.jl")
module PPLNamedMatrixProductReactantTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme, Test
using ..PPLNamedMatrixProductTests: BACKEND, SPELLINGS, build_case, reference,
    gradient

function mlir_inventory(text)
    counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+", text)
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    return counts
end

function executable_inventory(compiled)
    hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    counts = Dict{String,Int}()
    for m in eachmatch(
            r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", hlo)
        counts[m.captures[1]] = get(counts, m.captures[1], 0) + 1
    end
    @test !isempty(counts)
    return counts
end

const ReactantExt = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
executable_inventory(call::ReactantExt._ExternalizedADExecutable) =
    executable_inventory(call.compiled)

# The named and submodel-returned spellings compile exactly like the inline
# one: matching values and ordinary gradients, and the inline spelling's
# MLIR and default-executable inventories at every size. Optimized MLIR is
# fixed across sizes. At 8 rows XLA reduces in a single stage; from 40 rows
# it adds one `reduce-window` stage and then plateaus (measured identical at
# 40, 200 and 1000 rows), so the complete executables are compared there.
@testset "named matrix products compile with fixed structure" begin
    u = [0.3, -0.7]
    sizes = (0, 1, 8, 40, 200)
    inline = Dict{Int,Any}()
    for (label, _, body) in SPELLINGS
        @testset "$label" begin
            inventories = Dict{Int,Any}()
            for n in sizes
                bound, built, data = build_case(body, n)
                sampler = prepare_sampler(built, bound, u; backend=BACKEND)
                kernel, ad = sampler.kernel, sampler.ad
                both(w) = ad_value_and_gradient(ad, w)
                ru = Reactant.to_rarray(u)
                primal = Reactant.@compile kernel(ru)
                reverse = compile_ad_value_and_gradient(ad, ru)
                original = deepcopy(data)
                for w in (u, [-0.2, 0.4])
                    rw = Reactant.to_rarray(w)
                    @test Float64(primal(rw)) ≈ reference(data, w)
                    value, grad = reverse(rw)
                    @test Float64(value) ≈ reference(data, w)
                    @test Array(grad) ≈ gradient(data, w)
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
            label === :inline && merge!(inline, inventories)
            @test all(n -> inventories[n] == inline[n], sizes)
        end
    end
end

end
