isdefined(@__MODULE__, :PPLLiveMatrixAxesTests) || include("test_live_matrix_axes.jl")
module PPLLiveMatrixAxesReactantTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Test
using ..PPLLiveMatrixAxesTests: BACKEND, KINDS, build_case, reference, gradient

function inventory(text, pattern)
    counts = Dict{String,Int}()
    for m in eachmatch(pattern, text)
        op = isempty(m.captures) ? m.match : m.captures[1]
        counts[op] = get(counts, op, 0) + 1
    end
    @test !isempty(counts)
    return counts
end
mlir_inventory(text) = inventory(text,
    r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+")
function executable_inventory(compiled)
    hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    return inventory(hlo,
        r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(")
end
const ReactantExt = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
executable_inventory(call::ReactantExt._ExternalizedADExecutable) =
    executable_inventory(call.compiled)

function check_case(kind, n; fixed = false)
    bound, built, data = build_case(kind, n; fixed)
    names = coordinate_names(built.layout)
    u = [0.2sin(i) for i in eachindex(names)]
    sampler = prepare_sampler(built, bound, u; backend = BACKEND)
    kernel, ad = sampler.kernel, sampler.ad
    both(w) = ad_value_and_gradient(ad, w)
    ru = Reactant.to_rarray(u)
    primal = Reactant.@compile kernel(ru)
    reverse = compile_ad_value_and_gradient(ad, ru)
    original = deepcopy(data)
    for w in (u, u .+ 0.13)
        rw = Reactant.to_rarray(w)
        oracle = v -> reference(kind, data, names, v)
        @test Float64(primal(rw)) ≈ oracle(w) rtol = 1e-12
        value, grad = reverse(rw)
        @test Float64(value) ≈ oracle(w) rtol = 1e-12
        @test Array(grad) ≈ gradient(oracle, w) rtol = 1e-5 atol = 1e-7
        @test Array(rw) == w
        @test data == original
    end
    return (mlir_inventory(repr(Reactant.@code_hlo kernel(ru))),
        mlir_inventory(repr(Reactant.@code_hlo both(ru))),
        executable_inventory(primal), executable_inventory(reverse))
end

@testset "live matrix axes: default compiled values, reverse and structure" begin
    for kind in KINDS
        sizes = kind in (:alias, :named) ? (0, 1, 8, 40, 200) : (0, 1, 7)
        receipts = Dict{Int,Any}()
        for n in sizes
            @testset "$kind / $n" begin
                receipts[n] = check_case(kind, n)
                @test receipts[n] == check_case(kind, n; fixed = true)
            end
        end
        if kind in (:alias, :named)
            @test receipts[8][1:2] == receipts[40][1:2] == receipts[200][1:2]
            # XLA may add a reduction stage between small and larger arrays;
            # the full default-executable inventories at 40 and 200 agree.
            @test receipts[40] == receipts[200]
        end
    end
end
end
