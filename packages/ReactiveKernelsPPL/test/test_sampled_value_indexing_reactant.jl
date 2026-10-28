isdefined(@__MODULE__, :SampledValueIndexingTests) || include("test_sampled_value_indexing.jl")

module SampledValueIndexingReactantTests
using ..SampledValueIndexingTests: simplex_case, reference, differences, BACKEND
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Test

function mlir_inventory(text)
    ops = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith)\.\w+", text)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    ops
end

function executable_inventory(compiled)
    hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    ops = Dict{String,Int}()
    for m in eachmatch(r"(?m)^\s*[^\n=]+=[^\n]*?\s([a-z][a-z0-9-]*)\(", hlo)
        ops[m.captures[1]] = get(ops, m.captures[1], 0) + 1
    end
    ops
end

@testset "sampled-value indexing: default compiled values, Reverse and structure" begin
    inventories = []
    for n in (3, 9, 0)
        bound, built, data = simplex_case(; n)
        u = [0.2sin(i) for i in 1:built.layout.total]
        sampler = prepare_sampler(built, bound, u; backend=BACKEND)
        native, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        ru = Reactant.to_rarray(u)
        kernel = sampler.kernel
        compiled = Reactant.@compile kernel(ru)
        cad = compile_ad_value_and_gradient(sampler.ad, ru)
        primal_mlir = mlir_inventory(string(Reactant.@code_hlo kernel(ru)))
        # Inspect optimized primal MLIR and the actual default primal and AD
        # executables, including callees and fusion bodies in their HLO modules.
        primal_hlo = executable_inventory(compiled)
        reverse_hlo = executable_inventory(cad)
        @test !isempty(primal_mlir)
        @test !isempty(primal_hlo) && !isempty(reverse_hlo)
        println("sampled-value structure n=", n, " primal MLIR=", primal_mlir,
            " executable=", (primal_hlo, reverse_hlo))
        n == 0 || push!(inventories, (primal_mlir, primal_hlo, reverse_hlo))
        for shift in (0.0, -0.3)
            input = u .+ shift
            saved = copy(input)
            rinput = Reactant.to_rarray(input)
            oracle = v -> reference(built.layout, data, v).posterior
            value, gradient = cad(rinput)
            @test Float64(compiled(rinput)) ≈ oracle(input) rtol=1e-12
            @test Float64(value) ≈ oracle(input) rtol=1e-12
            @test Array(gradient) ≈ differences(oracle, input) rtol=1e-5 atol=1e-7
            @test Array(rinput) == saved
        end
        value, gradient = cad(ru)
        @test Float64(value) ≈ native
        @test Array(gradient) ≈ grad
    end
    # These fixed four-coordinate models differ only in observation count.
    # Complete inventories guard against replicated data-derived bodies.
    @test inventories[1] == inventories[2]
end
end
