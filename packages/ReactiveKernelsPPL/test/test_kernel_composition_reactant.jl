isdefined(@__MODULE__, :PPLKernelCompositionTests) || include("test_kernel_composition.jl")
module PPLKernelCompositionReactantTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme, Test
using ..PPLKernelCompositionTests: BACKEND, build_case, reference, gradient,
    build_subject_case, subject_reference, subject_gradient

function mlir_inventory(text)
    counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+", text)
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    return counts
end

function executable_inventory(compiled)
    hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
    instructions = collect(eachmatch(
        r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", hlo))
    assignments = collect(eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = ", hlo))
    @test !isempty(instructions)
    @test length(instructions) == length(assignments)
    counts = Dict{String,Int}()
    for m in instructions
        op = m.captures[1]
        counts[op] = get(counts, op, 0) + 1
    end
    return counts
end

const ReactantExt = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
executable_inventory(call::ReactantExt._ExternalizedADExecutable) =
    executable_inventory(call.compiled)

function inventory_does_not_grow(smaller, larger)
    all(zip(smaller, larger)) do (before, after)
        all(after) do (op, count)
            count <= get(before, op, 0)
        end
    end
end

@testset "PPL resolved scans retain default compiled bodies" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing
    for subjects in (false, true)
        inventories = []
        for (n, groups) in ((7, 3), (19, 7), (31, 11))
            bound, built, data = subjects ? build_subject_case(n, groups) : build_case(n)
            original = deepcopy(data)
            sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
            kernel, ad = sampler.kernel, sampler.ad
            both(u) = ad_value_and_gradient(ad, u)
            ru = Reactant.to_rarray([0.2])
            primal = Reactant.@compile kernel(ru)
            reverse = compile_ad_value_and_gradient(ad, ru)
            inventory = (
                mlir_inventory(repr(Reactant.@code_hlo kernel(ru))),
                mlir_inventory(repr(Reactant.@code_hlo both(ru))),
                executable_inventory(primal), executable_inventory(reverse))
            @test all(!isempty, inventory)
            @test get(inventory[1], "stablehlo.while", 0) >= 1
            @test get(inventory[3], "while", 0) >= 1
            push!(inventories, inventory)
            println("COMPOSITION_INVENTORY subjects=", subjects, " n=", n,
                " groups=", groups, " ", map(d -> sort!(collect(d); by=first), inventory))
            for a in (0.2, -0.4)
                u = [a]
                expected = subjects ? subject_reference(data, a) : reference(data, a)
                expected_grad = subjects ? subject_gradient(data, a) : gradient(data, a)
                native, grad = sampler_value_and_gradient!(sampler, [0.0], u)
                rw = Reactant.to_rarray(u)
                value, derivative = reverse(rw)
                @test native ≈ expected rtol=1e-10
                @test grad ≈ [expected_grad] rtol=1e-10
                @test Float64(primal(rw)) ≈ expected rtol=1e-9
                @test Float64(value) ≈ expected rtol=1e-9
                @test Array(derivative) ≈ [expected_grad] rtol=1e-8
                @test Array(rw) == u
            end
            @test data == original
        end
        if subjects
            # The default backend expands the three-subject plate into three
            # copies of the child scan. Larger plates retain nested loops.
            # Keep the small-shape structural gap visible alongside parity.
            @test_broken map(i -> get(inventories[1][i],
                i <= 2 ? "stablehlo.while" : "while", 0), 1:4) == [2, 4, 2, 4]
            for inventory in inventories[2:3]
                @test map(i -> get(inventory[i],
                    i <= 2 ? "stablehlo.while" : "while", 0), 1:4) == [2, 4, 2, 4]
            end
            @test inventories[2][1:2] == inventories[3][1:2]
            @test inventory_does_not_grow(inventories[2], inventories[3])
        else
            @test inventories[1] == inventories[2]
            @test inventories[2][1:2] == inventories[3][1:2]
            @test inventory_does_not_grow(inventories[2], inventories[3])
        end
    end
end
end
