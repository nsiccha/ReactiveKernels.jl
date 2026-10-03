# Backend-only boundary: XLA adds reduction stages as vector lengths grow,
# even when the optimized MLIR has one fixed reduction body.
using Reactant, Enzyme, Test
Reactant.set_default_backend("cpu")

reduction_loss(u) = sum(u .* u)
reduction_gradient(u) = only(Enzyme.gradient(Enzyme.Reverse,
    Enzyme.Const(reduction_loss), u))

function reduction_inventory(hlo; executable=false)
    counts = Dict{String,Int}()
    pattern = executable ?
        r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(" :
        r"((?:stablehlo|enzyme)\.[a-z_]+)"
    for m in eachmatch(pattern, repr(hlo))
        name = m.captures[1]
        counts[name] = get(counts, name, 0) + 1
    end
    return counts
end

@testset "Reactant vector reduction executable growth boundary" begin
    mlir, executable = [], []
    for n in (16, 64, 256, 2048)
        u = [0.2sin(i) for i in 1:n]
        ru = Reactant.to_rarray(u)
        primal = Reactant.@compile reduction_loss(ru)
        reverse = Reactant.@compile reduction_gradient(ru)
        @test Float64(primal(ru)) ≈ reduction_loss(u) rtol=1e-12
        @test Array(reverse(ru)) ≈ 2u rtol=1e-12
        @test Array(ru) == u
        push!(mlir, (
            reduction_inventory(Reactant.@code_hlo reduction_loss(ru)),
            reduction_inventory(Reactant.@code_hlo reduction_gradient(ru))))
        push!(executable, map((primal, reverse)) do compiled
            reduction_inventory(only(Reactant.XLA.get_hlo_modules(compiled.exec));
                executable=true)
        end)
        println("vector reduction inventory, n=", n,
            ", MLIR=", mlir[end], ", executable=", executable[end])
    end
    @test allequal(mlir)
    # Retaining scalar while/if bodies does not establish this separate
    # larger-shape requirement. Keep complete default executable counts.
    @test_broken allequal(executable)
end
