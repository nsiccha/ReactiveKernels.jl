module SourceWorldAgeReactantTests
using ReactiveKernels, Reactant, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

@kernel plus(x, y) = begin
    result = x + y
    return result
end

gradient(f, x) = only(Enzyme.gradient(Enzyme.Reverse, f, x))
function operation_inventory(text)
    names = [m.match for m in eachmatch(
        r"\b(?:stablehlo|chlo|enzyme|func|arith|scf|cf|tensor|math|linalg|memref)\.\w+", text)]
    sort(collect(Dict(name => count(==(name), names) for name in unique(names))))
end

function immediate_scan(xs)
    spec = Core.eval(@__MODULE__, :(@kernel dynamic(xs, gain) = begin
        shifted = plus(gain + 1, 0.25)
        updates = scan(xs, Ref(shifted); init=0.0) do carry, x, g
            next = 0.5 * carry + sin(g * x)
            (next, next)
        end
        result = sum(updates)
        return result
    end))
    kernel = prepare(spec; bound=(; xs))
    native_ad = prepare_ad(spec, AutoEnzyme(; mode=Enzyme.Reverse), xs, 0.3;
                           active=:gain, want=:result)
    g = 0.3
    rg = Reactant.to_rarray(g; track_numbers=true)
    reverse(x) = gradient(kernel, x)
    primal = Reactant.@compile kernel(rg)
    derivative = Reactant.@compile reverse(rg)
    saved = copy(xs)
    for gain in (0.3, -0.7)
        expected = 0.0
        expected_gradient = 0.0
        carry = 0.0
        dcarry = 0.0
        for x in xs
            carry = 0.5 * carry + sin((gain + 1.25) * x)
            dcarry = 0.5 * dcarry + x * cos((gain + 1.25) * x)
            expected += carry
            expected_gradient += dcarry
        end
        traced = Reactant.to_rarray(gain; track_numbers=true)
        @test kernel(gain) ≈ expected
        @test ad_gradient(native_ad, xs, gain) ≈ expected_gradient
        @test Float64(primal(traced)) ≈ expected
        @test Float64(derivative(traced)) ≈ expected_gradient
        @test xs == saved
    end
    modules = (repr(Reactant.@code_hlo kernel(rg)),
               repr(Reactant.@code_hlo reverse(rg)))
    executables = (repr(only(Reactant.XLA.get_hlo_modules(primal.exec))),
                   repr(only(Reactant.XLA.get_hlo_modules(derivative.exec))))
    @test all(text -> occursin("stablehlo.while", text), modules)
    @test all(text -> occursin("while(", text), executables)
    # Full operation inventories diagnose size growth; they are not a new
    # requirement that every pure arithmetic/reduction count be identical.
    println("world-age scan length=", length(xs), " MLIR=",
        operation_inventory.(modules))
    if haskey(ENV, "RK_WORLD_AGE_HLO_DIR")
        folder = ENV["RK_WORLD_AGE_HLO_DIR"]
        mkpath(folder)
        for (index, kind) in enumerate((:primal, :reverse))
            stem = joinpath(folder, "source-world-age-$(length(xs))-$kind")
            write(stem * ".mlir", modules[index])
            write(stem * ".hlo", executables[index])
        end
    end
end

@testset "dynamically authored scans retain default compiled primal and reverse" begin
    for n in (7, 19, 31)
        immediate_scan(sin.(1:n))
    end
end
end
