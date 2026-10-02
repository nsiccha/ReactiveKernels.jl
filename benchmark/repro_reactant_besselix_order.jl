# Backend-only periodic HSGP limitation; no ReactiveKernels imports.
# Run with Julia 1.10 and Reactant 0.2.289 plus SpecialFunctions available.
using Reactant, SpecialFunctions, Test

function periodic_weights(rho, harmonics)
    a = 1.0 / (rho * rho)
    return exp.(0.5 .* (log(2.0) .+
        log.(SpecialFunctions.besselix.(harmonics, a))))
end

rho = 0.8
harmonics = [1, 2, 3, 1, 2, 3]
expected = periodic_weights(rho, harmonics)
@test all(isfinite, expected)
rrho = Reactant.ConcreteRNumber(rho)
kernel = x -> periodic_weights(x, harmonics)
err = try
    compiled = Reactant.@compile kernel(rrho)
    @test Array(compiled(rrho)) ≈ expected
    nothing
catch e
    e
end
if err !== nothing
    @test err isa MethodError
    @test err.f === SpecialFunctions.besselix
    @test length(err.args) == 2
    @test err.args[2] isa Reactant.TracedRNumber
    println("PINNED_BACKEND_LIMITATION ", sprint(showerror, err))
else
    println("BACKEND_LIMITATION_LIFTED")
end
