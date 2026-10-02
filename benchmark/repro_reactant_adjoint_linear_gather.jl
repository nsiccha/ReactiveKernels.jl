# Backend-only linear gather from an adjoint vector; no ReactiveKernels imports.
# Run with Julia 1.10 and Reactant 0.2.290 available.
using Reactant, LinearAlgebra, Test

function linear_row(z, sd, g)
    b = (z * sd)'
    return b[g]
end

z = [0.2 -0.4; 0.7 0.5]
sd = [0.6, -0.3]
g = [1, 2, 1, 2, 2, 1]
expected = linear_row(z, sd, g)
@test expected ≈ (z * sd)[g]
rz, rsd = Reactant.to_rarray(z), Reactant.to_rarray(sd)

# Ordinary matrix linear indexing compiles; the failing wrapper is Adjoint.
control = x -> x[g]
compiled_control = Reactant.@compile control(rz)
@test Array(compiled_control(rz)) ≈ z[g]

kernel = (x, s) -> linear_row(x, s, g)
err = try
    compiled = Reactant.@compile kernel(rz, rsd)
    @test Array(compiled(rz, rsd)) ≈ expected
    nothing
catch e
    e
end
if err !== nothing
    @test err isa BoundsError
    @test err.a isa LinearIndices{1}
    @test err.i isa Tuple{AbstractVector{CartesianIndex{2}}}
    println("PINNED_BACKEND_LIMITATION ", sprint(showerror, err))
else
    println("BACKEND_LIMITATION_LIFTED")
end
