# Backend-only gather from an adjoint matrix; no ReactiveKernels imports.
using Reactant, LinearAlgebra, Test

point = collect(range(-0.4, 0.6; length = 6))
g = [3, 2, 1, 3, 2, 1]
kernel = u -> reshape(u, 2, 3)'[g, 1]
expected = kernel(point)
ru = Reactant.to_rarray(point)

# Gathering from a plain reshaped matrix along its second axis succeeds.
control = u -> reshape(u, 2, 3)[1, g]
compiled_control = Reactant.@compile control(ru)
@test Array(compiled_control(ru)) ≈ expected

err = try
    compiled = Reactant.@compile kernel(ru)
    @test Array(compiled(ru)) ≈ expected
    nothing
catch e
    e
end
if err !== nothing
    @test err isa BoundsError
    @test err.a isa LinearIndices{2}
    @test err.i isa Tuple{AbstractVector{CartesianIndex{2}}}
    println("PINNED_BACKEND_LIMITATION ", sprint(showerror, err))
else
    println("BACKEND_LIMITATION_LIFTED")
end
