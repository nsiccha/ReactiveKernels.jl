# Backend-only zero-coordinate reverse compilation; no ReactiveKernels imports.
# Reactant 0.2.290 / Enzyme 0.13.209: primal and native reverse succeed,
# compiled reverse leaves tensor.empty, which XLA export rejects.
using Reactant, Enzyme, Test

loss(x) = 0.0
gradient(x) = only(Enzyme.gradient(Enzyme.Reverse, loss, x))
x = Float64[]
rx = Reactant.to_rarray(x)
@test loss(x) == 0.0
@test isempty(gradient(x))
@test Float64((Reactant.@compile loss(rx))(rx)) == 0.0
err = try
    compiled = Reactant.@compile gradient(rx)
    @test isempty(Array(compiled(rx)))
    nothing
catch e
    e
end
if err === nothing
    println("BACKEND_LIMITATION_LIFTED")
else
    @test occursin("'tensor.empty' op unsupported op for export to XLA", sprint(showerror, err))
    println("PINNED_BACKEND_LIMITATION ", sprint(showerror, err))
end
