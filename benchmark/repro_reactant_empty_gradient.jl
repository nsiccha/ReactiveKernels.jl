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

# The same export boundary occurs with an empty multivariate slice batch
# beside a nonempty live factor; both gradients come from ordinary reverse.

function empty_batch(X, F)
    if size(X, 1) == 0
        0.0
    else
        sum(abs2, X) + sum(abs2, F)
    end
end

empty_batch_gradient(X, F) = Enzyme.gradient(Enzyme.Reverse, empty_batch, X, F)

X, F = zeros(0, 2), ones(2, 2)
rx, rf = Reactant.to_rarray(X), Reactant.to_rarray(F)
compiled = Reactant.@compile empty_batch(rx, rf)
@test Float64(compiled(rx, rf)) == 0.0
native = empty_batch_gradient(X, F)
@test native[1] == X
@test native[2] == zeros(2, 2)
println(Reactant.@code_hlo optimize = :only_enzyme empty_batch_gradient(rx, rf))
@test_throws r"'tensor.empty' op unsupported op for export to XLA" Reactant.@compile empty_batch_gradient(rx, rf)
