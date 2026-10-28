# Backend-only reproducer: Reactant 0.2.290 fails to broadcast over this
# SubArray with MethodError: reindex(::Tuple{UnitRange}, ::CartesianIndex).
# No ReactiveKernels code or derivative adapters are loaded.
using Reactant
Reactant.set_default_backend("cpu")

function view_break(u)
    return 1.0 ./ (1.0 .+ exp.(-(Float64.(view(u, 2:2)) .+
        log.(2 .- (1:1)))))
end
function slice_break(u)
    return 1.0 ./ (1.0 .+ exp.(-(u[2:2] .+
        log.(2 .- (1:1)))))
end

u = [0.2, 0.4]
ru = Reactant.to_rarray(u)
try
    compiled = Reactant.@compile view_break(ru)
    @assert Array(compiled(ru)) ≈ view_break(u)
    println("VIEW_BROADCAST_SUPPORTED")
catch err
    println("VIEW_BROADCAST_FAILURE: ", sprint(showerror, err))
end
compiled = Reactant.@compile slice_break(ru)
@assert Array(compiled(ru)) ≈ slice_break(u)
println("SLICE_BROADCAST_PARITY")
