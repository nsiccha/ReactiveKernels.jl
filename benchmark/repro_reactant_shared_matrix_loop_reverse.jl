# Reactant 0.2.290: reverse compilation of a shared matrix and a retained
# diagonal reduction under a live guard produces an invalid MLIR use before
# definition ("operand #0 does not dominate this use"). Native and compiled
# primal pass. No ReactiveKernels or PPL lowering is involved.
# Run in an environment with Enzyme and Reactant; no compiler flags or AD rules.

using Enzyme
using Reactant
using LinearAlgebra

function lkj_diagonal(eta, L, K)
    total = zero(eta)
    Reactant.@trace track_numbers=false for i in 2:K
        total += (K - i + 2 * eta - 2) * log(Reactant.@allowscalar L[i, i])
    end
    return total
end

function membership_loss(u)
    eta = exp(sum(view(u, 1:1)))
    a = sum(view(u, 2:2))
    sd = exp.(view(u, 3:4))
    p = tanh(sum(view(u, 5:5)))
    L = similar(u, 2, 2)
    fill!(L, 0.0)
    Reactant.@allowscalar begin
        L[1, 1] = 1.0
        L[2, 1] = p
        L[2, 2] = sqrt(1 - p * p)
    end
    z = reshape(view(u, 6:11), 3, 2)
    B = z * (sd .* L)'
    g, h = [1, 2, 3, 1, 2, 3, 1], [2, 3, 1, 2, 3, 1, 2]
    w1, w2 = [0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3], [0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7]
    R = (w1 .* B[g, :] .+ w2 .* B[h, :]) ./ (w1 .+ w2)
    x = collect(0.1:0.1:0.7)
    mu = R[:, 1] .+ (a .* x) .* R[:, 2]
    nu = .-R[:, 1] .+ (a .* x .* x) .* R[:, 2]
    scalar_term = 0.0
    diagonal = 0.0
    Reactant.@trace if isfinite(eta) && eta > 0
        # Ordinary algebra suffices; no distribution function is involved.
        scalar_term = eta * eta
        diagonal = lkj_diagonal(eta, L, 2)
    else
        scalar_term = -Inf
    end
    prior = scalar_term + diagonal
    return prior - sum(abs2, mu) - sum(abs2, nu) - sum(abs2, z) +
        sum(log.(sd) .- sd) - eta - a * a / 2 + sum(u[1:4])
end

gradient(u) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(membership_loss), u))
u = [0.2 * sin(i) for i in 1:11]
ru = Reactant.to_rarray(u)
primal = Reactant.@compile membership_loss(ru)
@assert Float64(primal(ru)) ≈ membership_loss(u)
h = 1e-5
fd = map(eachindex(u)) do i
    up, um = copy(u), copy(u)
    up[i] += h
    um[i] -= h
    (membership_loss(up) - membership_loss(um)) / (2h)
end
println("native and compiled primal pass")
# This default reverse compile fails with a dominance error on 0.2.290.
compiled = Reactant.@compile gradient(ru)
@assert Array(compiled(ru)) ≈ fd rtol = 1e-6 atol = 1e-8
println("compiled reverse passes: backend limitation lifted")
