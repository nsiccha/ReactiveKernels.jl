# Owned symmetric-eigendecomposition primitives with generated AD rules.
#
# Ordinary reverse Enzyme through `eigen(::Symmetric)` fails: the call lowers
# to LAPACK `syevr!` (`dsyevr_64_`), for which Enzyme has no derivative rule
# (`EnzymeNoDerivativeError: No augmented forward pass found for dsyevr_64_`;
# Enzyme's bitcode-replacement table covers BLAS plus `potrf` only). That is
# why a symmetric-eigendecomposition-based density (the posteriordb
# `kronecker_gp` marginal likelihood) has no native reverse gradient.
#
# These rules are the replacement path: one pure-math `@kernel` graph per
# output authors the primal plus its forward (JVP) and reverse (VJP)
# branches, and `derivative_rule` generates the owned callable plus every
# AD-protocol adapter from that graph's cuts. No hand-written rule, no rule
# on a foreign function (see `docs/src/constraints.md`). Each rule's primal
# calls LAPACK once; the `ccall` itself is never differentiated.
#
# Mathematics (symmetric-eigenvalue perturbation theory, e.g. Giles 2008,
# "An extended collection of matrix derivative results for forward and
# reverse mode AD"). For A = Q*Diagonal(R)*Q' with Q orthonormal:
#
#   values JVP:  R_dot = diag(Q' * A_dot * Q)
#   values VJP:  A_bar = Q * Diagonal(R_bar) * Q'
#   vectors JVP: Q_dot = Q * (F .* (Q' * A_dot * Q))
#   vectors VJP: A_bar = Q * Symmetric(F .* (Q' * Q_bar)) * Q'
#   with F[i, j] = 1 / (R[j] - R[i]) for i != j and F[i, i] = 0.
#
# Domain. Both rules read only the upper triangle (`Symmetric(A)`), so A
# must be symmetric. Eigenvalues must be distinct: a repeated eigenvalue
# makes F singular (the same documented restriction JAX's `eigh` rule
# carries). Eigenvector columns are defined up to sign; a vectors covector
# is only meaningful sign-aligned with the primal's columns.
using LinearAlgebra: Symmetric, eigen, diag, I

@kernel rk_symmetric_eigvals_rule(A::Matrix{Float64}, A_dot::Matrix{Float64},
        R_bar::Vector{Float64}) = begin
    S = Symmetric(A)
    F = eigen(S)
    R::Vector{Float64} = F.values
    Q::Matrix{Float64} = F.vectors
    T::Matrix{Float64} = A_dot * Q
    M::Matrix{Float64} = transpose(Q) * T
    R_dot::Vector{Float64} = diag(M)
    # Q * Diagonal(R_bar) without a diagonal constructor: scale the columns.
    QD::Matrix{Float64} = Q .* transpose(R_bar)
    A_bar::Matrix{Float64} = QD * transpose(Q)
    return R, R_dot, A_bar
end

@kernel rk_symmetric_eigvecs_rule(A::Matrix{Float64}, A_dot::Matrix{Float64},
        Q_bar::Matrix{Float64}) = begin
    S = Symmetric(A)
    F = eigen(S)
    R::Vector{Float64} = F.values
    Q::Matrix{Float64} = F.vectors
    n::Int = size(A, 1)
    # F[i, j] = 1 / (R[j] - R[i]) off the diagonal, 0 on it: the added
    # identity keeps the diagonal reciprocal finite before it is masked out.
    G::Matrix{Float64} = transpose(R) .- R
    Gn::Matrix{Float64} = G + Matrix{Float64}(I, n, n)
    Finv::Matrix{Float64} = 1 ./ Gn
    Fm::Matrix{Float64} = Finv .* (1 .- Matrix{Float64}(I, n, n))
    T::Matrix{Float64} = A_dot * Q
    M::Matrix{Float64} = transpose(Q) * T
    Q_dot::Matrix{Float64} = Q * (Fm .* M)
    C::Matrix{Float64} = transpose(Q) * Q_bar
    FC::Matrix{Float64} = Fm .* C
    H::Matrix{Float64} = (FC + transpose(FC)) / 2
    A_bar::Matrix{Float64} = Q * H * transpose(Q)
    return Q, Q_dot, A_bar
end

"""
    rk_symmetric_eigvals(A::Matrix{Float64}) -> Vector{Float64}

Eigenvalues of a symmetric matrix as an RK-owned primitive:
`rk_symmetric_eigvals(A) == eigen(Symmetric(A)).values`, with forward- and
reverse-mode rules generated from the one pure-math graph
`rk_symmetric_eigvals_rule`. Differentiating through it with Enzyme —
directly or via `DifferentiationInterface` with `AutoEnzyme` — uses those
generated cuts, so the LAPACK `syevr!` `ccall` inside is never
differentiated. No Reactant custom rule is emitted (see
`docs/src/manual-derivative-rules.md`).

`A` must be symmetric (only the upper triangle is read) with distinct
eigenvalues.
"""
const rk_symmetric_eigvals = derivative_rule(rk_symmetric_eigvals_rule;
    primal = :R, directions = (A = :A_dot,), tangent = :R_dot,
    covector = :R_bar, cotangents = (A = :A_bar,), name = :rk_symmetric_eigvals)

"""
    rk_symmetric_eigvecs(A::Matrix{Float64}) -> Matrix{Float64}

Orthonormal eigenvectors (columns) of a symmetric matrix as an RK-owned
primitive: `rk_symmetric_eigvecs(A) == eigen(Symmetric(A)).vectors`, with
forward- and reverse-mode rules generated from the one pure-math graph
`rk_symmetric_eigvecs_rule`. Differentiating through it with Enzyme —
directly or via `DifferentiationInterface` with `AutoEnzyme` — uses those
generated cuts, so the LAPACK `syevr!` `ccall` inside is never
differentiated. No Reactant custom rule is emitted (see
`docs/src/manual-derivative-rules.md`).

`A` must be symmetric (only the upper triangle is read) with distinct
eigenvalues. Columns are defined up to sign; align a covector's signs with
the primal's columns.
"""
const rk_symmetric_eigvecs = derivative_rule(rk_symmetric_eigvecs_rule;
    primal = :Q, directions = (A = :A_dot,), tangent = :Q_dot,
    covector = :Q_bar, cotangents = (A = :A_bar,), name = :rk_symmetric_eigvecs)
