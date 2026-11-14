"""
    rk_cholesky_lower(A::AbstractMatrix) -> LowerTriangular{Float64}

Compute the lower Cholesky factor of the symmetric matrix represented by
the lower triangle of `A`, using an ordinary pure-Julia factorization with
no LAPACK call. The arithmetic uses `Float64`. The result wraps a fresh
matrix whose strict upper triangle is zero; `A` is never mutated, and its
upper triangle does not participate in the factorization.

`A` must be square (`ArgumentError` otherwise). A non-positive or NaN pivot
throws `LinearAlgebra.PosDefException`, whose `info` identifies the pivot.
An empty square matrix has an empty factor. `jax_source` translates the
factorization with the same lower-triangle convention and runtime validity
checks. Both native and JAX execution use ordinary backend differentiation;
this callable has no custom AD rule.
"""
function rk_cholesky_lower(A::AbstractMatrix)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError(
        "rk_cholesky_lower needs a square matrix, got $(size(A))"))
    # Each entry of `L` is stored exactly once, after its sum is accumulated
    # in a scalar. Updating `L` in place (`L[j, j] -= …`) between the loop's
    # reads of `L` makes native Enzyme reverse tape those reads; RKPPL's
    # `_covariance_cholesky!` follows the same rule.
    L = zeros(Float64, n, n)
    @inbounds for j in 1:n
        s = Float64(A[j, j])
        for k in 1:j-1
            s -= L[j, k] * L[j, k]
        end
        s > 0 || throw(PosDefException(j))
        pivot = sqrt(s)
        L[j, j] = pivot
        for i in j+1:n
            t = Float64(A[i, j])
            for k in 1:j-1
                t -= L[i, k] * L[j, k]
            end
            L[i, j] = t / pivot
        end
    end
    return LowerTriangular(L)
end
