"""
    rk_cholesky_lower(A::AbstractMatrix) -> LowerTriangular{Float64}

Compute the lower Cholesky factor of the symmetric matrix represented by
the lower triangle of `A`, using an ordinary pure-Julia factorization with
no LAPACK call. The arithmetic uses `Float64`. The result wraps a fresh
matrix; `A` is never mutated, and its upper triangle does not participate
in the factorization.

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
    L = Matrix{Float64}(A)
    @inbounds for j in 1:n
        for k in 1:j-1
            L[j, j] -= L[j, k] * L[j, k]
        end
        L[j, j] > 0 || throw(PosDefException(j))
        L[j, j] = sqrt(L[j, j])
        for i in j+1:n
            for k in 1:j-1
                L[i, j] -= L[i, k] * L[j, k]
            end
            L[i, j] /= L[j, j]
        end
    end
    return LowerTriangular(L)
end
