# Centered multivariate-normal rows: the sum over rows `g` of
# `logpdf(MvNormal(0, F * F'), x_g)` for a lower-triangular factor `F`,
# by forward substitution. `input(g, i)` is entry `i` of row `g`,
# `factor(i, j)` the entry `F[i, j]` (`j ≤ i`), `logdet` the
# `sum(log.(diag(F)))`. The factor's construction is structural (`k`
# fixed), while the rows are traversed by an ordinary runtime loop.
@inline function _lower_solve_rows_logpdf(input, factor, logdet, k, groups)
    value = -groups*(0.5k*log(2pi)+logdet)
    z = zeros(Float64,k)
    for g in 1:groups
        for i in 1:k
            residual = input(g,i)
            for j in 1:(i-1)
                residual -= factor(i,j)*z[j]
            end
            z[i] = residual/factor(i,i)
            value -= 0.5z[i]^2
        end
    end
    return value
end

# Native centered correlated draws. The factor entries are row-major lower
# triangular, scaled per margin by `scales`.
@inline function _centered_correlated_logpdf(draws, scales, lower)
    ReactiveKernels._dynamic_tensorized_marker((draws,scales,lower)) === nothing ||
        throw(ArgumentError("centered correlated draws support native execution only"))
    k = length(scales)
    k > 0 && length(lower) == k*(k+1)÷2 && length(draws)%k == 0 ||
        throw(DimensionMismatch("centered correlated draws and factor dimensions disagree"))
    groups = length(draws)÷k
    logdet = 0.0
    for i in 1:k
        diagonal = lower[i*(i+1)÷2]
        scales[i] > 0 && diagonal > 0 ||
            throw(ArgumentError("centered correlated factor has a nonpositive scale or diagonal"))
        logdet += log(scales[i])+log(diagonal)
    end
    return _lower_solve_rows_logpdf((g,i) -> draws[(g-1)*k+i]/scales[i],
        (i,j) -> lower[i*(i-1)÷2+j], logdet, k, groups)
end

# `eachrow(B[a, b]) .~ MvNormalCholesky(mu, F)`: every row of `B` one
# draw of `MvNormal(mu, F * F')`, `F` the lower-triangular Cholesky factor
# of the covariance (Stan's `multi_normal_cholesky`). Native execution.
@inline function _mvnormal_cholesky_rows_logpdf(B::AbstractMatrix, mu, F)
    ReactiveKernels._dynamic_tensorized_marker((B,mu,F)) === nothing ||
        throw(ArgumentError("row-wise MvNormalCholesky arrays support native execution only"))
    groups, k = size(B)
    length(mu) == k && size(F) == (k,k) ||
        throw(DimensionMismatch("MvNormalCholesky rows have length $k, but the mean has " *
            "length $(length(mu)) and the factor size $(size(F))"))
    logdet = 0.0
    for i in 1:k
        F[i,i] > 0 || throw(ArgumentError("MvNormalCholesky factor has a nonpositive " *
            "diagonal entry F[$i, $i]"))
        for j in (i+1):k
            iszero(F[i,j]) || throw(ArgumentError("MvNormalCholesky factor is not lower " *
                "triangular: F[$i, $j] = $(F[i,j])"))
        end
        logdet += log(F[i,i])
    end
    return _lower_solve_rows_logpdf((g,i) -> B[g,i]-mu[i], (i,j) -> F[i,j], logdet,
        k, groups)
end
