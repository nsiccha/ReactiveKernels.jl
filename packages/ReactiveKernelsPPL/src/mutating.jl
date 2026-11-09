# Destination forms of RKPPL's own array-valued helpers. A non-allocating
# kernel (`ReactiveKernels.prepare_nonallocating`, which the sampler prepares
# by default: `prepare_sampler(...; nonallocating = :auto)`) keeps a buffer for
# a step only when a registered `MutatingFunctions.apply!!` method fills it.
# Each method below receives the step's previous result, writes the new one
# into its storage with the loops the ordinary helper runs, so values are
# bitwise unchanged, and falls back to the ordinary helper when the shape
# changed.

@inline function MutatingFunctions.apply!!(y::Vector{Float64},
        ::typeof(_ordered_constrain_kernel), u::AbstractVector{Float64},
        lo::Int, hi::Int)
    v = view(u, lo:hi)
    length(y) == length(v) || return _ordered_constrain_kernel(u, lo, hi)
    _ordered_constrain_native!(y, v)
end

@inline function MutatingFunctions.apply!!(cache::Tuple{Vector{Float64},Float64},
        ::typeof(_simplex_constrain_logjac_kernel), u::AbstractVector{Float64},
        lo::Int, hi::Int)
    p = cache[1]
    length(p) == hi - lo + 2 || return _simplex_constrain_logjac_kernel(u, lo, hi)
    _simplex_constrain_logjac_native!(p, u, lo, hi)
end

# The `MvNormal` covariance's lower Cholesky factor and the slices' solved
# margins, the two arrays its density builds.
@inline function MutatingFunctions.apply!!(F::Matrix{Float64},
        ::typeof(_covariance_cholesky), Sigma::AbstractMatrix{Float64}, k::Int)
    size(F) == (k, k) && !Base.mightalias(F, Sigma) ||
        return _covariance_cholesky(Sigma, k)
    _covariance_cholesky!(F, Sigma, k)
end

@inline function MutatingFunctions.apply!!(Z::Matrix{Float64},
        ::typeof(_lower_solve_rows), X::AbstractMatrix{Float64},
        F::AbstractMatrix{Float64})
    size(Z) == reverse(size(X)) && !Base.mightalias(Z, X) &&
        !Base.mightalias(Z, F) || return _lower_solve_rows(X, F)
    _lower_solve_rows!(Z, X, F)
end
