# Multivariate slice priors on declared arrays: `eachrow(B[a, b]) .~ D`,
# `eachcol(B[a, b]) .~ D` (every row / column of the two-axis array `B` one
# draw of the multivariate `D`) and `b[1:K] ~ D` (the vector `b` one draw).
# Julia's `eachrow` / `eachcol` make the slices the broadcast elements, and
# a distribution broadcasts as a scalar (Distributions.jl), so `D` is drawn
# once per slice. A dotted `D.(args...)` pairs slice `g` with slice `g` of a
# per-slice argument `eachrow(M)` / `eachcol(M)`; `Ref(x)` and the
# arguments of an undotted `D(args...)` are shared by every slice.
#
# This file holds the runtime half shared by the generated kernel and the
# host layout: slice orientation, per-slice arguments, the constrained
# transforms of simplex / ordered slices, and the slice densities. Every
# function here runs the same arithmetic in the graph and on the host.

# ── orientation ──────────────────────────────────────────────────────

"""Rows of a two-axis array are the slices (`eachrow`)."""
struct _SliceRows end
"""Columns of a two-axis array are the slices (`eachcol`)."""
struct _SliceCols end
"""A one-axis array is the one slice (`b[1:K] ~ D`)."""
struct _SliceWhole end

_slice_orientation(s::Symbol) = _slice_orientation(Val(s))
_slice_orientation(::Val{:rows}) = _SliceRows()
_slice_orientation(::Val{:cols}) = _SliceCols()
_slice_orientation(::Val{:vector}) = _SliceWhole()

_slice_count(::_SliceRows, B) = size(B, 1)
_slice_count(::_SliceCols, B) = size(B, 2)
_slice_count(::_SliceWhole, b) = 1
_slice_length(::_SliceRows, B) = size(B, 2)
_slice_length(::_SliceCols, B) = size(B, 1)
_slice_length(::_SliceWhole, b) = length(b)
@inline _slice_entry(::_SliceRows, B, g, i) = B[g, i]
@inline _slice_entry(::_SliceCols, B, g, i) = B[i, g]
@inline _slice_entry(::_SliceWhole, b, g, i) = b[i]

# The slices as the rows of a matrix (one slice per row) — the layout the
# vectorized densities below reduce over. Fresh for columns.
_slice_rows(::_SliceRows, B) = B
_slice_rows(::_SliceCols, B) = permutedims(B)
_slice_rows(::_SliceWhole, b) = reshape(b, 1, length(b))

# ── per-slice arguments ──────────────────────────────────────────────

"""
    _PerSlice(orientation, value)

A per-slice argument of a dotted slice prior: `eachrow(M)` / `eachcol(M)`
pairs slice `g` of the declared array with slice `g` of `M`.
"""
struct _PerSlice{O,A}
    orientation::O
    value::A
end

@inline _arg_entry(v::AbstractVector, g, i) = v[i]
@inline _arg_entry(a::_PerSlice, g, i) =
    _slice_entry(a.orientation, a.value, g, i)

# Slice length and count of an argument (`nothing`: shared by every slice).
_arg_length(v::AbstractVector) = length(v)
_arg_length(a::_PerSlice) = _slice_length(a.orientation, a.value)
_arg_count(::AbstractVector) = nothing
_arg_count(a::_PerSlice) = _slice_count(a.orientation, a.value)

# A vector argument as a 1×K row (shared: broadcasts over the slices) or
# a G×K matrix (per slice).
_arg_rows(v::AbstractVector) = reshape(v, 1, length(v))
_arg_rows(a::_PerSlice) = _slice_rows(a.orientation, a.value)

function _check_slice_vector_arg(what, a, groups, k)
    _arg_length(a) == k || throw(DimensionMismatch("$what has slices of " *
        "length $(_arg_length(a)), but the declared slices have length $k"))
    n = _arg_count(a)
    n === nothing || n == groups || throw(DimensionMismatch("$what has $n " *
        "slices, but the declared array has $groups"))
    return nothing
end

# ── multivariate normal slices (native) ──────────────────────────────

# Sum the row densities by forward substitution. The inputs and factor
# remain read-only; the solve buffer is fresh and local to this call.
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

# Σ log F[i, i] of a K×K lower-triangular Cholesky factor, checking the
# positive diagonal and the zero upper triangle.
function _cholesky_factor_logdet(F, k)
    size(F) == (k, k) || throw(DimensionMismatch("MvNormalCholesky slices " *
        "have length $k, but the factor has size $(size(F))"))
    logdet = 0.0
    for i in 1:k
        F[i, i] > 0 || throw(ArgumentError("MvNormalCholesky factor has a " *
            "nonpositive diagonal entry F[$i, $i]"))
        for j in (i + 1):k
            iszero(F[i, j]) || throw(ArgumentError("MvNormalCholesky factor " *
                "is not lower triangular: F[$i, $j] = $(F[i, j])"))
        end
        logdet += log(F[i, i])
    end
    return logdet
end

# The lower Cholesky factor of a K×K covariance (Cholesky–Banachiewicz) in
# a fresh local buffer. Reads the lower triangle; the upper must agree to
# Stan's symmetry tolerance (1e-8).
function _covariance_cholesky(Sigma, k)
    size(Sigma) == (k, k) || throw(DimensionMismatch("MvNormal slices " *
        "have length $k, but the covariance has size $(size(Sigma))"))
    F = zeros(eltype(Sigma), k, k)
    for i in 1:k
        for j in 1:(i - 1)
            abs(Sigma[i, j] - Sigma[j, i]) <= 1e-8 ||
                throw(ArgumentError("MvNormal covariance is not symmetric: " *
                    "Sigma[$i, $j] = $(Sigma[i, j]), Sigma[$j, $i] = " *
                    "$(Sigma[j, i])"))
        end
        for j in 1:i
            s = Sigma[i, j]
            for p in 1:(j - 1)
                s -= F[i, p] * F[j, p]
            end
            if i == j
                s > 0 || throw(ArgumentError("MvNormal covariance is not " *
                    "positive definite (pivot $i)"))
                F[i, i] = sqrt(s)
            else
                F[i, j] = s / F[j, j]
            end
        end
    end
    return F
end

@inline function _mvnormal_slices_core(o, B, mu, F, logdet)
    groups, k = _slice_count(o, B), _slice_length(o, B)
    return _lower_solve_rows_logpdf(
        (g, i) -> _slice_entry(o, B, g, i) - _arg_entry(mu, g, i),
        (i, j) -> F[i, j], logdet, k, groups)
end

"""
    _mvnormal_cholesky_slices_logpdf(o, B, mu, F)

Σ over the slices `x_g` of `B` (orientation `o`) of
`logpdf(MvNormal(mu_g, F * F'), x_g)`, `F` the lower-triangular Cholesky
factor of the covariance (Stan's `multi_normal_cholesky`). `mu` is a
shared K-vector or a [`_PerSlice`](@ref) argument. Native execution: the
slices are an ordinary runtime loop.
"""
@inline function _mvnormal_cholesky_slices_logpdf(o, B, mu, F)
    ReactiveKernels._dynamic_tensorized_marker((B, mu, F)) === nothing ||
        throw(ArgumentError("MvNormalCholesky slice priors support native " *
            "execution only"))
    groups, k = _slice_count(o, B), _slice_length(o, B)
    _check_slice_vector_arg("the MvNormalCholesky mean", mu, groups, k)
    return _mvnormal_slices_core(o, B, mu, F, _cholesky_factor_logdet(F, k))
end

"""
    _mvnormal_slices_logpdf(o, B, mu, Sigma)

Σ over the slices `x_g` of `B` of `logpdf(MvNormal(mu_g, Sigma), x_g)`
for a symmetric positive-definite covariance `Sigma`, through its lower
Cholesky factor. Native execution.
"""
@inline function _mvnormal_slices_logpdf(o, B, mu, Sigma)
    ReactiveKernels._dynamic_tensorized_marker((B, mu, Sigma)) === nothing ||
        throw(ArgumentError("MvNormal slice priors support native execution " *
            "only"))
    groups, k = _slice_count(o, B), _slice_length(o, B)
    _check_slice_vector_arg("the MvNormal mean", mu, groups, k)
    F = _covariance_cholesky(Sigma, k)
    logdet = 0.0
    for i in 1:k
        logdet += log(F[i, i])
    end
    return _mvnormal_slices_core(o, B, mu, F, logdet)
end

# ── simplex and ordered slices (vectorized) ──────────────────────────

@traceable function _dirichlet_slices_logpdf(o, B, alpha)
    groups, k = _slice_count(o, B), _slice_length(o, B)
    _check_slice_vector_arg("the Dirichlet concentration", alpha, groups, k)
    X = _slice_rows(o, B)
    A = _arg_rows(alpha)
    # Invalid live concentrations have zero density. The loggamma branch
    # stays inactive, including its derivative work.
    if all(isfinite.(A) .& (A .> 0))
        normalizer = DistributionKernelSources.loggamma.(sum(A; dims = 2)) .-
            sum(DistributionKernelSources.loggamma.(A); dims = 2)
        sum(normalizer .+ sum((A .- 1.0) .* log.(X); dims = 2))
    else
        -Inf
    end
end

@doc """
    _dirichlet_slices_logpdf(o, B, alpha)

Σ over the slices `x_g` of `B` of `logpdf(Dirichlet(alpha_g), x_g)`:
`loggamma(Σ α) − Σ loggamma(α) + Σ (α − 1) log x` per slice. `alpha` is a
shared K-vector or a per-slice argument. Whole-array broadcasts and
reductions, no loop over the slices.
""" _dirichlet_slices_logpdf

"""
    _ordered_normal_slices_logpdf(o, B, m, s)

Σ over every entry of the ordered slices of `B` of `logpdf(Normal(m, s), x)`
— the `Ordered(Normal(m, s), K)` convention (no `log(K!)` term), applied
per slice.
"""
function _ordered_normal_slices_logpdf(o, B, m, s)
    n = _slice_count(o, B) * _slice_length(o, B)
    return -0.5 * sum(((B .- m) ./ s) .^ 2) - n * (log(s) + 0.5 * log(2pi))
end

# Querying a broadcast of joint slice draws keeps one log density per
# slice. Whole-vector draws remain scalar in the query emitter. These
# native multivariate-normal helpers retain the existing compiled refusal.
function _mvnormal_cholesky_slices_pointwise(o, B, mu, F)
    ReactiveKernels._dynamic_tensorized_marker((B, mu, F)) === nothing ||
        throw(ArgumentError("MvNormalCholesky slice priors support native execution only"))
    groups, k = _slice_count(o, B), _slice_length(o, B)
    _check_slice_vector_arg("the MvNormalCholesky mean", mu, groups, k)
    logdet = _cholesky_factor_logdet(F, k)
    out = zeros(Float64, groups)
    for g in 1:groups
        out[g] = _lower_solve_rows_logpdf(
            (_, i) -> _slice_entry(o, B, g, i) - _arg_entry(mu, g, i),
            (i, j) -> F[i, j], logdet, k, 1)
    end
    return out
end

function _mvnormal_slices_pointwise(o, B, mu, Sigma)
    ReactiveKernels._dynamic_tensorized_marker((B, mu, Sigma)) === nothing ||
        throw(ArgumentError("MvNormal slice priors support native execution only"))
    return _mvnormal_cholesky_slices_pointwise(o, B, mu,
        _covariance_cholesky(Sigma, _slice_length(o, B)))
end

@traceable function _dirichlet_slices_pointwise(o, B, alpha)
    groups, k = _slice_count(o, B), _slice_length(o, B)
    _check_slice_vector_arg("the Dirichlet concentration", alpha, groups, k)
    X, A = _slice_rows(o, B), _arg_rows(alpha)
    if all(isfinite.(A) .& (A .> 0))
        normalizer = DistributionKernelSources.loggamma.(sum(A; dims = 2)) .-
            sum(DistributionKernelSources.loggamma.(A); dims = 2)
        vec(normalizer .+ sum((A .- 1.0) .* log.(X); dims = 2))
    else
        fill(-Inf, groups)
    end
end

function _ordered_normal_slices_pointwise(o, B, m, s)
    X = _slice_rows(o, B)
    k = _slice_length(o, B)
    return vec(-0.5 .* sum(((X .- m) ./ s) .^ 2; dims = 2) .-
        k * (log(s) + 0.5 * log(2pi)))
end

# Ordered slices: the first entry, then cumulative exp-increments along
# each slice (the vector `ordered_constrain`, per slice).
_ordered_slices_constrain(::_SliceRows, U) =
    cumsum(hcat(U[:, 1:1], exp.(U[:, 2:end])); dims = 2)
_ordered_slices_constrain(::_SliceCols, U) =
    cumsum(vcat(U[1:1, :], exp.(U[2:end, :])); dims = 1)
_ordered_slices_logjac(::_SliceRows, U) = sum(U[:, 2:end])
_ordered_slices_logjac(::_SliceCols, U) = sum(U[2:end, :])
_ordered_slices_unconstrain(::_SliceRows, X) =
    hcat(X[:, 1:1], log.(diff(X; dims = 2)))
_ordered_slices_unconstrain(::_SliceCols, X) =
    vcat(X[1:1, :], log.(diff(X; dims = 1)))

# Simplex slices: stick-breaking along each slice, the vector
# `simplex_constrain` per slice. `U` holds K − 1 coordinates per slice.
# The break fractions `z` and the log stick remainders `lr` feed both the
# value and the log-Jacobian. The constant leading column of `lr` and
# trailing column of the value are built from `U` (`0.0 .* U[:, 1:1]`),
# never as a host array, so the transform is broadcasts and reductions
# over `U` alone and traces under Reactant (a host column concatenated
# with a traced one falls back to scalar indexing). `U` is finite, so the
# column is exactly zero. A one-entry simplex (K = 1, no coordinates) is
# the constant 1.
_slice_zero(::_SliceRows, U) = 0.0 .* U[:, 1:1]
_slice_zero(::_SliceCols, U) = 0.0 .* U[1:1, :]
_slice_hcat(::_SliceRows, a, b) = hcat(a, b)
_slice_hcat(::_SliceCols, a, b) = vcat(a, b)
_slice_dims(::_SliceRows) = 2
_slice_dims(::_SliceCols) = 1
_slice_width(o, U) = size(U, _slice_dims(o))
# The K − 1 stick-breaking offsets `log(K − j)`, laid along the slices.
_simplex_offsets(::_SliceRows, K) = permutedims(log.(K .- (1:(K - 1))))
_simplex_offsets(::_SliceCols, K) = log.(K .- (1:(K - 1)))

function _simplex_slices_breaks(o, U)
    K = _slice_width(o, U) + 1
    z = 1.0 ./ (1.0 .+ exp.(-(U .+ _simplex_offsets(o, K))))
    l = log1p.(-z)
    lr = cumsum(_slice_hcat(o, _slice_zero(o, U), l); dims = _slice_dims(o))
    return z, l, lr
end
_slice_ones(::_SliceRows, G) = ones(G, 1)
_slice_ones(::_SliceCols, G) = ones(1, G)
_simplex_lr_head(::_SliceRows, lr) = lr[:, 1:(end - 1)]
_simplex_lr_head(::_SliceCols, lr) = lr[1:(end - 1), :]

function _simplex_slices_constrain(o, U)
    _slice_width(o, U) == 0 && return _slice_ones(o, _slice_count(o, U))
    z, _, lr = _simplex_slices_breaks(o, U)
    return exp.(lr) .* _slice_hcat(o, z, 1.0 .+ _slice_zero(o, U))
end
function _simplex_slices_logjac(o, U)
    _slice_width(o, U) == 0 && return 0.0
    z, l, lr = _simplex_slices_breaks(o, U)
    return sum(_simplex_lr_head(o, lr) .+ log.(z) .+ l)
end
# Host inverse: per slice, the vector `simplex_unconstrain`.
_simplex_slices_unconstrain(::_SliceRows, X) =
    permutedims(reduce(hcat, [simplex_unconstrain(X[g, :])
        for g in 1:size(X, 1)]))
_simplex_slices_unconstrain(::_SliceCols, X) =
    reduce(hcat, [simplex_unconstrain(X[:, g]) for g in 1:size(X, 2)])
