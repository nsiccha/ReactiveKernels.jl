# Gaussian algebra for selected-coordinate inference. The covariance is
# Diagonal(diagonal) + factors * core * factors'. Only the small rank-sized
# core is dense. In canonical form factors = sqrt.(diagonal) .* vectors,
# vectors have orthonormal columns and core = Diagonal(values).
# This is numerical machinery; authored coordinate identities live in PPL.
struct _CompactGaussian
    mean::Vector{Float64}
    diagonal::Vector{Float64}
    vectors::Matrix{Float64}
    values::Vector{Float64}
end

function _compact_gaussian(mean, diagonal, factors=zeros(length(mean), 0),
                           core=zeros(0, 0); maxrank::Integer=size(factors, 2))
    n, k = size(factors)
    n == length(mean) == length(diagonal) && size(core) == (k, k) ||
        throw(DimensionMismatch("Gaussian mean, diagonal and low-rank covariance"))
    maxrank >= 0 || throw(ArgumentError("Gaussian rank budget must be nonnegative"))
    all(isfinite, mean) && all(isfinite, factors) && all(isfinite, core) ||
        throw(ArgumentError("Gaussian moments must be finite"))
    all(x -> isfinite(x) && x > 0, diagonal) ||
        throw(ArgumentError("Gaussian diagonal must be finite and positive"))
    issymmetric(core) || throw(ArgumentError("Gaussian covariance core must be symmetric"))
    r = min(n, k)
    if r == 0
        return _CompactGaussian(Vector{Float64}(mean), Vector{Float64}(diagonal),
            zeros(n, 0), Float64[]), (; rank=0, discarded_rank=0, relative_loss=0.0)
    end
    # Never materialize the square Q of a tall QR. Multiplication by its first
    # r coordinate vectors allocates n*r, including when k exceeds n.
    decomposition = qr(Matrix{Float64}(factors) ./ sqrt.(diagonal))
    first_columns = zeros(n, r)
    for j in 1:r
        first_columns[j, j] = 1
    end
    Q = decomposition.Q * first_columns
    R = Matrix(decomposition.R)
    eig = eigen(Symmetric(R * core * R'))
    all(x -> isfinite(x) && x > -1, eig.values) ||
        throw(ArgumentError("Gaussian covariance is not positive definite at working precision"))
    order = sortperm(abs.(eig.values); rev=true)
    retained = order[1:min(maxrank, r)]
    omitted = order[min(maxrank, r)+1:end]
    vectors = Q * eig.vectors[:, retained]
    result = _CompactGaussian(Vector{Float64}(mean), Vector{Float64}(diagonal),
        vectors, Vector{Float64}(eig.values[retained]))
    # Loss is the spectral norm of the discarded covariance correction in
    # the diagonal-whitened frame, not a posterior-accuracy certificate.
    loss = isempty(omitted) ? 0.0 : maximum(abs, eig.values[omitted])
    return result, (; rank=length(retained), discarded_rank=length(omitted), relative_loss=loss)
end

_gaussian_factors(g::_CompactGaussian) = sqrt.(g.diagonal) .* g.vectors
_gaussian_core(g::_CompactGaussian) = Diagonal(g.values)

# A product H1*...*Hk*Diagonal(scale) is diagonal plus k outer
# products. Convert its covariance, never a principal slice of its scale.
function _gaussian_from_reflections(mean, scale, indices, reflections)
    n, k = length(mean), length(reflections)
    length(scale) == n && length(indices) == k ||
        throw(DimensionMismatch("Gaussian reflection scale"))
    all(x -> isfinite(x) && x != 0, scale) ||
        throw(ArgumentError("Gaussian scales must be finite and nonzero"))
    U, V = zeros(n, k), zeros(n, k)
    for (column, reflection) in enumerate(k:-1:1)
        rows, values = indices[reflection], reflections[reflection]
        length(rows) == length(values) && allunique(rows) &&
            all(i -> 1 <= i <= n, rows) && all(isfinite, values) ||
            throw(ArgumentError("invalid Gaussian reflection"))
        u, v = view(U, :, column), view(V, :, column)
        u[rows] = values
        v .= -2 .* scale .* u
        for previous in 1:column-1
            v .-= 2dot(u, view(U, :, previous)) .* view(V, :, previous)
        end
    end
    core = zeros(2k, 2k)
    core[1:k, 1:k] = V' * V
    for j in 1:k
        core[j, k+j] = core[k+j, j] = 1
    end
    first(_compact_gaussian(mean, abs2.(scale), hcat(U, scale .* V), core))
end

function _gaussian_indices(g::_CompactGaussian, indices)
    selected = Vector{Int}(indices)
    all(i -> 1 <= i <= length(g.mean), selected) && allunique(selected) ||
        throw(ArgumentError("Gaussian selection must contain distinct in-range coordinates"))
    selected
end

function _gaussian_marginal(g::_CompactGaussian, indices)
    selected = _gaussian_indices(g, indices)
    factors = sqrt.(g.diagonal[selected]) .* g.vectors[selected, :]
    first(_compact_gaussian(g.mean[selected], g.diagonal[selected], factors,
        _gaussian_core(g)))
end

# Data-only covariance, inverse, square-root and log-density operations.
# Ordinary AD differentiates these primal expressions; no derivative rule.
function _gaussian_covariance_mul(g::_CompactGaussian, x::AbstractVecOrMat)
    factors = _gaussian_factors(g)
    g.diagonal .* x + factors * (g.values .* (factors' * x))
end

function _gaussian_solve(g::_CompactGaussian, x::AbstractVecOrMat)
    sd = sqrt.(g.diagonal)
    z = x ./ sd
    (z - g.vectors * ((g.values ./ (1 .+ g.values)) .* (g.vectors' * z))) ./ sd
end

function _gaussian_whiten(diagonal, vectors, values, x)
    z = x ./ sqrt.(diagonal)
    z - vectors * ((1 .- inv.(sqrt.(1 .+ values))) .* (vectors' * z))
end

function _gaussian_color(g::_CompactGaussian, z::AbstractVector)
    sqrt.(g.diagonal) .* (z + g.vectors *
        ((sqrt.(1 .+ g.values) .- 1) .* (g.vectors' * z)))
end

function _gaussian_logdensity(mean, diagonal, vectors, values, x)
    z = _gaussian_whiten(diagonal, vectors, values, x .- mean)
    -(length(mean) * log(2pi) + sum(log, diagonal; init=0.0) +
        sum(log1p, values; init=0.0) + sum(abs2, z)) / 2
end
_gaussian_logdensity(g::_CompactGaussian, x) =
    _gaussian_logdensity(g.mean, g.diagonal, g.vectors, g.values, x)

function _gaussian_draw(rng, g::_CompactGaussian)
    g.mean + _gaussian_color(g, randn(rng, length(g.mean)))
end

# The inactive regression on active coordinates has rank at most the carry's
# rank: Cov(i,a) / Cov(a,a) = factors[i,:] * regression. Retain that small
# operator, not an inactive-by-active dense matrix or an old target wrapper.
function _gaussian_conditional(g::_CompactGaussian, indices)
    active = _gaussian_indices(g, indices)
    mask = trues(length(g.mean))
    mask[active] .= false
    inactive = findall(mask)
    marginal = _gaussian_marginal(g, active)
    factors = _gaussian_factors(g)
    A = factors[active, :]
    regression = g.values .* _gaussian_solve(marginal, A)'
    core = Matrix(_gaussian_core(g)) - regression * A * _gaussian_core(g)
    # Symmetrize roundoff from the small solve/product, not the input model.
    core = (core + core') / 2
    residual = first(_compact_gaussian(g.mean[inactive], g.diagonal[inactive],
        factors[inactive, :], core))
    return (; active, inactive, active_mean=copy(marginal.mean),
        factors=factors[inactive, :], regression, residual)
end

function _gaussian_recover(rng, conditional, active_values)
    length(active_values) == length(conditional.active) ||
        throw(DimensionMismatch("Gaussian conditional active point"))
    _gaussian_draw(rng, conditional.residual) + conditional.factors *
        (conditional.regression * (active_values - conditional.active_mean))
end

# Replace an active marginal by a Gaussian on active+fresh coordinates. New
# coordinates follow all old coordinates in the returned frame; new_active
# names the old-active coordinates' slots in the supplied new marginal.
# No inactive draws are needed to integrate the old conditional. Flatten once
# at the boundary and explicitly truncate the resulting rank if requested.
function _gaussian_replace_marginal(old::_CompactGaussian, active_indices,
        replacement::_CompactGaussian; new_active=1:length(active_indices),
        maxrank::Integer=2length(old.values)+length(replacement.values))
    active = _gaussian_indices(old, active_indices)
    slots = _gaussian_indices(replacement, new_active)
    length(slots) == length(active) || throw(DimensionMismatch("Gaussian replacement active slots"))
    conditional = _gaussian_conditional(old, active)
    inactive = conditional.inactive
    fresh_mask = trues(length(replacement.mean))
    fresh_mask[slots] .= false
    fresh = findall(fresh_mask)
    total = length(old.mean) + length(fresh)
    target = zeros(Int, length(replacement.mean))
    target[slots] = active
    target[fresh] = length(old.mean) .+ collect(eachindex(fresh))
    mean, diagonal = zeros(total), zeros(total)
    mean[target] = replacement.mean
    mean[inactive] = conditional.residual.mean + conditional.factors *
        (conditional.regression * (replacement.mean[slots] - conditional.active_mean))
    diagonal[target] = replacement.diagonal
    diagonal[inactive] = old.diagonal[inactive]
    k, q = size(conditional.factors, 2), length(replacement.values)
    # Joint covariance uses three column groups: inactive carry factors,
    # replacement factors, and Cov(replacement, active)*regression'.
    V = _gaussian_factors(replacement)
    selector = zeros(length(replacement.mean), k)
    selector[slots, :] = conditional.regression'
    cross = _gaussian_covariance_mul(replacement, selector)
    factors = zeros(total, 2k+q)
    factors[inactive, 1:k] = conditional.factors
    factors[target, k+1:k+q] = V
    factors[target, k+q+1:2k+q] = cross
    core = zeros(2k+q, 2k+q)
    A = _gaussian_factors(old)[active, :]
    residual_core = Matrix(_gaussian_core(old)) - conditional.regression * A * _gaussian_core(old)
    core[1:k, 1:k] = residual_core + conditional.regression * cross[slots, :]
    for j in 1:q
        core[k+j, k+j] = replacement.values[j]
    end
    for j in 1:k
        core[j, k+q+j] = core[k+q+j, j] = 1
    end
    _compact_gaussian(mean, diagonal, factors, (core + core') / 2; maxrank)
end
