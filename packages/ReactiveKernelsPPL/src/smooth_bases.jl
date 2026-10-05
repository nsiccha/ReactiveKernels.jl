# Temporary compatibility for the published consumer's three imports.
# BRM owns statistical preparation. Remove this compatibility after that
# consumer adopts BayesianRegressionModels.StatisticalPreparation.
const _HSGP_SQRT2PI = 2.5066282746310002

"""One axis's length-scale validity floor for `K` basis functions on the
half-width-`L` domain (SB `_brm_hsgp_rho_lower`): `(4L/pi) *
sqrt(log(100)/(K^2-1))`, `0.0` (unbounded) at `K == 1`. Shared by the
temporary [`hsgp_rho_floors`](@ref) helper."""
_hsgp_axis_floor(K::Integer, L::Real) =
    K == 1 ? 0.0 : (4 * L / pi) * sqrt(log(100.0) / (K * K - 1))

"""Exp-quad HSGP eigenvalue of basis function `k` on the half-width-`L`
domain (SB `lambda`): `(k*pi/(2L))^2`, used by the temporary
data-side [`hsgp_basis`](@ref) helper."""
_hsgp_lambda(k::Integer, L::Real) = (k * pi / (2.0 * L))^2

"""One HSGP axis fit (SB `_brm_fit_hsgp` 1-D verbatim): `mu =
mean(col)`, `L = c*max|col-mu|`, numeric/nonempty/finite/`L > 0`
gates for the temporary helper."""
function _hsgp_axis_fit(col::AbstractVector, c::Real, label::Symbol,
        where::String)
    eltype(col) <: Real ||
        _fail(label, "$where must be numeric, got $(eltype(col))")
    isempty(col) &&
        _fail(label, "$where is empty")
    mu = sum(col) / length(col)
    L = Float64(c) * maximum(abs.(col .- mu))
    isfinite(mu) && isfinite(L) ||
        _fail(label, "$where is non-finite (mu=$mu, L=$L)")
    L > 0 ||
        _fail(label, "$where is degenerate " *
              "(L == 0 — a constant column has no usable domain)")
    return (Float64(mu), L)
end

_per_axis(v::Number, d::Int, what) = fill(v, d)
function _per_axis(v::Union{Tuple,AbstractVector}, d::Int, what)
    length(v) == d || throw(ArgumentError(
        "`$what` takes one value or one per axis ($d), got " *
        repr(v)))
    return collect(v)
end

"""
    hsgp_basis(x...; k = 20, c = 1.5, domain = nothing, by = nothing) -> (PHI, lambda)

Hilbert-space approximate GP basis (exponentiated-quadratic kernel) over one
or more data vectors `x...` (the axes).

- `k`: basis functions per axis (an integer, or one per axis); the basis has
  `M = prod(k)` functions, the tensor products of the per-axis sines in
  `CartesianIndices(k)` order.
- `c`: the boundary factor (`> 1`, or one per axis). Each axis is centered at
  its mean and the domain half-width is `L = c * maximum(abs.(x .- mean(x)))`.
- `domain`: a fixed `(lo, hi)` (one axis) or one pair per axis instead of the
  data-derived domain (`c` is then unused); every value must lie inside.
- `by`: a grouping vector. The basis then repeats once per group level
  (`levels(by)` order), each copy zero outside its group: `PHI` has
  `G * M` columns, group fastest (column `(m - 1) * G + g` is basis
  function `m` on group `g`).

Returns `PHI` (`n × M`, or `n × G*M` with `by`) and `lambda` (`M × d`): the
per-axis eigenvalues `(k_j * pi / (2 L_j))^2` of each basis function.
[`hsgp_sqrt_spd`](@ref) turns `lambda` and the kernel's scale and length
scale into the coefficient weights, and [`hsgp_rho_floors`](@ref) gives the
smallest length scale the basis resolves. Statistical preparation belongs
to `BayesianRegressionModels.StatisticalPreparation`; import its helpers
into the model module and call them as ordinary functions.
"""
function hsgp_basis(axes_::AbstractVector{<:Real}...; k = 20, c = 1.5,
        domain = nothing, by = nothing)
    d = length(axes_)
    d >= 1 || throw(ArgumentError("hsgp_basis takes at least one axis"))
    n = length(first(axes_))
    all(a -> length(a) == n, axes_) || throw(ArgumentError(
        "hsgp_basis: axes have different lengths $(map(length, axes_))"))
    K = Int.(_per_axis(k, d, :k))
    all(>(0), K) || throw(ArgumentError(
        "hsgp_basis: `k` must be positive, got $(repr(k))"))
    fits = Vector{Tuple{Float64,Float64}}(undef, d)
    if domain === nothing
        C = Float64.(_per_axis(c, d, :c))
        all(>(1), C) || throw(ArgumentError(
            "hsgp_basis: `c` must exceed 1, got $(repr(c))"))
        for j in 1:d
            fits[j] = _hsgp_axis_fit(axes_[j], C[j], :hsgp_basis,
                "hsgp_basis axis $j")
        end
    else
        pairs = d == 1 && domain isa Tuple{Real,Real} ? [domain] :
            _per_axis(domain, d, :domain)
        for j in 1:d
            lo, hi = Float64.(pairs[j])
            isfinite(lo) && isfinite(hi) && lo < hi || throw(ArgumentError(
                "hsgp_basis: domain $j must be finite with lo < hi, got " *
                repr(pairs[j])))
            all(v -> lo <= v <= hi, axes_[j]) || throw(ArgumentError(
                "hsgp_basis: axis $j has values outside its domain " *
                "($lo, $hi)"))
            fits[j] = ((lo + hi) / 2, (hi - lo) / 2)
        end
    end
    # Per-axis sines (SB `_brm_apply_hsgp`):
    # `inv(sqrt(L)) * sin(sqrt(lambda_k) * (x - mu + L))`.
    cols = map(1:d) do j
        mu, L = fits[j]
        inv_sqrt_L = 1.0 / sqrt(L)
        xs = Float64.(axes_[j])
        [inv_sqrt_L .* sin.(sqrt(_hsgp_lambda(kk, L)) .* (xs .- mu .+ L))
            for kk in 1:K[j]]
    end
    idx = vec(collect(CartesianIndices(Tuple(K))))
    M = length(idx)
    PHI = Matrix{Float64}(undef, n, M)
    lambda = Matrix{Float64}(undef, M, d)
    for (b, I) in enumerate(idx)
        col = copy(cols[1][I[1]])
        for j in 2:d
            col .*= cols[j][I[j]]
        end
        PHI[:, b] = col
        for j in 1:d
            lambda[b, j] = _hsgp_lambda(I[j], fits[j][2])
        end
    end
    by === nothing && return (PHI, lambda)
    return (_grouped_basis(PHI, by), lambda)
end

# The basis repeated per group level, zero outside the group, group
# fastest: column `(m - 1) * G + g` is `PHI[:, m]` on the rows of level `g`.
function _grouped_basis(PHI::AbstractMatrix, by::AbstractVector)
    size(PHI, 1) == length(by) || throw(ArgumentError(
        "hsgp_basis: `by` has $(length(by)) values for $(size(PHI, 1)) rows"))
    levels = _grouping_levels(by)
    codes = _declared_codes(by, levels)
    G, M = length(levels), size(PHI, 2)
    out = zeros(Float64, size(PHI, 1), G * M)
    for m in 1:M, i in axes(PHI, 1)
        out[i, (m - 1) * G + codes[i]] = PHI[i, m]
    end
    return out
end

"""
    hsgp_sqrt_spd(lambda, sigma, rho) -> Vector

Coefficient weights of an [`hsgp_basis`](@ref): the square root of the
exponentiated-quadratic kernel's spectral density at each basis function's
eigenvalues `lambda` (`M × d`), for marginal scale `sigma` and length scale
`rho` (a scalar, isotropic, or one per axis):
`sigma * prod_j sqrt(rho_j * sqrt(2pi)) * exp(-rho_j^2 * lambda_j / 4)`.
"""
function hsgp_sqrt_spd(lambda::AbstractMatrix, sigma::Number, rho::Number)
    scale = sigma * sqrt(rho * _HSGP_SQRT2PI)^size(lambda, 2)
    return scale .* exp.(-0.25 .* (rho * rho) .* vec(sum(lambda; dims = 2)))
end

function hsgp_sqrt_spd(lambda::AbstractMatrix, sigma::Number,
        rho::AbstractVector)
    length(rho) == size(lambda, 2) || throw(ArgumentError(
        "hsgp_sqrt_spd: one length scale per axis ($(size(lambda, 2))), " *
        "got $(length(rho))"))
    scale = sigma * prod(sqrt.(rho .* _HSGP_SQRT2PI))
    return scale .* exp.(-0.25 .* (lambda * (rho .* rho)))
end

"""
    hsgp_rho_floors(lambda) -> Vector{Float64}

Per-axis validity floors of an [`hsgp_basis`](@ref) from its eigenvalues
`lambda` (`M × d`): the length scale below which axis `j`'s `K_j`-term
approximation degrades, `(4 L_j / pi) * sqrt(log(100) / (K_j^2 - 1))`
(`0.0` when `K_j == 1`). The isotropic floor is their `maximum`.
"""
function hsgp_rho_floors(lambda::AbstractMatrix)
    return map(axes(lambda, 2)) do j
        lo, hi = extrema(view(lambda, :, j))
        lo > 0 || throw(ArgumentError(
            "hsgp_rho_floors: eigenvalues must be positive"))
        _hsgp_axis_floor(round(Int, sqrt(hi / lo)), pi / (2 * sqrt(lo)))
    end
end
