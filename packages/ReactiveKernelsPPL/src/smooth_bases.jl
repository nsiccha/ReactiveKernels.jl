# Data-side smooth bases (user decision `1cmodra`, prong `smooths`): plain
# Julia functions that return the basis matrices of a penalized spline or a
# Hilbert-space GP approximation, plus the spectral weights and validity
# floors the shipped library submodels (`library.jl`) compute with. Called
# from an `@rkppl` definition over data only (`(Xf, Zp) = tps_basis(x; k = 4)`),
# they run once in `bind_data`; the spectral functions run in the kernel
# under generic AD. They reuse the built-in constructs' fit code
# (`preprocessing.jl`, `_hsgp_axis_fit`, `_hsgp_lambda`, `_hsgp_axis_floor`),
# so a basis column here equals the built-in's bound column on the same data.

"""
    tps_basis(x; k = 10) -> (X, Z)

Thin-plate regression spline basis over the data vector `x` with basis
dimension `k` (`k > 2`, at least `k` distinct values of `x`).

- `X` (`n × 1`): the unpenalized null space, the centered linear column.
  The constant column is left out; the model's intercept owns it.
- `Z` (`n × (k - 2)`): the penalized range space, whitened so that an iid
  `Normal(0, sd)` coefficient prior is the smoothing penalty.

These are the columns the built-in `spline_basis(:id, x; k)` binds as
`<id>_Xnull_1` and `<id>_Zpen_j`. Use them with
[`penalized_smooth`](@ref).
"""
function tps_basis(x::AbstractVector{<:Real}; k::Integer = 10)
    X, Z = _rk_apply_spline(_rk_fit_spline(x; k = Int(k)), x)
    return (Matrix{Float64}(X), Matrix{Float64}(Z))
end

"""
    t2_basis(x, z; k = (5, 5)) -> (X, Zrr, Zrn, Znr)

Tensor-product (`t2`) cubic regression spline basis over the data vectors
`x` and `z`, with per-margin basis dimensions `k` (an integer for both
margins, or a `(k1, k2)` pair, each `> 2`).

- `X` (`n × 3`): the unpenalized null-space products, column-centered (the
  constant product is left out; the model's intercept owns it).
- `Zrr`, `Zrn`, `Znr`: the three penalized blocks (range × range, range ×
  null, null × range), of widths `(k1-2)(k2-2)`, `2(k1-2)` and `2(k2-2)`,
  each penalized by its own smoothing sd.

These are the columns the built-in `spline_basis(:id, x, z; k)` binds. Use
them with [`t2_smooth`](@ref).
"""
function t2_basis(x::AbstractVector{<:Real}, z::AbstractVector{<:Real};
        k = (5, 5))
    kk = _t2_k(k)
    X, rr, rn, nr = _rk_apply_t2(_rk_fit_t2(x, z; k = kk), x, z)
    return (Matrix{Float64}(X), Matrix{Float64}(rr), Matrix{Float64}(rn),
        Matrix{Float64}(nr))
end

_t2_k(k::Integer) = (Int(k), Int(k))
_t2_k(k::Tuple{Integer,Integer}) = (Int(k[1]), Int(k[2]))
_t2_k(k) = throw(ArgumentError(
    "t2_basis: `k` is an integer or a `(k1, k2)` pair, got $(repr(k))"))

# One value per axis: a scalar applies to every axis, a tuple or vector
# gives one entry per axis.
_per_axis(v::Number, d::Int, what) = fill(v, d)
function _per_axis(v::Union{Tuple,AbstractVector}, d::Int, what)
    length(v) == d || throw(ArgumentError(
        "hsgp_basis: `$what` takes one value or one per axis ($d), got " *
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
smallest length scale the basis resolves. Use them with
[`hsgp_effect`](@ref) or [`hsgp_grouped_effect`](@ref). For one length scale
per axis, write the explicit statements documented under [`hsgp_effect`](@ref).
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
    # Per-axis sines (SB `_brm_apply_hsgp`, the built-in's in-graph form):
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
    hsgp_periodic_basis(x; k = 20, period) -> (PHI, harmonics)

Hilbert-space approximate GP basis for the periodic kernel with the given
`period` (on the scale of `x`): `k` harmonics of the angular frequency
`2pi / period` as `2k` columns, the cosines then the sines. `harmonics`
(length `2k`) is the harmonic index of each column (`1..k, 1..k`).
[`hsgp_periodic_sqrt_spd`](@ref) turns it into the coefficient weights. Use
them with [`hsgp_periodic_effect`](@ref).
"""
function hsgp_periodic_basis(x::AbstractVector{<:Real}; k::Integer = 20,
        period::Real)
    k > 0 || throw(ArgumentError("hsgp_periodic_basis: `k` must be positive"))
    isfinite(period) && period > 0 || throw(ArgumentError(
        "hsgp_periodic_basis: `period` must be finite and positive"))
    all(isfinite, x) || throw(ArgumentError(
        "hsgp_periodic_basis: x must be finite"))
    xs = Float64.(x)
    w0 = 2.0 * pi / Float64(period)
    PHI = hcat((cos.((w0 * j) .* xs) for j in 1:k)...,
        (sin.((w0 * j) .* xs) for j in 1:k)...)
    return (Matrix{Float64}(PHI), vcat(collect(1:Int(k)), collect(1:Int(k))))
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
    hsgp_grouped_sqrt_spd(lambda, sigma, rho) -> Vector

Coefficient weights of a grouped [`hsgp_basis`](@ref) (`by = g`): one
isotropic exponentiated-quadratic spectral weight per basis function and
group, group fastest (the grouped basis's column order). `sigma` and `rho`
are each one value per group (`length(levels(g))`) or one shared scalar.
"""
function hsgp_grouped_sqrt_spd(lambda::AbstractMatrix, sigma, rho)
    d = size(lambda, 2)
    scale = sigma .* sqrt.(rho .* _HSGP_SQRT2PI) .^ d
    E = exp.(-0.25 .* (rho .* rho) .* permutedims(vec(sum(lambda; dims = 2))))
    return vec(scale .* E)
end

"""
    hsgp_periodic_sqrt_spd(harmonics, sigma, rho) -> Vector

Coefficient weights of an [`hsgp_periodic_basis`](@ref): for harmonic `j`,
`sigma * sqrt(2 * exp(-a) * besseli(j, a))` with `a = 1 / rho^2`, evaluated
through the exponentially scaled `besselix` so it never overflows.
"""
function hsgp_periodic_sqrt_spd(harmonics::AbstractVector, sigma::Number,
        rho::Number)
    a = 1.0 / (rho * rho)
    return exp.(log(sigma) .+ 0.5 .* (0.6931471805599453 .+
        log.(SpecialFunctions.besselix.(harmonics, a))))
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

"""
    hsgp_periodic_rho_floor(harmonics) -> Float64

Validity floor of an [`hsgp_periodic_basis`](@ref): the length scale at
which the highest harmonic's spectral weight falls to 1/100 of the first's
(`0.0` for a single harmonic).
"""
hsgp_periodic_rho_floor(harmonics::AbstractVector) =
    _hsgp_periodic_rho_lower(maximum(harmonics))
