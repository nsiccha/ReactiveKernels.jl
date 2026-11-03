# Adapter from the public reference scale to RK's general Gaussian algebra.
const CompactGaussian = ReactiveKernels._CompactGaussian
const CompactScale = NamedTuple{(:kind, :diagonal, :vectors, :values)}

function PPL._smc_scale(scale::CompactScale)
    scale.kind === :diagonal_lowrank || throw(ArgumentError("unknown compact Gaussian scale kind"))
    n, k = size(scale.vectors)
    length(scale.diagonal) == n && length(scale.values) == k ||
        throw(DimensionMismatch("compact Gaussian scale arrays"))
    all(x -> isfinite(x) && x > 0, scale.diagonal) &&
        all(isfinite, scale.vectors) && all(x -> isfinite(x) && x > -1, scale.values) ||
        throw(ArgumentError("invalid compact Gaussian covariance"))
    isapprox(scale.vectors' * scale.vectors, Matrix{Float64}(LinearAlgebra.I,k,k);
        atol=1e-10, rtol=1e-10) || throw(ArgumentError("compact Gaussian directions must be orthonormal"))
    (; kind=:diagonal_lowrank, diagonal=Vector{Float64}(scale.diagonal),
        vectors=Matrix{Float64}(scale.vectors), values=Vector{Float64}(scale.values))
end

_gaussian(r::SMCReference{<:AbstractVector}) =
    first(ReactiveKernels._compact_gaussian(r.location, abs2.(r.scale)))
_gaussian(r::SMCReference{<:NamedTuple}) = ReactiveKernels._gaussian_from_reflections(
    r.location, r.scale.diagonal, r.scale.indices, r.scale.reflections)
_gaussian(r::SMCReference{<:CompactScale}) =
    CompactGaussian(copy(r.location), copy(r.scale.diagonal), copy(r.scale.vectors), copy(r.scale.values))
_gaussian_reference(g::CompactGaussian) = SMCReference(g.mean,
    (; kind=:diagonal_lowrank, diagonal=g.diagonal, vectors=g.vectors, values=g.values))

function _reference_sd(scale::CompactScale)
    sqrt.(scale.diagonal .* (1 .+ vec(sum(abs2.(scale.vectors) .* scale.values'; dims=2))))
end
_inflate_reference_scale(scale::CompactScale, factor) =
    (; scale..., diagonal=scale.diagonal .* factor^2)

@kernel _compact_normal_reference(u::Vector{Float64}, location::Vector{Float64},
        diagonal::Vector{Float64}, vectors::Matrix{Float64}, values::Vector{Float64}) = begin
    density = ReactiveKernels._gaussian_logdensity(location, diagonal, vectors, values, u)
end

function _reference(target, reference::SMCReference{<:CompactScale})
    _compact_reference(target.backend, _dimension(target), reference)
end

function _compact_reference(backend, dimension, reference::SMCReference{<:CompactScale})
    length(reference.location) == dimension ||
        throw(DimensionMismatch("reference and target dimensions differ"))
    r = SMCReference(reference.location, reference.scale)
    kernel = prepare(_compact_normal_reference; want=:density)
    arguments = (r.location, r.scale.diagonal, r.scale.vectors, r.scale.values)
    ad = Base.invokelatest(prepare_ad, kernel, backend, copy(r.location), arguments...; active=:u)
    ReferenceTarget(r, kernel, ad, arguments, _reference_sd(r.scale))
end

function _reference_draws(rng, reference::SMCReference{<:CompactScale}, d, n)
    g = _gaussian(reference)
    positions = Matrix{Float64}(undef, d, n)
    for j in 1:n
        positions[:,j] = ReactiveKernels._gaussian_draw(rng, g)
    end
    positions
end

# Leading correlation directions of a weighted cloud, as a diagonal-plus-low-rank
# scale: carry_smc(...; basis=:correlation), the learning filter's window
# boundary and its replenishment walk. A diagonal summary discards them: where
# the data inform only combinations of parameters (a hierarchical intercept and
# its group effects), a diagonal prior treats the marginal widths as independent
# information, and a diagonal random walk leaves the narrow combinations at once
# and is rejected. WarmupHMC's moment fitter finds the directions in the
# marginally standardized frame, so coordinates on different scales are treated
# alike. Retained directions keep their weighted variance; the remaining ones
# share the remaining variance equally (probabilistic PCA), which keeps the total
# standardized variance. At most min(max_reflections, d-1, distinct-position
# ESS - 2) directions. Work O(d*members*directions*iterations),
# storage O(d*directions); no d x d matrix is formed. `nothing` when no direction
# is estimable; then the caller uses the diagonal marginal moments.
function _correlation_scale(positions,mass,location,sd,max_reflections)
    d = length(location)
    r = min(max_reflections,d-1,floor(Int,_filter_distinct_ess(positions,mass))-2)
    r >= 1 || return nothing
    standardized = (positions .- location) ./ sd
    fit = WarmupHMC.fit_gaussian_moments(standardized;weights=mass,max_reflections=r)
    scale = fit.scale.diagonal
    k = length(fit.scale.reflections)
    k >= 1 || return nothing
    rotation = WarmupHMC.scale_from_data(fit.scale)
    vectors,unit,column = zeros(d,k),zeros(d),zeros(d)
    for j in 1:k
        fill!(unit,0.0)
        unit[j] = 1.0
        mul!(column,rotation,unit)
        vectors[:,j] .= column ./ scale[j]
    end
    remainder = sum(abs2,view(scale,k+1:d))/(d-k)
    isfinite(remainder) && remainder > 0 || return nothing
    return (; kind=:diagonal_lowrank,diagonal=abs2.(sd) .* remainder,
        vectors,values=abs2.(view(scale,1:k)) ./ remainder .- 1)
end
_correlation_directions(::Nothing) = 0
_correlation_directions(scale) = size(scale.vectors,2)

# A covariance projection is an importance reference, not one of WarmupHMC's
# portable mutation metrics. Never send it to the generic dense-Cholesky path.
_metric(::CompactScale) = throw(ArgumentError(
    "a projected Gaussian reference is not a mutation metric; use a diagonal metric or tune_smc's pilot metric"))
