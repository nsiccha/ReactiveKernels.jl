# Both native helpers and PPL calls use these same graphs. The locations
# broadcast along two independent axes; only the scalar covariance is a leaf.
# Bound data locations permit RK's ordinary inner-plate partial evaluation to
# cache distances. Live locations remain dependencies of every pair cell.
function _gp_indices(x::AbstractVector)
    isempty(x) && throw(ArgumentError("GP covariance needs at least one location"))
    collect(eachindex(x))
end

function _gp_positive(value::Real, name)
    value > 0 || throw(ArgumentError("GP $name must be positive, got $value"))
    Float64(value)
end

function _gp_period(period::Real)
    isfinite(period) && period > 0 || throw(ArgumentError(
        "GP period must be finite and positive, got $period"))
    Float64(period)
end

function _gp_jitter(jitter::Real)
    isfinite(jitter) && jitter >= 0 || throw(ArgumentError(
        "GP jitter must be finite and nonnegative, got $jitter"))
    Float64(jitter)
end

@kernel gp_pair_locations(x) = begin
    indices = _gp_indices(x)
    locations = Float64.(x)
    left = locations
    right = reshape(locations, 1, :)
    row = indices
    col = reshape(indices, 1, :)
    return left, right, row, col
end

@kernel gp_exp_quad_cov_graph(x, sigma, rho, jitter) = begin
    left, right, row, col = gp_pair_locations(x)
    scale = _gp_positive(sigma, :sigma)
    width = _gp_positive(rho, :rho)
    jit = _gp_jitter(jitter)
    variance = scale^2
    denominator = 2 * width^2
    covariance = plate(left, right, row, col, Ref(variance), Ref(denominator), Ref(jit)) do xi, xj, i, j, s2, denom, eps
        distance = xi - xj
        squared_distance = distance * distance
        cell = s2 * exp(-squared_distance / denom)
        diagonal_jitter = ifelse(i == j, eps, 0.0)
        cell + diagonal_jitter
    end
    return covariance
end

@kernel gp_periodic_cov_graph(x, sigma, rho, period, jitter) = begin
    left, right, row, col = gp_pair_locations(x)
    scale = _gp_positive(sigma, :sigma)
    width = _gp_positive(rho, :rho)
    per = _gp_period(period)
    jit = _gp_jitter(jitter)
    variance = scale^2
    squared_width = width^2
    covariance = plate(left, right, row, col, Ref(variance), Ref(squared_width), Ref(per), Ref(jit)) do xi, xj, i, j, s2, r2, p, eps
        distance = abs(xi - xj)
        sine = sin(pi * distance / p)
        cell = s2 * exp(-2 * sine * sine / r2)
        diagonal_jitter = ifelse(i == j, eps, 0.0)
        cell + diagonal_jitter
    end
    return covariance
end

const _GP_EXP_QUAD_COV = prepare(gp_exp_quad_cov_graph)
const _GP_PERIODIC_COV = prepare(gp_periodic_cov_graph)
