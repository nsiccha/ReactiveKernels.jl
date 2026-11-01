# Native-coordinate marginal standard deviations, without forming L or L*L'.
_reference_sd(scale::AbstractVector) = abs.(scale)

function _reference_sd(scale::NamedTuple)
    d, k = length(scale.diagonal), length(scale.reflections)
    (d == 0 || k == 0) && return abs.(scale.diagonal)
    magnitude = maximum(abs, scale.diagonal)
    diagonal = scale.diagonal ./ magnitude
    # L/magnitude = Diagonal(diagonal) + U*V'. Each left reflection adds
    # one outer product. The portable scale applies its reflections in reverse.
    U, V = zeros(d, k), zeros(d, k)
    for (column, reflection) in enumerate(k:-1:1)
        u, v = view(U, :, column), view(V, :, column)
        u[scale.indices[reflection]] .= scale.reflections[reflection]
        v .= -2 .* diagonal .* u
        for previous in 1:column-1
            coefficient = -2dot(u, view(U, :, previous))
            v .+= coefficient .* view(V, :, previous)
        end
    end
    # A thin QR gives ||V*a|| without a cancellation-prone Gram matrix.
    # Separate the diagonal entry before taking the norm of each row of L.
    R = qr(V).R
    rotated, row = zeros(size(R, 1)), zeros(d)
    result = zeros(d)
    for i in 1:d
        coefficients = view(U, i, :)
        mul!(rotated, R, coefficients)
        perturbation_norm = norm(rotated)
        diagonal_change = dot(coefficients, view(V, i, :))
        if abs(diagonal_change) > perturbation_norm / sqrt(2)
            # Avoid subtracting nearly equal squared norms. Such a row has
            # leverage > 1/2 in span(V), so there are at most 2k of them.
            # Direct row norms therefore keep the whole computation O(d*k^2).
            mul!(row, V, coefficients)
            row[i] += diagonal[i]
            result[i] = magnitude * norm(row)
        else
            off_diagonal = sqrt((perturbation_norm - abs(diagonal_change)) *
                (perturbation_norm + abs(diagonal_change)))
            result[i] = magnitude * hypot(diagonal[i] + diagonal_change, off_diagonal)
        end
    end
    return result
end

function _cloud_sd(positions)
    d, n = size(positions)
    location = vec(sum(positions; dims=2)) ./ n
    variance = zeros(d)
    for j in axes(positions, 2), i in 1:d
        variance[i] += abs2(positions[i, j] - location[i]) / n
    end
    return sqrt.(variance)
end

_inflate_reference_scale(scale::AbstractVector, factor) = factor .* scale
_inflate_reference_scale(scale::NamedTuple, factor) =
    (; scale..., diagonal=factor .* scale.diagonal)
