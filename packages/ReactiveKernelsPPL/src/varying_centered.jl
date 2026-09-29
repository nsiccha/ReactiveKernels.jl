# Native centered correlated draws. The factor entries are row-major lower
# triangular; their construction is structural, while groups/margins are
# traversed by ordinary runtime loops in this density.
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
    value = -groups*(0.5k*log(2pi)+logdet)
    z = zeros(Float64,k)
    for g in 1:groups
        for i in 1:k
            row = i*(i-1)÷2
            residual = draws[(g-1)*k+i]/scales[i]
            for j in 1:(i-1)
                residual -= lower[row+j]*z[j]
            end
            z[i] = residual/lower[row+i]
            value -= 0.5z[i]^2
        end
    end
    return value
end
