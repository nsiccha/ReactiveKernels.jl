export dirichlet_multinomial, dirichlet_multinomial_logdensity

# These are the ordinary constructor's checks, including its acceptance of
# zero concentrations. The density below additionally enforces count support.
_dm_concentrations(alpha::AbstractVector{<:Real}) = alpha
_dm_concentrations(k::Integer) = ones(k)

@kernel _dm_count_coefficient(counts, n::Int) = begin
    normalizers = plate(counts) do y
        isinteger(y) && y >= 0 ? loggamma(Float64(y) + 1.0) : Inf
    end
    coefficient::Float64 = sum(counts) == n ?
        loggamma(Float64(n) + 1.0) - sum(normalizers) : -Inf
    return coefficient
end

@kernel _dm_supported_density(counts, checked_alpha, alpha_loggamma,
        shared::Float64, coefficient::Float64) = begin
    cells = plate(counts, checked_alpha, alpha_loggamma) do y, a, lg
        loggamma(Float64(y) + a) - lg
    end
    return coefficient + shared + sum(cells)
end

@kernel dirichlet_multinomial(n::Int, alpha) = begin
    concentrations = _dm_concentrations(alpha)
    alpha_sum::Float64 = sum(abs, concentrations)
    checked_alpha = sum(concentrations) == alpha_sum ? concentrations :
        throw(DomainError(concentrations, "DirichletMultinomial: alpha must be a positive vector."))
    checked_n::Int = n > 0 ? n :
        throw(DomainError(n, "DirichletMultinomial: n must be a positive integer."))
    # Retaining this vector separates shared concentration terms from the
    # count-dependent inner plate when many events use the same alpha.
    alpha_loggamma = loggamma.(checked_alpha)
    shared::Float64 = loggamma(alpha_sum) - loggamma(checked_n + alpha_sum)

    logpdf(observed)::Float64 = begin
        # Count-only work must not acquire a dependency on alpha through a
        # dimension-checking alias of observed. Keep the check as its own gate.
        coefficient::Float64 = _dm_count_coefficient(observed, checked_n)
        valid::Bool = length(observed) == length(checked_alpha) ? coefficient != -Inf :
            throw(DimensionMismatch("counts and concentrations must have equal lengths"))
        if valid
            _dm_supported_density(observed, checked_alpha, alpha_loggamma, shared, coefficient)
        else
            -Inf
        end
    end
end

const dirichlet_multinomial_logdensity = extract(dirichlet_multinomial;
    have=(:observed, :n, :alpha), want=:logpdf)
