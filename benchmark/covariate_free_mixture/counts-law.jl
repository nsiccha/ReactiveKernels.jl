using ReactiveKernels: @kernel, plate
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli, _product_mixture_fold
@kernel counts_product_mixture(observed::Vector{Bool}, logits::Matrix{Float64}, weights::Vector{Float64}) = begin
    rows_checked = (size(logits, 1) == 1 || size(logits, 1) == length(observed)) ?
        logits : throw(DimensionMismatch("logits need one row or one per observation"))
    checked_logits = size(rows_checked, 2) == length(weights) ?
        rows_checked : throw(DimensionMismatch("mixture weights must match components"))
    nrows = size(checked_logits, 1)
    successes = sum(observed)
    failures = length(observed) - successes
    components = plate(1:length(weights)) do k
        cells = plate(1:nrows) do i
            logit = checked_logits[i, k]
            value = if nrows == 1
                # Zero counts must not evaluate an infinite or NaN log mass.
                success_mass = successes == 0 ? 0.0 :
                    successes * bernoulli(; logit).logpdf(true)
                failure_mass = failures == 0 ? 0.0 :
                    failures * bernoulli(; logit).logpdf(false)
                success_mass + failure_mass
            else
                bernoulli(; logit).logpdf(observed[i])
            end
            value
        end
        sum(cells)
    end
    return _product_mixture_fold(components, weights)
end
