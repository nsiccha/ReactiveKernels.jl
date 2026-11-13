# Frozen law from f91621d6472664039d6b498247bc97b58a0931f6.
using ReactiveKernels: @kernel, plate
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli, _product_mixture_fold
@kernel baseline_product_mixture(observed::Vector{Bool},
        logits::Matrix{Float64}, weights::Vector{Float64}) = begin
    rows_checked = (size(logits, 1) == 1 || size(logits, 1) == length(observed)) ?
        logits : throw(DimensionMismatch("logits need one row or one per observation"))
    checked_logits = size(rows_checked, 2) == length(weights) ?
        rows_checked : throw(DimensionMismatch("mixture weights must match components"))
    nrows = size(checked_logits, 1)
    components = plate(1:length(weights)) do k
        cells = plate(1:length(observed)) do i
            row = ifelse(nrows == 1, 1, i)
            bernoulli(; logit = checked_logits[row, k]).logpdf(observed[i])
        end
        sum(cells)
    end
    return _product_mixture_fold(components, weights)
end

