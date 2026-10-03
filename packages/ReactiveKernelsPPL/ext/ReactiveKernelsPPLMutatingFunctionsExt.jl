# Schedule-defined PK output packing and the legacy event-LP destination
# helper. The PK equations execute through RK's ordinary authored plate/scan
# caches; this extension owns no subject or event recurrence.
module ReactiveKernelsPPLMutatingFunctionsExt

using ReactiveKernelsPPL
import MutatingFunctions

const _PPL = ReactiveKernelsPPL

function MutatingFunctions.apply!!(cache::AbstractVector,
        ::typeof(_PPL._pk_pack_subjects), lanes, packing)
    length(cache) == length(packing) || resize!(cache, length(packing))
    isempty(packing) && return cache
    capacity = length(first(lanes))
    for p in eachindex(packing)
        subject, offset = divrem(packing[p] - 1, capacity)
        cache[p] = lanes[subject + 1][offset + 1]
    end
    cache
end

function MutatingFunctions.apply!!(cache::AbstractVector,
        f::typeof(_PPL.linear_pk_event_log_f),
        op_log_dose::AbstractVector, slope, rho,
        sigma, beta_raw::AbstractVector, mu, L, k::Integer)
    _PPL._pk_no_traced_marker((op_log_dose, beta_raw,),
        (slope, rho, sigma, mu, L)) ||
        return f(op_log_dose, slope, rho, sigma, beta_raw, mu, L, k)
    n = length(op_log_dose)
    cache isa Vector{Float64} && length(cache) == n &&
        length(beta_raw) == k && L > 0 ||
        return f(op_log_dose, slope, rho, sigma, beta_raw, mu, L, k)
    return _PPL.linear_pk_event_log_f!(cache,
        op_log_dose, slope, rho, sigma, beta_raw, mu, L, k)
end

end # module ReactiveKernelsPPLMutatingFunctionsExt
