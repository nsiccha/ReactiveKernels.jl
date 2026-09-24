# Buffer-reusing `MutatingFunctions.apply!!` methods for the grouped-PK cell
# vocabulary (`pkcells.jl`): the subject-batched runners, the per-subject cells,
# and the event-LP provider. Each method reuses its cache only when the shape
# and eltype match the allocating result exactly, so value parity is preserved
# and a mismatch simply reseeds through the allocating twin. Traced operands
# take the same fallback (the twin routes to the compiled path or throws the
# documented Reactant-pending error); the `!` variants are native-only.
#
# Argument shape mirrors the generated grouped statements exactly (fixed arity,
# no slurped tuples, no intermediate collections): per-subject values forward
# straight into the `!` variants, and only 1-based `view`s of the bound op
# columns — which the compiler stack-allocates, as in `_cell_over_subjects` —
# cross the slice boundary. Output slicing is offset-based (`off`) so no view
# of the destination is ever constructed.
module ReactiveKernelsPPLMutatingFunctionsExt

using ReactiveKernelsPPL
import MutatingFunctions

const _PPL = ReactiveKernelsPPL

# --- subject-batched runners ----------------------------------------------

function MutatingFunctions.apply!!(cache::AbstractVector,
        f::typeof(_PPL.linear_pk_read_locs_auc_over_subjects),
        op_ends::AbstractVector{<:Integer},
        op_type::AbstractVector, op_dt::AbstractVector,
        op_amount::AbstractVector, op_interval::AbstractVector,
        op_count::AbstractVector, op_read_idx::AbstractVector,
        m1, m2, m3, m4, m5, m6)
    opcols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    mvals = map(_PPL._subject_value, (m1, m2, m3, m4, m5, m6))
    _PPL._pk_no_traced_marker(opcols, mvals) ||
        return f(op_ends, opcols..., m1, m2, m3, m4, m5, m6)
    n_sub = length(op_ends)
    n_sub >= 1 || return f(op_ends, opcols..., m1, m2, m3, m4, m5, m6)
    total = 0
    prev = 0
    for s in 1:n_sub
        hi = Int(op_ends[s])
        total += 2 * _PPL._pk_count_reads(op_type, prev + 1, hi)
        prev = hi
    end
    cache isa Vector{Float64} && length(cache) == total ||
        return f(op_ends, opcols..., m1, m2, m3, m4, m5, m6)
    off = 0
    prev = 0
    for s in 1:n_sub
        hi = Int(op_ends[s])
        rng = (prev + 1):hi
        n_reads = _PPL._pk_count_reads(op_type, prev + 1, hi)
        _PPL.linear_pk_read_locs_auc!(cache, off,
            view(op_type, rng), view(op_dt, rng), view(op_amount, rng),
            view(op_interval, rng), view(op_count, rng),
            view(op_read_idx, rng),
            _PPL._subject_arg(m1, rng, s), _PPL._subject_arg(m2, rng, s),
            _PPL._subject_arg(m3, rng, s), _PPL._subject_arg(m4, rng, s),
            _PPL._subject_arg(m5, rng, s), _PPL._subject_arg(m6, rng, s))
        off += 2 * n_reads
        prev = hi
    end
    return cache
end

function MutatingFunctions.apply!!(cache::AbstractVector,
        f::typeof(_PPL.linear_pk_read_locs_over_subjects),
        op_ends::AbstractVector{<:Integer},
        op_type::AbstractVector, op_dt::AbstractVector,
        op_amount::AbstractVector, op_interval::AbstractVector,
        op_count::AbstractVector, op_read_idx::AbstractVector,
        m1, m2, m3, m4, m5, m6)
    opcols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    mvals = map(_PPL._subject_value, (m1, m2, m3, m4, m5, m6))
    _PPL._pk_no_traced_marker(opcols, mvals) ||
        return f(op_ends, opcols..., m1, m2, m3, m4, m5, m6)
    n_sub = length(op_ends)
    n_sub >= 1 || return f(op_ends, opcols..., m1, m2, m3, m4, m5, m6)
    total = 0
    prev = 0
    for s in 1:n_sub
        hi = Int(op_ends[s])
        total += _PPL._pk_count_reads(op_type, prev + 1, hi)
        prev = hi
    end
    cache isa Vector{Float64} && length(cache) == total ||
        return f(op_ends, opcols..., m1, m2, m3, m4, m5, m6)
    off = 0
    prev = 0
    for s in 1:n_sub
        hi = Int(op_ends[s])
        rng = (prev + 1):hi
        n_reads = _PPL._pk_count_reads(op_type, prev + 1, hi)
        _PPL.linear_pk_read_locs!(cache, off,
            view(op_type, rng), view(op_dt, rng), view(op_amount, rng),
            view(op_interval, rng), view(op_count, rng),
            view(op_read_idx, rng),
            _PPL._subject_arg(m1, rng, s), _PPL._subject_arg(m2, rng, s),
            _PPL._subject_arg(m3, rng, s), _PPL._subject_arg(m4, rng, s),
            _PPL._subject_arg(m5, rng, s), _PPL._subject_arg(m6, rng, s))
        off += n_reads
        prev = hi
    end
    return cache
end

function MutatingFunctions.apply!!(cache::AbstractVector,
        f::typeof(_PPL.linear_pk_read_locs_over_subjects),
        op_ends::AbstractVector{<:Integer},
        op_type::AbstractVector, op_dt::AbstractVector,
        op_amount::AbstractVector, op_interval::AbstractVector,
        op_count::AbstractVector, op_read_idx::AbstractVector,
        m1, m2, m3, m4, m5)
    opcols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    mvals = map(_PPL._subject_value, (m1, m2, m3, m4, m5))
    _PPL._pk_no_traced_marker(opcols, mvals) ||
        return f(op_ends, opcols..., m1, m2, m3, m4, m5)
    n_sub = length(op_ends)
    n_sub >= 1 || return f(op_ends, opcols..., m1, m2, m3, m4, m5)
    total = 0
    prev = 0
    for s in 1:n_sub
        hi = Int(op_ends[s])
        total += _PPL._pk_count_reads(op_type, prev + 1, hi)
        prev = hi
    end
    cache isa Vector{Float64} && length(cache) == total ||
        return f(op_ends, opcols..., m1, m2, m3, m4, m5)
    off = 0
    prev = 0
    for s in 1:n_sub
        hi = Int(op_ends[s])
        rng = (prev + 1):hi
        n_reads = _PPL._pk_count_reads(op_type, prev + 1, hi)
        _PPL.linear_pk_read_locs!(cache, off,
            view(op_type, rng), view(op_dt, rng), view(op_amount, rng),
            view(op_interval, rng), view(op_count, rng),
            view(op_read_idx, rng),
            _PPL._subject_arg(m1, rng, s), _PPL._subject_arg(m2, rng, s),
            _PPL._subject_arg(m3, rng, s), _PPL._subject_arg(m4, rng, s),
            _PPL._subject_arg(m5, rng, s))
        off += n_reads
        prev = hi
    end
    return cache
end

# --- per-subject cells -----------------------------------------------------

function MutatingFunctions.apply!!(cache::AbstractVector,
        f::typeof(_PPL.linear_pk_read_locs_auc),
        op_type::AbstractVector, op_dt::AbstractVector,
        op_amount::AbstractVector, op_interval::AbstractVector,
        op_count::AbstractVector, op_read_idx::AbstractVector,
        log_F::AbstractVector, log_Vc, log_k10, log_k12, log_k21, log_ka)
    opcols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    _PPL._pk_no_traced_marker(opcols, (log_F,)) ||
        return f(opcols..., log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
    n_reads = _PPL._pk_count_reads(op_type, 1, length(op_type))
    cache isa Vector{Float64} && length(cache) == 2 * n_reads ||
        return f(opcols..., log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
    return _PPL.linear_pk_read_locs_auc!(cache, 0,
        opcols..., log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
end

function MutatingFunctions.apply!!(cache::AbstractVector,
        f::typeof(_PPL.linear_pk_read_locs),
        op_type::AbstractVector, op_dt::AbstractVector,
        op_amount::AbstractVector, op_interval::AbstractVector,
        op_count::AbstractVector, op_read_idx::AbstractVector,
        log_F::AbstractVector, log_Vc, log_k10, log_k12, log_k21, log_ka)
    opcols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    _PPL._pk_no_traced_marker(opcols, (log_F,)) ||
        return f(opcols..., log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
    n_reads = _PPL._pk_count_reads(op_type, 1, length(op_type))
    cache isa Vector{Float64} && length(cache) == n_reads ||
        return f(opcols..., log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
    return _PPL.linear_pk_read_locs!(cache, 0,
        opcols..., log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
end

function MutatingFunctions.apply!!(cache::AbstractVector,
        f::typeof(_PPL.linear_pk_read_locs),
        op_type::AbstractVector, op_dt::AbstractVector,
        op_amount::AbstractVector, op_interval::AbstractVector,
        op_count::AbstractVector, op_read_idx::AbstractVector,
        log_Vc, log_k10, log_k12, log_k21, log_ka)
    opcols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    _PPL._pk_no_traced_marker(opcols, ()) ||
        return f(opcols..., log_Vc, log_k10, log_k12, log_k21, log_ka)
    n_reads = _PPL._pk_count_reads(op_type, 1, length(op_type))
    cache isa Vector{Float64} && length(cache) == n_reads ||
        return f(opcols..., log_Vc, log_k10, log_k12, log_k21, log_ka)
    return _PPL.linear_pk_read_locs!(cache, 0,
        opcols..., log_Vc, log_k10, log_k12, log_k21, log_ka)
end

# --- event-LP provider -----------------------------------------------------

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
