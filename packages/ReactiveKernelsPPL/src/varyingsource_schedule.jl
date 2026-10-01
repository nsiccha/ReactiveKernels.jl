"""
    build_varyingsource_pk_schedule(obs_subj, obs_time, dose_subj, dose_time,
                                   dose_amt, treatment)

Bind-time recipe for the varying-source PK slice. Observation subjects cover
1:n; doses refer to those subjects and remain in caller order, which must be
nondecreasing in time within each subject. Equal-time doses remain separate.
`treatment` supplies the combined treatment/diet key, renumbered in first
appearance order within each subject, as in the reference model.

Reference times are the sorted unique union of PK observations and dose times.
Each subject's lag grid includes zero and every nonnegative dose-to-reference
difference. All indices inside a subject block are local, except `dose_index`
(original dose rows) and `obs_map` (flat reference positions). Independent
cumulative ends describe the reference, dose, lag, and dose-major lag-index
axes. Duplicate observations and caller observation order survive in `obs_map`.
A subject with no doses retains its observation grid and yields zero PK.
"""
function build_varyingsource_pk_schedule(obs_subj::AbstractVector,
        obs_time::AbstractVector, dose_subj::AbstractVector,
        dose_time::AbstractVector, dose_amt::AbstractVector,
        treatment::AbstractVector)
    length(obs_subj) == length(obs_time) ||
        _pk_sched_fail("observation subject/time lengths differ")
    length(dose_subj) == length(dose_time) == length(dose_amt) == length(treatment) ||
        _pk_sched_fail("varyingsource dose column lengths differ")
    isempty(obs_subj) && _pk_sched_fail("at least one PK observation is required")
    for (label, ids) in (("observation", obs_subj), ("dose", dose_subj))
        all(x -> x isa Real && isfinite(x) && isinteger(x) && x > 0, ids) ||
            _pk_sched_fail("$label subject IDs must be positive integers")
    end
    all(x -> x isa Real && isfinite(x), obs_time) ||
        _pk_sched_fail("PK observation times must be finite")
    all(x -> x isa Real && isfinite(x), dose_time) ||
        _pk_sched_fail("PK dose times must be finite")
    all(x -> x isa Real && isfinite(x) && x > 0, dose_amt) ||
        _pk_sched_fail("varyingsource dose amounts must be finite and positive")
    all(x -> x isa Real && isfinite(x) && isinteger(x), treatment) ||
        _pk_sched_fail("treatment keys must be finite integers")
    subjects, dsubjects = Int.(obs_subj), Int.(dose_subj)
    times, dtimes, amounts = Float64.(obs_time), Float64.(dose_time), Float64.(dose_amt)
    n_subjects = maximum(subjects)
    sort(unique(subjects)) == collect(1:n_subjects) ||
        _pk_sched_fail("PK observation subject IDs must be contiguous 1:$n_subjects")
    all(<=(n_subjects), dsubjects) ||
        _pk_sched_fail("doses must refer to observed subjects 1:$n_subjects")
    reference_ends, dose_ends, lag_ends, concentration_ends = Int[], Int[], Int[], Int[]
    dose_amount, unique_dts = Float64[], Float64[]
    dose_index, treatment_map, concentration_idxs, dosing_time_idxs = Int[], Int[], Int[], Int[]
    obs_map = Vector{Int}(undef, length(subjects))
    n_reads = Int[]
    n_reference = 0
    for s in 1:n_subjects
        orows = findall(==(s), subjects)
        drows = findall(==(s), dsubjects)
        ds = dtimes[drows]
        issorted(ds) || _pk_sched_fail("dose times must be nondecreasing within subject $s")
        refs = sort(unique(vcat(times[orows], ds)))
        lags = Float64[0.0]
        for d in ds, t in refs
            t >= d && push!(lags, t - d)
        end
        sort!(unique!(lags))
        keys = unique(treatment[drows])
        for row in orows
            obs_map[row] = n_reference + searchsortedfirst(refs, times[row])
        end
        for row in drows
            push!(dose_index, row)
            push!(dose_amount, amounts[row])
            push!(treatment_map, findfirst(==(treatment[row]), keys))
            push!(dosing_time_idxs, searchsortedfirst(refs, dtimes[row]))
            for t in refs
                push!(concentration_idxs, searchsortedfirst(lags, max(0.0, t - dtimes[row])))
            end
        end
        append!(unique_dts, lags)
        n_reference += length(refs)
        push!(reference_ends, n_reference)
        push!(dose_ends, length(dose_amount))
        push!(lag_ends, length(unique_dts))
        push!(concentration_ends, length(concentration_idxs))
        push!(n_reads, length(refs))
    end
    return (; reference_ends, dose_ends, lag_ends, concentration_ends,
        dose_amount, dose_index, treatment_map, unique_dts, concentration_idxs,
        dosing_time_idxs, obs_map, n_subjects, n_reads, n_reads_total = n_reference)
end

@inline _vs_subject_value(a::SubjectScalar, s) = a.v[s]
@inline _vs_subject_value(a, s) = a
@inline _vs_dose_value(a::SubjectScalar, rows, s) = fill(a.v[s], length(rows))
@inline _vs_dose_value(a::AbstractVector, rows, s) = view(a, rows)
@inline _vs_dose_value(a::Number, rows, s) = fill(a, length(rows))
@inline _vs_block(ends, s) = (s == 1 ? 1 : ends[s - 1] + 1):ends[s]
function _vs_weight_matrix(a::AbstractMatrix)
    isempty(a) && throw(DimensionMismatch("varyingsource GP weights must be nonempty"))
    return a
end
function _vs_weight_matrix(a::AbstractVector)
    k = isqrt(length(a))
    k > 0 && k * k == length(a) ||
        throw(DimensionMismatch("varyingsource GP weights need a nonempty square vector"))
    return reshape(a, k, k)
end

"""
    varyingsource_pk_read_locs_over_subjects(reference_ends, dose_ends, lag_ends,
        concentration_ends, dose_amount, dose_index, treatment_map, unique_dts,
        concentration_idxs, dosing_time_idxs, dose_log_rate, dose_log_mode,
        dose_log_F, weights, dose_slope, conc_slope, log_Vc, log_k10, log_k12,
        log_k21, log_absorption_rate, log_absorption_mode)

Native subject-batched varying-source PK cell used by the grouped emitter.
The first ten vectors come from [`build_varyingsource_pk_schedule`](@ref).
Dose modifiers are vectors in original dose-row order, shared scalars, or
`SubjectScalar` subject predictors. The six log parameters and two GP slopes
are shared scalars or `SubjectScalar` predictors. GP coefficients are a shared
matrix or a square vector in Julia column order, already scaled by the model.
Gather the flat reference concentrations with the schedule's `obs_map`.

Subject, treatment, lag, dose, and reference traversal remain runtime loops.
Dose-free subjects skip all parameter and GP evaluation. Numerical controls
are those of `prepare_varyingsource_pk()` (series_rtol=1e-15, watson_terms=8);
they do not establish a production accuracy policy. Reactant is unsupported.
"""
function varyingsource_pk_read_locs_over_subjects(reference_ends, dose_ends,
        lag_ends, concentration_ends, dose_amount, dose_index, treatment_map,
        unique_dts, concentration_idxs, dosing_time_idxs, dose_log_rate,
        dose_log_mode, dose_log_F, weights, dose_slope, conc_slope, log_Vc,
        log_k10, log_k12, log_k21, log_absorption_rate, log_absorption_mode)
    marker = ReactiveKernels._dynamic_tensorized_marker(map(_subject_value,
        (dose_log_rate, dose_log_mode, dose_log_F, weights, dose_slope, conc_slope,
            log_Vc, log_k10, log_k12, log_k21, log_absorption_rate, log_absorption_mode)))
    marker === nothing || throw(ArgumentError("varyingsource PK grouped cells support native execution only"))
    n = length(reference_ends)
    n > 0 || throw(ArgumentError("varyingsource PK needs at least one subject"))
    length(dose_ends) == length(lag_ends) == length(concentration_ends) == n ||
        throw(DimensionMismatch("varyingsource schedule ends disagree"))
    out = zeros(Float64, reference_ends[end])
    for s in 1:n
        rr = _vs_block(reference_ends, s)
        dr = _vs_block(dose_ends, s)
        # This branch must stay lazy: no GP reshape, subject indexing, or
        # parameter reads are needed on the empty-dose arm.
        isempty(dr) && continue
        lr, cr = _vs_block(lag_ends, s), _vs_block(concentration_ends, s)
        rows = view(dose_index, dr)
        coefficients = _vs_weight_matrix(weights)
        ds, cs = _vs_subject_value(dose_slope, s), _vs_subject_value(conc_slope, s)
        normalizer = -ds - cs + _varyingsource_gp(coefficients, -1.0, -1.0)
        concentration = _varyingsource_pk_concentration(_VARYINGSOURCE_PK_CELL, length(rr),
            view(dose_amount, dr), view(treatment_map, dr), view(unique_dts, lr),
            view(concentration_idxs, cr), view(dosing_time_idxs, dr),
            _vs_dose_value(dose_log_rate, rows, s), _vs_dose_value(dose_log_mode, rows, s),
            _vs_dose_value(dose_log_F, rows, s), coefficients, ds, cs,
            normalizer, 10000.0, 200000.0, 2000.0,
            _vs_subject_value(log_Vc, s), _vs_subject_value(log_k10, s),
            _vs_subject_value(log_k12, s), _vs_subject_value(log_k21, s),
            _vs_subject_value(log_absorption_rate, s), _vs_subject_value(log_absorption_mode, s))
        for (i, row) in enumerate(rr)
            out[row] = concentration[i]
        end
    end
    return out
end
