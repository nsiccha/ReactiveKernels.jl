# Bind-time merge of measurements, doses and cumulative discretization lags.
# Keep equal-time dose boundaries: they produce zero-length PD steps in the
# reference transformed-data recipe.
function _vs_pkpd_times(measurements, doses, discretization)
    grid = Float64[]
    isempty(measurements) && return grid
    last_measurement = maximum(measurements)
    mi, di, qi = 1, 1, 1
    measurement = measurements[mi]
    last_dose = isempty(doses) ? 0.0 : doses[di]
    next_dose = isempty(doses) ? last_measurement : doses[di]
    step = last_dose + (isempty(discretization) ? last_measurement : discretization[qi])
    while mi <= length(measurements)
        time = min(measurement, next_dose, step)
        if time == measurement
            mi += 1
            measurement = mi > length(measurements) ? last_measurement : measurements[mi]
        end
        if time == next_dose
            last_dose = next_dose
            di += 1
            next_dose = di > length(doses) ? last_measurement : doses[di]
            qi = 1
            step = last_dose + (isempty(discretization) ? last_measurement : discretization[qi])
        elseif time == step
            qi += 1
            step = last_dose + (qi > length(discretization) ? last_measurement : discretization[qi])
            mi <= length(measurements) && step <= time &&
                measurement > time && next_dose > time && _pk_sched_fail(
                    "discretization lags do not cover the PD interval; the source grid would not advance")
        end
        push!(grid, time)
    end
    return grid
end

# The traversal consumes either subject-specific mathematical inputs or shared
# graph results. These local wrappers contain no cache or mutable state.
struct _VSSubjectEffectiveness{T}
    args::T
end
struct _VSSubjectPlacebo{T}
    args::T
end
struct _VSShared{T}
    value::T
end
struct _VSSharedEffectiveness{W,S}
    weights::W
    scalars::S
end
@inline function _vs_subject_effectiveness(p::_VSSharedEffectiveness, s)
    return (; min_dose=10000.0, max_dose=200000.0, max_conc=2000.0,
        dose_slope=p.scalars[1], conc_slope=p.scalars[2],
        weights=p.weights, normalizer=p.scalars[3])
end
@inline function _vs_subject_effectiveness(p::_VSSubjectEffectiveness, s)
    w, d, c, r_d, r_c, sd = p.args
    weights = varyingsource_gp_weights(w, _vs_subject_value(r_d,s),
        _vs_subject_value(r_c,s), _vs_subject_value(sd,s))
    return varyingsource_effectiveness(weights, _vs_subject_value(d,s), _vs_subject_value(c,s))
end
@inline _vs_subject_placebo(p::_VSShared, s, times, rows) = view(p.value, rows)
@inline function _vs_subject_placebo(p::_VSSubjectPlacebo, s, times, rows)
    w, r, sd, lo, hi = p.args
    return varyingsource_log_placebo(times, w, _vs_subject_value(r,s),
        _vs_subject_value(sd,s), _vs_subject_value(lo,s), _vs_subject_value(hi,s))
end

# Bound schedule facts and shared live mathematics are separate dependencies.
# Empty arms must not touch invalid or absent GP inputs.
_vs_pkpd_has_doses(schedule) = schedule.dose_ends[end] > 0
_vs_pkpd_placebo_times(schedule) = schedule.placebo_time
function _vs_pkpd_csf_times(schedule)
    times = Vector{Float64}(undef, length(schedule.pd3_dts))
    for s in eachindex(schedule.obs_ends)
        p2r = _vs_block(schedule.pd2_step_ends,s)
        p3r = _vs_block(schedule.pd3_step_ends,s)
        pr = _vs_block(schedule.placebo_ends,s)
        for (i,j) in enumerate(p3r)
            times[j] = schedule.placebo_time[first(pr) + length(p2r) + i - 1]
        end
    end
    return times
end
function _vs_gp_weights(has_doses, w, r_d, r_c, sd)
    has_doses || return nothing
    return varyingsource_gp_weights(w, r_d, r_c, sd)
end
function _vs_gp_normalizer(has_doses, weights, d, c)
    has_doses || return 0.0
    return -d - c + _varyingsource_gp(weights, -1.0, -1.0)
end

"""
    varyingsource_pkpd_read_locs_over_subjects(schedule, dose_log_rate,
        dose_log_mode, dose_log_F, gp_unit_weights, dose_slope, conc_slope,
        dose_scale, conc_scale, eff_scale, placebo_unit_weights,
        placebo_length_scale, placebo_sd, csf_unit_weights,
        csf_length_scale, csf_sd, placebo_lo, placebo_hi,
        log_Vc, log_k10, log_k12, log_k21, log_baseline_pbmc, log_kout,
        log_theta1_pbmc, log_theta2_pbmc, log_baseline_csf, log_theta1_csf,
        log_theta2_csf, log_absorption_rate, log_absorption_mode)

Native full PK/PD cell over a bound raw-column schedule. Output is in flat
subject-grouped observation order; gather with `schedule.obs_map` for caller
order. GP vectors contain unscaled standard-normal innovations. GP scales
and placebo scales are positive constrained values. The three dose modifiers
use original dose-row order, shared scalars, or `SubjectScalar`; scalar cell
parameters may likewise be shared or subject-specific.

Subject, treatment, dose, lag, observation and basis traversal are ordinary
runtime loops. Dose-free subjects skip PK/GP parameter reads. Missing PD
assays skip their parameters and placebo course, and initial-state-only PD
reads only its baseline. Priors and likelihood belong to the model.
"""
@inline function varyingsource_pkpd_read_locs_over_subjects(schedule, dose_log_rate,
        dose_log_mode, dose_log_F, gp_unit_weights, dose_slope, conc_slope,
        dose_scale, conc_scale, eff_scale, placebo_unit_weights,
        placebo_length_scale, placebo_sd, csf_unit_weights,
        csf_length_scale, csf_sd, placebo_lo, placebo_hi,
        log_Vc, log_k10, log_k12, log_k21, log_baseline_pbmc, log_kout,
        log_theta1_pbmc, log_theta2_pbmc, log_baseline_csf, log_theta1_csf,
        log_theta2_csf, log_absorption_rate, log_absorption_mode)
    _vs_native_math((schedule,))
    _vs_native_math(map(_subject_value,(dose_log_rate,dose_log_mode,dose_log_F)))
    _vs_native_math((gp_unit_weights,placebo_unit_weights,csf_unit_weights))
    _vs_native_math(map(_subject_value,(dose_slope,conc_slope,dose_scale,conc_scale,
        eff_scale,placebo_length_scale,placebo_sd,csf_length_scale,csf_sd,
        placebo_lo,placebo_hi)))
    _vs_native_math(map(_subject_value,(log_Vc,log_k10,log_k12,log_k21,
        log_baseline_pbmc,log_kout,log_theta1_pbmc,log_theta2_pbmc,
        log_baseline_csf,log_theta1_csf,log_theta2_csf,
        log_absorption_rate,log_absorption_mode)))
    return _vs_pkpd_read_math(schedule, dose_log_rate, dose_log_mode, dose_log_F,
        _VSSubjectEffectiveness((gp_unit_weights,dose_slope,conc_slope,
            dose_scale,conc_scale,eff_scale)),
        _VSSubjectPlacebo((placebo_unit_weights,placebo_length_scale,placebo_sd,placebo_lo,placebo_hi)),
        _VSSubjectPlacebo((csf_unit_weights,csf_length_scale,csf_sd,placebo_lo,placebo_hi)),
        log_Vc,log_k10,log_k12,log_k21,log_baseline_pbmc,log_kout,
        log_theta1_pbmc,log_theta2_pbmc,log_baseline_csf,log_theta1_csf,
        log_theta2_csf,log_absorption_rate,log_absorption_mode)
end

@inline function _vs_pkpd_read_math(schedule, dose_log_rate, dose_log_mode, dose_log_F,
        effectiveness, primary_placebo, secondary_placebo,
        log_Vc,log_k10,log_k12,log_k21,log_baseline_pbmc,log_kout,
        log_theta1_pbmc,log_theta2_pbmc,log_baseline_csf,log_theta1_csf,
        log_theta2_csf,log_absorption_rate,log_absorption_mode)
    n = length(schedule.obs_ends)
    n > 0 || throw(ArgumentError("varyingsource PK/PD needs at least one subject"))
    out = zeros(Float64, schedule.obs_ends[end])
    for s in 1:n
        rr, dr = _vs_block(schedule.reference_ends,s), _vs_block(schedule.dose_ends,s)
        or = _vs_block(schedule.obs_ends,s)
        concentration = zeros(Float64,length(rr))
        if !isempty(dr)
            rows = view(schedule.dose_index,dr)
            effect = _vs_subject_effectiveness(effectiveness,s)
            pk = _VARYINGSOURCE_PK_CELL(length(rr),view(schedule.dose_amount,dr),
                view(schedule.treatment_map,dr),view(schedule.unique_dts,_vs_block(schedule.lag_ends,s)),
                view(schedule.concentration_idxs,_vs_block(schedule.concentration_ends,s)),
                view(schedule.dosing_time_idxs,dr),
                _vs_dose_value(dose_log_rate,rows,s), _vs_dose_value(dose_log_mode,rows,s),
                _vs_dose_value(dose_log_F,rows,s), effect,
                _vs_subject_value(log_Vc,s), _vs_subject_value(log_k10,s),
                _vs_subject_value(log_k12,s), _vs_subject_value(log_k21,s),
                _vs_subject_value(log_absorption_rate,s), _vs_subject_value(log_absorption_mode,s))
            for i in eachindex(concentration)
                concentration[i] = pk[i]
            end
        end
        local_assay, values = view(schedule.assay,or), view(out,or)
        pkr = _vs_block(schedule.pk_ends,s)
        if !isempty(pkr)
            _vs_write_assay!(values,local_assay,1,concentration[view(schedule.pk_idxs,pkr)])
        end
        placebo_range = _vs_block(schedule.placebo_ends,s)
        p2r, p3r = _vs_block(schedule.pd2_step_ends,s), _vs_block(schedule.pd3_step_ends,s)
        w2r, w3r = _vs_block(schedule.pd2_write_ends,s), _vs_block(schedule.pd3_write_ends,s)
        if !isempty(w2r)
            placebo = zeros(Float64,length(p2r))
            if !isempty(p2r)
                rows = first(placebo_range):(first(placebo_range)+length(p2r)-1)
                times = view(schedule.placebo_time,rows)
                course = _vs_subject_placebo(primary_placebo,s,times,rows)
                for i in eachindex(placebo)
                    placebo[i] = course[i]
                end
            end
            pbmc = varyingsource_pd_locs(view(schedule.pd2_idxs,w2r),view(schedule.pd2_dts,p2r),
                concentration[view(schedule.pd2_center_idxs,p2r)],placebo,
                _vs_subject_value(log_baseline_pbmc,s),
                isempty(p2r) ? NaN : _vs_subject_value(log_kout,s),
                isempty(p2r) ? NaN : _vs_subject_value(log_theta1_pbmc,s),
                isempty(p2r) ? NaN : _vs_subject_value(log_theta2_pbmc,s))
            _vs_write_assay!(values,local_assay,2,pbmc)
        end
        if !isempty(w3r)
            placebo = zeros(Float64,length(p3r))
            if !isempty(p3r)
                rows = (first(placebo_range)+length(p2r)):last(placebo_range)
                times = view(schedule.placebo_time,rows)
                primary = _vs_subject_placebo(primary_placebo,s,times,rows)
                secondary = _vs_subject_placebo(secondary_placebo,s,times,p3r)
                for i in eachindex(placebo)
                    placebo[i] = primary[i] + secondary[i]
                end
            end
            csf = varyingsource_pd_locs(view(schedule.pd3_idxs,w3r),view(schedule.pd3_dts,p3r),
                concentration[view(schedule.pd3_center_idxs,p3r)],placebo,
                _vs_subject_value(log_baseline_csf,s),
                isempty(p3r) ? NaN : _vs_subject_value(log_kout,s),
                isempty(p3r) ? NaN : _vs_subject_value(log_theta1_csf,s),
                isempty(p3r) ? NaN : _vs_subject_value(log_theta2_csf,s))
            _vs_write_assay!(values,local_assay,3,csf)
        end
    end
    return out
end

# This emitted boundary receives ordinary mathematical results, so RK can fold
# bound producers or share live producers without inspecting the traversal.
@inline function varyingsource_pkpd_read_locs_over_subjects(schedule,
        dose_log_rate,dose_log_mode,dose_log_F,logs::AbstractMatrix,
        weights::Union{Nothing,AbstractMatrix},scalars::AbstractVector,
        primary::AbstractVector,secondary::AbstractVector)
    size(logs) == (length(schedule.obs_ends),13) && length(scalars) == 3 ||
        throw(DimensionMismatch("full PK/PD subject parameter dimensions disagree"))
    _vs_native_math((schedule,logs,weights,scalars,primary,secondary))
    _vs_native_math(map(_subject_value,(dose_log_rate,dose_log_mode,dose_log_F)))
    lp(i) = SubjectScalar(view(logs,:,i))
    return _vs_pkpd_read_math(schedule,dose_log_rate,dose_log_mode,dose_log_F,
        _VSSharedEffectiveness(weights,scalars),_VSShared(primary),_VSShared(secondary),
        lp(1),lp(2),lp(3),lp(4),lp(5),lp(6),lp(7),lp(8),lp(9),lp(10),lp(11),lp(12),lp(13))
end

# Raw compact spelling retained for callers supplying shared hyperparameters.
# Emission supplies graph-computed mathematics through the overload above.
@inline function varyingsource_pkpd_read_locs_over_subjects(schedule,
        dose_log_rate,dose_log_mode,dose_log_F,logs::AbstractMatrix,
        hyper::AbstractVector,gp_unit_weights,placebo_unit_weights,csf_unit_weights)
    size(logs) == (length(schedule.obs_ends),13) && length(hyper) == 11 ||
        throw(DimensionMismatch("full PK/PD subject/scalar parameter dimensions disagree"))
    lp(i) = SubjectScalar(view(logs,:,i))
    return varyingsource_pkpd_read_locs_over_subjects(schedule,
        dose_log_rate,dose_log_mode,dose_log_F,gp_unit_weights,
        hyper[1],hyper[2],hyper[3],hyper[4],hyper[5],placebo_unit_weights,
        hyper[6],hyper[7],csf_unit_weights,hyper[8],hyper[9],hyper[10],hyper[11],
        lp(1),lp(2),lp(3),lp(4),lp(5),lp(6),lp(7),lp(8),lp(9),lp(10),lp(11),lp(12),lp(13))
end

_vs_pkpd_subject_column(n,x::Number) = fill(x,n)
function _vs_pkpd_subject_column(n,x::AbstractVector)
    length(x) == n || throw(DimensionMismatch("full PK/PD subject LP column length differs"))
    x
end
@inline function _varyingsource_pkpd_subject_columns(n,args::Vararg{Any,13})
    hcat(map(x -> _vs_pkpd_subject_column(n,x),args)...)
end

# This data-only call is an ordinary graph operation. Preparation can bind
# its result without fusing schedule construction into active cell math.
function _varyingsource_pkpd_schedule_columns(reference_ends,dose_ends,
        lag_ends,concentration_ends,dose_amount,dose_index,treatment_map,
        unique_dts,concentration_idxs,dosing_time_idxs,obs_ends,assay,obs_map,
        pk_ends,pk_idxs,pd2_write_ends,pd2_step_ends,pd2_idxs,pd2_dts,
        pd2_center_idxs,pd3_write_ends,pd3_step_ends,pd3_idxs,pd3_dts,
        pd3_center_idxs,placebo_ends,placebo_time)
    return (;reference_ends,dose_ends,lag_ends,concentration_ends,
        dose_amount,dose_index,treatment_map,unique_dts,concentration_idxs,
        dosing_time_idxs,obs_ends,assay,obs_map,pk_ends,pk_idxs,
        pd2_write_ends,pd2_step_ends,pd2_idxs,pd2_dts,pd2_center_idxs,
        pd3_write_ends,pd3_step_ends,pd3_idxs,pd3_dts,pd3_center_idxs,
        placebo_ends,placebo_time)
end

"""
    build_varyingsource_pkpd_schedule(obs_subj, obs_time, obs_assay,
        dose_subj, dose_time, dose_amt, treatment, discretization_times)

Bind the full varying-source PK/PBMC/CSF grids from raw columns. Assay codes
are 1 (PK), 2 (PBMC), and 3 (CSF). Observation subjects cover 1:n; doses must
be nondecreasing in time within each subject. PD measurement times must
strictly increase within each subject and assay. PK observations may repeat
and all observations retain their caller order through `obs_map`.

`discretization_times` contains nonnegative cumulative lags from each dose,
in nondecreasing order. `treatment` contains combined vessel/diet keys, which
are renumbered by first appearance within each subject. Equal-time doses
remain distinct. The PK grid is the sorted unique union of doses, PK reads,
and both PD midpoint grids. A dose-free subject measured only at its initial
PD state may have an empty PK grid; no reference point is inserted.

All indices are local to their subject's block except `dose_index` (original
dose rows) and `obs_map` (flat, subject-grouped observation positions). Each
independent ragged axis has cumulative ends. Placebo times concatenate PBMC
and CSF midpoints within each subject. This is a data recipe, not a parameter
transform, and must run again when the raw data are rebound.
"""
function build_varyingsource_pkpd_schedule(obs_subj::AbstractVector,
        obs_time::AbstractVector, obs_assay::AbstractVector,
        dose_subj::AbstractVector, dose_time::AbstractVector,
        dose_amt::AbstractVector, treatment::AbstractVector,
        discretization_times::AbstractVector)
    length(obs_subj) == length(obs_time) == length(obs_assay) ||
        _pk_sched_fail("PK/PD observation column lengths differ")
    length(dose_subj) == length(dose_time) == length(dose_amt) == length(treatment) ||
        _pk_sched_fail("varyingsource dose column lengths differ")
    isempty(obs_subj) && _pk_sched_fail("at least one PK/PD observation is required")
    for (label, ids) in (("observation", obs_subj), ("dose", dose_subj))
        all(x -> x isa Real && isfinite(x) && isinteger(x) && x > 0, ids) ||
            _pk_sched_fail("$label subject IDs must be positive integers")
    end
    for (label, times) in (("observation", obs_time), ("dose", dose_time))
        all(x -> x isa Real && isfinite(x), times) ||
            _pk_sched_fail("$label times must be finite")
    end
    all(x -> x isa Real && isfinite(x) && isinteger(x) && 1 <= x <= 3, obs_assay) ||
        _pk_sched_fail("PK/PD assay codes must be integers in 1:3")
    all(x -> x isa Real && isfinite(x) && x > 0, dose_amt) ||
        _pk_sched_fail("varyingsource dose amounts must be finite and positive")
    all(x -> x isa Real && isfinite(x) && isinteger(x), treatment) ||
        _pk_sched_fail("treatment keys must be finite integers")
    all(x -> x isa Real && isfinite(x) && x >= 0, discretization_times) &&
        issorted(discretization_times) ||
        _pk_sched_fail("discretization lags must be finite, nonnegative and nondecreasing")
    subjects, dsubjects = Int.(obs_subj), Int.(dose_subj)
    times, assays = Float64.(obs_time), Int.(obs_assay)
    dtimes, amounts = Float64.(dose_time), Float64.(dose_amt)
    discretization = Float64.(discretization_times)
    n_subjects = maximum(subjects)
    sort(unique(subjects)) == collect(1:n_subjects) ||
        _pk_sched_fail("observation subject IDs must be contiguous 1:$n_subjects")
    all(<=(n_subjects), dsubjects) ||
        _pk_sched_fail("doses must refer to observed subjects 1:$n_subjects")

    reference_ends, dose_ends, lag_ends, concentration_ends = Int[], Int[], Int[], Int[]
    dose_amount, unique_dts = Float64[], Float64[]
    dose_index, treatment_map, concentration_idxs, dosing_time_idxs = Int[], Int[], Int[], Int[]
    obs_ends, assay, pk_ends, pk_idxs = Int[], Int[], Int[], Int[]
    pd2_write_ends, pd2_step_ends, pd2_idxs, pd2_center_idxs = Int[], Int[], Int[], Int[]
    pd3_write_ends, pd3_step_ends, pd3_idxs, pd3_center_idxs = Int[], Int[], Int[], Int[]
    pd2_dts, pd3_dts, placebo_time = Float64[], Float64[], Float64[]
    placebo_ends, n_reads = Int[], Int[]
    obs_map = Vector{Int}(undef, length(subjects))
    n_reference = 0
    for s in 1:n_subjects
        orows, drows = findall(==(s), subjects), findall(==(s), dsubjects)
        ds = dtimes[drows]
        issorted(ds) || _pk_sched_fail("dose times must be nondecreasing within subject $s")
        local_assay, local_time = assays[orows], times[orows]
        pk, p2, p3 = (local_time[local_assay .== a] for a in 1:3)
        for (code, ts) in ((2, p2), (3, p3))
            all(>(0), diff(ts)) || _pk_sched_fail(
                "PD observation times must strictly increase within subject $s, assay $code")
        end
        grid2, grid3 = _vs_pkpd_times(p2, ds, discretization), _vs_pkpd_times(p3, ds, discretization)
        mids2 = 0.5 .* (grid2[2:end] .+ grid2[1:end-1])
        mids3 = 0.5 .* (grid3[2:end] .+ grid3[1:end-1])
        refs = sort(unique(vcat(ds, pk, mids2, mids3)))
        lags = Float64[0.0]
        for d in ds, t in refs
            t >= d && push!(lags, t - d)
        end
        sort!(unique!(lags))
        keys = unique(treatment[drows])
        for (i, row) in enumerate(orows)
            obs_map[row] = length(assay) + i
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
        append!(assay, local_assay)
        append!(pk_idxs, searchsortedfirst.(Ref(refs), pk))
        append!(pd2_idxs, searchsortedfirst.(Ref(grid2), p2))
        append!(pd3_idxs, searchsortedfirst.(Ref(grid3), p3))
        append!(pd2_dts, diff(grid2))
        append!(pd3_dts, diff(grid3))
        append!(pd2_center_idxs, searchsortedfirst.(Ref(refs), mids2))
        append!(pd3_center_idxs, searchsortedfirst.(Ref(refs), mids3))
        append!(placebo_time, mids2)
        append!(placebo_time, mids3)
        n_reference += length(refs)
        push!(reference_ends, n_reference)
        push!(dose_ends, length(dose_amount))
        push!(lag_ends, length(unique_dts))
        push!(concentration_ends, length(concentration_idxs))
        push!(obs_ends, length(assay))
        push!(pk_ends, length(pk_idxs))
        push!(pd2_write_ends, length(pd2_idxs))
        push!(pd3_write_ends, length(pd3_idxs))
        push!(pd2_step_ends, length(pd2_dts))
        push!(pd3_step_ends, length(pd3_dts))
        push!(placebo_ends, length(placebo_time))
        push!(n_reads, length(orows))
    end
    return (; reference_ends, dose_ends, lag_ends, concentration_ends,
        dose_amount, dose_index, treatment_map, unique_dts, concentration_idxs,
        dosing_time_idxs, obs_ends, assay, obs_map, pk_ends, pk_idxs,
        pd2_write_ends, pd2_step_ends, pd2_idxs, pd2_dts, pd2_center_idxs,
        pd3_write_ends, pd3_step_ends, pd3_idxs, pd3_dts, pd3_center_idxs,
        placebo_ends, placebo_time, n_subjects, n_reads, n_reads_total=length(assay))
end
