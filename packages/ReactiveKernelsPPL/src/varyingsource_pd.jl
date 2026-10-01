# Native math for the varying-source PK/PD model.
# Basis, observation, and PD-grid sizes are data: retain ordinary loops.

function _vs_native_math(args)
    ReactiveKernels._dynamic_tensorized_marker(args) === nothing ||
        throw(ArgumentError("varyingsource PK/PD math supports native execution only"))
    return nothing
end

"""
    varyingsource_gp_weights(unit_weights, dose_scale, conc_scale, eff_scale)

Scale the model's square, column-major GP innovation vector into the matrix
consumed by varyingsource_effectiveness. Length scales and amplitude are
positive constrained values. This is the GP transform, not its prior density.
All basis traversal remains runtime iteration.
"""
function varyingsource_gp_weights(unit_weights::AbstractVector,
        dose_scale, conc_scale, eff_scale)
    k = isqrt(length(unit_weights))
    k >= 2 && k * k == length(unit_weights) || throw(ArgumentError(
        "varyingsource GP innovations need a square vector with at least 2x2 entries"))
    _vs_native_math((unit_weights, dose_scale, conc_scale, eff_scale))
    dose_scale > 0 && conc_scale > 0 && eff_scale > 0 || throw(ArgumentError(
        "varyingsource GP length scales and amplitude must be positive"))
    common = log(eff_scale) + 2 * 0.45946926660233633 +
        0.5 * log(dose_scale) + 0.5 * log(conc_scale)
    weights = Matrix{Float64}(undef, k, k)
    for j in 1:k
        for i in 1:k
            decay = -0.25 * (pi / 3)^2 *
                (i * i * dose_scale^2 + j * j * conc_scale^2)
            weights[i, j] = unit_weights[i + (j - 1) * k] * exp(common + decay)
        end
    end
    return weights
end

"""
    varyingsource_log_placebo(times, unit_weights, length_scale, sd, lo, hi)

Evaluate the clamped HSGP: clamp times to [lo, hi], map to [-1, 1],
and apply the padded sine basis (L=1.5) and exp-quad spectral weights. The
result is the log placebo course. Length scale and sd are constrained values.
An empty evaluation grid skips the weights, hyperparameters, and domain.
"""
function varyingsource_log_placebo(times::AbstractVector,
        unit_weights::AbstractVector, length_scale, sd, lo, hi)
    isempty(times) && return Float64[]
    _vs_native_math((times, unit_weights, length_scale, sd, lo, hi))
    length(unit_weights) >= 2 || throw(ArgumentError(
        "varyingsource placebo needs at least two basis innovations"))
    length_scale > 0 && sd > 0 && hi > lo || throw(ArgumentError(
        "varyingsource placebo needs positive scales and an increasing domain"))
    common = log(sd) + 0.45946926660233633 + 0.5 * log(length_scale)
    values = zeros(Float64, length(times))
    for j in eachindex(unit_weights)
        weight = unit_weights[j] * exp(common -
            0.25 * j * j * (length_scale * pi / 3)^2)
        for i in eachindex(times)
            t = if times[i] < lo
                lo
            elseif times[i] > hi
                hi
            else
                times[i]
            end
            x = 2 * (t - lo) / (hi - lo) - 1
            values[i] += sin((pi / 3) * (x + 1.5) * j) * weight / sqrt(1.5)
        end
    end
    return values
end

"""
    varyingsource_pd_locs(write_idxs, dts, concentration, log_placebo,
        log_baseline, log_kout, log_theta1, log_theta2)

Piecewise-exact indirect-response PD at selected grid boundaries. Each step
uses the centre-point concentration and log placebo. Indices address the
initial state (1) and post-step states (2:length(dts)+1), and must increase
strictly. Duplicate PD writes are rejected: the deployed pd_approx does not
fill repeated write indices. A missing assay skips all PD input evaluation.
"""
function varyingsource_pd_locs(write_idxs::AbstractVector{<:Integer},
        dts::AbstractVector, concentration::AbstractVector,
        log_placebo::AbstractVector, log_baseline, log_kout,
        log_theta1, log_theta2)
    isempty(write_idxs) && return Float64[]
    _vs_native_math((dts, concentration, log_placebo,
        log_baseline, log_kout, log_theta1, log_theta2))
    m = length(dts)
    length(concentration) == length(log_placebo) == m ||
        throw(DimensionMismatch("varyingsource PD step columns disagree"))
    previous = 0
    for idx in write_idxs
        previous < idx <= m + 1 || throw(ArgumentError(
            "varyingsource PD writes must strictly increase within the grid"))
        previous = idx
    end
    baseline = exp(log_baseline)
    m == 0 && return [baseline]
    kout, theta1, theta2 = exp(log_kout), exp(log_theta1), exp(log_theta2)
    kin = baseline * kout
    state = baseline
    values = Vector{Float64}(undef, length(write_idxs))
    next = 1
    if write_idxs[next] == 1
        values[next] = state
        next += 1
    end
    for i in eachindex(dts)
        dts[i] >= 0 || throw(ArgumentError("varyingsource PD step sizes must be nonnegative"))
        c = concentration[i]
        c2 = kout * exp(log_placebo[i]) * (1 + c / (theta1 * c + theta2))
        # The source recurrence, with expm1 avoiding subtraction near dt=0.
        state = state * exp(-c2 * dts[i]) + kin * (-expm1(-c2 * dts[i])) / c2
        if next <= length(write_idxs) && write_idxs[next] == i + 1
            values[next] = state
            next += 1
        end
    end
    return values
end

struct VaryingSourcePKPDCell{C}
    pk::C
end

"""
    prepare_varyingsource_pkpd(; series_rtol=1e-15, watson_terms=8)

Prepare the native full-subject PK/PD locations cell with bound transit-rule
controls. Its signature matches varyingsource_pkpd_locs. Priors, subject/dose
predictors, grid construction, and likelihood are supplied by the caller.
"""
prepare_varyingsource_pkpd(; series_rtol=1e-15, watson_terms=8) =
    VaryingSourcePKPDCell(prepare_varyingsource_pk(; series_rtol, watson_terms))

const _VARYINGSOURCE_PKPD_CELL = prepare_varyingsource_pkpd()

function _vs_write_assay!(values, assay, code, samples)
    next = 1
    for i in eachindex(assay)
        if assay[i] == code
            values[i] = samples[next]
            next += 1
        end
    end
    return nothing
end

"""
    varyingsource_pkpd_locs(n_times, assay, dose, treatment_map, unique_dts,
        conc_idxs, dosing_time_idxs, pk_idxs, pd2_idxs, pd2_dts, pd2_center_idxs,
        pd3_idxs, pd3_dts, pd3_center_idxs, dose_log_rate, dose_log_mode,
        dose_log_F, log_placebo, log_placebo_csf, effectiveness,
        log_Vc, log_k10, log_k12, log_k21, log_baseline_pbmc, log_kout,
        log_theta1_pbmc, log_theta2_pbmc, log_baseline_csf, log_theta1_csf,
        log_theta2_csf, log_absorption_rate, log_absorption_mode)

Compose the model's PK and PBMC/CSF PD locations in original assay order.
Placebo arrays contain PBMC centres followed by CSF centres. PBMC uses
log_placebo; CSF uses log_placebo + log_placebo_csf. A dose-free subject skips
PK math while retaining its PD/placebo response. Absent assays skip their
grid reads and parameter arithmetic. Execution is native only.
"""
@inline varyingsource_pkpd_locs(args...) = _VARYINGSOURCE_PKPD_CELL(args...)

@inline function (cell::VaryingSourcePKPDCell)(n_times, assay::AbstractVector{<:Integer},
        dose, treatment_map, unique_dts, conc_idxs, dosing_time_idxs, pk_idxs,
        pd2_idxs, pd2_dts, pd2_center_idxs, pd3_idxs, pd3_dts, pd3_center_idxs,
        dose_log_rate, dose_log_mode, dose_log_F, log_placebo, log_placebo_csf,
        effectiveness, log_Vc, log_k10, log_k12, log_k21, log_baseline_pbmc,
        log_kout, log_theta1_pbmc, log_theta2_pbmc, log_baseline_csf,
        log_theta1_csf, log_theta2_csf, log_absorption_rate, log_absorption_mode)
    isempty(assay) && return Float64[]
    _vs_native_math((dose_log_rate, dose_log_mode, dose_log_F,
        log_placebo, log_placebo_csf, effectiveness, log_Vc, log_k10, log_k12,
        log_k21, log_baseline_pbmc, log_kout, log_theta1_pbmc, log_theta2_pbmc,
        log_baseline_csf, log_theta1_csf, log_theta2_csf,
        log_absorption_rate, log_absorption_mode))
    counts = zeros(Int, 3)
    for a in assay
        1 <= a <= 3 || throw(ArgumentError("varyingsource assay codes must be 1:3"))
        counts[a] += 1
    end
    (length(pk_idxs), length(pd2_idxs), length(pd3_idxs)) == Tuple(counts) ||
        throw(DimensionMismatch("varyingsource assay reads disagree with observation counts"))
    # Allocate the shared buffers once; each taken arm writes its own values.
    # Empty temporary arrays never merge with active stream pointers.
    concentration = zeros(Float64, n_times)
    values = zeros(Float64, length(assay))
    if !isempty(dose)
        pk_concentration = cell.pk(n_times, dose, treatment_map, unique_dts, conc_idxs,
            dosing_time_idxs, dose_log_rate, dose_log_mode, dose_log_F,
            effectiveness, log_Vc, log_k10, log_k12, log_k21,
            log_absorption_rate, log_absorption_mode)
        for i in eachindex(concentration)
            concentration[i] = pk_concentration[i]
        end
    end
    if !isempty(pk_idxs)
        _vs_write_assay!(values, assay, 1, concentration[pk_idxs])
    end
    p2 = length(pd2_dts)
    if !isempty(pd2_idxs)
        pbmc = varyingsource_pd_locs(pd2_idxs, pd2_dts, concentration[pd2_center_idxs],
            view(log_placebo, 1:p2), log_baseline_pbmc, log_kout,
            log_theta1_pbmc, log_theta2_pbmc)
        _vs_write_assay!(values, assay, 2, pbmc)
    end
    if !isempty(pd3_idxs)
        r = (p2 + 1):(p2 + length(pd3_dts))
        csf = varyingsource_pd_locs(pd3_idxs, pd3_dts, concentration[pd3_center_idxs],
            view(log_placebo, r) .+ view(log_placebo_csf, r),
            log_baseline_csf, log_kout, log_theta1_csf, log_theta2_csf)
        _vs_write_assay!(values, assay, 3, csf)
    end
    return values
end
