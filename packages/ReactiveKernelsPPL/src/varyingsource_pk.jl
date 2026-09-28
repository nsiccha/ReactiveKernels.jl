# Native PK cell of Bruno's varyingsource3/4 twin, source checkpoint 896137dd:
# src/pkpd_models.jl:690-709, 2199-2204, 2285-2304, 3231-3263.
# Treatment j selects source column j, not the first dose carrying label j.
# Retain that deployed indexing, the normalized 2-D GP, and ordered feedback.

function _varyingsource_gp(weights, x, y)
    value = 0.0
    for i in axes(weights, 1)
        row = 0.0
        for j in axes(weights, 2)
            row += weights[i, j] * sin((pi / 3) * (y + 1.5) * j)
        end
        value += sin((pi / 3) * (x + 1.5) * i) * row
    end
    return value / 1.5
end

"""
    varyingsource_effectiveness(weights, dose_slope, conc_slope)

The varyingsource3 dose/concentration effectiveness surface with its
transformed GP weight matrix. The basis has padding 1.5, the dose domain is
10000–200000, and concentration is clamped to 0–2000 before its log1p map.
The normalizer makes a dose of 10000 at concentration zero unchanged.
Weights are the already scaled coefficients, not the model's innovations.
All basis traversal remains runtime iteration over the matrix dimensions.
"""
# Inline convenience packing/unpacking so constant coefficients and active
# slopes stay separate inputs at the mathematical kernel call.
@inline function varyingsource_effectiveness(weights::AbstractMatrix,
        dose_slope, conc_slope)
    isempty(weights) && throw(ArgumentError("effectiveness weights must be nonempty"))
    normalizer = -dose_slope - conc_slope + _varyingsource_gp(weights, -1.0, -1.0)
    return (; min_dose = 10000.0, max_dose = 200000.0, max_conc = 2000.0,
        dose_slope, conc_slope, weights, normalizer)
end

"""
    varyingsource_effective_dose(dose, concentration, effectiveness)

Evaluate the normalized 2-D dose/concentration GP from
[`varyingsource_effectiveness`](@ref). The dose coordinate is not clamped;
only the concentration coordinate is clamped, exactly as in the twin.
"""
function varyingsource_effective_dose(dose, concentration, effectiveness)
    e = effectiveness
    return _varyingsource_effective_dose(dose, concentration, e.weights,
        e.dose_slope, e.conc_slope, e.normalizer, e.min_dose, e.max_dose, e.max_conc)
end

function _varyingsource_effective_dose(dose, concentration, weights,
        dose_slope, conc_slope, normalizer, min_dose, max_dose, max_conc)
    dose > 0 || throw(ArgumentError("varyingsource doses must be positive"))
    x = 2 * (log(dose) - log(min_dose)) /
        (log(max_dose) - log(min_dose)) - 1
    clamped = if concentration < 0
        0.0
    elseif concentration > max_conc
        max_conc
    else
        concentration
    end
    y = 2 * log1p(clamped) / log1p(max_conc) - 1
    inner = dose_slope * x + conc_slope * y + _varyingsource_gp(weights, x, y)
    return dose * exp(inner - normalizer)
end

struct VaryingSourcePKCell{R}
    unit_response::R
end

"""
    prepare_varyingsource_pk(; series_rtol=1e-15, watson_terms=8)

Prepare a native varying-source PK concentration cell with numerical controls
bound in its generated transit unit-response rule. Call it with the arguments
of [`varyingsource_pk_concentration`](@ref). The series and Watson errors are
separate; these defaults do not establish a production approximation policy.
Compiled Reactant execution is not supported by this native cell.
"""
prepare_varyingsource_pk(; series_rtol = 1e-15, watson_terms = 8) =
    VaryingSourcePKCell(prepare_transit_twocmt_rule(; series_rtol, watson_terms))

const _VARYINGSOURCE_PK_CELL = prepare_varyingsource_pk()

"""
    varyingsource_pk_concentration(n_times, dose, treatment_map, unique_dts,
        concentration_idxs, dosing_time_idxs, dose_log_rate, dose_log_mode,
        dose_log_F, effectiveness, log_Vc, log_k10, log_k12, log_k21,
        log_absorption_rate, log_absorption_mode)

One subject's PK concentrations on its `n_times` reference-time grid. Reuse
one Gamma-transit/twocmt unit solve per treatment, then accumulate doses in
order. Each dose's effectiveness reads the concentration produced by earlier
doses at `dosing_time_idxs[i]`; its bioavailability is `exp(dose_log_F[i]-log_Vc)`.

`concentration_idxs` contains `n_times` lag indices per dose, in dose-major
order. Nonpositive time differences select the zero-lag unit response.
Treatment-map value `j` selects source column `j`, assembled from
`dose_log_rate[j]` and `dose_log_mode[j]`. This preserves the deployed twin's
first-`maximum(treatment_map)` source-column selection, even when label `j`
first appears at another dose. The rate is `exp(log_absorption_rate+modifier)`;
Gamma shape is `1 + rate*exp(log_absorption_mode+modifier)`.

An empty dose sequence returns zeros without evaluating the unit solve,
effectiveness surface, or log parameters. Indices and grid sizes remain data;
all treatment, lag, dose, and reference-time traversals are runtime loops.
Use [`prepare_varyingsource_pk`](@ref) to bind different accuracy controls.
This native cell supplies the PK component; PD and BRM emission are separate.
"""
@inline function varyingsource_pk_concentration(n_times, dose, treatment_map, unique_dts,
        concentration_idxs, dosing_time_idxs, dose_log_rate, dose_log_mode,
        dose_log_F, effectiveness, log_Vc, log_k10, log_k12, log_k21,
        log_absorption_rate, log_absorption_mode)
    return _VARYINGSOURCE_PK_CELL(n_times, dose, treatment_map, unique_dts,
        concentration_idxs, dosing_time_idxs, dose_log_rate, dose_log_mode,
        dose_log_F, effectiveness, log_Vc, log_k10, log_k12, log_k21,
        log_absorption_rate, log_absorption_mode)
end

@inline function (cell::VaryingSourcePKCell)(n_times::Integer, dose::AbstractVector,
        treatment_map::AbstractVector{<:Integer}, unique_dts::AbstractVector,
        concentration_idxs::AbstractVector{<:Integer},
        dosing_time_idxs::AbstractVector{<:Integer},
        dose_log_rate::AbstractVector, dose_log_mode::AbstractVector,
        dose_log_F::AbstractVector, effectiveness, log_Vc, log_k10, log_k12,
        log_k21, log_absorption_rate, log_absorption_mode)
    _vs_pk_check_lengths(n_times, dose, treatment_map, concentration_idxs,
        dosing_time_idxs, dose_log_rate, dose_log_mode, dose_log_F)
    isempty(dose) && return zeros(Float64, n_times)
    e = effectiveness
    return _varyingsource_pk_concentration(cell, n_times, dose, treatment_map,
        unique_dts, concentration_idxs, dosing_time_idxs, dose_log_rate,
        dose_log_mode, dose_log_F, e.weights, e.dose_slope, e.conc_slope,
        e.normalizer, e.min_dose, e.max_dose, e.max_conc, log_Vc, log_k10,
        log_k12, log_k21, log_absorption_rate, log_absorption_mode)
end

function _vs_pk_check_lengths(n_times, dose, treatment_map, concentration_idxs,
        dosing_time_idxs, dose_log_rate, dose_log_mode, dose_log_F)
    n_times >= 0 || throw(ArgumentError("n_times must be nonnegative"))
    m = length(dose)
    length(treatment_map) == length(dosing_time_idxs) == length(dose_log_rate) ==
        length(dose_log_mode) == length(dose_log_F) == m ||
        throw(DimensionMismatch("varyingsource per-dose columns disagree"))
    length(concentration_idxs) == m * n_times ||
        throw(DimensionMismatch("concentration_idxs needs n_times indices per dose"))
    return nothing
end

# Keep GP coefficients and scalar parameters as separate mathematical inputs.
# A temporary boxed effectiveness tuple with constant coefficients and active
# slopes obscures field activity across the subject loop in native Enzyme.
function _varyingsource_pk_concentration(cell::VaryingSourcePKCell, n_times,
        dose, treatment_map, unique_dts, concentration_idxs, dosing_time_idxs,
        dose_log_rate, dose_log_mode, dose_log_F, weights, dose_slope, conc_slope,
        normalizer, min_dose, max_dose, max_conc, log_Vc, log_k10, log_k12,
        log_k21, log_absorption_rate, log_absorption_mode)
    _vs_pk_check_lengths(n_times, dose, treatment_map, concentration_idxs,
        dosing_time_idxs, dose_log_rate, dose_log_mode, dose_log_F)
    m = length(dose)
    # The source's no-dose arm is lazy, including all GP and unit-solve work.
    m == 0 && return zeros(Float64, n_times)
    marker = ReactiveKernels._dynamic_tensorized_marker((dose_log_rate,
        dose_log_mode, dose_log_F, weights, dose_slope, conc_slope, normalizer,
        min_dose, max_dose, max_conc, log_Vc, log_k10,
        log_k12, log_k21, log_absorption_rate, log_absorption_mode))
    marker === nothing || throw(ArgumentError(
        "varyingsource PK cells support native execution only; compiled " *
        "transit-rule and sequential-dose control flow is not established"))
    g = length(unique_dts)
    g > 0 || throw(ArgumentError("a dosed subject needs a lag grid"))
    nt = maximum(treatment_map)
    1 <= nt <= m || throw(ArgumentError("treatment map is outside source columns"))
    for i in eachindex(dose)
        dose[i] > 0 || throw(ArgumentError("varyingsource doses must be positive"))
        1 <= treatment_map[i] <= nt || throw(ArgumentError("invalid treatment index"))
        1 <= dosing_time_idxs[i] <= n_times ||
            throw(ArgumentError("dosing time index is outside the reference grid"))
    end
    for idx in concentration_idxs
        1 <= idx <= g || throw(ArgumentError("concentration index is outside the lag grid"))
    end
    # The generated transit rule accepts native Float64 vectors. Copy bound
    # views with an ordinary loop; Base.collect's source/destination alias
    # branch obscures static activity when nested in the subject loop.
    ts = Vector{Float64}(undef, g)
    for i in 1:g
        ts[i] = Float64(unique_dts[i])
    end
    units = Matrix{Float64}(undef, g, nt)
    k10, k12, k21 = exp(log_k10), exp(log_k12), exp(log_k21)
    for j in 1:nt
        rate = exp(log_absorption_rate + dose_log_rate[j])
        mode = exp(log_absorption_mode + dose_log_mode[j])
        response = cell.unit_response(ts, [k10, k12, k21, rate, 1 + rate * mode])
        for k in 1:g
            units[k, j] = response[k]
        end
    end
    concentration = zeros(Float64, n_times)
    for i in 1:m
        effective = _varyingsource_effective_dose(dose[i],
            concentration[dosing_time_idxs[i]], weights, dose_slope, conc_slope,
            normalizer, min_dose, max_dose, max_conc)
        factor = effective * exp(dose_log_F[i] - log_Vc)
        j = treatment_map[i]
        offset = (i - 1) * n_times
        for k in 1:n_times
            concentration[k] += factor * units[concentration_idxs[offset + k], j]
        end
    end
    return concentration
end

"""
    varyingsource_pk_locs(pk_idxs, args...)

Gather PK observation locations, preserving duplicates and observation order,
from [`varyingsource_pk_concentration`](@ref). Empty-dose locations are zero.
"""
function varyingsource_pk_locs(pk_idxs::AbstractVector{<:Integer}, n_times, dose,
        treatment_map, unique_dts, concentration_idxs, dosing_time_idxs,
        dose_log_rate, dose_log_mode, dose_log_F, effectiveness, log_Vc,
        log_k10, log_k12, log_k21, log_absorption_rate, log_absorption_mode)
    return varyingsource_pk_concentration(n_times, dose, treatment_map, unique_dts,
        concentration_idxs, dosing_time_idxs, dose_log_rate, dose_log_mode,
        dose_log_F, effectiveness, log_Vc, log_k10, log_k12, log_k21,
        log_absorption_rate, log_absorption_mode)[pk_idxs]
end
