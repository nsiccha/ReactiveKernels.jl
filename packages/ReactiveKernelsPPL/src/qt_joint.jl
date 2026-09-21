# QT coupling + PK/QT in-cell observations for the joint PK+QT(+TGI) thin
# layer — SB-mirror of the brm2 V2 (`raw_axes`) kernel cell at DEFAULT
# config only (direct_linear spine, Gaussian QT obs family, additive QT
# amplitude, no time effect / comed / R² / regime variants — those are
# sequenced separately).
#
# SB source of truth (bruno `kb-impl/Bruno-arv393-tgi` @ `3af22846`, audit
# triple `web-pkpd/test/stan_audit_triples/joint_brm2_default.md`):
# - cell (`brm_integration.jl:4042-4045`, triple §1):
#     qt_loc = qbase + qslope * (conc[ecg_idx] ./ 0.8)
#     qt_y ~ normal(qt_loc, qt_scale .* qt_weight)
#   with the training-derived `reference_exposure` (max observed PK conc)
#   baked as a literal (`brm_integration.jl:4029-4041`, decision
#   `2026-08-19T10-50-49-773-0ec3tx1`).
# - PK likelihood (top-level over the collected `conc[pk_idx]`,
#   `brm_integration.jl:4064-4066`, triple §1):
#     ragged(pk_conc, obs_subject) ~ censored(Normal(pk_loc,
#         addprop(pk_loc, sigma_add, sigma_prop)); lower=pk_lloq)
#   lowering to Stan `lower_clamping_normal` (triple §3): per element,
#   `y == lo` takes `normal_lcdf_stable(lo, loc, scale) =
#   log(erfc(-(lo-loc)/(scale*√2))) - log(2)`, `y > lo` takes
#   `normal_lpdf`, else `-inf`; `addprop(loc, add, prop) =
#   sqrt(add^2 + (loc*prop)^2)` elementwise.
# - prep (`brm_integration.jl:2404,3363,2975-2976`): BLQ rows are clamped
#   (`pk_conc = max.(pk_conc, pk_lloq)`, so `y >= lo` by construction and
#   the `-inf` arm is unreachable); `inv_sqrt_k` (the per-ECG-row
#   `qt_weight` data) must be finite and positive.
#
# This file owns the QT slice of the joint cell: the coupling builder, the
# two observation-statement spellings, their fail-closed admission, the
# prep contract, and the generator-level likelihood lowering the
# grouped-kernel emitter calls. It assumes per-subject concentration
# columns and the `qt_base`/`qt_slope` LP values as HAVE ports (sibling
# slices own the PK recurrence, the TGI block, and the grouped-kernel
# foundation: LP args, ragged data args, multi-obs cells, non-Gaussian
# in-cell families, `[...]` gathers, and `erfc` in the cell vocabulary).

"""QT coupling spines the thin layer lowers (default config: direct-linear
C–QT only; `direct_emax` / `direct_logistic` are sequenced separately)."""
const QT_COUPLING_SPINES = (:direct_linear,)

"""QT observation families the thin layer lowers (default config: Gaussian
only; `student_t` is sequenced separately)."""
const QT_OBS_FAMILIES = (:gaussian,)

"""Admit a QT coupling spine, fail-closed naming the admitted spellings."""
function admit_qt_spine(spine::Symbol)
    spine in QT_COUPLING_SPINES ||
        throw(ContractValidationError("[qt_joint] QT coupling spine " *
              "`$spine` is not admitted (admitted: " *
              join(map(repr, QT_COUPLING_SPINES), ", ") * "; " *
              "`direct_emax` / `direct_logistic` are sequenced separately)"))
    return spine
end

"""Admit a QT observation family, fail-closed naming the admitted spellings."""
function admit_qt_obs_family(family::Symbol)
    family in QT_OBS_FAMILIES ||
        throw(ContractValidationError("[qt_joint] QT observation family " *
              "`$family` is not admitted (admitted: " *
              join(map(repr, QT_OBS_FAMILIES), ", ") * "; " *
              "`student_t` is sequenced separately)"))
    return family
end

"""Reference-exposure prep value: finite and positive (the training max PK
conc SB bakes as a literal; the thin layer takes it as given)."""
function _qt_joint_reference_exposure(reference_exposure::Real)
    ref = Float64(reference_exposure)
    isfinite(ref) && ref > 0 ||
        throw(ContractValidationError("[qt_joint] reference_exposure must " *
              "be finite and positive (training max PK conc), got " *
              repr(reference_exposure)))
    return ref
end

"""
    qt_loc_assignment(spine = :direct_linear; base = :qbase, slope = :qslope,
        conc = :conc, ecg_idx = :ecg_idx, reference_exposure) -> Pair{Symbol,Expr}

The `direct_linear` C–QT coupling cell assignment, SB-mirror of
`brm_integration.jl:4042-4045`: `qt_loc = qbase + qslope *
(conc[ecg_idx] ./ <ref>)` with the prep `reference_exposure` baked as a
`Float64` literal. Scalar-context `*` (the surface dotifies per
slice-kind provenance, the Ex1 `ke = CLi / Vci` precedent). `base` /
`slope` are the per-subject `qt_base` / `qt_slope` LP HAVE ports.
"""
function qt_loc_assignment(spine::Symbol = :direct_linear;
        base::Symbol = :qbase, slope::Symbol = :qslope,
        conc::Symbol = :conc, ecg_idx::Symbol = :ecg_idx,
        reference_exposure::Real)
    admit_qt_spine(spine)
    ref = _qt_joint_reference_exposure(reference_exposure)
    rhs = Expr(:call, :+, base, Expr(:call, :*, slope,
        Expr(:call, :./, Expr(:ref, conc, ecg_idx), ref)))
    return :qt_loc => rhs
end

"""
    qt_obs_statement(family = :gaussian; response = :qt_y, location = :qt_loc,
        scale = :qt_scale, weight = :qt_weight) -> Expr

The default-config in-cell QT observation, SB-mirror of
`_joint_pk_qt_obs_statement("gaussian", "qt_scale")`
(`brm_integration.jl:3591-3594`): `qt_y .~ Normal.(qt_loc, qt_scale .*
qt_weight)`. Dotted per the explicit-dots ruling (scalar `~` over vectors
is rejected); `weight` is the per-ECG-row `inv_sqrt_k` data HAVE.
"""
function qt_obs_statement(family::Symbol = :gaussian;
        response::Symbol = :qt_y, location::Symbol = :qt_loc,
        scale::Symbol = :qt_scale, weight::Symbol = :qt_weight)
    admit_qt_obs_family(family)
    dist = Expr(:., :Normal, Expr(:tuple, location,
        Expr(:call, :.*, scale, weight)))
    return Expr(:call, :.~, response, dist)
end

"""
    pk_obs_statement(; response = :pk_y, location = :pk_loc,
        add = :sigma_add, prop = :sigma_prop, lloq = :pk_lloq) -> Expr

The in-cell PK observation, SB-mirror of the V2 top-level
`censored(Normal(pk_loc, addprop(pk_loc, sigma_add, sigma_prop));
lower=pk_lloq)` (`brm_integration.jl:4064-4066`) over the collected
`conc[pk_idx]`: `pk_y .~ CensoredAddpropnormal.(pk_loc, sigma_add,
sigma_prop, pk_lloq)`. The family name mirrors bruno's V1
`censored_addpropnormal` adapter (positional `(location, add, prop,
lloq)`, julianic CamelCase + dots). Exactly one spelling is admitted.
"""
function pk_obs_statement(;
        response::Symbol = :pk_y, location::Symbol = :pk_loc,
        add::Symbol = :sigma_add, prop::Symbol = :sigma_prop,
        lloq::Symbol = :pk_lloq)
    dist = Expr(:., :CensoredAddpropnormal,
        Expr(:tuple, location, add, prop, lloq))
    return Expr(:call, :.~, response, dist)
end

"""
    validate_qt_joint_prep(; reference_exposure, pk_conc, pk_lloq, inv_sqrt_k)

Bind-time prep contract for the QT slice (SB prep mirror,
`brm_integration.jl:2404,3363,2975-2976`): positive finite reference
exposure; `pk_conc`/`pk_lloq` pairwise (BLQ rows clamped to LLOQ, so
`pk_conc .>= pk_lloq` — the SB `-inf` arm is unreachable by
construction); finite positive per-ECG-row `inv_sqrt_k`. Ragged
structure/lengths are the grouped-kernel bind's own rules.
"""
function validate_qt_joint_prep(;
        reference_exposure::Real,
        pk_conc::AbstractVector{<:Real},
        pk_lloq::AbstractVector{<:Real},
        inv_sqrt_k::AbstractVector{<:Real})
    _qt_joint_reference_exposure(reference_exposure)
    length(pk_conc) == length(pk_lloq) ||
        throw(ContractValidationError("[qt_joint] pk_conc / pk_lloq length " *
              "mismatch ($(length(pk_conc)) vs $(length(pk_lloq)): " *
              "one LLOQ per PK row)"))
    all(pk_conc .>= pk_lloq) ||
        throw(ContractValidationError("[qt_joint] pk_conc below pk_lloq: " *
              "clamp BLQ rows to LLOQ at prep (`max.(pk_conc, pk_lloq)`)"))
    all(x -> isfinite(x) && x > 0, inv_sqrt_k) ||
        throw(ContractValidationError("[qt_joint] inv_sqrt_k must be " *
              "finite and positive (per-ECG-row replicate weight)"))
    return nothing
end

# Generator-level in-cell likelihood lowering for the QT slice. The
# grouped-kernel emitter calls these with resolved flat refs; each returns
# `(stmts, term)` in the `_kernel_plate_likelihood` style (`term` joins the
# `_likelihood_statements` sum). Plates mirror `_plate_sum_stmts`
# (pointwise plate + scalar sum node).

"""QT likelihood: `qt_scale .* qt_weight` rides a flat pre-local (the
computed-flat-local-as-plate-input precedent), then a plain normal plate
(SB model block: `ecg_y ~ normal(qt_loc, qt_scale .* inv_sqrt_k)`)."""
function _qt_joint_qt_likelihood_stmts(;
        response::Symbol, location::Symbol, scale::Symbol,
        weight::Symbol, label::Symbol)
    sc = Symbol(:_ppl_qt_scale_, label)
    pre = Expr(:(=), sc, Expr(:call, :.*, scale, weight))
    klabel = Symbol(:qt_joint_qt_, label)
    pw = _pw_name(klabel)
    node = _lik_name(klabel)
    inputs = Any[response, location, sc]
    yv, lpv, sv = _dovar(1), _dovar(2), _dovar(3)
    cell = :(normal($lpv, $sv).logpdf($yv))
    return Expr[pre, _plate_sum_stmts(pw, node, inputs, cell)...], node
end

"""PK likelihood: `addprop` rides a flat pre-local
(`sqrt.(add^2 .+ (loc .* prop).^2)`, SB's `addprop`), then a per-element
`lower_clamping` plate (SB's `lower_clamping_normal_lpdf` branch
structure: at-bound `==` takes the log-cdf arm, above-bound takes the
logpdf arm, below-bound is unreachable — prep clamps). The log-cdf arm
reuses the landed censored-Gaussian `log(normal(...).cdf(...))` form
(`_gaussian_cell`; ≤2ulp from SB's `log(erfc) - log(2)` spelling, no new
formula); the `==` boundary differs deliberately from top-level
`:censored` (`<`), where at-bound rows take the density — here every BLQ
row sits exactly at LLOQ and SB takes the cdf there."""
function _qt_joint_pk_likelihood_stmts(;
        response::Symbol, location::Symbol, add::Symbol, prop::Symbol,
        lloq::Symbol, label::Symbol)
    sc = Symbol(:_ppl_pk_scale_, label)
    pre = Expr(:(=), sc, Expr(:., :sqrt, Expr(:tuple, Expr(:call, :.+,
        Expr(:call, :^, add, 2),
        Expr(:call, :.^, Expr(:call, :.*, location, prop), 2)))))
    klabel = Symbol(:qt_joint_pk_, label)
    pw = _pw_name(klabel)
    node = _lik_name(klabel)
    inputs = Any[response, location, sc, lloq]
    yv, lpv, sv, lov = _dovar(1), _dovar(2), _dovar(3), _dovar(4)
    base = :(normal($lpv, $sv).logpdf($yv))
    cell = :(ifelse($yv == $lov, log(normal($lpv, $sv).cdf($lov)), $base))
    return Expr[pre, _plate_sum_stmts(pw, node, inputs, cell)...], node
end
