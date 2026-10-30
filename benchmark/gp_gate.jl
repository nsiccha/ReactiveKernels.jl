#!/usr/bin/env julia

# gp_gate.jl — authoritative acceptance gate for the GP posteriordb `@kernel`
# translations (gp_regr, gp_pois_regr, hierarchical_gp, accel_gp). Each model is
# checked, on its FULL real posteriordb data, against the ACTUAL reference
# `.stan` via BridgeStan (propto = false, jacobian = true) along two axes:
#
#   1. NATIVE VALUE     — RK primal vs BridgeStan log_density            (rel < 1e-6)
#   2. NATIVE GRADIENT  — RK plain-DifferentiationInterface+Enzyme reverse
#                         vs BridgeStan log_density_gradient             (rel < 1e-3)
#
# Every axis is a hard `@assert`; a regression exits nonzero. Graphs come from
# the installed package's `build_*_graph` (templates built eagerly at module
# load, a world boundary), then `prepare`d here at top level, so no world-age
# hazard. The gate binds ALL raw data (raw-data bound entry).
#
# Run:
#   julia --project=benchmark/all80-env benchmark/gp_gate.jl
# Optional GP_GATE_MODELS=gp_regr,accel_gp,gp_pois_regr,hierarchical_gp
# restricts the set. Selection is exact and defaults to all four.

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse)
const VALUE_TOL = 1e-6
const GRAD_TOL = 1e-3

const GP_MODEL_WHITELIST = (
    "gp_regr", "accel_gp", "gp_pois_regr", "hierarchical_gp",
)
const GP_GATE_DEFAULT_MODELS = join(GP_MODEL_WHITELIST, ',')
const MODELS = Tuple(String.(strip.(split(
    get(ENV, "GP_GATE_MODELS", GP_GATE_DEFAULT_MODELS), ',',
))))
all(!isempty, MODELS) || error("GP_GATE_MODELS contains an empty selection")
const INVALID_MODELS = filter(m -> !(m in GP_MODEL_WHITELIST), MODELS)
isempty(INVALID_MODELS) || error(
    "GP_GATE_MODELS must select only $(join(GP_MODEL_WHITELIST, ", ")); got $(repr(MODELS))",
)
length(MODELS) == length(unique(MODELS)) || error("GP_GATE_MODELS contains duplicates")
length(MODELS) > 0 || error("GP_GATE_MODELS selected no models")
_want(m) = m in MODELS
const EXECUTED_MODELS = String[]

sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_relv(a, b) = abs(a - b) / max(abs(b), 1.0)
_mat(x) = x isa AbstractMatrix ? Float64.(x) :
          reduce(vcat, [permutedims(Float64.(r)) for r in x])

function bridge(name, seed; accel_prior_only::Union{Nothing,Int} = nothing)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    sp = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan"))
    data = PosteriorDB.load(PosteriorDB.dataset(post), String)
    if accel_prior_only !== nothing
        old = accel_prior_only == 1 ? 0 : 1
        pattern = Regex("\"prior_only\"\\s*:\\s*$old")
        length(collect(eachmatch(pattern, data))) == 1 || error(
            "accel_gp data must contain exactly one prior_only=$old field",
        )
        data = replace(
            data, pattern => SubstitutionString("\"prior_only\":$accel_prior_only");
            count = 1,
        )
    end
    sm = BridgeStan.StanModel(sp, data, seed)
    sm, post
end

# probes selected by REFERENCE (BridgeStan) validity, never RK finiteness
function reference_points(sm, dim, seed, name, npts, scale)
    rng = Xoshiro(seed + sum(codeunits(name)))
    pts = Vector{Vector{Float64}}(); tries = 0
    while length(pts) < npts && tries < 40_000
        tries += 1; q = scale .* randn(rng, dim)
        @assert all(isfinite, q) "$name generated a non-finite probe"
        v = try sval(sm, q) catch; NaN end
        isfinite(v) && push!(pts, q)
    end
    length(pts) == npts || error("$name: only $(length(pts))/$npts reference-finite probes")
    pts
end

# gate one model. `native_bound` binds all raw data.
function gate(name; posterior_name = name, key = name, want = :posterior,
              accel_prior_only = nothing, graph, have,
              native_bound, scale = 0.5, npts = 8, seed = 468,
              boundary_point = nothing)
    println("\n########## $name ##########"); flush(stdout)
    key in MODELS && push!(EXECUTED_MODELS, key)
    sm, _ = bridge(posterior_name, seed; accel_prior_only)
    dim = Int(BridgeStan.param_unc_num(sm))
    pts = reference_points(sm, dim, seed, name, npts, scale)
    boundary_q = boundary_point === nothing ? nothing : boundary_point(dim)

    kb = prepare(graph; have, want, bound = native_bound)
    prep = prepare_ad(kb, AE, pts[1]; active = :unconstrained)
    max_v = 0.0; max_g = 0.0
    for q in pts
        vr = kb(q); vs = sval(sm, q); rv = _relv(vr, vs)
        @assert isfinite(vs) "$name: BridgeStan value not finite at selected reference point"
        @assert isfinite(vr) "$name: native value not finite at Stan-finite point"
        @assert rv < VALUE_TOL "$name: native value rel=$rv ≥ $VALUE_TOL"
        gr = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        gs = sgrad(sm, q); rg = relerr(gr, gs)
        @assert all(isfinite, gs) "$name: BridgeStan gradient not finite"
        @assert all(isfinite, gr) "$name: native gradient not finite"
        @assert rg < GRAD_TOL "$name: native gradient rel=$rg ≥ $GRAD_TOL"
        max_v = max(max_v, rv); max_g = max(max_g, rg)
    end
    println("  [1] native value max_rel=$(round(max_v; sigdigits = 4)) (< $VALUE_TOL) PASS")
    println("  [2] native grad  max_rel=$(round(max_g; sigdigits = 4)) (< $GRAD_TOL) PASS"); flush(stdout)

    if boundary_q !== nothing
        qb = boundary_q
        vb = kb(qb); gbnd = ReactiveKernels.ad_value_and_gradient!(prep, similar(qb), qb)[2]
        sb = sval(sm, qb); gs_b = sgrad(sm, qb)
        @assert isfinite(sb) && all(isfinite, gs_b) "$name: boundary BridgeStan reference not finite"
        @assert isfinite(vb) && all(isfinite, gbnd) "$name: boundary value/grad not finite"
        @assert _relv(vb, sb) < VALUE_TOL "$name: boundary value ≠ Stan"
        @assert relerr(gbnd, gs_b) < GRAD_TOL "$name: boundary gradient ≠ Stan"
        println("  [b] support-boundary probe finite + matches Stan PASS"); flush(stdout)
    end

    return
end

# gp_regr — bind raw x (folds the in-graph squared-distance design). Boundary
# probe: extreme tiny scales (still Stan-finite; the covariance stays PD by
# construction).
_want("gp_regr") && gate("gp_pois_regr-gp_regr";
    key = "gp_regr",
    graph = PE.GPRegrExample.build_gp_regr_graph(),
    have = (:unconstrained, :x, :y),
    native_bound = (; x = PE.GPRegrExample.GP_REGR_X, y = PE.GPRegrExample.GP_REGR_Y),
    scale = 1.0, npts = 8, seed = 468,
    boundary_point = _ -> [-4.0, -4.0, -8.0])

# accel_gp — HSGP (no Cholesky). Boundary probe: near-zero GP amplitudes (tiny
# sdgp on both mean and sigma GPs), still Stan-finite.
_want("accel_gp") && gate("mcycle_gp-accel_gp";
    key = "accel_gp",
    graph = PE.AccelGPExample.build_accel_gp_graph(),
    have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1, :slambda_sigma_1),
    native_bound = (; Y = PE.AccelGPExample.ACCEL_GP_Y,
        Xgp_1 = PE.AccelGPExample.ACCEL_GP_XGP, slambda_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA,
        Xgp_sigma_1 = PE.AccelGPExample.ACCEL_GP_XGP_SIGMA,
        slambda_sigma_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA),
    scale = 0.3, npts = 8, seed = 468,
    boundary_point = _ -> (qb = zeros(66); qb[1] = -13.0; qb[2] = -12.0; qb[45] = -12.0; qb))

# The same graph with Stan's actual prior_only=1 data switch: the posterior and
# gradient come from the prior + Jacobian and are invariant to the response.
_want("accel_gp") && gate("mcycle_gp-accel_gp (prior_only=1)";
    posterior_name = "mcycle_gp-accel_gp", key = "accel_gp_prior_only",
    accel_prior_only = 1,
    want = PE.AccelGPExample.accel_gp_posterior_want(true),
    graph = PE.AccelGPExample.build_accel_gp_graph(),
    have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1, :slambda_sigma_1),
    native_bound = (; Y = PE.AccelGPExample.ACCEL_GP_Y,
        Xgp_1 = PE.AccelGPExample.ACCEL_GP_XGP, slambda_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA,
        Xgp_sigma_1 = PE.AccelGPExample.ACCEL_GP_XGP_SIGMA,
        slambda_sigma_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA),
    scale = 0.3, npts = 5, seed = 469,
    boundary_point = _ -> begin
        qb = zeros(66)
        nb1 = size(PE.AccelGPExample.ACCEL_GP_XGP, 2)
        qb[1] = -13.0
        qb[2] = -12.0
        qb[4 + nb1] = -800.0
        qb
    end)

# gp_pois_regr — non-centered latent GP + Poisson-log. bind raw x (folds the
# squared-distance design) + counts k; f_tilde rides q. Boundary probe: tiny
# length-scale (near-diagonal covariance), Stan-finite.
_want("gp_pois_regr") && gate("gp_pois_regr-gp_pois_regr";
    key = "gp_pois_regr",
    graph = PE.GPPoisRegrExample.build_gp_pois_regr_graph(),
    have = (:unconstrained, :x, :k),
    native_bound = (; x = PE.GPPoisRegrExample.GP_POIS_X, k = PE.GPPoisRegrExample.GP_POIS_K),
    scale = 0.7, npts = 8, seed = 468,
    boundary_point = _ -> (qb = zeros(13); qb[1] = -4.0; qb))

# hierarchical_gp — dim 933. All data bound. No separate support-boundary
# probe: the finite
# reference-validity probes span the sampled region and are checked against Stan;
# a hard boundary in unconstrained q is not asserted here (finite-probe evidence
# only, not a proof over all q).
_want("hierarchical_gp") && gate("state_wide_presidential_votes-hierarchical_gp";
    key = "hierarchical_gp",
    graph = PE.HierarchicalGPExample.build_hierarchical_gp_graph(),
    have = (:unconstrained, :y, :year_ind, :state_ind, :region_ind, :state_region_ind,
            :N_years, :N_regions, :N_states, :N_years_obs),
    native_bound = (; y = PE.HierarchicalGPExample.HGP_Y,
        year_ind = PE.HierarchicalGPExample.HGP_YEAR_IND,
        state_ind = PE.HierarchicalGPExample.HGP_STATE_IND,
        region_ind = PE.HierarchicalGPExample.HGP_REGION_IND,
        state_region_ind = PE.HierarchicalGPExample.HGP_STATE_REGION_IND,
        N_years = PE.HierarchicalGPExample.HGP_N_YEARS,
        N_regions = PE.HierarchicalGPExample.HGP_N_REGIONS,
        N_states = PE.HierarchicalGPExample.HGP_N_STATES,
        N_years_obs = PE.HierarchicalGPExample.HGP_N_YEARS_OBS),
    scale = 0.3, npts = 5, seed = 468,
    boundary_point = nothing)

@assert (length(EXECUTED_MODELS) == length(MODELS) &&
         Set(EXECUTED_MODELS) == Set(MODELS)) "GP_GATE_MODELS selection did not execute exactly once: requested=$MODELS executed=$EXECUTED_MODELS"
println("\n== gp_gate: requested checks PASSED ==")
