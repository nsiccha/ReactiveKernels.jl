#!/usr/bin/env julia

# gp_gate.jl — authoritative acceptance gate for the GP posteriordb `@kernel`
# translations (gp_regr, gp_pois_regr, hierarchical_gp, accel_gp). Each model is
# checked, on its FULL real posteriordb data, against the ACTUAL reference
# `.stan` via BridgeStan (propto = false, jacobian = true) along four axes:
#
#   1. NATIVE VALUE     — RK primal vs BridgeStan log_density            (rel < 1e-6)
#   2. NATIVE GRADIENT  — RK plain-DifferentiationInterface+Enzyme reverse (no
#                         Reactant) vs BridgeStan log_density_gradient   (rel < 1e-3)
#   3. REACTANT PRIMAL  — Reactant-compiled primal vs the native primal  (rel < 1e-6)
#   4. REACTANT GRADIENT— Reactant-compiled gradient vs BridgeStan       (rel < 2e-3)
#
# Every axis is a hard `@assert`; a regression exits nonzero. Graphs come from
# the installed package's `build_*_graph` (templates built eagerly at module
# load, a world boundary), then `prepare`d here at top level, so no world-age
# hazard. Native axes bind ALL raw data (raw-data bound entry). Reactant axes
# bind only the design-driving data and pass the response(s) TRACED (the
# mvnormal-cholesky pattern; a bound host response falls to a scalar
# `generic_trimatdiv!` path).
#
# KNOWN GAP — snag `reactant-compile-f877fcfd` on ReactiveKernels: the exact-GP
# (dense in-graph Cholesky) models lower the Reactant PRIMAL but NOT the
# compiled REVERSE gradient — the Reactant/EnzymeMLIR pass has no adjoint for
# `stablehlo.triangular_solve`. Those entries set `chol_grad_gap = true`, and
# the gate asserts axis 4 FAILS with exactly that diagnostic (so a regression to
# a different failure, or a silent fix, is flagged and the gate must be
# updated). `accel_gp` (HSGP, no Cholesky) sets `chol_grad_gap = false` and its
# axis 4 is hard-asserted to match Stan.
#
# Reactant split at a real load boundary (Reactant cannot be unloaded once
# imported): `GP_GATE_REACTANT=0` never imports Reactant and asserts it is not
# loaded — axes 1, 2 certify genuinely Reactant-unloaded native execution;
# `GP_GATE_REACTANT=1` imports Reactant and runs axes 3, 4. The Reactant axes
# live in a SEPARATE file `include`d only from the `import Reactant` branch — a
# plain `if` does not defer `Reactant.@compile` macro-expansion. Authoritative
# acceptance = one run of each phase.
#
# Run:
#   julia --project=benchmark/all80-env benchmark/gp_gate.jl                 # phase 1
#   GP_GATE_REACTANT=0 julia --project=benchmark/all80-env benchmark/gp_gate.jl  # phase 0
# Optional GP_GATE_MODELS=gp_regr,accel_gp restricts the set.

const DO_REACTANT = get(ENV, "GP_GATE_REACTANT", "1") == "1"

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                      function_annotation = Enzyme.Const)
const VALUE_TOL = 1e-6
const GRAD_TOL = 1e-3
const RPRIMAL_TOL = 1e-6
const RGRAD_TOL = 2e-3

sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_relv(a, b) = abs(a - b) / max(abs(b), 1.0)
_mat(x) = x isa AbstractMatrix ? Float64.(x) :
          reduce(vcat, [permutedims(Float64.(r)) for r in x])

_reactant_loaded() = any(id -> id.name == "Reactant", keys(Base.loaded_modules))
if !DO_REACTANT
    @assert !_reactant_loaded() "GP_GATE_REACTANT=0 native phase must run with Reactant UNLOADED"
end

function bridge(name, seed)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    sp = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan"))
    sm = BridgeStan.StanModel(sp, PosteriorDB.load(PosteriorDB.dataset(post), String), seed)
    sm, post
end

# probes selected by REFERENCE (BridgeStan) validity, never RK finiteness
function reference_points(sm, dim, seed, name, npts, scale)
    rng = Xoshiro(seed + sum(codeunits(name)))
    pts = Vector{Vector{Float64}}(); tries = 0
    while length(pts) < npts && tries < 40_000
        tries += 1; q = scale .* randn(rng, dim)
        v = try sval(sm, q) catch; NaN end
        isfinite(v) && push!(pts, q)
    end
    length(pts) == npts || error("$name: only $(length(pts))/$npts reference-finite probes")
    pts
end

if DO_REACTANT
    import Reactant
    include(joinpath(@__DIR__, "gp_gate_reactant.jl"))
else
    reactant_axes(args...; kwargs...) = nothing
end

# gate one model. `native_bound` binds all raw data. `reactant_bound` binds only
# the design-driving data; `reactant_runtime` are the traced response(s) in HAVE
# order after the bound ports. `chol_grad_gap` selects the axis-4 behavior.
function gate(name; graph, have, native_bound, reactant_bound, reactant_runtime,
              scale = 0.5, npts = 8, seed = 468, boundary_point = nothing,
              chol_grad_gap = false)
    println("\n########## $name (REACTANT=$DO_REACTANT) ##########"); flush(stdout)
    sm, _ = bridge(name, seed)
    dim = Int(BridgeStan.param_unc_num(sm))
    pts = reference_points(sm, dim, seed, name, npts, scale)

    kb = prepare(graph; have, want = :posterior, bound = native_bound)
    prep = prepare_ad(kb, AE, pts[1]; active = :unconstrained)
    max_v = 0.0; max_g = 0.0
    for q in pts
        vr = kb(q); vs = sval(sm, q); rv = _relv(vr, vs)
        @assert isfinite(vr) "$name: native value not finite at Stan-finite point"
        @assert rv < VALUE_TOL "$name: native value rel=$rv ≥ $VALUE_TOL"
        gr = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        gs = sgrad(sm, q); rg = relerr(gr, gs)
        @assert all(isfinite, gr) "$name: native gradient not finite"
        @assert rg < GRAD_TOL "$name: native gradient rel=$rg ≥ $GRAD_TOL"
        max_v = max(max_v, rv); max_g = max(max_g, rg)
    end
    println("  [1] native value max_rel=$(round(max_v; sigdigits = 4)) (< $VALUE_TOL) PASS")
    println("  [2] native grad  max_rel=$(round(max_g; sigdigits = 4)) (< $GRAD_TOL) PASS"); flush(stdout)

    if boundary_point !== nothing
        qb = boundary_point(dim)
        vb = kb(qb); gbnd = ReactiveKernels.ad_value_and_gradient!(prep, similar(qb), qb)[2]
        @assert isfinite(vb) && all(isfinite, gbnd) "$name: boundary value/grad not finite"
        @assert _relv(vb, sval(sm, qb)) < VALUE_TOL "$name: boundary value ≠ Stan"
        @assert relerr(gbnd, sgrad(sm, qb)) < GRAD_TOL "$name: boundary gradient ≠ Stan"
        println("  [b] support-boundary probe finite + matches Stan PASS"); flush(stdout)
    end

    if DO_REACTANT
        kb_r = prepare(graph; have, want = :posterior, bound = reactant_bound)
        prep_r = prepare_ad(kb_r, AE, pts[1], reactant_runtime...; active = :unconstrained)
        @assert _relv(kb_r(pts[1], reactant_runtime...), kb(pts[1])) < 1e-10 "$name: reactant-kernel disagrees with native"
        reactant_axes(name, kb_r, prep_r, sm, pts, reactant_runtime; chol_grad_gap)
    end
    return
end

const MODELS = split(get(ENV, "GP_GATE_MODELS", "gp_regr"), ',')
_want(m) = any(x -> occursin(m, x), MODELS)
isempty(filter(!isempty, MODELS)) && error("GP_GATE_MODELS selected no models")

# gp_regr — bind raw x (folds the in-graph squared-distance design); response y
# traced for the Reactant axes. Boundary probe: extreme tiny scales (still
# Stan-finite; the covariance stays PD by construction).
_want("gp_regr") && gate("gp_pois_regr-gp_regr";
    graph = PE.GPRegrExample.build_gp_regr_graph(),
    have = (:unconstrained, :x, :y),
    native_bound = (; x = PE.GPRegrExample.GP_REGR_X, y = PE.GPRegrExample.GP_REGR_Y),
    reactant_bound = (; x = PE.GPRegrExample.GP_REGR_X),
    reactant_runtime = (PE.GPRegrExample.GP_REGR_Y,),
    scale = 1.0, npts = 8, seed = 468,
    boundary_point = _ -> [-4.0, -4.0, -8.0],
    chol_grad_gap = true)

# accel_gp — HSGP (no Cholesky). All data bound; only q traced for Reactant
# (reactant_runtime empty). The full graph lowers, so axis 4 is hard-asserted
# vs Stan (chol_grad_gap = false). Boundary probe: near-zero GP amplitudes
# (tiny sdgp on both mean and sigma GPs), still Stan-finite.
_want("accel_gp") && gate("mcycle_gp-accel_gp";
    graph = PE.AccelGPExample.build_accel_gp_graph(),
    have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1, :slambda_sigma_1),
    native_bound = (; Y = PE.AccelGPExample.ACCEL_GP_Y,
        Xgp_1 = PE.AccelGPExample.ACCEL_GP_XGP, slambda_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA,
        Xgp_sigma_1 = PE.AccelGPExample.ACCEL_GP_XGP_SIGMA,
        slambda_sigma_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA),
    reactant_bound = (; Y = PE.AccelGPExample.ACCEL_GP_Y,
        Xgp_1 = PE.AccelGPExample.ACCEL_GP_XGP, slambda_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA,
        Xgp_sigma_1 = PE.AccelGPExample.ACCEL_GP_XGP_SIGMA,
        slambda_sigma_1 = PE.AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA),
    reactant_runtime = (),
    scale = 0.3, npts = 8, seed = 468,
    boundary_point = _ -> (qb = zeros(66); qb[1] = -13.0; qb[2] = -12.0; qb[45] = -12.0; qb),
    chol_grad_gap = false)

# gp_pois_regr — non-centered latent GP + Poisson-log. bind raw x (folds the
# squared-distance design) + counts k; f_tilde rides q (traced). Reactant
# gradient is the KNOWN Cholesky-FACTOR-adjoint gap (chol_grad_gap = true).
# Boundary probe: tiny length-scale (near-diagonal covariance), Stan-finite.
_want("gp_pois_regr") && gate("gp_pois_regr-gp_pois_regr";
    graph = PE.GPPoisRegrExample.build_gp_pois_regr_graph(),
    have = (:unconstrained, :x, :k),
    native_bound = (; x = PE.GPPoisRegrExample.GP_POIS_X, k = PE.GPPoisRegrExample.GP_POIS_K),
    reactant_bound = (; x = PE.GPPoisRegrExample.GP_POIS_X, k = PE.GPPoisRegrExample.GP_POIS_K),
    reactant_runtime = (),
    scale = 0.7, npts = 8, seed = 468,
    boundary_point = _ -> (qb = zeros(13); qb[1] = -4.0; qb),
    chol_grad_gap = true)

# hierarchical_gp — dim 933. All data bound; only q traced for Reactant. The
# Reactant PRIMAL lowers (ILR simplex, dual Cholesky-factor matmul, reshape,
# index gathers), the compiled gradient is the KNOWN Cholesky-factor-adjoint gap
# (chol_grad_gap = true). No separate support-boundary probe: the finite
# reference-validity probes span the sampled region and are checked against Stan;
# a hard boundary in unconstrained q is not asserted here (finite-probe evidence
# only, not a proof over all q).
_want("hierarchical_gp") && gate("state_wide_presidential_votes-hierarchical_gp";
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
    reactant_bound = (; y = PE.HierarchicalGPExample.HGP_Y,
        year_ind = PE.HierarchicalGPExample.HGP_YEAR_IND,
        state_ind = PE.HierarchicalGPExample.HGP_STATE_IND,
        region_ind = PE.HierarchicalGPExample.HGP_REGION_IND,
        state_region_ind = PE.HierarchicalGPExample.HGP_STATE_REGION_IND,
        N_years = PE.HierarchicalGPExample.HGP_N_YEARS,
        N_regions = PE.HierarchicalGPExample.HGP_N_REGIONS,
        N_states = PE.HierarchicalGPExample.HGP_N_STATES,
        N_years_obs = PE.HierarchicalGPExample.HGP_N_YEARS_OBS),
    reactant_runtime = (),
    scale = 0.3, npts = 5, seed = 468,
    boundary_point = nothing,
    chol_grad_gap = true)

println("\n== gp_gate: all requested models PASSED (phase REACTANT=$DO_REACTANT) ==")
