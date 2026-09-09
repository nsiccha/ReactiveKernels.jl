#!/usr/bin/env julia

# batch_latent_gate.jl — the four-part correctness gate for the latent/spatial
# posteriordb `@kernel` PPL translations (GLMM1, bym2_offset_only, bones,
# multi_occupancy). Each model is checked, on its FULL real posteriordb data,
# against the ACTUAL reference `.stan` via BridgeStan (propto = false,
# jacobian = true) along four independent axes:
#
#   1. NATIVE VALUE      — RK primal log-density vs BridgeStan log_density
#                          (relative error < 1e-6).
#   2. NATIVE GRADIENT   — RK gradient via plain DifferentiationInterface + Enzyme
#                          reverse (no Reactant) vs BridgeStan log_density_gradient
#                          (relative error < 1e-3), finite at every probe.
#   3. REACTANT PRIMAL   — the Reactant-compiled primal vs the native primal
#                          (relative error < 1e-6).
#   4. REACTANT GRADIENT — the Reactant-compiled gradient vs BridgeStan
#                          (relative error < 2e-3), finite.
#
# Every axis is a hard `@assert`, so a regression exits nonzero. The native axes
# evaluate exactly `NATIVE_PROBE_COUNT` (six) reference-finite unconstrained
# points; the Reactant axes evaluate exactly the FIRST of those points. Those
# finite-point checks are bounded evidence, not an all-input proof. Graphs are
# built from the installed package's `build_*_graph` (its templates are evaluated
# at module-load, a world boundary), then `prepare`d here at top level, so no
# world-age hazard arises. Each model binds ONLY the RAW posteriordb data ports
# it declares — every structural coordinate and derived quantity is in-graph.
#
# This is the AUTHORITATIVE real-package acceptance driver, distinct from any
# exploratory stub loader and from the per-model regression testsets in
# packages/ReactiveKernelsPPLExamples/test (which assert value equality vs
# independent reference oracles, not vs the .stan reference over probes).
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Reactant, Enzyme — all in
# the all80 environment):
#
#   julia --project=benchmark/all80-env benchmark/batch_latent_gate.jl
#
# Optional: BATCH_LATENT_MODELS=glmm1,bones restricts the set.
#
# The Reactant axes (3, 4) and the two native axes (1, 2) MUST be certified in
# SEPARATE processes: `BATCH_LATENT_REACTANT=0` never imports Reactant and asserts
# it is not loaded, so axes 1 & 2 certify plain-native execution with the Reactant
# extension genuinely absent; `BATCH_LATENT_REACTANT=1` imports Reactant and runs
# all four. Authoritative acceptance = one run of each. Because Reactant cannot be
# unloaded once imported, the split is a real load boundary, not a runtime flag.

const DO_REACTANT = get(ENV, "BATCH_LATENT_REACTANT", "1") == "1"

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

if DO_REACTANT
    import Reactant
end

_reactant_loaded() = any(id -> id.name == "Reactant", keys(Base.loaded_modules))
if !DO_REACTANT
    @assert !_reactant_loaded() "BATCH_LATENT_REACTANT=0 native phase must run with Reactant UNLOADED, but it is loaded"
end

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse)
const NATIVE_PROBE_COUNT = 6
const REACTANT_PROBE_COUNT = 1

# Package load must have built its graph templates MODEL-ONLY (no per-source
# prepare/execute demo tail); assert the startup contract here so the gate also
# certifies cheap import.
@assert PE._DEMO_TAIL_EXECUTIONS[] == 0 "package import ran a demo tail — __init__ is not model_only"

const VALUE_TOL  = 1e-6      # native value vs Stan
const GRAD_TOL   = 1e-3      # native plain-Enzyme gradient vs Stan
const RPRIMAL_TOL = 1e-6     # Reactant primal vs native
const RGRAD_TOL  = 2e-3      # Reactant gradient vs Stan

sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_relv(a, b) = abs(a - b) / max(abs(b), 1.0)
_mat(x) = x isa AbstractMatrix ? x :
          reduce(vcat, [permutedims(r) for r in x])

function bridge(name, seed)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    sp = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan"))
    sm = BridgeStan.StanModel(sp, PosteriorDB.load(PosteriorDB.dataset(post), String), seed)
    sm, post
end

# Draw `npts` unconstrained points selected by REFERENCE (BridgeStan) validity —
# never by RK finiteness. Selecting on the reference means a point where Stan is
# finite but RK is not is NOT silently discarded: it survives into the gate loop,
# where kb(q) is asserted finite and hard-fails. (This rejects only points where
# the reference log-density itself overflows, independent of the RK graph.)
function reference_points(sm, dim, seed, name, npts, scale)
    rng = Xoshiro(seed + sum(codeunits(name)))
    pts = Vector{Vector{Float64}}()
    tries = 0
    while length(pts) < npts && tries < 40_000
        tries += 1
        q = scale .* randn(rng, dim)
        v = try sval(sm, q) catch; NaN end
        isfinite(v) && push!(pts, q)
    end
    length(pts) == npts || error("$name: only $(length(pts))/$npts reference-finite probes")
    pts
end

function gate(name; graph, have, bind, scale = 0.3, npts = NATIVE_PROBE_COUNT,
              seed = 468, do_reactant = true)
    println("\n########## $name ##########"); flush(stdout)
    sm, post = bridge(name, seed)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    bound = bind(data)
    kb = prepare(graph; have, want = :posterior, bound)
    dim = Int(BridgeStan.param_unc_num(sm))
    pts = reference_points(sm, dim, seed, name, npts, scale)
    @assert length(pts) == NATIVE_PROBE_COUNT
    println("  raw bound ports: $(keys(bound))  dim=$dim  native_probes=$(length(pts)) reactant_probes=$REACTANT_PROBE_COUNT(first point)"); flush(stdout)

    # ---- axes 1 & 2: native value + native plain-Enzyme gradient vs Stan ----
    prep = prepare_ad(kb, AE, pts[1]; active = :unconstrained)
    max_v = 0.0; max_g = 0.0
    for q in pts
        vr = kb(q); vs = sval(sm, q)
        rv = _relv(vr, vs)
        @assert isfinite(vr) "$name: native value not finite"
        @assert rv < VALUE_TOL "$name: native value rel=$rv ≥ $VALUE_TOL"
        gr = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        gs = sgrad(sm, q)
        rg = relerr(gr, gs)
        @assert all(isfinite, gr) "$name: native gradient not finite"
        @assert rg < GRAD_TOL "$name: native gradient rel=$rg ≥ $GRAD_TOL"
        max_v = max(max_v, rv); max_g = max(max_g, rg)
    end
    println("  [1] native value   max_rel=$(round(max_v; sigdigits = 4))  (< $VALUE_TOL) PASS")
    println("  [2] native grad    max_rel=$(round(max_g; sigdigits = 4))  (< $GRAD_TOL) PASS"); flush(stdout)

    do_reactant && _reactant_axes(name, kb, prep, sm, pts)
    _RAN[] += 1
    return
end

# The Reactant axes use `Reactant.@compile`, a macro expanded when the code
# containing it is LOWERED. A plain `if DO_REACTANT … end` guard does NOT help:
# a top-level `if` is macro-expanded (both branches) before it is evaluated, so
# the macro would still expand — and abort — in the Reactant-UNLOADED process.
# The only sound guard is a separate file INCLUDED ONLY after `import Reactant`,
# whose body is lowered solely at include time (which happens only when Reactant
# is loaded). Under BATCH_LATENT_REACTANT=0 the no-op method below stands in.
if DO_REACTANT
    include(joinpath(@__DIR__, "batch_latent_gate_reactant.jl"))
else
    _reactant_axes(args...) = nothing
end

const KNOWN_MODELS = ("glmm1", "bym2", "bones", "occ")
const SEL = strip.(split(get(ENV, "BATCH_LATENT_MODELS", join(KNOWN_MODELS, ',')), ','))
for s in SEL
    isempty(s) && error("BATCH_LATENT_MODELS has an empty entry (got $(repr(get(ENV, "BATCH_LATENT_MODELS", "")))).")
    s in KNOWN_MODELS ||
        error("BATCH_LATENT_MODELS: unknown model '$s' (known: $(join(KNOWN_MODELS, ", ")))")
end
isempty(SEL) && error("BATCH_LATENT_MODELS selected no models")
_want(m) = m in SEL
const _RAN = Ref(0)

# GLMM1 — bind ONLY raw obs counts + site indices; the site-effect gather
# alpha[obssite] (the year index drops out) is in-graph.
_want("glmm1") && gate("GLMM_data-GLMM1_model";
    graph = PE.GLMM1ModelExample.build_glmm1_model_graph(),
    have = (:unconstrained, :obs, :obssite),
    bind = d -> (obs = Int.(d["obs"]), obssite = Int.(d["obssite"])),
    scale = 0.3, do_reactant = DO_REACTANT)

# bym2_offset_only — bind ONLY raw adjacency, counts, exposure, scaling; log_E,
# convolved_re, and the ICAR phi[node1]/phi[node2] gathers are in-graph.
_want("bym2") && gate("traffic_accident_nyc-bym2_offset_only";
    graph = PE.Bym2OffsetOnlyExample.build_bym2_offset_only_graph(),
    have = (:unconstrained, :node1, :node2, :y, :E, :scaling_factor),
    bind = d -> (node1 = Int.(d["node1"]), node2 = Int.(d["node2"]), y = Int.(d["y"]),
                 E = Float64.(d["E"]), scaling_factor = Float64(d["scaling_factor"])),
    scale = 0.3, do_reactant = DO_REACTANT)

# bones — bind ONLY the raw grade/gamma/delta/ncat block; grid coordinates, the
# ragged cut selection, gathers and masks are all derived in-graph.
_want("bones") && gate("bones_data-bones_model";
    graph = PE.BonesModelExample.build_bones_model_graph(),
    have = (:unconstrained, :GRADE, :GAMMA, :DELTA, :NCAT),
    bind = d -> PE.BonesModelExample.bones_inputs(d),
    scale = 0.5, do_reactant = DO_REACTANT)

# multi_occupancy — bind ONLY the raw n×J detection matrix + dims; the flat
# counts, species coordinate, and binomial normalizer are all in-graph.
_want("occ") && gate("butterfly-multi_occupancy";
    graph = PE.MultiOccupancyExample.build_multi_occupancy_graph(),
    have = (:unconstrained, :X, :n, :J, :K),
    bind = d -> PE.MultiOccupancyExample.multi_occupancy_inputs(d),
    scale = 0.3, do_reactant = DO_REACTANT)

@assert _RAN[] == length(SEL) "$(_RAN[]) gate(s) ran but $(length(SEL)) were selected"
@assert _RAN[] > 0 "no gates ran"
if !DO_REACTANT
    @assert !_reactant_loaded() "BATCH_LATENT_REACTANT=0 native phase finished with Reactant UNLOADED, but it was loaded after the gates"
end
const _PHASE = DO_REACTANT ?
    "all FOUR parts (native value+grad, Reactant primal+grad)" :
    "the TWO native parts (Reactant UNLOADED; parts 3-4 deferred to a BATCH_LATENT_REACTANT=1 run)"
println("\n== batch_latent_gate: $(_RAN[]) model(s) [$(join(SEL, ", "))] PASSED $_PHASE =="); flush(stdout)
