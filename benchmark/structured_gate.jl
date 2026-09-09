#!/usr/bin/env julia

# structured_gate.jl — the four-part correctness gate for the final structured
# posteriordb `@kernel` PPL translations (hmm_drive_1 first; the remaining
# kronecker_gp / grsm_latent_reg_irt / hmm_drive_0 slots are added as their
# modules land). Each model is checked, on its FULL real posteriordb data,
# against the ACTUAL reference `.stan` via BridgeStan (propto = false,
# jacobian = true) along four independent axes:
#
#   1. NATIVE VALUE      — RK primal log-density vs BridgeStan log_density
#                          (relative error < 1e-6).
#   2. NATIVE GRADIENT   — RK gradient via plain DifferentiationInterface +
#                          Enzyme reverse (no Reactant) vs BridgeStan
#                          (relative error < 1e-3), finite at every probe.
#   3. REACTANT PRIMAL   — the Reactant-compiled primal vs the native primal
#                          (relative error < 1e-6).
#   4. REACTANT GRADIENT — the Reactant-compiled gradient vs BridgeStan
#                          (relative error < 2e-3), finite.
#
# A model axis with a DOCUMENTED, snagged limitation is not silently skipped:
# the gate HARD-ASSERTS the documented failure signature itself (so a different
# failure, or an unnoticed fix, is loud) and, where a correct alternative mode
# exists, asserts that mode against Stan instead of leaving the math unproven.
# Current documented gap (snag scan-prior-enzym-d67d4ac1): hmm_drive_1's
# authored-scan forward filter combined with its >=4 prior endpoint terms
# trips Enzyme's STATIC activity analysis in BOTH gradient axes; the
# runtime-activity gradient matches Stan, and both primal axes pass.
#
# Run (BridgeStan 2.9 / Stan 2.39, PosteriorDB, Reactant, Enzyme — all in
# the all80 environment):
#
#   julia --project=benchmark/all80-env benchmark/structured_gate.jl
#
# Optional: STRUCTURED_MODELS=hmm_drive_1 restricts the set.
#
# The Reactant axes (3, 4) and the two native axes (1, 2) MUST be certified in
# SEPARATE processes: `STRUCTURED_REACTANT=0` never imports Reactant and asserts
# it is not loaded; `STRUCTURED_REACTANT=1` imports Reactant and runs all four.
# Authoritative acceptance = one run of each.

const DO_REACTANT = get(ENV, "STRUCTURED_REACTANT", "1") == "1"

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
    @assert !_reactant_loaded() "STRUCTURED_REACTANT=0 native phase must run with Reactant UNLOADED, but it is loaded"
end

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
const AE_RTA = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                          function_annotation = Enzyme.Const)

const VALUE_TOL   = 1e-6   # native value vs Stan
const GRAD_TOL    = 1e-3   # native gradient vs Stan
const RPRIMAL_TOL = 1e-6   # Reactant primal vs native
const RGRAD_TOL   = 2e-3   # Reactant gradient vs Stan

sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_relv(a, b) = abs(a - b) / max(abs(b), 1.0)
_mat(x) = x isa AbstractMatrix ? Float64.(x) :
          reduce(vcat, [permutedims(Float64.(r)) for r in x])

function bridge(name, seed)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    sp = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan"))
    sm = BridgeStan.StanModel(sp, PosteriorDB.load(PosteriorDB.dataset(post), String), seed)
    sm, post
end

function reference_points(sm, dim, seed, name, npts, scale)
    rng = Xoshiro(seed + sum(codeunits(name)))
    pts = Vector{Vector{Float64}}()
    tries = 0
    while length(pts) < npts && tries < 20_000
        tries += 1
        q = scale .* randn(rng, dim)
        v = try sval(sm, q) catch; NaN end
        isfinite(v) && push!(pts, q)
    end
    length(pts) == npts || error("$name: only $(length(pts))/$npts reference-finite probes")
    pts
end

# Assert the DOCUMENTED failure signature — never a silent skip. A different
# error (regression) or success (the snag was fixed) is loud and actionable.
_assert_documented_gap(name, axis, thunk, needle) = begin
    err = try
        thunk()
        nothing
    catch e
        sprint(showerror, e)
    end
    @assert err !== nothing "$name [$axis]: documented gap no longer reproduces (snag fixed?) — promote this axis to a full assertion"
    @assert occursin(needle, err) "$name [$axis]: expected documented $needle, got: $(first(err, 300))"
    println("  [$axis] DOCUMENTED GAP reproduces exactly ($needle) — see snag scan-prior-enzym-d67d4ac1"); flush(stdout)
end

function gate(name; graph, have, bind, scale = 0.3, npts = 6, seed = 468,
              do_reactant = true, grad_gap = false, boundary_point = nothing)
    println("\n########## $name ##########"); flush(stdout)
    sm, post = bridge(name, seed)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    kb = prepare(graph; have, want = :posterior, bound = bind(data))
    dim = Int(BridgeStan.param_unc_num(sm))
    pts = reference_points(sm, dim, seed, name, npts, scale)

    # ---- axis 1: native value vs Stan (always a hard assertion) ----
    max_v = 0.0
    for q in pts
        vr = kb(q); vs = sval(sm, q)
        rv = _relv(vr, vs)
        @assert isfinite(vr) "$name: native value not finite"
        @assert rv < VALUE_TOL "$name: native value rel=$rv ≥ $VALUE_TOL"
        max_v = max(max_v, rv)
    end
    println("  [1] native value   max_rel=$(round(max_v; sigdigits = 4))  (< $VALUE_TOL) PASS"); flush(stdout)

    # ---- axis 2: native gradient vs Stan ----
    prep = prepare_ad(kb, AE, pts[1]; active = :unconstrained)
    if grad_gap
        _assert_documented_gap(name, 2,
            () -> ReactiveKernels.ad_value_and_gradient!(prep, similar(pts[1]), pts[1])[2],
            "EnzymeRuntimeActivityError")
        # The gradient MATH is still proven: runtime-activity Enzyme matches
        # Stan at the same tolerance the plain axis would have used.
        prep_rta = prepare_ad(kb, AE_RTA, pts[1]; active = :unconstrained)
        max_g = 0.0
        for q in pts
            gr = ReactiveKernels.ad_value_and_gradient!(prep_rta, similar(q), q)[2]
            gs = sgrad(sm, q)
            rg = relerr(gr, gs)
            @assert all(isfinite, gr) "$name: runtime-activity gradient not finite"
            @assert rg < GRAD_TOL "$name: runtime-activity gradient rel=$rg ≥ $GRAD_TOL"
            max_g = max(max_g, rg)
        end
        println("  [2r] runtime-activity grad max_rel=$(round(max_g; sigdigits = 4))  (< $GRAD_TOL) PASS (diagnostic math check)"); flush(stdout)
    else
        max_g = 0.0
        for q in pts
            gr = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
            gs = sgrad(sm, q)
            rg = relerr(gr, gs)
            @assert all(isfinite, gr) "$name: native gradient not finite"
            @assert rg < GRAD_TOL "$name: native gradient rel=$rg ≥ $GRAD_TOL"
            max_g = max(max_g, rg)
        end
        println("  [2] native grad    max_rel=$(round(max_g; sigdigits = 4))  (< $GRAD_TOL) PASS"); flush(stdout)
    end

    # ---- support-boundary probe (where applicable) ----
    if boundary_point !== nothing && !grad_gap
        qb = boundary_point(dim)
        vb = kb(qb); gbnd = ReactiveKernels.ad_value_and_gradient!(prep, similar(qb), qb)[2]
        @assert isfinite(vb) && all(isfinite, gbnd) "$name: boundary value/grad not finite"
        @assert _relv(vb, sval(sm, qb)) < VALUE_TOL "$name: boundary value ≠ Stan"
        @assert relerr(gbnd, sgrad(sm, qb)) < GRAD_TOL "$name: boundary gradient ≠ Stan"
        println("  [b] support boundary value+grad finite and match Stan PASS"); flush(stdout)
    end

    do_reactant && _reactant_axes(name, kb, prep, sm, pts)
    _RAN[] += 1
    return
end

if DO_REACTANT
    include(joinpath(@__DIR__, "structured_gate_reactant.jl"))
else
    _reactant_axes(args...; kwargs...) = nothing
end

const KNOWN_MODELS = ("hmm_drive_1",)
const SEL = strip.(split(get(ENV, "STRUCTURED_MODELS", join(KNOWN_MODELS, ',')), ','))
for s in SEL
    isempty(s) && error("STRUCTURED_MODELS has an empty entry (got $(repr(get(ENV, "STRUCTURED_MODELS", "")))).")
    s in KNOWN_MODELS ||
        error("STRUCTURED_MODELS: unknown model '$s' (known: $(join(KNOWN_MODELS, ", ")))")
end
isempty(SEL) && error("STRUCTURED_MODELS selected no models")
_want(m) = m in SEL
const _RAN = Ref(0)

# hmm_drive_1 — bind ONLY the raw data (u, v streams; the alpha transit-prior
# rows; the fixed emission scales tau, rho). The forward algorithm is authored
# with the vector-carry `scan`; both gradient axes carry the documented
# static-activity gap (snag scan-prior-enzym-d67d4ac1) and are pinned by
# _assert_documented_gap, while the runtime-activity gradient proves the math.
_want("hmm_drive_1") && gate("bball_drive_event_1-hmm_drive_1";
    graph = PE.HmmDrive1Example.build_hmm_drive_1_graph(),
    have = (:unconstrained, :u, :v, :alpha, :tau, :rho),
    bind = d -> (u = Float64.(d["u"]), v = Float64.(d["v"]), alpha = _mat(d["alpha"]),
                 tau = Float64(d["tau"]), rho = Float64(d["rho"])),
    scale = 0.5, do_reactant = DO_REACTANT, grad_gap = true)

@assert _RAN[] > 0 "structured gate ran 0 models"
println("\nstructured gate done: $(_RAN[]) model(s) certified"); flush(stdout)
