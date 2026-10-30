#!/usr/bin/env julia

# batch_dynamics_gate.jl — correctness gate for the three adaptive-ODE
# posteriordb `@kernel` PPL translations (lotka_volterra, sir,
# soil_incubation; lotka back in after core fix 96eaf14d).
# Each model is checked, on its FULL real posteriordb
# data, against the ACTUAL reference `.stan` via BridgeStan (propto = false,
# jacobian = true) along two axes:
#
#   1. NATIVE VALUE    — RK primal log-density vs BridgeStan log_density.
#   2. NATIVE GRADIENT — RK gradient via plain DifferentiationInterface +
#                        Enzyme reverse vs BridgeStan log_density_gradient.
#
# Solver-policy basis: user-resolved decision 008vhy5 authorizes the Julia
# DP5/FBDF-with-explicit-controls opaque nodes as the delivery vehicle —
# these are NOT Stan's Boost dopri5 / CVODES integrators, so axes 1-2 REPORT
# measured RK-vs-Stan discrepancies rather than asserting the static-graph
# family's 1e-6/1e-3 tolerances. What IS hard-asserted on every axis: RK
# finiteness wherever Stan is finite (never filter an RK-nonfinite point
# where Stan is finite) and finite gradients.
#
# Graphs are built from the installed package's `build_*_graph` and `prepare`d
# here at top level (no world-age hazard). Only the raw data ports each model
# declares are bound.
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Enzyme — all
# in the all80 environment):
#   julia --project=benchmark/all80-env benchmark/batch_dynamics_gate.jl
#
# Optional: DYNAMICS_MODELS=sir,soil restricts the set.

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse)
const AD_LABEL = "AutoEnzyme(mode=Enzyme.Reverse) [default annotation]"

# Finiteness is the hard gate on axes 1-2; discrepancies are REPORTED
# (max over probes printed) under the 008vhy5 exception, not asserted
# against the static-graph family's tolerances.

println("== batch_dynamics_gate provenance ==")
println("  Julia ", VERSION, " | BridgeStan ", pkgversion(BridgeStan),
        " (Stan 2.39.0) | Enzyme ", pkgversion(Enzyme),
        " | DifferentiationInterface ", pkgversion(DifferentiationInterface))
println("  AD backend: ", AD_LABEL)
println("  reference oracle: BridgeStan.log_density(propto=false, jacobian=true) on the actual .stan")
println("  solver policy: user-resolved 008vhy5 exception (Julia DP5/FBDF nodes, explicit controls)")
println("  native axes use scale*randn probes selected by REFERENCE (Stan) finiteness.")
flush(stdout)

sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_relv(a, b) = abs(a - b) / max(abs(b), 1.0)

function bridge(name, seed)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    sp = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan"))
    sm = BridgeStan.StanModel(sp, PosteriorDB.load(PosteriorDB.dataset(post), String), seed)
    sm, post
end

function reference_points(sm, dim, seed, name, npts, scale, center = zeros(dim))
    length(center) == dim || error("$name: center dim $(length(center)) != $dim")
    rng = Xoshiro(seed + sum(codeunits(name)))
    pts = Vector{Vector{Float64}}()
    tries = 0
    while length(pts) < npts && tries < 20_000
        tries += 1
        q = center .+ scale .* randn(rng, dim)
        v = try sval(sm, q) catch; NaN end
        isfinite(v) && push!(pts, q)
    end
    length(pts) == npts || error("$name: only $(length(pts))/$npts reference-finite probes")
    pts
end

function gate(name; graph, have, bind, scale = 0.3, npts = 4, seed = 468,
              center = nothing, do_grad = true)
    println("\n########## $name ##########"); flush(stdout)
    sm, post = bridge(name, seed)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    fullbound = bind(data)
    kb = prepare(graph; have, want = :posterior, bound = fullbound)
    dim = Int(BridgeStan.param_unc_num(sm))
    c = isnothing(center) ? zeros(dim) : Float64.(center)
    pts = reference_points(sm, dim, seed, name, npts, scale, c)
    println("  dim=$dim  seed=$seed  scale=$scale  center=$c  native probes=$npts (reference-finite)  AD=$AD_LABEL"); flush(stdout)

    # ---- axis 1: native value vs Stan (all models) ----
    # ---- axis 2: native plain-Enzyme gradient vs Stan (skipped per model
    # via do_grad=false where the boundary marks gradients unsupported) ----
    prep = do_grad ? prepare_ad(kb, AE, pts[1]; active = :unconstrained) : nothing
    max_v = 0.0; max_g = 0.0
    for (i, q) in enumerate(pts)
        vr = kb(q); vs = sval(sm, q)
        @assert isfinite(vr) "$name pt$i: native value not finite where Stan is finite (Stan=$vs)"
        rv = _relv(vr, vs)
        if do_grad
            gr = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
            gs = sgrad(sm, q)
            @assert all(isfinite, gr) "$name pt$i: native gradient not finite"
            rg = relerr(gr, gs)
            max_g = max(max_g, rg)
            println("  pt$i native=$vr stan=$vs val_rel=$rv grad_rel=$rg"); flush(stdout)
        else
            println("  pt$i native=$vr stan=$vs val_rel=$rv grad=SKIPPED-per-boundary"); flush(stdout)
        end
        max_v = max(max_v, rv)
    end
    println("  [1] native value max_rel=$max_v over $npts probes (REPORTED under 008vhy5)")
    if do_grad
        println("  [2] native grad  max_rel=$max_g over $npts probes (REPORTED under 008vhy5)  [$AD_LABEL]"); flush(stdout)
    else
        println("  [2] native grad SKIPPED-per-boundary (ordinary-Reverse unsupported; see model boundary)"); flush(stdout)
    end

    _RAN[] += 1
    return
end

# ---- model registry -------------------------------------------------------
# The pharmacokinetic example and its scientific oracle now live in
# RKPPLBench/examples/posteriordb_pk/.
const KNOWN_MODELS = ("lotka", "sir", "soil")
const SEL = strip.(split(get(ENV, "DYNAMICS_MODELS", join(KNOWN_MODELS, ',')), ','))
for s in SEL
    isempty(s) && error("DYNAMICS_MODELS has an empty entry")
    s in KNOWN_MODELS || error("DYNAMICS_MODELS: unknown model '$s' (known: $(join(KNOWN_MODELS, ", ")))")
end
isempty(SEL) && error("DYNAMICS_MODELS selected no models")
_want(m) = m in SEL
const _RAN = Ref(0)

# lotka: reference-checked native value + ordinary-Reverse gradient, enabled
# by landed core fix 55da875/f8acaa1 (snag plain-enzyme-rev-3dc5d563: bound
# views externalized as owning copies for prepared AD).
_want("lotka") && gate("hudson_lynx_hare-lotka_volterra";
    graph = PE.LotkaVolterraExample.build_lotka_volterra_graph(),
    have = (:unconstrained, :ts, :y_init, :y),
    bind = d -> (ts = Float64.(d["ts"]), y_init = Float64.(d["y_init"]),
                 y = Float64.(d["y"])))

# sir: the posterior is concentrated away from the origin (0/20000
# reference-finite at scale 0.3 around zeros; 1739/2000 around the demo
# point, scan log kb-run-compact.K9aqhS). Probes center on the demo q with
# selection still purely by Stan finiteness — no RK-side filtering.
_want("sir") && gate("sir-sir";
    graph = PE.SIRExample.build_sir_graph(),
    have = (:unconstrained, :t, :y0, :stoi_hat, :B_hat),
    bind = d -> (t = Float64.(d["t"]), y0 = Float64.(d["y0"]),
                 stoi_hat = Int.(d["stoi_hat"]), B_hat = Float64.(d["B_hat"])),
    center = [log(0.1), log(0.1), log(1.0), log(1.0)])

_want("soil") && gate("soil_carbon-soil_incubation";
    graph = PE.SoilIncubationExample.build_soil_incubation_graph(),
    have = (:unconstrained, :ts, :total_c_t0, :eco2_mean),
    bind = d -> (ts = Float64.(d["ts"]),
                 total_c_t0 = Float64(d["totalC_t0"]),
                 eco2_mean = Float64.(d["eCO2mean"])))

@assert _RAN[] == length(SEL) "$(_RAN[]) gate(s) ran but $(length(SEL)) selected"
@assert _RAN[] > 0 "no gates ran"
println("\n== batch_dynamics_gate: $(_RAN[]) model(s) [$(join(SEL, ", "))] DONE the two native parts =="); flush(stdout)
