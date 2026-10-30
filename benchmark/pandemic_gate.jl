#!/usr/bin/env julia

# pandemic_gate.jl — the correctness gate for the posteriordb `covid19imperial`
# `@kernel` translation (packages/ReactiveKernelsPPLExamples
# src/covid19imperial.jl). One data-generic graph is checked, on FULL real
# posteriordb data, against the ACTUAL reference `.stan` via BridgeStan
# (propto = false, jacobian = true):
#
#   1. NATIVE VALUE      — RK primal log-density vs BridgeStan log_density
#                          (relative error < 1e-6), finite at every probe.
#   2. NATIVE GRADIENT   — RK gradient via ORDINARY DifferentiationInterface +
#                          AutoEnzyme(mode = Enzyme.Reverse) — no
#                          function_annotation, runtime_activity, or priming
#                          substitution — vs BridgeStan log_density_gradient
#                          (relative error < 1e-3), finite at every probe.
#
# plus BOUNDARY CONTROLS on small synthetic datasets against actual Stan:
# early observations inside the imputation window (including day 1's
# E_deaths = 1e-15 * prediction[1] special case) and the empty renewal tail
# N2 = N0, each with value AND ordinary-gradient parity.
#
# Every axis is a hard `@assert`, so a regression exits nonzero. The graph
# binds ONLY the raw posteriordb arrays (X and pop bind as REAL — Stan declares
# them real — alongside the int arrays and the dimension scalars); every
# model-specific preprocessing step (covariate reshape, full observed-grid
# mask, deaths grid, count log-factorial) is a named in-graph node.
#
# The four posteriors of the pandemic family are covered: the bundled
# covid19imperial_v2.stan and covid19imperial_v3.stan are byte-identical, so
# the same graph serves both model names; the sibling datasets ecdc0401 and
# ecdc0501 are separately bound through the same raw ports. Both axes run for
# every selected posterior.
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Enzyme — all in the all80
# environment):
#
#   julia --project=benchmark/all80-env benchmark/pandemic_gate.jl
#
# Optional: PANDEMIC_MODELS=ecdc0401_v2 restricts the set. Probe scope: 5
# reference-valid (Stan-finite) probes per posterior.

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
# ORDINARY reverse-mode Enzyme: no function_annotation=Const, no
# runtime_activity, no priming. (Earlier exploratory receipts used
# function_annotation = Enzyme.Const; those are historical/config-labelled,
# not this gate's configuration.)
const AE = AutoEnzyme(mode = Enzyme.Reverse)

const VALUE_TOL   = 1e-6      # native value vs Stan
const GRAD_TOL    = 1e-3      # native plain-Enzyme gradient vs Stan

sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_relv(a, b) = abs(a - b) / max(abs(b), 1.0)

println("== pandemic_gate identity ==")
println("  worktree HEAD = ", try readchomp(`git rev-parse HEAD`) catch; "unknown" end)
println("  julia = ", VERSION, "  BridgeStan = ", Base.pkgversion(BridgeStan),
        "  PosteriorDB = ", Base.pkgversion(PosteriorDB))
println("  Enzyme = ", Base.pkgversion(Enzyme),
        "  DifferentiationInterface = ", Base.pkgversion(DifferentiationInterface))
println("  AD configuration: ordinary AutoEnzyme(mode = Enzyme.Reverse); no annotation/runtime_activity/priming")
println("  probe scope: 5 reference-valid probes per posterior")
flush(stdout)

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
    while length(pts) < npts && tries < 20_000
        tries += 1
        q = scale .* randn(rng, dim)
        v = try sval(sm, q) catch; NaN end
        isfinite(v) && push!(pts, q)
    end
    length(pts) == npts || error("$name: only $(length(pts))/$npts reference-finite probes")
    pts
end

_have() = (:unconstrained, :X, :EpidemicStart, :N, :deaths, :SI, :fmat, :pop,
           :M, :P, :N0, :N2)
_bind(d) = (X = Float64.(d["X"]), EpidemicStart = Int.(d["EpidemicStart"]),
            N = Int.(d["N"]), deaths = Int.(d["deaths"]), SI = Float64.(d["SI"]),
            fmat = Float64.(d["f"]), pop = Float64.(d["pop"]), M = Int(d["M"]),
            P = Int(d["P"]), N0 = Int(d["N0"]), N2 = Int(d["N2"]))

const _NATIVE_RAN = Ref(0)

function gate(name)
    println("\n########## $name ##########"); flush(stdout)
    sm, post = bridge(name, 468)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    kb = prepare(PE.Covid19ImperialExample.build_covid19imperial_graph();
                 have = _have(), want = :posterior, bound = _bind(data))
    dim = Int(BridgeStan.param_unc_num(sm))
    @assert dim == 51 "$name: expected 51 unconstrained parameters, got $dim"
    pts = reference_points(sm, dim, 468, name, 5, 0.1)

    # ---- axes 1 & 2: native value + native ordinary-Enzyme gradient vs Stan ----
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
    println("  [2] native grad    max_rel=$(round(max_g; sigdigits = 4))  (< $GRAD_TOL) PASS  [ordinary AutoEnzyme(Reverse)]"); flush(stdout)

    _NATIVE_RAN[] += 1
    return
end

# ---- small-shape boundary controls against ACTUAL Stan (synthetic data) ----
const _BC_STAN = PosteriorDB.path(PosteriorDB.implementation(
    PosteriorDB.model(PosteriorDB.posterior(PosteriorDB.database(),
        "ecdc0401-covid19imperial_v2")), "stan"))

function boundary_control(label; n0, n2, es, last, population, q)
    jrows = "[" * join(fill("[1]", n2), ",") * "]"
    frows = "[" * join(fill("[0.1]", n2), ",") * "]"
    xrows = "[[" * join(fill("[0]", n2), ",") * "]]"
    si = "[" * join(fill("0.1", n2), ",") * "]"
    json = "{\"M\":1,\"P\":1,\"N0\":$n0,\"N2\":$n2,\"N\":[$last],\"cases\":$jrows,\"deaths\":$jrows,\"f\":$frows,\"X\":$xrows,\"EpidemicStart\":[$es],\"pop\":[$population],\"SI\":$si}"
    sm = BridgeStan.StanModel(_BC_STAN, json, 468)
    vs, gs = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)
    @assert isfinite(vs) && all(isfinite, gs) "$label: reference control not finite"
    bound = (; X = zeros(Float64, 1, n2, 1), EpidemicStart = [es], N = [last],
             deaths = ones(Int, n2, 1), SI = fill(0.1, n2), fmat = fill(0.1, n2, 1),
             pop = [Float64(population)], M = 1, P = 1, N0 = n0, N2 = n2)
    kb = prepare(PE.Covid19ImperialExample.build_covid19imperial_graph();
                 have = _have(), want = :posterior, bound = bound)
    vr = kb(q)
    @assert isfinite(vr) "$label: RK value not finite"
    rv = _relv(vr, vs)
    @assert rv < VALUE_TOL "$label: RK value rel=$rv ≥ $VALUE_TOL (Stan=$vs RK=$vr)"
    prep = prepare_ad(kb, AE, q; active = :unconstrained)
    gr = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
    rg = relerr(gr, gs)
    @assert all(isfinite, gr) "$label: RK gradient not finite"
    @assert rg < GRAD_TOL "$label: RK gradient rel=$rg ≥ $GRAD_TOL"
    println("  [boundary] $label: value_rel=$(round(rv; sigdigits=4)) grad_rel=$(round(rg; sigdigits=4)) PASS"); flush(stdout)
end

println("\n== boundary controls (actual Stan, synthetic small shapes) =="); flush(stdout)
boundary_control("observations after N0"; n0 = 3, n2 = 6, es = 4, last = 6,
                 population = 10000, q = zeros(7))
boundary_control("early observations incl. day 1"; n0 = 3, n2 = 6, es = 1, last = 6,
                 population = 10000, q = zeros(7))
boundary_control("empty renewal tail N2 = N0"; n0 = 3, n2 = 3, es = 1, last = 3,
                 population = 10000, q = zeros(7))

# ---- prospective selection + accounting ----
const _POSTERIORS = (
    (key = "ecdc0401_v2", name = "ecdc0401-covid19imperial_v2", dataset = "ecdc0401"),
    (key = "ecdc0401_v3", name = "ecdc0401-covid19imperial_v3", dataset = "ecdc0401"),
    (key = "ecdc0501_v2", name = "ecdc0501-covid19imperial_v2", dataset = "ecdc0501"),
    (key = "ecdc0501_v3", name = "ecdc0501-covid19imperial_v3", dataset = "ecdc0501"),
)
const SEL = strip.(split(get(ENV, "PANDEMIC_MODELS",
                             join((p.key for p in _POSTERIORS), ',')), ','))
for s in SEL
    isempty(s) && error("PANDEMIC_MODELS has an empty entry (got $(repr(get(ENV, "PANDEMIC_MODELS", "")))).")
    any(p -> p.key == s, _POSTERIORS) ||
        error("PANDEMIC_MODELS: unknown model '$s' (known: $(join((p.key for p in _POSTERIORS), ", ")))")
end
isempty(SEL) && error("PANDEMIC_MODELS selected no models")
const SELECTION = [p for p in _POSTERIORS if p.key in SEL]
length(SELECTION) == length(SEL) ||
    error("selection bookkeeping mismatch: $(length(SELECTION)) resolved for $(length(SEL)) requested")

for p in SELECTION
    gate(p.name)
end

@assert _NATIVE_RAN[] == length(SELECTION) "$(_NATIVE_RAN[]) native gate(s) ran but $(length(SELECTION)) were selected"
println("\n== pandemic_gate: $(length(SELECTION)) posterior(s) [$(join((p.key for p in SELECTION), ", "))] PASSED the native value + gradient parts and boundary controls =="); flush(stdout)
