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
#   3. REACTANT PRIMAL   — the Reactant-compiled primal vs the native primal
#                          (relative error < 1e-6).
#   4. REACTANT GRADIENT — the Reactant-compiled gradient vs BridgeStan
#                          (relative error < 2e-3), finite.
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
# ecdc0501 are separately bound through the same raw ports. Native axes (1-2)
# run for every selected posterior. Reactant axes (3-4) are compiled ONCE per
# DISTINCT dataset (its first selected posterior, whichever model name that
# is); later selected posteriors of the same dataset assert .stan byte equality
# and identical dataset with the covering entry and are explicitly labelled as
# NOT independently compiled — they never silently claim axis 3-4 execution.
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Reactant, Enzyme — all in
# the all80 environment):
#
#   julia --project=benchmark/all80-env benchmark/pandemic_gate.jl
#
# Optional: PANDEMIC_MODELS=ecdc0401_v2 restricts the set. Probe scope: 5
# reference-valid (Stan-finite) probes per posterior for native axes, 1 probe
# per compiled dataset for the Reactant axes.
#
# The Reactant axes (3, 4) and the two native axes (1, 2) MUST be certified in
# SEPARATE processes: `PANDEMIC_REACTANT=0` asserts Reactant is UNLOADED both
# BEFORE any package import and AFTER the native phase, so axes 1-2 certify
# plain-native execution with the Reactant extension genuinely absent;
# `PANDEMIC_REACTANT=1` imports Reactant and runs all four. Authoritative
# acceptance = one run of each.

const DO_REACTANT = get(ENV, "PANDEMIC_REACTANT", "1") == "1"

_reactant_loaded() = any(id -> id.name == "Reactant", keys(Base.loaded_modules))
if !DO_REACTANT
    @assert !_reactant_loaded() "PANDEMIC_REACTANT=0 native phase must START with Reactant UNLOADED (before any package import), but it is loaded"
end

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

if DO_REACTANT
    import Reactant
else
    @assert !_reactant_loaded() "PANDEMIC_REACTANT=0 native phase must keep Reactant UNLOADED after package imports"
end

const PE = ReactiveKernelsPPLExamples
# ORDINARY reverse-mode Enzyme: no function_annotation=Const, no
# runtime_activity, no priming. (Earlier exploratory receipts used
# function_annotation = Enzyme.Const; those are historical/config-labelled,
# not this gate's configuration.)
const AE = AutoEnzyme(mode = Enzyme.Reverse)

const VALUE_TOL   = 1e-6      # native value vs Stan
const GRAD_TOL    = 1e-3      # native plain-Enzyme gradient vs Stan
const RPRIMAL_TOL = 1e-6      # Reactant primal vs native
const RGRAD_TOL   = 2e-3      # Reactant gradient vs Stan

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
if DO_REACTANT
    println("  Reactant = ", Base.pkgversion(Reactant))
end
println("  AD configuration: ordinary AutoEnzyme(mode = Enzyme.Reverse); no annotation/runtime_activity/priming")
println("  probe scope: 5 reference-valid probes per posterior (native axes); 1 probe per compiled dataset (Reactant axes)")
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
const _REACTANT_DATASETS_RUN = Set{String}()
const _REACTANT_COVERED_BY = Dict{String,String}()

function gate(name; do_reactant, reactant_note)
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

    if do_reactant
        _reactant_axes(name, kb, prep, sm, pts)
    else
        println("  [3,4] Reactant axes NOT executed here — $reactant_note"); flush(stdout)
    end
    _NATIVE_RAN[] += 1
    return
end

# The Reactant axes use `Reactant.@compile`, a macro expanded when the code
# containing it is LOWERED. A plain `if DO_REACTANT … end` guard does NOT help:
# a top-level `if` is macro-expanded (both branches) before it is evaluated, so
# the macro would still expand — and abort — in the Reactant-UNLOADED process.
# The only sound guard is a separate file INCLUDED ONLY after `import Reactant`,
# whose body is lowered solely at include time (which happens only when Reactant
# is loaded). Under PANDEMIC_REACTANT=0 the no-op method below stands in and no
# Reactant macro or binding is ever referenced.
if DO_REACTANT
    include(joinpath(@__DIR__, "pandemic_gate_reactant.jl"))
else
    _reactant_axes(args...) = nothing
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

# Reactant compiles once per DISTINCT dataset, on that dataset's FIRST selected
# posterior (whichever model name that is). Every later selected posterior of
# the same dataset must prove source+data equivalence with the covering entry
# and is labelled as not independently compiled.
const _REACTANT_OWNER = Dict{String,String}()
for p in SELECTION
    get!(_REACTANT_OWNER, p.dataset) do; p.name; end
end
function _stan_bytes(name)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    read(PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan")))
end

for p in SELECTION
    owner = _REACTANT_OWNER[p.dataset]
    do_reactant = DO_REACTANT && p.name == owner
    if do_reactant
        note = "compiled for this dataset"
    else
        # Same dataset is guaranteed by the grouping key; the load-bearing
        # equivalence for sharing a compiled graph is the model source bytes.
        @assert _stan_bytes(owner) == _stan_bytes(p.name) "$(p.name): .stan bytes differ from covering entry $(owner)"
        note = "identical .stan bytes and dataset ($(p.dataset)), compiled as $(owner)"
    end
    gate(p.name; do_reactant, reactant_note = note)
end

@assert _NATIVE_RAN[] == length(SELECTION) "$(_NATIVE_RAN[]) native gate(s) ran but $(length(SELECTION)) were selected"
const _EXPECTED_REACTANT = DO_REACTANT ? length(_REACTANT_OWNER) : 0
@assert length(_REACTANT_DATASETS_RUN) == _EXPECTED_REACTANT "Reactant accounting mismatch: $(length(_REACTANT_DATASETS_RUN)) dataset compile(s) ran, expected $(_EXPECTED_REACTANT)"
const _PHASE = DO_REACTANT ?
    "all FOUR parts (native value+grad on every selected posterior; Reactant primal+grad compiled once per distinct dataset: $(join(sort(collect(_REACTANT_DATASETS_RUN)), ", ")))" :
    "the TWO native parts + boundary controls (Reactant UNLOADED before imports and after the phase; parts 3-4 deferred to a PANDEMIC_REACTANT=1 run)"
println("\n== pandemic_gate: $(length(SELECTION)) posterior(s) [$(join((p.key for p in SELECTION), ", "))] PASSED $_PHASE =="); flush(stdout)

if !DO_REACTANT
    @assert !_reactant_loaded() "PANDEMIC_REACTANT=0 native phase must END with Reactant still UNLOADED"
end
