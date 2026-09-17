#!/usr/bin/env julia

# seq_batch_gate.jl — the four-part correctness gate for the sequential
# posteriordb `@kernel` PPL translations (hmm_example, hmm_gaussian, garch11,
# iohmm_reg). Each model is checked, on its FULL real posteriordb data, against
# the ACTUAL reference `.stan` via BridgeStan (propto = false, jacobian = true)
# along four independent axes:
#
#   1. NATIVE VALUE      — RK primal log-density vs BridgeStan log_density
#                          (relative error < 1e-6).
#   2. NATIVE GRADIENT   — RK gradient via plain DifferentiationInterface +
#                          ordinary Enzyme reverse (no Reactant, no function
#                          annotation, no FD substitution) vs BridgeStan
#                          log_density_gradient (relative error < 1e-3),
#                          finite at every probe.
#   3. REACTANT PRIMAL   — the Reactant-compiled primal vs the native primal
#                          (relative error < 1e-6).
#   4. REACTANT GRADIENT — the Reactant-compiled gradient vs BridgeStan
#                          (relative error < 2e-3), finite.
#
# Every axis is a hard `@assert`, so a regression exits nonzero. Graphs are built
# from the installed package's `build_*_graph` (its templates are evaluated at
# module-load, a world boundary), then `prepare`d here at top level, so no
# world-age hazard arises. The all-bound query below keeps the raw-data entry
# partial-evaluated; the traced raw-series query (which retains the scan's
# `stablehlo.while` for garch11/hmm_example/hmm_gaussian) is asserted separately
# in test/test_ppl_examples_reactant.jl. iohmm_reg's per-step scan inputs are
# K-vectors, so its compiled Reactant program is the correct unrolled form.
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Reactant, Enzyme — all in
# the all80 environment):
#
#   julia --project=benchmark/all80-env benchmark/seq_batch_gate.jl
#
# Optional: SEQ_MODELS=garch11,hmm_example restricts the set.
#
# The Reactant axes (3, 4) and the two native axes (1, 2) MUST be certified in
# SEPARATE processes: `SEQ_REACTANT=0` never imports Reactant and asserts it is
# not loaded, so axes 1 & 2 certify plain-native execution with the Reactant
# extension genuinely absent; `SEQ_REACTANT=1` imports Reactant and runs all
# four. Authoritative acceptance = one run of each. Because Reactant cannot be
# unloaded once imported, the split is a real load boundary, not a runtime flag.

const DO_REACTANT = get(ENV, "SEQ_REACTANT", "1") == "1"

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
    @assert !_reactant_loaded() "SEQ_REACTANT=0 native phase must run with Reactant UNLOADED, but it is loaded"
end

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse)

const VALUE_TOL   = 1e-6      # native value vs Stan
const GRAD_TOL    = 1e-3      # native plain-Enzyme gradient vs Stan
const RPRIMAL_TOL = 1e-6      # Reactant primal vs native
const RGRAD_TOL   = 2e-3      # Reactant gradient vs Stan

sval(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
sgrad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_relv(a, b) = abs(a - b) / max(abs(b), 1.0)
F(d, k) = Float64.(d[k])
_mat(x) = x isa AbstractMatrix ? Float64.(x) :
          reduce(vcat, [permutedims(Float64.(r)) for r in x])

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
    while length(pts) < npts && tries < 20_000
        tries += 1
        q = scale .* randn(rng, dim)
        v = try sval(sm, q) catch; NaN end
        isfinite(v) && push!(pts, q)
    end
    length(pts) == npts || error("$name: only $(length(pts))/$npts reference-finite probes")
    pts
end

function gate(name; graph, have, bind, scale = 0.3, npts = 6, seed = 468)
    println("\n########## $name ##########"); flush(stdout)
    tb = time()
    sm, post = bridge(name, seed)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    kb = prepare(graph; have, want = :posterior, bound = bind(data))
    dim = Int(BridgeStan.param_unc_num(sm))
    pts = reference_points(sm, dim, seed, name, npts, scale)
    println("  build+prepare: $(round(time()-tb;digits=1))s  dim=$dim  npts=$(length(pts))"); flush(stdout)

    prep = prepare_ad(kb, AE, pts[1]; active = :unconstrained)
    max_v = 0.0; max_g = 0.0
    for q in pts
        vr = kb(q); vs = sval(sm, q)
        rv = _relv(vr, vs)
        @assert isfinite(vr) "$name: native value not finite at a Stan-finite point"
        @assert rv < VALUE_TOL "$name: native value rel=$rv ≥ $VALUE_TOL"
        gr = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        gs = sgrad(sm, q)
        rg = relerr(gr, gs)
        @assert all(isfinite, gr) "$name: native gradient not finite"
        @assert rg < GRAD_TOL "$name: native gradient rel=$rg ≥ $GRAD_TOL"
        max_v = max(max_v, rv); max_g = max(max_g, rg)
    end
    println("  [1] native value   max_rel=$(round(max_v; sigdigits=4))  (< $VALUE_TOL) PASS")
    println("  [2] native grad    max_rel=$(round(max_g; sigdigits=4))  (< $GRAD_TOL) PASS"); flush(stdout)

    DO_REACTANT && _reactant_axes(name, kb, prep, sm, pts)
    return
end

# Reactant axes are macro-expanded (Reactant.@compile) only when Reactant is
# loaded, so they live in a file that is `include`d ONLY under SEQ_REACTANT=1. A
# plain `if DO_REACTANT` guard does NOT defer macro expansion of a top-level `if`.
if DO_REACTANT
    include(joinpath(@__DIR__, "seq_batch_gate_reactant.jl"))
else
    _reactant_axes(args...) = nothing
end

const KNOWN_MODELS = ("garch11", "hmm_example", "hmm_gaussian", "iohmm_reg")
const SEL = strip.(split(get(ENV, "SEQ_MODELS", join(KNOWN_MODELS, ',')), ','))
for s in SEL
    isempty(s) && error("SEQ_MODELS has an empty entry (got $(repr(get(ENV, "SEQ_MODELS", "")))).")
    s in KNOWN_MODELS || error("SEQ_MODELS: unknown model '$s' (known: $(join(KNOWN_MODELS, ", ")))")
end
isempty(SEL) && error("SEQ_MODELS selected no models")
_want(m) = m in SEL

_want("garch11") && gate("garch-garch11";
    graph = PE.GARCH11Example.build_garch11_graph(),
    have = (:unconstrained, :y, :sigma1),
    bind = d -> (y = F(d, "y"), sigma1 = Float64(d["sigma1"])),
    scale = 0.5)

_want("hmm_example") && gate("hmm_example-hmm_example";
    graph = PE.HmmExampleExample.build_hmm_example_graph(),
    have = (:unconstrained, :y, :K),
    bind = d -> (y = F(d, "y"), K = Int(d["K"])),
    scale = 0.5)

_want("hmm_gaussian") && gate("hmm_gaussian_simulated-hmm_gaussian";
    graph = PE.HmmGaussianExample.build_hmm_gaussian_graph(),
    have = (:unconstrained, :y, :K),
    bind = d -> (y = F(d, "y"), K = Int(d["K"])),
    scale = 0.3)

_want("iohmm_reg") && gate("iohmm_reg_simulated-iohmm_reg";
    graph = PE.IohmmRegExample.build_iohmm_reg_graph(),
    have = (:unconstrained, :y, :u, :K),
    bind = d -> (y = F(d, "y"), u = _mat(d["u"]), K = Int(d["K"])),
    scale = 0.3)

println("\n== seq_batch_gate: all requested models PASSED all requested axes =="); flush(stdout)
