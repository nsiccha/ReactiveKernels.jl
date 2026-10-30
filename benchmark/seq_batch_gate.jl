#!/usr/bin/env julia

# seq_batch_gate.jl — the two-part correctness gate for the sequential
# posteriordb `@kernel` PPL translations (hmm_example, hmm_gaussian, garch11,
# iohmm_reg). Each model is checked, on its FULL real posteriordb data, against
# the ACTUAL reference `.stan` via BridgeStan (propto = false, jacobian = true)
# along two independent axes:
#
#   1. NATIVE VALUE      — RK primal log-density vs BridgeStan log_density
#                          (relative error < 1e-6).
#   2. NATIVE GRADIENT   — RK gradient via plain DifferentiationInterface +
#                          ordinary Enzyme reverse (no function annotation,
#                          no FD substitution) vs BridgeStan
#                          log_density_gradient (relative error < 1e-3),
#                          finite at every probe.
#
# Every axis is a hard `@assert`, so a regression exits nonzero. Graphs are built
# from the installed package's `build_*_graph` (its templates are evaluated at
# module-load, a world boundary), then `prepare`d here at top level, so no
# world-age hazard arises. The all-bound query below keeps the raw-data entry
# partial-evaluated.
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Enzyme — all in
# the all80 environment):
#
#   julia --project=benchmark/all80-env benchmark/seq_batch_gate.jl
#
# Optional: SEQ_MODELS=garch11,hmm_example restricts the set.

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse)

const VALUE_TOL   = 1e-6      # native value vs Stan
const GRAD_TOL    = 1e-3      # native plain-Enzyme gradient vs Stan

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

    return
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
