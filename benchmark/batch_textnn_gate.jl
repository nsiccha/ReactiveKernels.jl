#!/usr/bin/env julia

# batch_textnn_gate.jl — the two-part correctness gate for the text/neural
# posteriordb `@kernel` PPL translations (ldaK2, ldaK5, nn_rbm1bJ10,
# nn_rbm1bJ100). Each model is checked, on its FULL real posteriordb data,
# against the ACTUAL reference `.stan` via BridgeStan (propto = false,
# jacobian = true) along two independent axes:
#
#   1. NATIVE VALUE      — RK primal log-density vs BridgeStan log_density
#                          (relative error < 1e-6).
#   2. NATIVE GRADIENT   — RK gradient via plain DifferentiationInterface + Enzyme
#                          reverse vs BridgeStan log_density_gradient
#                          (relative error < 1e-3), finite at every probe.
#
# Every axis is a hard `@assert`, so a regression exits nonzero. Graphs are built
# from the installed package's `build_*_graph` (its templates are evaluated at
# module-load, a world boundary), then `prepare`d here at top level, so no
# world-age hazard arises. Only the raw data ports each model declares are bound;
# in-graph preprocessing (the simplex sum-to-zero basis, the per-word gather
# indices, the transformed-data prior scales) is exercised by binding raw data.
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Enzyme — all in
# the all80 environment):
#   julia --project=benchmark/all80-env benchmark/batch_textnn_gate.jl
#
# Optional: TEXTNN_MODELS=ldaK2,rbmJ10 restricts the set.

# Standard consumer acceptance uses the DEFAULT AutoEnzyme(mode=Enzyme.Reverse).
# TEXTNN_AD_CONST=1 selects the stricter function_annotation=Const variant (what
# the batch startup/acceptance drivers use); the two are labeled distinctly.
const AD_CONST = get(ENV, "TEXTNN_AD_CONST", "0") == "1"

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
const AE = AD_CONST ?
    AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const) :
    AutoEnzyme(mode = Enzyme.Reverse)
const AD_LABEL = AD_CONST ?
    "AutoEnzyme(mode=Enzyme.Reverse, function_annotation=Enzyme.Const)" :
    "AutoEnzyme(mode=Enzyme.Reverse) [default annotation]"

const VALUE_TOL   = 1e-6      # native value vs Stan
const GRAD_TOL    = 1e-3      # native plain-Enzyme gradient vs Stan

# Receipt provenance (correction 6): exact toolchain + reference identity.
println("== batch_textnn_gate provenance ==")
println("  Julia ", VERSION, " | BridgeStan ", pkgversion(BridgeStan),
        " (Stan 2.39.0) | Enzyme ", pkgversion(Enzyme),
        " | DifferentiationInterface ", pkgversion(DifferentiationInterface))
println("  AD backend: ", AD_LABEL)
println("  reference oracle: BridgeStan.log_density(propto=false, jacobian=true) on the actual .stan")
println("  probes are scale*randn draws selected by REFERENCE (Stan) finiteness;")
println("  point counts printed per model below.")
flush(stdout)

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

function gate(name; graph, have, bind, scale = 0.3, npts = 4, seed = 468)
    println("\n########## $name ##########"); flush(stdout)
    sm, post = bridge(name, seed)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    fullbound = bind(data)
    kb = prepare(graph; have, want = :posterior, bound = fullbound)
    dim = Int(BridgeStan.param_unc_num(sm))
    pts = reference_points(sm, dim, seed, name, npts, scale)
    println("  dim=$dim  seed=$seed  scale=$scale  native probes=$npts (reference-finite)  " *
            "AD=$AD_LABEL"); flush(stdout)

    # ---- axes 1 & 2: native value + native plain-Enzyme gradient vs Stan ----
    # (native axes bind ALL raw data — the faithful entry query.)
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
    println("  [1] native value max_rel=$(round(max_v; sigdigits = 4)) (< $VALUE_TOL) PASS over $npts probes")
    println("  [2] native grad  max_rel=$(round(max_g; sigdigits = 4)) (< $GRAD_TOL) PASS over $npts probes  [$AD_LABEL]"); flush(stdout)

    _RAN[] += 1
    return
end

# ---- model registry -------------------------------------------------------
_lda_have = (:unconstrained, :doc, :w, :alpha, :beta, :M)
_rbm_have = (:unconstrained, :x, :y, :K, :J)

const KNOWN_MODELS = ("ldaK2", "ldaK5", "rbmJ10", "rbmJ100")
const SEL = strip.(split(get(ENV, "TEXTNN_MODELS", join(KNOWN_MODELS, ',')), ','))
for s in SEL
    isempty(s) && error("TEXTNN_MODELS has an empty entry")
    s in KNOWN_MODELS || error("TEXTNN_MODELS: unknown model '$s' (known: $(join(KNOWN_MODELS, ", ")))")
end
isempty(SEL) && error("TEXTNN_MODELS selected no models")
_want(m) = m in SEL
const _RAN = Ref(0)

# ldaK2 — alpha, beta are the .stan transformed-data ones-vectors (bound here).
_want("ldaK2") && gate("three_men1-ldaK2";
    graph = PE.LDAExample.build_lda_graph(), have = _lda_have,
    bind = d -> (doc = Int.(d["doc"]), w = Int.(d["w"]),
                 alpha = ones(2), beta = ones(Int(d["V"])), M = Int(d["M"])),
    scale = 0.3)

# ldaK5 — alpha (5), beta (V) are data.
_want("ldaK5") && gate("prideprejudice_chapter-ldaK5";
    graph = PE.LDAExample.build_lda_graph(), have = _lda_have,
    bind = d -> (doc = Int.(d["doc"]), w = Int.(d["w"]),
                 alpha = Float64.(d["alpha"]), beta = Float64.(d["beta"]), M = Int(d["M"])),
    scale = 0.3)

# rbmJ10 — mnist_100, J = 10 (transformed-data hidden-unit count).
_want("rbmJ10") && gate("mnist_100-nn_rbm1bJ10";
    graph = PE.NNRBMExample.build_nn_rbm_graph(), have = _rbm_have,
    bind = d -> (x = _mat(d["x"]), y = Int.(d["y"]), K = Int(d["K"]), J = 10),
    scale = 0.3, npts = 3)

# rbmJ100 — full MNIST (N=60000), J = 100.
_want("rbmJ100") && gate("mnist-nn_rbm1bJ100";
    graph = PE.NNRBMExample.build_nn_rbm_graph(), have = _rbm_have,
    bind = d -> (x = _mat(d["x"]), y = Int.(d["y"]), K = Int(d["K"]), J = 100),
    scale = 0.3, npts = 2)

@assert _RAN[] == length(SEL) "$(_RAN[]) gate(s) ran but $(length(SEL)) selected"
@assert _RAN[] > 0 "no gates ran"
println("\n== batch_textnn_gate: $(_RAN[]) model(s) [$(join(SEL, ", "))] PASSED the two native parts =="); flush(stdout)
