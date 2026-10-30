#!/usr/bin/env julia

# batch1_gate.jl — the two-part correctness gate for the batch-1 posteriordb
# `@kernel` PPL translations (diamonds, normal_mixture_k, dogs_nonhierarchical,
# logistic_regression_rhs). Each model is checked, on its FULL real posteriordb
# data, against the ACTUAL reference `.stan` via BridgeStan (propto = false,
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
# world-age hazard arises. In-graph data preprocessing is exercised by binding
# ONLY the raw data ports each model declares.
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Enzyme — all in
# the all80 environment):
#
#   julia --project=benchmark/all80-env benchmark/batch1_gate.jl
#
# Optional: BATCH1_MODELS=diamonds,dogs restricts the set.

using Random, LinearAlgebra
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPLExamples
using DifferentiationInterface

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

const VALUE_TOL  = 1e-6      # native value vs Stan
const GRAD_TOL   = 1e-3      # native plain-Enzyme gradient vs Stan

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

function gate(name; graph, have, bind, scale = 0.3, npts = 6, seed = 468,
              boundary_point = nothing)
    println("\n########## $name ##########"); flush(stdout)
    sm, post = bridge(name, seed)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    kb = prepare(graph; have, want = :posterior, bound = bind(data))
    dim = Int(BridgeStan.param_unc_num(sm))
    pts = reference_points(sm, dim, seed, name, npts, scale)

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

    # ---- support-boundary probe (where applicable) ----
    if boundary_point !== nothing
        qb = boundary_point(dim)
        vb = kb(qb); gbnd = ReactiveKernels.ad_value_and_gradient!(prep, similar(qb), qb)[2]
        @assert isfinite(vb) && all(isfinite, gbnd) "$name: boundary value/grad not finite"
        @assert _relv(vb, sval(sm, qb)) < VALUE_TOL "$name: boundary value ≠ Stan"
        @assert relerr(gbnd, sgrad(sm, qb)) < GRAD_TOL "$name: boundary gradient ≠ Stan"
        println("  [b] support boundary value+grad finite and match Stan PASS"); flush(stdout)
    end

    _RAN[] += 1
    return
end


const KNOWN_MODELS = ("diamonds", "mixture", "dogs", "logistic")
const SEL = strip.(split(get(ENV, "BATCH1_MODELS", join(KNOWN_MODELS, ',')), ','))
for s in SEL
    isempty(s) && error("BATCH1_MODELS has an empty entry (got $(repr(get(ENV, "BATCH1_MODELS", "")))).")
    s in KNOWN_MODELS ||
        error("BATCH1_MODELS: unknown model '$s' (known: $(join(KNOWN_MODELS, ", ")))")
end
isempty(SEL) && error("BATCH1_MODELS selected no models")
_want(m) = m in SEL
const _RAN = Ref(0)

# diamonds — bind the raw design matrix X (in-graph column-drop + centering) and
# the prior_only flag (in-graph data-directed likelihood inclusion); Y stays free.
_want("diamonds") && gate("diamonds-diamonds";
    graph = PE.DiamondsExample.build_diamonds_graph(),
    have = (:unconstrained, :X, :Y, :prior_only),
    bind = d -> (X = _mat(d["X"]), Y = Float64.(d["Y"]), prior_only = Int(d["prior_only"])),
    scale = 0.3)

# normal_mixture_k — natural K-dimensional, bind y and the component count K.
_want("mixture") && gate("normal_5-normal_mixture_k";
    graph = PE.NormalMixtureKExample.build_normal_mixture_k_graph(),
    have = (:unconstrained, :y, :K),
    bind = d -> (y = Float64.(d["y"]), K = Int(d["K"])),
    scale = 0.5)

# dogs_nonhierarchical — bind ONLY the raw y matrix; the running-count design
# (C = strict-upper-triangular, prev_shock = y·C, …) is derived in-graph. The
# boundary probe puts z = 0 so every t = 1 cell has p = 1 exactly (y[:,1] == 1),
# the case where a naive log(p) gradient would be NaN — the branch-selecting
# Bernoulli endpoint must keep it finite and matching Stan.
_want("dogs") && gate("dogs-dogs_nonhierarchical";
    graph = PE.DogsNonhierarchicalExample.build_dogs_nonhierarchical_graph(),
    have = (:unconstrained, :y),
    bind = d -> (y = Bool.(d["y"]),),
    scale = 0.3,
    boundary_point = dim -> vcat(-1.0, 0.5, log(0.5), log(0.4), 0.2, zeros(dim - 5)))

# logistic_regression_rhs — bind x, y and the horseshoe hyper-scalars.
_want("logistic") && gate("ovarian-logistic_regression_rhs";
    graph = PE.LogisticRegressionRHSExample.build_logistic_regression_rhs_graph(),
    have = (:unconstrained, :x, :y, :scale_icept, :scale_global, :nu_global,
            :nu_local, :slab_scale, :slab_df),
    bind = d -> (x = _mat(d["x"]), y = Bool.(d["y"]),
                 scale_icept = Float64(d["scale_icept"]), scale_global = Float64(d["scale_global"]),
                 nu_global = Float64(d["nu_global"]), nu_local = Float64(d["nu_local"]),
                 slab_scale = Float64(d["slab_scale"]), slab_df = Float64(d["slab_df"])),
    scale = 0.3)

@assert _RAN[] == length(SEL) "$(_RAN[]) gate(s) ran but $(length(SEL)) were selected"
@assert _RAN[] > 0 "no gates ran"
println("\n== batch1_gate: $(_RAN[]) model(s) [$(join(SEL, ", "))] PASSED the two native parts + boundary =="); flush(stdout)
