#!/usr/bin/env julia

# forecast_batch_gate.jl — the four-part correctness gate for the posteriordb
# FORECASTING batch `@kernel` PPL translations (losscurve_sislob, accel_splines,
# state_space_stochastic_level_stochastic_seasonal, prophet). Each model is
# checked, on its FULL real posteriordb data, against the ACTUAL reference `.stan`
# via BridgeStan (propto = false, jacobian = true) along four independent axes:
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
# Every axis is a hard `@assert`, so a regression exits nonzero. Graphs are built
# from the REAL installed package's `build_*_graph` (each model's template is
# evaluated MODEL-ONLY at module-load, a world boundary), then `prepare`d here at
# top level (build->prepare->execute in one ordinary function per model), so no
# world-age hazard arises. In-graph data preprocessing is exercised by binding
# ONLY the raw data ports each model declares. This is the PUBLIC-package AD gate;
# it is distinct from the package's committed native-oracle regression tests
# (`test/runtests.jl`) and from the model-only import sentinel
# (`test/test_startup_initialization.jl`).
#
# Run (needs BridgeStan 2.9 / Stan 2.39, PosteriorDB, Reactant, Enzyme, DI — all
# in the all80 environment):
#
#   julia --project=benchmark/all80-env benchmark/forecast_batch_gate.jl
#
# Optional: FORECAST_MODELS=losscurve,prophet restricts the set.
#
# The Reactant axes (3, 4) and the two native axes (1, 2) MUST be certified in
# SEPARATE processes: `FORECAST_REACTANT=0` never imports Reactant and asserts it
# is not loaded, so axes 1 & 2 certify plain-native execution with the Reactant
# extension genuinely absent; `FORECAST_REACTANT=1` imports Reactant and runs all
# four. Authoritative acceptance = one run of each. Because Reactant cannot be
# unloaded once imported, the split is a real load boundary, not a runtime flag.

const DO_REACTANT = get(ENV, "FORECAST_REACTANT", "1") == "1"

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
    @assert !_reactant_loaded() "FORECAST_REACTANT=0 native phase must run with Reactant UNLOADED, but it is loaded"
end

const PE = ReactiveKernelsPPLExamples
const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

const VALUE_TOL   = 1e-6     # native value vs Stan
const GRAD_TOL    = 1e-3     # native plain-Enzyme gradient vs Stan
const RPRIMAL_TOL = 1e-6     # Reactant primal vs native
const RGRAD_TOL   = 2e-3     # Reactant gradient vs Stan

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

# ALTERNATE-FLAG Stan oracle: instantiate the SAME `.stan` with a scalar data
# flag flipped (e.g. `prior_only` 0→1, `growthmodel_id` 1→0). BridgeStan then
# evaluates the alternate model MODE, giving an independent Stan oracle for the
# eager-ifelse branch the real posterior does not select. The flag is a top-level
# JSON scalar, so a bounded regex replacement is a reliable edit of the data.
function bridge_altflag(name, flag_regex::Regex, flag_replacement::AbstractString, seed)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    sp = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan"))
    orig = PosteriorDB.load(PosteriorDB.dataset(post), String)
    modified = replace(orig, flag_regex => flag_replacement)
    modified == orig && error("$name: alt-flag regex $flag_regex changed nothing in the data JSON")
    BridgeStan.StanModel(sp, modified, seed), post
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
              do_reactant = true, boundary_point = nothing, altflag = nothing)
    println("\n########## $name ##########"); flush(stdout)
    if altflag === nothing
        sm, post = bridge(name, seed)
    else
        sm, post = bridge_altflag(name, altflag.regex, altflag.replacement, seed)
        println("  ALT-FLAG: $(altflag.regex) => $(altflag.replacement); RK bound $(altflag.bind)"); flush(stdout)
    end
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    bound = bind(data)
    altflag === nothing || (bound = merge(bound, altflag.bind))   # RK uses the flipped flag too
    kb = prepare(graph; have, want = :posterior, bound = bound)
    dim = Int(BridgeStan.param_unc_num(sm))
    println("  q identity: dim=$dim scale=$scale seed=$seed npts=$npts"); flush(stdout)
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
# is loaded). Under FORECAST_REACTANT=0 the no-op method below stands in and no
# Reactant macro or binding is ever referenced.
if DO_REACTANT
    include(joinpath(@__DIR__, "forecast_batch_gate_reactant.jl"))
else
    _reactant_axes(args...) = nothing
end

# None of the four forecasting models has a reachable support boundary in the
# unconstrained parameterization: every constraining transform (exp for
# positivity, the data-bounded logit level, the positive_ordered cumulative-exp
# scales) maps every finite unconstrained input to a strictly-interior
# constrained point, so there is no reference-finite / RK-nonfinite point to
# force through the boundary probe. Reference-validity probe selection still
# guards the general case.

const KNOWN_MODELS = ("losscurve", "accel", "state_space", "prophet",
                      "losscurve_altflag", "accel_altflag")
const SEL = strip.(split(get(ENV, "FORECAST_MODELS", join(KNOWN_MODELS, ',')), ','))
for s in SEL
    isempty(s) && error("FORECAST_MODELS has an empty entry (got $(repr(get(ENV, "FORECAST_MODELS", "")))).")
    s in KNOWN_MODELS ||
        error("FORECAST_MODELS: unknown model '$s' (known: $(join(KNOWN_MODELS, ", ")))")
end
isempty(SEL) && error("FORECAST_MODELS selected no models")
_want(m) = m in SEL
const _RAN = Ref(0)

# losscurve_sislob — hierarchical loss-development curve; bind the raw cohort/time
# indices + the growthmodel_id flag (in-graph flag-selected growth factor) + the
# premium/loss/t_value data; the unconstrained parameter vector stays free.
_want("losscurve") && gate("loss_curves-losscurve_sislob";
    graph = PE.LosscurveSislobExample.build_losscurve_sislob_graph(),
    have = (:unconstrained, :growthmodel_id, :cohort_id, :t_idx, :t_value, :premium, :loss),
    bind = d -> (growthmodel_id = Int(d["growthmodel_id"]), cohort_id = Int.(d["cohort_id"]),
                 t_idx = Int.(d["t_idx"]), t_value = Float64.(d["t_value"]),
                 premium = Float64.(d["premium"]), loss = Float64.(d["loss"])),
    scale = 0.3, do_reactant = DO_REACTANT)

# accel_splines — brms penalized-spline regression; bind the response + the four
# design/basis matrices (in-graph matrix-vector products) + the prior_only flag.
_want("accel") && gate("mcycle_splines-accel_splines";
    graph = PE.AccelSplinesExample.build_accel_splines_graph(),
    have = (:unconstrained, :Y, :Xs, :Zs_1_1, :Xs_sigma, :Zs_sigma_1_1, :prior_only),
    bind = d -> (Y = Float64.(d["Y"]), Xs = _mat(d["Xs"]), Zs_1_1 = _mat(d["Zs_1_1"]),
                 Xs_sigma = _mat(d["Xs_sigma"]), Zs_sigma_1_1 = _mat(d["Zs_sigma_1_1"]),
                 prior_only = Int(d["prior_only"])),
    scale = 0.3, do_reactant = DO_REACTANT)

# state_space — structural DLM; bind only the raw series y, x, w (the bounded
# level transform, positive_ordered scales, random-walk level and trailing-window
# seasonal are all derived in-graph).
_want("state_space") && gate("uk_drivers-state_space_stochastic_level_stochastic_seasonal";
    graph = PE.StateSpaceStochasticExample.build_state_space_stochastic_graph(),
    have = (:unconstrained, :y, :x, :w),
    bind = d -> (y = Float64.(d["y"]), x = Float64.(d["x"]), w = Float64.(d["w"])),
    scale = 0.2, do_reactant = DO_REACTANT)

# prophet — Facebook Prophet linear-trend forecasting; bind the raw time/
# changepoint/design data (the changepoint incidence matrix + trend + seasonality
# are derived in-graph). LINEAR trend only (build_prophet_graph validates the
# trend flag); the logistic mode is not part of the graph.
_want("prophet") && gate("rstan_downloads-prophet";
    graph = PE.ProphetExample.build_prophet_graph(),
    have = (:unconstrained, :t, :t_change, :X, :sigmas, :tau, :s_a, :s_m, :y),
    bind = d -> (t = Float64.(d["t"]), t_change = Float64.(d["t_change"]), X = _mat(d["X"]),
                 sigmas = Float64.(d["sigmas"]), tau = Float64(d["tau"]),
                 s_a = Float64.(d["s_a"]), s_m = Float64.(d["s_m"]), y = Float64.(d["y"])),
    scale = 0.2, do_reactant = DO_REACTANT)

# ALTERNATE-FLAG Stan-oracle axes. The graph's eager `ifelse` evaluates BOTH
# branches; the real posterior selects one (losscurve growthmodel_id=1 Weibull;
# accel prior_only=0 likelihood-included). These entries flip the flag in BOTH
# the BridgeStan data and the RK bound port, so the four-axis gate certifies the
# gradient of the OTHER branch against an actual Stan oracle — the independent
# check that the eagerly-evaluated-but-unselected branch does not poison the
# reverse gradient.
_want("losscurve_altflag") && gate("loss_curves-losscurve_sislob";
    graph = PE.LosscurveSislobExample.build_losscurve_sislob_graph(),
    have = (:unconstrained, :growthmodel_id, :cohort_id, :t_idx, :t_value, :premium, :loss),
    bind = d -> (growthmodel_id = Int(d["growthmodel_id"]), cohort_id = Int.(d["cohort_id"]),
                 t_idx = Int.(d["t_idx"]), t_value = Float64.(d["t_value"]),
                 premium = Float64.(d["premium"]), loss = Float64.(d["loss"])),
    altflag = (regex = r"\"growthmodel_id\"\s*:\s*\d+", replacement = "\"growthmodel_id\": 0",
               bind = (; growthmodel_id = 0)),
    scale = 0.3, do_reactant = DO_REACTANT)

_want("accel_altflag") && gate("mcycle_splines-accel_splines";
    graph = PE.AccelSplinesExample.build_accel_splines_graph(),
    have = (:unconstrained, :Y, :Xs, :Zs_1_1, :Xs_sigma, :Zs_sigma_1_1, :prior_only),
    bind = d -> (Y = Float64.(d["Y"]), Xs = _mat(d["Xs"]), Zs_1_1 = _mat(d["Zs_1_1"]),
                 Xs_sigma = _mat(d["Xs_sigma"]), Zs_sigma_1_1 = _mat(d["Zs_sigma_1_1"]),
                 prior_only = Int(d["prior_only"])),
    altflag = (regex = r"\"prior_only\"\s*:\s*\d+", replacement = "\"prior_only\": 1",
               bind = (; prior_only = 1)),
    scale = 0.3, do_reactant = DO_REACTANT)

@assert _RAN[] == length(SEL) "$(_RAN[]) gate(s) ran but $(length(SEL)) were selected"
@assert _RAN[] > 0 "no gates ran"

# Native phase: assert Reactant is STILL unloaded AFTER every model ran, not only
# before — nothing in the native value/gradient path may have pulled the Reactant
# extension in. (Before + after bracket the whole native certification.)
if !DO_REACTANT
    @assert !_reactant_loaded() "FORECAST_REACTANT=0 native phase loaded Reactant during execution (was unloaded at start)"
    println("  [native] Reactant asserted UNLOADED before AND after all models"); flush(stdout)
end

const _PHASE = DO_REACTANT ?
    "all FOUR parts (native value+grad, Reactant primal+grad)" :
    "the TWO native parts (Reactant UNLOADED; parts 3-4 deferred to a FORECAST_REACTANT=1 run)"
println("\n== forecast_batch_gate: $(_RAN[]) model(s) [$(join(SEL, ", "))] PASSED $_PHASE =="); flush(stdout)
