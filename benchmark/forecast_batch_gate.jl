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
# Every axis is a hard `@assert`, so a regression exits nonzero. Each case builds
# its graph from the REAL installed package's `build_*_graph`, prepares it, and
# executes it (the first `kb(q)`) ALL INSIDE the one ordinary `gate` function —
# the graph is passed as a thunk `() -> build_*_graph()` and called there, not at
# top level. Each model's template was evaluated MODEL-ONLY at package load (a
# world boundary), so this is world-age-safe. In-graph data preprocessing is
# exercised by binding
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
# Ordinary reverse-mode Enzyme (no function_annotation), matching the package's
# committed AD checks. Non-active model data travel through DI as constants.
const AE = AutoEnzyme(mode = Enzyme.Reverse)

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

function gate(name; build, have, bind, want = :posterior, scale = 0.3, npts = 6, seed = 468,
              do_reactant = true, boundary_point = nothing, altflag = nothing)
    println("\n########## $name ##########"); flush(stdout)
    if altflag === nothing
        sm, post = bridge(name, seed)
    else
        sm, post = bridge_altflag(name, altflag.regex, altflag.replacement, seed)
        println("  ALT-FLAG: $(altflag.regex) => $(altflag.replacement); RK bind $(get(altflag, :bind, nothing))"); flush(stdout)
    end
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    bound = bind(data)
    # An alt-flag `bind` merges the flipped flag into the RK bound ports; for a
    # WANT-selected flag (accel prior_only) there is no bind — only the Stan
    # oracle is instantiated with the flipped data.
    if altflag !== nothing && get(altflag, :bind, nothing) !== nothing
        bound = merge(bound, altflag.bind)
    end
    # build -> prepare -> execute ALL inside this one ordinary function (the
    # benchmark run_one shape): `build` is a thunk `() -> build_*_graph()`, so
    # the graph is constructed here, not at top level. The templates were
    # evaluated MODEL-ONLY at package load (a world boundary), so this is
    # world-age-safe. The first `kb(q)` below is the execute.
    graph = build()
    kb = prepare(graph; have, want, bound = bound)
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

    # ---- stress probe (where supplied): reference validity FIRST ----
    # The BridgeStan oracle must be value- AND gradient-finite at the probe
    # before anything is claimed about the RK graph there — the same
    # reference-validity selection rule as the random probes, applied to the
    # deliberately extreme point.
    stress_q = boundary_point === nothing ? nothing : boundary_point(dim)
    if stress_q !== nothing
        vsb = sval(sm, stress_q); gsb = sgrad(sm, stress_q)
        @assert isfinite(vsb) && all(isfinite, gsb) "$name: stress-point BridgeStan reference not finite"
        vb = kb(stress_q)
        gbnd = ReactiveKernels.ad_value_and_gradient!(prep, similar(stress_q), stress_q)[2]
        @assert isfinite(vb) && all(isfinite, gbnd) "$name: stress-point RK value/grad not finite"
        @assert _relv(vb, vsb) < VALUE_TOL "$name: stress-point value ≠ Stan"
        @assert relerr(gbnd, gsb) < GRAD_TOL "$name: stress-point gradient ≠ Stan"
        println("  [b] stress-point (reference-valid) value+grad finite and match Stan PASS"); flush(stdout)
    end

    do_reactant && _reactant_axes(name, kb, prep, sm, pts, stress_q)
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

# Two INACTIVE-BRANCH STRESS probes are supplied (deliberately extreme points,
# not ordinary support boundaries). The graph's eager `ifelse` evaluates BOTH
# arms, so a naive transcription's unselected arm can overflow/underflow and
# poison the reverse gradient while Stan — whose ternary selects lazily — stays
# finite. Each stress probe establishes the Stan oracle's value+gradient
# finiteness FIRST, then asserts ordinary-native and (in the Reactant process)
# compiled-Reactant value+gradient parity:
#   * losscurve (data growthmodel_id=1, Weibull selected): q₁=log(1000),
#     q₂=log(max t_value) — at ω=1000 the unused log-logistic arm's
#     t^ω/θ^ω overflows; the authored overflow-safe forms must stay finite.
#   * accel_altflag (prior_only=1, WANT-pruned prior-only node mirroring Stan's
#     `if (!prior_only)`): all-zero q except the unconstrained Intercept_sigma
#     at -800 — exp(-800) would underflow the deselected likelihood's σ to 0;
#     the planned prior-only program must stay finite and match the prior-only
#     Stan oracle.
# No FURTHER boundary_point is supplied: none of the four models carries a
# SEPARATE hard prior-support restriction in the tested modes beyond the
# ordinary constraining transforms (exp for positivity, the data-bounded logit
# level, the positive_ordered cumulative-exp scales). This is NOT a claim that
# every finite Float64 q maps strictly interior (exp overflow/underflow and
# logistic saturation exist). Correctness at the sampled points is what the gate
# asserts; reference-validity probe selection (never RK finiteness) still guards
# the general case, so a Stan-finite / RK-nonfinite point would survive and
# hard-fail.

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
    build = () -> PE.LosscurveSislobExample.build_losscurve_sislob_graph(),
    have = (:unconstrained, :growthmodel_id, :cohort_id, :t_idx, :t_value, :premium, :loss),
    bind = d -> (growthmodel_id = Int(d["growthmodel_id"]), cohort_id = Int.(d["cohort_id"]),
                 t_idx = Int.(d["t_idx"]), t_value = Float64.(d["t_value"]),
                 premium = Float64.(d["premium"]), loss = Float64.(d["loss"])),
    boundary_point = dim -> begin
        # Inactive-branch stress: ω=1000, θ=max(t_value) keeps the SELECTED
        # Weibull arm bounded ((t/θ)^ω ≤ 1) while the unused log-logistic arm
        # would overflow in the naive t^ω/θ^ω transcription.
        q = zeros(dim)
        q[1] = log(1000.0)
        q[2] = log(maximum(PE.LosscurveSislobExample.LOSSCURVE_T_VALUE))
        q
    end,
    scale = 0.3, do_reactant = DO_REACTANT)

# accel_splines — brms penalized-spline regression; bind the response + the four
# design/basis matrices (in-graph matrix-vector products).
_want("accel") && gate("mcycle_splines-accel_splines";
    build = () -> PE.AccelSplinesExample.build_accel_splines_graph(),
    have = (:unconstrained, :Y, :Xs, :Zs_1_1, :Xs_sigma, :Zs_sigma_1_1),
    bind = d -> (Y = Float64.(d["Y"]), Xs = _mat(d["Xs"]), Zs_1_1 = _mat(d["Zs_1_1"]),
                 Xs_sigma = _mat(d["Xs_sigma"]), Zs_sigma_1_1 = _mat(d["Zs_sigma_1_1"])),
    scale = 0.3, do_reactant = DO_REACTANT)

# state_space — structural DLM; bind only the raw series y, x, w (the bounded
# level transform, positive_ordered scales, random-walk level and trailing-window
# seasonal are all derived in-graph).
_want("state_space") && gate("uk_drivers-state_space_stochastic_level_stochastic_seasonal";
    build = () -> PE.StateSpaceStochasticExample.build_state_space_stochastic_graph(),
    have = (:unconstrained, :y, :x, :w),
    bind = d -> (y = Float64.(d["y"]), x = Float64.(d["x"]), w = Float64.(d["w"])),
    scale = 0.2, do_reactant = DO_REACTANT)

# prophet — Facebook Prophet linear-trend forecasting; bind the raw time/
# changepoint/design data (the changepoint incidence matrix + trend + seasonality
# are derived in-graph). LINEAR trend only (build_prophet_graph validates the
# trend flag); the logistic mode is not part of the graph.
_want("prophet") && gate("rstan_downloads-prophet";
    build = () -> PE.ProphetExample.build_prophet_graph(),
    have = (:unconstrained, :t, :t_change, :X, :sigmas, :tau, :s_a, :s_m, :y),
    bind = d -> (t = Float64.(d["t"]), t_change = Float64.(d["t_change"]), X = _mat(d["X"]),
                 sigmas = Float64.(d["sigmas"]), tau = Float64(d["tau"]),
                 s_a = Float64.(d["s_a"]), s_m = Float64.(d["s_m"]), y = Float64.(d["y"])),
    scale = 0.2, do_reactant = DO_REACTANT)

# ALTERNATE-FLAG Stan-oracle axes. The real posterior selects one mode
# (losscurve growthmodel_id=1 Weibull; accel prior_only=0 likelihood-included).
# These entries flip the flag in the BridgeStan data, so the four-axis gate
# certifies the OTHER mode against an actual Stan oracle. Losscurve's eager
# per-cell `ifelse` needs the flipped flag bound in RK too; accel's prior_only
# is a WANT-selected node (`accel_splines_posterior_want(true)`), mirroring
# Stan's `if (!prior_only)` — planning `:prior_only_posterior` does not compute
# the likelihood at all.
_want("losscurve_altflag") && gate("loss_curves-losscurve_sislob";
    build = () -> PE.LosscurveSislobExample.build_losscurve_sislob_graph(),
    have = (:unconstrained, :growthmodel_id, :cohort_id, :t_idx, :t_value, :premium, :loss),
    bind = d -> (growthmodel_id = Int(d["growthmodel_id"]), cohort_id = Int.(d["cohort_id"]),
                 t_idx = Int.(d["t_idx"]), t_value = Float64.(d["t_value"]),
                 premium = Float64.(d["premium"]), loss = Float64.(d["loss"])),
    altflag = (regex = r"\"growthmodel_id\"\s*:\s*\d+", replacement = "\"growthmodel_id\": 0",
               bind = (; growthmodel_id = 0)),
    scale = 0.3, do_reactant = DO_REACTANT)

_want("accel_altflag") && gate("mcycle_splines-accel_splines";
    build = () -> PE.AccelSplinesExample.build_accel_splines_graph(),
    have = (:unconstrained, :Y, :Xs, :Zs_1_1, :Xs_sigma, :Zs_sigma_1_1),
    want = PE.AccelSplinesExample.accel_splines_posterior_want(true),
    bind = d -> (Y = Float64.(d["Y"]), Xs = _mat(d["Xs"]), Zs_1_1 = _mat(d["Zs_1_1"]),
                 Xs_sigma = _mat(d["Xs_sigma"]), Zs_sigma_1_1 = _mat(d["Zs_sigma_1_1"])),
    altflag = (regex = r"\"prior_only\"\s*:\s*\d+", replacement = "\"prior_only\": 1",
               bind = nothing),
    boundary_point = dim -> begin
        # Inactive-branch stress on the WANT-pruned prior-only program: an
        # all-zero q except the unconstrained Intercept_sigma at -800 underflows
        # the deselected likelihood's σ=exp(-800) to 0 — the planned node must
        # stay finite and match the prior-only Stan oracle there.
        q = zeros(dim)
        q[3 + size(PE.AccelSplinesExample.ACCEL_XS, 2) +
              size(PE.AccelSplinesExample.ACCEL_ZS_1_1, 2)] = -800.0
        q
    end,
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
println("\n== forecast_batch_gate: $(_RAN[]) case(s) [$(join(SEL, ", "))] PASSED $_PHASE ==")
println("   (native = 6 reference-valid probes/case, plus the reference-valid stress probe where supplied;")
println("    Reactant axes = first probe, plus the stress probe where supplied)"); flush(stdout)
