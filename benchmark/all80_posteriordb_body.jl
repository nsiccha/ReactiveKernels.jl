# All-82 measurement body — ONE PHASE per subprocess (RK_ALL80_PHASE), so loading Reactant
# never perturbs native timings. Reuses the pinned upstream driver's helpers
# (bridge_model / make_model / CoordinateMap / stable_ldf / random_valid_points / draw_center)
# for reference-Stan + upstream-Turing at a shared q, and the faithful registry graphs for RK.
# HMC uses the SAME authored program as benchmark/reactant_hmc_loop_table.jl (transpiled_endpoint
# / prepare_transpiled / multinomial_hmc_state, backend :native|:reactant) fed the faithful
# kb/prep. AHMC-Turing uses AdvancedHMC on the Turing LDF. All timings via Chairmarks/@elapsed;
# every side parity-gated against reference Stan (propto=false, jacobian=true) before timing.
using Random, LinearAlgebra, Statistics
import TOML, SHA
using Chairmarks: @be
import BridgeStan, PosteriorDB, DynamicPPL
using ReactiveKernels, ReactiveKernelsPPLExamples
import Enzyme
using DifferentiationInterface

const UP = ENV["RK_ALL80_UPSTREAM"]
const PHASE = get(ENV, "RK_ALL80_PHASE", "native")
const RECEIPT = get(ENV, "RK_ALL80_RECEIPT", "")
# DISCOVER: measurement/offset-discovery mode — RECORD parity_pass + the observed offset per
# model and CATCH per-model, instead of erroring on the first undeclared offset. DEFAULT (unset)
# is the HARD gate (the publication contract). The observed offsets guide SOURCE derivation of
# each off_rk/off_tu; the final hard-gated run then verifies measured == source-derived.
const DISCOVER = get(ENV, "RK_ALL80_DISCOVER", "") == "1"
include(joinpath(UP, "posteriordb.jl"))            # main-guarded; helpers + make_model + models + PDB
include(joinpath(@__DIR__, "all80_registry.jl"))   # All80Registry.REGISTRY (82 executable-gated entries)
include(joinpath(@__DIR__, "all80_receipt.jl"))    # All80Receipt.write_phase
include(joinpath(@__DIR__, "all80_axes.jl"))       # All80Axes.measure_native (native phase)
include(joinpath(@__DIR__, "all80_metadata.jl"))   # All80Metadata.meta(key): family/note + optional provenance
const RK = All80Registry.REGISTRY

# HMC transpiler machinery (Reactant-FREE to include; the :native backend needs no Reactant).
const STD = joinpath(@__DIR__, "sampler_transpiler")
include(joinpath(STD, "eight_schools_density.jl"))            # Potential, Gradient, CallbackHandle
include(joinpath(@__DIR__, "nuts_kernel_authoring_fixture_b.jl"))  # F.euclidean_phasepoint, F.leapfrog!
include(joinpath(STD, "position_multinomial_hmc_kernel.jl"))  # PositionMultinomialHMCAuthoring.multinomial_hmc_state
using .EightSchoolsDensity: Potential, Gradient, CallbackHandle
const F = NUTSBMutationAuthoringFixture

# Phase-conditional HEAVY imports: native loads AdvancedHMC (NO Reactant); reactant loads Reactant
# + the per-operation Reactant.@compile cells (reactant_cells), kept in a separate file so the
# macro is lowered ONLY where Reactant is loaded (a Reactant macro in run_one's body would be
# expanded at run_one's DEFINITION and would fail in the native phase).
if PHASE == "native"
    @eval import AdvancedHMC
elseif PHASE == "reactant"
    @eval import Reactant
    include(joinpath(@__DIR__, "all80_reactant_evals.jl"))
else
    error("all80 body: unknown RK_ALL80_PHASE=$(PHASE) (expected native|reactant)")
end

# FAIL-CLOSED isolation: the native phase must NOT have Reactant loaded (executable, not prose).
_reactant_loaded() = any(m -> String(nameof(m)) == "Reactant", values(Base.loaded_modules))
PHASE == "native" && _reactant_loaded() &&
    error("native-phase isolation violated: Reactant is loaded before measurements")

# Standing RK policy (user, 2026-09-07): RK does NOT use Enzyme runtime activity.
# A correct RK graph has statically-resolvable activity (data ports are DI.Constant
# contexts; the active port is the unconstrained vector), so plain reverse is right;
# needing set_runtime_activity would itself signal an activity defect to fix, not mask.
const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
med(b) = median(b).time * 1e9   # Chairmarks `.time` is SECONDS; receipt/fmt use NANOSECONDS
fmt(ns) = ns < 1e3 ? "$(round(ns; digits=1)) ns" : ns < 1e6 ? "$(round(ns/1e3; digits=2)) µs" : "$(round(ns/1e6; digits=3)) ms"

stan_val(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
stan_grad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

# Scale-aware native parity floor (scale_ok / parity_tol / STAB_ATOL / STAB_ULP_C / classify_boundary)
# — SHARED with the pure fixtures so they exercise the PRODUCTION helpers.
include(joinpath(@__DIR__, "all80_parity.jl"))
# Protocol identity stamped on every row measured under the scale-aware / per-cell native gate
# (Fix A rescale, C NaN-gradient resilience, D primal-defect preservation, E per-side boundary).
const PARITY_PROTOCOL = "native-parity-scaleaware-peraxis-v1-2026-09-08"

# Full-diagnostic artifact for a THROWN RK-cell failure. The cell string is a 200-char SUMMARY; the
# COMPLETE exception + backtrace is written to a CONTENT-ADDRESSED raw log linked from the cell —
# `diagnostics/<key>__<cell>__<sha16>.log`, where <sha16> is the first 16 hex of the sha256 of the
# full text. Content-addressing makes the artifact IMMUTABLE: a later fix run with a DIFFERENT stack
# writes a DIFFERENT file, so it cannot silently replace the exact diagnostic an older receipt links
# to; the hash is persisted in the cell string alongside the path (performance review 2026-09-08).
function _rk_full_diag(cell, name, err, bt)
    full = sprint(showerror, err, bt)
    ref = ""
    if RECEIPT != ""
        dir = joinpath(dirname(RECEIPT), "diagnostics"); mkpath(dir)
        digest = bytes2hex(SHA.sha256(full)); h = digest[1:16]   # h = 16-hex PREFIX (filename only)
        open(joinpath(dir, "$(name)__$(cell)__$(h).log"), "w") do io; write(io, full); end
        ref = " [full stack sha256:$(digest) (filename uses the 16-hex prefix) -> diagnostics/$(name)__$(cell)__$(h).log]"
    end
    string(cell, ": ", first(replace(full, "\n" => " "), 200), ref)
end

# ---- HMC loop (µs/transition) via the shared transpiled program, either backend ----
# rng_factory yields a fresh sampler RNG (Xoshiro for :native; ReactantRNG for :reactant —
# constructed only in the reactant phase, so `Reactant` is never referenced under :native).
function hmc_loop(kb, prep, q, backend, rng_factory; T = 1000, steps = 16, rounds = 6)
    D = length(q)
    point = transpiled_endpoint(F.euclidean_phasepoint, F.leapfrog!,
        CallbackHandle(Potential(kb)), CallbackHandle(Gradient(prep)),
        Diagonal(ones(D)), copy(q), zeros(D))
    prog = prepare_transpiled(PositionMultinomialHMCAuthoring.multinomial_hmc_state, point;
        backend, method = :step!, argument = rng_factory(), iterations = T,
        kernel_kwargs = (n_steps = steps, step_f = F.leapfrog!, stepsize = 0.03),
        outputs = (position = (:init, :pos),))
    st = initial_transpiled_state(prog)
    warm = prog(st, rng_factory())                    # warmup (compile+run for :reactant)
    all(isfinite, Array(warm.outputs.position)) ||
        error("$backend RK HMC produced a non-finite warmup position (chain exploded)")
    # FAIL-CLOSED per TIMED round: validate the final position of every measured result
    # before accepting its duration (timing NaN/Inf is not a valid HMC cell). Fresh seed
    # each round (deterministic, symmetric with AHMC).
    times = Float64[]
    for _ in 1:rounds
        local r
        t = @elapsed (r = prog(st, rng_factory()))
        all(isfinite, Array(r.outputs.position)) ||
            error("$backend RK HMC produced a non-finite position on a timed round (chain exploded)")
        push!(times, t)
    end
    median(times) / T * 1e6
end
_native_rng() = Xoshiro(91)

# ---- AHMC-Turing HMC (µs/transition): fixed-L multinomial HMC over the Turing LDF, MATCHING
# the RK program (L=steps, stepsize=0.03, no adaptation, warmup excluded). ----
function ahmc_loop(tldf, q0; T = 1000, steps = 16, stepsize = 0.03, rounds = 6)
    D = length(q0)
    val(q) = DynamicPPL.LogDensityProblems.logdensity(tldf, q)
    valgrad(q) = DynamicPPL.LogDensityProblems.logdensity_and_gradient(tldf, q)
    metric = AdvancedHMC.DiagEuclideanMetric(D)
    ham = AdvancedHMC.Hamiltonian(metric, val, valgrad)
    integ = AdvancedHMC.Leapfrog(stepsize)
    kern = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integ, AdvancedHMC.FixedNSteps(steps)))
    loop() = AdvancedHMC.sample(Xoshiro(91), ham, kern, collect(q0), T;
        progress = false, verbose = false)
    _finalpos(r) = r isa Tuple ? r[1][end] : r[end]
    all(isfinite, _finalpos(loop())) ||               # warmup + finite check
        error("AHMC-Turing HMC produced a non-finite warmup sample (chain exploded)")
    times = Float64[]
    for _ in 1:rounds                                 # per-round finite check (symmetry with RK HMC)
        local r
        t = @elapsed (r = loop())
        all(isfinite, _finalpos(r)) ||
            error("AHMC-Turing HMC produced a non-finite sample on a timed round (chain exploded)")
        push!(times, t)
    end
    median(times) / T * 1e6
end

function run_one(name; seed = 468, scale = 0.2, draws = 3)
    println("\n===== [$PHASE] $name =====")
    post = PosteriorDB.posterior(PDB, name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    model = make_model(Val(Symbol(name)), data)
    model_name = PosteriorDB.name(PosteriorDB.model(post))
    sm = bridge_model(post, seed)
    dim = Int(BridgeStan.param_unc_num(sm))
    rng = Xoshiro(seed + sum(codeunits(name)))
    points = random_valid_points(sm, rng, draws, scale; center = draw_center(name, dim))
    map_ldf = stable_ldf(model; logdensity = DynamicPPL.getlogjoint)   # transforms only
    tldf = stable_ldf(model; adtype = AutoMooncake())                  # getlogjoint_internal (unconstrained)
    e = RK[name]
    # Artifact-Stan parameter order need not match the RK unconstrained packing
    # (registry `stan_perm`; eight_schools_centered declares theta[8],mu,tau). The Stan
    # oracle (value AND gradient) is always contacted in Stan order; gradients map back.
    # The Turing CoordinateMap also consumes Stan order — every cmap/groups contact below
    # uses sq(q), never raw RK-order q.
    sq = q -> e.stan_perm === nothing ? q : q[e.stan_perm]
    _back = e.stan_perm === nothing ? nothing : sortperm(e.stan_perm)
    bs = g -> _back === nothing ? g : g[_back]
    cmap = CoordinateMap(sm, BridgeStan.param_names(sm; include_tp = false, include_gq = false),
                         model_name, DynamicPPL.get_all_ranges_and_transforms(map_ldf))
    groups = classify_coordinate_groups(cmap, sq.(points))
    graph = getproperty(getproperty(ReactiveKernelsPPLExamples, e.mod), e.build)()
    kb = prepare(graph; have = e.have, want = :posterior, bound = e.bind(data))
    prep = prepare_ad(kb, AE, points[1]; active = :unconstrained)

    # ---- HARD GATE vs reference Stan (propto=false jacobian=true): declared offset + stability + gradient ----
    svj = [stan_val(sm, sq(q)) for q in points]
    rk_vals = [kb(q) for q in points]
    tu_vals = [DynamicPPL.LogDensityProblems.logdensity(tldf, cmap(sq(q))) for q in points]
    rk_c = svj .- rk_vals
    tu_c = svj .- tu_vals
    rk_off = mean(rk_c); tu_off = mean(tu_c)
    rk_off_err = abs(rk_off - e.off_rk); tu_off_err = abs(tu_off - e.off_tu)
    rk_stab = maximum(abs, rk_c .- rk_off); tu_stab = maximum(abs, tu_c .- tu_off)
    # DISTINCT roundoff magnitudes for the RK-side vs Turing-side floor (never one RK-derived mag
    # for both — performance review): each is set by the values actually differenced on that side.
    mag_rk = max(maximum(abs, svj), maximum(abs, rk_vals))
    mag_tu = max(maximum(abs, svj), maximum(abs, tu_vals))
    # RK reverse gradient parity — resilient to the KNOWN authored-plate core defect
    # (snag authored-plate-i-4556ee01: Int-axis accumulator / Any-materialization Enzyme
    # failure). On THAT failure record the exact diagnostic; under DISCOVER the row keeps
    # every unaffected cell and only gradient_rk + hmc_rk_native carry the string (to be
    # republished as numbers once the core fix lands). Any OTHER error still throws.
    rk_grad_diag = nothing
    rk_grad_err = try
        maximum(relerr(ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2], bs(stan_grad(sm, sq(q)))) for q in points)
    catch err
        rk_grad_diag = _rk_full_diag("gradient_rk", name, err, catch_backtrace())
        NaN
    end
    # A NON-FINITE gradient RETURNED (not thrown — e.g. dogs_hier's genuine direct-p boundary
    # defect, fixed on canonical 1349092/2ba4a639) routes to the SAME diagnostic REPORTING PATH as a
    # thrown Enzyme error — NOT the same ROOT CAUSE (dogs_hier: boundary; arma11: authored-scan
    # activity; GLMM: authored-plate marker-six). Same path so DISCOVER keeps every unaffected cell.
    if rk_grad_diag === nothing && !isfinite(rk_grad_err)
        rk_grad_diag = "gradient_rk: RK reverse returned a non-finite gradient at an in-support point (genuine RK-reverse defect)"
    end
    tu_grad_err = maximum(relerr(gradient_in_stan_coordinates(cmap, groups, q,
                   DynamicPPL.LogDensityProblems.logdensity_and_gradient(tldf, cmap(sq(q)))[2]), stan_grad(sm, sq(q))) for q in points)
    # INDEPENDENT PER-CELL failure preservation (Fix D, performance contract 2026-09-08): each RK
    # cell is a number only if THAT operation is verified vs reference Stan; else an exact diagnostic.
    #  · gradient_rk: unavailable (thrown/non-finite, above) OR finite-but-WRONG (relerr ≥ 2e-3).
    #  · primal_rk:   primal not a constant offset (stab fails the scale-aware floor).
    # Under DISCOVER the row is KEPT with the UNAFFECTED reference cells (Turing/Stan) intact and
    # parity_pass=false; a failed RK cell never prints a passing verdict or a benchmark ratio
    # (All80Axes emits the diagnostic string, never a number). NON-DISCOVER still fails CLOSED.
    if rk_grad_diag === nothing && !(rk_grad_err < 2e-3)
        rk_grad_diag = "gradient_rk: RK reverse gradient disagrees with reference Stan (relerr $rk_grad_err ≥ 2e-3) — genuine RK-reverse defect"
    end
    rk_primal_diag = scale_ok(rk_stab, mag_rk) ? nothing :
        (isfinite(rk_stab) && isfinite(mag_rk)) ?
        "primal_rk: RK primal not a constant offset vs reference Stan across in-support draws " *
        "(stab $rk_stab > tol $(parity_tol(mag_rk)) at mag $mag_rk) — genuine RK-primal defect" :
        "primal_rk: RK primal scale check FAILED — non-finite residual/magnitude at an in-support draw " *
        "(stab=$rk_stab, mag=$mag_rk; RK primal values=$(rk_vals); reference Stan values=$(svj)) — genuine RK-primal defect"
    if !DISCOVER
        rk_primal_diag === nothing || error("RK PRIMAL parity FAIL $name: $rk_primal_diag")
        rk_grad_diag === nothing || error("RK gradient parity FAIL $name: $rk_grad_diag")
    end
    # Turing/Stan REFERENCE structural integrity stays HARD in both modes: a reference-side failure
    # invalidates the row's comparison entirely (not an isolated RK-cell defect).
    scale_ok(tu_stab, mag_tu) && tu_grad_err < 2e-3 ||
        error("Turing STRUCTURAL parity FAIL $name: stab $tu_stab (tol $(parity_tol(mag_tu)) at mag $mag_tu) grad $tu_grad_err (not a constant offset)")
    # The value-offset-vs-declared is a soft check under DISCOVER.
    rk_off_ok = scale_ok(rk_off_err, mag_rk; atol = 1e-4); tu_off_ok = scale_ok(tu_off_err, mag_tu; atol = 1e-4)
    # SUPPORT-boundary probe (per-side; Fix E / option 1). Stan is the REFERENCE and stays HARD; a
    # per-side RK or Turing failure is owned by THAT side (kept under DISCOVER as a diagnostic +
    # correctness=false; hard error under strict). A finite Turing where RK+Stan are -Inf is
    # NON-EQUIVALENT support — not a valid reference success (no RK/Turing ratio, no HMC claim).
    turing_support_ok = true; turing_support_diag = nothing
    if e.boundary !== nothing
        qb = e.boundary(points[1])
        vb_r = kb(qb); vb_s = stan_val(sm, sq(qb)); vb_t = DynamicPPL.LogDensityProblems.logdensity(tldf, cmap(sq(qb)))
        rk_boundary_diag, turing_support_ok, turing_support_diag = classify_boundary(vb_r, vb_s, vb_t; discover = DISCOVER)
        rk_boundary_diag === nothing || (rk_primal_diag = rk_primal_diag === nothing ? rk_boundary_diag : rk_primal_diag)
        (rk_boundary_diag === nothing && turing_support_ok) &&
            println("  support-boundary probe: OK (RK, Stan, Turing all -Inf)")
    end
    # MODEL-CORRECTNESS flag (distinct from report-completeness): FALSE on ANY required RK
    # primal/gradient diagnostic, a declared-offset mismatch, OR a non-equivalent Turing support.
    # A failed axis never carries a passing verdict, though the row stays report-complete via its
    # reference cells (Turing timings, if kept, are labeled non-equivalent — not passing cells).
    rk_primal_ok = rk_primal_diag === nothing; rk_grad_ok = rk_grad_diag === nothing
    parity_pass = rk_off_ok && tu_off_ok && rk_primal_ok && rk_grad_ok && turing_support_ok
    if !DISCOVER
        rk_off_ok || error("RK offset FAIL $name: measured $(rk_off) != declared $(e.off_rk)")
        tu_off_ok || error("Turing offset FAIL $name: measured $(tu_off) != declared $(e.off_tu)")
    end
    println("  parity: RK off=$(round(rk_off;sigdigits=4))(want $(e.off_rk)) grad=$(round(rk_grad_err;sigdigits=2)) $(rk_off_ok ? "OK" : "OFFSET") | Turing off=$(round(tu_off;sigdigits=6))(want $(e.off_tu)) grad=$(round(tu_grad_err;sigdigits=2)) $(tu_off_ok ? "OK" : "OFFSET")$(turing_support_ok ? "" : " | TURING-SUPPORT INVALID")")

    q = points[1]; qt = cmap(sq(q)); gbuf = similar(q)
    md = All80Metadata.meta(name)
    ctx = (; dim = dim, family = md.family, note = md.note,
        parity_pass = parity_pass, rk_off = rk_off, tu_off = tu_off, off_reason = e.off_reason,
        rk_grad_relerr = rk_grad_err, tu_grad_relerr = tu_grad_err,
        # scale-aware parity evidence: the offset-residual spread + the magnitude that sets its
        # roundoff floor (tol = STAB_ATOL + STAB_ULP_C·mag·eps), so any row's gate call is auditable.
        rk_stab = rk_stab, tu_stab = tu_stab, mag_rk = mag_rk, mag_tu = mag_tu,
        # per-cell RK failure preservation (Fix D): each diagnostic is `nothing` where that RK
        # operation is verified, else the exact diagnostic string. All80Axes emits a NUMBER only
        # where the corresponding *_ok is true; a failed cell carries the diagnostic, never a ratio.
        rk_grad_diag = rk_grad_diag, rk_primal_diag = rk_primal_diag,
        rk_primal_ok = rk_primal_ok, rk_grad_ok = rk_grad_ok,
        # Fix E: Turing-side support equivalence. false ⇒ upstream Turing has a NON-EQUIVALENT support
        # vs reference Stan/RK; Turing timings (if kept) are observability only — NO RK/Turing ratio.
        turing_support_ok = turing_support_ok, turing_support_diag = turing_support_diag,
        # protocol stamp: rows measured under the scale-aware / per-cell (Fix A/C/D/E) native protocol.
        # Rows WITHOUT this field predate it — their absent scale-aware fields are "not recorded under
        # the old protocol", not a defect (performance directive: no full-82 rerun just to backfill).
        protocol = PARITY_PROTOCOL,
        # native single-eval closures (timed by All80Axes); HMC returns µs/transition directly
        rk_primal = () -> kb(q),
        tu_primal = () -> DynamicPPL.LogDensityProblems.logdensity(tldf, qt),
        stan_primal = () -> stan_val(sm, sq(q)),
        rk_grad = () -> ReactiveKernels.ad_value_and_gradient!(prep, gbuf, q),
        tu_grad = () -> DynamicPPL.LogDensityProblems.logdensity_and_gradient(tldf, qt),
        stan_grad = () -> BridgeStan.log_density_gradient(sm, sq(q); propto = false, jacobian = true),
        hmc_time_loop = (backend, transitions) ->
            hmc_loop(kb, prep, q, backend, _native_rng;
                T = transitions, steps = All80Axes.HMC_STEPS,
                rounds = All80Axes.HMC_ROUNDS),
        ahmc_time_loop = transitions ->
            ahmc_loop(tldf, qt; T = transitions, steps = All80Axes.HMC_STEPS,
                rounds = All80Axes.HMC_ROUNDS))

    row = if PHASE == "native"
        r = All80Axes.measure_native(ctx)
        # optional optimized-Stan / further-Turing cells: user-directive deferral provenance
        # (phase-independent metadata; attached in the native phase, carried by the aggregate).
        r["primal_opt_stan"] = md.primal_opt_stan; r["primal_further_turing"] = md.primal_further_turing
        r["gradient_opt_stan"] = md.gradient_opt_stan; r["gradient_further_turing"] = md.gradient_further_turing
        r
    else   # reactant: the 3 cells, each numeric-or-diagnostic, caught independently
        reactant_cells(kb, prep, q)
    end
    for (k, v) in sort(collect(row); by = first)
        v isa Real && occursin(r"^(primal|gradient)_", k) && println("  $(rpad(k,24)) $(fmt(v))")
        v isa Real && startswith(k, "hmc_") && println("  $(rpad(k,24)) $(round(v;sigdigits=3)) µs/it")
        (v isa AbstractString && occursin("reactant", k)) && println("  $(rpad(k,24)) N/A → $(first(v,90))")
    end
    PHASE == "native" && println("  HMC protocol: $(row["hmc_transitions"]) transitions × $(row["hmc_steps"]) steps × $(row["hmc_rounds"]) rounds")
    PHASE == "native" && _reactant_loaded() &&
        error("native-phase isolation violated: Reactant loaded DURING $name (transpiler :native must not pull it)")
    println("MODEL_OK [$PHASE] $name")
    row
end

# Empty ARGS => ALL 82 keys (sorted). Any requested key MUST exist (reject unknown, never
# silently filter). Keep an explicit one-key smoke working.
_req = [n for n in ARGS if !startswith(n, "-")]
_unknown = [k for k in _req if !haskey(RK, k)]
isempty(_unknown) || error("all80: unknown registry key(s) requested: $(join(_unknown, ", "))")
targets = isempty(_req) ? sort(collect(keys(RK))) : _req
retry_requested = get(ENV, "RK_ALL80_RETRY", "") == "1"
retry_requested && isempty(_req) &&
    error("RK_ALL80_RETRY=1 requires explicit model keys; refusing to replay all 82 implicitly")
rows = Dict{String,Any}()
if get(ENV, "RK_ALL80_RESUME", "") == "1" && RECEIPT != "" && isfile(RECEIPT)
    prior = TOML.parsefile(RECEIPT)
    get(prior, "schema", "") == All80Receipt.SCHEMA ||
        error("all80 resume: schema mismatch in $RECEIPT")
    get(prior, "phase", "") == PHASE ||
        error("all80 resume: phase mismatch in $RECEIPT")
    for (name, cells) in get(prior, "models", Dict())
        rows[String(name)] = Dict{String,Any}(String(k) => v for (k, v) in cells)
    end
    println("resumed $PHASE receipt with $(length(rows)) existing rows: $RECEIPT")
end
pending = [name for name in targets if retry_requested || !haskey(rows, name)]
for (i, name) in enumerate(pending)
    if DISCOVER
        try
            rows[name] = run_one(name)
        catch err
            msg = first(replace(sprint(showerror, err), "\n" => " "), 220)
            println("  DISCOVER-CATCH $name → $msg")
            rows[name] = Dict{String,Any}("error" => msg, "parity_pass" => false)
        end
    else
        rows[name] = run_one(name)
    end
    # INCREMENTAL: rewrite the phase receipt after EACH model so a late crash/OOM cannot erase
    # earlier completed rows or exact errors (parent-mandated failure isolation).
    RECEIPT != "" && All80Receipt.write_phase(RECEIPT, PHASE, rows)
    println("  [$(length(rows))/$(length(targets)) complete; $i/$(length(pending)) this run] $(name) recorded" *
            (RECEIPT != "" ? " → $RECEIPT" : ""))
end
RECEIPT != "" && println("wrote phase receipt ($PHASE, $(length(rows)) rows): $RECEIPT")
println("\nALL80_PHASE_DONE $PHASE")
