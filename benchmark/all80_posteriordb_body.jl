# All-82 measurement body — ONE PHASE per subprocess (RK_ALL80_PHASE), so loading Reactant
# never perturbs native timings. Reuses the pinned upstream driver's helpers
# (bridge_model / make_model / CoordinateMap / stable_ldf / random_valid_points / draw_center)
# for reference-Stan + upstream-Turing at a shared q, and the faithful registry graphs for RK.
# HMC uses the SAME authored program as benchmark/reactant_hmc_loop_table.jl (transpiled_endpoint
# / prepare_transpiled / multinomial_hmc_state, backend :native|:reactant) fed the faithful
# kb/prep. AHMC-Turing uses AdvancedHMC on the Turing LDF. All timings via Chairmarks/@elapsed;
# every side parity-gated against reference Stan (propto=false, jacobian=true) before timing.
using Random, LinearAlgebra, Statistics
using Chairmarks: @be
import BridgeStan, PosteriorDB, DynamicPPL
using ReactiveKernels, ReactiveKernelsPPLExamples
import Enzyme
using DifferentiationInterface

const UP = ENV["RK_ALL80_UPSTREAM"]
const PHASE = get(ENV, "RK_ALL80_PHASE", "native")
const RECEIPT = get(ENV, "RK_ALL80_RECEIPT", "")
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

const AE = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const)
med(b) = median(b).time * 1e9   # Chairmarks `.time` is SECONDS; receipt/fmt use NANOSECONDS
fmt(ns) = ns < 1e3 ? "$(round(ns; digits=1)) ns" : ns < 1e6 ? "$(round(ns/1e3; digits=2)) µs" : "$(round(ns/1e6; digits=3)) ms"

stan_val(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
stan_grad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

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
    cmap = CoordinateMap(sm, BridgeStan.param_names(sm; include_tp = false, include_gq = false),
                         model_name, DynamicPPL.get_all_ranges_and_transforms(map_ldf))
    groups = classify_coordinate_groups(cmap, points)

    e = RK[name]
    graph = getproperty(getproperty(ReactiveKernelsPPLExamples, e.mod), e.build)()
    kb = prepare(graph; have = e.have, want = :posterior, bound = e.bind(data))
    prep = prepare_ad(kb, AE, points[1]; active = :unconstrained)

    # ---- HARD GATE vs reference Stan (propto=false jacobian=true): declared offset + stability + gradient ----
    svj = [stan_val(sm, q) for q in points]
    rk_c = svj .- [kb(q) for q in points]
    tu_c = svj .- [DynamicPPL.LogDensityProblems.logdensity(tldf, cmap(q)) for q in points]
    rk_off = mean(rk_c); tu_off = mean(tu_c)
    rk_off_err = abs(rk_off - e.off_rk); tu_off_err = abs(tu_off - e.off_tu)
    rk_stab = maximum(abs, rk_c .- rk_off); tu_stab = maximum(abs, tu_c .- tu_off)
    rk_grad_err = maximum(relerr(ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2], stan_grad(sm, q)) for q in points)
    tu_grad_err = maximum(relerr(gradient_in_stan_coordinates(cmap, groups, q,
                   DynamicPPL.LogDensityProblems.logdensity_and_gradient(tldf, cmap(q))[2]), stan_grad(sm, q)) for q in points)
    (rk_off_err < 1e-4 && rk_stab < 1e-6 && rk_grad_err < 2e-3) ||
        error("RK parity FAIL $name: off $(rk_off) (declared $(e.off_rk)) stab $rk_stab grad $rk_grad_err")
    (tu_off_err < 1e-4 && tu_stab < 1e-6 && tu_grad_err < 2e-3) ||
        error("Turing parity FAIL $name: off $(tu_off) (declared $(e.off_tu)) stab $tu_stab grad $tu_grad_err")
    println("  parity: RK off=$(round(rk_off;sigdigits=3))(want $(e.off_rk)) grad=$(round(rk_grad_err;sigdigits=2)) | Turing off=$(round(tu_off;sigdigits=3))(want $(e.off_tu)) grad=$(round(tu_grad_err;sigdigits=2))")
    if e.boundary !== nothing
        qb = e.boundary(points[1])
        vb_r = kb(qb); vb_s = stan_val(sm, qb); vb_t = DynamicPPL.LogDensityProblems.logdensity(tldf, cmap(qb))
        (vb_r == -Inf && vb_s == -Inf && vb_t == -Inf) || error("boundary probe FAIL $name rk=$vb_r stan=$vb_s turing=$vb_t")
        println("  support-boundary probe: OK (RK, Stan, Turing all -Inf)")
    end

    q = points[1]; qt = cmap(q); gbuf = similar(q)
    md = All80Metadata.meta(name)
    ctx = (; dim = dim, family = md.family, note = md.note,
        parity_pass = true, rk_off = rk_off, tu_off = tu_off, off_reason = e.off_reason,
        rk_grad_relerr = rk_grad_err, tu_grad_relerr = tu_grad_err,
        # native single-eval closures (timed by All80Axes); HMC returns µs/transition directly
        rk_primal = () -> kb(q),
        tu_primal = () -> DynamicPPL.LogDensityProblems.logdensity(tldf, qt),
        stan_primal = () -> stan_val(sm, q),
        rk_grad = () -> ReactiveKernels.ad_value_and_gradient!(prep, gbuf, q),
        tu_grad = () -> DynamicPPL.LogDensityProblems.logdensity_and_gradient(tldf, qt),
        stan_grad = () -> BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true),
        hmc_time_loop = backend -> hmc_loop(kb, prep, q, backend, _native_rng),  # native phase
        ahmc_time_loop = () -> ahmc_loop(tldf, qt))

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
rows = Dict{String,Any}()
for name in targets
    rows[name] = run_one(name)
end
if RECEIPT != ""
    All80Receipt.write_phase(RECEIPT, PHASE, rows)
    println("wrote phase receipt ($PHASE): $RECEIPT")
end
println("\nALL80_PHASE_DONE $PHASE")
