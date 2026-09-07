# All-80 body — SMOKE stage: validate the scaled pattern on one model before
# wiring the full RK registry. Reuses the pinned upstream driver's helpers
# (bridge_model / make_model / CoordinateMap / stable_ldf / random_valid_points /
# draw_center / verify_against_stan) for reference-Stan + upstream-Turing at a
# shared q, and adds the idiomatic-RK column (RK shares Stan's unconstrained
# coordinates per posteriordb's own gate, so it evaluates at the same q, jacobian
# inclusive). All timings via Chairmarks @be; parity gated against reference Stan.
using Random, LinearAlgebra, Statistics
using Chairmarks: @be
import BridgeStan, PosteriorDB, DynamicPPL
using ReactiveKernels, ReactiveKernelsPPLExamples
import Enzyme
using DifferentiationInterface

const UP = ENV["RK_ALL80_UPSTREAM"]
include(joinpath(UP, "posteriordb.jl"))   # main-guarded; defines helpers + make_model + models
include(joinpath(@__DIR__, "all80_registry.jl"))   # All80Registry.REGISTRY (82 executable-gated entries)
const AE = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)
med(b) = median(b).time
fmt(ns) = ns < 1e3 ? "$(round(ns; digits=1)) ns" : ns < 1e6 ? "$(round(ns/1e3; digits=2)) µs" : "$(round(ns/1e6; digits=3)) ms"

# RK-binding registry (per posterior name) — the 82 executable-gated entries from
# all80_registry.jl. Each NamedTuple: mod::Symbol, build::Symbol (RKE.<mod>.<build>()),
# have::Tuple, bind::(data->NamedTuple over FULL data), boundary::(nothing|q->q_boundary),
# off_rk/off_tu::Float64 (EXPECTED Stan−side offset; 0 unless a SOURCE-derived
# parameter-independent normalization constant applies), off_reason::String. off_rk==0 and
# off_tu matches the declared source-derived value; any UNDECLARED offset is a HARD failure.
const RK = All80Registry.REGISTRY

stan_val(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
stan_grad(sm, q) = BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]
relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

function run_one(name; seed = 468, scale = 0.2, draws = 3)
    println("\n===== $name =====")
    post = PosteriorDB.posterior(PDB, name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    model = make_model(Val(Symbol(name)), data)
    model_name = PosteriorDB.name(PosteriorDB.model(post))
    sm = bridge_model(post, seed)
    dim = Int(BridgeStan.param_unc_num(sm))
    rng = Xoshiro(seed + sum(codeunits(name)))
    points = random_valid_points(sm, rng, draws, scale; center = draw_center(name, dim))

    # map_ldf: only for the CoordinateMap's constrained<->unconstrained transforms (convention-independent).
    map_ldf = stable_ldf(model; logdensity = DynamicPPL.getlogjoint)
    # tldf: stable_ldf DEFAULT logdensity = getlogjoint_INTERNAL = the UNCONSTRAINED posterior
    # (Jacobian-inclusive) — the SAME density RK and Stan (jacobian=true) evaluate. Same convention
    # on every side (the already-green nine-model harness used exactly this).
    tldf = stable_ldf(model; adtype = AutoMooncake())
    cmap = CoordinateMap(sm, BridgeStan.param_names(sm; include_tp = false, include_gq = false),
                         model_name, DynamicPPL.get_all_ranges_and_transforms(map_ldf))
    groups = classify_coordinate_groups(cmap, points)

    e = RK[name]
    graph = getproperty(getproperty(ReactiveKernelsPPLExamples, e.mod), e.build)()
    kb = prepare(graph; have = e.have, want = :posterior, bound = e.bind(data))
    prep = prepare_ad(kb, AE, points[1]; active = :unconstrained)

    # ---- HARD GATE (every side vs reference Stan, propto=false jacobian=true): value offset must
    #      match the DECLARED expectation (default 0) AND be stable across draws; + gradient. ----
    # SAME density on every side: the unconstrained posterior (Jacobian-inclusive) =
    # Stan jacobian=true = RK posterior = Turing getlogjoint_internal(qt). Offsets expected 0.
    svj = [stan_val(sm, q) for q in points]                          # Stan jacobian=true (all sides gate against this)
    rk_c = svj .- [kb(q) for q in points]
    tu_c = svj .- [DynamicPPL.LogDensityProblems.logdensity(tldf, cmap(q)) for q in points]
    rk_off = mean(rk_c); tu_off = mean(tu_c)
    rk_off_err = abs(rk_off - e.off_rk); tu_off_err = abs(tu_off - e.off_tu)         # agree with DECLARED expectation
    rk_stab = maximum(abs, rk_c .- rk_off); tu_stab = maximum(abs, tu_c .- tu_off)   # stability across draws
    rk_grad_err = maximum(relerr(ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2], stan_grad(sm, q)) for q in points)
    tu_grad_err = maximum(
        relerr(gradient_in_stan_coordinates(cmap, groups, q,
                   DynamicPPL.LogDensityProblems.logdensity_and_gradient(tldf, cmap(q))[2]),
               stan_grad(sm, q)) for q in points)
    (rk_off_err < 1e-4 && rk_stab < 1e-6 && rk_grad_err < 2e-3) ||
        error("RK parity FAIL $name: offset $(rk_off) (declared $(e.off_rk), err $rk_off_err) stab $rk_stab grad $rk_grad_err")
    (tu_off_err < 1e-4 && tu_stab < 1e-6 && tu_grad_err < 2e-3) ||
        error("Turing parity FAIL $name: offset $(tu_off) (declared $(e.off_tu), err $tu_off_err) stab $tu_stab grad $tu_grad_err")
    println("  parity: RK off=$(round(rk_off;sigdigits=3))(want $(e.off_rk)) stab=$(round(rk_stab;sigdigits=2)) grad=$(round(rk_grad_err;sigdigits=2)) | Turing off=$(round(tu_off;sigdigits=3))(want $(e.off_tu)) stab=$(round(tu_stab;sigdigits=2)) grad=$(round(tu_grad_err;sigdigits=2))")
    # support-boundary probe: RK, Stan AND upstream Turing (boundary q via CoordinateMap) all -Inf.
    if e.boundary !== nothing
        qb = e.boundary(points[1])
        vb_r = kb(qb); vb_s = stan_val(sm, qb); vb_t = DynamicPPL.LogDensityProblems.logdensity(tldf, cmap(qb))
        (vb_r == -Inf && vb_s == -Inf && vb_t == -Inf) || error("boundary probe FAIL $name rk=$vb_r stan=$vb_s turing=$vb_t")
        println("  support-boundary probe: OK (RK, Stan, Turing all -Inf)")
    end

    # ---- timings (jacobian-inclusive; primal + gradient) at the first draw ----
    q = points[1]; qt = cmap(q); gbuf = similar(q)
    sp = med(@be BridgeStan.log_density($sm, $q; propto = false, jacobian = true))
    sgd = med(@be BridgeStan.log_density_gradient($sm, $q; propto = false, jacobian = true))
    tp = med(@be DynamicPPL.LogDensityProblems.logdensity($tldf, $qt))
    tgd = med(@be DynamicPPL.LogDensityProblems.logdensity_and_gradient($tldf, $qt))
    rp = med(@be $kb($q))
    rgd = med(@be ReactiveKernels.ad_value_and_gradient!($prep, $gbuf, $q))
    println("  primal   : reference-Stan $(fmt(sp))  upstream-Turing $(fmt(tp))  RK $(fmt(rp))")
    println("  gradient : reference-Stan $(fmt(sgd))  upstream-Turing $(fmt(tgd))  RK $(fmt(rgd))")
    println("SMOKE_MODEL_OK $name")
end

for name in ARGS
    startswith(name, "-") && continue
    haskey(RK, name) || continue
    run_one(name)
end
# default smoke target when no name given
any(a -> haskey(RK, a), ARGS) || run_one("GLM_Poisson_Data-GLM_Poisson_model")
println("\nALL80_SMOKE_DONE")
