# Same-T HMC investigation harness (todo 0yyv0u3; performance co-owns diagnosis via 0dur9gk).
#
# Answers whether Reactant's compiled-HMC-loop advantage materializes at MATCHED, long batches, or
# whether "Reactant+HMC loses" is only the capability-probe artifact (Reactant fixed T=4 vs native
# adaptive T=30–1000). Drives the SAME authored `multinomial_hmc_state` program on BOTH backends at
# matched T ∈ {4, 100, 1000}, same q / metric (Diagonal I) / L=16 / ε=0.03.
#
# Two fixes over the production hmc_loop the spec calls out:
#   (1) COMPILE (prepare_transpiled + first warmup) is timed SEPARATELY from EXECUTE — compile is
#       never folded into the per-transition figure.
#   (2) The per-round device RNG is PRECONSTRUCTED OUTSIDE the timed loop — the production loop calls
#       `rng_factory()` (→ `to_rarray` for :reactant) INSIDE `@elapsed` every round, a per-round
#       marshalling cost amortized over only T transitions that inflates small-T Reactant figures.
#
# One BACKEND per process (SAME_T_BACKEND=native|reactant) so loading Reactant never perturbs the
# :native timings (same isolation rationale as the main two-phase benchmark). Reactant is imported
# ONLY for the reactant backend. NO cross-backend trajectory-agreement gate (same input + same work,
# not identical draws). Timings are observed-load / provenance-labelled (shared host, run-freely).
#
# WHAT THIS DOES AND DOES NOT SHOW (performance review 2026-09-08):
#   - It measures matched-T native-vs-Reactant per-transition wall-clock on FIVE representative
#     models. At T=100/1000 all five are faster on Reactant; at T=4 four of five are faster on
#     native (GLM_Binomial is faster on Reactant already at T=4). This is OBSERVED on these reps,
#     NOT proof that every main-suite Reactant-HMC loss was a batch artifact, nor that no lowering
#     deficiency remains anywhere.
#   - The crossover for the four losing-at-4 models is only BRACKETED in (4, 100] — the grid does
#     not resolve where in that interval it happens; do not state a specific crossover T.
#   - The per-round timer is NOT a pure kernel execute: the transpiled wrapper still calls
#     _device_argument + deepcopy(outputs) inside prog(...). XLA launch cost is NOT isolated from
#     those wrapper costs. Preconstructing the RNG changes the construction boundary but the
#     old-vs-new boundary is NOT measured here (see OPEN control below).
#   - prep_warmup_s is preparation + FIRST execution (warmup), not a standalone cold compile.
#   - Provenance recorded is partial (versions + medians + raw rounds + q). It omits data hashes,
#     loaded RK/distribution roots+SHAs, DI version, effective compile options, per-round start/end,
#     and a load trace. Preserve the completed receipts/logs WITH these limitations; capture the
#     missing fields on any follow-on execution rather than rerunning solely for paperwork.
#
# OPEN (spec item not completed): a matched RNG-construction BOUNDARY control (same T, same q, only
# the RNG-construction boundary changed: preconstructed vs constructed-in-timer) with honest
# sample/load/source recording. Left explicitly open — the matched-T sub-result stands without it.
#
# Run:  SAME_T_BACKEND=native   SAME_T_RECEIPT=.../same_t_hmc_native.toml   julia --project=benchmark/all80-env benchmark/all80_same_t_hmc.jl
#       SAME_T_BACKEND=reactant SAME_T_RECEIPT=.../same_t_hmc_reactant.toml julia --project=benchmark/all80-env benchmark/all80_same_t_hmc.jl
using Random, LinearAlgebra, Statistics
import TOML, Dates
import BridgeStan, PosteriorDB, Enzyme
using ReactiveKernels, ReactiveKernelsPPLExamples
using DifferentiationInterface

const BACKEND = Symbol(get(ENV, "SAME_T_BACKEND", "native"))
const RECEIPT = get(ENV, "SAME_T_RECEIPT", "")
BACKEND in (:native, :reactant) || error("SAME_T_BACKEND must be native|reactant, got $BACKEND")
if BACKEND == :reactant
    @eval import Reactant
end

# Same transpiler machinery + authored program as the two benchmark phases.
const STD = joinpath(@__DIR__, "sampler_transpiler")
include(joinpath(STD, "eight_schools_density.jl"))            # Potential, Gradient, CallbackHandle
include(joinpath(@__DIR__, "nuts_kernel_authoring_fixture_b.jl"))
include(joinpath(STD, "position_multinomial_hmc_kernel.jl"))  # PositionMultinomialHMCAuthoring.multinomial_hmc_state
include(joinpath(@__DIR__, "all80_registry.jl"))
using .EightSchoolsDensity: Potential, Gradient, CallbackHandle
const F = NUTSBMutationAuthoringFixture
const RK = All80Registry.REGISTRY
const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

const STEPS = 16
const STEPSIZE = 0.03
const ROUNDS = 6
# SAME_T_TS / SAME_T_MODELS override the grid for smoke tests + targeted reruns (comma-separated).
const TS = haskey(ENV, "SAME_T_TS") ?
    Tuple(parse.(Int, split(ENV["SAME_T_TS"], ","))) : (4, 100, 1000)
const REP_SET = haskey(ENV, "SAME_T_MODELS") ? String.(split(ENV["SAME_T_MODELS"], ",")) : [
    "eight_schools-eight_schools_centered",
    "eight_schools-eight_schools_noncentered",
    "kilpisjarvi_mod-kilpisjarvi",
    "radon_mn-radon_variable_intercept_slope_centered",   # larger hierarchical
    "GLM_Binomial_data-GLM_Binomial_model",               # control (a winning GLM)
]

# Per-backend device RNG factory. :native → Xoshiro (matches _native_rng); :reactant → ReactantRNG
# from a to_rarray'd seed. Constructed via this factory OUTSIDE the timed loop (the fix).
mkrng() = BACKEND == :native ? Xoshiro(91) :
          Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77]))

function find_point(kb, dim, entry)
    okq(x) = length(x) == dim && isfinite(try kb(x) catch; NaN end)
    entry.probe_q !== nothing && okq(entry.probe_q) && return collect(Float64, entry.probe_q)
    rng = Xoshiro(0xC0FFEE)
    for cand in Iterators.flatten(([fill(0.1, dim), fill(-0.1, dim)],
                                   (0.5 .* randn(rng, dim) for _ in 1:96), (zeros(dim),)))
        okq(cand) && return collect(Float64, cand)
    end
    error("no valid probe point")
end

function build_model(name)
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    entry = RK[name]
    graph = getproperty(getproperty(ReactiveKernelsPPLExamples, entry.mod), entry.build)()
    kb = prepare(graph; have = entry.have, want = :posterior, bound = entry.bind(data))
    stan = PosteriorDB.implementation(PosteriorDB.model(post), "stan")
    lib = first(splitext(PosteriorDB.path(stan))) * "_model.so"
    sm = BridgeStan.StanModel(lib, PosteriorDB.load(PosteriorDB.dataset(post), String), 468)
    q = find_point(kb, Int(BridgeStan.param_unc_num(sm)), entry)
    prep = prepare_ad(kb, AE, q; active = :unconstrained)
    (kb, prep, q)
end

# Matched-T measurement: compile timed separately, RNGs preconstructed outside the execute timer.
function same_t_measure(kb, prep, q; T, steps = STEPS, rounds = ROUNDS)
    D = length(q)
    point = transpiled_endpoint(F.euclidean_phasepoint, F.leapfrog!,
        CallbackHandle(Potential(kb)), CallbackHandle(Gradient(prep)),
        Diagonal(ones(D)), copy(q), zeros(D))
    local prog, st, warm
    # prep_warmup_s = prepare_transpiled + FIRST execution (warmup). This is NOT a standalone cold
    # compile cost: for :reactant it bundles compile + one run; and at T=100/1000 in a warm process
    # (small numbers below) the executable is already compiled, so this is mostly the warmup run.
    # Do not read it as a cold-vs-warm per-executable comparison.
    prep_warmup_s = @elapsed begin
        prog = prepare_transpiled(PositionMultinomialHMCAuthoring.multinomial_hmc_state, point;
            backend = BACKEND, method = :step!, argument = mkrng(), iterations = T,
            kernel_kwargs = (n_steps = steps, step_f = F.leapfrog!, stepsize = STEPSIZE),
            outputs = (position = (:init, :pos),))
        st = initial_transpiled_state(prog)
        warm = prog(st, mkrng())
    end
    all(isfinite, Array(warm.outputs.position)) || error("$BACKEND T=$T warmup non-finite")
    # RNG preconstructed outside the execute timer. NOTE (performance review): this changes the
    # RNG-construction boundary vs the production hmc_loop, but the harness does NOT measure the
    # old-vs-new boundary, and the transpiled wrapper STILL calls _device_argument + deepcopy of
    # the outputs inside `prog(...)` — so the timer is NOT a pure kernel-execute and the per-round
    # figure is not an isolated marshalling gain. See the boundary control (OPEN) in the header.
    rngs = [mkrng() for _ in 1:rounds]
    times = Float64[]
    for i in 1:rounds
        local r
        t = @elapsed (r = prog(st, rngs[i]))
        all(isfinite, Array(r.outputs.position)) || error("$BACKEND T=$T timed result non-finite")
        push!(times, t)
    end
    (; prep_warmup_s, exec_round_s_median = median(times), us_per_transition = median(times) / T * 1e6,
       raw_round_s = times)
end

_pkgroot(m) = try string(pkgdir(m)) catch; "?" end
_pkgver(m) = try string(pkgversion(m)) catch; "?" end
const VERSIONS = Dict{String,Any}(
    "julia" => string(VERSION), "backend" => String(BACKEND),
    "enzyme" => _pkgver(Enzyme),
    "differentiationinterface" => _pkgver(DifferentiationInterface),
    "reactant" => BACKEND == :reactant ? _pkgver(@eval Reactant) : "not-loaded",
    "reactivekernels_version" => _pkgver(ReactiveKernels),
    "reactivekernels_root" => _pkgroot(ReactiveKernels),
    "pplexamples_root" => _pkgroot(ReactiveKernelsPPLExamples),
    "device" => "CPU", "steps" => STEPS, "stepsize" => STEPSIZE, "rounds" => ROUNDS,
    "started_at" => string(Dates.now()),
    # HONEST GAPS (performance review): NOT recorded here — data hashes, exact loaded package SHAs,
    # effective XLA/compile options, per-round start/end timestamps, host load trace. Capture on any
    # follow-on execution; not reconstructable after the fact for the completed run.
    "provenance_note" => "partial; see harness header LIMITATIONS")

rows = Dict{String,Any}()
for name in REP_SET
    println("\n===== [same-T $BACKEND] $name =====")
    local kb, prep, q
    try
        kb, prep, q = build_model(name)
    catch err
        msg = first(replace(sprint(showerror, err), "\n" => " "), 200)
        println("  BUILD-FAIL $name → $msg")
        for T in TS; rows["$name|T=$T"] = Dict("error" => "build: $msg", "T" => T); end
        continue
    end
    for T in TS
        cell = try
            m = same_t_measure(kb, prep, q; T = T)
            println("  T=$(rpad(T,4)) prep+warmup=$(round(m.prep_warmup_s;sigdigits=3))s  " *
                    "$(round(m.us_per_transition;sigdigits=4)) µs/transition")
            Dict("T" => T, "prep_warmup_s" => m.prep_warmup_s,
                 "exec_round_s_median" => m.exec_round_s_median,
                 "us_per_transition" => m.us_per_transition, "dim" => length(q),
                 "raw_round_s" => m.raw_round_s, "q" => collect(Float64, q))
        catch err
            msg = first(replace(sprint(showerror, err), "\n" => " "), 220)
            println("  T=$(rpad(T,4)) FAIL → $msg")
            Dict("T" => T, "error" => msg, "dim" => length(q))
        end
        rows["$name|T=$T"] = cell
        if RECEIPT != ""
            doc = Dict("schema" => "same-t-hmc-v1", "backend" => String(BACKEND),
                       "versions" => VERSIONS, "cells" => rows,
                       "generated_at" => string(Dates.now()))
            open(RECEIPT, "w") do io; TOML.print(io, doc; sorted = true); end
        end
    end
end
println("\nSAME_T_HMC_DONE backend=$BACKEND cells=$(length(rows))")
