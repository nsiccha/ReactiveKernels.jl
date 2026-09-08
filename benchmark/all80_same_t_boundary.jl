# RNG-construction BOUNDARY control (0yyv0u3 remaining item; performance spec 2026-09-08).
#
# Estimates ONLY the RNG-construction boundary contribution to the Reactant per-transition timer —
# NOT pure XLA launch cost, NOT all host overhead. Reactant backend only. Two models (a short-batch
# loser + a control), T ∈ {4, 1000}.
#
#   Arm A: device RNG PRECONSTRUCTED outside the timer  (what all80_same_t_hmc.jl uses).
#   Arm B: mkrng()/device RNG constructed INSIDE the timer (the production hmc_loop boundary).
#
# Contract (per review): SAME compiled `prog`, SAME immutable starting state (fresh
# initial_transpiled_state each round), SAME q/data, SAME RNG seed+algorithm. Both measurement
# closures are compiled/warmed first. A/B are ALTERNATED. ALL raw samples are kept. Output
# finiteness AND exact equality are verified OUTSIDE the timer (same seed+state ⇒ identical draws;
# this is a same-backend/same-seed control, NOT a cross-backend trajectory gate). Nothing else
# changes between arms — projection/copies/_device_argument, backend options, source are identical.
# Reports absolute delta and ratio at each T; makes NO assumption that construction loses.
#
# Run: SAME_T_BND_RECEIPT=.../same_t_boundary.toml julia --project=benchmark/all80-env benchmark/all80_same_t_boundary.jl
using Random, LinearAlgebra, Statistics
import TOML, Dates, SHA
import BridgeStan, PosteriorDB, Enzyme, Reactant
using ReactiveKernels, ReactiveKernelsPPLExamples
using DifferentiationInterface

const RECEIPT = get(ENV, "SAME_T_BND_RECEIPT", "")
const STD = joinpath(@__DIR__, "sampler_transpiler")
include(joinpath(STD, "eight_schools_density.jl"))
include(joinpath(@__DIR__, "nuts_kernel_authoring_fixture_b.jl"))
include(joinpath(STD, "position_multinomial_hmc_kernel.jl"))
include(joinpath(@__DIR__, "all80_registry.jl"))
using .EightSchoolsDensity: Potential, Gradient, CallbackHandle
const F = NUTSBMutationAuthoringFixture
const RK = All80Registry.REGISTRY
const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

const STEPS = 16
const STEPSIZE = 0.03
const ROUNDS = 30          # per arm, per (model, T)
const TS = (4, 1000)
const MODELS = ["eight_schools-eight_schools_centered",   # short-batch loser
                "GLM_Binomial_data-GLM_Binomial_model"]   # control
const SEED = UInt64[91, 77]
mkrng() = Reactant.ReactantRNG(Reactant.to_rarray(copy(SEED)))   # SAME seed + algorithm every call

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
    (kb, prep, q, data)
end

_hash(x) = bytes2hex(SHA.sha256(codeunits(string(x))))

# One compiled prog; alternate A/B; each round starts from a FRESH initial state so both arms
# measure identical work from identical conditions — the ONLY difference is where mkrng() is called.
function boundary_measure(kb, prep, q; T)
    D = length(q)
    point = transpiled_endpoint(F.euclidean_phasepoint, F.leapfrog!,
        CallbackHandle(Potential(kb)), CallbackHandle(Gradient(prep)),
        Diagonal(ones(D)), copy(q), zeros(D))
    prog = prepare_transpiled(PositionMultinomialHMCAuthoring.multinomial_hmc_state, point;
        backend = :reactant, method = :step!, argument = mkrng(), iterations = T,
        kernel_kwargs = (n_steps = STEPS, step_f = F.leapfrog!, stepsize = STEPSIZE),
        outputs = (position = (:init, :pos),))
    runA(st, rng) = prog(st, rng)                 # rng preconstructed (passed in)
    runB(st)      = prog(st, mkrng())             # rng constructed INSIDE
    # compile/warm BOTH closures before timing
    runA(initial_transpiled_state(prog), mkrng())
    runB(initial_transpiled_state(prog))
    tsA = Float64[]; tsB = Float64[]; outs = Vector{Vector{Float64}}()
    for i in 1:(2 * ROUNDS)
        st = initial_transpiled_state(prog)       # fresh immutable start (outside timer)
        if isodd(i)                               # ALTERNATE A / B
            rng = mkrng()                         # A: constructed OUTSIDE the timer
            t = @elapsed (r = runA(st, rng))
            push!(tsA, t)
        else
            t = @elapsed (r = runB(st))           # B: mkrng() INSIDE the timer
            push!(tsB, t)
        end
        push!(outs, Array{Float64}(r.outputs.position))   # equality/finiteness checked OUTSIDE timer
    end
    all(o -> all(isfinite, o), outs) || error("boundary T=$T: non-finite output")
    # same prog+state+seed ⇒ every round's output must be identical (same-backend/same-seed control)
    ref = outs[1]
    equal = all(o -> o == ref, outs)
    (; T,
       A_us_per_transition = median(tsA) / T * 1e6, B_us_per_transition = median(tsB) / T * 1e6,
       A_round_s_median = median(tsA), B_round_s_median = median(tsB),
       delta_us_per_transition = (median(tsB) - median(tsA)) / T * 1e6,
       ratio_B_over_A = median(tsB) / median(tsA),
       raw_A_round_s = tsA, raw_B_round_s = tsB, output_equal = equal)
end

# --- provenance recorded at process start (missing fields explicit; no retrospective stamping) ---
_git(args...) = try strip(read(setenv(`git $(collect(args))`; dir = @__DIR__), String)) catch; "?" end
const GIT_HEAD = _git("rev-parse", "HEAD")
const GIT_DIRTY = _git("status", "--porcelain") != ""
const PROV = Dict{String,Any}(
    "julia" => string(VERSION), "enzyme" => string(pkgversion(Enzyme)),
    "reactant" => string(pkgversion(Reactant)),
    "differentiationinterface" => string(pkgversion(DifferentiationInterface)),
    "reactivekernels_root" => try string(pkgdir(ReactiveKernels)) catch; "?" end,
    "git_head" => GIT_HEAD, "git_dirty" => GIT_DIRTY,
    "seed" => string(SEED), "steps" => STEPS, "stepsize" => STEPSIZE, "rounds_per_arm" => ROUNDS,
    "device" => "CPU", "backend" => "reactant", "started_at" => string(Dates.now()),
    "missing_fields" => "effective XLA/compile options; per-round wall timestamps; continuous host-load trace (only a start-of-run competing-julia sample is taken below)")

# start-of-run observed-load sample (competing julia processes), for the preliminary load qualification
function competing_julia()
    try
        out = read(`bash -c "ps -eo pcpu,comm | awk '\$2==\"julia\" && \$1>15{n++} END{print n+0}'"`, String)
        parse(Int, strip(out))
    catch; -1 end
end

rows = Dict{String,Any}()
PROV["competing_julia_at_start"] = competing_julia()
for name in MODELS
    println("\n===== [boundary $name] =====")
    kb, prep, q, data = build_model(name)
    qhash = _hash(q); dhash = _hash(sort(collect(keys(data))))
    for T in TS
        cell = try
            m = boundary_measure(kb, prep, q; T = T)
            println("  T=$(rpad(T,4)) A(pre)=$(round(m.A_us_per_transition;sigdigits=4)) " *
                    "B(in-timer)=$(round(m.B_us_per_transition;sigdigits=4)) µs/it  " *
                    "Δ=$(round(m.delta_us_per_transition;sigdigits=3)) ratioB/A=$(round(m.ratio_B_over_A;sigdigits=4)) " *
                    "equal=$(m.output_equal)")
            Dict("T" => T, "dim" => length(q), "q_sha256" => qhash, "data_keys_sha256" => dhash,
                 "A_us_per_transition" => m.A_us_per_transition, "B_us_per_transition" => m.B_us_per_transition,
                 "delta_us_per_transition" => m.delta_us_per_transition, "ratio_B_over_A" => m.ratio_B_over_A,
                 "raw_A_round_s" => m.raw_A_round_s, "raw_B_round_s" => m.raw_B_round_s,
                 "output_equal" => m.output_equal, "competing_julia_after" => competing_julia())
        catch err
            Dict("T" => T, "error" => first(replace(sprint(showerror, err), "\n" => " "), 220))
        end
        rows["$name|T=$T"] = cell
        RECEIPT != "" && open(RECEIPT, "w") do io
            TOML.print(io, Dict("schema" => "same-t-boundary-v1", "provenance" => PROV, "cells" => rows); sorted = true)
        end
    end
end
println("\nSAME_T_BOUNDARY_DONE cells=$(length(rows))")
