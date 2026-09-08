# Fast RK-focused Reactant phase for the all-82 posteriordb benchmark.
#
# This process deliberately does NOT load DynamicPPL/Turing or evaluate reference Stan. Native
# parity is established by the native phase and every model package test. Here BridgeStan is
# used only to read the unconstrained dimension from the already-cached PosteriorDB `.so`; the
# faithful RK graph receives the complete real dataset and each Reactant operation is attempted
# independently. A cell is either a timing or its exact lowering/setup diagnostic.
using Random, LinearAlgebra, Statistics
import TOML
using Chairmarks: @be
import BridgeStan, PosteriorDB, Reactant, Enzyme
using ReactiveKernels, ReactiveKernelsPPLExamples
using DifferentiationInterface

const PHASE = "reactant"
const RECEIPT = get(ENV, "RK_ALL80_RECEIPT", "")
include(joinpath(@__DIR__, "all80_registry.jl"))
include(joinpath(@__DIR__, "all80_receipt.jl"))
include(joinpath(@__DIR__, "all80_axes.jl"))
const RK = All80Registry.REGISTRY

const STD = joinpath(@__DIR__, "sampler_transpiler")
include(joinpath(STD, "eight_schools_density.jl"))
include(joinpath(@__DIR__, "nuts_kernel_authoring_fixture_b.jl"))
include(joinpath(STD, "position_multinomial_hmc_kernel.jl"))
using .EightSchoolsDensity: Potential, Gradient, CallbackHandle
const F = NUTSBMutationAuthoringFixture

const AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
med(b) = median(b).time * 1e9

# A Julia/Enzyme process abort is NOT catchable as an exception (signal 6 kills the whole
# subprocess), so a model whose AD genuinely aborts must be pre-declared here to preserve the
# other rows' coverage. The mechanism is retained for that case.
#
# EMPTY as of the f1e8b83 re-test (HEAD c75242b, 2026-09-08): the sole prior entry,
# `Survey_data-Survey_model`, was re-attempted IN ISOLATION (survey_ad_probe.jl — the exact
# run_reactant_one setup, real `prepare_ad` + gradient `@compile` + gradient eval, no skip) and
# it SURVIVES: prepare_ad OK, gradient @compile OK, finite gradient (first=-1.603). The
# historical signal-6 abort in Enzyme generated-call/GC marking (Survey is a 1-D discrete-count
# marginalization) no longer reproduces on this base; the referenced compact-run logs
# (kb-run-compact.tsoAmU / .6LkART) are gone. So Survey now runs its real AD like every other
# model. Re-populate this dict ONLY for a model whose abort is freshly reproduced.
const PROCESS_ABORTING_AD = Dict{String,String}()

function hmc_loop(kb, prep, q, backend, rng_factory;
                  T = 4, steps = All80Axes.HMC_STEPS, rounds = All80Axes.HMC_ROUNDS)
    D = length(q)
    point = transpiled_endpoint(F.euclidean_phasepoint, F.leapfrog!,
        CallbackHandle(Potential(kb)), CallbackHandle(Gradient(prep)),
        Diagonal(ones(D)), copy(q), zeros(D))
    prog = prepare_transpiled(PositionMultinomialHMCAuthoring.multinomial_hmc_state, point;
        backend, method = :step!, argument = rng_factory(), iterations = T,
        kernel_kwargs = (n_steps = steps, step_f = F.leapfrog!, stepsize = 0.03),
        outputs = (position = (:init, :pos),))
    st = initial_transpiled_state(prog)
    warm = prog(st, rng_factory())
    all(isfinite, Array(warm.outputs.position)) || error("$backend HMC warmup was non-finite")
    times = Float64[]
    for _ in 1:rounds
        local result
        elapsed = @elapsed (result = prog(st, rng_factory()))
        all(isfinite, Array(result.outputs.position)) ||
            error("$backend HMC timed result was non-finite")
        push!(times, elapsed)
    end
    median(times) / T * 1e6
end

include(joinpath(@__DIR__, "all80_reactant_evals.jl"))

function cached_bridge_model(post)
    stan = PosteriorDB.implementation(PosteriorDB.model(post), "stan")
    stan_path = PosteriorDB.path(stan)
    library = first(splitext(stan_path)) * "_model.so"
    isfile(library) || error("cached BridgeStan library missing: $library")
    BridgeStan.StanModel(library, PosteriorDB.load(PosteriorDB.dataset(post), String), 468)
end

function valid_rk_point(kb, dim, entry; rng = Xoshiro(0xC0FFEE))
    # STRICT-INTERIOR probe. zeros(dim) is DEGENERATE for several models: a hard Uniform
    # boundary (dogs_log's mixed-sign box), the a^0·b^0 = 1 endpoint singularity
    # (dogs_hierarchical), or a symmetric logsumexp TIE where the AD subgradient is ambiguous
    # (mixtures at mu1==mu2) — all poor gradient probes (performance audit 2026-09-08). A
    # registry `probe_q` pins the model source's own separated probe; else search small random
    # MIXED-SIGN points; zeros is only the last resort.
    ok(q) = length(q) == dim && isfinite(try kb(q) catch; NaN end)
    entry.probe_q !== nothing && ok(entry.probe_q) && return collect(Float64, entry.probe_q)
    for q in Iterators.flatten(([fill(0.1, dim), fill(-0.1, dim)],
                                (0.5 .* randn(rng, dim) for _ in 1:96), (zeros(dim),)))
        ok(q) && return collect(Float64, q)
    end
    error("no strict-interior finite RK probe point for a $dim-dim model")
end

function run_reactant_one(name)
    println("\n===== [reactant-fast] $name =====")
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    entry = RK[name]
    graph = getproperty(getproperty(ReactiveKernelsPPLExamples, entry.mod), entry.build)()
    kb = prepare(graph; have = entry.have, want = :posterior, bound = entry.bind(data))
    sm = cached_bridge_model(post)
    q = valid_rk_point(kb, Int(BridgeStan.param_unc_num(sm)), entry)
    abort_reason = get(PROCESS_ABORTING_AD, name, nothing)
    prep = abort_reason === nothing ? try
        prepare_ad(kb, AE, q; active = :unconstrained)
    catch err
        err
    end : ErrorException(abort_reason)
    # Capability-first Reactant HMC probe: four transitions are enough to compile, execute,
    # finite-check, and time the fixed 16-step program. Native HMC owns the adaptive
    # throughput protocol; running native Enzyme here would duplicate work and can abort on
    # exactly the full-data AD shapes this pass is meant to classify.
    # Reference Stan gradient oracle (BridgeStan, propto=false jacobian=true) mapped to RK
    # unconstrained order via the registry stan_perm — the SAME oracle the native phase uses
    # (all80_posteriordb_body.jl). NO finite differences; the .so (sm) is already loaded.
    sperm = entry.stan_perm
    stan_order_q = sperm === nothing ? q : q[sperm]
    grad_oracle = try
        g = BridgeStan.log_density_gradient(sm, stan_order_q; propto = false, jacobian = true)[2]
        sperm === nothing ? g : g[sortperm(sperm)]
    catch err
        err
    end
    transitions = All80Axes.HMC_MIN_TRANSITIONS
    row = reactant_cells(kb, prep, q; transitions, grad_oracle)
    row["hmc_steps"] = All80Axes.HMC_STEPS
    row["hmc_rounds"] = All80Axes.HMC_ROUNDS
    row["hmc_target_round_seconds"] = All80Axes.HMC_TARGET_ROUND_SECONDS
    row["hmc_reactant_transitions"] = transitions   # DISTINCT from native's calibrated hmc_transitions
    # (fixed at HMC_MIN_TRANSITIONS=4; surfaces the batch-size asymmetry + avoids an aggregate cell clash)
    for key in ("primal_rk_reactant", "gradient_rk_reactant", "hmc_rk_reactant")
        value = row[key]
        println("  $key = ", value isa Real ? string(round(value; sigdigits = 4)) : first(value, 180))
    end
    row
end

requested = [name for name in ARGS if !startswith(name, "-")]
unknown = [name for name in requested if !haskey(RK, name)]
isempty(unknown) || error("all80 reactant: unknown key(s): $(join(unknown, ", "))")
targets = isempty(requested) ? sort(collect(keys(RK))) : requested
retry_requested = get(ENV, "RK_ALL80_RETRY", "") == "1"
retry_requested && isempty(requested) &&
    error("RK_ALL80_RETRY=1 requires explicit model keys; refusing to replay all 82 implicitly")
rows = Dict{String,Any}()
if get(ENV, "RK_ALL80_RESUME", "") == "1" && RECEIPT != "" && isfile(RECEIPT)
    prior = TOML.parsefile(RECEIPT)
    get(prior, "schema", "") == All80Receipt.SCHEMA || error("reactant resume schema mismatch")
    get(prior, "phase", "") == PHASE || error("reactant resume phase mismatch")
    for (name, cells) in get(prior, "models", Dict())
        rows[String(name)] = Dict{String,Any}(String(k) => v for (k, v) in cells)
    end
    println("resumed reactant receipt with $(length(rows)) existing rows")
end

pending = [name for name in targets if retry_requested || !haskey(rows, name)]
for (index, name) in enumerate(pending)
    rows[name] = try
        run_reactant_one(name)
    catch err
        reason = string("reactant setup: ", first(replace(sprint(showerror, err), "\n" => " "), 220))
        Dict(cell => reason for cell in All80Receipt.REACTANT_CELLS)
    end
    All80Receipt.write_phase(RECEIPT, PHASE, rows)
    println("  [$(length(rows))/$(length(targets)) complete; $index/$(length(pending)) this run] $name recorded")
    flush(stdout)
end
println("REACTANT_PHASE_DONE rows=$(length(rows))")
