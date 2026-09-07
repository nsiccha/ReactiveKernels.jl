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

# A Julia/Enzyme process abort is not catchable as an exception. This exact model was
# attempted twice: both runs reached Survey and terminated with signal 6 in the generated
# AD call (logs kb-run-compact.tsoAmU and kb-run-compact.6LkART). Preserve primal coverage,
# but record that evidenced process-level diagnostic for the two AD-dependent cells.
const PROCESS_ABORTING_AD = Dict(
    "Survey_data-Survey_model" =>
        "Survey Reactant AD process-abort: Julia signal 6 in Enzyme generated-call/GC marking; reproduced twice",
)

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

function valid_rk_point(kb, dim)
    for q in (zeros(dim), fill(0.1, dim), fill(-0.1, dim), fill(0.25, dim))
        value = try kb(q) catch; NaN end
        isfinite(value) && return q
    end
    error("no finite RK value at the bounded deterministic Reactant probe points")
end

function run_reactant_one(name)
    println("\n===== [reactant-fast] $name =====")
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    entry = RK[name]
    graph = getproperty(getproperty(ReactiveKernelsPPLExamples, entry.mod), entry.build)()
    kb = prepare(graph; have = entry.have, want = :posterior, bound = entry.bind(data))
    sm = cached_bridge_model(post)
    q = valid_rk_point(kb, Int(BridgeStan.param_unc_num(sm)))
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
    transitions = All80Axes.HMC_MIN_TRANSITIONS
    row = reactant_cells(kb, prep, q; transitions)
    row["hmc_steps"] = All80Axes.HMC_STEPS
    row["hmc_rounds"] = All80Axes.HMC_ROUNDS
    row["hmc_target_round_seconds"] = All80Axes.HMC_TARGET_ROUND_SECONDS
    row["hmc_transitions"] = transitions
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
