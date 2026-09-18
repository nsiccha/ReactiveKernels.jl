# Fast RK-focused Reactant phase for the all-82 posteriordb benchmark.
#
# This process deliberately does NOT load DynamicPPL/Turing or evaluate reference Stan. Native
# parity is established by the native phase and every model package test. Here BridgeStan is
# used only to read the unconstrained dimension from the already-cached PosteriorDB `.so`; the
# faithful RK graph receives the complete real dataset and each Reactant operation is attempted
# independently. A cell is either a timing or its exact lowering/setup diagnostic.
using Random, LinearAlgebra, Statistics
import TOML
import SHA

const PHASE = "reactant"
const RECEIPT = get(ENV, "RK_ALL80_RECEIPT", "")
const BATCH = get(ENV, "RK_ALL80_BATCH", "")
include(joinpath(@__DIR__, "all80_receipt.jl"))    # stdlib-only; load the parent lock before measured imports
if BATCH != ""
    All80Receipt.load_provenance!(ENV["RK_ALL80_SOURCE_LOCK"])
end
using Chairmarks: @be
import BridgeStan, PosteriorDB, Reactant, Enzyme
using ReactiveKernels, ReactiveKernelsPPLExamples
using DifferentiationInterface
include(joinpath(@__DIR__, "all80_registry.jl"))
include(joinpath(@__DIR__, "all80_axes.jl"))
const RK = All80Registry.REGISTRY

const STD = joinpath(@__DIR__, "sampler_transpiler")
include(joinpath(STD, "eight_schools_density.jl"))
include(joinpath(@__DIR__, "nuts_kernel_authoring_fixture_b.jl"))
include(joinpath(STD, "position_multinomial_hmc_kernel.jl"))
using .EightSchoolsDensity: Potential, Gradient, CallbackHandle
const F = NUTSBMutationAuthoringFixture

const AE = AutoEnzyme(mode = Enzyme.Reverse)
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
if BATCH != ""
    All80Receipt.certify_loaded_modules!(Dict(
        "ReactiveKernels" => ReactiveKernels,
        "ReactiveKernelsDistributionKernels" => "ReactiveKernelsDistributionKernels",
        "ReactiveKernelsPPLExamples" => ReactiveKernelsPPLExamples))
    snapshot = All80Receipt.verify_provenance()
    snapshot["ad_backend"] == All80Receipt.ORDINARY_AD_BACKEND ||
        error("all80 Reactant batch AD configuration mismatch: expected ordinary reverse without annotation")
end
_phase_prov() = BATCH == "" ? nothing : All80Receipt.verify_provenance()

function cached_bridge_model(post)
    stan = PosteriorDB.implementation(PosteriorDB.model(post), "stan")
    stan_path = PosteriorDB.path(stan)
    library = first(splitext(stan_path)) * "_model.so"
    isfile(library) || error("cached BridgeStan library missing: $library")
    BridgeStan.StanModel(library, PosteriorDB.load(PosteriorDB.dataset(post), String), 468)
end

stan_val(sm, q) = BridgeStan.log_density(sm, q; propto = false, jacobian = true)
stan_grad(sm, q) = BridgeStan.log_density_gradient(
    sm, q; propto = false, jacobian = true)[2]
_numeric_vector_sha256(values) =
    bytes2hex(SHA.sha256(sprint(show, Float64.(values))))

function reference_valid_probe(sm, dim, entry; rng = Xoshiro(0xC0FFEE))
    # Select by REFERENCE validity, never by RK success. A registry probe is authoritative and
    # must itself be reference-valid; generic candidates are tried in deterministic order. An RK
    # failure at the selected point is preserved by `prepare_ad`/`reactant_cells`, not resampled.
    sperm = entry.stan_perm
    back = sperm === nothing ? nothing : sortperm(sperm)
    candidates = entry.probe_q === nothing ?
        Iterators.flatten((
            (fill(0.1, dim), fill(-0.1, dim)),
            (0.5 .* randn(rng, dim) for _ in 1:96),
            (zeros(dim),))) :
        (collect(Float64, entry.probe_q),)
    tested = 0
    for candidate in candidates
        q = collect(Float64, candidate)
        length(q) == dim || continue
        tested += 1
        stan_q = sperm === nothing ? q : q[sperm]
        value = try stan_val(sm, stan_q) catch; NaN end
        gradient = try stan_grad(sm, stan_q) catch; fill(NaN, dim) end
        mapped_gradient = back === nothing ? gradient : gradient[back]
        if isfinite(value) && length(mapped_gradient) == dim &&
                all(isfinite, mapped_gradient)
            return (; q = q, candidates_tested = tested,
                selected_source = entry.probe_q === nothing ? "generated" : "registry",
                reference_value = value, reference_gradient = mapped_gradient,
                reference_gradient_sha256 = _numeric_vector_sha256(mapped_gradient))
        end
    end
    error("no reference-valid finite value/gradient probe for a $dim-dim model after $tested candidate(s)")
end

function _input_identity(post, probe, name, entry)
    dataset = PosteriorDB.dataset(post)
    stan_path = PosteriorDB.path(PosteriorDB.implementation(
        PosteriorDB.model(post), "stan"))
    data_json = PosteriorDB.load(dataset, String)
    library = first(splitext(stan_path)) * "_model.so"
    All80Receipt.input_identity(;
        phase = PHASE, posterior = name,
        model = PosteriorDB.name(PosteriorDB.model(post)),
        stan_path = stan_path, stan_sha256 = bytes2hex(SHA.sha256(read(stan_path))),
        library = library,
        library_sha256 = isfile(library) ? bytes2hex(SHA.sha256(read(library))) : "not-present",
        dataset_path = try PosteriorDB.path(dataset) catch; "unavailable" end,
        dataset_json_sha256 = bytes2hex(SHA.sha256(data_json)),
        dataset_json_bytes = sizeof(data_json),
        query = Dict{String,Any}(
            "selected_point" => probe.q,
            "selected_source" => probe.selected_source,
            "candidates_tested" => probe.candidates_tested,
            "reference_value" => probe.reference_value,
            "reference_gradient_sha256" => probe.reference_gradient_sha256,
            "selection_rule" => "first reference-valid finite BridgeStan value and gradient; RK failures are not skipped",
            "ad_backend" => All80Receipt.ORDINARY_AD_BACKEND),
        stan_perm = entry.stan_perm)
end

function run_reactant_one(name)
    println("\n===== [reactant-fast] $name =====")
    post = PosteriorDB.posterior(PosteriorDB.database(), name)
    data = PosteriorDB.load(PosteriorDB.dataset(post))
    entry = RK[name]
    graph = getproperty(getproperty(ReactiveKernelsPPLExamples, entry.mod), entry.build)()
    kb = prepare(graph; have = entry.have, want = :posterior, bound = entry.bind(data))
    sm = cached_bridge_model(post)
    probe = reference_valid_probe(sm, Int(BridgeStan.param_unc_num(sm)), entry)
    q = probe.q
    input_identity = _input_identity(post, probe, name, entry)
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
    # The reference-valid selector already produced the BridgeStan gradient mapped to RK order.
    grad_oracle = probe.reference_gradient
    transitions = All80Axes.HMC_MIN_TRANSITIONS
    row = reactant_cells(kb, prep, q; transitions, grad_oracle,
        stan_value = probe.reference_value, rk_offset = entry.off_rk)
    row["hmc_steps"] = All80Axes.HMC_STEPS
    row["hmc_rounds"] = All80Axes.HMC_ROUNDS
    row["hmc_target_round_seconds"] = All80Axes.HMC_TARGET_ROUND_SECONDS
    row["hmc_reactant_transitions"] = transitions   # DISTINCT from native's calibrated hmc_transitions
    # (fixed at HMC_MIN_TRANSITIONS=4; surfaces the batch-size asymmetry + avoids an aggregate cell clash)
    for key in ("primal_rk_reactant", "gradient_rk_reactant", "hmc_rk_reactant")
        value = row[key]
        println("  $key = ", value isa Real ? string(round(value; sigdigits = 4)) : first(value, 180))
    end
    row["input_identity_reactant"] = input_identity
    row
end

requested = [name for name in ARGS if !startswith(name, "-")]
duplicates = unique(filter(name -> count(==(name), requested) > 1, requested))
isempty(duplicates) ||
    error("all80 reactant: duplicate key(s): $(join(duplicates, ", "))")
unknown = [name for name in requested if !haskey(RK, name)]
isempty(unknown) || error("all80 reactant: unknown key(s): $(join(unknown, ", "))")
# Match the native default: the immutable frozen-82 sweep excludes incremental batch keys.
targets = isempty(requested) ?
    sort(collect(setdiff(keys(RK), All80Registry.BATCH1_KEYS))) : requested
retry_requested = get(ENV, "RK_ALL80_RETRY", "") == "1"
retry_requested && isempty(requested) &&
    error("RK_ALL80_RETRY=1 requires explicit model keys; refusing to replay all 82 implicitly")
rows = Dict{String,Any}()
if get(ENV, "RK_ALL80_RESUME", "") == "1" && RECEIPT != "" && isfile(RECEIPT)
    prior = TOML.parsefile(RECEIPT)
    get(prior, "schema", "") == All80Receipt.SCHEMA || error("reactant resume schema mismatch")
    get(prior, "phase", "") == PHASE || error("reactant resume phase mismatch")
    BATCH == "" || All80Receipt.assert_batch_resume!(prior, RECEIPT;
        phase = PHASE, targets = targets, provenance = All80Receipt.verify_provenance())
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
    All80Receipt.write_phase(RECEIPT, PHASE, rows; provenance = _phase_prov())
    println("  [$(length(rows))/$(length(targets)) complete; $index/$(length(pending)) this run] $name recorded")
    flush(stdout)
end
if BATCH != ""
    sort(collect(keys(rows))) == sort(targets) ||
        error("all80 Reactant batch incomplete: got $(sort(collect(keys(rows)))) expected $(sort(targets))")
end
println("REACTANT_PHASE_DONE rows=$(length(rows))")
