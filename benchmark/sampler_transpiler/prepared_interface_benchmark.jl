# Focused wrapper-cost comparison. Both cases consume the same emitted program;
# the direct compiler baseline returns final output/RNG, without reusable state.
using ReactiveKernels, Reactant, Random, Statistics
include("prepared_hmc.jl")
const NSC = ReactiveKernels.NativeSlotCompiler
const TSC = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt).TracedSlotCompiler

function record!(rows, backend, steps, sample, path, phase, run)
    measurement = @timed run()
    push!(rows, (; backend, steps, sample, path, phase,
        seconds=measurement.time, allocated_bytes=measurement.bytes))
    measurement.value
end

function native_samples!(rows, prepared, steps)
    initial = initial_transpiled_state(prepared)
    seed = Xoshiro(91)
    prepared(initial, seed)
    function direct_input()
        owned = deepcopy(initial.storage)
        stores = map((x, old) -> (x, last(old)), owned, prepared.program.stores)
        merge(prepared.program, (; stores))
    end
    NSC.slot_chain(direct_input(), copy(seed), prepared.iterations)
    for sample in 1:7
        for path in (isodd(sample) ? (:prepared, :direct) : (:direct, :prepared))
            raw, rng = direct_input(), copy(seed)
            record!(rows, "native", steps, sample, string(path), "execute",
                path === :prepared ? () -> prepared(initial, rng) :
                    () -> NSC.slot_chain(raw, rng, prepared.iterations))
        end
    end
end

function reactant_samples!(rows, prepared, steps, transitions)
    traced = prepared.traced
    batch = TSC.TranspiledBatch(traced.f, traced.projector, traced.metadata, transitions)
    driver = (state, rng) -> begin
        result = batch(state, rng)
        (; outputs=result.outputs, argument=result.argument)
    end
    initial = initial_transpiled_state(prepared)
    rng = Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77]))
    direct = record!(rows, "reactant", steps, 0, "direct", "compile",
        () -> Reactant.compile(driver, (initial.storage, rng);
            TSC.cpu_compile_options()..., sync=true, donated_args=:none))
    prepared(initial, rng)
    direct(initial.storage, rng)
    for sample in 1:7
        for path in (isodd(sample) ? (:prepared, :direct) : (:direct, :prepared))
            record!(rows, "reactant", steps, sample, string(path), "execute",
                path === :prepared ? () -> prepared(initial, rng) :
                    () -> direct(initial.storage, rng))
        end
    end
end

function benchmark(path; transitions=10000)
    rows = NamedTuple[]
    for backend in (:native, :reactant), steps in (4, 16)
        rng = backend === :native ? Xoshiro(91) :
            Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77]))
        println("preparing ", backend, " steps=", steps); flush(stdout)
        prepared = record!(rows, string(backend), steps, 0, "prepared", "prepare",
            () -> PreparedHMCExample.prepare_hmc(rng; backend, transitions, steps))
        if backend === :native
            Base.invokelatest(native_samples!, rows, prepared, steps)
        else
            Base.invokelatest(reactant_samples!, rows, prepared, steps, transitions)
        end
        for output in ("direct", "prepared")
            times = [r.seconds for r in rows if r.backend == string(backend) &&
                r.steps == steps && r.path == output && r.phase == "execute"]
            println(backend, " steps=", steps, " ", output,
                " median_us=", median(times)*1e6/transitions)
        end
        flush(stdout)
    end
    open(path, "w") do io
        println(io, "backend,steps,sample,path,phase,seconds,allocated_bytes,transitions")
        for row in rows
            println(io, join(values(row), ','), ',', transitions)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(only(ARGS))
end
