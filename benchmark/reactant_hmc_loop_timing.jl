# Fair timing of the native vs Reactant HMC loop. The SAME authored @kernel is
# lowered to both backends by the merged transpiler (no hand-written Reactant
# kernel) — multinomial HMC on the bound Eight Schools density via the sampling
# lane's `sampler_transpiler` consumer. Compile + first-exec warmup excluded;
# whole batch synchronized; inputs preserved so state/rng are reused across
# replicates. Reports median µs per transition.
#
# Run (after the sampler_transpiler env is built):
#   julia benchmark/sampler_transpiler/setup.jl        # JULIA_NUM_PRECOMPILE_TASKS=1
#   julia --project=benchmark/sampler_transpiler benchmark/reactant_hmc_loop_timing.jl
# Optional AS_OUTPUT=<path.toml> writes a receipt.
import Reactant                       # load before any :reactant preparation
using ReactiveKernels
using Random, Statistics, Dates
import TOML
include(joinpath(@__DIR__, "sampler_transpiler", "prepared_hmc.jl"))
const P = PreparedHMCExample

function time_backend(backend, transitions, steps, rounds)
    rng = backend === :reactant ? Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77])) : Xoshiro(91)
    prog = P.prepare_hmc(rng; backend, transitions, steps)
    st = initial_transpiled_state(prog)
    prog(st, rng)                       # warmup (compile + first exec) — excluded
    ts = Float64[]
    for _ in 1:rounds
        push!(ts, @elapsed prog(st, rng))   # inputs preserved; batch synchronizes
    end
    med = median(ts)
    (per_transition_us = med / transitions * 1e6, batch_s = med)
end

const T = 1000
const STEPS = 16
const ROUNDS = 8
n = time_backend(:native, T, STEPS, ROUNDS)
r = time_backend(:reactant, T, STEPS, ROUNDS)
ratio = r.per_transition_us / n.per_transition_us
println("\n===== HMC loop (multinomial, Eight Schools density; L=$STEPS leapfrog, $T transitions/batch) =====")
println("  native   : $(round(n.per_transition_us; digits=3)) µs/transition")
println("  reactant : $(round(r.per_transition_us; digits=3)) µs/transition")
println("  reactant/native = $(round(ratio; digits=3))×  (lower = Reactant faster)")

outp = get(ENV, "AS_OUTPUT", "")
if outp != ""
    receipt = Dict("schema" => "reactant-hmc-loop-timing-v1", "generated_at" => string(now()),
        "methodology" => "same authored @kernel lowered to native + Reactant via the merged transpiler (no hand-written reactant kernel); multinomial HMC on the bound Eight Schools density; compile+first-exec warmup excluded; whole batch synchronized; median of $ROUNDS batches of $T transitions × L$STEPS",
        "environment" => Dict("julia" => string(VERSION), "arch" => string(Sys.ARCH), "cpu" => Sys.cpu_info()[1].model, "reactant_backend" => "default CPU"),
        "result" => Dict("native_us_per_transition" => n.per_transition_us,
                         "reactant_us_per_transition" => r.per_transition_us,
                         "reactant_over_native" => ratio,
                         "transitions_per_batch" => T, "leapfrog_steps" => STEPS))
    open(outp, "w") do io; TOML.print(io, receipt); end
    println("\nwrote receipt: $outp")
end
println("\nREACTANT_HMC_TIMING_DONE")
