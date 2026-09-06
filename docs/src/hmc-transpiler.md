# HMC from reactive mathematical source

This experimental compiler lowers the same captured HMC source to native Julia
and Reactant. The measured workloads chain endpoint or multinomial HMC transitions over the
[centered Eight Schools model](eight-schools.md). The compiler owns state,
cache reuse, control flow and backend lowering; the sampler remains mathematical
`@kernel` source.

## The kernel being compiled

This block is read directly from the executable benchmark source. `step_f` is
the captured Euclidean leapfrog method, whose position, momentum, potential,
gradient and Hamiltonian dependencies come from the reactive phasepoint.

```@eval
Main.HMCTranspilerDocs.render_source()
```

Each transition refreshes momentum, chooses a forward/backward trajectory split,
and selects a position using streaming multinomial weights. Only the selected
position is retained between transitions. Momentum is refreshed next time; this
source does not return the selected joint phasepoint. The original full-phasepoint
variant remains in the
[benchmark family](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/benchmark/sampler_transpiler).

## Multinomial HMC throughput

These are microseconds per transition at **10,000 chained transitions**. Native
and Reactant columns are medians of seven execution samples. AdvancedHMC uses
its fastest sample across both separately paired comparison groups. The table
is computed from the committed data during the docs build.

```@eval
Main.HMCTranspilerDocs.render_results()
```

The run used strato2 CPU, Julia 1.10.11, Reactant 0.2.284,
AdvancedHMC 0.8.6, ten Float64 parameters, a unit diagonal metric and step size
0.03. Each backend uses the same RK density and gradient, with the indicated
leapfrog count. AdvancedHMC additionally materializes trajectories and computes
acceptance statistics; this minimal kernel streams selection and omits those
statistics. The measurements cover fixed integration work, without adaptation,
chain history or effective-sample-size estimates.

The complete chain executes in **one synchronized Reactant call**, including
gradients and random draws. Preparation, input resets and compilation are outside
execution timing. Both generated backends beat the strongest paired AdvancedHMC
sample at all six measured workloads: 100, 1,000 and 10,000 transitions at four
and sixteen steps. Reactant still loses to native for 100 four-step transitions:
2.32 versus 1.70 μs per transition.

[Raw samples](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/sampler_transpiler/position-multinomial-scaling-v2.csv)
retain all 168 execution observations and six separate compilation rows. The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/sampler_transpiler/position-multinomial-scaling-v2.toml)
pins source files and records the environment and measurement contract.

## Matched endpoint HMC throughput

Endpoint HMC is compared separately with **endpoint HMC**. This minimal source
uses the same captured phasepoint/leapfrog and the same generic compiler. It
performs exactly the requested number of steps and one final Metropolis
acceptance decision, without per-step divergence observations or early returns.
The older endpoint source with those observations remains a diagnostic fixture.

```@eval
Main.HMCTranspilerDocs.render_source("endpoint_hmc_kernel.jl")
```

The following values are medians of seven alternating execution samples at
**10,000 transitions**. All four cases run in the same process, with the same
model, metric, step size, fixed step count and final-position/RNG output.
Reactant calls are synchronized. Preparation, reset and compilation are outside
timing; adaptation and chain history are absent. RK step counters are disabled.
Work counts follow the fixed source/emitted loops and AdvancedHMC's `FixedNSteps`.

```@eval
Main.HMCTranspilerDocs.render_endpoint_results()
```

ProbProg uses `mcmc_logpdf(...; algorithm=:HMC)`, supplying the initial potential
and gradient outside timing. Both Reactant cases use the same 64 KiB CPU loop
policy. Native RK uses the same mathematical source with native lowering;
AdvancedHMC uses `EndPointTS`. Implementations use their own random streams and
may differ in internal handling of unused statistics. These are execution-time
comparisons, not claims of equal sampling efficiency.

The [endpoint scaling data](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/sampler_transpiler/matched-endpoint-scaling-v1.csv)
retain every execution replicate at 100, 1,000 and 10,000 transitions. The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/sampler_transpiler/matched-endpoint-scaling-v1.toml)
records source hashes, compilation order and emitted loops. The earlier chart
placed ProbProg endpoint timings beside RK multinomial timings; that comparison
did not isolate compiler overhead. The sampler families are now separate.

## Generated execution and compilation cost

Native lowering uses fixed typed stores and prepared derivative destinations.
Compiler-known source copies preserve reusable caches, avoiding extra gradient
evaluations. Reactant consumes that generated native program, carrying numerical
values through retained loops. Fixed metadata stays outside the traced state;
preparation proves which cache checks can disappear. Neither compiler contains
HMC or Eight Schools cases.

On this CPU, XLA initially dispatched many small operations inside each leapfrog
loop. The optional per-compilation `cpu_compile_options()` helper lets XLA pack
eligible loops into computation kernels using a 64 KiB small-loop threshold.
It preserves explicit user options and XLA's eligibility checks. Sixteen-step
execution improves from roughly 12–14 to 2.6–2.8 μs per transition in the
controlled ablation, without expanding loops by trip count. The
[loop-packing receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/sampler_transpiler/cpu-loop-packing-v1.toml)
records both measurements and emitted-HLO evidence.

**Cold Reactant compilation still takes about 44 seconds.** Later compilations
in the same process reuse emitted Julia code and are measured separately.
ProbProg's first compilation in its measurement process takes 21.86 seconds;
its five subsequent compilations take 0.52–0.61 seconds.

## Reproduce and current limits

```sh
julia benchmark/sampler_transpiler/setup.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/position_multinomial_scaling.jl /absolute/output.csv
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/cpu_loop_packing.jl /absolute/ablation-directory
timeout --signal=TERM --kill-after=15s 600s julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/matched_endpoint_comparison.jl /absolute/endpoint-scaling scaling
```

The compiler assumes fixed types, array shapes, call graph and compiler-known
mutation. Unsupported syntax or unknown mutation rejects at preparation.
`ReactiveKernels.NativeSlotCompiler` and the optional Reactant lowering remain
internal experimental entry points. Backend RNG specialization preserves the
source's random operations without requiring matching streams or trajectories.

This is the working HMC checkpoint. Optimized NUTS, then WALNUTS/WALNUTPIE,
remain further work; user-defined online statistics are a separate extension of
the minimal source. The
[executable guide](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/sampler_transpiler/README.md)
describes compiler entry points, focused probes and earlier experiments.
