# HMC / Eight Schools transpiler prototype

This executable prototype connects the existing reactive mathematical HMC and
leapfrog sources to the generic compiler and the bound Eight Schools density.
It reports preparation, compilation, and warm execution separately. The current
priority is efficient native lowering, followed by Reactant consuming that
simplified program. This remains an experimental compiler prototype.

The reusable compiler is owned by the package: the internal
`ReactiveKernels.NativeSlotCompiler` module prepares and emits native programs,
and the existing optional Reactant extension owns the traced backend. The
benchmark files `native_slots.jl`, `native_slots_factory.jl`, and
`traced_slots.jl` are import shims. Loading the core compiler does not load
Reactant. These internal entry points retain the prototype's finite support
limits; this move does not declare a stable public sampler or compiler API.

The emitted traced function accepts its ordinary runtime argument and returns
that argument alongside the updated state and counters. RNG packing is a
caller adapter, outside source lowering. The standalone
[`scalar_argument_probe.jl`](scalar_argument_probe.jl) uses the same native and
traced compiler on a scalar accumulation kernel, then reuses one traced
executable with two different runtime increments. It checks the source's
closed-form result and preservation of caller inputs independently in each
backend. `runtime-argument-v1.toml` records that probe and the HMC/control
rechecks after separating the RNG adapter from emitted code.

From the repository root:

```sh
julia benchmark/sampler_transpiler/setup.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl compare 1000
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl native-slots 1000
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl reactant-slots 1000
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl reactant-slots 1000 16
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/loop_control_probe.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/scalar_argument_probe.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/code_reuse_probe.jl checks
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/code_reuse_probe.jl timings
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/multinomial_eight_schools.jl native 1000 4
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/multinomial_eight_schools.jl reactant 1000 4
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/multinomial_eight_schools.jl diagnostics 1000 4
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/constant_binding_probe.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/range_draw_probe.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl reactant
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl ahmc
```

Append a positive batch length to any command, for example `reactant 1000`
and `ahmc 1000`, to compare the same number of chained transitions. The
`compare`, `native-slots`, and `reactant-slots` commands also accept a final
positive leapfrog count; the default is four. The comparison always uses that
same count for AdvancedHMC.

The environment pins Reactant 0.2.284 and Enzyme 0.13.199 and develops the local
compiler, distribution kernels, and PPL example packages. Setup creates a local
Manifest; it does not change the root project.

The mathematical sources are
[`hmc_state`](../reactivehmc_hmc_kernel_fixture_b.jl) and
[`euclidean_phasepoint` / `leapfrog!`](../nuts_kernel_authoring_fixture_b.jl).
The prototype compiles those sources, binds the generated endpoint and prepared
Eight Schools gradient, and supplies backend RNG state. The numerical-state
adapter, native type specialization, and compiled endpoint effect binding
currently use internal compiler APIs. Small typed callback handles keep prepared
program metadata outside the numerical state; their captured programs remain
fixed throughout execution.

`native-slots` composes captured HMC and free-method integrator MethodIR over
the existing typed canonical stores and prepared recipe handles. Derived values
are computed on demand, gradients use their destination form, and captured
integrator writes become in-place broadcasts inside the generated loop.
Preparation hoists pure expressions whose operands are fixed shared values.
A diagonal product can reuse its dying owned vector when earlier local reads
do not retain that vector. The emitter is in
[`src/native_slots.jl`](../../src/native_slots.jl); it
contains no sampler-name or model-name cases. Unsupported expression/control
forms fail during preparation.

The [native factory](../../src/native_slots_factory.jl) constructs these stores directly from the captured constructor
and bound callable sources. Consecutive transfers of all authoritative fields
between matching endpoint layouts also transfer valid derived caches. This
removes an unnecessary gradient evaluation at the start of each HMC transition:
the diagnostic run performs 4,000 gradients for 4,000 leapfrog steps across 1,000
transitions. Disabling this generic transfer optimization takes 5,000 gradients.

The program owns its stores and fixed compiler metadata for its entire lifetime.
Its private stores and constants are not an API for external mutation. The
source's known aliases and destination-preserving copies remain part of the
lowering contract.

`reactant-slots` consumes the same emitted native program. Its
[extension backend](../../ext/ReactiveKernelsReactantExt/traced_slots.jl)
turns numerical slots into local variables, derives explicit branch inputs and
outputs by backward liveness, and proves cache-validity facts across construction
and every possible transition exit. Only proven facts replace runtime checks.
Prepared gradient calls stage AD inside the enclosing Reactant compilation.
For this backend, the native emitter's `peel_loops=true` option emits the first
iteration separately, in source order. The remaining loop inherits established
cache facts. This adds at most one body, independent of the trip count. Native
execution keeps this option off because its measurements did not improve.
Fixed compiler metadata stays outside the numerical state. Preparation-fixed
integer ranges retain one traced loop body, with explicit live values carried
between iterations and an authored early return terminating the loop. Cache
facts account for loop entry, the backedge, and zero-trip execution. Dynamic
ranges remain outside this backend's support. The batch driver also uses a
traced loop. The optional `unroll_limit` compiler keyword permits bounded loop
expansion for ablation; the default retains every nonempty loop.

Repeated uses of a fixed authority share one metadata entry, using identity
to preserve distinct mutable objects. Cache flags proved constant at every
transition entry and exit stay out of the traced state interface. Temporarily
varying values inside control flow still follow the ordinary liveness analysis.

Repeated preparation of the same emitted syntax reuses its Julia function
within the process. The cache retains code only: every preparation still owns
its state and fixed metadata, and every Reactant compilation constructs a new
executable. Literal objects compare by exact identity, so equal contents do
not merge distinct mutable authorities. Inspection ASTs are separate from
cached keys. `reuse_code=false` on `compile_traced_slots` retains fresh function
emission for ablation. The cache does not change first-compilation cost or
persist across Julia processes.

`code_reuse_probe.jl checks` exercises independent initial states/bound gains,
changed loop bounds, distinct mutable literals, and inspection-tree edits.
The `timings` mode alternates code reuse with fresh emission for HMC in one
process and checks actual integration counts and caller inputs. Its first
compilation and subsequent warm compilations are reported separately.

`code-reuse-v1.toml` records a fresh process whose first Reactant compilation
takes 38.49 seconds. Repeated compilation with cached emitted code takes
0.619 and 0.625 seconds; interleaved fresh emissions of the same code take
8.06 and 8.73 seconds. Each case constructs a new executable, executes 4,000
integration steps, and preserves its caller inputs. This reduces repeated
compilation within one process; the first-compilation cost remains open.

The `reactant-slots` command interleaves synchronized execution with AdvancedHMC.
Every sample constructs fresh device inputs outside timing. The batch takes
private working copies once at entry and returns the final chain-phasepoint
fields, RNG seed, and counts. This keeps scratch mutations inside the compiled
batch and preserves its input state. Copies are outside the transition loop.
The underlying state ABI is private: callers must preserve its cached values
and compiler-proven currentness contract. `phasepoint_output=false` on
`run_traced_comparison` retains the earlier full-state mutation interface for
ablation.

`reactant-slots-v1.toml` records the fresh-process checkpoint: 41.9 seconds
of Reactant compilation and 2.43–2.84 ms per 1,000 transitions, with every batch
executing 4,000 leapfrog steps. AdvancedHMC's strongest batch in that run takes
2.13 ms. This establishes working execution of the simplified program; a clear
Reactant throughput win was still open at that checkpoint. The receipt separates model preparation,
native preparation, backend lowering, compilation, and first execution.

`reactant-slots-v2.toml` records retained-loop runs in two fresh processes.
Four and sixteen steps both compile in about 40 seconds; their IR sizes are
29,126 and 29,127 bytes. Four-step batches take 2.59–3.35 ms per 1,000
transitions, and sixteen-step batches take 6.94–9.46 ms. Every batch executes
the requested number of integrator calls. AdvancedHMC's strongest samples are
2.05 and 7.31 ms, respectively. This removes the previous small-loop limit
without code growth proportional to the trip count; a clear Reactant
throughput win was still open at that checkpoint.

`reactant-slots-v3.toml` records the first local throughput checkpoint in both
backends. Native batches take 1.46–1.55 ms per 1,000 transitions (median 1.52 ms),
and Reactant batches take 1.84–2.07 ms (median 1.92 ms). The strongest AdvancedHMC
batches in the respective fresh processes take 2.00 and 2.04 ms. Reactant is
faster in each of its seven paired samples; its margin is modest. Its compile
stage remains 40.2 seconds. Each batch executes 4,000 actual integration steps.
The updated zero-trip, divergence-return, and caller-input preservation probes
pass. This remains an experimental compiler checkpoint, without adaptation or
public sampler integration.

The same receipt includes a fresh sixteen-step run: Reactant takes 5.19–5.86 ms
per 1,000 transitions versus AdvancedHMC's strongest 7.49 ms. All batches
execute 16,000 steps. Compilation takes 41.8 seconds and the IR grows by one
byte, to 38,561 bytes.

`compilation-profile-v1.toml` records the subsequent representation cleanup
and compilation-cost investigation. Fixed metadata shrinks from 32 entries to
9, and 16 proved cache flags leave the traced state interface. Compilation
still takes about 40 seconds. Its allocations fall from about 3.95 to 3.79 GB,
and synchronized execution allocates 2,160 host bytes per batch instead of
2,608. These measurements do not establish a compilation-time speedup.

CPU sampling of the preceding checkpoint points mainly to Julia compiling
and running the tracing functions; XLA compilation accounts for a smaller
share. Sample counts overlap along call stacks and are not stage wall times.
The reusable attribution driver is:

```sh
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/profile_compile.jl /tmp/sampler-compile-profile
```

It writes a serialized profile and text views after compilation, then checks
the actual integration count and preservation of caller inputs. Profiling
substantially slows JIT compilation: use the ordinary `reactant-slots` command
for elapsed-time measurements. No backend optimization settings are changed.

`package-owned-v1.toml` records execution after moving these lowering passes
into the package and optional extension. The mathematical source and compiler
transformations are unchanged; the benchmark imports the package-owned code.
Historical receipts retain the source paths and hashes from their recorded
commits, before this move.

All modes use centered Eight Schools, ten Float64 parameters, a unit diagonal
metric, and step size 0.03. Four leapfrog steps are the default. This is endpoint-Metropolis HMC;
the separate multinomial case is described below. NUTS remains subsequent work.

## Captured multinomial HMC

[`multinomial_hmc_kernel.jl`](multinomial_hmc_kernel.jl) is a second mathematical
source for the same compiler. It places the initial point uniformly among the
`L+1` positions on a fixed trajectory, integrates forwards and backwards from
that point, and uses streaming Hamiltonian-weighted selection. Its two bound
integrator calls use the existing captured `leapfrog!` with opposite step sizes.
No sampler logic is implemented in a backend replacement.

[`multinomial_eight_schools.jl`](multinomial_eight_schools.jl) measures 1,000
consecutive transitions, seven interleaved samples, using the same model,
callbacks, metric, step size and leapfrog count as `AdvancedHMC.MultinomialTS`.
State/seed resets are outside timing; Reactant compiles the entire chained batch
and synchronizes its calls. Only final state is retained. Compilation, adaptation
and full chain storage are outside the timed workload. AdvancedHMC materializes
each trajectory and computes its usual acceptance statistics; the authored
streaming source currently omits those statistics. This is an execution
comparison with matched integration work, not complete sampler feature parity
or tuned posterior-sampling efficiency.

`multinomial-v1.toml` retains all seven samples from each initial comparison.
At four steps, native takes a median 1.83 ms per 1,000 transitions; the fastest
AdvancedHMC sample in that process takes 7.20 ms. At sixteen steps, Reactant
takes a median 15.18 ms versus AdvancedHMC's fastest 12.15 ms, so this longer
trajectory remains an optimization target. First Reactant compilation takes
44.49 seconds separately. An earlier four-step Reactant run takes a median
4.39 ms versus AdvancedHMC's fastest 6.64 ms; its recorded provenance predates
the final generalization of bound-range recognition. These are fixed-work
throughput observations, not effective-sample-size comparisons.

An optional fourth command-line argument writes the actual emitted `native.jl`
expression and, in Reactant mode, `traced.jl` and the final `kernel.mlir`:

```sh
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/multinomial_eight_schools.jl reactant 1000 16 /absolute/scratch/emissions
```

The Julia expressions are inspection artifacts whose stores, resources, and
metadata belong to their preparation; they are not standalone programs.

The compiler now accepts captured constant builtin scalar values/types in
method bodies. Preparation rejects nonconstant bindings and mutable global
objects. `constant_binding_probe.jl` exercises that boundary with actual captured
consumers; data and mutable state still belong in explicit prepared inputs.

The Reactant backend lowers preparation-fixed integer-range draws to unsigned
64-bit draws with rejection before reduction. This preserves a uniform result
without Julia's widened integer range-sampler intermediate, which this Reactant
version cannot trace. Signed/unsigned builtin ranges up to 64 bits are supported,
including full-width ranges; empty ranges and wider/custom integer types reject
at preparation. No RNG sequence matching is required. `range_draw_probe.jl`
exercises signed/full-width output, a range with frequent rejection, and caller
seed/output preservation. Its local scalar-indexing annotation only stores the
probe's draws; sampler execution does not use that annotation.

`loop_control_probe.jl` checks the same authored HMC source with zero steps and
with a divergence threshold that forces its early return. It exercises the
peeled loop and the batch's private state. Each backend must
execute the expected number of integrator calls and leave the initial position
unchanged. The traced batch must also preserve its caller inputs. These are
source-control invariants, without comparing random
trajectories or floating-point computations between implementations.

`compare` interleaves the new native program with AdvancedHMC using the same
prepared RK model and gradient, initial point, and integration settings.
Both implementations use the same small callback-handle representation.
Construction and resets occur outside warm timing. It reports actual integrator
calls to make a shortened trajectory visible in the benchmark.

The second native checkpoint (`native-first-v2.toml`) records all seven samples:
native execution takes 1.44–1.50 ms per 1,000 transitions, with 752 allocated
bytes per batch. AdvancedHMC's strongest batches take about 2.04–2.07 ms; slower
and GC-affected batches are retained in the receipt. Native endpoint preparation,
factory construction, and source emission together take about 16.7 seconds in
that fresh process, plus model preparation and first-call compilation. These
stage timings are measurements of this prototype, not an isolated comparison
of Julia compilation costs.

The earlier functional native path remains available as `native`. `reactant`
currently compiles a driver around that functional path; it has not yet adopted
the new slot-based program. Each batch restarts from the same initial state and
seed; transitions within it are chained. Reactant calls synchronize before
returning. The default batch length is 100; use the same explicit length for
comparisons.

These are execution probes without adaptation or ESS measurement. They do not
establish tuned sampling performance.

The first milestone is higher HMC throughput than AdvancedHMC in both backends.
The third receipt reaches this local execution checkpoint; reducing preparation
and compilation cost and integrating the generic lowering remain work.
The `ahmc` mode uses AdvancedHMC 0.8.6 endpoint HMC with the same bound RK density,
prepared gradient, initial position, metric, step size, and leapfrog count.
Initialization is outside its warm timing. Compilation is reported separately.
