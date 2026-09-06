# HMC / Eight Schools transpiler prototype

This executable prototype connects the existing reactive mathematical HMC and
leapfrog sources to the generic compiler and the bound Eight Schools density.
It reports preparation, compilation, and warm execution separately. The current
priority is efficient native lowering, followed by Reactant consuming that
simplified program. This remains an experimental compiler prototype.

From the repository root:

```sh
julia benchmark/sampler_transpiler/setup.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl compare 1000
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl native-slots 1000
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl reactant-slots 1000
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl reactant
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/hmc_eight_schools.jl ahmc
```

Append a positive batch length to any command, for example `reactant 1000`
and `ahmc 1000`, to compare the same number of chained transitions.

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
do not retain that vector. This prototype emitter is in `native_slots.jl`; it
contains no sampler-name or model-name cases. Unsupported expression/control
forms fail during preparation.

The native factory constructs these stores directly from the captured constructor
and bound callable sources. Consecutive transfers of all authoritative fields
between matching endpoint layouts also transfer valid derived caches. This
removes an unnecessary gradient evaluation at the start of each HMC transition:
the diagnostic run performs 4,000 gradients for 4,000 leapfrog steps across 1,000
transitions. Disabling this generic transfer optimization takes 5,000 gradients.

The program owns its stores and fixed compiler metadata for its entire lifetime.
Its private stores and constants are not an API for external mutation. The
source's known aliases and destination-preserving copies remain part of the
lowering contract.

`reactant-slots` consumes the same emitted native program. Its backend pass
turns numerical slots into local variables, derives explicit branch inputs and
outputs by backward liveness, and proves cache-validity facts across construction
and every possible transition exit. Only proven facts replace runtime checks.
Prepared gradient calls stage AD inside the enclosing Reactant compilation.
Fixed compiler metadata stays outside the numerical state. The pilot expands
fixed integer loops of at most eight iterations; larger or dynamic loops are
outside this backend's current support. The batch driver uses a traced loop.

This command interleaves synchronized Reactant execution with AdvancedHMC.
Every sample constructs fresh device inputs outside timing, because this
generated program mutates its owned buffers. The state ABI is private: callers
must preserve its cached values and compiler-proven currentness contract.

`reactant-slots-v1.toml` records the fresh-process checkpoint: 41.9 seconds
of Reactant compilation and 2.43–2.84 ms per 1,000 transitions, with every batch
executing 4,000 leapfrog steps. AdvancedHMC's strongest batch in that run takes
2.13 ms. This establishes working execution of the simplified program; a clear
Reactant throughput win remains open. The receipt separates model preparation,
native preparation, backend lowering, compilation, and first execution.

All modes use centered Eight Schools, ten Float64 parameters, a unit diagonal
metric, step size 0.03, and four leapfrog steps. This is endpoint-Metropolis HMC;
multinomial HMC and NUTS remain subsequent work.

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
The `ahmc` mode uses AdvancedHMC 0.8.6 endpoint HMC with the same bound RK density,
prepared gradient, initial position, metric, step size, and leapfrog count.
Initialization is outside its warm timing. Compilation is reported separately.
