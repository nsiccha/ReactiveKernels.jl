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

The program owns its stores and fixed compiler metadata for its entire lifetime.
Its private stores and constants are not an API for external mutation. The
source's known aliases and destination-preserving copies remain part of the
lowering contract.

All modes use centered Eight Schools, ten Float64 parameters, a unit diagonal
metric, step size 0.03, and four leapfrog steps. This is endpoint-Metropolis HMC;
multinomial HMC and NUTS remain subsequent work.

`compare` interleaves the new native program with AdvancedHMC using the same
prepared RK model and gradient, initial point, and integration settings.
Construction and resets occur outside warm timing. It reports actual integrator
calls to make a shortened trajectory visible in the benchmark.

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
