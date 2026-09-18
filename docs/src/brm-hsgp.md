# BRM motorcycle: two HSGPs with partial centering

This executable example translates BRM's motorcycle case study into one ordinary
ReactiveKernel. All 133 `MASS::mcycle` observations are retained. Time is scaled
to `[-1,1]`, acceleration is divided by its sample standard deviation, and the
mean and log standard deviation each use 20 squared-exponential HSGP basis
functions on `(-1.5,1.5)`. There are no population intercepts. The four positive
hyperparameters have `LogNormal(0,4)` priors.

The authored kernel below is read directly from
[`examples/brm_hsgp.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/brm_hsgp.jl).
The data checksum and source provenance are recorded with the benchmark.

```@eval
using Markdown
source = read(joinpath(@__DIR__, "..", "..", "examples", "brm_hsgp.jl"), String)
body = split(split(source, "# BEGIN MOTORCYCLE KERNEL\n")[2], "# END MOTORCYCLE KERNEL")[1]
Markdown.parse("```julia\n" * body * "```\n")
```

For basis frequency `ωⱼ = jπ/3`, the spectral log standard deviation is

```math
\ell_j = \log s + \tfrac12\log\rho + \tfrac14\log(2\pi)
          - \tfrac14\rho^2\omega_j^2.
```

The working coordinate is `vⱼ = zⱼ exp(cⱼ ℓⱼ)` with `zⱼ ~ Normal(0,1)`.
Thus `c=0` is noncentered, `c=1` is centered, and the basis weight is
`vⱼ exp((1-cⱼ)ℓⱼ)`. The working-coordinate log density includes
`-sum(c .* ℓ)` as its Jacobian. The four log hyperparameters have normalized
`Normal(0,4)` densities after combining each positive transform's Jacobian
with its `LogNormal` prior.

## Prepare and evaluate

The packed coordinate order is `(log ρμ, log sμ, vμ[1:20], log ρσ, log sσ,
vσ[1:20])`. The centeredness vector has the 20 mean entries followed by the
20 log-scale entries. Data-only basis and frequency work is folded once by
`bound=`. Both `q` and `c` remain runtime inputs, so a new online-selected
centeredness vector reuses the same executable.

```@example brm_hsgp
using ReactiveKernels
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "examples", "brm_hsgp.jl")) # hide
data = BRMHSGPExample.motorcycle_data(
    joinpath(@__DIR__, "..", "..", "examples", "data", "mcycle.csv"))
kernel = prepare(BRMHSGPExample.model;
    have=(:q, :c, :x, :y, :modes, :half_width), want=:posterior,
    bound=(; data..., modes=collect(1.0:20.0), half_width=1.5))
q = zeros(44)
q[[1,23]] .= -2
c = zeros(40)
kernel(q, c)
```

## Compile the value and gradient

The following uses the same graph and the public prepared AD boundary. The
focused Reactant test executes this path; the documentation build executes
the native interaction above.

```julia
using Reactant, Enzyme
using DifferentiationInterface: AutoEnzyme
backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
prepared = prepare_ad(kernel, backend, q, c; active=:q)
native_value, native_gradient = ad_value_and_gradient!(prepared, similar(q), q, c)
rq, rc = Reactant.to_rarray(q), Reactant.to_rarray(c)
compiled_primal = Reactant.@compile sync=true kernel(rq, rc)
value = compiled_primal(rq, rc)
compiled = compile_ad_value_and_gradient(prepared, rq, rc)
value, gradient = compiled(rq, rc)
```

Centeredness is held fixed when differentiating with respect to `q`. Changing
coordinates also requires transporting the position: for `c_old → c_new`,
multiply each working coefficient by `exp((c_new-c_old)*ℓ)`. Merely changing
`c` at a fixed `q` evaluates a different physical point.

## Measured performance

On the CPU benchmark, native Julia is faster for the primal. Native Enzyme
value-and-gradient evaluation is 1.02–1.20× StanBlocks in the longer measurement,
meeting the case study's `≤1.25×` target. The resident Reactant path is
1.50–1.98× StanBlocks and does not meet that target. The first, shorter receipt
overstated the native Enzyme gap; both receipts are retained for comparison.

The table reads the committed receipt. Times are warmed median microseconds
at one posterior position per frame, with up to 100,000 single-evaluation samples
and a two-second budget per measurement,
CPU affinity 6 on `strato2`, Julia 1.10.11, and one BLAS thread. The shared host
was not reserved. Resident Reactant calls synchronize before returning; the
host row also transfers the position and materializes both results. Julia
allocation counts exclude backend-managed memory. These are CPU measurements.

```@eval
using Markdown, TOML
receipt = TOML.parsefile(joinpath(@__DIR__, "..", "..", "benchmark", "receipts",
    "brm-hsgp-reactant-v2.toml"))
frames = ("noncentered", "selected_partial", "mixed", "centered")
paths = (("Julia primal", "rk_native_primal"),
    ("Reactant primal (resident)", "reactant_primal"),
    ("Enzyme value + gradient", "rk_native_value_gradient"),
    ("Enzyme + Reactant value + gradient (resident)", "reactant_resident"),
    ("Enzyme + Reactant value + gradient (host)", "reactant_host"),
    ("StanBlocks value + gradient", "stanblocks"),
    ("Native Turing + adaptive gradient wrapper", "native_turing"))
rows = ["| Evaluation | NCP (μs) | Selected partial (μs) | Mixed (μs) | Centered (μs) | NCP Julia bytes / allocations |",
        "|---|---:|---:|---:|---:|---:|"]
for (label, key) in paths
    measurements = [receipt[frame]["runtime"][key] for frame in frames]
    times = [string(round(m["median_ns"] / 1000; digits=2)) for m in measurements]
    m = first(measurements)
    push!(rows, "| $label | " * join(times, " | ") * " | $(m["bytes"]) / $(m["allocations"]) |")
end
Markdown.parse(join(rows, "\n"))
```

Preparation took 2.73 s for the bound kernel and 1.33 s for AD preparation.
The first native Enzyme call took 26.69 s. Reactant primal compilation took
22.53 s; subsequent value-and-gradient compilation took 10.03 s, followed by
a 0.30 s first execution. These stages ran in that order in one process, so
the second compiler measurement benefits from the first one's initialization.
None of those costs is included in the warmed table.

The compiled primal and value-and-gradient paths passed 40,072 comparisons
against both normalized StanBlocks and native Turing: both supplied 10,000-draw posterior bundles,
10,000 NCP positions transported to each of mixed and centered coordinates,
and 18 adversarial points in every frame. Maximum absolute density error was
`9.10e-13`; maximum componentwise gradient error scaled by `1+abs(reference)`
was `1.55e-11`. Absolute gradient errors are recorded too: centered coordinates
can produce enormous gradients, making an absolute tolerance alone misleading.

Native Turing parity is verified at BRM
`d137c326fa6a30cf173bf81fd6767e440bf025c0`, which corrected the explicit-prior
length-scale support mismatch. The generated fixed-frame DynamicPPL density
is checked directly. Its gradient control uses BRM's supported adaptive wrapper:
construct the NCP target, set the source centeredness, and retain the target
frame. This combines the generated density with an audited analytic HSGP
gradient and Enzyme coordinate transport; it is not direct Enzyme AD of
DynamicPPL. Those reference checks evaluate fixed posterior positions.

## Why the timings differ

The targets compute the same normalized density and derivatives, but the
executed operations differ. BRM specializes its generated NCP model to
standardized weights, while the RK example keeps centeredness live. The RK
lowering folds the basis once: its optimized StableHLO has a constant
`133×20` basis and no sine operation. It still performs the parameter-dependent
rescaling and projections on every call.

The native Enzyme allocation profile records 40 arrays per call: eight slices,
28 broadcast results, and four matrix-product results. AD preparation reuses
differentiation setup, not all those intermediates. Binding NCP centeredness
in a diagnostic reduces allocations from 40 to 32 and the native gradient
median from 8.42 to 6.95 μs. The corresponding Reactant median changes from
16.26 to 14.76 μs. These controls ran on CPU7 and should be compared within
their own run.

A cheap Reactant control returning the same scalar-plus-44-vector shape takes
5.76 μs. A quadratic control containing just the two basis projections and their
reverse projections already takes 15.98 μs through Reactant, versus 1.37 μs
for its native analytic evaluation. Grouping the right-hand sides changes
the Reactant control to 15.67 μs. Thus call overhead is material, and this
small dense linear-algebra workload reproduces most of the compiled model's
latency. This is a controlled comparison, not an additive timing breakdown or
a general conclusion about Enzyme or Reactant. See the
[diagnostic procedures and receipts](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/DIAGNOSTICS.md).

The [benchmark runner](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/compare.jl)
contains both the native Turing and StanBlocks comparisons. See its
[reproduction instructions](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/README.md)
for coordinate maps, posterior provenance, numerical errors, setup costs, and
warmed runtime measurements.

## Compile the HMC loop

The same motorcycle kernel also runs through the
[authored HMC transpiler](hmc-transpiler.md). Native uses prepared Enzyme
gradients; Reactant compiles differentiation together with momentum refreshes,
leapfrog integration, and multinomial proposal selection. This removes the
standalone gradient-call boundary from every leapfrog step.

The shared benchmark helper is
[`HMCBenchmark.prepare_hmc` / `benchmark_hmc`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/sampler_transpiler/hmc_benchmark.jl),
also used by the Eight Schools timing driver. It remains outside the package
API. This comparison uses the existing mathematical sampler source on both
backends, with 16 leapfrog steps per transition and step size 0.03.

Times below are warmed median **microseconds per transition**, from nine
batches on CPU6 of the shared host. The bracketed values are observed minimum
and maximum, not confidence intervals. Compilation garbage is collected once;
at least one second and three batches of additional warmup precede timing.
Each prepared-program call preserves its input state and returns final output
snapshots. Those wrapper costs are timed; Reactant synchronizes once per batch.

```@eval
using Markdown, TOML
receipts = Dict(backend => TOML.parsefile(joinpath(@__DIR__, "..", "..",
    "benchmark", "receipts", "brm-hsgp-hmc-$backend-v1.toml"))
    for backend in ("native", "reactant"))
rows = ["| Frame | Transitions/batch | Native + Enzyme (μs) | Reactant + Enzyme (μs) | Native / Reactant |",
        "|---|---:|---:|---:|---:|"]
for (frame, label) in (("noncentered", "NCP"), ("partial", "Selected partial")), t in (4, 100, 1000)
    measured = [receipts[b]["frames"][frame]["results"][string(t)] for b in ("native", "reactant")]
    format(m) = string(round(m["median_us_per_transition"]; digits=2), " [",
        round(minimum(m["raw_batch_seconds"]) * 1e6 / t; digits=2), "–",
        round(maximum(m["raw_batch_seconds"]) * 1e6 / t; digits=2), "]")
    ratio = round(measured[1]["median_us_per_transition"] / measured[2]["median_us_per_transition"]; digits=2)
    push!(rows, "| $label | $t | $(format(measured[1])) | $(format(measured[2])) | $(ratio)× |")
end
Markdown.parse(join(rows, "\n"))
```

At 1,000 transitions, Reactant is about 2.3× faster in both frames. The shared
host produced substantial variation, so these results support the throughput
advantage more strongly than a precise scaling curve. First HMC preparation
took 14.75 s native and 66.74 s Reactant, in addition to density/AD preparation
and the first native gradient. Subsequent preparations benefited from compiler
initialization. Raw timings and all preparation stages are in the
[full HMC report](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/HMC.md).

The density retains `(q,c)` inputs and folds data-only transformations once;
the sampler holds centeredness fixed in its callback context. Both backends
start at posterior column 5,000 and use the same frame-specific diagonal mass,
the inverse coordinate variance of the supplied 10,000 draws. This fixed mass
is posterior-informed benchmark setup. All 108 timed batches and 48 continued
batches ended at finite, moved positions with finite densities. Different RNG
engines are used; trajectories are not required to match.

This measures fixed-work HMC throughput. Adaptation, history storage, and ESS
are outside the measurement, so the timing ratio is not an effective-samples
ratio.
