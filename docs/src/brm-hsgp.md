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
value-and-gradient evaluation is faster than the compiled Reactant path or
similar to it, depending on the coordinate frame. Reactant does not meet the
case study's `≤1.25× StanBlocks` value-and-gradient runtime target in this run.

The table reads the committed receipt. Times are warmed median microseconds
at one posterior position per frame, with 1,000 single-evaluation samples,
CPU affinity 6 on `strato2`, Julia 1.10.11, and one BLAS thread. The shared host
was not reserved. Resident Reactant calls synchronize before returning; the
host row also transfers the position and materializes both results. Julia
allocation counts exclude backend-managed memory. These are CPU measurements.

```@eval
using Markdown, TOML
receipt = TOML.parsefile(joinpath(@__DIR__, "..", "..", "benchmark", "receipts",
    "brm-hsgp-reactant-v1.toml"))
frames = ("noncentered", "selected_partial", "mixed", "centered")
paths = (("Julia primal", "rk_native_primal"),
    ("Reactant primal (resident)", "reactant_primal"),
    ("Enzyme value + gradient", "rk_native_value_gradient"),
    ("Enzyme + Reactant value + gradient (resident)", "reactant_resident"),
    ("Enzyme + Reactant value + gradient (host)", "reactant_host"),
    ("StanBlocks value + gradient", "stanblocks"))
rows = ["| Evaluation | NCP (μs) | Selected partial (μs) | Mixed (μs) | Centered (μs) | Julia bytes / allocations |",
        "|---|---:|---:|---:|---:|---:|"]
for (label, key) in paths
    measurements = [receipt[frame]["runtime"][key] for frame in frames]
    times = [string(round(m["median_ns"] / 1000; digits=2)) for m in measurements]
    m = first(measurements)
    push!(rows, "| $label | " * join(times, " | ") * " | $(m["bytes"]) / $(m["allocations"]) |")
end
Markdown.parse(join(rows, "\n"))
```

Preparation took 4.20 s for the bound kernel and 1.82 s for AD preparation.
The first native Enzyme call took 29.80 s. Reactant primal compilation took
24.93 s; subsequent value-and-gradient compilation took 12.42 s, followed by
a 0.39 s first execution. These stages ran in that order in one process, so
the second compiler measurement benefits from the first one's initialization.
None of those costs is included in the warmed table.

The compiled primal and value-and-gradient paths passed 40,072 comparisons
against normalized StanBlocks: both supplied 10,000-draw posterior bundles,
10,000 NCP positions transported to each of mixed and centered coordinates,
and 18 adversarial points in every frame. Maximum absolute density error was
`9.10e-13`; maximum componentwise gradient error scaled by `1+abs(reference)`
was `1.55e-11`. Absolute gradient errors are recorded too: centered coordinates
can produce enormous gradients, making an absolute tolerance alone misleading.

Native Turing parity remains unverified. At the recorded BRM revision, its
generated target retains a length-scale floor near `0.20518` despite the explicit
prior, while Stan uses the required zero lower bound. Initialization at `ρ=0.2`
therefore fails. The receipt explicitly records `native_verified=false`; the
runner's default native comparison fails rather than accepting this mismatch.

The [benchmark runner](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/compare.jl)
contains both the native Turing and StanBlocks comparisons. See its
[reproduction instructions](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/README.md)
for coordinate maps, posterior provenance, numerical errors, setup costs, and
warmed runtime measurements.
