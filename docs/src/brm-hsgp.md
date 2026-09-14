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

The [benchmark runner](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/compare.jl)
compares normalized values and all 44 gradients against the independently
generated BRM native Turing and StanBlocks targets. See its
[reproduction instructions](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/brm_hsgp/README.md)
for coordinate maps, posterior provenance, numerical errors, setup costs, and
warmed runtime measurements.
