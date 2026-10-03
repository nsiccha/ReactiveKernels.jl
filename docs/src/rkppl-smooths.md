# Smooths and HSGPs with `@rkppl`

`ReactiveKernelsPPL` provides spline and approximate Gaussian-process bases
as data functions, with small library submodels that state every prior.
The basis is computed once when data is bound; the model contains ordinary
matrix products, sampled coefficients and spectral weights.

## A penalized spline

```julia
using ReactiveKernelsPPL

model = @rkppl begin
    (Xf, Zp) = tps_basis(x; k = 4)
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    f ~ penalized_smooth(Xf, Zp)
    mu = a .+ f
    y .~ Normal.(mu, sigma)
end
plan = model(; x = X, y = Y)
built = build_kernel(plan)
```

`Xf` contains the centered linear null-space column; the model's intercept
owns the constant column. `Zp` contains the whitened penalized columns.
The submodel is exactly these statements:

```julia
@rkppl penalized_smooth(X, Z) = begin
    b[axes(X, 2)] .~ Flat.()
    sd ~ HalfNormal(1)
    z[axes(Z, 2)] .~ Normal.(0, 1)
    return X * b .+ Z * (sd .* z)
end
```

At `f ~ penalized_smooth(Xf, Zp)`, the sampled names are `f_b`, `f_sd`
and `f_z`. Write the statements directly to choose different priors.
Destructuring evaluates its data-only right-hand side once per bind.

## An approximate Gaussian process

```julia
model = @rkppl begin
    a ~ Normal(0, 1)
    (PHI, lambda) = hsgp_basis(x; k = 4)
    f ~ hsgp_effect(PHI, lambda)
    mu = a .+ f
    y .~ Normal.(mu, 1.5)
end
```

`PHI` holds the basis columns and `lambda` their per-axis eigenvalues.
The length-scale floor and every prior are visible in the library body:

```julia
@rkppl hsgp_effect(PHI, lambda) = begin
    rho_floor = maximum(hsgp_rho_floors(lambda))
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ LogNormal(0, 1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_sqrt_spd(lambda, sigma, rho) .* z)
end
```

The truncation is normalized as in Distributions.jl. Its lower bound may
be a nonnegative literal or a model-level data name. Multiple axes share
one length scale in `hsgp_effect`. For one length scale per axis, write
the sampled statements explicitly and use
`hsgp_sqrt_spd(lambda, sigma, [rho_1, rho_2])`.

## Tensor, periodic and grouped effects

| Data basis | Library submodel | Coefficient and scale priors |
| --- | --- | --- |
| `tps_basis(x; k)` → `(X, Z)` | `penalized_smooth(X, Z)` | Flat null space, standard Normal offsets, one HalfNormal scale |
| `t2_basis(x, z; k=(k1, k2))` → `(X, rr, rn, nr)` | `t2_smooth(X, rr, rn, nr)` | Flat null space, three standard Normal offset blocks and HalfNormal scales |
| `hsgp_basis(x...; k, c, domain)` → `(PHI, lambda)` | `hsgp_effect(PHI, lambda)` | Floored LogNormal length scale, LogNormal marginal scale, standard Normal offsets |
| `hsgp_periodic_basis(x; k, period)` → `(PHI, harmonics)` | `hsgp_periodic_effect(PHI, harmonics)` | Floored LogNormal length scale, LogNormal marginal scale, standard Normal offsets |
| `hsgp_basis(x; k, by=g)` → `(PHI, lambda)` | `hsgp_grouped_effect(PHI, lambda, g)` | Log-linear group length scales and marginal scales with Normal locations, HalfNormal scales and standard Normal offsets |

The grouped effect states its hyperparameters as ordinary sampled arrays:

```julia
@rkppl hsgp_grouped_effect(PHI, lambda, g) = begin
    rho_floor = maximum(hsgp_rho_floors(lambda))
    rho_mu ~ Normal(0, 1)
    rho_sd ~ HalfNormal(1)
    rho_z[levels(g)] .~ Normal.(0, 1)
    rho = max.(exp.(rho_mu .+ rho_sd .* rho_z), rho_floor)
    sigma_mu ~ Normal(0, 1)
    sigma_sd ~ HalfNormal(1)
    sigma_z[levels(g)] .~ Normal.(0, 1)
    sigma = exp.(sigma_mu .+ sigma_sd .* sigma_z)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_grouped_sqrt_spd(lambda, sigma, rho) .* z)
end
```

Group offsets follow `levels(g)` order. Group is the fastest column index
in the basis: column `(m - 1) * G + g` holds basis function `m` for group
`g`. The length scale is clamped at the shared validity floor, with zero
gradient below it.

The existing built-in smooth constructs remain available. On identical
data their columns match the data functions. At matched parameter values,
the library and built-ins use the same normalized priors, including
half-Normal scales and truncation at the fitted length-scale floor.

The [library bodies](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPL/src/library.jl)
and [ten corpus examples](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/packages/ReactiveKernelsPPL/test/corpus)
provide the complete statements for each variant. Native density and
gradient tests compare them with Distributions.jl and the built-in bases.

Spline, tensor, isotropic HSGP and grouped HSGP effects also compile with
Reactant, with native primal and gradient parity and fixed operation counts
as observation, basis and group sizes grow. Periodic HSGP currently supports
native execution and Enzyme gradients only: Reactant 0.2.289 lacks
`besselix(order, traced_x)`. The [backend-only reproducer](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/repro_reactant_besselix_order.jl)
and [core constraints](constraints.md) record that boundary.
