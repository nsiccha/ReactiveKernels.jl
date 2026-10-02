# Shipped @rkppl library submodels: ordinary submodels written in the
# surface language with every default prior stated in the body, so a
# program reads (and rewrites) exactly what the use site expands to. A use
# site `y ~ sm(...)` / `z ~ sm(...)` resolves here through the exported
# binding (the caller's module sees it via `using ReactiveKernelsPPL`) and
# lowers to the identical plan as the hand-inlined body (see
# [`RKPPLSubmodel`](@ref)): body names namespace under the use-site
# left-hand side. Replace a local declaration with `merge(model, :(z.tau ~
# HalfCauchy(0.5)))`, pin it with `merge(model, (; var"z.tau" = 0.3))`, or
# derive a reusable variant with `merge(submodel, :(tau ~ HalfCauchy(0.5)))`.

"""
    log_F ~ linear_pk_log_f(sched; k = 5, c = 1.5)

Linear dose effect plus an exponentiated-quadratic HSGP over the schedule's
log-dose axis, in schedule operation order. Every parameter and prior is
stated in the library body; `rho_floor` is the data-derived validity floor.
Use `merge(linear_pk_log_f, :(slope ~ Normal(0, 0.5)))` to make a new
submodel with a different slope prior. The original body is unchanged.
"""
@rkppl linear_pk_log_f(sched; k = 5, c = 1.5) = begin
    reference_dose = 1
    op_log_dose = linear_pk_op_log_dose(sched.op_type, sched.op_amount;
        reference_dose = reference_dose)
    (PHI, lambda) = hsgp_basis(op_log_dose; k = k, c = c)
    rho_floor = maximum(hsgp_rho_floors(lambda))
    slope ~ Normal(0, 1)
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ HalfNormal(1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return slope .* op_log_dose .+
        PHI * (hsgp_sqrt_spd(lambda, sigma, rho) .* z)
end

"""
    y ~ ordered_logistic(eta)

Shipped observation-stream submodel for a cumulative-logit ordinal response
(levels `1..K`). Its body states the default cutpoint prior and observes the
response:

```julia
@rkppl ordered_logistic(eta) = begin
    cutpoints ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
    y .~ OrderedLogistic.(eta, Ref(cutpoints))
    return y
end
```

At `y ~ ordered_logistic(eta)` the cutpoints are local `y.cutpoints` and
restored as `nt.y.cutpoints`. Replace their prior through `y.cutpoints`
with `merge`, or derive a variant of `ordered_logistic`.
"""
@rkppl ordered_logistic(eta) = begin
    cutpoints ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
    y .~ OrderedLogistic.(eta, Ref(cutpoints))
    return y
end

"""
    f ~ penalized_smooth(X, Z)

Shipped latent submodel for a penalized spline over the bases of
[`tps_basis`](@ref): `X` the unpenalized null-space columns, `Z` the
penalized (whitened) range-space columns. Its body states every prior:

```julia
@rkppl penalized_smooth(X, Z) = begin
    b[axes(X, 2)] .~ Flat.()
    sd ~ HalfNormal(1)
    z[axes(Z, 2)] .~ Normal.(0, 1)
    return X * b .+ Z * (sd .* z)
end
```

At `f ~ penalized_smooth(Xf, Zp)` the parameters are `f.b`, `f.sd` and
`f.z` (restored as `nt.f.b`, `nt.f.sd`, `nt.f.z`), and bare `f` is one value
per observation:

```julia
(Xf, Zp) = tps_basis(x; k = 10)
f ~ penalized_smooth(Xf, Zp)
mu = a .+ f
```

The null-space coefficients are flat, as in the built-in `spline_basis`;
the smoothing sd is a proper half-Normal (the built-in's is Stan's
unnormalized half, `log(2)` lower). To change a prior, write the
statements yourself.
"""
@rkppl penalized_smooth(X, Z) = begin
    b[axes(X, 2)] .~ Flat.()
    sd ~ HalfNormal(1)
    z[axes(Z, 2)] .~ Normal.(0, 1)
    return X * b .+ Z * (sd .* z)
end

"""
    f ~ t2_smooth(X, Zrr, Zrn, Znr)

Shipped latent submodel for a tensor-product (`t2`) spline over the bases
of [`t2_basis`](@ref): one flat null-space block and three penalized
blocks, each with its own smoothing sd. Its body:

```julia
@rkppl t2_smooth(X, Zrr, Zrn, Znr) = begin
    b[axes(X, 2)] .~ Flat.()
    sd[1:3] .~ HalfNormal.(1)
    z_rr[axes(Zrr, 2)] .~ Normal.(0, 1)
    z_rn[axes(Zrn, 2)] .~ Normal.(0, 1)
    z_nr[axes(Znr, 2)] .~ Normal.(0, 1)
    return X * b .+ Zrr * (sd[1] .* z_rr) .+ Zrn * (sd[2] .* z_rn) .+
        Znr * (sd[3] .* z_nr)
end
```
"""
@rkppl t2_smooth(X, Zrr, Zrn, Znr) = begin
    b[axes(X, 2)] .~ Flat.()
    sd[1:3] .~ HalfNormal.(1)
    z_rr[axes(Zrr, 2)] .~ Normal.(0, 1)
    z_rn[axes(Zrn, 2)] .~ Normal.(0, 1)
    z_nr[axes(Znr, 2)] .~ Normal.(0, 1)
    return X * b .+ Zrr * (sd[1] .* z_rr) .+ Zrn * (sd[2] .* z_rn) .+
        Znr * (sd[3] .* z_nr)
end

"""
    f ~ hsgp_effect(PHI, lambda)

Shipped latent submodel for an isotropic Hilbert-space approximate GP
(exponentiated-quadratic kernel) over the basis of [`hsgp_basis`](@ref):
`PHI` the basis columns, `lambda` their eigenvalues. Its body states every
prior, including the length scale's validity floor:

```julia
@rkppl hsgp_effect(PHI, lambda) = begin
    rho_floor = maximum(hsgp_rho_floors(lambda))
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ LogNormal(0, 1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_sqrt_spd(lambda, sigma, rho) .* z)
end
```

At `f ~ hsgp_effect(PHI, lambda)` the parameters are `f.rho`, `f.sigma` and
`f.z` (restored under `nt.f`), and bare `f` is one value per observation:

```julia
(PHI, lambda) = hsgp_basis(x; k = 20, c = 1.5)
f ~ hsgp_effect(PHI, lambda)
mu = a .+ f
```

The floor `rho_floor` is the smallest length scale the `k`-term basis
resolves; the built-in `hsgp_basis(:id, x)` applies the same floor with
Stan's unnormalized lower-bound kernel (here the truncation is normalized,
a data constant apart). Several axes take one shared length scale
(`hsgp_basis(x, z; ...)`); for one length scale per axis, write the
statements with `hsgp_sqrt_spd(lambda, sigma, [rho_1, rho_2])`.
"""
@rkppl hsgp_effect(PHI, lambda) = begin
    rho_floor = maximum(hsgp_rho_floors(lambda))
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ LogNormal(0, 1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_sqrt_spd(lambda, sigma, rho) .* z)
end

"""
    f ~ hsgp_periodic_effect(PHI, harmonics)

Shipped latent submodel for a periodic-kernel Hilbert-space approximate GP
over the basis of [`hsgp_periodic_basis`](@ref). Its body:

```julia
@rkppl hsgp_periodic_effect(PHI, harmonics) = begin
    rho_floor = hsgp_periodic_rho_floor(harmonics)
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ LogNormal(0, 1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_periodic_sqrt_spd(harmonics, sigma, rho) .* z)
end
```
"""
@rkppl hsgp_periodic_effect(PHI, harmonics) = begin
    rho_floor = hsgp_periodic_rho_floor(harmonics)
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ LogNormal(0, 1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_periodic_sqrt_spd(harmonics, sigma, rho) .* z)
end

"""
    f ~ hsgp_grouped_effect(PHI, lambda, g)

Shipped latent submodel for a grouped Hilbert-space approximate GP: one
curve per level of `g`, over the grouped basis `hsgp_basis(x; k, by = g)`
(see [`hsgp_basis`](@ref)), with per-group length scales and marginal
scales from log-linear hyper-predictors. Every prior is a statement:

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

The per-group offsets `rho_z` and `sigma_z` have one element per level of
`g`, in `levels(g)` order (the grouped basis's group order). Each group's
length scale is clamped at the validity floor, as in the
built-in `hsgp_basis(...; by = g, length_scale = 1 + (1 | g))` (decision
`0z5bsqi`, prong `group_floor`); the clamp has zero gradient below the
floor. For a shared length scale or marginal scale, write the statements
with a scalar `rho` or `sigma`.
"""
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

"""
    m ~ monotonic(c, zeta)

Shipped latent submodel for a monotonic effect of an ordinal predictor:
level `k` of the integer index column `c` (levels `1..K`) reads the
cumulative sum of the first `k - 1` increments of the simplex `zeta`
(`K - 1` elements), so level 1 is 0 and level `K` is 1. Its body is one
value; the simplex and its prior are the caller's:

```julia
@rkppl monotonic(c, zeta) = begin
    return cumsum(vcat(0.0, zeta))[c]
end
```

```julia
zeta ~ Dirichlet([1.0, 2.0])     # K = 3 levels
m ~ monotonic(c, zeta)
mu = a .+ b .* m                 # a scaled monotonic effect
# or: mu = a .+ m                # unscaled
```

At the use site `m` is the observation column `cumsum(vcat(0.0, zeta))[c]`;
the plan is the hand-written gather.
"""
@rkppl monotonic(c, zeta) = begin
    return cumsum(vcat(0.0, zeta))[c]
end

"""
    x ~ differenced_ar1(beta, sigma)

Shipped latent submodel for a zero-started differenced-AR(1) trajectory: the
increments follow an AR(1), `d[t] = beta * d[t - 1] + sigma * z[t]`, and the
level integrates them, `x[t] = x[t - 1] + d[t]`, from `x[1] = d[1] = 0`. The
trajectory has one element per observation (the loop runs `2:T`, `T` the
observation count). Its body states the innovation prior; the persistence
`beta` and scale `sigma` are the caller's, with the priors as written:

```julia
@rkppl differenced_ar1(beta, sigma) = begin
    @scan begin
        level[1] = 0.0
        increment[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            increment[t] = beta * increment[t - 1] + sigma * z
            level[t] = level[t - 1] + increment[t]
        end
    end
    return level
end
```

```julia
beta ~ truncated(Normal(0.5, 0.2), 0, 1)
sigma_d ~ HalfNormal(0.2)
x ~ differenced_ar1(beta, sigma_d)
mu = a .+ x
```

At `x ~ differenced_ar1(...)` the carried arrays are local `x.level` and
`x.increment`. The existing non-centered scan packing stores its `T - 1`
innovations as `nt.x._ppl_scan_z_level`; coordinate labels start with
`x._ppl_scan_z_level`. Bare `x` returns the level trajectory.
If an authored local already has a generated block's name, the generated
draw name takes the first free numeric suffix; the authored local keeps its name.
"""
@rkppl differenced_ar1(beta, sigma) = begin
    @scan begin
        level[1] = 0.0
        increment[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            increment[t] = beta * increment[t - 1] + sigma * z
            level[t] = level[t - 1] + increment[t]
        end
    end
    return level
end

"""
    b ~ r2d2_coefs(X, alpha)

Shipped R2D2 shrinkage prior on the coefficients of the columns of a
matrix `X` (an `X = hcat(x1, x2, ...)` definition or a bound data matrix),
with `alpha` the Dirichlet concentration: a literal vector with one entry
per column. The body states every prior:

```julia
@rkppl r2d2_coefs(X, alpha) = begin
    R2 ~ Beta(1, 1)
    phi ~ Dirichlet(alpha)
    tau ~ HalfNormal(1)
    varx = var.(eachcol(X))
    b[axes(X, 2)] .~ Normal.(0, sqrt.(phi .* R2 .* tau^2 ./ varx))
    return b
end
```

Use it as `b ~ r2d2_coefs(X, [1.0, 1.0])` and `mu = a .+ X * b`, with the
intercept `a` and its prior stated outside. At that use site the draws are
`nt.b.R2`, `nt.b.phi`, `nt.b.tau` and the coefficients `nt.b.b`; `varx` is the sample
variance (N − 1) of each column, computed once at `bind_data`.

The coefficient sd is `tau * sqrt(phi_k * R2 / var(x_k))` with an
independent `tau`, the StanBlocks algebra of the `r2d2(...)` built-in. So
`R2` is a share parameter, not the model's R². For a different prior, write
the statements yourself.
"""
@rkppl r2d2_coefs(X, alpha) = begin
    R2 ~ Beta(1, 1)
    phi ~ Dirichlet(alpha)
    tau ~ HalfNormal(1)
    varx = var.(eachcol(X))
    b[axes(X, 2)] .~ Normal.(0, sqrt.(phi .* R2 .* tau^2 ./ varx))
    return b
end

"""
    b ~ horseshoe_coefs(X)

Shipped horseshoe shrinkage prior on the coefficients of the columns of a
matrix `X` (an `X = hcat(x1, x2, ...)` definition or a bound data matrix):
one global scale `tau`, one local scale per column, non-centered. The body
states every prior:

```julia
@rkppl horseshoe_coefs(X) = begin
    tau ~ HalfCauchy(1)
    lambda[axes(X, 2)] .~ HalfCauchy.(1)
    z[axes(X, 2)] .~ Normal.(0, 1)
    return z .* lambda .* tau
end
```

Use it as `b ~ horseshoe_coefs(X)` and `mu = a .+ X * b`, with the
intercept `a` and its prior stated outside. At that use site the draws are
`nt.b.tau`, `nt.b.lambda` and `nt.b.z`, and bare `b` returns `z .* lambda .* tau`
from that scope. This is the
coefficient vector, so each coefficient is `Normal(0, lambda_k * tau)`
given its scales. The halves are Distributions-normalized
(`truncated(Cauchy(0, s), 0, Inf)`). For a different prior, write the
statements yourself.
"""
@rkppl horseshoe_coefs(X) = begin
    tau ~ HalfCauchy(1)
    lambda[axes(X, 2)] .~ HalfCauchy.(1)
    z[axes(X, 2)] .~ Normal.(0, 1)
    return z .* lambda .* tau
end

# ── Varying effects ──────────────────────────────────────────────────
# Each submodel returns per-level coefficients over `levels(g)`: plain
# array values the use site reads per observation with ordinary indexing
# (`b[g]`, `x .* b[g]`, `b[g, 1] .+ x .* b[g, 2]`). Intercepts, slopes,
# derived or indicator margins (`w .* b[g]`, `(c .== 2) .* b[g, 2]`),
# multi-membership (`b[g1] ./ 2 .+ b[g2] ./ 2` over `gg = vcat(g1, g2)`)
# and draws shared by several predictors are all plain Julia at the use
# site. Defaults: `sd ~ HalfNormal(1)` per margin, `LKJCholesky(K, 1.0)`
# for K ≥ 2 margins.

"""
    b ~ varying_coefs(g)

Non-centered per-level coefficients for one margin: `b[j] = sd * z[j]`
with `z[j] ~ Normal(0, 1)` for every level `j` of `g`. Its body:

```julia
@rkppl varying_coefs(g) = begin
    sd ~ HalfNormal(1)
    z[levels(g)] .~ Normal.(0, 1)
    return sd .* z
end
```

`b` is a vector over `levels(g)`: `b[g]` is each observation's
coefficient (a varying intercept), `x .* b[g]` a varying slope. Draws are
`nt.b.sd` and `nt.b.z`. For multi-membership, pass the union of the membership
columns as one data definition (`gg = vcat(g1, g2)`; `b ~
varying_coefs(gg)`) and weight the gathers (`b[g1] ./ 2 .+ b[g2] ./ 2`).
"""
@rkppl varying_coefs(g) = begin
    sd ~ HalfNormal(1)
    z[levels(g)] .~ Normal.(0, 1)
    return sd .* z
end

"""
    b ~ varying_coefs_correlated(g, K)

Non-centered correlated per-level coefficients for `K ≥ 2` margins (`K` a
literal): every level's row is `diagm(sd) * L * z[j, :]`, so rows are
`MvNormal(0, (sd .* L) * (sd .* L)')`. Its body:

```julia
@rkppl varying_coefs_correlated(g, K) = begin
    sd[1:K] .~ HalfNormal.(1)
    L ~ LKJCholesky(K, 1.0)
    z[levels(g), 1:K] .~ Normal.(0, 1)
    return z * (sd .* L)'
end
```

`b` is a `levels(g) × K` matrix: margin `k` of observation `i` is
`b[g, k]`, so a correlated varying intercept and slope read
`b[g, 1] .+ x .* b[g, 2]`. Draws are
`nt.b.sd`, `nt.b.L` and `nt.b.z`.
"""
@rkppl varying_coefs_correlated(g, K) = begin
    sd[1:K] .~ HalfNormal.(1)
    L ~ LKJCholesky(K, 1.0)
    z[levels(g), 1:K] .~ Normal.(0, 1)
    return z * (sd .* L)'
end

"""
    b ~ varying_coefs_centered(g)

Centered per-level coefficients for one margin: `b[j] ~ Normal(0, sd)`
for every level `j` of `g`, the coefficients themselves sampled. Its
body:

```julia
@rkppl varying_coefs_centered(g) = begin
    sd ~ HalfNormal(1)
    c[levels(g)] .~ Normal.(0, sd)
    return c
end
```

Read it like [`varying_coefs`](@ref): `b[g]`, `x .* b[g]`. Draws are
`nt.b.sd` and `nt.b.c`.
"""
@rkppl varying_coefs_centered(g) = begin
    sd ~ HalfNormal(1)
    c[levels(g)] .~ Normal.(0, sd)
    return c
end

"""
    b ~ varying_coefs_centered_correlated(g, K)

Centered correlated per-level coefficients for `K ≥ 2` margins (`K` a
literal): every level's row is one draw of
`MvNormal(0, (sd .* L) * (sd .* L)')`, the coefficients themselves
sampled. Its body:

```julia
@rkppl varying_coefs_centered_correlated(g, K) = begin
    sd[1:K] .~ HalfNormal.(1)
    L ~ LKJCholesky(K, 1.0)
    F = sd .* L
    eachrow(c[levels(g), 1:K]) .~ MvNormalCholesky(zeros(K), F)
    return c
end
```

Read it like [`varying_coefs_correlated`](@ref): margin `k` of
observation `i` is `b[g, k]`, so `b[g, 1] .+ x .* b[g, 2]`. Draws are
`nt.b.sd`, `nt.b.L` and `nt.b.c`.
"""
@rkppl varying_coefs_centered_correlated(g, K) = begin
    sd[1:K] .~ HalfNormal.(1)
    L ~ LKJCholesky(K, 1.0)
    F = sd .* L
    eachrow(c[levels(g), 1:K]) .~ MvNormalCholesky(zeros(K), F)
    return c
end

"""
    u ~ varying_stratified(g, s)

One stratified margin: each stratum (level of `s`) has its own sd, and
observation `i` reads `sd[s[i]] * z[g[i]]`. Its body:

```julia
@rkppl varying_stratified(g, s) = begin
    sd[levels(s)] .~ HalfNormal.(1)
    z[levels(g)] .~ Normal.(0, 1)
    return sd[s] .* z[g]
end
```

`u` is one value per observation: a varying intercept is `u`, a varying
slope `x .* u`. Draws are `nt.u.sd` and `nt.u.z`.
"""
@rkppl varying_stratified(g, s) = begin
    sd[levels(s)] .~ HalfNormal.(1)
    z[levels(g)] .~ Normal.(0, 1)
    return sd[s] .* z[g]
end

"""
    u ~ varying_stratified_correlated(g, s, K)

Stratified correlated margins: each stratum (level of `s`) has its own
`K` sds and its own `LKJCholesky(K, 1.0)` factor, the levels of `g` share
`z`, and observation `i` reads the row
`(sd[s[i], :] .* L[s[i]]) * z[g[i], :]`. Its body:

```julia
@rkppl varying_stratified_correlated(g, s, K) = begin
    sd[levels(s), 1:K] .~ HalfNormal.(1)
    @plate for k in levels(s)
        L[k] ~ LKJCholesky(K, 1.0)
    end
    z[levels(g), 1:K] .~ Normal.(0, 1)
    @plate for i in eachindex(g)
        b[i, 1:K] = (sd[s[i], :] .* L[s[i]]) * z[g[i], :]
    end
    return b
end
```

`u` has one row per observation: margin `k` is the column `u[:, k]`, so a
correlated intercept and slope read `u[:, 1] .+ x .* u[:, 2]`. Draws are
`nt.u.sd`, `nt.u.L` (`nt.u.L[:, :, k]` stratum k's factor) and `nt.u.z`.
"""
@rkppl varying_stratified_correlated(g, s, K) = begin
    sd[levels(s), 1:K] .~ HalfNormal.(1)
    @plate for k in levels(s)
        L[k] ~ LKJCholesky(K, 1.0)
    end
    z[levels(g), 1:K] .~ Normal.(0, 1)
    @plate for i in eachindex(g)
        b[i, 1:K] = (sd[s[i], :] .* L[s[i]]) * z[g[i], :]
    end
    return b
end
