# Shipped @rkppl library submodels: ordinary submodels written in the
# surface language with every default prior stated in the body, so a
# program reads (and rewrites) exactly what the use site expands to. A use
# site `y ~ sm(...)` / `z ~ sm(...)` resolves here through the exported
# binding (the caller's module sees it via `using ReactiveKernelsPPL`) and
# lowers to the identical plan as the hand-inlined body (see
# [`RKPPLSubmodel`](@ref)): body names namespace under the use-site
# left-hand side.

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

At `y ~ ordered_logistic(eta)` the cutpoints are `y_cutpoints`; the plan is
the hand-written two statements above with `y_cutpoints`. To use a different
prior, write those statements yourself.
"""
@rkppl ordered_logistic(eta) = begin
    cutpoints ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
    y .~ OrderedLogistic.(eta, Ref(cutpoints))
    return y
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

At `x ~ differenced_ar1(...)` the carried arrays are `x_level` and
`x_increment` and the innovations are the `T - 1` coordinates of
`_ppl_scan_z_x_level`.
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
`b_R2`, `b_phi`, `b_tau` and the coefficients `b_b`; `varx` is the sample
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
`b_tau`, `b_lambda` and `b_z`, and `b = b_z .* b_lambda .* b_tau` is the
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
