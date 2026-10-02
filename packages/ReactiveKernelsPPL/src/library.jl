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
