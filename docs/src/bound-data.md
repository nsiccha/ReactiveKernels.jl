# Bound data

Prepare data-dependent work once while leaving parameters as runtime inputs.

## Preparation

This graph centers its observations before comparing them with a coefficient.
Binding `x` removes it from the prepared call's arguments.

```@example binding
using ReactiveKernels, Statistics

@kernel centered_error(x, coefficient) = begin
    center = mean(x)
    centered = x .- center
    residual = centered .- coefficient
    loss = sum(abs2, residual)
end
x = [1.0, 2.0, 4.0]
fixed = prepare(centered_error; bound=(; x), want=:loss)
ordinary = prepare(centered_error; want=:loss)
(; bound=fixed(0.5), ordinary=ordinary(x, 0.5),
   same_result=fixed(0.5) ≈ ordinary(x, 0.5))
```

Recipes that depend only on bound values can be evaluated during preparation.
The coefficient-dependent residual remains in the called program. Binding is a
preparation choice; it does not rewrite the authored model with array lengths
or data values.

## Rebinding

Treat bound data as read-only for the lifetime of the prepared object.
Construct another preparation for changed data instead of mutating a captured
array and expecting folded results to update.

```@example binding
new_x = [2.0, 3.0, 8.0]
new_fixed = prepare(centered_error; bound=(; x=new_x), want=:loss)
(; original=fixed(0.5), rebound=new_fixed(0.5),
   matches_unbound=new_fixed(0.5) ≈ ordinary(new_x, 0.5))
```

The new object owns its preparation state. Keep a separate object for each
binding when callers use several datasets concurrently.

## Requested outputs

An intermediate can remain named and requestable even when another query folds
or eliminates it.

```@example binding
preprocessing = prepare(centered_error;
    have=:x, bound=(; x), want=(:center, :centered))
preprocessing()
```

Expose preprocessing as graph values when it is part of the model. That makes
its dependencies inspectable and lets the planner select it without copying
the preprocessing code into another function.

## Gaussian observation plates

A summed plate of Gaussian observations over a bound response adds
`a + c * (y[i] - mu[i])^2` per row, with the scale's terms `a` and `c` shared
by every row. When `mu` is linear in live values with bound coefficients (a
bound matrix times a live vector, live numbers, or live vectors gathered
through one bound index, each times bound coefficients, plus bound offsets),
the sum of squared residuals is a quadratic polynomial whose coefficients are
sums over the bound rows. A bound preparation computes least-squares
statistics of the rows once (a triangular factor of the design and the
least-squares solution carried at extended precision). Each call then costs
time in the number of parameters instead of the number of rows, in the value
and in its ordinary reverse derivative.

```@example binding
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel regression_density(X, y, beta, sigma) = begin
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total = sum(lp)
    return total
end
X = [ones(200) range(-1, 1; length=200)]
y = X * [0.5, 2.0] .+ 0.1 .* sin.(1:200)
collapsed = prepare(regression_density; bound=(; X, y))
per_row = prepare(regression_density; bound=(; X, y), bound_statistics=false)
beta = [0.4, 2.1]
(; collapsed=collapsed(beta, 0.3), per_row=per_row(beta, 0.3),
   same_result=collapsed(beta, 0.3) ≈ per_row(beta, 0.3),
   reads_statistics=occursin("_ls_rss", string(code_expr(collapsed))))
```

Values and derivatives agree with the per-row plate to rounding. They are
more accurate where the response is far from zero, where the per-row sum
itself loses digits. A binding whose data do not fit compiles the per-row
plate as before: fewer rows than twice the parameters per block, a non-finite
entry, or data of another shape. The plate's own loop also still runs where
the factored sum is not used, for example at a scale of zero. Per-row and
per-cell queries keep the per-row plate. `bound_statistics = false` on a
preparation, or `bound_statistics = false` under `[ReactiveKernels]` in the
project's `LocalPreferences.toml`, keeps the per-row plate everywhere.

[Bound regression](bound-regression.md) applies this pattern to a design
matrix and separates prediction from residual evaluation.
