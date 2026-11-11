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

[Bound regression](bound-regression.md) applies this pattern to a design
matrix and separates prediction from residual evaluation.
