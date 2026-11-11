# Graph inspection

Use `visualize` to inspect dependencies and the recipes selected by a plan. Print `code_expr` when the question concerns the emitted Julia program.

## Plan view

The graph below contains a value and its square. Selecting only the square retains the dependency on the value.

```@example inspection
using ReactiveKernels
using Main.Tutorials

@kernel shifted_square(x, shift) = begin
    value = x + shift
    squared = value^2
    return squared
end

selected = plan(shifted_square)
Tutorials.html(visualize(selected))
```

The visualization is generated from the current plan, so it changes whenever the plan does. A supplied intermediate input changes the plan boundary, so its producing recipe is no longer needed for that selection.

## Generated expression

Print the actual expression returned by the prepared callable. Removing line annotations from a copy improves readability without changing the displayed operations.

```@example inspection
prepared = prepare(shifted_square; have=:value, want=:squared)
Tutorials.generated(prepared)
```

The callable's output checks that the selected boundary consumes the supplied intermediate directly.

```@example inspection
prepared(4.0)
```

`dot_source` and `save_visualization` support exported graph views; see the [reference](api.md) for formats and keyword arguments. A saved view describes the plan that produced it and does not update when a different plan is prepared.

The [bound-regression example](bound-regression.md) places a generated dependency view beside its source, runtime outputs and numerical checks.
