# Graphs and queries

Prepare different computations from one authored graph by selecting its inputs
and outputs.

## Definition

Every signature argument is an input port, and each assigned name is a graph
value. An explicit `return` chooses the default output without hiding the
other names.

```@example graphs
using ReactiveKernels

@kernel calibration(x, slope, intercept) = begin
    scaled = slope * x
    prediction = scaled + intercept
    squared = prediction * prediction
    return prediction
end
predict = prepare(calibration)
(; prediction=predict(2.0, 3.0, 1.0),
   changed_input=predict(4.0, 3.0, 1.0))
```

The call supplies arguments in signature order. Type annotations are optional;
the prepared function specializes on the values passed to it.

## Boundary

`have` selects supplied values; `want` selects results. A value computed in one
query can be supplied directly to another.

```@example graphs
square = prepare(calibration; have=:prediction, want=:squared)
parts = prepare(calibration; want=(:scaled, :prediction))
(; square=square(7.0), parts=parts(2.0, 3.0, 1.0))
```

A scalar `want` returns one value. A tuple of requested names returns a tuple in
that order. Supplying `prediction` means the square query needs neither `x`
nor the calibration coefficients.

## Plan

The plan exposes the selected recipes. The graph below contains the
prediction-to-square computation, so it has no coefficient inputs.

```@example graphs
using Main.Tutorials # hide
selected = plan(calibration; have=:prediction, want=:squared)
Tutorials.html(visualize(selected))
```

Planning chooses an executable set of recipes from the available boundary.
Dependencies shared by several requested values are computed once. Declared
alternative producers are selected by cost; a cost describes planning
preference, not a timing measurement.

## Generated expression

The expression below is the program returned by `code_expr`, with only line
number nodes removed from a copy.

```@example graphs
Tutorials.generated(square)
```

A generated expression may reference the prepared object's `constants` and
`callables` tuples. It is an inspection view of that object, not necessarily a
standalone function that can be pasted into another session. The
[visualization guide](visualization.md) covers graph and expression exports.

Use [bound regression](bound-regression.md) to select predictions and residuals
from a graph that also prepares its data.
