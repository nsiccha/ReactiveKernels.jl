# Bound regression

This example uses a deterministic dataset: a predictor measured in arbitrary units and an observed response in the same units. The observations lie near a straight line. The coefficient vector is supplied by the caller; there is no fitted posterior or optimization step. This keeps the example focused on a compiler question: which parts of the calculation depend on the observations, and which must run again when the coefficients change?

## Graph

The graph names the design matrix, prediction, residual and squared error separately. These names are ports that can become input or output boundaries when preparing a callable. Array lengths remain runtime properties of the data; the graph does not contain a separate recipe for each observation.

```@example regression
using ReactiveKernels, LinearAlgebra, AlgebraOfVega
using Main.Tutorials

@kernel regression(x, y, coefficients) = begin
    design = hcat(ones(length(x)), x)
    prediction = design * coefficients
    residual = y - prediction
    squared_error = sum(abs2, residual)
    return squared_error
end

x = collect(-2.0:1.0:2.0)
y = [0.1, 0.8, 2.2, 2.9, 4.1]
coefficients = [2.0, 1.0]
observations = (; x, y)
loss = prepare(regression; bound=observations)
loss(coefficients)
```

Binding makes the observations part of the prepared object's state. The remaining argument is the coefficient vector. The planner can compute the design matrix from the bound predictor during preparation, since that matrix has no dependency on the coefficients. Binding does not change the mathematical result or turn the observations into constants written into generated source.

The generated view exposes the selected dependencies. A reader can inspect which values enter the residual and which recipe supplies the design, rather than infer that structure from a timing result.

```@example regression
Tutorials.html(visualize(plan(regression)))
```

The callable's generated expression is also available. It is printed using Julia's expression printer, with source line annotations removed from a copy. Every name and operation shown is one the callable executes.

```@example regression
Tutorials.generated(loss)
```

## Evaluation

Prepare an unbound callable as an independent check of the binding boundary. Both evaluations receive the same coefficient values and observations, so their squared errors must agree. A second coefficient vector exercises the remaining runtime input; it is not necessary to rebuild the bound callable for that change.

```@example regression
unbound_loss = prepare(regression)
other_coefficients = [1.5, 0.8]
comparison = (
    first_bound=loss(coefficients),
    first_unbound=unbound_loss(x, y, coefficients),
    second_bound=loss(other_coefficients),
    second_unbound=unbound_loss(x, y, other_coefficients),
)
@assert comparison.first_bound ≈ comparison.first_unbound
@assert comparison.second_bound ≈ comparison.second_unbound
comparison
```

Prediction and residual inspection use other output selections from the same graph. Requesting several outputs together lets the planner share their common computations. The resulting tuple follows the requested output order, making the boundary explicit in both the preparation and the assignment.

```@example regression
inspect = prepare(regression;
    want=(:prediction, :residual, :squared_error), bound=observations)
prediction, residual, squared_error = inspect(coefficients)
(; prediction, residual, squared_error)
```

The predictions below are conditional on the supplied coefficients. The plot compares each observed response with its corresponding prediction; it does not represent posterior uncertainty, a confidence interval or evidence about generalization. The horizontal axis uses the predictor's arbitrary units, and the vertical axis uses response units. Both plotted series come from the values evaluated above.

```@example regression
plot_data = (
    predictor=vcat(x, x),
    response=vcat(y, prediction),
    series=vcat(fill("Observed", length(x)), fill("Prediction", length(x))),
)
plot = data(plot_data) *
    mapping(:predictor => "Predictor (arbitrary units)",
            :response => "Response (arbitrary units)"; color=:series) *
    visual(Scatter)
Tutorials.figure(plot, "regression-response",
    "Observed responses and predictions at the supplied coefficient vector; paired points expose the residuals.")
```

## Diagnostics

The residual definition and the scalar reduction are compared directly against ordinary array arithmetic. Each displayed discrepancy is an absolute difference from that reference, the largest one for the residual vector. The assertions stop the build if either relationship changes.

```@example regression
residual_error = maximum(abs, residual - (y - prediction))
reduction_error = abs(squared_error - sum(abs2, residual))
@assert residual_error == 0
@assert reduction_error == 0
(; residual_error, reduction_error)
```

Sometimes a caller already has predictions and wants to evaluate residuals for several observation vectors. Supplying `prediction` as an input boundary makes its producer unnecessary for that query. The following callable consequently needs neither the predictor nor the coefficients. Its result must still equal the residual produced by the original graph when the supplied predictions agree.

```@example regression
residual_from_prediction = prepare(regression;
    have=(:y, :prediction), want=:residual)
reused_residual = residual_from_prediction(y, prediction)
@assert reused_residual == residual
(; reused_residual, agrees=reused_residual == residual)
```

## Rebinding

A new dataset requires a new binding. Reuse the graph definition and prepare it with the new values; do not mutate arrays held by a prepared object and assume every cached data-dependent value will refresh. The new binding can have a different observation count because that count is runtime data. The unbound callable provides the same cross-check for this dataset as for the original one.

```@example regression
new_x = [-1.0, 0.0, 1.0]
new_y = [1.2, 1.9, 3.2]
rebound = prepare(regression; bound=(x=new_x, y=new_y))
rebound_value = rebound(coefficients)
reference_value = unbound_loss(new_x, new_y, coefficients)
@assert rebound_value ≈ reference_value
(; rebound_value, reference_value)
```

The [bound-data guide](bound-data.md) describes the binding boundary, and the [graph guide](graphs.md) explains input and output selection. For a probabilistic graph with constrained parameters and a log density, continue to [Eight schools](eight-schools.md). Those coordinate transformations add another graph layer while preserving the same principle: supplied values and requested outputs determine the computation to prepare.

```@example regression
Tutorials.receipt("bound-regression"; checks=(
    bound_agrees=comparison.first_bound ≈ comparison.first_unbound,
    residual_error, reduction_error,
    rebound_agrees=rebound_value ≈ reference_value,
))
```

## Preparation boundary

Preparation belongs outside the repeated coefficient-evaluation loop. It selects recipes, arranges their inputs and outputs, and evaluates work that depends only on bound values. Evaluation then supplies the changing coefficients to that prepared computation. Timing those two phases together answers a different question from timing repeated evaluations, so this example makes no throughput claim from its small dataset.

The design matrix is intentionally visible as a named intermediate. A larger application might construct basis functions, standardize predictors or factor a fixed covariance at the corresponding point. Those operations belong in the graph when the compiler should identify their data dependence. A supplied intermediate is authoritative: if a caller provides predictions that came from another coefficient vector, the residual query uses those predictions. It does not silently recompute or validate their provenance.

The squared-error output is a scalar objective, not a normal log likelihood. Adding a probabilistic observation model requires its scale, normalizing terms and parameter priors. Keeping that distinction explicit prevents an optimization objective from being mistaken for a sampler target.

```@raw html
<details><summary>Reproduce</summary>
```

Run these shell commands from the repository root. The build executes this page; it does not run a statistical fit.

```sh
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
```

Download the <a href="./receipts/bound-regression.toml" download>execution receipt</a>. The [research README](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/docs-history/README.md) separates historical performance experiments from this example.

```@raw html
</details>
```
