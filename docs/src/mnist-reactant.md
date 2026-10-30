# MNIST kernels through Reactant (historical receipt)

> **Historical receipt.** ReactiveKernels no longer has a Reactant path: its
> Reactant extension, tensorized lowering and Reactant tests were removed. This
> page keeps the measurements as recorded. Nothing on it runs Reactant, and the
> RK functions and test files it names no longer exist.

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page measures Reactant on the exact executable model documented on the
[MNIST multinomial-logistic kernel page](mnist-logistic.md). The benchmark
imports both MNIST source authorities and graph builders from
`ReactiveKernelsPPLExamples`; it does not copy the model, rewrite a density, or
introduce a Reactant-only mathematical path.

The comparison mirrors the complete two-by-four capability matrix in the
[native primal receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-primal-v3.toml):
the packed unconstrained vector and the structured `(W, b)` coefficients as
input boundaries, and the joint density, prior, summed likelihood, and
pointwise likelihood as requested outputs, on the full 60000-image MNIST
training split. Every cell is evaluated natively and attempted through
Reactant.

All primal outcomes compile for both model sources, both parameter boundaries,
and both data modes. Native execution and Reactant consume the same natural
`plate(eachcol(logits), y)` likelihood source. Reactant's generic authored-plate
lowering preserves the column structure and lowers the scalar endpoints to
batched operations with traced gathers; there is no alternate recipe body or
private MNIST adapter. Bound-data prior rows are N/A because their selected
graph slice has no data ports. No host fallback or Reactant-only model is used.

## Primal performance and support

```@eval
Main.ReactiveKernelsDocs.render_mnist_reactant_benchmark()
```

The outcome-specific plots and tables contain steady-state synchronous call
time only, reported by the same uncontended-cost estimator as the native MNIST
receipt (the minimum of per-round BenchmarkTools minimums). Each row uses its
matched native RK twin as the 1.00× baseline, reports `Reactant ÷ native`, and
states the faster/slower interpretation. Host-to-device conversion, kernel
preparation, Reactant compilation, the first synchronous call, and result
readback are outside that timing and remain recorded in the receipt.

The section above times primal densities only. The Reactant-compiled gradient
column follows below; native AD (RK vs Turing vs manual, no Reactant) lives on
the [automatic-differentiation page](automatic-differentiation.md), whose
derivative matrix this page reuses exactly.

## Reactant-compiled automatic differentiation

```@eval
Main.ReactiveKernelsDocs.render_mnist_reactant_ad_benchmark()
```

This section is the AD analog of the primal table above. It consumes the
first-class RK verb `compile_ad_value_and_gradient` (the AD companion of the
primal `@compile` path) — no gradient is hand-rolled — and reuses the exact
differentiable outcome/boundary protocol published by the native MNIST AD
receipt
([`mnist-logistic-ad-v2`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-ad-v2.toml)):
the value and gradient of each scalar output with respect to the packed
coefficient vector. Non-scalar `pointwise` outputs and the two-active-port
structured `(W, b)` boundary stay unsupported, exactly as on the AD receipt.

The packed joint, prior, and likelihood gradients compile for both model
sources; packed joint and likelihood also compile with `X`, `y`, and the class
count fixed during preparation. Pointwise remains vector-valued and the
structured `(W, b)` boundary has two active ports, so those rows remain explicit
unsupported cells rather than invented Jacobian/VJP or multi-active APIs.

## Wren-compatible PCA-40 workload

The additional workload preserves the exact data representation used by
Wren's `bench/mnist.csv` without committing that private file. Starting from
the same MLDatasets training split, the generator centers all 60000 raw images,
fits PCA in the 784-pixel space, and projects the first 1000 images onto the top
40 components without whitening. The publication receipt records zero label
mismatches against the reference CSV and a maximum feature error of
`1.10e-13`; the retained components explain 78.6108% of training-set pixel
variance.

This separately identified `1000×40` workload is not a replacement for the
full `60000×784` two-model matrix above. Its retained v1 receipt uses the same
authored model and runtime-data boundary, while the default generator route
continues to produce the complete v2 matrix.

The PCA-40 primal Reactant receipt cross-pins the standalone
`mnist-logistic-wren-pca40-v1` RK/manual/Turing matrix, and the compiled-AD
receipt cross-pins `mnist-logistic-ad-wren-pca40-v1`. Thus the native columns
on this page and the standalone comparison pages share the same dataset and
receipt authority rather than borrowing the full-data baseline.

### PCA-40 primal performance

```@eval
Main.ReactiveKernelsDocs.render_mnist_reactant_wren_benchmark()
```

### PCA-40 Reactant-compiled automatic differentiation

```@eval
Main.ReactiveKernelsDocs.render_mnist_reactant_ad_wren_benchmark()
```

## Provenance

Each receipt was generated from a clean detached checkout of the
ReactiveKernels commit it pins, in a fresh environment with Reactant 0.2.289,
by `benchmark/mnist_reactant_comparison.jl` and
`benchmark/mnist_reactant_ad_comparison.jl` at that commit (the Wren-compatible
receipts with `--dataset=wren-pca40`). Both scripts were removed together with
ReactiveKernels' Reactant path. The full-resolution receipts keep the
publication protocol of the full training split and at least ten rounds, and
`benchmark/receipts/validate_mnist_reactant.jl` and
`validate_mnist_reactant_ad.jl` still validate them.
