# MNIST kernels through Reactant

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
and both data modes. The native half of each categorical recipe remains the
allocation-friendly authored scalar-object plate; Reactant receives the
explicitly equivalent tensor gather/reduction. Bound-data prior rows are N/A
because their selected graph slice has no data ports. No host fallback or
Reactant-only model is used.

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

## Reproduce

The receipt generator requires a clean detached candidate so its full commit
pin and source blob are immutable. From a sibling directory:

```sh
git -C ReactiveKernels.jl worktree add --detach ReactiveKernels-mnist-receipt HEAD
cd ReactiveKernels-mnist-receipt
julia --startup-file=no benchmark/mnist_reactant_comparison.jl \
  --output=benchmark/receipts/mnist-reactant-v2.toml
julia --startup-file=no benchmark/receipts/validate_mnist_reactant.jl \
  benchmark/receipts/mnist-reactant-v2.toml
```

The script provisions a fresh environment with Reactant 0.2.278, develops the
exact candidate plus its two example packages, loads the full MNIST training
split via MLDatasets, and records environment setup, package precompilation,
and data loading separately. For a quick non-publication smoke run, set
`RK_MNIST_REACTANT_N=64` and `RK_MNIST_REACTANT_ROUNDS=2`; the checked-in
receipt keeps the publication protocol of the full training split and at least
ten rounds.

The Reactant-compiled-AD receipt is generated the same way from the same clean
detached candidate, and additionally resolves Enzyme and
DifferentiationInterface into the pinned environment:

```sh
julia --startup-file=no benchmark/mnist_reactant_ad_comparison.jl \
  --output=benchmark/receipts/mnist-reactant-ad-v2.toml
julia --startup-file=no benchmark/receipts/validate_mnist_reactant_ad.jl \
  benchmark/receipts/mnist-reactant-ad-v2.toml
```

Its quick-smoke knobs are `RK_MNIST_REACTANT_AD_N` and
`RK_MNIST_REACTANT_AD_ROUNDS`.

Generate the Wren-compatible receipts through the same wrappers with the
explicit dataset selector. Supplying the copied CSV performs the
publication-time equality check; PCA itself is reconstructed from MLDatasets.

```sh
julia --startup-file=no benchmark/mnist_reactant_comparison.jl \
  --dataset=wren-pca40 --wren-reference=/path/to/mnist.csv \
  --output=benchmark/receipts/mnist-reactant-wren-pca40-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_reactant.jl \
  benchmark/receipts/mnist-reactant-wren-pca40-v1.toml

julia --startup-file=no benchmark/mnist_reactant_ad_comparison.jl \
  --dataset=wren-pca40 --wren-reference=/path/to/mnist.csv \
  --output=benchmark/receipts/mnist-reactant-ad-wren-pca40-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_reactant_ad.jl \
  benchmark/receipts/mnist-reactant-ad-wren-pca40-v1.toml
```
