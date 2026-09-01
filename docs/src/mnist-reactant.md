# MNIST kernels through Reactant

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page measures Reactant on the exact executable model documented on the
[MNIST multinomial-logistic kernel page](mnist-logistic.md). The benchmark
imports `MNIST_LOGISTIC_SOURCE` and `build_mnist_logistic_graph` from
`ReactiveKernelsPPLExamples`; it does not copy the model, rewrite a density, or
introduce a Reactant-only mathematical path.

The comparison mirrors the complete two-by-four capability matrix in the
[native primal receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-v1.toml):
the packed unconstrained vector and the structured `(W, b)` coefficients as
input boundaries, and the joint density, prior, summed likelihood, and
pointwise likelihood as requested outputs, on the full 60000-image MNIST
training split. Every cell is evaluated natively and attempted through
Reactant.

Unsupported compiler cells stay in the table with their actual diagnostic.
They are not silently omitted, replaced with a different HAVE boundary, or
timed through host fallback. That distinction matters: the table describes
which views of this one graph compile today as well as how the compiled views
perform. On the current receipt the likelihood-bearing cells do not compile:
the per-observation likelihood plate over `eachcol(logits)` with the observed
class index stops at the compiler's scalar-indexing/gather frontier (the
model's earlier constant reference-row `vcat` blocker is fixed), so their rows
carry that diagnostic while the capability lands upstream.

## Primal performance and support

```@eval
Main.ReactiveKernelsDocs.render_mnist_reactant_benchmark()
```

The first table contains steady-state synchronous call time only, reported by
the same uncontended-cost estimator as the native MNIST receipt (the minimum
of per-round BenchmarkTools minimums). Host-to-device conversion, kernel
preparation, Reactant compilation, the first synchronous call, and result
readback are outside that timing. The second table and setup summary report
those costs separately, including failed compile attempts.

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
([`mnist-logistic-ad-v1`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-ad-v1.toml)):
the value and gradient of each scalar output with respect to the packed
coefficient vector. Non-scalar `pointwise` outputs and the two-active-port
structured `(W, b)` boundary stay unsupported, exactly as on the AD receipt.

A Reactant-compiled gradient exists only where the primal kernel itself
compiles through Reactant, so the compiled cells are a subset of the native-AD
cells: the packed joint and likelihood — whose primals stop at the
plate/gather frontier above — keep native AD but no Reactant gradient. As with
the primal table, AD preparation, host transfers, gradient compilation, the
first synchronous call, and readback are excluded from the steady-state timing
and reported separately.

## Reproduce

The receipt generator requires a clean detached candidate so its full commit
pin and source blob are immutable. From a sibling directory:

```sh
git -C ReactiveKernels.jl worktree add --detach ReactiveKernels-mnist-receipt HEAD
cd ReactiveKernels-mnist-receipt
julia --startup-file=no benchmark/mnist_reactant_comparison.jl \
  --output=benchmark/receipts/mnist-reactant-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_reactant.jl \
  benchmark/receipts/mnist-reactant-v1.toml
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
  --output=benchmark/receipts/mnist-reactant-ad-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_reactant_ad.jl \
  benchmark/receipts/mnist-reactant-ad-v1.toml
```

Its quick-smoke knobs are `RK_MNIST_REACTANT_AD_N` and
`RK_MNIST_REACTANT_AD_ROUNDS`.
