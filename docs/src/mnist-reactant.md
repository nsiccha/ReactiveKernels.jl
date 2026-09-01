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
perform. On the current receipt the likelihood-bearing cells do not compile —
the model's constant reference-logits row reaches Base's generic elementwise
`vcat` during tracing — so their rows carry that scalar-indexing diagnostic
while the fixed-capability report tracks the upstream fix.

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

This page times primal densities only — no gradients. Native AD for this model
belongs to the [automatic-differentiation page](automatic-differentiation.md);
a Reactant-compiled AD column mirroring the
[Eight Schools one](eight-schools-reactant.md) follows once the native MNIST
AD matrix is published there.

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
