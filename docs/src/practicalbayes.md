# PracticalBayes external comparator

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page adds `EvoArt/PracticalBayes` as an external comparator across the
same PPL surfaces used elsewhere in these docs: Eight Schools, full raw-MNIST
multinomial logistic regression, iid-Normal evaluation throughput, and the
Eight Schools plus Wren PCA-40 NUTS workloads. The receipt pins PracticalBayes
0.1.0 at commit
`c6b340baef4f4a9e3d26cd0ea5082a2baf26dcf9` and Julia 1.10. It uses only the
documented public `@model`, `build_layout`, `LogDensityFunction`, fixed-parameter
density views, `returned`, and `sample(..., NUTS(...))` APIs.

The upstream repository has no license file at that commit. Consequently this
repository copies or vendors none of its source: the model definitions below
are independently authored comparator clients of the public API.

PracticalBayes fixes Bijectors 0.15.24, which Julia's resolver reports as
incompatible with this repository's Turing 0.47.1 / DynamicPPL 0.42.6 benchmark
environment. It therefore runs in its own fresh exact-pin environment and
writes a separate receipt. PracticalBayes remains a benchmark-only external
package, never a ReactiveKernels dependency.

## Eight Schools model matrix

The linked vector is exactly `(μ, log τ, θ₁, …, θ₈)`, and the linked full joint
includes the `log τ` Jacobian. PracticalBayes exposes that packed joint for
primal and prepared Enzyme value-and-gradient evaluation. Its constrained
NamedTuple APIs expose joint, prior, summed likelihood, and pointwise
likelihoods for primal evaluation. Packed partial-density views, constrained AD,
and the `θ`-only likelihood boundary remain explicit unsupported rows.

```@eval
Main.ReactiveKernelsDocs.render_practicalbayes_eight_schools_benchmark()
```

The executed public model definition:

```@eval
Main.ReactiveKernelsDocs.render_practicalbayes_eight_schools_baseline()
```

## Full raw-MNIST model matrix

The full 60,000 × 784 evaluation workload uses the same standard-normal
coefficient prior, reference-coded logits, one-based class labels, Float64
precision, parameter order `[vec(W); b]`, and deterministic parameter seed as
the primary MNIST matrix. Both the idiomatic materialized-softmax and vcat-free
stable-log-sum-exp models expose packed joint primal and Enzyme AD. Constrained
primal joint/prior/likelihood cells are supported for both. The idiomatic model
also has a public pointwise observation-site vector; the optimized model's one
scalar `@addlogprob!` likelihood does not, so that cell retains its runtime
diagnostic instead of synthesizing a new interface.

```@eval
Main.ReactiveKernelsDocs.render_practicalbayes_mnist_benchmark()
```

The executed public model definitions:

```@eval
Main.ReactiveKernelsDocs.render_practicalbayes_mnist_baselines()
```

## Evaluation throughput

The native comparison uses the same iid-Normal model, position sizes 16, 256,
and 4096, Float64 inputs, and three requested outputs as the primary receipt:
linked log density, analytic-parity-checked linked gradient, and the returned
pointwise log-density vector. All nine native cells are measured. The pinned
public API has no Reactant compiler boundary, so the nine compiled cells are
explicitly unsupported.

```@eval
Main.ReactiveKernelsDocs.render_practicalbayes_eval_benchmark()
```

The executed public model definition:

```@eval
Main.ReactiveKernelsDocs.render_practicalbayes_eval_baseline()
```

## NUTS sampling

The public sampler cells call `PracticalBayes.sample` with
`AdvancedHMC.NUTS(0.8)`, the Enzyme backend, matched initial positions, seed
20260901, 1,000 adaptation transitions, and 1,000 retained draws. The sampler
object is checked at runtime for a diagonal metric, depth cap 10, and maximum
energy error 1000. Eight Schools uses the linked centered posterior; MNIST uses
the same first-1,000-image, 40-component unwhitened PCA workload as the ProbProg
sampling matrix. Density parity is checked at deterministic linked points before
sampling, and divergence counts remain diagnostics rather than pass/fail gates.

```@eval
Main.ReactiveKernelsDocs.render_practicalbayes_mcmc_benchmark()
```

Setup, package resolution, precompilation, layout construction, AD preparation,
and first calls are recorded separately. Model/evaluation steady-state rows use
ten independent BenchmarkTools rounds and retain minimum and median runtimes,
raw rounds, allocated bytes, and allocation counts. The NUTS sampling time is
the complete public `sample` call, including its internal layout/AD preparation,
adaptation, and retained draws; a separate short JIT warm-start is also retained.

## Reproduce

The generator requires a clean detached candidate so the receipt's RK source
pin is immutable:

```sh
julia --startup-file=no benchmark/practicalbayes_comparison.jl \
  --output=benchmark/receipts/practicalbayes-comparison-v1.toml
julia --startup-file=no benchmark/receipts/validate_practicalbayes.jl \
  benchmark/receipts/practicalbayes-comparison-v1.toml
```

The checked-in publication receipt is
[`benchmark/receipts/practicalbayes-comparison-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/practicalbayes-comparison-v1.toml).
For a reduced diagnostic run, set `RK_PRACTICALBAYES_MNIST_N`,
`RK_PRACTICALBAYES_ROUNDS`, `RK_PRACTICALBAYES_SAMPLES_PER_ROUND`,
`RK_PRACTICALBAYES_MCMC_WARMUP`, and `RK_PRACTICALBAYES_MCMC_SAMPLES`; the
validator deliberately rejects those reduced settings as publication evidence.
