# PPL automatic differentiation

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

The reviewed PPL gradient evidence is ordered like the model documentation:
[Eight Schools](eight-schools.md) first, then
[MNIST multinomial logistic regression](mnist-logistic.md). Both reuse the exact
published primal graph and comparator authorities; there is no AD-specific copy
of either model.

## Eight Schools model gradient matrix

`prepare_ad` differentiates the exact generated callable and operation table
used by primal execution. The four scalar cells are the packed unconstrained
joint, prior, and likelihood plus the minimal θ-only likelihood. Each is checked
against central finite differences before timing.

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_ad_benchmarks()
```

### Baseline implementations

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_ad_baselines()
```

The constrained parameter object is not a supported active storage type, and
pointwise output remains blank because the compared public surfaces do not share
a matched Jacobian/VJP contract. The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-ad-v1.toml)
retains raw rounds, source and primal pins, parity errors, preparation costs,
timings, and allocations.

## MNIST full-data model gradients

The sampler-relevant packed `[vec(W); b]` vector is the one active port; the
60,000 training images, labels, and class count are rebound as constants. An
independent analytic reference-class softmax score checks all RK, manual Julia,
and Turing gradients before timing.

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_ad_benchmarks()
```

The structured `(W, b)` boundary remains unsupported because public
ReactiveKernels AD selects exactly one active HAVE port. Pointwise output is not
replaced by a benchmark-only surrogate. The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-ad-v1.toml)
retains the complete data shape, 7,065 active coefficients, analytic parity,
preparation costs, timings, and allocations.

## Other model coverage

The remaining declarative walkthroughs are correctness-tested through the same
prepared boundary in
[`test_ppl_enzyme.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/test/test_ppl_enzyme.jl).
They remain after the reviewed Eight Schools and MNIST material in navigation;
no additional benchmark claim is inferred from those checks.
