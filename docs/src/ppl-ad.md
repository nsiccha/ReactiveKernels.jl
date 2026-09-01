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
used by primal execution. Scalar value-and-gradient cells cover the packed
unconstrained joint, prior, and likelihood; the constrained parameter
`NamedTuple` joint, prior, and likelihood; and the minimal θ-only likelihood,
with data unbound or fixed during preparation where applicable. Packed and
minimal pointwise likelihoods use the public reverse-pullback surface with one
fixed receipt cotangent. Every supported cell is checked against central finite
differences before timing.

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_ad_benchmarks()
```

### Baseline implementations

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_ad_baselines()
```

Structured native sensitivities preserve the constrained `NamedTuple`; array
sensitivities use caller-owned destinations. Pointwise rows report one VJP
(`J' * output_cotangent`), not a full Jacobian. The constrained/pointwise
cross-product remains explicitly unsupported because the current Enzyme backend
cannot represent that `MixedDuplicated` combination. Turing stays visible only
where its public density boundary supplies a genuinely matched comparison. The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-ad-v2.toml)
retains raw rounds, source and primal pins, parity errors, preparation costs,
timings, and allocations.

## MNIST full-data model gradients

The sampler-relevant packed `[vec(W); b]` vector is the one active port. Both
model sources are measured with the 60,000 training images, labels, and class
count either ordinary HAVE arguments or fixed during preparation. An
independent analytic reference-class softmax score checks all RK, manual Julia,
and Turing gradients before timing.

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_ad_benchmarks()
```

### Baseline implementations

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_ad_baselines()
```

The structured `(W, b)` boundary remains unsupported because public
ReactiveKernels AD selects exactly one active HAVE port. Pointwise output is not
replaced by a benchmark-only surrogate. The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-ad-v2.toml)
retains the complete data shape, 7,065 active coefficients, analytic parity,
preparation costs, timings, and allocations.

## Other model coverage

The remaining declarative walkthroughs are correctness-tested through the same
prepared boundary in
[`test_ppl_enzyme.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/test/test_ppl_enzyme.jl).
They remain after the reviewed Eight Schools and MNIST material in navigation;
no additional benchmark claim is inferred from those checks.
