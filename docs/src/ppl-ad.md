# PPL automatic differentiation

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

The reviewed PPL gradient evidence is ordered like the model documentation:
[Eight Schools](eight-schools.md) first, then
[MNIST multinomial logistic regression](mnist-logistic.md). Both reuse the exact
published primal graph and comparator authorities; there is no AD-specific copy
of either model.

## Eight Schools focused value-and-gradient comparisons

`prepare_ad` differentiates the exact generated callable and operation table
used by primal execution. Scalar value-and-gradient cells cover the packed
unconstrained joint, prior, and likelihood; the constrained parameter
`NamedTuple` joint, prior, and likelihood; and the minimal θ-only likelihood,
with data unbound or fixed during preparation where applicable. Packed and
minimal pointwise likelihoods use the public reverse-pullback surface with one
fixed receipt cotangent. Every supported cell is checked against central finite
differences before timing.

The page leads with four separate packed-boundary outcome panels. Each compact
table reports absolute runtime, the matching RK baseline, `runtime ÷ baseline`,
allocation evidence, and a faster/slower interpretation. The full derivative
boundary matrix, unsupported reasons, preparation costs, and raw rounds remain
available in the receipt.

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

Joint, prior, and likelihood each have their own relative-runtime plot and
compact comparison table. RK + Enzyme is the explicit 1.00× baseline; tables
also state the faster/slower interpretation and use three-significant-digit
runtime and allocation units. Unsupported pointwise and structured cells remain
in the receipt rather than sharing a misleading scale with scalar timings.

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

## MNIST Wren-compatible PCA-40 gradients

The same RK/Enzyme, handwritten/Enzyme, and Turing/Enzyme value-and-gradient
comparison is repeated on the first 1,000 MNIST observations projected onto
Wren's 40-component unwhitened PCA basis. The supported-cell inventory is
unchanged; only the shared data profile and resulting 369-element packed
coefficient vector differ.

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_ad_wren_benchmarks()
```

The separate
[PCA-40 receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/mnist-logistic-ad-wren-pca40-v1.toml)
cross-pins the matching standalone PCA-40 primal receipt and retains the same
analytic parity, preparation, timing, and allocation evidence.

Reproduce it from the same clean detached checkout used for the primal receipt:

```sh
julia --startup-file=no benchmark/mnist_logistic_ad_comparison.jl \
  --dataset=wren-pca40 --wren-reference=/path/to/mnist.csv \
  --output=benchmark/receipts/mnist-logistic-ad-wren-pca40-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_logistic_ad.jl \
  benchmark/receipts/mnist-logistic-ad-wren-pca40-v1.toml
```

## Other model coverage

The remaining declarative walkthroughs are correctness-tested through the same
prepared boundary in
[`test_ppl_enzyme.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/test/test_ppl_enzyme.jl).
They remain after the reviewed Eight Schools and MNIST material in navigation;
no additional benchmark claim is inferred from those checks.
