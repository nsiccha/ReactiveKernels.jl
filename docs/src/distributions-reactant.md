# Distribution kernels through Reactant

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

The mathematical and executable source authorities remain
[Distribution kernels](distributions.md) and
[Batched log densities](batched.md). Reactant traces those same prepared scalar,
structured, and plated kernels; there is no Reactant-specific distribution
formula or parallel source tree. This page owns the Reactant benchmark panels so
the native source pages stay focused on authoring and semantics.

Every benchmark below uses focused plots rather than one mixed-scale summary:
scalar families are separated by distribution, and native, compiled, authored,
and amortized comparisons use execution-matched baselines. Compact tables keep
absolute runtime, baseline runtime, `runtime ÷ baseline`, allocations where
available, and a faster/slower interpretation beside each plot.

## Scalar families

The matched comparison covers Cauchy, Laplace, Bernoulli, LogNormal,
Exponential, Geometric, and Uniform measures from Distributions,
ProbabilityMeasures, and the generic ReactiveKernels `plate` generated from the
public objects. Parameters are traced runtime inputs; compilation and transfers
are excluded from execution timings.

```@eval
Main.ReactiveKernelsDocs.render_scalar_gallery_benchmarks()
```

Unsupported Distributions + Reactant cells retain their constructor diagnostic
instead of disappearing. The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/scalar-distribution-gallery-v1.toml)
contains raw samples, allocations, support results, and exact pins.

## Structured families

The covariance-Cholesky multivariate-Normal comparison measures evaluation, not
construction or factorization. AR(1) is not benchmarked because neither
comparison package exposes a matched native AR(1) distribution.

```@eval
Main.ReactiveKernelsDocs.render_structured_distribution_benchmarks()
```

Unsupported traced-array cells remain visible measured compatibility results.
The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/structured-distribution-logdensity-v1.toml)
retains raw samples, allocations, compiler diagnostics, and pins.

## Scalar plate and call-boundary amortization

This comparison evaluates the same batched Normal log density with shared
location and scale. It includes the shared `normal.logpdf` object and a one-off
ReactiveKernels formula as a matched control alongside each library's idiomatic
vectorized public interface.

```@eval
Main.ReactiveKernelsDocs.render_distribution_benchmarks()
```

For a tiny scalar density, fixed host/runtime overhead dominates. Lifting the
same scalar `PreparedKernel` with `replica(...; batched = :x)` amortizes that
boundary over independent evaluations without changing the isolated-call
latency.

```@eval
Main.ReactiveKernelsDocs.render_distribution_amortization()
```

Reactant timings synchronize before measurement and exclude compilation and
transfers. Host allocations are Julia wrapper allocations, not device memory.
The
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/distribution-logdensity-v1.toml)
contains the raw samples and acceptance pins.

## Batched authored graph parity

The authored return-only graph and the established `plate(normal.logpdf; ...)`
path run in one process, on the same data and rounds. The receipt compiles both
through Reactant and requires numerical parity while retaining the native
zero-allocation and throughput gates.

```@eval
Main.ReactiveKernelsDocs.render_batched_benchmarks()
```

The benchmark also rejects a native `similar` output or a Reactant lowering that
is not a tensorized broadcast chain consumed by `sum`. Reproduction commands
and the authored source remain on [Batched log densities](batched.md).

## Support boundary

Reactant support is optional and covers accepted fixed-shape numeric inputs and
the documented prepared boundaries. Unsupported traced scalar indexing or
storage shapes fail closed; a separate formula is not substituted merely to
make compilation succeed.
