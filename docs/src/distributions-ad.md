# Distribution AD: scalar and batched

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page collects the reviewed automatic-differentiation results for
[distribution objects](distributions.md) and
[batched log densities](batched.md). ReactiveKernels owns the prepared-kernel
boundary; `DifferentiationInterface` and the selected backend own the reverse
pass.

## One reverse pass over a plated objective

The same API used by a scalar kernel applies to an authored likelihood whose
`plate` result is summed. The build-executed source below is the exact primal
source from [Batched log densities](batched.md): one authored graph supports its
return, pointwise, combined, and prepared-gradient boundaries.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BatchedExamples.BATCHED_AD_SOURCE,
)
```

The result is checked against the analytic score
`-(xᵢ - location)/scale²`. The generated-kernel pane comes from the exact plan
executed during this docs build.

## Distribution gradient latency and allocation

The receipt below reuses the same inventories as the distribution page: Normal
plates, seven scalar families, and covariance-Cholesky multivariate Normal.
Continuous families differentiate observations; Bernoulli and Geometric
differentiate their scalar logit ports because integer observations cannot be
active. Every row is checked against an analytic gradient before timing.

Each distribution family is rendered as its own relative-runtime plot and
compact table. The baseline is `ad_gradient` for that same family and plate
size; absolute runtime, allocation evidence, the normalized ratio, and a plain
faster/slower interpretation stay together.

```@eval
Main.ReactiveKernelsDocs.render_distribution_gradient_benchmarks()
```

Each displayed value uses three significant digits and comes from the median
of five minimum-time measurements after preparation.
`ad_gradient` returns only the derivative;
`ad_value_and_gradient!` also returns the value while filling caller-owned
storage. The checked-in
[receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/distribution-gradient-v1.toml)
retains raw times, allocations, analytic errors, source links, and package pins.

## Storage boundary

Do not differentiate a `NonAllocatingKernel` whose recipe caches are borrowed
and overwritten on every call: the backward pass needs forward intermediates to
remain valid. When a derivative needs reusable batch storage, keep the primal
operation explicit and give DifferentiationInterface an owned `Cache`.

The executable authority is
[`test_batched_nonallocating.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl).
It checks caller-owned storage, backend cache ownership, the analytic score, and
zero steady-state allocations.
