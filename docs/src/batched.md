# Batched log densities, for free

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

Write the likelihood once as ordinary Julia: a `plate` computes the
per-observation values and the kernel returns their sum. The resulting graph has
three useful views without a second density formula:

- ordinary application or `prepare(normal_loglik)` returns the scalar total;
- `extract(normal_loglik; want = :pointwise)` returns the pointwise values used
  by LOO, WAIC, and PSIS; and
- `extract(normal_loglik; want = (:pointwise, :__return__))` returns both in one
  traversal.

The scalar endpoint is the canonical transparent `normal` distribution object
from [Distribution kernels](distributions.md). `Distributions.jl` remains an
independent numerical oracle; it is absent from every generated compute path.

## One authored graph, three queries

This panel executes the exact source used by the nested example package. It
checks ordinary application, return-only preparation, pointwise extraction, and
the combined query against the same oracle. The generated-kernel view is the
actual return-only lowering from this docs build.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BatchedExamples.BATCHED_PRIMAL_SOURCE,
)
```

The query changes the prepared boundary, not the model. Return-only lowering
has one reduction loop and no pointwise output allocation. Pointwise-only has
one loop and materializes the requested array. Asking for both still has one
loop: each pointwise value is stored and immediately added to the return.

## Broadcasting is the batching contract

`plate` follows Julia broadcasting semantics. Equal array dimensions align and
zip; singleton dimensions expand; scalars repeat. `Ref(value)` marks an
array-valued argument as one atomic value rather than a batch axis. Incompatible
shapes raise `DimensionMismatch`.

There is no separate public axis or scheduling language. The one-axis example
above is the simplest case of that contract; multidimensional inputs use the
same broadcast rules.

## Measured parity with the established plate path

The checked-in Normal receipt compares the authored return-only kernel with the
established `plate(normal.logpdf; ...)` reduction in the **same process, same
rounds, and same data** at
`N = 1, 1,000, 10,000, 30,000, 100,000, 1,000,000`. Each cell retains raw
per-round timings, allocation bytes, and allocation counts. The benchmark also
compiles both paths with Reactant and checks numerical parity.

```@eval
Main.ReactiveKernelsDocs.render_batched_benchmarks()
```

The native hard gate requires the authored path to stay within 10% of the
established plate path for every `N ≥ 1,000`; `N = 1` is reported but excluded
from that ratio gate because timer quantization dominates such a short call.
Return-only authored execution must remain zero-byte and zero-allocation at all
six sizes. Before timing, the harness also rejects a native `similar` output or
a Reactant lowering that is not a tensorized broadcast chain consumed by
`sum`.

## Reproduce the receipt

From the repository root:

```sh
julia --startup-file=no --project=benchmark/distributions \
  benchmark/distributions/setup.jl
julia --startup-file=no --project=benchmark/distributions \
  benchmark/distributions_comparison.jl \
  --output=benchmark/receipts/distribution-logdensity-v1.toml
julia --startup-file=no benchmark/receipts/validate_distributions.jl \
  benchmark/receipts/distribution-logdensity-v1.toml
```

For a reusable pointwise output buffer, the optional MutatingFunctions
integration prepares the pointwise extraction non-allocatingly. The focused
fixture is
[`packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl).
