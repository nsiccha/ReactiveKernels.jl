# Batched log densities, for free

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

A density can expose both a per-element recipe over an array input and a fused
scalar reduction. The author writes one scalar `normal_logpdf`; from that one
graph, asking for a different `want` gives either the per-observation log density
(`want = :per_obs`, for LOO/WAIC/PSIS) or the total (`want = :logdensity`).

As in [Native log densities as recipes](distributions.md), the **compute path
contains no `Distributions.jl` call**: the density is written out directly and
`Distributions.jl` appears only as an independent oracle.

## One graph, two want boundaries

The panel below prepares the same graph twice — `want = :logdensity` for the
total and `want = :per_obs` for the length-`N` vectorized pointwise density —
and checks both against a `Distributions.jl` oracle. The **Generated kernel**
view confirms the lowered kernel names no distribution library.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BatchedExamples.BATCHED_PRIMAL_SOURCE,
)
```

The pointwise density is passed in as a **typed port**
(`pointwise::typeof(normal_logpdf)`) rather than hard-coded into the recipe. This
small convention keeps the per-element step a bare `broadcast` call, which the
non-allocating version below can give a reusable buffer; hard-coding the function
instead would produce an anonymous closure that allocates.

## No unnecessary work: pruning by want

`want = :per_obs` and `want = :logdensity` come from the *same* graph. The
pruning happens when the kernel is prepared, not as an `if` at run time: the
per-observation plan selects the broadcast producer, while the total plan
selects a fused scalar-loop producer and never allocates the pointwise vector.
Both compute the shared `σ = exp(logσ)` once. The PPL examples use the same
two-producer pattern when they expose both pointwise and density-only queries.

## No memory we don't need: zero-allocation value

The plain [`prepare`](api.md) path builds the per-observation vector on every
call. The non-allocating version under
[Non-allocating kernels](nonallocating.md) reuses one batch buffer across calls,
driving **both** wants to **zero bytes** in steady state (measured behind a
function barrier after warm-up):

```@eval
Main.ReactiveKernelsDocs.render_batched_allocations()
```

This is the synergy the batched pattern was designed around: a preallocated
batch buffer reused across the millions of serial density evaluations a sampler
makes.

## Reproducing the zero-allocation claims

The zero-allocation value claim needs the pinned `MutatingFunctions` revision.
Reproduce it with:

```sh
julia --startup-file=no test/run_nonallocating_integration.jl
```
