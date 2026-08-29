# Batched log densities, for free

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

A density can expose both a per-element recipe over an array input and a fused
scalar reduction. The author writes one scalar `normal_logpdf`; from that one
graph, asking for a different `want` gives either the per-observation log density
(`want = :per_obs`, for LOO/WAIC/PSIS) or the total (`want = :logdensity`). A
single `Enzyme` reverse-mode pass takes the gradient of the whole batch at once.

As in [Native log densities as recipes](distributions.md), the **compute path
contains no `Distributions.jl` call**: the density is written out directly and
`Distributions.jl` appears only as an independent oracle. The native recipe is
prepared through `prepare_ad` and evaluated through `ad_gradient`, using the
plain `Enzyme` reverse-mode backend behind DifferentiationInterface.

## One graph, two want boundaries

The panel below prepares the same graph twice — `want = :logdensity` for the
total and `want = :per_obs` for the length-`N` vectorized pointwise density —
checks both against a `Distributions.jl` oracle, and checks a full reverse-mode
gradient over the `N`-dimensional batch against the analytic score
`-(xᵢ - μ)/σ²`. The **Generated kernel** view confirms the lowered kernel names
no distribution library.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BatchedExamples.BATCHED_SOURCE,
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

## One reverse pass, zero-allocation gradient

The gradient over the whole `N`-dimensional batch is a **single** `Enzyme`
reverse-mode pass — its cost is essentially independent of `N`, not the `N×` of a
scalar loop (that is what reverse-mode buys you). The owned buffer and shared
parameters are classified explicitly at the DifferentiationInterface boundary,
so ordinary reverse mode is sufficient:

```julia
AutoEnzyme(; mode = Enzyme.Reverse)
```

The zero-allocation gradient reuses a preallocated batch buffer, but **not** the
borrowed cache of a `prepare_nonallocating` kernel. Differentiating that kernel
is unsound: its caches get overwritten between the forward and backward passes
and silently corrupt the gradient — the automatic-differentiation version of the
"owned state, not borrowed caches" rule in
[Non-allocating kernels](nonallocating.md). Instead the batch buffer is a
preallocated `DifferentiationInterface` `Cache` that `Enzyme` manages itself,
giving a **zero-byte** steady-state gradient that matches the analytic score to
machine precision:

```julia
using DifferentiationInterface: Cache, Constant

primal(x, buffer, μ, logσ) = begin
    σ = exp(logσ)
    broadcast!(normal_logpdf, buffer, x, μ, σ)
    sum(buffer)
end

buffer = similar(x)
prep = prepare_gradient(primal, backend, x, Cache(buffer), Constant(μ), Constant(logσ))
gradient!(primal, grad, prep, backend, x, Cache(buffer), Constant(μ), Constant(logσ))
```

## Reproducing the zero-allocation claims

The zero-allocation value and gradient claims need the pinned `MutatingFunctions`
revision. Reproduce them with:

```sh
julia --startup-file=no test/run_nonallocating_integration.jl
```
