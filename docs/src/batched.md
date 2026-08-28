# Batched log densities, for free

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

A density written as **(1) an elementwise pointwise recipe over an array-typed
port → (2) a `sum` reduction** plans to a single straight-line vectorized
kernel. The author writes one *scalar* `normal_logpdf`; **batching is just
passing a `Vector` where a scalar went**. From that one graph, want-set pruning
yields either the per-observation log density (`want = :per_obs`, for
LOO/WAIC/PSIS) or the total (`want = :logdensity`), and a single `Enzyme`
reverse pass differentiates the whole batch at once. This is exactly Stan's
mechanism — a vectorized density statement plus reverse-mode AD — with no
batching-specific user code.

As in [Native log densities as recipes](distributions.md), the **compute path
contains no `Distributions.jl` call**: the density is written out directly and
`Distributions.jl` appears only as an independent oracle. Differentiation goes
through `DifferentiationInterface` with the `Enzyme` reverse-mode backend, over
the native recipe only.

## One graph, two want boundaries

The panel below is build-executed. It prepares the same graph twice —
`want = :logdensity` for the total and `want = :per_obs` for the length-`N`
vectorized pointwise density — checks both against a `Distributions.jl` oracle,
and checks a full reverse-mode gradient over the `N`-dimensional batch against
the analytic score `-(xᵢ - μ)/σ²`. The **Generated kernel** view confirms the
lowered kernel names no distribution library.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BatchedExamples.BATCHED_SOURCE,
)
```

The pointwise density is passed as a **typed port**
(`pointwise::typeof(normal_logpdf)`) rather than baked into the recipe. That is
the small broadcast-lift convention that keeps the elementwise operation the
bare `broadcast`, which the non-allocating lowering below can reuse a buffer
for; a captured constant would instead lower to an anonymous closure that
allocates.

## No unnecessary work: want-set pruning

`want = :per_obs` and `want = :logdensity` are the *same* graph. Pruning is
structural, not a runtime branch: the per-observation plan holds strictly fewer
recipes (the `sum` reduction is dropped, not merely skipped), and the total plan
computes the pointwise vector only as the necessary intermediate for its sum.
There is no duplicated arithmetic and the shared `σ = exp(logσ)` subexpression
is emitted once. The `eight_schools.jl` example splits `pointwise_log_likelihood`
→ `sum_log_likelihood` the same way.

## No memory we don't need: zero-allocation value

The plain [`prepare`](api.md) path materializes the per-observation vector on
every call. The non-allocating lowering under
[Non-allocating kernels](nonallocating.md) reuses that batch buffer across
calls, driving **both** want boundaries to **zero bytes** in steady state
(measured behind a function barrier after warm-up):

```@eval
Main.ReactiveKernelsDocs.render_batched_allocations()
```

This is the synergy the batched pattern was designed around: a preallocated
batch buffer reused across the millions of serial density evaluations a sampler
makes.

## One reverse pass, zero-allocation gradient

The gradient over the whole `N`-dimensional batch is a **single** `Enzyme`
reverse pass — its cost is essentially independent of `N` (the reverse-mode
signature), not the `N×` of a scalar loop. The batch mixes the active parameter
vector with the constant `μ`, `σ` inside the broadcast, so the backend needs
runtime activity, and the closed-over kernel/primal is `Const`:

```julia
AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
             function_annotation = Enzyme.Const)
```

The zero-allocation gradient reuses a preallocated batch buffer, but **not** the
borrowed cache of a `prepare_nonallocating` kernel. Differentiating that kernel
is unsound: its per-recipe caches are overwritten between the forward and
reverse sweeps and silently corrupt the adjoint — the reverse-mode instance of
the "owned state, not borrowed caches" rule in
[Non-allocating kernels](nonallocating.md). Instead the batch buffer is a
preallocated `DifferentiationInterface` `Cache`
that `Enzyme` owns and shadows itself, giving a **zero-byte** steady-state
gradient that matches the analytic score to machine precision:

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

## Reproducible integration gate

The default package tests stay independent of the unregistered weak dependency;
the `Distributions.jl`-free value, want-pruning, and gradient checks above run in
the default suite (`test/test_batched_example.jl`). The zero-allocation value and
gradient claims need the pinned `MutatingFunctions` revision and are proven
separately in `test/test_batched_nonallocating.jl`, run through the same gate as
the other non-allocating tests:

```sh
julia --startup-file=no test/run_nonallocating_integration.jl
```
