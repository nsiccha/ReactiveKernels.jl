# Eight Schools kernels through Reactant

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page measures Reactant on the exact executable model documented on the
[Eight Schools kernel page](eight-schools.md). The benchmark imports
`EIGHT_SCHOOLS_SOURCE` and `build_eight_schools_graph` from
`ReactiveKernelsPPLExamples`; it does not copy the model, rewrite a density, or
introduce a Reactant-only mathematical path.

The comparison mirrors the complete three-by-four capability matrix in the
[native primal receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-primal-v1.toml).
Its input boundaries are the packed unconstrained vector,
the constrained parameter `NamedTuple`, and the minimal likelihood boundary
(`θ`, observations, scales). Its requested outputs are the joint density,
prior, summed likelihood, and pointwise likelihood. Two minimal-boundary cells
are mathematically undefined because the prior parameters are absent. Every
other cell is evaluated natively and attempted through Reactant.

Unsupported compiler cells stay in the table with their actual diagnostic.
They are not silently omitted, replaced with a different HAVE boundary, or
timed through host fallback. That distinction matters: the table describes
which views of this one graph compile today as well as how the compiled views
perform.

## Primal performance and support

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_reactant_benchmark()
```

The first table contains steady-state synchronous call time only. Host-to-device
conversion, kernel preparation, Reactant compilation, the first synchronous
call, and result readback are outside that timing. The second table and setup
summary report those costs separately, including failed compile attempts. This
keeps a small CPU kernel's fixed compiler/runtime costs visible without mixing
them into repeated-call performance.

The section above times primal densities only — no gradients, and nothing about
sampler throughput, adaptation, draws, ESS, accelerators, or
time-to-effective-sample. The Reactant-compiled gradient column follows below;
native AD (RK vs Turing vs manual, no Reactant) lives on the separate
[automatic-differentiation page](automatic-differentiation.md), whose derivative
matrix this page reuses exactly.

## Reactant-compiled automatic differentiation

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_reactant_ad_benchmark()
```

This section is the AD analog of the primal table above. It consumes the
first-class RK verb `compile_ad_value_and_gradient` (the AD companion of the
primal `@compile` path) — no gradient is hand-rolled — and reuses the exact
differentiable outcome/boundary protocol published on the
[automatic-differentiation page](automatic-differentiation.md)
([`eight-schools-ad-v1`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-ad-v1.toml)):
the value and gradient of each scalar output with respect to a single active
port (the packed unconstrained vector, or `θ`). Non-scalar `pointwise` outputs,
the constrained parameter `NamedTuple`, and the undefined minimal joint/prior
stay unsupported, exactly as on the AD page.

A Reactant-compiled gradient exists only where the primal kernel itself compiles
through Reactant, so the compiled cells are a subset of the native-AD cells: the
packed joint and prior — which fail the primal Reactant path with "Scalar
indexing is disallowed." — keep native AD but no Reactant gradient. Every
compiled cell matches the native RK reverse pass bit-for-bit (gradient and value
max-abs-error 0). As with the primal table, AD preparation, host transfers,
gradient compilation, the first synchronous call, and readback are excluded from
the steady-state timing and reported separately.

For the exact model graph see the [Eight Schools kernel page](eight-schools.md);
for native-AD (non-Reactant) timing see the
[automatic-differentiation page](automatic-differentiation.md).

## Reproduce

The receipt generator requires a clean detached candidate so its full commit
pin and source blob are immutable. From a sibling directory:

```sh
git -C ReactiveKernels.jl worktree add --detach ReactiveKernels-eight-schools-receipt HEAD
cd ReactiveKernels-eight-schools-receipt
julia --startup-file=no benchmark/eight_schools_reactant_comparison.jl \
  --output=benchmark/receipts/eight-schools-reactant-v1.toml
julia --startup-file=no benchmark/receipts/validate_eight_schools_reactant.jl \
  benchmark/receipts/eight-schools-reactant-v1.toml
```

The script provisions a fresh environment with Reactant 0.2.278, develops the
exact candidate plus its two example packages, and records environment setup
and package precompilation separately. For a quick non-publication smoke run,
set `RK_EIGHT_SCHOOLS_REACTANT_ROUNDS=2` and
`RK_EIGHT_SCHOOLS_REACTANT_TARGET_SECONDS=0.002`; the checked-in receipt keeps
the publication protocol of at least 20 rounds.

The Reactant-compiled-AD receipt is generated the same way from the same clean
detached candidate, and additionally resolves Enzyme and DifferentiationInterface
into the pinned environment:

```sh
julia --startup-file=no benchmark/eight_schools_reactant_ad_comparison.jl \
  --output=benchmark/receipts/eight-schools-reactant-ad-v1.toml
julia --startup-file=no benchmark/receipts/validate_eight_schools_reactant_ad.jl \
  benchmark/receipts/eight-schools-reactant-ad-v1.toml
```

Its quick-smoke knobs are `RK_EIGHT_SCHOOLS_REACTANT_AD_ROUNDS` and
`RK_EIGHT_SCHOOLS_REACTANT_AD_TARGET_SECONDS`.
