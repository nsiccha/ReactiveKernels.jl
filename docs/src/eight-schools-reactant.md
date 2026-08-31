# Eight Schools primal kernels through Reactant

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page measures Reactant on the exact executable model documented on the
[Eight Schools kernel page](eight-schools.md). The benchmark imports
`EIGHT_SCHOOLS_SOURCE` and `evaluate_eight_schools_source` from
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

This page is primal-only. It does not time gradients or claim anything about
sampler throughput, adaptation, draws, ESS, accelerators, or
time-to-effective-sample. The separate
[automatic-differentiation page](automatic-differentiation.md) owns AD
comparisons and their backend contract.

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
