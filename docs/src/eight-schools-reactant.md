# Eight Schools kernels through Reactant

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page measures Reactant on the exact executable model documented on the
[Eight Schools kernel page](eight-schools.md). The benchmark imports
`EIGHT_SCHOOLS_SOURCE` and `build_eight_schools_graph` from
`ReactiveKernelsPPLExamples`; it does not copy the model, rewrite a density, or
introduce a Reactant-only mathematical path.

The receipt mirrors the complete three-by-four capability matrix in the
[native primal receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-primal-v2.toml).
Its input boundaries are the packed unconstrained vector,
the constrained parameter `NamedTuple`, and the minimal likelihood boundary
(`θ`, observations, scales). Its requested outputs are the joint density,
prior, summed likelihood, and pointwise likelihood. Two minimal-boundary cells
are mathematically undefined because the prior parameters are absent. Every
other cell is evaluated natively and attempted through Reactant.

Every mathematically defined primal cell compiles in both data modes. The two
minimal-boundary joint/prior cells remain unsupported by definition, and a
bound-data prior is N/A because its backward slice has no data ports. Those
states stay in the table rather than being omitted or replaced with a different
HAVE boundary.

## Primal performance and support

The headline view separates joint, prior, summed-likelihood, and pointwise
outcomes at the packed boundary. Runtime-input and bound-input compiled calls
are each normalized to their embedded, configuration-matched native RK control;
the complete capability matrix and compiler diagnostics remain in the receipt.

```@eval
Main.ReactiveKernelsDocs.render_eight_schools_reactant_benchmark()
```

The plots and compact tables contain steady-state synchronous call time only. Host-to-device
conversion, kernel preparation, Reactant compilation, the first synchronous
call, and result readback are outside that timing and remain recorded in the
receipt. This keeps a small CPU kernel's fixed compiler/runtime costs visible
without mixing them into repeated-call performance.

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

This section is the AD analog of the primal panels above. It consumes the
first-class RK verb `compile_ad_value_and_gradient` (the AD companion of the
primal `@compile` path) — no gradient is hand-rolled — and reuses the exact
differentiable outcome/boundary protocol published on the
[automatic-differentiation page](automatic-differentiation.md)
([`eight-schools-ad-v2`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-ad-v2.toml)).
Native RK supports scalar gradients for packed, constrained-`NamedTuple`, and
minimal boundaries, plus packed/minimal pointwise VJPs. The compiled public
surface is deliberately narrower: it exposes scalar value-and-gradient for one
array active port, but no compiled reverse-pullback verb and no structured
active-argument/result ABI.

Every declared scalar array-active AD cell now compiles through Reactant in both
unbound and partially evaluated data modes, including the packed full joint.
Pointwise VJPs and constrained structured gradients remain explicit compiled-API
unsupported cells even though their native counterparts are public and measured;
the undefined minimal joint/prior remain unsupported by definition. As with the
primal table, AD preparation, host transfers, gradient compilation, the first
synchronous call, and readback are excluded from steady-state timing.

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
  --output=benchmark/receipts/eight-schools-reactant-v2.toml
julia --startup-file=no benchmark/receipts/validate_eight_schools_reactant.jl \
  benchmark/receipts/eight-schools-reactant-v2.toml
```

The script provisions a fresh environment with Reactant 0.2.284, develops the
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
  --output=benchmark/receipts/eight-schools-reactant-ad-v2.toml
julia --startup-file=no benchmark/receipts/validate_eight_schools_reactant_ad.jl \
  benchmark/receipts/eight-schools-reactant-ad-v2.toml
```

Its quick-smoke knobs are `RK_EIGHT_SCHOOLS_REACTANT_AD_ROUNDS` and
`RK_EIGHT_SCHOOLS_REACTANT_AD_TARGET_SECONDS`.
