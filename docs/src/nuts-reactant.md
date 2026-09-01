# Adaptive NUTS through Reactant: static receipt

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page preserves the frozen adaptive-NUTS Reactant receipt without executing
NUTS compiler or runtime code during the docs build. It is historical
compiler-acceptance evidence, not a claim that the current moving NUTS or
generic structural-state frontier should gate documentation publication.

The source authority is
[`kernel_nuts_reactant.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsNUTSExamples/src/nuts_runtime/kernel_nuts_reactant.jl),
and the executable acceptance authority is
[`test_kernel_nuts_reactant.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/test/test_kernel_nuts_reactant.jl).
Neither is loaded or run by this page.

## Frozen matched-control result

```@eval
Main.ReactiveKernelsDocs.render_nuts_reactant_benchmark()
```

The checked-in receipt compares one synchronous CPU call per full-depth
transition, with identical pre-generated momentum, direction, and exponential
bundles. Compilation, transfers, state setup, random generation, and result
readback are reported outside steady-state timing.

The immutable input is
[`nuts-reactant-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-reactant-v1.toml).
It is not adaptation, retained-draw, ESS, accelerator-transfer, or
time-to-effective-sample evidence. Sampling and WALNUTS compiler work remains
outside the docs execution gate.
