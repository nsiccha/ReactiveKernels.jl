# ProbProg MCMC: compiled NUTS over RK densities (historical receipt)

> **Historical receipt.** ReactiveKernels no longer has a Reactant path: its
> Reactant extension, tensorized lowering and Reactant tests were removed. This
> page keeps the measurements as recorded. Nothing on it runs Reactant, and the
> RK functions and test files it names no longer exist.

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

Reactant's [`ProbProg` module](https://enzymead.github.io/Reactant.jl/dev/tutorials/probprog/mcmc)
ships an end-to-end compiled MCMC engine: `ProbProg.mcmc_logpdf` takes an
**arbitrary traced log density**, and lowers the *entire* warmup + sampling
loop — NUTS tree building, dual-averaging step-size adaptation, and windowed
Welford diagonal mass-matrix adaptation — into one MLIR operation whose
gradients Enzyme takes inside the compiled program. One synchronous call
returns every retained draw.

That makes it a natural *consumer* of ReactiveKernels' compiled-density
boundary: the packed-unconstrained prepared kernels documented on the
[Eight Schools](eight-schools.md) and [MNIST](mnist-logistic.md) pages are
exactly the log densities `mcmc_logpdf` wants. This page samples every cell
that compiles at the pinned version that way — and records the one that does
not as an explicit rejection — comparing against two established native
harnesses:

- **AdvancedHMC NUTS over the same RK density** — the standard Julia sampler
  ecosystem consuming RK natively: the density closure is the RK prepared
  kernel, the gradient closure is the RK prepared DI+Enzyme reverse gradient
  from the [automatic-differentiation page](automatic-differentiation.md).
- **Turing NUTS** — the full PPL stack on the source-attested Turing twin of
  the same model, with the suite's established Enzyme configuration.

ProbProg is *not* a port of ReactiveKernels' own samplers: it is an
upstream-maintained NUTS with its own adaptation schedule and tree
implementation. It complements, and does not replace, the
[adaptive Reactant NUTS](nuts-reactant.md) that proves RK's compiler can lower
RK's *own* sealed sampler semantics. The receipt below is also the
outer-loop-amortization counterpoint to that page's one-synchronous-call-per-
transition measurement: here the whole chain is one compiled call.

## What is compared, and how honestly

Before any timing is recorded, the benchmark gates **density parity
deterministically**: the RK packed density and the linked Turing joint must
agree at exact points (`rtol = 1e-8`), and ProbProg's per-draw log densities
must match the native RK kernel at every retained draw (`atol = 1e-8`; the
measured gap is at floating-point roundoff). Sampler *trajectories* are
compared statistically only — the three harnesses use different adaptation
schedules and tree implementations, so exact-trajectory parity is not a
meaningful contract (and XLA's reduction ordering rules it out by
construction). Divergence counts are reported, never gated: the centered Eight
Schools posterior is a funnel, and its divergences are real sampler
diagnostics, comparable across harnesses because all three use the same
energy-error threshold of 1000 and depth cap of 10.

The MNIST sampling workload is the
[Wren-compatible PCA-40 profile](mnist-logistic.md) — the first 1000 training
images projected onto the top 40 unwhitened principal components, a 369-dim
packed posterior — not the 60000×784 full-resolution matrix workload, which
remains the evaluation-benchmark authority. The two workload identities are
kept explicitly distinct, as on the MNIST pages.

## Results

```@eval
Main.ReactiveKernelsDocs.render_probprog_mcmc_benchmark()
```

The MNIST × ProbProg cell is a **recorded compiler rejection at this pin**,
not an omission: `optimize = :probprog` tracing of the 369-dimensional packed
joint fails in the Enzyme-interpreter inference pass — a stack overflow, or an
unconverging grind, depending on inference-cache state — while the *same*
prepared kernel compiles primal and reverse through the plain Reactant AD path
in seconds (see the [MNIST Reactant page](mnist-reactant.md)). Because the
grinding mode has no interruption points, every ProbProg attempt runs in a
subprocess under a hard wall-clock budget (recorded in the receipt protocol);
exceeding it is the failure the cell records. The boundary is specific to the
ProbProg tracing path, and the native AdvancedHMC and Turing rows for MNIST
remain fully measured.

The compile / JIT warm-start column is the one-time cost of each harness:
XLA compilation of the whole sampling program for ProbProg, and a small
JIT-warming pre-run for the native harnesses. Treat every number as a
hardware- and version-specific receipt, not a universal performance claim —
and note that effective-sample-size per second on *one* model/dimension pair
does not generalize.

## The executed harnesses

The sampling harnesses and Turing twins the receipt executed are in
`benchmark/probprog_mcmc_comparison_body.jl` at the receipt's pinned
ReactiveKernels commit `958d9344234784a2380a84cea05e779364d43607`. The script
was removed together with ReactiveKernels' Reactant path.

The RK density entered each harness unmodified: `prepare` binds the data ports
at preparation (`bound = (; observations, observation_scales)` for Eight
Schools, `bound = (; X, y, num_classes)` for MNIST), so every harness samples
a single-argument packed log density. Constraint handling needs no sampler
support — the packed boundary already folds the `τ = exp(log τ)` transform and
its Jacobian into the density, which is also why the unconstrained spaces
match across harnesses.

## Provenance

The receipt was generated from a clean detached checkout of the pinned commit
above, in a fresh environment with Reactant pinned at 0.2.289 plus
AdvancedHMC, Turing and MCMCDiagnosticTools, with the publication protocol of
1000 warmup and 1000 retained draws. Each model's ProbProg attempt ran in a
subprocess under a hard wall-clock budget (recorded in the receipt protocol).
`benchmark/receipts/validate_probprog_mcmc.jl` still validates the checked-in
receipt.
