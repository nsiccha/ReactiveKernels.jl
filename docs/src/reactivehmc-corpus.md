# ReactiveHMC kernel corpus

```@eval
Main.ReactiveKernelsDocs.render_review_status(:frozen_sampling)
```

This page is the transparent source index for the ReactiveHMC-derived compiler
corpus. It starts from ReactiveHMC.jl revision
`ca9ea4ca41924bb0e1fadc01c717e1333916aba6`, whose nine source-file digests are
locked in
[`benchmark/reactivehmc_algorithm_corpus.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/reactivehmc_algorithm_corpus.jl).
Each rendered kernel below is read from its real package or benchmark authority
during the docs build. The build also loads that file, checks its captured
MethodIR, and requires exactly one source-locked user interaction per example.
Supported prepared/compiler paths execute natively; fixtures whose compiler
acceptance is still held stop visibly at construction, MethodIR, and independent
receipt inspection.

The corpus separates three kinds of evidence:

- The pinned upstream file and line range define the mathematical authority.
- The RK source file is the authored compiler input; it is never copied into
  this Markdown page.
- Independent receipts establish expected values and control behavior outside
  the candidate compiler. A terminal `native_and_reactant` field is the corpus
  acceptance contract, not permission to infer an unrecorded backend result.

## Complete algorithm inventory

The cards are generated from the corpus tuple at build time. They include all 17
logical algorithms: relativistic kinetic energy; six phase-point variants;
leapfrog, generalized leapfrog, implicit midpoint, and multistep integration;
fixed-step HMC and NUTS; dual averaging and Welford adaptation; and trajectory
and sampling statistics.

```@eval
Main.ReactiveKernelsDocs.render_reactivehmc_inventory()
```

## Six executable phase-point variants

The six panels come from three parameterized `@kernel spec` definitions in the
nested compatibility package: Euclidean, Riemannian, and SoftAbs geometry, each
with Gaussian and relativistic kinetic energy. Every panel executes an actual
`PreparedKernel`, derives the Generated kernel from its `code_expr`, and
renders that same kernel's selected `Plan` as the Compute DAG.

Open **Raw input** for the exact source extracted from
[`packages/ReactiveKernelsCompatibilityExamples/src/preexisting_reactivehmc.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsCompatibilityExamples/src/preexisting_reactivehmc.jl).
The two kinetic variants deliberately share authored source but have distinct
captured callable/scalar authorities and executable plans. Each Raw input pane
also contains the exact constructor, `prepare`, and call expression that produced
its displayed output.

```@eval
Main.ReactiveKernelsDocs.render_reactivehmc_phasepoints()
```

## Relativistic kinetic energy

ReactiveHMC's relativistic kinetic-energy construction is captured as an object
kernel with nested energy, density, CDF, and quantile methods. The Lambert-W
implementation remains an explicit callable port, so ReactiveKernels does not
gain a hidden package dependency or an algorithm-name special case. The
interaction shown below compiles the loaded fixture, calls its quantile method,
and checks the observed value against the independent receipt.

```@eval
Main.ReactiveKernelsDocs.render_reactivehmc_captured_sources(:rke)
```

## Generalized and implicit integrators

These are the source-faithful nonseparable integrators. Their fixed-point loops
and ordered phase-point writes remain ordinary authored control; `multistep`
stays an ordinary higher-order Julia wrapper because it owns no reactive state.
The compiler-capture block is generated from the loaded kernel's MethodIR, not
from a documentation-only description. The displayed interactions compile and
run each transition against its pinned receipt case.

```@eval
Main.ReactiveKernelsDocs.render_reactivehmc_captured_sources(:integrators)
```

## Trajectory and sampling statistics

One fixed-capacity state kernel covers both trajectory recording and
per-transition sampling summaries. Capacity exhaustion is explicit and sticky;
direction-dependent prepend/append order, reveal indices, acceptance reduction,
and detached history copies are all visible in the source. This replaces dynamic
container growth with an explicit compiler boundary while preserving the pinned
callback order. The displayed interaction resets the compiled state, records the
receipt's trajectory and sample, then reads back the resulting statistics.

```@eval
Main.ReactiveKernelsDocs.render_reactivehmc_captured_sources(:statistics)
```

## Fixed-step HMC: accepted source and IR boundary

The fixed-step HMC fixture and its independent control/RNG receipt are accepted.
The source below exposes momentum refresh, ordered integration, statistics before
divergence exit, and the final log-Bernoulli copy decision. Its compiler-capture
pane is the accepted MethodIR/oracle evidence. Its displayed interaction is
deliberately limited to fixture construction, MethodIR, and receipt-control
inspection; it does not run the fixed-HMC kernel.

This is intentionally **not** a claim that the later generic callable/RNG
functional compiler implementation is accepted: that implementation remains
outside this published boundary until its separate semantic review has a GO
revision.

```@eval
Main.ReactiveKernelsDocs.render_reactivehmc_captured_sources(:hmc)
```

## Full samplers and external integrations

The larger kernels retain dedicated pages so their complete source and
independent evidence remain readable:

- [NUTS sampling](nuts.md) build-loads the eight-kernel authoring fixture,
  including leapfrog, momentum refresh, diagnostics, NUTS state/entry, dual
  averaging, and Welford variance. It also compiles and executes a source-locked
  `nuts!!` interaction; its full reactive group remains the primary three-pane
  NUTS view.
- [WALNUTS-D](walnuts.md) shows fixed-macro-time dyadic refinement, reverse-grid
  rejection, the depth-10 NUTS leaf, explicit replay streams, and the complete
  authored source. Its current interaction is honest fixture/MethodIR/receipt
  inspection only, with no native or Reactant compiler-success claim.
- [Nutpie diagonal adaptation](nutpie-diagonal.md) executes its initialization
  and adaptation kernels against an independent Rust oracle.
- [Pathfinder approximation](pathfinder.md) executes two WANT cuts of one
  authored inverse-BFGS/local-Gaussian kernel against an independent Python
  oracle.

ReactiveKernels remains the compiler/planning library. These pages are external
compiler-acceptance examples, not new sampler APIs.
