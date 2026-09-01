# WALNUTS-D as mathematical `@kernel` source

WALNUTS-D changes the numerical integrator used at each NUTS leaf. Instead of
committing to one leapfrog step size, it keeps a fixed macro time and tries
dyadic micro-step grids until the endpoint Hamiltonian error is small enough.
It then runs the accepted endpoint backward on every coarser grid: if a coarser
reverse grid would also pass, the leaf is rejected. Accepted leaves are combined
by the ordinary depth-10 multinomial NUTS recursion shown on the
[NUTS sampling](nuts.md) page.

This is an external compiler-acceptance candidate fixture, not a sampler API
exported by ReactiveKernels. The source authority is the released
[`walnuts.hpp` at `4f051db`](https://github.com/flatironinstitute/walnutpie/blob/4f051db7df57762a58ac851b0274fe57de342198/include/walnutpie/walnuts.hpp),
with Bob Carpenter's earlier
[`walnuts.py` at `895a9b7`](https://github.com/bob-carpenter/walnuts/blob/895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a/walnuts/walnuts.py)
retained as research lineage. The independent acceptance executable compiles
the pinned C++ header in a separate process; the Julia fixture and compiler do
not provide their own expected answers.

Every code block below is read as inert text at build time from
[`benchmark/walnuts_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/walnuts_kernel_authoring_fixture.jl).
The docs build validates textual source anchors so GitHub Pages cannot silently
publish a stale transcription, but it does not include the file, load its
definitions, or request captured MethodIR.

## State, bounds, and observable replay

The authored state fixes the depth-10 default, macro-step refinement bound, and
the independent observables a future accepted native/Reactant comparison must
use. All randomness is supplied as runtime data; stream positions and overflow
are part of the result rather than hidden host RNG state.

```@eval
Main.ReactiveKernelsDocs.render_walnuts_source(:state)
```

## Fixed macro time, dyadic micro grids

This is the numerical heart of WALNUTS-D. The same `macro_time` is integrated
with `m`, `2m`, `4m`, … micro steps. The first forward endpoint satisfying the
absolute Hamiltonian-error tolerance becomes the only candidate. Starting from
that endpoint with negated momentum, every coarser grid must fail; otherwise the
candidate is not reversible at a unique critical grid and the leaf is rejected.

```@eval
Main.ReactiveKernelsDocs.render_walnuts_source(:macro_step)
```

The compiler sees ordinary loops, branches, `copy!!`, broadcasts, the captured
builtin `abs(::Float64)`, and same-type integer `div`. Those calls are admitted
by the compiler's exact primitive-specialization rules, not by their names
alone. There is no WALNUTS operation, program-counter table, field list, or
handler in compiler code.

## The macro leaf inside depth-10 NUTS

The outer transition consumes direction and exponential streams only on the
source paths that need them. It grows one NUTS subtree per depth and preserves
the same multinomial proposal and U-turn structure as the existing mathematical
NUTS fixture.

```@eval
Main.ReactiveKernelsDocs.render_walnuts_source(:nuts_step)
```

At recursion depth one, `macro_step!` is the leaf integrator. A rejected macro
leaf stops the current trajectory; an accepted leaf records the Hamiltonian
difference, runs the diagnostics callback, and becomes a weighted proposal.
The WALNUTS tolerance (`max_error`) is deliberately distinct from the outer
NUTS divergence threshold (`min_dham`).

```@eval
Main.ReactiveKernelsDocs.render_walnuts_source(:nuts_leaf)
```

## Public fixture entry: stochastic values are explicit

The fixture entry receives momentum, directions, and exponentials as typed
runtime arguments. This makes identical pre-generated streams and exact counter
comparison part of the intended native/Reactant acceptance contract; it is not,
by itself, evidence that either generic compiler path is accepted.

```@eval
Main.ReactiveKernelsDocs.render_walnuts_source(:entry)
```

## Compiler status: not executed during the docs build

The docs build does not include, parse, lower, compile, or execute any part of
this fixture; it only reads the displayed source bytes above. The recursive
`step!` / `start!` / `finish!` strongly connected component sits at the active
compiler frontier: bounded recursive state-machine lowering is under
development and its contract is still moving, so a build-executed frontier
probe would pin the docs to a boundary that changes underneath them. This page
therefore intentionally does **not** execute `walnuts!!`, does not compile a
WALNUTS transition natively or with Reactant, and does not claim full
recursive-SCC or WALNUTS compiler support. The compiler-side boundary is owned
by the focused compiler tests, not by this page.

## Independent control receipt

The pinned C++ oracle covers four distinguishing paths:

- base-grid acceptance: `1` target evaluation;
- refinement through `1 + 2 + 4` forward steps and `2 + 1` reverse checks:
  `10` target evaluations;
- rejection because the coarser reverse grid passes: `1 + 2 + 1 = 4` target
  evaluations;
- failure of every allowed forward grid: `1 + 2 + 4 + 8 = 15` target
  evaluations.

Those counts and the exact endpoints live in
[`benchmark/receipts/walnuts-upstream-macro-v1.tsv`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/walnuts-upstream-macro-v1.tsv).
They are independent expected results, not output from either RK compiler.
The receipt also locks its build-only Eigen dependency to full commit
`bc3b39870ecb690a623a3f49149a358b95c5781d` from the vetted GitHub mirror;
the capture does not trust a matching version tag alone.

## Complete authored WALNUTS-D fixture

The complete mathematical source—including the shared Euclidean phase point,
leapfrog integrator, depth-10 tree recursion, dyadic macro step, diagnostics,
and explicit replay entry—is available here for review.

::: details Show the complete source consumed by the compiler

```@eval
Main.ReactiveKernelsDocs.render_walnuts_complete_source()
```

:::
