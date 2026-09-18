# ReactiveKernelsReactantODESolvers

RK-native, explicitly adaptive ODE solvers that lower through Reactant.

## Scope

This is a self-contained example subpackage. It owns an explicit adaptive
Tsit5 implementation whose adaptive control is written in Reactant-traceable
form: fixed-shape buffers, traceable control flow, no scalar indexing into
traced arrays, no host-side branching on tensor values, and no
`runtime_activity`/priming workaround. Ordinary reverse-mode gradients
(`AutoEnzyme(mode=Enzyme.Reverse)`) through the solve are part of acceptance.

Nothing here is PosteriorDB support until the solver itself is independently
proven and separately reviewed.

## Milestones

1. Scaffold (this package skeleton, registration, smoke test).
2. Native explicit adaptive Tsit5, validated against OrdinaryDiffEq references
   on Lotka–Volterra, Van der Pol with non-small μ, and at least one larger
   non-toy nonstiff system.
3. Reactant lowering of the adaptive solve plus ordinary reverse gradients.
4. Refactor so the tableau/stage/update structure is explicit and can be
   driven from the repository's standard RK-kernel formulation.
5. Parent review package; no landing without a fresh exact-tip user GO.

## Kernel coverage

The Tsit5 stage block — seven stage evaluations, propagator row, embedded
error estimate — is expressed once as a standard `@kernel` graph
(`src/kernels.jl`, `tsit5_stage`) over explicit ports, and that graph drives
both hot paths; there is no separate step implementation beside it:

- Native (`solve_ode` via `tsit5_step`): the `lower`ed plan body evaluated
  functionally, so plain-Enzyme reverse works through the solve. Prepared
  kernels cannot serve here: the stateful executor breaks Enzyme
  (`IllegalTypeAnalysisException` on its `Union{Missing,Bool}` restart
  bookkeeping, plus mutation discipline), and the contract forbids working
  around it.
- Traced (the Reactant ext): the prepared kernel itself, called per step.
  Prepared calls lower through Reactant — primal and reverse — via the core
  traced-slot machinery.

`test_kernels.jl` proves the two executors bit-for-bit identical across RHS
shapes, parameter shapes, and float types. The adaptive loop
(accept/reject, PI control, guards) is driver-level code in both paths:
kernels express straight-line dataflow, not data-dependent control.

One structural cost, measured honestly: the traced loop always runs
`maxiters` iterations (per-iteration freeze instead of early exit — the only
loop shape whose checkpointed reverse Enzyme lowers). Values are
bitwise-identical to early exit; execution wall time scales with the bound
(~0.28 µs per frozen iteration on a 2-state problem: 0.20 ms at
maxiters=150 vs 0.72 ms at maxiters=2000, strato2 2026-09-18);
status/exhaustion semantics are unchanged.

## Out of scope until separately authorized

- Stiff/BDF solvers.
- Discontinuous events/callbacks.
- Replacing OrdinaryDiffEq delivery for the four PosteriorDB adaptive-ODE models.
- Core API changes beyond this additive subpackage.

## Layout

```text
ReactiveKernelsReactantODESolvers
├── Project.toml   # deps: ReactiveKernels + LinearAlgebra; test-only reference/AD deps in extras
├── src/
│   ├── ReactiveKernelsReactantODESolvers.jl
│   ├── tableau.jl     # Tsit5 coefficients + dense-output coefficients
│   ├── controller.jl  # error norm/estimate, PI factors, initial step
│   ├── step.jl        # validated native entry to the functional step
│   ├── kernels.jl     # tsit5_stage graph + functional/prepared executors
│   ├── dense.jl       # dense output + saveat emission
│   ├── solve.jl       # native adaptive driver
│   └── reactant.jl    # traced config (driver lives in ext/)
├── ext/
│   └── ReactiveKernelsReactantODESolversReactantExt.jl  # traced driver
└── test/
    ├── runtests.jl  # problems, reference, controller, kernels, agreement,
    │                # guards, enzyme, reactant
    └── test_*.jl
```

On Julia 1.10, materialize the local paths explicitly (see `packages/README.md`):

```sh
julia --startup-file=no --project=packages packages/setup.jl
julia --startup-file=no --project=packages -e 'using Pkg; Pkg.test("ReactiveKernelsReactantODESolvers"; coverage=false)'
```
