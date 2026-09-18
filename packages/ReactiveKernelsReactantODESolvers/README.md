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

## Kernel coverage (milestone 4)

The Tsit5 stage block — seven stage evaluations, propagator row, embedded
error estimate — is expressed as a standard `@kernel` graph (`src/kernels.jl`,
`tsit5_stage`) over explicit ports, and `test_kernels.jl` proves it
bit-for-bit against the plain step across RHS shapes, parameter shapes, and
float types. That is as far as current lowering goes; the rest is honest
fallback, not RK-native execution:

- The native and traced hot paths both execute the plain functional step
  (`tsit5_step`). Routing native execution through the prepared kernel breaks
  plain-Enzyme reverse through the solve (the executor's
  `Union{Missing,Bool}` restart bookkeeping fails Enzyme type analysis, and
  the contract forbids working around it); the prepared executor is in-place
  (0-alloc) and therefore untraceable by Reactant in any case.
- The adaptive loop (accept/reject, PI control, guards) is driver-level code
  in both paths: kernels express straight-line dataflow, not data-dependent
  control.

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
│   ├── step.jl        # plain functional Tsit5 step (both hot paths)
│   ├── kernels.jl     # standard-@kernel mirror of the step + boundary note
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
