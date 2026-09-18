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
│   └── ReactiveKernelsReactantODESolvers.jl
└── test/
    └── runtests.jl
```

On Julia 1.10, materialize the local paths explicitly (see `packages/README.md`):

```sh
julia --startup-file=no --project=packages packages/setup.jl
julia --startup-file=no --project=packages -e 'using Pkg; Pkg.test("ReactiveKernelsReactantODESolvers"; coverage=false)'
```
