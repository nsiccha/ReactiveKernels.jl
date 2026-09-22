# ReactiveKernelsReactantODESolvers

RK-native, explicitly adaptive ODE solvers that lower through Reactant.

## Scope

This is a self-contained example subpackage. It owns an explicit adaptive
Tsit5 implementation whose adaptive control is written in Reactant-traceable
form: fixed-shape buffers, traceable control flow, no scalar indexing into
traced arrays, no host-side branching on tensor values, and no
`runtime_activity`/priming workaround. Gradients are delivered by a
backsolve-style adjoint (`compile_backsolve_gradient`): a second
early-exit compiled program re-solves the augmented `[u; λ; μ]` system
backward in reverse time. Direct differentiation through the adaptive
solve is not supported.

Nothing here is PosteriorDB support until the solver itself is independently
proven and separately reviewed.

## Milestones

1. Scaffold (this package skeleton, registration, smoke test).
2. Native explicit adaptive Tsit5, validated against OrdinaryDiffEq references
   on Lotka–Volterra, Van der Pol with non-small μ, and at least one larger
   non-toy nonstiff system.
3. Reactant lowering of the adaptive solve plus backsolve adjoint gradients.
4. Refactor so the tableau/stage/update structure is explicit and can be
   driven from the repository's standard RK-kernel formulation.
5. Parent review package; no landing without a fresh exact-tip user GO.

## Kernel coverage

The Tsit5 stage block — seven stage evaluations, propagator row, embedded
error estimate — is expressed once as a standard `@kernel` graph
(`src/kernels.jl`, `tsit5_stage`) over explicit ports, and that graph drives
both hot paths; there is no separate step implementation beside it:

- Native (`solve_ode` via `tsit5_step`): the `lower`ed plan body evaluated
  functionally (plain-Enzyme reverse works through the native solve, but
  the supported compiled gradient path is the backsolve adjoint).
  Prepared kernels cannot serve here: the stateful executor breaks Enzyme
  (`IllegalTypeAnalysisException` on its `Union{Missing,Bool}` restart
  bookkeeping, plus mutation discipline), and the contract forbids working
  around it.
- Traced (the Reactant ext): the prepared kernel itself, called per step.
  Prepared calls lower through Reactant via the core traced-slot
  machinery. The traced path differentiates only the loop-free RHS VJP
  inside the step (for the backsolve adjoint); through-solve reverse is
  not supported.

`test_kernels.jl` proves the two executors bit-for-bit identical across RHS
shapes, parameter shapes, and float types. The adaptive loop
(accept/reject, PI control, guards) is driver-level code in both paths:
kernels express straight-line dataflow, not data-dependent control.

Reverse behavior: gradients come from the backsolve adjoint
(`compile_backsolve_gradient`), not from differentiating through the
adaptive loop. Both directions run the early-exit primal: the forward
solve exits on `(n < maxiters) & (t < t1)`, and each backward segment
re-solves the augmented `[u; λ; μ]` system in reverse time with the same
early-exit shape. The only `Enzyme.autodiff` differentiates the loop-free
RHS once per stage evaluation (a vector-Jacobian product lowered to
straight-line code). Supported losses are `:endpoint` (one backward
segment `t1 → t0`) and `:saveat` (one segment per saveat interval, the
adjoint jumping by `1` at each saveat point). Because continuous
backsolve is not discretisation differentiation, saveat gradients agree
with references up to a tolerance-scaled bar, not bit-for-bit.

Known limitation: Enzyme reverse *through* the retained adaptive `while`
loop does not lower (Reactant's reverse-mode `while` handling needs a
statically known iteration count, which a data-dependent exit cannot
provide). Reactant/Enzyme-only reproducer:
`benchmark/repro_reactant_adaptive_while_reverse.jl` at the repository
root. The former workarounds — a fixed-N straight-line unroll of the
solver and post-exit dummy-`dt` masked iterations — were removed, since
both are shapes `docs/src/constraints.md` forbids.

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
│   ├── dense.jl       # dense output + vectorized saveat emission
│   ├── solve.jl       # native adaptive driver
│   └── reactant.jl    # traced config (driver lives in ext/)
├── ext/
│   └── ReactiveKernelsReactantODESolversReactantExt.jl  # traced driver
└── test/
    ├── runtests.jl  # problems, reference, controller, kernels, dense,
    │                # agreement, guards, fixedn, enzyme, reactant
    └── test_*.jl
```

On Julia 1.10, materialize the local paths explicitly (see `packages/README.md`):

```sh
julia --startup-file=no --project=packages packages/setup.jl
julia --startup-file=no --project=packages -e 'using Pkg; Pkg.test("ReactiveKernelsReactantODESolvers"; coverage=false)'
```
