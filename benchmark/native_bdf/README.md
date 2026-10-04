# Native BDF implementation experiments

The user selected Julia's standard BDF implementation on 2026-10-04. The
generic numerical bridge now lives in `ext/ReactiveKernelsBDFExt.jl` as
`ReactiveKernels.rk_ode_bdf_tol`, delegating to `OrdinaryDiffEqBDF.FBDF()`.
Its public acceptance is `test/test_native_bdf.jl`. The CVODE experiments below
are preserved research; their solver-selection recommendation is superseded.
The compatibility signature does not require consumers to retain old helper
layouts or argument counts. Equivalent source binding may restructure those
details while preserving the scientific model, accuracy controls and full
application acceptance. Recorded dependency versions are qualification
evidence; supported upgrades require qualification, not permanently frozen pins.

These public experiments investigate snag `native-bdf-origi-82b23b10`.
They are research candidates, not installed package capabilities or an
application replacement. No method is added to `StanBlocks.ode_bdf_tol`.
The original application, its numerical controls, and its private source
remain the responsibility of the existing BRM/RKPPLBench lanes.

The exact uploaded public call fails with `MethodError` on Julia 1.10.12 and
StanBlocks 0.2.1 at `4d3f6bff78d3b7b3638b612e55ddb4951f9ff125`.
Reproduction bundle SHA256:
`c59c299b63fa792dc2d7aa52f6196860729011d3c1ddcf563921565320aea8ff`.
This is an absent native numerical/source bridge; StanBlocks documents its
token as Stan-only.

## Concrete implementation choices

Both experiments use Sundials CVODE BDF, caller tolerances, ordinary runtime
loops, fresh state/parameter buffers, and the library's Enzyme-backed
continuous adjoint. They author no backend derivative rules, attach no rule
to a foreign callable, and use no ForwardDiff sensitivity fallback.
Finite differences in `check_interval.jl` are independent test oracles for
the matrix-exponential control, never the gradient implementation.

1. **Continuous multistep history.** A single solve retains CVODE's history
   between output times. The owned ODE algorithm delegates to Sundials and
   supplies ODE initialization rather than the unsupported DAE initialization
   requested by SciMLSensitivity. Base values and ordinary Reverse pass.
   A per-output step counter works in the primal but the callback currently
   fails ordinary Reverse. The library reverse also returns zero for active
   output times. Completing this route requires keeping the counter inside
   the undifferentiated solver adapter and providing time sensitivities on an
   owned mathematical graph, through RK's existing generated-rule mechanism.
   No handwritten protocol adapter or consumer derivative declaration is an
   acceptable repair.

2. **Interval restart.** Each requested interval is solved with CVODE BDF,
   with its independent variable mapped to `[0, 1]`. The original times enter
   the RHS parameter vector, so the existing library adjoint differentiates
   initial and output times as well as initial states and RHS arguments.
   This passes the public scalar and stiff matrix/vector argument checks.
   It naturally resets the step budget at every output. It also restarts the
   multistep history at every output, unlike Stan's continuous integration;
   that is a material numerical/performance choice, requiring user direction
   before this prototype becomes production code.

At the time of these experiments, the first route was recommended because it
preserved more of the original integration behavior. The second was a smaller,
demonstrated increment requiring a choice about interval restarts. That
recommendation is historical; the USER selected standard Julia FBDF. Neither is
an analytical PK substitute or a switch to an explicit/non-BDF solver.

## Run

Use a scratch consumer environment with ReactiveKernels developed from the
reviewed worktree and these direct dependencies:

```toml
[deps]
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
SciMLSensitivity = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
StanBlocks = "2e771a56-c23a-4e0b-9282-20c2e37157e9"
Sundials = "c3572dad-4567-51f8-b174-8c6c989267f4"
```

The qualified receipts use Enzyme 0.13.209, SciMLBase 3.57.0,
SciMLSensitivity 7.119.12 and Sundials 6.7.1 / Sundials_jll 7.5.0.

```sh
julia --startup-file=no --project=<scratch-env> benchmark/native_bdf/check_interval.jl
julia --startup-file=no --project=<scratch-env> benchmark/native_bdf/check_continuous.jl
julia --startup-file=no --project=<scratch-env> benchmark/native_bdf/compare_paths.jl
```

`check_interval.jl` exits nonzero on any failed assertion. The continuous
script first checks its supported base case, then catches and explicitly
reports the two unresolved diagnostics. Its zero exit is **not** acceptance
of the callback/time diagnostics. `DIAGNOSTIC_FAIL` is load-bearing evidence.

The public checks establish feasibility and unresolved boundaries. They do
not establish equivalent application/source binding, private application numerical
or density parity, full input-type/exception parity, backend portability,
or the user's original-model performance target. Those remain acceptance
requirements for the implementation, not waived constraints.
