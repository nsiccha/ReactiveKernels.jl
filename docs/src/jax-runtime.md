# JAX runtime and derivatives

This guide continues [JAX source](jax.md): runtime checks on accelerators, derivative rules in emitted modules, stable rewrites, and ODE solves.

## Accelerator runtime checks

Data-dependent checks use `jax.debug.callback`: their predicates are computed
on the numerical device and checked on the host. JAX needs a local CPU device
for these callbacks, even for valid inputs on CUDA. Start the Python process
with `JAX_PLATFORMS=cuda,cpu`; `JAX_PLATFORMS=cuda` alone fails when a runtime
check executes. CPU-only execution remains supported.

An explicit platform list makes CUDA initialization mandatory and CUDA the
default; it does not silently fall back to CPU if CUDA initialization fails.
For a GPU-required consumer, select `gpu = jax.devices("gpu")[0]`, place
numerical inputs with `jax.device_put(value, gpu)`, and verify result and
gradient arrays with `array.devices() == {gpu}`. Evaluate inside
`with jax.default_device(gpu):` so newly created constants use that device too.
Numerical function results, including constant terms and numeric leaves of
tuple/dict results, are JAX arrays with the emitted dtype. `prepare_data`,
`dimension`, `parameters` and module metadata remain host values.
The host validates the check predicates; it does not evaluate the numerical kernel. Call
`jax.block_until_ready(result)` and `jax.effects_barrier()` when checking
completion and callback errors. Host callbacks incur transfer and
synchronization costs; `prepare_data` also runs on the host.

See JAX's [callback semantics](https://docs.jax.dev/en/latest/external-callbacks.html)
and [platform configuration](https://docs.jax.dev/en/latest/config_options.html#platforms).
`python benchmark/jax_runtime_checks.py` verifies the helper's values,
gradients, device placement, invalid-input rejection and missing-GPU failure
with a CUDA-enabled JAX installation.

## Derivative rules

For `scalar_derivative_rule`, the module generates a `jax.custom_jvp` adapter
from the rule's mathematical graph. It selects the primal and named partials
for the active arguments, then applies the scalar chain rule. An inactive
argument's partial is not contracted with a numeric zero. Calls share their
generated helpers, including a joint recipe that serves several activity
patterns. Host data preparation uses the primal cut. General `derivative_rule`
calls currently emit their primal cut for ordinary JAX differentiation.

The incomplete-beta continued fraction used by distribution tails supplies
its value and shape/argument partials through such a shared graph. A first
JAX gradient retains those partials rather than a reverse tape of all 300
fraction iterations. The primal and partials use fixed loops, so their
higher derivatives remain available through ordinary JAX differentiation.

## Stable rewrites

Unless `simplify = false`, each emitted recipe computes numerically fragile compositions in the stable form native preparation computes, together with the statements that compute its inputs, also from a lazy arm or a plate's cell: `log(exp(u))` is emitted as `u`, and `log1p(-p)` of `p = logistic(u)` reads `u`, so the value and `jax.grad` stay finite where `exp(u)` overflows. A law's term on a value computed outside the law (`Normal(0, s)`'s `log(s)` of `s = exp(a)`) is still translated as written: the law's endpoint is translated from its compiled lowered code, which the rewrites do not read yet.

## ODE solves

A call of the native ODE bridges `rk_ode_rk45_tol(f, y0, t0, ts, reltol, abstol, maxsteps, args...)` or `rk_ode_bdf_tol` with the same arguments, also inside a plate cell and through a helper such as BayesianRegressionModels' `ode_rk45`, is one [diffrax](https://docs.kidger.site/diffrax/) solve. A module that solves an ODE imports `diffrax` (`pip install diffrax`); other modules do not. Emitting needs none of the Julia solver packages the native bridges load. The right-hand side `f(t, y, args...)` is translated like any helper into a nested Python function, and the extra arguments that are runtime values reach it through diffrax's `args`.

- `rk_ode_rk45_tol` is `diffrax.Dopri5()`, the Dormand–Prince 5(4) method of the native `DP5()`. `rk_ode_bdf_tol` is `diffrax.Kvaerno5()`, diffrax's stiff (L-stable implicit Runge–Kutta) method: diffrax has no BDF method, so the stiff solve uses a different method than the native `FBDF()`, with the same tolerances.
- The caller's tolerances are passed unchanged and may be traced (data or parameters). The step size follows a PID controller; outputs come from the solver's dense interpolant, so the native and JAX solves agree to the tolerances, not to the last bit.
- `maxsteps` limits the accepted steps between consecutive requested outputs, as natively, and may be traced too. diffrax's own static bound on attempted steps is the largest int32, which costs nothing: a gradient keeps 64 checkpoints whatever the bound.
- `jax.grad` differentiates the discrete solve (diffrax's `RecursiveCheckpointAdjoint`, 64 checkpoints a solve), including with respect to the initial and output times, so a gradient's accuracy follows the caller's tolerances, as the native forward sensitivities' does. A backward-solve adjoint such as `jax.experimental.ode.odeint`'s does not: on a seven-state pharmacokinetic model its gradient was about 35 times less accurate at the same tolerances.
- A ragged row of output times solves to the row's last time; its padded rows repeat that time's state, a column `sol[:, j]` keeps the padding, and reductions mask it.
- Invalid inputs (non-finite or non-increasing times, invalid tolerances or `maxsteps`), an exceeded per-output budget and a failed solve raise at run time through the module's runtime checks, as the native bridge throws (a failed solve or exceeded budget natively as `ReactiveKernels.ODESolveFailure`). A row without output times raises nothing. A module's log density therefore raises at a failed solve, where ReactiveKernelsPPL's native sampler targets answer `-Inf`.
