# JAX runtime and derivatives

This guide continues [JAX source](jax.md): runtime checks on accelerators, chains sampled as one batch, derivative rules in emitted modules and stable rewrites. [JAX ODE solves](jax-ode.md) covers the ODE bridges.

## Accelerator runtime checks

Ordinary runtime checks reduce the active lanes' predicates on the numerical
device. If a check fails, `jax.debug.callback` reports its error on the host.
A module that uses `checkify` for numerical failures checks the accumulated
error on the host at the function boundary. Set `JAX_PLATFORMS=cuda,cpu` for
CUDA execution so JAX has a CPU device whenever it runs a callback.

An explicit platform list makes CUDA initialization mandatory and CUDA the
default; it does not silently fall back to CPU if CUDA initialization fails.
For a GPU-required consumer, select `gpu = jax.devices("gpu")[0]`, place
numerical inputs with `jax.device_put(value, gpu)`, and verify result and
gradient arrays with `array.devices() == {gpu}`. Evaluate inside
`with jax.default_device(gpu):` so newly created constants use that device too.
Numerical function results, including constant terms and numeric leaves of
tuple/dict results, are JAX arrays with the emitted dtype. `prepare_data`,
`dimension`, `parameters` and module metadata remain host values.
Call `jax.block_until_ready(result)` and `jax.effects_barrier()` when checking
completion and callback errors. Executed callbacks incur transfer and
synchronization costs. `prepare_data` also runs on the host.

See JAX's [callback semantics](https://docs.jax.dev/en/latest/external-callbacks.html)
and [platform configuration](https://docs.jax.dev/en/latest/config_options.html#platforms).
`python benchmark/jax_runtime_checks.py` verifies the helper's values,
gradients, device placement, invalid-input rejection and missing-GPU failure
with a CUDA-enabled JAX installation.

## Selected station or subject densities

For station or subject minibatches, use `jax_source(built, bound, observations;
by, partitioned, global_factors, unit_factors, expected_layout)`. Write
`out.source` and send `out.partition_data` with the model data. In Python,
`q = model.prepare_partition(p, partition_data)` takes the result of
`p = model.prepare_data(**raw)`. `model.global_log_density(u, **q)` sums the
shared events. `model.unit_log_density(u, label, **q)` sums that label's
observations and assigned prior and Jacobian events at its own full packed
point. Both support `jax.jit`, `jax.vmap` and reverse differentiation.
A complete assignment at one point reconstructs `model.log_density(u, **p)`.
An unsupported indexed primitive raises during tracing.

A selected read evaluates the requested rows of each event, so it never
reaches a runtime check, which has no output. A check that reads only data,
such as a cell's length, has the same outcome at every point:
`prepare_partition` runs it once over all cells and raises if it fails. A
check that reads the parameters, such as the pivots of a factor, runs over
the whole event at every selected read of that event, as `log_density` runs
every check at each evaluation. A check under a condition on the
parameters runs where that condition takes its arm.

Selected rows read coupled values whole: each selected matrix of a batch of
Cholesky factors, the whole axis of a cumulative sum, and the entire output
of a loop or a scatter. These costs do not grow with the number of units
when the values are shared, such as an LKJ factor or the coherence factors
of an HSGP basis. A check, loop or scatter inside a unit's own cell is
evaluated for every cell of the event at each read. A read computes each
distinct entry it requests once, so many rows that read one shared factor
compute it once, and an entry it does not request is never computed. The
derivative of a selected read keeps only the selected rows' residuals.

## Plate layouts

A plate over units with ragged rows (subjects with their own event
schedules) is one `jax.vmap` of its cell, so every unit is stored and computed
at the longest unit's length. When the lengths differ widely, most of that
work, and of what reverse mode stores for each step of a scan, is padding.

`jax_source(...; data, plate_layout = :groups)` emits a module whose caller
chooses how such plates run, after `prepare_data` and without regenerating it:

```python
prepared = module.prepare_data(**raw)
chosen = module.prepare_plate_layout(prepared, groups=4)
value, gradient = jax.value_and_grad(module.kernel)(u, **chosen)
```

`prepare_plate_layout(prepared, groups, sequential=False)` sorts each plate's
units by length and splits them into at most `groups` groups that minimize
the padded size. Each group runs as one `jax.vmap` over rows cut to its
longest unit, or with `sequential=True` as one `jax.lax.map`, its units one
after another. The groups' values are joined in the plate's order and padded
to the plate's own shapes. `groups=1` without `sequential` is the plate as
without the option, and `prepare_data`'s own result selects it.

Every layout calls the plate's own cell, so values and ordinary derivatives
are the plate's. The layouts differ in cost: more groups move fewer bytes and
store less per chain, and run more loop rounds in sequence. Which is fastest
depends on the device, the model and the number of chains, so time the
candidates there rather than choosing by rule.

`select_plate_layout(prepared, make_step, chains=None)` runs that timing. The
caller supplies `make_step(prepared)`, or `make_step(prepared, count)` with
`chains`, returning a function that runs its own work once: one batched
gradient, or one sampler step over all chains, so the sampler's memory and
cost are measured too. Each candidate is compiled, run and timed; with
`chains` they are compared by time per chain. A candidate the device has no
memory for is recorded and its larger chain counts are skipped. The result is
the prepared data with the fastest layout and a report of every run:

```python
def make_step(data, count):
    gradient = jax.jit(jax.vmap(jax.value_and_grad(lambda u: module.kernel(u, **data))))
    positions = starts[:count]
    return lambda: gradient(positions)

chosen, report = module.select_plate_layout(prepared, make_step, chains=(256, 1024, 4096))
```

`GROUPED_PLATES` names the plates that take a layout, each with the prepared
row lengths that group its units. A plate can group numeric rows with
prepared lengths, including computed rows that inherit the lengths of
`rows[i]` from a prepared index. A cell that reads `rows[i]` directly
can also use the selected group's width. When a mapped numeric row or a
recorded row read has unknown lengths, the plate keeps its full widths so
that nested operations over paired rows see compatible axes.
A jitted function can take the prepared data as arguments. The member
indices and the permutation that restores the output order are array inputs.
Reordering units with the same group shapes reuses the compiled function.
Each selected group contributes one plate body to the trace, up to the
caller's group limit.

## Chains under `vmap`

When a run-time branch calls the matrix exponential on a traced matrix, the
exported density tests whether any chain or plate cell selects it. If every lane skips it,
`jax.vmap(jax.value_and_grad(density))` skips both the exponential and its
reverse pass. In a mixed batch, JAX evaluates the exponential in a batch,
using zero matrices for inactive lanes. The original branch selects each
lane's result. Reverse AD can recompute the exponential from its input matrix.

A sampler that advances several chains as one batch, `jax.vmap` over its
transition (BlackJAX's NUTS, for example), runs each of its while loops until
the last chain is done. JAX runs a batched loop's body for every lane and keeps
the old carry where a lane's loop has ended, so a chain that is done is still
evaluated on every later step of the batch, and the result is discarded. In
NUTS such a chain stays at the state where it stopped: after a divergence,
every remaining leapfrog step of the batch evaluates the density one step past
the divergent point, and a chain that is done can even keep an inner loop
running after every live chain has left it.

This costs nothing extra when a density costs the same everywhere. It does
when the cost depends on the position, as an ODE solve's does
([JAX ODE solves](jax-ode.md)): past a divergence the system can be stiff, and
one lane's solve then holds every call of the batch. In a four-chain warm-up on
CUDA of a density with a five-state solve in each of 24 plate cells (float64,
tolerances `1e-8`, at most 100,000 steps per output), one chain diverged after
5 of 127 leapfrog steps. Its remaining evaluations took ten times the solver
attempts of each running chain's, and the transition took 25 s where the
others took about 4 s. At a saved proposal whose solves exhaust the step budget,
one value and gradient takes 16.7 s on the card and 0.11 s on one CPU core.

Guard the sampler, not the model: give a chain that is done a state it has
already evaluated. In BlackJAX 1.7.1 that is two changes in
`blackjax.mcmc.trajectory`. In `dynamic_progressive_integration`'s loop body,
integrate from the subtree's initial state instead of
`trajectory.rightmost_state` when the lane had already diverged, terminated or
taken its steps. In `dynamic_multiplicative_expansion`'s, start a finished
lane's subtree at the transition's initial state with zero steps. Only
discarded results change. Compiled to round alike, the guarded and unguarded
samplers gave bitwise identical draws on Neal's funnel, with 420 instead of
2,694 evaluations far out in it (`|v| > 6`).

## Derivative rules

For `scalar_derivative_rule`, the module generates a `jax.custom_jvp` adapter
from the rule's mathematical graph. It selects the primal and named partials
for the active arguments, then applies the scalar chain rule. An inactive
argument's partial is not contracted with a numeric zero. Calls share their
generated helpers, including a joint recipe that serves several activity
patterns. Host data preparation uses the primal cut.

General `derivative_rule` calls generate adapters from the authored forward
branch and activity-selected staged reverse branches. Ordinary `jax.grad` and
`jax.value_and_grad` compute the joint primal and residuals once, then contract
the selected reverse cut. JAX does not differentiate the primal's convergence
loop in place of the supplied derivative. Forward mode uses the authored
forward branch; a missing branch or differentiation of a declared data-only
input raises. Batching maps whole calls and retains each lane's lazy branches.
These adapters use `jax.experimental.hijax.HiPrim` and are tested with JAX
0.12.0.1. Only modules containing a general rule import that API.

Ragged array ports carry their logical lengths through primal, forward and
reverse cuts. Lengths have zero tangent; padded cotangents do not contribute
to upstream gradients. Runtime checks inside a rule execute only for active
calls, including rules reached through a lazy branch or an empty plate.
Rules containing array regions retain their source plans as build-time
metadata. Native invocation reads only the compiled body and its concrete
execution tables.

The incomplete-beta continued fraction used by distribution tails supplies
its value and shape/argument partials through such a shared graph. A first
JAX gradient retains those partials rather than a reverse tape of all 300
fraction iterations. The primal and partials use fixed loops, so their
higher derivatives remain available through ordinary JAX differentiation.

## Stable rewrites

Unless `simplify = false`, each emitted recipe computes numerically fragile compositions in the stable form native preparation computes, together with the statements that compute its inputs, also from a lazy arm or a plate's cell: `log(exp(u))` is emitted as `u`, and `log1p(-p)` of `p = logistic(u)` reads `u`, so the value and `jax.grad` stay finite where `exp(u)` overflows. A law's term on a value computed outside the law (`Normal(0, s)`'s `log(s)` of `s = exp(a)`) is still translated as written: the law's endpoint is translated from its compiled lowered code, which the rewrites do not read yet.
