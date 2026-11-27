# JAX plate layouts

A plate over units with ragged rows (subjects with their own event
schedules) is one `jax.vmap` of its cell, so every unit is stored and computed
at the longest unit's length. When the lengths differ widely, most of that
work, and of what reverse mode stores for each step of a scan, is padding.

An ordinary `plate_layout = :padded` export that schedules guarded table reads
across subjects also supplies a choice of walks. After preparing the data, call
`chosen, timings = module.select_plate_layout(prepared, make_step, chains=[64])`.
Use `make_step(chosen, 64)()` to repeat the winning batch. The builder
`make_step(data, count)` returns a zero-argument gradient or sampler step.
The first timing row names the winning `variant`. Each successful row gives
the chain count, the blocked warm calls in `samples_seconds`, and their
median in `median_seconds`. The `"padded"` walk gathers records before scanning each
subject up to the common row width. The `"events"` walk schedules reads at real
events wherever the scan supports it. `prepare_data` uses the latter by default.
To repeat a particular walk, call
`module.prepare_plate_layout(prepared, subject_walk="padded")` or pass
`subject_walk="events"`. Neither call times the model. The selected data can
be passed directly to a jitted function as one dictionary argument.

`jax_source(...; data, plate_layout = :groups)` emits a module whose caller
chooses how such plates run. After `prepared = module.prepare_data(**raw)`,
use `chosen = module.prepare_plate_layout(prepared, groups=4)` and pass
`**chosen` to `jax.value_and_grad(module.kernel)`.

`prepare_plate_layout(prepared, groups, sequential=False)` sorts each plate's
units by length and splits them into at most `groups` groups that minimize
the padded size. Each group runs as one `jax.vmap` over rows cut to its
longest unit, or with `sequential=True` as one `jax.lax.map`, its units one
after another. The groups' values are joined in the plate's order and padded
to the plate's own shapes. `groups=1` without `sequential` is the plate as
without the option, and `prepare_data`'s own result selects it.

Each layout calls the same cell. Grouping can reduce padded work and storage
per chain. The groups run in sequence, so their scan lengths add to the
number of rounds. Time the candidates on the target device at the chain
counts the sampler will use.

For batched gradients, create `gradient = module.bind_gradient(module.log_density,
prepared, device=device, memory_budget_bytes=budget)` and call
`values, grads = gradient(positions)`. A scalar kernel can supply
`module.kernel` in place of the density. Device and budget are optional.
The binding snapshots the prepared data and supplies its arrays to the
executable as device inputs. Its first call at a device, shape
and dtype tries the supplied layout and only tries alternatives if memory
prevents execution. If `select_plate_layout` has already measured this
workload on the same device at this width, pass its returned prepared data
to the binder. Later batches reuse the executable with new positions.

For a sampler that traces a scalar log density, use the module's
`select_plate_layout` when available. The callable from `bind_gradient`
runs a compiled batched gradient that JAX cannot trace inside the sampler. Call
`chosen, rows = module.select_plate_layout(prepared, make_step, chains=[n])`
at the fit's chain count `n`. The `make_step` in `tune_gradient` below
shows how to time a batched gradient. To time a sampler transition, have
`make_step` run the same transition with each candidate.

Pass `chosen` as an argument to the outermost compiled sampler step.
For example, a jitted `step(state, data)` can define its unary callback
as `lambda q: module.log_density(q, **data)` inside the step. Call
`step(state, chosen)`, or `step(state, prepared)` for a fixed layout.
Here `data` already contains traced inputs when JAX traces the callback.
If the sampler owns the compilation, its interface must carry the data
through that boundary. A callback created outside the step that captures
concrete arrays can embed their values in the program.

JAX flattens the dictionary and its registered layout objects as a pytree.
Observations, schedule indices and permutations are array inputs.
The tree records structure such as an absent schedule, group widths and
prepared `StaticInt` values. Reuse across datasets requires matching
prepared shapes, dtypes and static metadata on the same compilation target.
For example, a change that moves observations between censoring groups can
change their prepared shapes even when the raw observation count stays fixed.

Call `rows = gradient.calibrate(positions, repeats=3)`. The first row names the
installed layout, or the list is empty for a fixed module.
Calibration compares median warm latencies at this signature. Layouts whose
interquartile ranges overlap the fastest layout's enter 12 synchronized rounds
with it in rotating order. The lowest confirmation median wins. To replace
a current finalist, a challenger must also win at least 10 paired rounds.
The current layout is initially the supplied partition, including custom ones,
then the layout installed for this signature.

Each row marks the incumbent with `current` and records
`samples_seconds` and `median_seconds`. Read extra calls in round order from
`confirmation_samples_seconds` and their median from `confirmation_median_seconds`.
The winning row gives its `selection_reason`. `calibration_seconds` includes
confirmation attempts.

Make a new binding when the data or budget
changes. The byte budget covers XLA’s reported device buffer allocation
for the executable. Allow separate headroom for other live arrays and
allocator overhead. A backend must supply a buffer estimate to run with a
budget.

`select_plate_layout(prepared, make_step, chains=None)` runs that timing. The
caller supplies `make_step(prepared)`, or `make_step(prepared, count)` with
`chains`. The builder returns a zero-argument function for one batched
gradient or one sampler step over all chains. Timings and memory failures
describe the work that function runs. For a jitted gradient, budget one compilation or
cache load per distinct candidate and chain count tried, one call to warm
it up, and `repeats` timed calls (three by default). Gradient calibration
does this at the width of `positions`. With `chains`, the selector compares
layouts by time per chain. If a candidate runs out of memory, the selector
records the failure and skips its larger chain counts. The result is
the prepared data with the fastest layout and a report of every run.

Modules with eligible guarded reads expose `prepare_guarded_reads`.
`prepare_data` defaults to `"compact"`, using the largest match count in
any row as each read's capacity. When guards and indices come directly from
padded data and the scan and table extents agree, compact mode batches valid
requests across subjects. Other eligible compact reads use tiles.

Pass `**module.prepare_guarded_reads(prepared, "dense")` to the numerical
function to gather full producer tables. Either mode applies to all eligible
reads and can be chosen again with new data. If compact and dense capacities
differ, the tuner tries both, including on padded exports. The winner's
`guarded_reads` field names the mode to replay. Pass `guarded_reads="compact"`
or `guarded_reads="dense"` to `select_plate_layout` to hold the mode fixed.

The example also compares separately exported sampler modules in a mapping
keyed by layout name. Supply identical raw inputs and at least as many
starting positions as the largest batch. `tune_sampler` returns the module,
prepared data, chain count and timing table. It ranks each measured width by
the median gradient-call time multiplied by the projected calls to finish
warmup and sampling. Its inputs are warmup gradients per chain, the total
draw count and mean gradients per draw. An application can pass its own
`calls(count)` to `select_plate_layout` to reflect its adaptation schedule.
A grouped export with no ragged plates still measures each candidate chain
count once.

Each row also gives two elapsed wall times in seconds. `setup_seconds` starts
just before the tuner calls the builder and ends when the first result is
ready. It includes any eager compilation or cache loading the builder does.
`calibration_seconds` covers that setup and all timed repetitions for the
row. A memory failure ends its interval at the failed attempt. The tuner
ranks successful rows by warm performance.

```@example jax_plate_runtime
using ReactiveKernels, Markdown # hide
path = joinpath(pkgdir(ReactiveKernels), "examples", "jax_plate_layout.py") # hide
Markdown.parse("```python\n" * read(path, String) * "\n```") # hide
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


`plate_layout = :rounds` packs independent subjects’ recurrent events into
at most two fixed-width scans. The first can run one more subject per round than the
second. A subject that crosses between lanes carries its state across the
boundary and advances in its original order.
Every true event is visited once and the longest subject sets the number of
rounds. Empty subjects keep their original initial-state semantics. A scan
with observable checks keeps its serial execution order. Compare the
resulting module against the other layouts at the chain counts of interest.
