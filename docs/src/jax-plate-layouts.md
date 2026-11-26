# JAX plate layouts

A plate over units with ragged rows (subjects with their own event
schedules) is one `jax.vmap` of its cell, so every unit is stored and computed
at the longest unit's length. When the lengths differ widely, most of that
work, and of what reverse mode stores for each step of a scan, is padding.

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

`select_plate_layout(prepared, make_step, chains=None)` runs that timing. The
caller supplies `make_step(prepared)`, or `make_step(prepared, count)` with
`chains`, returning a function that runs its own work once: one batched
gradient, or one sampler step over all chains, so the sampler's memory and
cost are measured too. Each candidate is compiled, run and timed; with
`chains` they are compared by time per chain. A candidate the device has no
memory for is recorded and its larger chain counts are skipped. The result is
the prepared data with the fastest layout and a report of every run.

When a prepared scan can gather a producer's full table or compute just its
requested rows, the module also emits `prepare_guarded_reads`. For example,
`dense = module.prepare_guarded_reads(prepared, "dense")` selects the full
table. Pass `**dense` to the numerical function. Use `"compact"` to request
the tiled computation. Both choices apply to every eligible read in that
module and can be made again after preparing different data. The tuner times
both when their prepared bounds differ, including on a padded export. The
winning row's `guarded_reads` field records the mode to replay.

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
