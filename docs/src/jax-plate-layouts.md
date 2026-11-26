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
This example times independent chain gradients at each requested chain count.
Supply at least that many rows of `starts`.

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
streams that advance in a few fixed-width segments. If the stream lengths
would require more than eight segments, a two-segment schedule selects
subjects by index. Each subject advances in its original order.
Every true event is visited once and the longest subject sets the number of
rounds. Empty subjects keep their original initial-state semantics. A scan
with observable checks keeps its serial execution order. Compare the
resulting module against the other layouts at the chain counts of interest.
