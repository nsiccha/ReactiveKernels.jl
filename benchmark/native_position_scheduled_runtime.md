# Opt-in native runtime scheduling receipt

Run the public synthetic recurrence and storage check with:

```sh
julia --threads=4 --project=. benchmark/native_position_scheduled_runtime.jl
```

This exercises `vectorize(...; schedule=NativeScheduling(workers=4,
chunk_size=8))`. Ordinary batching remains serial. The source graph, fixture
and independent mathematical reference come from the generic original
`native_position_scheduling.jl` probe. No private consumer code or data is
included. The original probe remains unchanged for its acceptance provenance.

On 2026-10-03, Linux x86-64, Julia 1.10.12, four execution threads, all 86
checks passed. Summary and full trajectories match serial results bit for bit
at 32, 129 and 512 positions and 13 and 64 recurrence steps. Worker output
width stays at most eight, and the retained native AST does not change with
these lengths. Signed zero, infinities and NaNs retain serial semantics.

The 512-position, 1024-step full-output case retained:

| Variant | Worker slots, bytes | Owned final result, bytes |
| --- | ---: | ---: |
| Ordinary serial | 0 | 4,198,440 |
| Original whole-range probe | 4,231,944 | 4,198,440 |
| Opt-in runtime, chunk size 8 | 295,944 | 4,198,440 |

The runtime retains 93% less worker-slot storage than the original probe.
These counts are `Base.summarysize` of worker caches and the final result;
they are not peak RSS or a process memory-budget proof. The final result
still contains every trajectory. No streaming reduction is implemented.

Seven warmed operation samples alternated order on a shared host. Each
scheduled/probe operation included a fresh `copy(template)` and its first
complete call. These times are exploratory, without a production speedup claim:

| Variant | Median, seconds | Minimum | Maximum | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| Ordinary serial | 0.002405041 | 0.001738273 | 0.002725609 | 4,240,752 |
| Original whole-range probe | 0.001656834 | 0.001105496 | 0.044599996 | 8,483,200 |
| Opt-in runtime, chunk size 8 | 0.001094946 | 0.000845367 | 0.017047233 | 4,587,536 |

The observed maxima show substantial shared-host variation. This receipt does
not calibrate worker or chunk hints and does not replace whole-operation
measurement in the first consumer. Private acceptance of the original probe
is separate evidence; it has not been repeated for this runtime API.

`test/test_native_scheduling.jl` covers owned and borrowed lifetimes, nested
outputs, output-to-input alias detachment, independent copies, read-only dense
lane dispatch, task failure recovery, lazy branches and replicated scalar AD.
`test/test_position_batching_reactant.jl` checks compiled parity and a retained
position loop whose operation sequence stays fixed between 3 and 23 positions.
