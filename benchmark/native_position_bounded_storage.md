# Bounded native position worker storage

`native_position_bounded_storage.jl` is a producer-only follow-up to the
[original scheduling probe](native_position_scheduling.md). It imports that
probe unchanged and uses its generic recurrence, shared-prefix split,
independent borrowed workers and owning-output helpers. No package API,
default, lowering or extension changes.

Each worker's assigned position range is traversed in bounded runtime chunks.
One initial chunk discovers output shapes; the wrapper allocates a fresh full
result and copies each completed chunk immediately into its disjoint region.
Every task is joined before the result escapes, including on failure. Shared
inputs and prefix values remain read-only; copied instances have independent
worker slots. The same instance is not reentrant.

This retains at most the configured chunk width per worker output instead of
each worker's full position range. It still allocates and stitches a fresh full
result, and still uses the original count-only scheduling threshold. Chunk
width eight is an experimental control, not a calibrated runtime default.
The initial chunk executes serially; larger chunks therefore delay the other
workers. Uneven tails can also change the borrowed slot shapes and cause
additional allocations. No scan fusion or arithmetic change is introduced.

Run from the package environment:

```sh
julia --startup-file=no --threads=4 --project=. benchmark/native_position_bounded_storage.jl
```

The accepted x86-64 Linux / Julia 1.10.12 / four-thread run passed **181 original
checks and 538 new checks**, with child and wrapper exit zero. New acceptance
covers full trajectories and scalar summaries, mixed shared/mapped outputs
with two array shapes, bitwise serial parity, changed inputs/shapes, immutable
inputs, output lifetime and nonaliasing, independent concurrent copied
instances, strict ties, special floats, empty scalar batches and seed-only
sequences. It distinguishes an initial inline failure from joined task
failures and verifies recovery. The existing generated residual stays identical
as runtime sequence lengths increase. Fixed chunk bounds hold at growing
position counts through 512 with remainder chunks.

Full untyped array outputs retain the original empty-position boundary. This
experiment establishes native primal behavior only; threaded AD and compiled
scheduling remain unestablished.

For 512 positions and 1,024 recurrence steps, seven warm samples per arm used
alternating forward/reverse orders. These are shared-host exploratory timings,
not exclusive performance evidence. The complete final summary is:

| Arm | Median [min–max], ms | Median allocated bytes | Retained worker slots, bytes | Owned result, bytes |
| --- | ---: | ---: | ---: | ---: |
| Owning serial | 2.349757 [1.818020–2.679935] | 4,240,752 | 0 | 4,198,440 |
| Original full chunk | 1.540031 [1.296593–20.168868] | 4,245,408 | 4,231,944 | 4,198,440 |
| Bounded width 1 | 1.149904 [1.125834–6.126926] | 4,802,352 | 66,344 | 4,198,440 |
| Bounded width 8 | 0.981215 [0.697356–1.045954] | 4,322,096 | 295,944 | 4,198,440 |
| Bounded width 32 | 1.106804 [0.771506–50.077821] | 4,270,640 | 1,083,144 | 4,198,440 |
| Bounded width 128 | 1.689610 [1.414852–2.021799] | 4,257,776 | 4,231,944 | 4,198,440 |

At width eight, worker slots shrink **93.0%**, and the sum of worker slots plus
one owning result falls from **8,430,384 B to 4,494,384 B**. Cumulative
allocation rises **1.81%** relative to the original full-chunk wrapper. A
one-position chunk saves more retained storage but increases loop/slice/copy
allocation. These results establish the storage tradeoff, not a universal
optimal chunk width or a consumer speedup.

Slot sizes are `Base.summarysize` of worker cache tuples after the calls;
the owned result is measured separately. They exclude read-only prepared
metadata, inputs, intermediate lifetimes, the measurement harness, Julia/JIT
memory and process RSS. Timed samples keep their returned values in the
harness; the table is not a measurement of whole-process retained memory or
peak RSS. Allocated bytes are cumulative per call. Owning serial has no
retained borrowed slots but may use transient scratch.

The exact accepted run window was `2026-10-03T01:26:39Z`–`01:26:57Z`; the
compact runner recorded `exit=0 elapsed=18s`. This Markdown records all final
sample summaries and correctness counts so the result survives the local
runner log. Production promotion still needs a scheduling policy that avoids
cheap work, controlled consumer verification of any changed implementation,
and a clear owning/borrowed result contract.
