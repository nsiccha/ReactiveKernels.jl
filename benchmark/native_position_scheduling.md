# Native position scheduling probe

`native_position_scheduling.jl` is a producer-only experiment, not a public
scheduling API or a production default. It uses a generic scalar recurrence
authored once as an ordinary `@kernel`. It compares the same prepared graph
under existing serial `vectorize` and a chunk scheduling wrapper.

The wrapper uses the existing graph dependency split to evaluate the shared
prefix once. It maps the residual over runtime position ranges with one
independent borrowed execution instance per worker, joins every task, and
copies chunk results into a fresh owning result. `copy(candidate)` shares
prepared computation and creates empty worker slots for another concurrent
caller. An instance itself is not reentrant. Runtime lengths and counts do
not generate graph bodies. This prototype uses internal RK interfaces and
supports native primal execution only; it establishes no threaded AD or
compiled scheduling contract.

Run from the package environment:

```sh
julia --startup-file=no --threads=4 --project=. benchmark/native_position_scheduling.jl
```

The final Julia 1.10.12 run on x86-64 Linux passed **181 assertions**. These
cover bitwise agreement with an independent scalar loop, agreement with serial
graph evaluation, empty/seed-only sequences, declared empty scalar outputs,
singleton and remainder position chunks, changed inputs and shapes, strict
threshold ties, NaNs/infinities/signed zeros, immutable inputs, independent
concurrent instances, output ownership across later calls, lazy inactive
indexing, joined task failures and recovery. The generated residual is reused
across increasing sequence lengths. Full untyped array outputs retain the
existing requirement for an output declaration to admit an empty position
batch; the probe does not alter that boundary.

For 512 positions and 1024 recurrence steps, five warmed samples per arm
recorded the following allocation and retained-slot measurements:

| Cut | Serial allocated bytes | Candidate allocated bytes | Candidate retained worker slots |
| --- | ---: | ---: | ---: |
| Two scalar summaries | 4,306,176 | 4,324,272 | 5,640 |
| Full trajectories | 4,240,128 | 4,246,656 | 4,231,944 |
| Cheap scalar square control | 4,224 | 8,656 | not measured |

Retained worker slots include cached chunk outputs and recyclable lanes;
they exclude read-only prepared metadata, inputs, the fresh owning result,
Julia/JIT memory and process RSS. Allocation is cumulative per warmed call,
not retained memory. The summary cut still materializes its intermediate
trajectory: no scan-extrema fusion is introduced.

Timing samples varied with shared-host load. The final run's minima were
2.521/1.906 ms for serial/candidate summaries, 3.051/2.817 ms for full outputs,
and 4.059/14.81 microseconds for the cheap control. Earlier samples differed
substantially. These are exploratory measurements, not exclusive timings or
a consumer speedup claim. The cheap control demonstrates that a position
count threshold alone is insufficient for a production scheduling policy.
The full-output cut demonstrates the storage cost of retaining chunk outputs
and then stitching an owning result.

Before promotion to a runtime API, the first real consumer must measure its
existing authored graph against its serial reference in private scope. That
acceptance must separate request time, cumulative allocation, retained storage
and task overhead, and preserve fractions/counts, ties, seeds, remainders,
changed inputs and request ownership. The producer design must then address
useful scheduling decisions and the extra full-output storage. The withdrawn
extrema experiment and deferred consumer-specific arithmetic optimizations
are outside this probe.
