# Censored Student-t JAX gradients

This reproduces the three-parameter toy and seeds supplied with snag
`jax-student-t-cd-035aeb56`. It compares the candidate with ordinary JAX differentiation
of the **same value-only fraction and current reflection logic**. It isolates
the generated derivative rule's effect. The optional original artifact bundle
also compares the reporter's emitted module on the identical data and point.
The real two-site model was not supplied.

Use a Julia environment containing this checkout's ReactiveKernels,
ReactiveKernelsDistributionKernels, and ReactiveKernelsPPL, and a Python with
JAX and NumPy. Run under the host's normal compute budget:

```sh
julia --project=ENV benchmark/jax_student_t/emit.jl RESULTS
python benchmark/jax_student_t/measure.py RESULTS
python benchmark/jax_student_t/trace.py RESULTS
python benchmark/jax_student_t/check.py RESULTS/measurements.json
```

For the original-module comparison, pass `--original ARTIFACT_DIRECTORY` to
`measure.py`. The directory must contain the reporter's
`out/jax/toy_t_{100,1000}.py` and `out/runs/toy_t_<N>.request.json` files.
The driver checks exact equality of every input and the parameter layout
before comparing values and gradients. The supplied bundle is recorded in
the snag as SHA-256
`022abcf53f36f4cc940ac28beddbb21816fea627baca926986c001bcec2c6350`.

The emitter writes baseline/candidate modules and their identical inputs for
100 and 1,000 rows, prints native values and source inventories, and keeps the
ordinary-AD override confined to its benchmark process. The Python driver
enables float64, checks values against native Julia and all 16 gradients
against the baseline and optional original, then records warmed, synchronized
minimum and median timings, compilation times, and XLA temporary memory in
`measurements.json`. It uses the reporter's five-call/0.3-second minimum
sampling duration and 16 seeded points. The same file records Python/JAX
versions, device, thread settings, and host load.

The trace driver records loop/branch inventories for both data sizes and
single/batched gradients. It rejects a reverse loop in the candidate's
first-order gradient. The durable regression tests additionally cover all
activity masks, forward mode, Hessians, and absence of a 300-step tape axis.

The optional timing check implements the report's requested ratio regression:
gradient/value at most 8 and batch-per-point/single at most 1.5, at both sizes.
These are workload-specific guardrails with margin for shared-host noise,
not semantic constraints or universal performance guarantees. Run it on a
representative CPU measurement; both bounds can be set explicitly. The saved
candidate run passes, while `--tag original` and `--tag baseline` fail.

## Measured comparison

[Saved measurements](measurements-2026-11-12-strato2.json) were collected on
2026-11-12 on strato2 (8 logical CPUs, 4 cores), with the run limited to one
CPU, Python 3.14.4, JAX 0.12.0.1, float64, and the thread settings above.
The host was shared: load averaged about 8 during measurement. These are
minimum warmed times from the original and candidate modules measured in
the same run, rather than a comparison against the reporter's earlier host.
All authored data, the parameter layout, native values, and all 16 gradients
matched, including the tail's derivative in `nu`.

| Rows | Module | Value (ms) | Gradient (ms) | Batch of 16, per point (ms) |
| --- | --- | ---: | ---: | ---: |
| 100 | Original | 0.0691 | 0.4501 | 1.6697 |
| 100 | Candidate | 0.0464 | 0.1752 | 0.1064 |
| 1,000 | Original | 0.1622 | 1.6083 | 12.2553 |
| 1,000 | Candidate | 0.1087 | 0.5449 | 0.5283 |

The candidate gradient costs 3.78–5.01 times its value; batching costs
0.61–0.97 times the single gradient per point. XLA's temporary memory for
the batch falls from 39.7 MB to 0.163 MB at 100 rows and from 281.4 MB to
1.635 MB at 1,000 rows. This is compiler-reported temporary storage, not
whole-process peak memory.

The candidate emits 1,062 lines, two fraction loops and 58 conditions at
either size; the original emits 874 lines, four loops and 58 conditions.
The extra helper/adapter text is shared, and the entire candidate source is
identical across the two data sizes. Its first-gradient traces contain no
reverse fraction loop, both singly and under the outer batch. The ordinary-AD
diagnostic baseline retains a reverse loop. These measurements cover the
supplied toy; they do not establish a result for the unavailable two-site
model or for a GPU.
