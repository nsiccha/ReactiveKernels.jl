# JAX HMM forward investigation, 2026-11-12

The candidate combines plate validity checks across active lanes and uses a
rolling primal history buffer where the scan's reads have provably bounded
lag. On the reporter's 1,600-observation input, the warmed gradient median
falls from 300.83 ms to 5.30 ms. **This is a partial performance repair:** the
row-based reference takes 0.186 ms, and the candidate still has steep gradient
scaling. It does not implement a row-vector HMM rewrite or custom derivatives.

The change is general RK lowering. The authored HMM law, its pairwise
tie-exact log-add-exp, and its native Julia execution remain the authority.
Both the candidate and reference use ordinary `jax.grad`; native comparisons
use ordinary Enzyme reverse AD.

## Evidence and conditions

- Report: `ReactiveKernels/snags/jax-hmm-forward-809555a2`, ODLProject.
- Reporter module: RK `e727fe13`, RKPPL 0.1.154, supplied standalone Python
  and exact request JSON. Its three module files have SHA256
  `46385c0c242a3619be4a20c844a7d842af77f5132b6590d28fc96101df557d56`.
- Additional pre-change emission: RK
  `8c849da1c8e8786d24c4630476c90608d08fb571`, SHA256
  `b3d0559a5b6e781a643fa86a54553ad37445d1bc6e8dcb457ba91e8130cf6bd4`.
- Initial tested candidate: `dfdcdf295ad873f65794804a1f2bcb2377ad8040`, based on
  canonical `aab5993018cd0241487cdf0daaa379c2d9f49d5f`. The initial reviewed
  tip `28ff263e30ffcd95503c8b901f44965e4ce8959c` adds comments, this report,
  retained results and diagnostic reporting without changing production code.
  The later scan-bound integration is described below.
- Candidate source SHA256:
  `cd45ce43f582147986feaa4b0f4a8db90c27ddc2e9d08569fdc5943d015d0d20`.
  All four emissions at N = 4, 100, 400, 1600 are identical, 47,582 bytes.
- gordito CPU, one CPU of time, single-thread XLA, x64, JAX 0.12.0.1,
  Python 3.14.4, Julia 1.10.12-kb3, Enzyme 0.13.219, RKPPL 0.1.160.
  The host was shared and loaded; a token does not reserve a physical core.
- Each function was compiled and warmed, then called nine times with
  `block_until_ready` and an effects barrier. The retained results include
  minima, medians, temporary bytes, values, gradients and trace inventories.
  The reporter originally used minima; both statistics are given here.
- Baselines and candidate were timed in the same Python process on the
  same inputs. Their RK revisions differ as listed above; these are not an
  isolated benchmark of only this diff against `aab59930`.

Full observations: [jax_hmm_forward_results.json](jax_hmm_forward_results.json).
The immutable reporter bundle on the KB host is
`/home/niko/.local/state/kb-agents/uploads/fd2cc78bfd4d0fd7a2293ac3a20ebfb794ca0f6639e059cd9d8cc28e38c0315c.zip`.
Its README records the original producer commands and environment.

### Exact reporter inputs, milliseconds

| N | Original value median | Candidate value median | Original gradient median | Candidate gradient median | Row-reference gradient median |
|---:|---:|---:|---:|---:|---:|
| 100 | 27.228 | 0.258 | 22.410 | 0.301 | 0.178 |
| 400 | 72.229 | 0.634 | 84.194 | 1.626 | 0.110 |
| 1600 | 322.308 | 0.523 | 300.829 | 5.304 | 0.186 |

| N | Original gradient minimum | Candidate gradient minimum | Row-reference gradient minimum |
|---:|---:|---:|---:|
| 100 | 16.788 | 0.221 | 0.046 |
| 400 | 56.361 | 0.700 | 0.051 |
| 1600 | 236.382 | 5.087 | 0.124 |

The 1,600-row improvement is 56.7 times by medians and 46.5 times by minima.
The candidate is still 28.5 times the row reference by medians and 41.0
times by minima. Small timings and the N = 400 median are visibly noisy.

On the independently generated `sin.(1:N)` inputs, the 8c849da1 baseline
gradient medians are 18.766, 62.414 and 320.140 ms; the candidate's are
0.367, 0.636 and 6.425 ms. See the JSON for every value and gradient.

## What changed

The vector observation's extra unit-axis check was a `jax.debug.callback`
inside nested maps. There were 2N callback regions in the traced HMM primal:
200, 800 and 3,200. Boolean batching now reduces each check over active
lanes, including nested maps, before one callback. Empty domains run none;
branch and padding masks exclude inactive lanes. The predicate is Boolean,
so this batching rule does not author a numerical derivative.

History-index provenance proves a fixed lag or a lag bounded by a runtime
row width. The scan then carries a rolling buffer and stacks its scalar
outputs through `lax.scan`. Unknown indices, slices, and whole-history uses
retain the original full buffer. Static guards retain the fallback where
an index proof requires a sign or divisor condition. Seeds and unwritten
entries keep the authored values. Zero-step scans do not trace indexed
reads from their unused body.

The HMM primal's history carry shrinks from 2N entries to three entries.
This is **not** evidence of a quadratic-memory repair. XLA gradient temporary
memory was already linear: the reporter baseline uses 48,944 / 188,912 /
750,448 bytes; the candidate uses 51,272 / 197,192 / 782,664 bytes.

## Remaining gradient scaling

The candidate's ordinary reverse scan still carries gradients for the
complete emission and transition arrays: 2N and 4(N-1) entries. Those arrays
are shared by the scan and gathered under lazy branches. A diagnostic using
`lax.scan(..., _split_transpose=True)` left both the carry inventory and
temporary memory unchanged and did not improve the 1,600-row gradient
(5.34 ms versus 5.43 ms medians in that run). It is not part of the fix.

[jax_scan_shared_gradient.py](jax_scan_shared_gradient.py) isolates the
guarded shared-array pattern without RK or an HMM. Both programs compute
the same nonlinear scalar recurrence, preserve the initial-step branch,
and compare value and ordinary reverse gradients. Passing each entry as a
scan input removes the full-array gradient carry:

| N | Guarded shared gather, minimum ms | Iterated entries, minimum ms |
|---:|---:|---:|
| 100 | 0.0125 | 0.0125 |
| 400 | 0.0344 | 0.0161 |
| 1600 | 0.3542 | 0.0434 |
| 6400 | 4.8278 | 0.1483 |

This supports a follow-up on general scan-input extraction. It does not
prove that the HMM's entire residual cost has one cause, or that JAX cannot
optimize an unguarded shared-array gather: the unguarded diagnostic did not
show comparable scaling. The HMM still performs one scalar step per table
cell and a predecessor-state loop per cell, with their own costs.

A general fix must move eligible per-step gathers/work to iterated inputs
without evaluating untaken invalid accesses, duplicating effects, changing
the authored reduction/tie semantics, or unrolling data-sized axes. This
candidate leaves that work open. There is no model-specific replacement,
backend derivative rule, or unsupported compiler flag in the production path.

## Validation

A fresh Julia process on the tested commit ran these 14 JAX test files:
`source`, `checks`, `history`, `join`, `allocations`, `keep`, `empty_plate`,
`translations`, `elementwise`, `fma`, `columns`, `data_ranges`,
`padded_fields`, and `padded_loops`. All completed successfully; pre-existing
broken cases were not changed. New history coverage has 56 assertions;
four check-test drivers each run Python numerical, AD and effect assertions.

The selected existing PPL HMM testsets passed: `hmm_law` (4),
`hmm_unreached` (4), and one module across observation counts and state
widths (5). This was an HMM subset, not the complete PPL suite. Benchmark
native Enzyme gradients at N = 4, 100, 400 and 1600 also passed.

Coverage includes seeds, empty inputs, current/future fill values, lagged
and block reads, unknown-index and whole-history fallbacks, bound data,
mixed branches, ragged padding, nested maps, `grad(vmap)` and `vmap(grad)`,
active invalid checks and inactive invalid branches. Source equality and
trace-carry checks protect retained iteration without a machine-time test.
Numerical comparisons assert 1e-12 relative value tolerance and 1e-10
relative/absolute gradient tolerance against the reference and native AD.
The exact reporter requests are independently checked against the reference.

The acceptance driver ended with `[hmm-validation] exit=0`. The final
diagnostic benchmark and guarded shared-array reproducer each exited zero.

## Integration with prepared scan bounds

The recovery starts from canonical
`4d08ffd8497fdb795f962018867611d65c2d5dcf`, which retains prepared integer
bounds through scans. History translation passes its unique scan position
to that path and retains ordinary integer provenance alongside its lag proof.
For example, a scanned range element can both read `history[n - 1]` and bound
`for j in n:(n + 2)`. Merely adapting the helper's signature lost the loop
bound and rejected that program; the combined representation preserves both.
The regression covers empty and changed ranges, unchanged emitted source,
values and ordinary native/JAX gradients.

## Reproduction

Use an environment with this checkout, RKPPL, DistributionKernels,
DifferentiationInterface and Enzyme resolved. Under the site's compute-token
wrapper where required, run:

```sh
julia --startup-file=no --project="$RK_ENV" benchmark/jax_hmm_forward.jl "$HMM_OUTPUT"
kb-jax-python benchmark/jax_hmm_forward.py "$HMM_OUTPUT/hmm_4.py"
kb-jax-python benchmark/jax_hmm_forward.py "$HMM_OUTPUT/hmm_4.py" \
  --baseline "$REPORTER/out/jax/toy_hmm_100.py" \
  --requests "$REPORTER/out/runs" --sizes 100,400,1600
kb-jax-python benchmark/jax_scan_shared_gradient.py
```

Set `RK_JAX_PYTHON` to the JAX interpreter for Julia's JAX tests. The generated
module has no RK dependency in Python. The benchmark scripts set x64 and
single-thread XLA defaults; they report the effective JAX platform and x64
setting. Compilation is excluded from the measured calls.
