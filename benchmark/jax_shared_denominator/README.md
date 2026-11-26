# Shared denominators in the rkbound2 gradient

The archived QTcF log-density gradient has one complete volume-related division that can be reused across event rounds. The other volume divisions, the site rate-gap divisions and the observation divisions examined here have varying numerators. The checkpoint experiment reduced temporary storage and showed mixed timing results. We retained the existing policy for this workload.

## What repeats

The original optimized GPU program at 1,024 chains gives the following counts per chain. These are float64 elements at divide instructions, weighted by observed fusion executions over three warm gradient calls.

| Source | Divide elements | What the dataflow establishes |
| --- | ---: | --- |
| Event scan, module line 3291 | 20,776 | The two readouts divide changing states by one subject volume. Their transpose divides changing cotangents by that volume. |
| Differentiated scan, attributed to line 3180 | 5,194 | The forward loop takes the volume from tuple slot 17, squares it, and divides the constant one by the square. The loop returns the same slot. It writes one copy of the factor per subject per round for the reverse pass. |
| Site row, lines 1866–1868 | 15,741 | The rate gap is already broadcast from the chain and subject axes. The three readouts also depend on interval-specific quantities. The authored branches for positive duration and rate separation govern their evaluation. |
| Other observation divisions | 27,095 | The denominator repeats across observations for 27,088 elements. The census does not prove these complete quotients invariant. The separately investigated log-beta normalizer is excluded. |

For the inverse-square factor alone, retaining one division for each of 53 subjects would remove **5,141** of the **5,194** divisions across 98 rounds per chain. This is an opportunity in the differentiated program. The source line attributed by the compiler contains a different quotient, so attributing the inverse square requires following the fusion operands and the returned loop tuple. The site constants already contain authored, guarded reciprocals of a determinant and a rate difference at module lines 1508–1509.

## Recomputing the existing inverse square

The experiment changed the generated table-scan checkpoint policy to recompute JAX's existing `integer_pow(y=-2)` during reverse evaluation. It kept the authored divisions, guards and ordinary differentiation. At **all three widths**, the optimized forward loop lost one inverse-square divide site and the optimized reverse loop gained one. The two varying quotient sites in each loop remained. This experiment measures a change in what is saved for reverse evaluation. A distinct optimization would recognize a saved derivative factor whose operands are invariant, compute it once under the original demand conditions, and pass that value to the reverse scan. This experiment does not establish its timing.

| Chains | Original ms | Replay ms | Change in median | Original / replay temporary MiB |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 20.863 | 21.429 | +2.71% | 333.26 / 330.72 |
| 256 | 71.387 | 71.271 | −0.16% | 1,359.62 / 1,349.48 |
| 1,024 | 296.918 | 296.704 | −0.07% | 5,297.31 / 5,290.25 |

Measurements on 2026-11-26 used an exclusive NVIDIA GeForce RTX 5060 Ti, float64 and JAX `0.12.0.1.dev20261116+a67d713e1`. Each cell is one compilation and the median of 25 synchronized calls in a fresh process after warm-up. Replay ran first at 256 chains and second at the other widths. The interquartile timing ranges overlapped at both larger widths. Compilation took 124/123, 134/131 and 146/135 seconds respectively (original/replay). Temporary storage is the compiler's estimate. CUDA reported timer warnings during compilation. The table uses host elapsed time around synchronized execution.

All full-model values and all 3,888 gradient coordinates were finite. At each width the first `min(width, 256)` points had native Julia references. Across these points and both policies the largest absolute value difference was `5.46e-12` and the largest scaled gradient difference was `4.33e-11`, using `abs(actual - reference) / (1 + abs(reference))`. Comparing the two GPU policies directly across all points gave a maximum absolute value difference of `9.10e-13` and a maximum scaled gradient difference of `8.06e-13`.

## Arithmetic counterexamples

Run `JAX_PLATFORMS=cpu python edges.py` from this directory with float64-capable JAX. The measured version gave the following results.

- For `(x / d) * weight` with `x = 2^600`, `d = 2^400` and `weight = 2^450`, direct division and reciprocal multiplication both returned `2^650`. The denominator derivative was `-2^250` with division and negative infinity with reciprocal multiplication. Even a finite, exactly representable power-of-two reciprocal was insufficient with an active denominator.
- A 32-round scan with alternating `±2^903` and denominator `2^-60` returned zero with a zero denominator derivative. Moving the same guarded readout divisions after the scan into a vectorized map returned zero with a NaN derivative. This change exposes the sum of cotangents to a different reduction order.

## Repeat the full measurement

Use Bruno's archived `real_qtcf_rkbound2_w256` run, including its native reference files, and its `benchmarks/jax_emission` directory for input preparation and start generation. The measured module's SHA-256 is `c1872254d70e46e8a6d9714618a464b9cd7118b7f18d49f2f8bab2d9c59bf7da`. From this directory, with `RUN` and `DRIVER` pointing to those two directories, run the commands below on an exclusive CUDA device.

```sh
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
for width in 64 256 1024; do
    modes="original replay"
    if [ "$width" = 256 ]; then modes="replay original"; fi
    for mode in $modes; do
        python measure.py "$RUN" "$DRIVER" "$width" "$mode" --hlo "$mode-$width.hlo"
    done
done
```

The driver also accepts `--trace-only` to inspect the ordinary gradient graph on a CPU. The archive's 256 starts are used in order. For wider batches the Bruno driver repeats starts with seeded jitter on the additional chains.
