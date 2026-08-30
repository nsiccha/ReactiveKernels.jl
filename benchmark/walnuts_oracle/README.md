# WALNUTS-D upstream macro-step oracle

This harness compiles against the exact released Walnutpie header rather than
against ReactiveKernels or a Julia transcription.  It exercises four control
shapes: base-grid acceptance, dyadic refinement plus reverse checks, rejection
by a coarser reverse grid, and exhaustion of every allowed forward grid.

From an exact checkout of Walnutpie revision
`4f051db7df57762a58ac851b0274fe57de342198` and an exact checkout of Eigen
5.0.1 at `bc3b39870ecb690a623a3f49149a358b95c5781d` from the vetted
cross-forge mirror `https://github.com/eigen-mirror/eigen.git`:

```sh
julia --startup-file=no benchmark/walnuts_oracle/capture.jl \
  /path/to/walnutpie /path/to/eigen-mirror-at-bc3b398
```

The capture rejects either tracked-dirty checkout and prints both exact source
revisions into the receipt. A matching `5.0.1` tag name alone is not accepted.

The committed receipt is
`benchmark/receipts/walnuts-upstream-macro-v1.tsv`.  `logp_grad_calls` is an
ordered-control observable because every leapfrog micro step performs exactly
one target evaluation in the pinned source.  Thus 10 calls in the dyadic case
means `1+2+4` forward attempts followed by `2+1` reverse checks; the four-call
rejection means `1+2` forward followed by the coarser one-step reverse grid.
The test also locks every numeric endpoint and the base-grid adaptation value.
