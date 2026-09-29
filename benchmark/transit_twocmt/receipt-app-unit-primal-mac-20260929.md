# Actual application transit-unit primal acceptance, 2026-09-29

The existing ShinyRK `SIMULATION_GRAPH` unit-response cut and
`SIMULATION_UNITS_BATCH` pass on the CPU backend with the two transit runtime
files from `937d1d9fb4cc638dc326a1c71c2ad1aeac3c9ba4` applied to published
ReactiveKernels `b6cea65732aa27d849eea89077b6b9cd8b196955`.
This is application unit-response acceptance, not full simulation, dose
superposition, likelihood, posterior, or compiled reverse acceptance.

## Exact execution

- Application source: `79c80150a4619f6a6b26210a2bb3ecc601bf8950`.
- Mac job: `f61dd966fbaa4e2a`, terminal exit 0 at 2026-09-29 20:39:34 UTC.
- Julia 1.10.11, four Julia threads, CPU backend on ARM64.
- Reactant 0.2.289 / Core 0.1.23 / Reactant_jll 0.0.413+0;
  Enzyme 0.13.205 / Enzyme_jll 0.0.294+0.
- Selected `libReactantExtra` SHA256:
  `d42844109d0cc17c391c5dfcac4b1040bb14b1e49a944a6f2b3bc49caf2872f2`.
- Full log SHA256:
  `254005f348e5eef274883fa8852b60583effa6cd6e6a9d49ff0694c5c20ec591`
  (10,454 bytes; KB `/mac_jobs/log?job=f61dd966fbaa4e2a`).
- Output archive SHA256:
  `379fef65f7ec5e555346bebe861678aff15a860cc62aaf2a824dbea771e5ba79`
  (111,412 bytes; KB `/mac_jobs/outputs?job=f61dd966fbaa4e2a`).

Source guards verify both original and replacement Git blobs and SHA256 hashes,
and the complete runtime-tree delta contains exactly these two files:

| Path under `packages/ReactiveKernelsPPL/src` | Accepted SHA256 |
| --- | --- |
| `transit_twocmt.jl` | `9b8f4a78575718ea7b486ef14e8381e22f3c8e21d3b5645d7aba9bb4501d006d` |
| `transit_twocmt_rule.jl` | `8c8f82a80ef6eb360447cfeb295bda5a2d155e3cbbb285b64de90cb7a8d0150e` |

## Numerical and ownership checks

The runner's 139 native checks pass: 84 graph/reference checks, 36 host dose
validation checks and 19 bound-schedule sweep checks. Strict isolated precompile
passes. The same authored application graph supplies the native scalar
`want=:units` reference and position-batched candidate; no second simulator is
substituted.

For both lattice (97 lags) and exact (247 lags) schedules, three actual PK
positions and a changed set of live positions are evaluated. The driver checks
that the changed positions alter the transit parameters and responses, while
shared inputs remain equal. Native batched outputs agree with the native scalar
cut, and default-optimized compiled outputs agree with native values for the
first and changed inputs (`rtol=2e-11`, `atol=2e-10`). Each executable is reused;
the first returned device output remains unchanged after the second call, and
both input arrays remain unchanged. All three case logs end in PASS.

## Retained-loop evidence

The audit independently counted the archived HLO, including every StableHLO
operation rather than only `while`:

| Schedule | Lags | Preoptimization lines / bytes | Default lines / bytes | While count in each |
| --- | ---: | ---: | ---: | ---: |
| lattice | 97 | 1,946 / 192,295 | 735 / 56,720 | 15 |
| exact | 247 | 1,946 / 204,714 | 735 / 62,930 | 15 |

Within each optimization phase the complete StableHLO operation histograms are
identical across these two lag sizes. Shapes and constant payloads differ, so
HLO byte counts differ. These observations support retained iteration in this
unit cut; they are not a full-app performance measurement or evidence of
shape-independent compilation cost.

## Environment and limits

The isolated environment adds the 21 compiler dependencies recorded in
`resolver-delta.txt` and selects the candidate RK monorepo packages. Existing
non-RK package versions and source trees are preserved; ADTypes/QuadGK weakdep
serialization is normalized. The production Project and Manifest are unchanged
before/after (SHA256 `c020297c11cef9d2d90d7348c800043fbdad2ee820f688970a3b7b3020fc1b79`
and `d9171d7054ce21be9891ff44f0e4423f682d85f7945d61f2a71a8aec1f15894c`).
The isolated Manifest SHA256 is
`219dbf17d8e2b56eeedacebdc65ffe6a44348b2c5473bd9af232100ca84c8413`.

The audit reviewed the exact driver, complete log, case logs, source/lock
receipts and HLO artifacts and verified both artifact hashes. It did not rerun
the Mac job. The earlier standalone component receipt supplies empty-lag,
changed-lag, lazy-regime, explicit-accuracy and ordinary native reverse checks.
This application gate changes no production pin and does not resolve the
separate embedded concentration-superposition failure.
