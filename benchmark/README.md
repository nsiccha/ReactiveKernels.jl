# Benchmarks and research

These programs investigate compiler behavior and evaluation cost. They are not
run by the documentation build. Historical measurements describe the revision,
backend and workload recorded with them; rerun a driver before making a claim
about a current release.

| Area | Entry point |
| --- | --- |
| Distribution kernels | [Setup and workloads](distributions/README.md) |
| Native differential equations | [BDF experiments](native_bdf/README.md) |
| Gaussian-process consumer | [Public motorcycle-data comparison](brm_hsgp/README.md) |
| Gibbs experiments | [Prototype](gibbs_poc/README.md) |
| Weak-component mixture inference | [Frozen SMC schedules and mode mass](smc_weak_component/README.md) |
| Documentation history | [Research separation](docs-history/README.md) |

Run a selected driver in the environment documented beside it. Write newly
generated receipts and figures to a scratch or output directory rather than
committing regenerable dumps. The small retained `receipts/` collection is an
input fixture for `test/test_benchmark_smoke.jl`, including its receipt chains.
It is not a current performance catalogue. Historical unreferenced dumps were
removed during the documentation rewrite.

Data must be public with a stated source or generated synthetically by the
program. Keep private application models and data out of this repository.
Recorded host descriptions use “a Linux fleet host”; environment-specific
absolute paths are redacted in historical fixtures.
