# Registered and dogfood Enzyme benchmark runs

Benchmarks that time Enzyme gradients run twice: once with registered Enzyme
and once with the host's local Enzyme `dogfood` source. Both runs use the same
benchmark body, inputs, arguments, Julia command and host settings.

| Variant | Enzyme |
|---|---|
| `registered` | The suite's own registry selection, such as its exact `Enzyme v0.13.199` pin. This is the baseline. |
| `dogfood` | The same environment with `Enzyme` and its in-tree `lib/EnzymeCore` developed from one exact local commit. |

The dogfood-versus-registered ratio is a change within one implementation. No
variant is chosen by speed, and a variant that fails still appears in the
results. Mooncake is not part of these comparisons.

## Running

An applicable wrapper runs both variants by default, with unchanged arguments:

```sh
julia --startup-file=no benchmark/eight_schools_ad_comparison.jl \
  --output=<dir>/eight-schools-ad.toml
```

A script that runs inside a project environment goes through the generic
entry point, which builds each variant's environment from that project:

```sh
julia --startup-file=no benchmark/enzyme_variants.jl \
  --project=benchmark/distributions benchmark/distribution_gradients.jl \
  --output=<dir>/distribution-gradient.toml
```

Wrappers that require a clean, detached candidate still do; see
`_repro_guard.jl`.

Each run writes the following files:

- Registered receipt: the requested `--output=` path, in the suite's
  existing format.
- Dogfood receipt: `<stem>.enzyme-dogfood.toml` beside it, in the same format.
- Matrix: `<stem>.enzyme-variants.toml` beside them, with the schema
  `reactivekernels.enzyme-variants/v1`.
- Run directory: the run's logs, plus each variant's `Project.toml` and
  `Manifest.toml`. It defaults to
  `~/.local/state/reactivekernels-benchmarks/enzyme-variants/<suite>/<UTC time>Z-<RK SHA>/`.

A suite that only prints its results (`nuts_comparison.jl`,
`gradient_probe.jl`) keeps them in each variant's `body.log`. Its matrix has
no paired measurements.

Variant outputs are first written to the run directory and copied to the
requested destination after both runs. A file written by the first run
therefore cannot mark the source tree dirty for the second.

### Settings

| Environment variable | Meaning |
|---|---|
| `RK_ENZYME_VARIANTS` | Comma-separated subset of `registered,dogfood`. The default runs both. `registered` alone reproduces a historical single-environment receipt. |
| `RK_ENZYME_DOGFOOD_MIRROR` | Local `Enzyme.jl` Git mirror. The default is the KB host mirror, `~/.local/state/kb-agents/git-mirrors/Enzyme.jl.git`. |
| `RK_ENZYME_DOGFOOD_REVISION` | A ref or full SHA in that mirror. The default is `refs/kb-fetched/local/dogfood`. Set it to a full SHA to replay a recorded run. |
| `RK_ENZYME_DOGFOOD_CHECKOUTS` | Directory of immutable per-SHA checkouts. Each must be clean and at its SHA. |
| `RK_ENZYME_VARIANTS_DIR` | Run directory. |

The process fails when the baseline fails: registered, or dogfood when it runs
alone. A dogfood failure beside a completed registered baseline is recorded and
printed, and the process exits successfully.

## Matrix contents

`variants.<name>` records the following for each variant:

- `status`: `completed`, `failed`, `environment_failed` or `not_requested`.
- On failure, `error`, the exit code or signal, and the tail of the log.
- Setup and precompile seconds.
- The manifest's SHA-256 and the resolved dependencies.
- The exact body command, with the variant's paths made symbolic. Its
  configuration variables are listed by value; the rest of the environment is
  recorded only as a hash.
- A probe of the environment: Julia version, CPU target, CPU model, Julia and
  BLAS threads, bounds checking, and the Enzyme, EnzymeCore and Enzyme_jll
  sources.

The dogfood variant also records the mirror, ref, full `source_revision`,
`source_dirty = false` and the source's Enzyme and EnzymeCore versions.

`comparison` holds the cross-variant results:

- `status`:
  - `complete`: both variants completed.
  - `incomplete`: a variant failed or was not requested.
  - `invalid`: the variants ran under different conditions. This covers a
    different Julia version, CPU, thread count or bounds-checking mode, a
    different body command, or a benchmark source change during the run.
- `dependency_differences`: every package whose version or source differs
  between the two manifests, not only the Enzyme packages. For example, the
  dogfood source also moves Enzyme_jll and GPUCompiler.
- `measurements`: present only for a `complete` comparison with two receipts.
  It pairs every numeric timing or allocation field present in both receipts,
  and gives `dogfood_over_registered` where the registered value is nonzero.
  A failed variant has no ratio.

The benchmark source revision and its tracked-diff hash are recorded once at
the top of the file.

## Suite inventory

### Covered

These suites time native Enzyme gradients and build their own environment, or
run in one:

| Entry point | Timed Enzyme work | Registered Enzyme |
|---|---|---|
| `eight_schools_ad_comparison.jl` | RK, DI and Turing value-and-gradient and pullback | `0.13.199` |
| `sum_to_zero_comparison.jl` | RK, DI and Turing value-and-gradient | `0.13.199` |
| `mnist_logistic_ad_comparison.jl` (both datasets) | RK, DI and Turing value-and-gradient | `0.13.199` |
| `nuts_comparison.jl` | NUTS sampling, every sampler using DI+Enzyme gradients | `0.13.199` |
| `practicalbayes_comparison.jl` | PracticalBayes and RK gradients | `0.13.199` |
| `gradient_probe.jl` | RK DI+Enzyme gradient invocation paths | `0.13.199` |
| `distribution_gradients.jl` (via `enzyme_variants.jl`, `--project=benchmark/distributions`) | prepared RK reverse gradients for the distribution corpus | `=0.13.199` in the committed project |

The covered wrappers also add ReactiveKernels' exact source dependencies
(`deps/source_dependencies.jl`). On Julia 1.10, which ignores `[sources]`,
ReactiveKernels does not resolve without them.

The caller-environment drivers `inner_plate_partial_evaluation.jl`,
`rkppl_retained_lkj.jl` and `compiler_metadata_slice.jl` do not own an
environment. Pass the consumer project they document to `enzyme_variants.jl`
with `--project=<dir>`.

### Not covered

| Entry points | Reason |
|---|---|
| `eight_schools_primal_comparison.jl`, `mnist_logistic_comparison.jl`, `partial_evaluation_comparison.jl`, `distributions_comparison.jl`, `scalar_distribution_gallery_comparison.jl`, `structured_distributions_comparison.jl` | Primal only; Enzyme is not in the timed region. |
| `nuts_microbench.jl`, `nuts_microbench_ca9.jl`, `nuts_reactant_comparison.jl`, `cggibbs/`, `gibbs_poc/` | Analytic gradients; no Enzyme. |
| `fair_posteriordb_{b3,glm,gp,linear,mh,more}.jl`, `faithful_posteriordb.jl`, `faithful_staging.jl`, `all_sides_gaussian_regression.jl`, `all80_posteriordb.jl` and the `all80-env` scripts | They install and time a Mooncake comparator side, so running either variant would also run Mooncake. They become eligible once that side is removed. |
| `*_reactant_*_comparison.jl`, `eval_throughput_comparison.jl`, `probprog_mcmc_comparison.jl`, `sampler_transpiler/`, `reactant_hmc_loop_*.jl`, `brm_hsgp/` | They depend on Reactant; Reactant suites are not part of this rollout. |
| `repro_*`, `probe_*`, `*_mwe.jl`, `joint_stan_tiled/`, `native_bdf/` | Reproducers and feasibility checks, not timing suites. |
| `*_gate*.jl`, `*_fixture*.jl`, `*_corpus.jl`, `test_*_fixture.jl`, `verify_rename.jl` | Acceptance gates and fixtures; no timed comparison. |
| `native_position_*`, `benchmark_nested_construction.jl`, `borrowed_batch_instances.jl`, `dose_outer_superposition/`, `reactant_handoff/`, `pathfinder_oracle/`, `walnuts_oracle/` | No Enzyme. |

Historical receipts in `receipts/` stay unchanged as records of their
single-environment runs.
