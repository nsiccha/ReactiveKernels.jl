# Public BRM documentation artifact reproductions

`probe.jl` exercises the four existing public models reported by BRM's
documentation builder. It retains their original data, priors and helpers,
and checks model construction, retained artifact emission, RKPPL translation,
kernel construction and JAX source emission separately. It does not fit any
model. JAX source emission alone does not establish numerical or AD parity.

The reproduction source is BayesianRegressionModels commit
`c0158b4fc44faf03bc490955293a78e602bff07c` (version 0.2.278). The source tree
must match the BRM package loaded by the consumer environment. The probe reads
the mesh documentation's original Julia block, the station script's model-only
prefix, and the complete public OU and wastewater model sources. It writes
the actual Julia expressions with their ordinary printers.

Use a consumer environment resolved with this ReactiveKernels worktree,
ReactiveKernelsPPL, the pinned BRM source, and StanBlocks. Run through the
usual compute-token wrapper on the KB fleet:

```sh
KB_COMPACT_KEEP_LOG=1 kb-acquire-compute-token --here \
  --why 'Reproduce public BRM retained documentation models' -- \
  kb-run-compact julia --startup-file=no --project=/path/to/consumer-env \
  benchmark/brm_docs_retained/probe.jl /path/to/pinned/BRM /path/to/results
```

Optional arguments after the output directory select cases: `mesh`, `station`,
`ou`, `wastewater`. Every selected case runs even if an earlier case fails;
any failure gives the process exit code 1. Successful sources and intermediate
artifact/program expressions remain in the chosen output directory. The log
records package versions and each failing stage.

The original reporter used Julia 1.11.9. The local investigation uses
Julia 1.10.12-kb3; the version printed by Julia is 1.10.12. Neither environment
is evidence for the other's compiler behavior.
