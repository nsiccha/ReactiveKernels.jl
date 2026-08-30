# ReactiveKernels example packages

The examples are ordinary nested Julia packages in this monorepo. Core
`ReactiveKernels` never depends on them; the root documentation environment is
the one integration sink and still builds one Documenter/VitePress site.

```text
ReactiveKernels
├── ReactiveKernelsDistributionKernels
│   ├── ReactiveKernelsKernelExamples
│   └── ReactiveKernelsPPLExamples
├── ReactiveKernelsBatchingExamples
└── ReactiveKernelsCompatibilityExamples

docs → every package above
```

`ReactiveKernelsDistributionKernels` owns the reusable mathematical
`KernelSpec`s. The PPL package imports those specs and owns seven nested model
modules; it does not copy their formulas or flatten their colliding exports.
The general kernel-example package owns the distribution gallery, bijectors,
fixed-step HMC, and runnable core walkthrough. Batching remains a separate
package because its public boundary includes DifferentiationInterface cache
semantics. Compatibility examples are isolated from current authoring examples.

Julia 1.10 is still supported, so its package manager ignores `[sources]` and
does not provide the newer workspace mechanism. Materialize the local paths
explicitly before loading or testing them:

```sh
julia --startup-file=no --project=packages packages/setup.jl
julia --startup-file=no --project=packages packages/test.jl
julia --startup-file=no --project=docs docs/make.jl
```

The checked-in `[sources]` entries become useful automatically on newer Julia
versions; `setup.jl` remains the Julia-1.10-compatible source of local path
development. CI runs the same path development explicitly rather than assuming
that root `Pkg.test()` recurses into nested packages.

The NUTS compiler exemplar and its online-diagnostics consumers remain at the
legacy `examples/nuts_runtime` boundary for the moment. Their current loader
injects private backend definitions into `ReactiveKernels`; moving those files
unchanged would create a package that only pretends to own its namespace. The
active generic-control extraction must first expose the reviewed callable
backend hook. Once it does, the remaining DAG is:

```text
ReactiveKernelsNUTSExamples → ReactiveKernels
ReactiveKernelsStreamingStats → ReactiveKernels
ReactiveKernelsHMCDiagnostics → {
    ReactiveKernelsStreamingStats,
    ReactiveKernelsNUTSExamples,
}
docs → all three
```

That staging boundary preserves the byte-locked NUTS source, independent eager
oracle, native/Reactant parity, and the rule that bare `using ReactiveKernels`
loads no sampler API.
