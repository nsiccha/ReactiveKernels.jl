# Reactant integration

Reactant is an optional compiler backend, not a dependency of native
ReactiveKernels. The reviewed surface starts with scalar and batched
distribution kernels, then the Eight Schools and MNIST PPL models, followed by
compiled automatic differentiation.

The same `PreparedKernel` or `PreparedADKernel` boundary is traced; Reactant does
not introduce a second model language. A Reactant path is supported only where
the corresponding primal kernel compiles, and unsupported traced storage or
control fails closed.

## Read in this order

1. [Distribution kernels through Reactant](distributions-reactant.md) — scalar,
   structured, and batched boundaries.
2. [Eight Schools through Reactant](eight-schools-reactant.md) — reviewed PPL
   primal and value-and-gradient evidence.
3. [MNIST through Reactant](mnist-reactant.md) — reviewed full-data model
   evidence.
4. [AD + Reactant](reactant-ad.md) — the compiled prepared-AD API and its limits.

Older throughput and experimental sampling receipts appear afterward. They are
not promoted into the reviewed capability set, and sampling compiler/runtime
code is not executed by the docs build.

## Current boundary

- Prepared scalar kernels and tensorized distribution plates are accepted.
- A plate with at most 16 static lanes lowers as per-lane scalar recipes with
  a scalar reduction, so a small posterior fuses into one CPU kernel; larger
  plates keep the batched lowering. The automatic AD compile keeps bound
  arrays of at most 4096 elements embedded as compiler literals.
- Whole-kernel `replica` preserves the scalar kernel as its source authority.
- Compiled AD reuses the native single-active-port, scalar-WANT validation.
- Unsupported scalar indexing, unbounded control, or structural state rejects;
  the docs do not paper over those errors with alternate implementations.
- NUTS/WALNUTS pages are source- and receipt-only during documentation builds.
