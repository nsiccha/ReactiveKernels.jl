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
- Every plate keeps the batched or broadcast lowering whatever its lane
  count: a lane count is a data length, so no plate is unrolled into per-lane
  scalar recipes (see [core constraints](constraints.md)). The automatic AD
  compile keeps bound arrays of at most 4096 elements embedded as compiler
  literals.
- A plate with one lane axis and array-valued cells retains its layout
  through outer consumers.
  Ordinary helpers can use `stack(lanes)` or `stack(lanes; dims=d)` to place
  the lane axis as in native Julia; `vec(stack(lanes))` packs each complete
  lane consecutively. An authored `sum(lanes)` adds arrays across lanes, preserving
  their per-lane shape. A directly returned compiled plate still materializes
  to its dense storage with the lane axis first.
- Authored `if`, `?:`, `&&` and `||` lower to lazy `stablehlo.if` regions
  (also inside a batched plate cell), retained through ordinary MLIR AD.
  Default CPU XLA on Reactant 0.2.290 can subsequently replace pure live
  guards with eager selection, including inactive logarithms and division.
  Correct values/gradients and matching inventories alone do not establish
  executable laziness. Preserve the authored guard; use native execution when
  inactive arithmetic must stay inactive. `Base.ifelse` remains an eager
  select of two already valid values. See the executable boundary and removal
  criteria in [core constraints](constraints.md) and the backend-only
  `benchmark/repro_reactant_pure_lazy_guard.jl`.
- An authored `for`/`while` inside a recipe keeps its iteration: the
  tensorized companion expands it with `ReactantCore.@trace` at kernel
  definition, so it becomes one `stablehlo.while` region whatever the trip
  count; the loop body is never replicated per iteration. A carry may start
  on the host (`acc = 0.0`, `zeros(n)`): it is re-bound to a fresh traced copy
  before the loop, because `@trace` carries a value only by updating a traced
  object that exists before the loop (before, such a loop silently returned
  its seed). Inside a recipe or `@traceable` helper, a conditional assignment
  such as `if flag; acc = acc + x; end` carries the updated binding out of
  the selected arm; the other arm keeps its existing value. Assignments to
  several locals keep their order. A branch used as an expression also
  returns its authored value. These branches stay lazy inside the retained
  loop, including an empty loop. A loop may read a host struct such as a
  schedule plan; it crosses the loop untraced, and a traced value it reads
  enters as a fresh
  tracer, so a zero-sized input is not returned as an aliased output (which
  XLA export rejects). A tuple or named tuple it reads is opened leaf by
  leaf, so a partly traced model keeps a host matrix host instead of handing
  the loop one traced scalar per element. A loop over an empty host range is
  skipped, not traced: `@trace` would trace its body once, indexing
  zero-length arrays.
  A loop over `eachindex(x)`, `axes(x, d)` or
  `a:b` whose scope rebinds `first`, `step`, `last`, `one`, `zero`, `div`,
  `isqrt` or `error` is traced per iteration instead, since `@trace for`
  resolves those names where it expands.
- A generator reduction with `init` over a data-length iterator,
  `sum(term for i in eachindex(x) [if cond]; init = x0)`, is the same retained
  loop with an explicit accumulator (the filter stays a lazy branch), so a sum
  over doses does not grow the program with the dose count. The loop index is
  then traced: indexing written in the term is lowered, host tables included.
  An element read with one integer index per dimension, any of them traced, is
  one gather, so `p.shifts[j]` and `W[i, j]` lower alike, for a host table, a
  traced array, or a host container of traced scalars (stacked first).
  A helper the term calls with it is lowered the same way when it is defined
  with `@traceable` (`@traceable lag(t, p::Lattice, j) = t - p.shifts[j]`);
  an ordinary helper indexing with it fails with `Scalar indexing is
  disallowed`. A generator over a helper-produced iterator keeps the
  per-element trace.
- A kernel calls `ReactiveKernels.traced(f, args...)` in place of a helper
  call `f(args...)` when a tracing backend compiles it (only for functions
  Base, Core and ReactiveKernels do not own, and only positional calls). Its
  default is `f(args...)`. A method for your own function gives it a tracing
  implementation while native execution keeps calling `f`: an opaque
  component, such as a hand-tuned in-place loop, can run under tracing as a
  prepared kernel or backend code in the package's Reactant extension.
  `@traceable` defines that method from the helper's own body. The two
  implementations are separate code, so test the traced one against the
  native one.
- A traced Cholesky factorization is a wrapper type owned by the Reactant
  extension, with `LinearAlgebra.Cholesky`'s accessors (`.L`, `.U`, `.UL`,
  `.factors`) and solves (`\`, `ldiv!`; a diagonal factor divides
  elementwise). A Cholesky passed as state, a factorization stored into
  compiled reactive state, and a `cholesky(...)` call inside a larger
  `@kernel` expression (`cholesky(Symmetric(K)).L`) produce it; the extension
  defines no methods on Reactant's own factorization type, which has no
  `.L`/`.U`. Within one kernel call, a `cholesky` call outside that lowering —
  in a helper function the kernel calls, or a statement that is exactly
  `F = cholesky(A)` — yields Reactant's type, whose packed factor is
  `F.factors`. A structured-state Cholesky field leaves a raw compiled
  transition as that wrapper; `validated_compiled_transition` restores the
  source `LinearAlgebra.Cholesky` at its host bridge, so the returned state
  can be passed to the next guarded call.
- Whole-kernel `replica` preserves the scalar kernel as its source authority.
- Compiled AD reuses the native single-active-port, scalar-WANT validation.
- Unsupported scalar indexing, unbounded control, or structural state rejects;
  the docs do not paper over those errors with alternate implementations.
- NUTS/WALNUTS pages are source- and receipt-only during documentation builds.
