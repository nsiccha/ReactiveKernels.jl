# Core constraints

These constraints apply to every ReactiveKernels lowering, extension, and
example package. They also apply when data is bound during preparation.
Backend limitations must be reported explicitly; they do not relax these rules.

## Preserve data-dependent iteration

**Do not statically unroll a loop whose trip count derives from data.** This
includes runtime inputs, bound arrays, shapes, subject or observation counts,
operation tables, ragged ranges, dose counts, and capacities computed from any
of them. Knowing a length at preparation or trace time does not make it a
structural constant. For example, expanding one statement per bit of the largest
bound dose count is forbidden just as expanding one statement per subject is.

Retain a runtime loop, scan, or equivalent backend control-flow operation with
explicit carried state and output buffers. Shapes may specialize an executable;
the lowering must not duplicate the body according to those shapes or values.
Native execution retains ordinary iteration. Fixed structural algebra, such as
the scalar entries of an intrinsically three-compartment operator, is distinct
from data-derived iteration and may be expanded.

## Preserve lazy branches

**Do not replace required lazy control flow with eager `ifelse`, masked
evaluation of both branches, or equivalent predication to bypass a compiler
failure.** An inactive branch must remain inactive, including its indexing,
allocation, mutation, potentially undefined arithmetic, and derivative work.
Clamping indices or inventing dummy buffers does not make such a rewrite valid.

Ordinary elementwise selection between already valid values remains a selection;
it does not authorize evaluating an otherwise inactive computation. Preserve
the authored Julia branch and loop semantics in both primal and derivative
execution. If a backend cannot express them, report the unsupported case and
isolate the backend failure instead of adopting an eager workaround.

## Acceptance and existing limitations

A lowering change must demonstrate that increasing relevant data lengths or
capacities does not replicate loop bodies or control-flow regions. Check the
generated backend structure as well as primal and AD parity with native Julia,
including inactive branches and empty or ragged cases where supported. A small
Julia statement count alone does not establish this: tracing can still expand
a host loop.

These are required constraints, not a claim that every existing path already
conforms. Known gaps include host-bound Reactant `scan` fallbacks and legacy
bounded stateful unrolling when their bounds derive from data. The
[scan](scan.md) and [compiler](compiler.md) pages describe their present behavior.
Those paths require retained control flow or explicit rejection; their current
behavior is not an exception or a template for new implementations.

The experimental rectangular PK path retains its loops and lazy branches but
still fails reverse compilation. Its eager-branch and data-derived unrolling
workarounds are not acceptable fixes. See the [scan limitations](scan.md).
