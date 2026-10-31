# Changelog

All notable changes to ReactiveKernels.jl are documented here, newest first.
The nested packages under `packages/` keep their own versions and changelogs.

Every landing that adds or changes public API (an export, a macro, a keyword,
a method, or documented behaviour a dependant can rely on) bumps the patch
digit in `Project.toml` and adds its entry under a new heading at the top of
this file. An entry that breaks something a dependant relies on says
**Breaking:** and what to change. The minor digit, the breaking one in
[semantic versioning](https://semver.org/) before 1.0, has not moved yet.
A package or app that calls that API states the version as its floor, for
example `[compat] ReactiveKernels = "0.1.2"`. An environment that pins an
older ReactiveKernels commit is then refused when it resolves, with
`Unsatisfiable requirements detected for package ReactiveKernels`, instead of
failing or silently behaving differently at run time. Changes before 0.1.2
are recorded in the git history only.

## 0.1.35

- A plan holds every value computed from the bound data and the label alone:
  the index bookkeeping and label-constant data (a data column's rows, a
  design block's rows and their `hcat`), computed once per label by pure
  operations (sources, plain functions, partition ops; plates, scans and
  cells stay in the body); never a value computed from an unbound port, its
  alias, a type the body computes, or a value the body writes into.
- `plate_partition(…; plan_constant = (:port, …))` names HAVE ports whose
  values are constant per label, which the plan may read; a lazy view of
  inputs serves its rows from its parts through its own
  `_cell_lazy_rows(op, inputs, rows, bind, constant)` method:
  ReactiveKernelsPPL's compact overlay reads its active vector at slots
  planned per label instead of a slot load and a branch per element.
- Planned partitions use ordinary Enzyme Reverse. Enzyme dogfood from
  `1280e6ee` repairs constant plan fields; grouped compact partitions on
  Julia 1.10 also need `a6458e9c` (snag `static-reverse-g-173719b3`).
  Registered 0.13.216 has the upstream activity-analysis defects; runtime
  activity is not an accepted remedy.
- Plan splitting preserves indexed/property writes, writes through aliases,
  mutating calls and repeated assignments in the evaluation body. Empty and
  nonempty plate portions share one plan type rather than exporting the
  final loop state (`nothing` or a `CartesianIndex`). Reusing a
  plan with another label throws `ArgumentError` instead of silently returning
  the original label's value.

## 0.1.34

- Bound preparation caches shared data-only scalar work from nested plate
  cells, including pure expressions inside lazy branch arms. A speculative
  read that throws or produces a nonfinite floating-point value stays under
  its original loop and branch, so empty groups and inactive invalid inputs
  keep their values and ordinary derivatives. Both plate loops remain
  runtime loops, and array results retain their per-evaluation storage.

## 0.1.33

- Generated programs can compose an authored graph reconstructed by an
  explicitly registered structural constructor. The compiler resolves only
  that constructor's static syntax; ordinary calls remain ordinary calls.
  RKPPL uses this internal integration point to emit replayable caller-owned
  parameter geometry without adding global bindings or opaque runtime graph
  preparation. Nested graph recipes retain their normal locality analysis.

## 0.1.32

- Retain sum and eager-operand source boundaries for factored cell queries,
  separately from the boundaries used by ordinary partial evaluation. The
  cut pieces keep their authored error policy and use the existing graph,
  cell row lowering and derivative operators. Cell result types propagate
  from their scalar bodies when one cell supplies another cell's row index.
  ReactiveKernelsPPL 0.1.25 exposes the outer and summand queries.

## 0.1.31

- Authored source companions infer and execute when ReactiveKernels loads
  from source or a consumer prepares and warms a kernel during package
  precompilation on Julia 1.12+. The precompile wrapper type is defined before
  the source-call generator, and the generator reads a consumer's constant
  expression-cache binding as compiler metadata even when that binding is
  newer than its definition world. Strict type-stability checks remain
  enabled, and the generated numerical body is unchanged.


## 0.1.30

- Strict plate cells and partitions accept vector-valued products with an
  inline right operand, such as `PHI * (weights .* z)`, and multi-factor
  products such as `A * W * z`. When Julia infers a vector result, its full
  shape follows from the left factor's row axis; plate-computed weights keep
  the rank inferred from their batched inputs. Dotted readers can therefore
  consume only the requested rows without computing the whole product or
  evaluating its right operand to discover its shape. This fixes strict
  partition preparation for BRM HSGP predictors.
