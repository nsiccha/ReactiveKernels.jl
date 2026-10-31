# Changelog

All notable changes to ReactiveKernelsPPL.jl are documented here, newest
first.

Every landing that adds or changes public API (an export, a macro, a keyword,
a method, or documented behaviour a dependant can rely on) bumps the patch
digit in `Project.toml` and adds its entry under a new heading at the top of
this file. An entry that breaks something a dependant relies on says
**Breaking:** and what to change. The minor digit, the breaking one in
[semantic versioning](https://semver.org/) before 1.0, has not moved yet.
A package or app that calls that API states the version as its floor, for
example `[compat] ReactiveKernelsPPL = "0.1.1"`. An environment that pins an
older ReactiveKernelsPPL commit is then refused when it resolves, with
`Unsatisfiable requirements detected for package ReactiveKernelsPPL`, instead
of failing or silently behaving differently at run time. Changes before 0.1.1
are recorded in the git history only.

## 0.1.28

- A partition's plan holds the bound data at the label's rows too (design
  blocks included), and the compact sampler reads a packed block's rows from
  its active vector at slots computed once per label from that label's slot
  map, not through the overlay's per-element slot load and branch; a
  partition reading a packed coordinate outside its footprint is refused by
  name at preparation. Requires ReactiveKernels 0.1.35.
- The earlier 2.04× sweep-share measurement used runtime activity and does
  not establish ordinary-Reverse acceptance. Grouped compact partitions on
  Julia 1.10 require Enzyme dogfood from `a6458e9c`, whose constant-field
  analysis repairs the remaining `EnzymeRuntimeActivityError` after
  `1280e6ee`. The partition tests exercise ordinary Reverse.

## 0.1.27

- `ParameterGeometry` accepts function-shaped `@kernel` endpoints in its
  existing `constrain`, `unconstrain` and `logjac` slots. Host transforms retain
  prepared callables; emitted transforms compose the authored graph at a full
  argument boundary, including explicitly supplied keywords. This exposes
  identity and row-local vector transforms to strict partitions while preserving
  actual coupled dependencies and the caller's joint prior. Scalar cells
  and joint cell latents use the same endpoints. Printed `kernel_expr` source
  reconstructs the geometry through `sampling_geometry`, including parametric
  law constructors. Ordinary callbacks,
  omitted defaults and splatted arguments retain their numerical call path.
  Requires ReactiveKernels 0.1.33.

## 0.1.26

- `prepare_cell_factor_query(f, :outer)` and
  `prepare_cell_factor_query(f, j)` return scalar `CellFactorQuery` objects
  for the outer density and one summand, before compact AD indices are
  known. Call `outer(u, sums, cell)` or `term(u, cell, term_position)`.
  Recording `AbstractVector{Float64}` points pass through without copying;
  the public `kernel` supports RK's structural footprint API for discovering
  each factor's packed-coordinate selection, including retained branches.
  This completes the value/support preparation path for the sum-factored
  derivatives added in 0.1.25.

## 0.1.25

- `prepare_cell_factors(built, plan, observation)`, `cell_reductions` and
  `reduction_rows` discover a cell's sums, including inline expressions,
  local names and earlier plate outputs. `prepare_cell_factor_ad` prepares
  any RK derivative operator for the outer density (`:outer`, active over
  the sum vector and selected direct packed coordinates) or one summand
  (a reduction index, active over selected packed coordinates). Calls use
  the ordinary RK derivative verbs and compact coordinate selections.
  Empty reductions make no summand calls; no-sum cells have an ordinary
  outer density. The pieces support Hessian and third-order chain rules for
  sparse Laplace without a dense block over every coordinate a sum reads.
  Requires ReactiveKernels 0.1.32.

## 0.1.24

- Document and test the corrected upstream `DirichletMultinomial` dependency
  used by the optional Distributions adapter. The repository acceptance setup
  pins Distributions commit `83cec5e3a8a35167d54f19da9dc5afb54f59273e`, whose
  `logpdf` rejects invalid counts and mismatched totals before evaluating the
  density. Native Enzyme gradients on valid observations are unchanged.
  Other environments must select that dependency revision explicitly;
  upgrading RKPPL alone does not repair Distributions 0.25.132. The adapter
  continues to delegate directly to the distribution's public `logpdf`.
