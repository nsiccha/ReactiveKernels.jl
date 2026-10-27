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

## 0.1.2

- A summed plate whose cells hold arrays adds them elementwise, as Julia's
  `sum` does: with `cells = plate(groups) do xs; b .* xs; end`,
  `total = sum(cells)` is `b .* sum(groups)`, and a cell `[b * u, b * v]` or
  a matrix cell `xs * xs'` sums the same way, also when the plate consumes a
  scan or runs by index runs. The fused total starts from the first cell; it
  was seeded with `zero(T)` for the cell type, which raised
  `MethodError: no method matching zero(::Type{Vector{Float64}})` before any
  cell ran. Cells of different lengths raise Julia's `DimensionMismatch`, and
  an empty plate fails as `sum(Vector{Float64}[])` does. Numeric cells keep
  their `zero(T)` start. Native values and ordinary Enzyme Reverse gradients.
