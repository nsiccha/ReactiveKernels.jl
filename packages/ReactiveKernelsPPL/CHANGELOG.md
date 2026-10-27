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

## 0.1.1

- A `@plate` output whose cells are vectors is one vector per cell at model
  scope, as the Julia loop leaves it: `sum(cells)` adds the cells
  elementwise, `cells[g]` is cell `g`'s vector and `cells[k]` (a data index
  vector) the cells `k`, and a value computed from them is a location like
  any other vector (`obs .~ Normal.(sum(cells), s)`,
  `obs .~ Normal.(sum(cells[k]) .+ cells[3], s)`). Before, a location reading
  `cells[1]` or `sum(cells)` failed `bind_data` with `term references missing
  column _rkppl_leaf_1`, and `sum(cells)` over cells that read one array per
  index raised `MethodError` from `zero(Vector{Float64})`. Native values and
  ordinary Enzyme Reverse gradients.
- A cell that builds an array literal from per-index values
  (`cells[i] = [b * a1[i], b * a2[i]]`, `[x[i]; y[i]]`, a comprehension) runs
  once per index, as a retained RK plate cell. It was evaluated over whole
  columns, which built one array per literal entry (`[b .* a1, b .* a2]`), so
  `cells[k]` was checked against each cell's length instead of the cells.
- Requires ReactiveKernels 0.1.2.
