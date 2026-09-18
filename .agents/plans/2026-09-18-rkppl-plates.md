# Joint plate + group-levels design (APPROVED 2026-09-18 — F1 no-preference → recommendation stands; F2 full-rank by ruling)

Date: 2026-09-18. Follows the accelerated todo
`2026-09-18T16-01-51-424-1ooske6` (dual plate spellings + `c[<levels>]`).
Design-first: no implementation until the user approves this doc (open
forks F1/F2 are separate decision rows).

## 0. Binding user directives (all recorded on the todo/decision trail)

- D1: `.~` stays AND an explicit plate form joins it; both lower to the
  plate correctly and automatically.
- D2: Leading explicit candidate is slice-LHS: `y[1:n] .~ ...` — the LHS
  slice declares the broadcast range in plain Julia indexing.
- D3: Single-LHS-ownership: `y` on the LHS of exactly one sampling
  statement. `a.b ~ ...` out of scope. (Both `y[1:n]` and `a.b` were
  "for later", then accelerated by the group example.)
- D4: Scalar-looking vector priors must go: `c ~ Normal(0, 2)` for a
  vector `c` is rejected; replacement shape is `c[<levels>] .~ ...`,
  where the index expression sizes `c` and fixes the level→position
  mapping. `unique` is a placeholder, probably not the final function.
- D5: `@plate`/`@scan` shapes copied from StanBlocks verbatim (standing
  constraint c2); `@scan` stays reserved.

## 1. Spelling A: range-explicit broadcast `y[R] .~ ...`

The julianic explicit form for the common case (no per-cell structure).

- Admitted ranges `R`: `1:N` (integer literals), `eachindex(y)`,
  `axes(y, 1)`. Rationale: no unbound names — there is no `n` bound in
  the surface, so `1:n` as literally written is rejected (naming
  `eachindex(y)`); anything else (calls, arithmetic, names) fails
  closed with the admitted list.
- Coverage rule: `R` must equal `eachindex(y)` exactly. Partial
  (`y[1:3]`), over-wide, or offset ranges fail closed — every cell is
  sampled exactly once, and the ownership rule below makes that
  checkable.
- Ownership rule (D3): a response column appears on the LHS of exactly
  one sampling statement per block, counting `~`, `.~`, and `y[R]`
  forms together. A second LHS use fails closed naming the first
  statement's line.
- Bind check: `length(R) == n_obs`, else a bind-time error naming both
  lengths. (`R` is data-dependent only through `y` itself, so this is
  nearly free.)
- Lowering: byte-identical plan to `y .~ ...` (same `LikelihoodSpec`).
  Equivalence `y .~ f` ≡ `y[eachindex(y)] .~ f` is pinned by
  plan-equality tests. DEVIATION (slice A implementation): literal
  `1:N` needs its `N` at bind, so `LikelihoodSpec` gained one
  defaulted field (`range::Union{Nothing,UnitRange{Int}}`, `nothing` =
  whole column); structure + data validators enforce the cover rule.
  `eachindex`/`axes` carry no range (self-covering).

## 2. Spelling B: `@plate for i in R ... end` (StanBlocks-verbatim)

The cell-explicit form, for bodies with per-cell structure. Shape copied
verbatim from StanBlocks (`@slic` docstring):

```julia
@plate for i in eachindex(y)
    y[i] ~ Normal.(mu[i], sigma)
end
```

- `R`: same admitted forms and coverage rule as spelling A.
- `~` inside a cell is scalar (cell-level) — consistent with the
  standing scalar-`~` / broadcast-`.~` split. `.~` inside a cell fails
  closed ("cells are scalar; broadcast at top level").
- Cells mirror top-level spelling exactly (dots as written): the cell
  object must already be dotted (`Normal.(...)`, never scalar
  `Normal(...)`); the desugar strips refs and never invents dots.
  DEVIATION from the draft (which showed a scalar object): scalar
  objects would need an auto-vectorizing desugar, contradicting the
  explicit-dots ruling. For-loop shape and cell rules stay
  StanBlocks-verbatim; only the object spelling is julianic-dotted.
- Cell rules (StanBlocks, adapted to immutability): arrays indexed by
  the loop variable are model-scope; bare fresh names are per-cell
  locals; writes to outer-bound names are rejected; reads only at the
  loop index — `x[j]`, `x[i-1]`, and whole-vector refs fail closed
  ("cross-index reads need `@scan`", which stays reserved).
- This slice: observations (`y[i] ~ ...` on data) + deterministic cell
  assignments only. These desugar to the existing nodes (same
  `LikelihoodSpec` / derived assignments the `.~` form produces) — no
  IR growth. Value over spelling A: the loop is written, and the shape
  is the growth path.
- Deferred (needs IR growth, sketched in §5): per-cell sampled arrays
  (`x[i] ~ ...` where `x` is model-scope, not data) and multi-index /
  values-iteration plates (`for i in ..., j in ...`, `for d in doses`
  — fail closed as "planned" for now).

## 3. Spelling C: `c[L] .~ ...` broadcast priors over levels (D4)

The directed replacement for scalar-looking vector priors:

```julia
c[levels(g)] .~ Normal.(0, 2)   # all K entries, full-rank (§4); no
mu = c[g] .+ o                 # intercept (else unidentified → §4)
```

- One expression, two jobs: `L` sizes `c` AND fixes the level→position
  mapping (position `k` ↔ `L[k]`). This also settles the reference
  question §4 derives from.
- Predictor-side `c[g]` keeps working over the mapped levels: the
  generator emits one dummy column per mapped value, so rows whose
  codes fall outside a strict subset contribute 0 (the subset is
  explicit on the page — reference rows under an intercept work this
  way). Unobserved mapped levels are allowed like any zero-variance
  column.
- `L` is evaluated at bind from the grouping column (pure, total
  function of data — no sampling, no parameters).
- Migration: any `coef ~ ...` scalar prior whose LHS is used in vector
  position (`c[...]`) fails closed directing to `c[L] .~ ...`. The old
  sugar is removed, not deprecated (D4: "must go").
- Fork F1 (decision row): the levels function. Candidates:
  (a) `unique(g)` — Base, order of appearance;
  (b) `sort(unique(g))` — sorted values;
  (c) a new surface word `levels(g)` — defined as (b), self-describing.
  Recommendation: (c) — the mapping order is load-bearing (it fixes
  positions AND the reference level), so it deserves a name that says
  what it guarantees rather than whichever accident `unique` has.
  `unique`/`sort(unique(...))` spellings are rejected (naming `levels`)
  so there is exactly one mapping.

## 4. Reference/contrast rule (fork F2 — RESOLVED: full-rank)

User ruling (~16:53): rkppl carries nothing BRM-specific, and the
treatment/reference machinery (R `contr.treatment` tradition, silent
ref-1 drop, `treatment(g, ref)` AST vocabulary) is exactly that. So:

- FULL-RANK, no dropping anywhere in rkppl, ever. `c[L] .~ ...` means
  exactly `|L|` coefficients over exactly the written levels. The old
  K-vs-K−1 crux dissolves: `c[levels(g)]` has all K entries, period.
- The `treatment(g, ref)` index vocabulary is REMOVED from the surface
  (BRM-specific); its use fails closed directing to explicit level
  subsets. IR `ref`/dropping machinery dies with it (`DesignBlock`
  width = `|L|`; `LevelMap` carries exactly the written levels).
- Identifiability is enforced, not assumed: a full-cover factor term
  alongside an intercept in one predictor fails closed ("unidentified:
  drop the intercept or index a strict subset of levels"). A
  user-chosen strict subset (e.g. `c[levels(g)[2:end]]`) is an
  explicit user choice over fewer coefficients — not a hidden
  reference — and lowers fine.
- NOT behavior-preserving: existing factor models (including the
  peer's corpus factor cases and `test_surface.jl` factor tests) lower
  differently and the peer's emitter must stop emitting the treatment
  sugar. Cross-repo impact flagged to the peer; their 26/26 corpus
  will need factor-case rework on their side.

## 5. IR growth (concrete, minimal)

- `y[R]` / `@plate`-observations: NO new IR (same nodes, §1–§2).
- Spelling C needs one node: `LevelMap(grouping::Symbol,
  values::Vector, source::Symbol)` — plan-level, binder-evaluated from
  `L`, carried alongside the `FactorTerm` it sizes. The generator's
  factor design-matrix build reads the map instead of assuming `1:K`.
  `source` records the levels function (`:levels`) so a future second
  function cannot silently mix mappings.
- Deferred: `PlatedParameter` (per-cell sampled arrays, §2) and
  `@scan` recurrence nodes. Sketched, not specified — later slice.

## 6. Failure catalog (each fails closed naming the fix)

Ranges/ownership: unadmitted range form; `1:n` (unbound `n`);
partial/over-wide range; second LHS for one response; `length(R) ≠
n_obs` at bind. Plate cells: `.~` in a cell; cross-index/lag read
(`@scan` reserved); whole-vector ref in a cell; write to an outer name
in a cell; multi-index or values-iteration plate ("planned"). Levels:
non-`levels` index in `c[L]`; scalar `~` prior for vector-used coef;
bind-time code outside `L`. Standing: `a.b` LHS stays rejected.

## 7. Equivalence + test plan

- `y .~ f` ≡ `y[eachindex(y)] .~ f` ≡ `@plate` observation form:
  plan-equality tests across all three.
- Spelling C is NOT plan-equal to today's sugar (full-rank §4 vs
  treatment-dropped): factor tests are rewritten to the new semantics
  with independent oracles, and the old sugar + `treatment()` vocabulary
  are removed per D4.
- Transpile-report coverage for the new spellings (reporter input
  modes already AST-capable; add a levels demo).
- Hand oracles stay independent (Distributions.jl loops, never the
  fused forms), per standing test convention.

## 8. Sequencing (after approval)

A (ranges + ownership, smallest, no IR) → C (levels + `LevelMap`) → B
(`@plate` observations, desugar) → deferred: per-cell sampled arrays,
`@scan`. BRM emitter untouched throughout (keeps emitting `.~`);
explicit forms are user surface first, future emitter vocabulary later.

## 9. Open forks (decision rows, filed alongside)

- F1: RESOLVED (~17:02, user chose "no preference") — recommendation
  stands: `levels(g)`, the DataAPI/CategoricalArrays word, adopted.
- F2: RESOLVED by user ruling (~16:53, row marked moot) — full-rank,
  no treatment/reference machinery in rkppl (§4).

## 10. Slice C implementation notes (landed)

- Unmapped rows contribute 0 (subset explicit on the page); the §3
  draft line about bind errors for outside codes was wrong and is
  corrected above. Unobserved mapped levels allowed (zero-variance
  precedent).
- Subset goes INSIDE: `c[levels(g)[S]]` with `S` in {`a:end`, `a:b`,
  `[i, j]`} (all literal, 1-based); outside-chained
  `c[levels(g)][S]` rejected (one way). `end` is a surface Symbol
  resolved at bind against the observed count.
- Required-prior rule: factor coefficients need the broadcast prior
  (no Normal(0,1) default — the default cannot size the block);
  scalar priors for vector-used coefs fail closed (migration);
  unused levels priors fail closed.
- One map per (predictor, column): duplicate factor terms over one
  column fail closed (unidentified sums — merge them).
- Slice A note (kept): `y[:]` rejected in both positions (full cover
  is the bare form — one way); 3-line follow-up if wanted.

## 11. Slice B implementation notes (landed)

- Desugar, no IR: plates expand in a partition pre-pass to top-level
  statements; literal ranges reuse the slice-A `y[a:b]` path
  (start-1, literal endpoints, bind-time cover); ownership flows
  through single-assignment automatically.
- Dotted-object rule (deviation, §2): cells mirror top-level spelling
  — scalar objects rejected, no auto-vectorization (would contradict
  the explicit-dots ruling).
- Bare-vector check is post-analysis (side table): data/predictors/
  derived/factor-coefs read bare in a cell fail naming `[i]`. The
  load-bearing case (bare predictor, silently valid when stripped) is
  pinned by message test.
- Cell assignments leak model-wide (documented looseness): desugared
  locals are plain top-level dets, visible outside the plate.
  Per-cell sampled arrays + `@scan` stay deferred; multi-index and
  values-iteration fail closed as planned.
