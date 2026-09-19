# Ranef Stage A: bucket IR + surface + lowering (design record)

Date: 2026-09-19. GO: `BayesianRegressionModels:rk:sb:ranef/decisions/2026-09-19T03-19-04-749-0mlx9gh`
(user chose `yes, go`). Lane todo:
`/agents/ReactiveKernels:brm/todos/2026-09-19T03-23-10-620-0o9gqpa` (in-progress).
Stage A = bucket IR + `@rkppl` surface + lowering + validation + committed
tests + corpus. Codegen is NOT Stage A: `build_kernel` fails closed on
non-empty `ranef_buckets` (K=1 codegen = Stage B, LKJ = Stage C). This
staging matches the todo's "define the thin-layer IR/`@rkppl` surface,
then notify so the BRM lane can emit against it".

## 0. SB mirror (verified reads, BRM repo `src/sbimpl.jl`)

- Geometry: `ranef_correlated_draws` (:494): `L ~ lkj_corr_cholesky(eta)`,
  `tau ~ half-normal(K)`, `z_flat ~ std_normal(K*G)`, `z = to_matrix` (K×G
  column-major), `b = (diag_pre_multiply(tau,L)*z)'` (G×K). K=1 fast paths:
  `ranef_intercept` (:401, `log_scale`/`xi`), `ranef_slope` (:410,
  `tau[1]`/`xi`). Plain blocks (:8232-8275): `(1|g)` → intercept path,
  single-slope → slope path, else correlated. ID buckets (:9527+): shared
  draws block + static per-target column ranges + sliced gathers
  (`b[idx,col]` / `rows_dot_product(Z, b[idx,cols])`, :9834).
- Group codes: `_brm_level_index` = `sort(unique)` for plain vectors
  (`backend_plan.jl:1488+`) — matches thin-layer `levels()`/`_grouping_levels`.
  `CategoricalVector` keeps DECLARED order SB-side; the thin layer sorts
  always (open item §5).
- `|ID|` + K=1: the ID emission path has NO K=1 fast path — a single-margin
  ID bucket goes through `ranef_correlated_draws` with `n_terms=1` (vacuous
  1×1 LKJ). Mirrored exactly (open confirmation §5).

## 1. Surface spelling (this lane's call; peer to confirm before emitting)

Compound do-block declaration (plain Julia, no new macro) + explicit gather
atoms in predictor affines (no name mangling anywhere):

```julia
@rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1)
    mu = a .+ b .* x .+ ranef(:ID, g)      # this predictor's slice of bucket (ID,g)
    y .~ Normal.(mu, sigma)
    ranef_bucket(:ID, g; eta=1.0) do       # ID bucket; eta default 1.0
        mu => [1, x]                        # margins for predictor mu
        sigma => [1]                        # second target slice (distributional)
    end
    ranef_bucket(h) do                      # plain bucket (no ID): one slice
        mu => [1]
    end
end
```

- `ranef_bucket(id, group; eta)` / `ranef_bucket(group; eta)`: `id` a literal
  `Symbol` (`:ID`), `group` a bare data column. Body: 1+ `target => [...]`
  margin lines, targets in any order (checked against predictors at
  structure validation — buckets may precede their predictors textually).
- Margin elements: `1` (intercept), bare column (continuous Z), explicit
  `dummy(c, k)` with `k::Union{Int,AbstractString}` (single indicator
  column; level VALUE for `Int`, exact match for strings). NO coding
  inference in the thin layer: BRM lowers treatment/cell-means INTO explicit
  `dummy()`s. Bare categorical columns fail closed directing to `dummy()`.
  Interactions fail closed as planned.
- `ranef(id, group)` / `ranef(group)` gather: vector atom, additive only
  (negated gathers rejected). The target is the enclosing predictor —
  the lowering verifies bucket + target slice exist.
- `eta`: literal `Real > 0`, admitted iff bucket kind is `:correlated`
  (K=1 plain buckets take NO eta — no correlation to parameterize; K=1 ID
  buckets keep it, mirroring SB's uniform correlated path).

## 2. IR (contract.jl)

```julia
struct RanefZRecipe          # one Z column recipe (never materialized pre-codegen)
    kind::Symbol             # :ones | :column | :dummy
    column::Symbol           # :none for :ones
    level::Union{Nothing,Int,AbstractString}  # dummy only
end
struct RanefMargin           # ranefcoefnames order per target
    predictor::Symbol
    coefficient::Symbol      # :Intercept | column | dummy label
    z::RanefZRecipe
end
struct RanefBucket
    id::Union{Nothing,Symbol}
    group::ColumnRef
    kind::Symbol             # :intercept1 | :slope1 | :correlated
    margins::Vector{RanefMargin}
    slices::Vector{Tuple{Symbol,UnitRange{Int}}}  # predictor => static cols
    lkj_eta::Float64         # correlated only (NaN otherwise)
    label::Symbol            # :bucket_<suffix>
end
```

- `StructuralPlan.ranef_buckets::Vector{RanefBucket}` (defaulted `[]`).
- Kind dispatch (structural, data-free): plain single-`1` → `:intercept1`;
  plain single non-`1` → `:slope1`; everything else (incl. ALL ID buckets)
  → `:correlated`.
- Gather terms: `TermKind` += `RanefGatherTerm`; `TermSpec(columns=[group],
  options=(bucket_id=..., bucket_group=...), addressee=label, label=r_...)`.
  Gathers take no `PopulationPrior` (validator exempts, like `LatentTerm`).
- In-graph names (Stage B): `b_<suffix>`, `r_<target>_<suffix>`,
  `L_<suffix>`, `tau_<suffix>`, `z_<suffix>`, `Z_<target>_<suffix>` (SB's
  `r_`/`Z_`/`b_` spellings); suffix = `ID_g` or plain `g`.
- Intended packed layout (Stage B, thin-layer-owned): per bucket
  `[L_unconstrained (C(K,2), Stan vine row-major), log tau (K),
  z_flat (K*G column-major)]`; K=1 intercept `[log_scale, xi(G)]`; K=1
  slope `[log_tau, xi(G)]`. LKJ math pinned to Stan exactly (vine forward +
  logJ from the reference manual §10.12; `lkj_corr_cholesky_lpdf` kernel +
  `do_lkj_constant` from Stan Math source, fetched 2026-09-19).

## 3. Validation (fail-closed, mirroring triple/ContractValidationError style)

Structure (`_validate_ranef_buckets`): bucket keys unique; one margin list
per target; slices partition `1:K` contiguously in body order; K≥1; kinds
consistent with margins (`:intercept1` ⟺ plain single-`1`, etc.); eta rule
(admitted ⟺ `:correlated`, `> 0`); every gather references an existing
bucket + a slice for its own predictor; no duplicate (predictor, bucket)
gathers; no dangling buckets (every slice gathered ≥ once); targets are
real predictors; margin columns are Symbols.
Data (`_validate_ranef_buckets_data`): group column bound; G≥1 via
`_grouping_levels`; margin columns bound; dummy levels are members
(`Int` value / string exact); gather group columns pass the generic
presence check with NO numeric-eltype requirement (strings allowed).

## 4. Stage A file plan

- `contract.jl`: kinds/structs/plan field/validators/handshake
  (`admitted_terms` += gather, `TERM_NAMES[:ranef_gather]`).
- `surface.jl`: partition `ranef_bucket` do-blocks; lower buckets+margins;
  `ranef()` gather branch in `_classify_summand` (+`:vector` shape,
  rejection allowlist); explicit fail-closed messages with admitted forms.
- `generator.jl`: `build_kernel` fails closed on `ranef_buckets ≠ []`
  (Stage B/C pointer). `_term_block`'s `else` already fails closed.
- `ReactiveKernelsPPL.jl`: export new types + kind.
- `test/test_ranef.jl`: acceptance item 1 (bucket IR + lowering: shape,
  slicing, fail-closed). Corpus: 3 ranef programs + goldens (rebless all —
  new plan field changes every golden).

## 5. Open items for the ranef peer (in the Stage-A notify)

1. Spelling confirmation (or counter-proposal) before BRM emits.
2. Confirm K=1 ID buckets take the correlated path (no fast path).
3. `CategoricalVector` grouping order: thin layer sorts always; BRM
   normalizes the RK-bound column or we add the dep later.
4. Ext serializer constructs `RanefBucket`/`RanefGatherTerm` 1:1 (field
   names in §2); unmapped keys stay internal errors per the ext contract.
