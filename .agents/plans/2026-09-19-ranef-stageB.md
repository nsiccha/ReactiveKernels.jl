# Ranef Stage B: K=1 codegen + parity (design record)

Date: 2026-09-19. Lane todo: `/agents/ReactiveKernels:brm/todos/2026-09-19T03-23-10-620-0o9gqpa`
(spec: `BayesianRegressionModels:rk:sb:ranef/decisions/2026-09-19T03-19-04-749-0mlx9gh`).
Stage A (bucket IR + surface + lowering, no codegen) landed in `0cde428`.

## SB mirror (verbatim shapes, `sbimpl.jl:401-417`)

- `ranef_intercept`: `log_scale ~ std_normal()`, `xi ~ std_normal(; n=n_groups)`,
  `exp(log_scale) * xi[group_idx]`. Log-normal scale, NO Jacobian (the `exp`
  is inside the returned expression, not a declared transformed parameter),
  NO renormalizer.
- `ranef_slope`: `tau ~ std_normal(; n=1, lower=0.0)`, `xi ~ std_normal(;
  n=n_groups)`, `tau[1] * (xi[group_idx] .* Z[:, 1])`. Stan lower-bound
  kernel semantics: `+log(tau)` Jacobian via the exp transform, NO
  truncation normalizer (NOT the thin layer's proper-half `+log(2)`).
- K=1 ⟹ plain only (`id` forces `:correlated`, which stays refused for
  Stage C). K=1 ⟹ exactly one slice `(T, 1:1)` ⟹ exactly one gather in T
  (Stage-A partition + linkage checks already enforce this).

## Thin-layer mapping

- Names (claimed at bucket lowering, suffix = group): `log_scale_<s>` /
  `tau_<suffix>` (scalar) + `xi_<suffix>` (G-vector). `tau` is a SCALAR,
  not SB's 1-vector (identical values, simpler layout; documented).
- Layout: scale rides `:sampled` (`:identity` for log_scale, `:exp` for
  tau — Jacobian `u = log(tau)` matches Stan); `xi` rides new kind
  `:ranef` (plate-mirror, `:identity`, width G from bind levels).
  Entries after spline, before scans.
- Encoder: in-graph indicator sum over bind-known `_grouping_levels`
  order, one named node per group `_ppl_gidx_<group>` (data-only,
  bound-folded; strings native-only, same as factors).
- Z: `:column` bare (raw or derived local), `:dummy` `(c .== level)`
  with `_level_literal`; `:ones` never materializes (intercept needs
  no Z multiply; slope1 margins are never `:ones`).
- Gather (SB-literal association, no `b` node): intercept
  `exp(log_scale) * xi[idx]`; slope `tau * (xi[idx] .* Z)`.
- Priors per bucket: scale scalar node (`normal(0,1)`, tau WITHOUT
  `+log(2)` — SB Stan-convention, documented loudly) + `xi` std-normal
  plate via the shared `_vector_prior_stmts!` helper.
- Design: `RanefGatherTerm` → 0-width block (`column` = group);
  generator reads TERMS for the full `(bucket_id, bucket_group)` key.
- Contract: K=1 name-collision checks (expected names ∩ union +
  mutual uniqueness); kind↔margin dispatch already enforced.
- Gate: `build_kernel` refuses `kind == :correlated` (Stage C owns
  LKJ packing); K=1 kinds emit.

## Validation / parity

- Committed: surface claims, contract mutants, bind/levels, layout
  entries/roundtrip/Jacobian, tps-style e2e vs independent SB-shape
  hand refs + Enzyme-vs-findiff, corpus 26/27 (intercept/slope).
- Joint: SB-value parity case offered to `rk:sb:ranef` (their StanBlocks
  side computes; thin layer matches bit-exact incl. the tau convention).
