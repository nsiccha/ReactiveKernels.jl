# Ranef Stage C: LKJ-correlated codegen + parity (design record)

Date: 2026-09-19. Lane todo: `/agents/ReactiveKernels:brm/todos/2026-09-19T03-23-10-620-0o9gqpa`
(spec: `BayesianRegressionModels:rk:sb:ranef/decisions/2026-09-19T03-19-04-749-0mlx9gh`).
Stage A (bucket IR + surface + lowering, no codegen) landed in `0cde428`;
Stage B (K=1 codegen + parity) landed in `58e1a9c` / merge `6cd58bc4`.

## SB mirror (verbatim shapes, `sbimpl.jl` `ranef_correlated_draws`)

- `L ~ lkj_corr_cholesky(eta; n=K)`, `tau ~ std_normal(; n=K, lower=0)`,
  `z_flat ~ std_normal(; n=K*G)`, `z = to_matrix(z_flat, K, G)`
  (column-major: `z[t,g] = z_flat[t + (g-1)*K]`), draws
  `b = (diag_pre_multiply(tau,L)*z)'` (GxK). Per-slice:
  `r = b[idx,cols]` / `rows_dot_product(Z, b[idx,col])`.
- ID buckets take NO K=1 fast path (K=1 ID is `:correlated` with a
  1x1 `L = [1]`, LKJ term exactly 0).

## Thin-layer mapping (deviations from SB are thin-layer-owned per contract)

- Names (claimed at bucket lowering): `L_<s>` / `tau_<s>` (K-vector) /
  `z_flat_<s>` (SB vocabulary, bucket-qualified).
- Layout per correlated bucket in plan bucket order (SB declaration
  order L/tau/z): `L` is a new `:ranef_corr` kind with `:lkj`
  transform packing K(K-1)/2 thetas (K derived from size; K=1 packs
  zero and constrains to `[1.0]`); `tau` rides `:ranef` with `:exp`
  (Stan lower-bound kernel semantics, NO `+log(2)` — Stage-B
  convention extended); `z_flat` rides `:ranef` `:identity` in SB
  column-major order.
- L PARAMETERIZATION IS OURS, NOT STAN'S (contract: middle layer owns
  layout+transforms): hyperspherical rows — row 1 is `[1,0,...]`,
  row i>=2 is a unit vector from i-1 logistic angles
  `theta = pi*sigma(u)`, `L[i,j] = cos(theta_j)* prod sin`,
  `L[i,i] = prod sin`. Log-Jacobian = per-angle Gram factor
  `(i-1-j)*log(sin theta)` (hyperspherical volume element) +
  logistic `log pi + log s + log1p(-s)`. Posterior parity with SB
  holds IN DISTRIBUTION (same posterior, different coords); value
  parity is measured at CONSTRAINED values, where the LKJ lpdf
  (diagonal sum + Stan's `do_lkj_constant` ported verbatim, both
  `eta==1.0` and general branches) matches Stan bit-exact.
- Gather (SB-literal association, no `b` node — Stage-B precedent,
  draws stay implicit): per slice margin j, `Z_j .* sum_s
  (tau[j]*L[j,s]) .* z_flat[s + (gidx-1)*K]` over `s in 1:j`
  (L lower-triangular); `:ones` Z drops the factor. All K, slice
  ranges static; tau reads are scalar refs (coefficient precedent).
- Priors: LKJ node (diagonal sum + host-computed constant literal;
  K=1 emits `0.0`); tau/z plates via the shared vector-prior helper
  with NO support correction (Stan kernel semantics).
- `constrain` returns `L` (KxK), `tau`, `z_flat`, plus DERIVED
  `b_<s>` (GxK draws — the posterior quantity; ignored by
  `unconstrain`).

## Validation / parity

- Committed: surface claims, contract mutants, layout entries +
  roundtrip, LKJ Gram-factor vs finite-difference 1/2 logdet(J'J),
  constant vs K=2 Beta-integral closed form + hand numbers,
  K=2 e2e vs independent hand ref (Beta-form LKJ, hand Gram) +
  Enzyme-vs-findiff, K=1-ID e2e, corpus 28 (multi-slice ID).
- Joint: SB-value parity case offered to `rk:sb:ranef` at 1e-12
  (NOT bit-exact — gather flop order differs from Stan's matmul).
