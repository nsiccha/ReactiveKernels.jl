# HSGP thin-layer contract (design record, staged A/B)

Date: 2026-09-19. Lane todo: `/agents/ReactiveKernels:brm/todos/2026-09-19T09-27-44-303-11dsahf`
(spec: `BayesianRegressionModels:rk:sb:hsgp/decisions/2026-09-19T03-44-48-998-02e64eo`).
Staged like ranef: Stage A (IR + surface + admission, codegen fails
closed) lands first; Stage B (layout + codegen + parity) follows.

## SB mirror (verbatim shapes)

- Fit per axis (`preparation_hsgp.jl` `_brm_fit_hsgp`): `mu =
  mean(x)`, `L = c*max|x-mu|` (`c > 1`, `L > 0` or degenerate).
- 1D basis (`_brm_apply_hsgp`): `lambda_k = (k*pi/(2L))^2`,
  `PHI[i,k] = sin(sqrt(lambda_k)*(x[i]-mu+L))/sqrt(L)`, `k=1..K`.
- Tensor product over axes in `CartesianIndices(K)` order (basis-major
  `b`, `omega2[b,axis]`); `M = prod(K)`.
- Floor (`_brm_hsgp_rho_lower`): `(4L/pi)*sqrt(log(100)/(K^2-1))`
  per axis (`K=1` → `0.0`, unbounded); iso takes the max.
- `_sb_hsgp`: `rho_iso ~ lognormal(0,1; lower=rho_lower)`, `sigma ~
  lognormal(0,1; lower=0)`, `beta_raw ~ std_normal(; n=M)`,
  summand `PHI * (sqrt_spd .* beta_raw)`; `brm_hsgp_sqrt_spd`:
  `scale = sigma*prod(sqrt(rho*2.5066282746310002))`,
  `rv[b] = scale*exp(-0.25*sum(rho^2*omega2[b,:]))`.
- Aniso: `rho :: vector[d]` with per-axis floors. Defaults: `k=20`,
  `c=1.5`, `iso=true` (scalar broadcasts per axis).

## Thin-layer mapping

- IR (contract): `HSGPBasis(id, axes, K::Vector{Int},
  c::Vector{Float64}, iso::Bool, fits, label)` — fits `(mu,L)` per
  axis, filled at bind (empty pre-bind; spline-bind precedent). New
  `TermKind HSGPSummandTerm` (0-width design block; use-site
  `hsgp(:id)`). Term OWNS its params (ranef-bucket precedent):
  `beta_raw_<id>` (M-vector), iso `rho_<id>` scalar, aniso
  `rho_<id>_1..d` scalars (d scalars, NOT a floors-vector — no new
  vector machinery), `sigma_<id>` scalar. Single source
  `_hsgp_names`. `StructuralPlan` gains `hsgp_bases`.
- Surface: `hsgp_basis(:id, x...; k=..., c=..., iso=...)` bare-call
  decl (spline_basis mirror: quoted id, bare raw axes, literal
  k/c with SB defaults 20/1.5, `iso=true`; scalar-or-tuple
  per-axis) + `hsgp(:id)` summand (spline-use mirror). Claims
  basis label, use labels, param names up front.
- Admission: axes are data; k positive ints; c > 1; iso Bool;
  bind fails closed on degenerate axes (`L == 0`); every basis is
  used at least once (no dead parameters — ranef-linkage mirror).
- Bind (host-side transformed-data mirror — spline precedent):
  fit `(mu,L)` per axis from raw columns; floors iso/aniso
  (`K=1` → `0.0`). Basis EVALUATION stays in-graph (trig is
  elementwise-expressible — unlike spline eigen).
- Layout (Stage B): beta M-vector (new kind `:hsgp`, identity —
  mirrors `:spline`/`:ranef` arms); sigma rides `:exp`; rho
  rides new parameterized `:floored` support (`x = lo+exp(u)`,
  logjac `u` — Stan lower-bound KERNEL semantics, NO truncation
  normalizer, exactly like Stage-B tau; floor `0.0` routes to
  `:exp`, bit-identical). Entry order per basis (SB declaration
  order): rho, sigma, beta.
- Generator (Stage B, spec-literal matmul): per-axis 1D columns
  from frozen literals (`lam_sqrt`, `shift=L-mu`, `inv_sqrt_L`),
  tensor-product columns by elementwise products,
  `PHI = hcat(cols...)`, `S = [s_1..s_M]` vect of unrolled
  `sqrt_spd` scalars, `w = S .* beta`, summand `PHI * w`.
  Priors: rho/sigma lognormal(0,1) nodes (no normalizer), beta
  std-normal plate. RISK: `Expr(:vect)` lowering in-graph —
  verify early in Stage B (fallback: dotted sum, no PHI).
- Deferred (spec + this lane): `by=`, periodic, latent-axis,
  partial centering, hyper-predictors, prior overrides
  (floor-zeroing needs a surface design — follow-up).

## Validation / parity

- Stage A: lowering tests (shapes/defaults/fail-closed), bind
  fit/floor tests, corpus 1D + aniso, build_kernel gate test.
- Stage B: floored roundtrip/Jacobian, e2e vs independent hand
  refs (SB-shape loops + Distributions) + Enzyme-vs-findiff,
  joint SB parity case at 1e-12 (flop-order), land + notify BRM.
