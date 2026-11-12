# Changelog

## 0.1.8

- The Student-t log kernel declares its form at `log|z|` to ReactiveKernels'
  stable rewrites, so a half-Student or half-Cauchy prior at `exp(u)` reads
  `u` directly. Requires ReactiveKernels 0.1.204.

## 0.1.7

- Incomplete-beta and Student-t tails use ReactiveKernels' shared fraction
  value/partials graph. Reflection selects the fraction's arguments before
  its call, so batched JAX gradients avoid two reflected fraction evaluations
  and a reverse tape over the iterations. Endpoint guards remain lazy.
  Requires ReactiveKernels 0.1.202.

## 0.1.6

- Add `icar_rows_logdensity(phi, node1, node2)` and
  `icar_cols_logdensity(phi, node1, node2)`: independent unit-scale ICAR
  fields over one graph, the rows (field `k` is `phi[k, :]`) or the columns
  (field `k` is `phi[:, k]`) of a matrix. The density is the sum of
  `icar_logdensity` over the fields, computed in one plate over the edges
  with a plate over the fields inside each edge cell; no per-edge or
  per-field vector is filled.

## 0.1.5

- The hidden-Markov laws and their state-posterior twins' helpers record
  their sources (`ReactiveKernels.@_record_source`), so ReactiveKernels'
  `jax_source` (0.1.132) translates the laws. Native execution is unchanged;
  the helpers stay `@inline`. Requires ReactiveKernels 0.1.132.

## 0.1.4

- Add a state-posterior twin for each hidden Markov law, taking the law's own
  arguments: `hmm_state_posterior`, `normal_hmm_state_posterior` and
  `exponential_hmm_state_posterior`, and the log-weight twins
  `hmm_log_weights_state_posterior`, `normal_hmm_log_weights_state_posterior`
  and `exponential_hmm_log_weights_state_posterior`. Each returns the smoothed
  state probabilities (`probabilities[t, k]`), the filtered log probabilities
  (`log_filtered[t, k]`) and the `log_transition` weights the law reads.
  `hmm_state_posterior_name(law)` names a law's twin.
- Add `draw_hmm_states(rng, posterior)` and `draw_hmm_states!`: one exact draw
  of the state path from a twin's value (forward filtering, backward
  sampling), with `rng` as the only source of randomness.
- The laws and their twins share one forward-pass kernel and one log-emission
  kernel per family. Densities are unchanged.

## 0.1.3

- Add Gumbel, Rayleigh, Chi, Chi-squared, inverse-Gaussian, BetaPrime and F
  density/tail objects with shared parameter normalization. BetaPrime and F
  share log-domain beta tails that retain extreme odds without rounding away
  the smaller tail; inverse-Gaussian tails retain the scaled-normal difference.
- Add inverse-Gamma log tails, share Student-t normalization and log tails,
  and preserve Weibull's exact density limits at zero.

## 0.1.2

- Add lower- and upper-truncated LogNormal objects with zero-argument
  `mean()` and `var()` endpoints. Nonpositive lower bounds give ordinary
  moments. Stable tail arithmetic remains visible to binding and native AD.
- Add Gamma log-CDF and log-survival graphs with shared normalization and
  runtime series/continued-fraction loops. Preserve shape derivatives at
  shape one and the support boundary at zero.
- Add a DirichletMultinomial graph with shared concentration normalization
  and count-only coefficients evaluated at binding. Reject invalid counts
  and totals with `-Inf`, retaining dimension errors.
- Expose shared and data-only arithmetic in scalar family densities. Require
  RK 0.1.59 for Boolean negation in the new moment support guards.

## 0.1.1

- Add stable `logcdf` and `logccdf` endpoints for Normal, Cauchy, Laplace,
  Logistic, LogNormal, Exponential and Weibull. Normal and probit ordinal
  log tails use the scaled complementary error function through
  `SpecialFunctions.logerfc`, so positive masses remain representable in
  log space beyond probability-space underflow.
