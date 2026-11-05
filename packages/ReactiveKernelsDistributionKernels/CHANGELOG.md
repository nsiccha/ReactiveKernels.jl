# Changelog

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
