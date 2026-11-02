# Changelog

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
  RK 0.1.58 for Boolean negation in the new moment support guards.

## 0.1.1

- Add stable `logcdf` and `logccdf` endpoints for Normal, Cauchy, Laplace,
  Logistic, LogNormal, Exponential and Weibull. Normal and probit ordinal
  log tails use the scaled complementary error function through
  `SpecialFunctions.logerfc`, so positive masses remain representable in
  log space beyond probability-space underflow.
