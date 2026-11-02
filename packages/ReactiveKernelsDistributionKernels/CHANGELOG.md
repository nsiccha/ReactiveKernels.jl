# Changelog

## 0.1.1

- Add stable `logcdf` and `logccdf` endpoints for Normal, Cauchy, Laplace,
  Logistic, LogNormal, Exponential and Weibull. Normal and probit ordinal
  log tails use the scaled complementary error function through
  `SpecialFunctions.logerfc`, so positive masses remain representable in
  log space beyond probability-space underflow.
