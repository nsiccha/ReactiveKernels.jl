# Statistical model ownership

BayesianRegressionModels owns spline, HSGP, GP and varying-effect statistical
preparation and model bodies for both backends. RK-PPL compiles ordinary
model statements and does not ship a statistical model catalogue.

Obtain a BRM-owned body with `BayesianRegressionModels.rkppl_model`, bind it
in the defining module, or pass prepared data matrices to a hand-authored
model. Explicit coefficient priors, matrix products, truncated distributions,
array indexing and function calls retain their authored Julia semantics.
The [RK-PPL guide](rkppl.md) describes these general language capabilities.

Model-specific bases, spectral weights, defaults and scientific acceptance
belong with the BRM implementation. PK-specific models and helpers belong to
downstream RKPPLBench. Producer compatibility for a still-used historical API
is temporary and does not make that implementation a generic compiler primitive.

Import preparation helpers from
`BayesianRegressionModels.StatisticalPreparation` into the model module.
The compiler no longer carries spline, HSGP, monotonic, or DAR model IR.
Three HSGP helpers (`hsgp_basis`, `hsgp_sqrt_spd`, `hsgp_rho_floors`) and
manual varying-effect IR temporarily retain their existing behavior for a
published consumer that still imports them.
