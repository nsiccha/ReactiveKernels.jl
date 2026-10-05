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
downstream RKPPLBench.

The PosteriorDB `one_comp_mm_elim_abs` model, full observation fixture and
scientific oracle are maintained in RKPPLBench under
`examples/posteriordb_pk/`. Its former RK-PPL example module and executable
walkthrough are removed after that downstream adoption. General ODE and
compiler machinery remain in ReactiveKernels.

Import preparation helpers from
`BayesianRegressionModels.StatisticalPreparation` into the model module.
The compiler no longer carries spline, HSGP, monotonic, DAR or varying-effect
model IR. Varying effects are ordinary declared arrays and level gathers, or a
BRM-owned `rkppl_model` body.
