module ReactiveKernelsPPLExamples

using ReactiveKernels: KernelSpec, PreparedKernel

include("_ppl_source_authority.jl")
include("ppl_workflow.jl")
include("eight_schools.jl")
include("sum_to_zero.jl")
include("linear_regression.jl")
include("beta_binomial.jl")
include("poisson_gamma.jl")
include("glm_poisson.jl")
include("glm_binomial.jl")
include("eight_schools_noncentered.jl")
include("glmm_poisson.jl")
include("kidscore_momhs.jl")
include("kidscore_momiq.jl")
include("kidscore_momhsiq.jl")
include("kidscore_interaction.jl")
include("blr.jl")
include("mesquite.jl")
include("logmesquite.jl")
include("logmesquite_logvolume.jl")
include("kilpisjarvi.jl")
include("earn_height.jl")
include("logearn_height.jl")
include("log10earn_height.jl")
include("logearn_interaction.jl")
include("ark.jl")
include("mh.jl")
include("nes_logit.jl")
include("wells_dist.jl")
include("wells_dist100_model.jl")
include("dogs_log.jl")
include("nes.jl")
include("kidscore_mom_work.jl")
include("rate_1.jl")
include("radon_pooled.jl")
include("radon_partially_pooled_centered.jl")
include("radon_partially_pooled_noncentered.jl")
include("radon_variable_intercept_centered.jl")
include("rate_2.jl")
include("rate_3.jl")
include("rate_4.jl")
include("rate_5.jl")
include("dugongs_growth.jl")
include("arma11.jl")
include("gaussian_mixture.jl")
include("mnist_logistic.jl")
include("mvnormal_regression.jl")
include("bound_regression.jl")

export PPLWorkflow

# EXPERIMENTAL, NOT REVIEWED / NOT APPROVED — do NOT build on this.
# `PPLMacro.@ppl` is an in-progress sb-like PPL front-end (first cut). It is
# deliberately NOT exported and NOT part of the consumer API (`reactivekernels-use`
# does not mention it), so no consuming agent picks it up and it introduces no
# churn while it is unreviewed. Reach it only via the fully qualified
# `ReactiveKernelsPPLExamples.PPLMacro.@ppl`. See the `PPLMacro` module docstring.
include("ppl_macro.jl")

# EXPERIMENTAL, NOT REVIEWED / NOT APPROVED — do NOT build on this. `PPLGibbs`
# is an in-progress PPL-style Gibbs layer over `@ppl` models (user decision C,
# handed off from ReactiveKernels:sampling:gibbs). Also NOT exported and NOT in
# the consumer API; reach it via `ReactiveKernelsPPLExamples.PPLGibbs`. Never in
# rk core. See the `PPLGibbs` module docstring.
include("ppl_gibbs.jl")

export EightSchoolsExample
export SumToZeroExample
export LinearRegressionExample
export BetaBinomialExample
export PoissonGammaExample
export GLMPoissonExample
export GLMBinomialExample
export EightSchoolsNoncenteredExample
export GLMMPoissonExample
export KidscoreMomhsExample
export KidscoreMomiqExample
export KidscoreMomhsiqExample
export KidscoreInteractionExample
export BLRExample
export MesquiteExample
export LogmesquiteExample
export LogmesquiteLogvolumeExample
export KilpisjarviExample
export EarnHeightExample
export LogearnHeightExample
export Log10earnHeightExample
export LogearnInteractionExample
export ARKExample
export MhExample
export NesLogitExample
export WellsDistExample
export WellsDist100Example
export DogsLogExample
export NESExample
export KidscoreMomWorkExample
export Rate1Example
export RadonPooledExample
export RadonPartiallyPooledCenteredExample
export RadonPartiallyPooledNoncenteredExample
export RadonVariableInterceptCenteredExample
export Rate2Example
export Rate3Example
export Rate4Example
export Rate5Example
export DugongsGrowthExample
export ARMA11Example
export GaussianMixtureExample
export MNISTLogisticExample
export MVNormalRegressionExample
export BoundRegressionExample

end # module ReactiveKernelsPPLExamples
