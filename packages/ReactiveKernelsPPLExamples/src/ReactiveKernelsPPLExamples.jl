module ReactiveKernelsPPLExamples

using ReactiveKernels: KernelSpec, PreparedKernel

include("_ppl_source_authority.jl")
include("ppl_workflow.jl")
include("eight_schools.jl")
include("sum_to_zero.jl")
include("linear_regression.jl")
include("beta_binomial.jl")
include("poisson_gamma.jl")
include("dugongs_growth.jl")
include("arma11.jl")
include("gaussian_mixture.jl")
include("mnist_logistic.jl")

export PPLWorkflow

# EXPERIMENTAL, NOT REVIEWED / NOT APPROVED — do NOT build on this.
# `PPLMacro.@ppl` is an in-progress sb-like PPL front-end (first cut). It is
# deliberately NOT exported and NOT part of the consumer API (`reactivekernels-use`
# does not mention it), so no consuming agent picks it up and it introduces no
# churn while it is unreviewed. Reach it only via the fully qualified
# `ReactiveKernelsPPLExamples.PPLMacro.@ppl`. See the `PPLMacro` module docstring.
include("ppl_macro.jl")

export EightSchoolsExample
export SumToZeroExample
export LinearRegressionExample
export BetaBinomialExample
export PoissonGammaExample
export DugongsGrowthExample
export ARMA11Example
export GaussianMixtureExample
export MNISTLogisticExample

end # module ReactiveKernelsPPLExamples
