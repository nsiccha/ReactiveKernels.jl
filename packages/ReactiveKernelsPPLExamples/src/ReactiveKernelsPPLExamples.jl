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

# RK-native sb-like PPL front-end (first cut): `@ppl` lowers a declarative `~`
# model to a `KernelSpec` exposing the canonical PPL workflow node set.
include("ppl_macro.jl")
using .PPLMacro: @ppl

export EightSchoolsExample
export SumToZeroExample
export LinearRegressionExample
export BetaBinomialExample
export PoissonGammaExample
export DugongsGrowthExample
export ARMA11Example
export GaussianMixtureExample
export MNISTLogisticExample
export PPLMacro, @ppl

end # module ReactiveKernelsPPLExamples
