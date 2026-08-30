module ReactiveKernelsPPLExamples

using ReactiveKernels: KernelSpec, PreparedKernel

include("_ppl_source_authority.jl")
include("eight_schools.jl")
include("linear_regression.jl")
include("beta_binomial.jl")
include("poisson_gamma.jl")
include("dugongs_growth.jl")
include("arma11.jl")
include("gaussian_mixture.jl")

export EightSchoolsExample
export LinearRegressionExample
export BetaBinomialExample
export PoissonGammaExample
export DugongsGrowthExample
export ARMA11Example
export GaussianMixtureExample

end # module ReactiveKernelsPPLExamples
