using ReactiveKernels
using ReactiveKernelsPPLExamples
using ReactiveKernelsPPLExamples: EightSchoolsExample,
    SumToZeroExample,
    LinearRegressionExample, BetaBinomialExample, PoissonGammaExample,
    DugongsGrowthExample, ARMA11Example, GaussianMixtureExample,
    MNISTLogisticExample
using Test

include("test_eight_schools_example.jl")
include("test_sum_to_zero_example.jl")
include("test_mnist_logistic_example.jl")
include("test_linear_regression_example.jl")
include("test_beta_binomial_example.jl")
include("test_poisson_gamma_example.jl")
include("test_dugongs_example.jl")
include("test_arma11_example.jl")
include("test_gaussian_mixture_example.jl")
include("test_ppl_enzyme.jl")
include("test_ppl_docs_source_authority.jl")
