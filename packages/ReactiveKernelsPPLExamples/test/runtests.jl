using ReactiveKernels
using ReactiveKernelsPPLExamples
using ReactiveKernelsPPLExamples: EightSchoolsExample,
    SumToZeroExample,
    LinearRegressionExample, BetaBinomialExample, PoissonGammaExample,
    GLMPoissonExample, GLMBinomialExample, EightSchoolsNoncenteredExample,
    GLMMPoissonExample, BLRExample,
    MesquiteExample, LogmesquiteExample, LogmesquiteLogvolumeExample, KilpisjarviExample, EarnHeightExample, LogearnHeightExample, Log10earnHeightExample, LogearnInteractionExample, ARKExample, MhExample,
    NesLogitExample, WellsDistExample, WellsDist100Example, DogsLogExample, NESExample, KidscoreMomWorkExample, Rate1Example,
    RadonPooledExample, RadonPartiallyPooledCenteredExample, RadonPartiallyPooledNoncenteredExample, RadonVariableInterceptCenteredExample,
    KidscoreMomhsExample, KidscoreMomiqExample, KidscoreMomhsiqExample, KidscoreInteractionExample,
    DugongsGrowthExample, ARMA11Example, GaussianMixtureExample,
    MNISTLogisticExample, MVNormalRegressionExample, BoundRegressionExample
using Test

include("test_ppl_workflow.jl")
include("test_eight_schools_example.jl")
include("test_sum_to_zero_example.jl")
include("test_mnist_logistic_example.jl")
include("test_linear_regression_example.jl")
include("test_beta_binomial_example.jl")
include("test_poisson_gamma_example.jl")
include("test_glm_poisson_example.jl")
include("test_glm_binomial_example.jl")
include("test_eight_schools_noncentered_example.jl")
include("test_glmm_poisson_example.jl")
include("test_blr_example.jl")
include("test_mesquite_example.jl")
include("test_logmesquite_example.jl")
include("test_logmesquite_logvolume_example.jl")
include("test_kilpisjarvi_example.jl")
include("test_earn_height_example.jl")
include("test_logearn_height_example.jl")
include("test_log10earn_height_example.jl")
include("test_logearn_interaction_example.jl")
include("test_ark_example.jl")
include("test_mh_example.jl")
include("test_nes_logit_example.jl")
include("test_wells_dist_example.jl")
include("test_wells_dist100_example.jl")
include("test_dogs_log_example.jl")
include("test_nes_example.jl")
include("test_kidscore_mom_work_example.jl")
include("test_rate_1_example.jl")
include("test_radon_pooled_example.jl")
include("test_radon_partially_pooled_centered_example.jl")
include("test_radon_partially_pooled_noncentered_example.jl")
include("test_radon_variable_intercept_centered_example.jl")
include("test_kidscore_momhs_example.jl")
include("test_kidscore_momiq_example.jl")
include("test_kidscore_momhsiq_example.jl")
include("test_kidscore_interaction_example.jl")
include("test_dugongs_example.jl")
include("test_arma11_example.jl")
include("test_gaussian_mixture_example.jl")
include("test_mvnormal_regression_example.jl")
include("test_bound_regression_example.jl")
include("test_ppl_macro.jl")
include("test_ppl_gibbs.jl")
include("test_ppl_enzyme.jl")
include("test_ppl_docs_source_authority.jl")
