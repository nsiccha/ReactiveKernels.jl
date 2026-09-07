using ReactiveKernels
using ReactiveKernelsPPLExamples
using ReactiveKernelsPPLExamples: EightSchoolsExample,
    SumToZeroExample,
    LinearRegressionExample, BetaBinomialExample, PoissonGammaExample,
    GLMPoissonExample, GLMBinomialExample, EightSchoolsNoncenteredExample,
    GLMMPoissonExample, BLRExample,
    MesquiteExample, LogmesquiteExample, LogmesquiteLogvolumeExample, LogmesquiteLogvaExample, LogmesquiteLogvasExample, LogmesquiteLogvashExample, KilpisjarviExample, EarnHeightExample, LogearnHeightExample, Log10earnHeightExample, LogearnInteractionExample, LogearnHeightMaleExample, LogearnLogheightMaleExample, LogearnInteractionZExample, ARKExample, MhExample,
    NesLogitExample, WellsDistExample, WellsDist100Example, DogsLogExample, NESExample, KidscoreMomWorkExample, Rate1Example,
    RadonPooledExample, RadonPartiallyPooledCenteredExample, RadonPartiallyPooledNoncenteredExample, RadonVariableInterceptCenteredExample,
    RadonCountyExample, RadonCountyInterceptExample, RadonVariableInterceptNoncenteredExample, RadonVariableSlopeCenteredExample, RadonVariableSlopeNoncenteredExample,
    RadonVariableInterceptSlopeCenteredExample, RadonVariableInterceptSlopeNoncenteredExample, RadonHierarchicalInterceptCenteredExample, RadonHierarchicalInterceptNoncenteredExample,
    Rate2Example, Rate3Example, Rate4Example, Rate5Example,
    DogsExample, DogsHierarchicalExample,
    SeedsExample, PilotsExample, LsatExample, SurgicalExample,
    SeedsCenteredExample, SeedsStanifiedExample, RatsModelExample, SesameOnePredAExample,
    M0Example, MbExample, MtExample, MthModelExample, MtbhModelExample, WellsDaeExample, WellsDaeCExample, WellsInteractionExample, WellsDaaeCExample,
    WellsInteractionCExample, WellsDaeInterExample, WellsDist100arsExample, Election88FullExample,
    KidscoreMomhsExample, KidscoreMomiqExample, KidscoreMomhsiqExample, KidscoreInteractionExample,
    KidscoreInteractionCExample, KidscoreInteractionC2Example, KidscoreInteractionZExample,
    DugongsGrowthExample, ARMA11Example,
    NormalMixtureExample, LowDimGaussMixCollapseExample, LowDimGaussMixExample,
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
include("test_logmesquite_logva_example.jl")
include("test_logmesquite_logvas_example.jl")
include("test_logmesquite_logvash_example.jl")
include("test_logearn_height_male_example.jl")
include("test_logearn_logheight_male_example.jl")
include("test_logearn_interaction_z_example.jl")
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
include("test_radon_county_example.jl")
include("test_radon_county_intercept_example.jl")
include("test_radon_variable_intercept_noncentered_example.jl")
include("test_radon_variable_slope_centered_example.jl")
include("test_radon_variable_slope_noncentered_example.jl")
include("test_radon_variable_intercept_slope_centered_example.jl")
include("test_radon_variable_intercept_slope_noncentered_example.jl")
include("test_radon_hierarchical_intercept_centered_example.jl")
include("test_radon_hierarchical_intercept_noncentered_example.jl")
include("test_rate_2_example.jl")
include("test_rate_3_example.jl")
include("test_rate_4_example.jl")
include("test_rate_5_example.jl")
include("test_dogs_example.jl")
include("test_dogs_hierarchical_example.jl")
include("test_seeds_example.jl")
include("test_seeds_centered_model_example.jl")
include("test_seeds_stanified_model_example.jl")
include("test_rats_model_example.jl")
include("test_sesame_one_pred_a_example.jl")
include("test_pilots_example.jl")
include("test_lsat_example.jl")
include("test_surgical_example.jl")
include("test_m0_example.jl")
include("test_mb_example.jl")
include("test_mt_example.jl")
include("test_mth_model_example.jl")
include("test_mtbh_model_example.jl")
include("test_wells_dae_example.jl")
include("test_wells_dae_c_example.jl")
include("test_wells_interaction_example.jl")
include("test_wells_daae_c_example.jl")
include("test_wells_interaction_c_example.jl")
include("test_wells_dae_inter_example.jl")
include("test_wells_dist100ars_example.jl")
include("test_election88_full_example.jl")
include("test_kidscore_momhs_example.jl")
include("test_kidscore_momiq_example.jl")
include("test_kidscore_momhsiq_example.jl")
include("test_kidscore_interaction_example.jl")
include("test_kidscore_interaction_c_example.jl")
include("test_kidscore_interaction_c2_example.jl")
include("test_kidscore_interaction_z_example.jl")
include("test_dugongs_example.jl")
include("test_arma11_example.jl")
include("test_normal_mixture_example.jl")
include("test_low_dim_gauss_mix_collapse_example.jl")
include("test_low_dim_gauss_mix_example.jl")
include("test_survey_model_example.jl")
include("test_mvnormal_regression_example.jl")
include("test_bound_regression_example.jl")
include("test_ppl_macro.jl")
include("test_ppl_gibbs.jl")
include("test_ppl_enzyme.jl")
include("test_ppl_docs_source_authority.jl")
