using ReactiveKernels
using Reactant
using Reactant: @compile
using Test
using ReactiveKernelsPPLExamples.LinearRegressionExample:
    evaluate_linear_regression_source
using ReactiveKernelsPPLExamples.ARMA11Example: evaluate_arma11_source
using ReactiveKernelsPPLExamples.PoissonGammaExample: evaluate_poisson_gamma_source
using ReactiveKernelsPPLExamples.GLMPoissonExample: evaluate_glm_poisson_source
using ReactiveKernelsPPLExamples.GLMBinomialExample: evaluate_glm_binomial_source
using ReactiveKernelsPPLExamples.EightSchoolsNoncenteredExample:
    evaluate_eight_schools_noncentered_source
using ReactiveKernelsPPLExamples.GLMMPoissonExample: evaluate_glmm_poisson_source
using ReactiveKernelsPPLExamples.BLRExample: evaluate_blr_source
using ReactiveKernelsPPLExamples.MesquiteExample: evaluate_mesquite_source
using ReactiveKernelsPPLExamples.LogmesquiteExample: evaluate_logmesquite_source
using ReactiveKernelsPPLExamples.LogmesquiteLogvolumeExample: evaluate_logmesquite_logvolume_source
using ReactiveKernelsPPLExamples.LogmesquiteLogvaExample: evaluate_logmesquite_logva_source
using ReactiveKernelsPPLExamples.LogmesquiteLogvasExample: evaluate_logmesquite_logvas_source
using ReactiveKernelsPPLExamples.LogmesquiteLogvashExample: evaluate_logmesquite_logvash_source
using ReactiveKernelsPPLExamples.KilpisjarviExample: evaluate_kilpisjarvi_source
using ReactiveKernelsPPLExamples.EarnHeightExample: evaluate_earn_height_source
using ReactiveKernelsPPLExamples.LogearnHeightExample: evaluate_logearn_height_source
using ReactiveKernelsPPLExamples.Log10earnHeightExample: evaluate_log10earn_height_source
using ReactiveKernelsPPLExamples.LogearnInteractionExample: evaluate_logearn_interaction_source
using ReactiveKernelsPPLExamples.LogearnHeightMaleExample: evaluate_logearn_height_male_source
using ReactiveKernelsPPLExamples.LogearnLogheightMaleExample: evaluate_logearn_logheight_male_source
using ReactiveKernelsPPLExamples.LogearnInteractionZExample: evaluate_logearn_interaction_z_source
using ReactiveKernelsPPLExamples.ARKExample: evaluate_ark_source
using ReactiveKernelsPPLExamples.MhExample: evaluate_mh_source
using ReactiveKernelsPPLExamples.NesLogitExample: evaluate_nes_logit_source
using ReactiveKernelsPPLExamples.WellsDistExample: evaluate_wells_dist_source
using ReactiveKernelsPPLExamples.WellsDist100Example: evaluate_wells_dist100_source
using ReactiveKernelsPPLExamples.DogsLogExample: evaluate_dogs_log_source
using ReactiveKernelsPPLExamples.NESExample: evaluate_nes_source
using ReactiveKernelsPPLExamples.KidscoreMomWorkExample: evaluate_kidscore_mom_work_source
using ReactiveKernelsPPLExamples.Rate1Example: evaluate_rate_1_source
using ReactiveKernelsPPLExamples.RadonPooledExample: evaluate_radon_pooled_source
using ReactiveKernelsPPLExamples.RadonPartiallyPooledCenteredExample: evaluate_radon_partially_pooled_centered_source
using ReactiveKernelsPPLExamples.RadonPartiallyPooledNoncenteredExample: evaluate_radon_partially_pooled_noncentered_source
using ReactiveKernelsPPLExamples.RadonVariableInterceptCenteredExample: evaluate_radon_variable_intercept_centered_source
using ReactiveKernelsPPLExamples.RadonCountyExample: evaluate_radon_county_source
using ReactiveKernelsPPLExamples.RadonCountyInterceptExample: evaluate_radon_county_intercept_source
using ReactiveKernelsPPLExamples.RadonVariableInterceptNoncenteredExample: evaluate_radon_variable_intercept_noncentered_source
using ReactiveKernelsPPLExamples.RadonVariableSlopeCenteredExample: evaluate_radon_variable_slope_centered_source
using ReactiveKernelsPPLExamples.RadonVariableSlopeNoncenteredExample: evaluate_radon_variable_slope_noncentered_source
using ReactiveKernelsPPLExamples.RadonVariableInterceptSlopeCenteredExample: evaluate_radon_variable_intercept_slope_centered_source
using ReactiveKernelsPPLExamples.RadonVariableInterceptSlopeNoncenteredExample: evaluate_radon_variable_intercept_slope_noncentered_source
using ReactiveKernelsPPLExamples.RadonHierarchicalInterceptCenteredExample: evaluate_radon_hierarchical_intercept_centered_source
using ReactiveKernelsPPLExamples.RadonHierarchicalInterceptNoncenteredExample: evaluate_radon_hierarchical_intercept_noncentered_source
using ReactiveKernelsPPLExamples.Rate2Example: evaluate_rate_2_source
using ReactiveKernelsPPLExamples.Rate3Example: evaluate_rate_3_source
using ReactiveKernelsPPLExamples.Rate4Example: evaluate_rate_4_source
using ReactiveKernelsPPLExamples.Rate5Example: evaluate_rate_5_source
using ReactiveKernelsPPLExamples.DogsExample: evaluate_dogs_source
using ReactiveKernelsPPLExamples.DogsHierarchicalExample: evaluate_dogs_hierarchical_source
using ReactiveKernelsPPLExamples.SeedsExample: evaluate_seeds_source
using ReactiveKernelsPPLExamples.SeedsCenteredExample: evaluate_seeds_centered_model_source
using ReactiveKernelsPPLExamples.SeedsStanifiedExample: evaluate_seeds_stanified_model_source
using ReactiveKernelsPPLExamples.RatsModelExample: evaluate_rats_model_source
using ReactiveKernelsPPLExamples.SesameOnePredAExample: evaluate_sesame_one_pred_a_source
using ReactiveKernelsPPLExamples.PilotsExample: evaluate_pilots_source
using ReactiveKernelsPPLExamples.LsatExample: evaluate_lsat_source
using ReactiveKernelsPPLExamples.SurgicalExample: evaluate_surgical_source
using ReactiveKernelsPPLExamples.M0Example: evaluate_m0_source
using ReactiveKernelsPPLExamples.MbExample: evaluate_mb_source
using ReactiveKernelsPPLExamples.MtExample: evaluate_mt_source
using ReactiveKernelsPPLExamples.WellsDaeExample: evaluate_wells_dae_source
using ReactiveKernelsPPLExamples.WellsDaeCExample: evaluate_wells_dae_c_source
using ReactiveKernelsPPLExamples.WellsInteractionExample: evaluate_wells_interaction_source
using ReactiveKernelsPPLExamples.WellsDaaeCExample: evaluate_wells_daae_c_source
using ReactiveKernelsPPLExamples.WellsInteractionCExample: evaluate_wells_interaction_c_source
using ReactiveKernelsPPLExamples.WellsDaeInterExample: evaluate_wells_dae_inter_source
using ReactiveKernelsPPLExamples.WellsDist100arsExample: evaluate_wells_dist100ars_source
using ReactiveKernelsPPLExamples.Election88FullExample: evaluate_election88_full_source
using ReactiveKernelsPPLExamples.KidscoreMomhsExample: evaluate_kidscore_momhs_source
using ReactiveKernelsPPLExamples.KidscoreMomiqExample: evaluate_kidscore_momiq_source
using ReactiveKernelsPPLExamples.KidscoreMomhsiqExample: evaluate_kidscore_momhsiq_source
using ReactiveKernelsPPLExamples.KidscoreInteractionExample: evaluate_kidscore_interaction_source
using ReactiveKernelsPPLExamples.KidscoreInteractionCExample: evaluate_kidscore_interaction_c_source
using ReactiveKernelsPPLExamples.KidscoreInteractionC2Example: evaluate_kidscore_interaction_c2_source
using ReactiveKernelsPPLExamples.KidscoreInteractionZExample: evaluate_kidscore_interaction_z_source
using ReactiveKernelsPPLExamples.BetaBinomialExample: evaluate_beta_binomial_source
using ReactiveKernelsPPLExamples.DugongsGrowthExample: evaluate_dugongs_source
using ReactiveKernelsPPLExamples.GaussianMixtureExample: evaluate_gaussian_mixture_source
using ReactiveKernelsPPLExamples.NormalMixtureExample: evaluate_normal_mixture_source
using ReactiveKernelsPPLExamples.LowDimGaussMixCollapseExample: evaluate_low_dim_gauss_mix_collapse_source
using ReactiveKernelsPPLExamples.LowDimGaussMixExample: evaluate_low_dim_gauss_mix_source
using ReactiveKernelsPPLExamples.MVNormalRegressionExample:
    build_mvnormal_regression_graph, MVREG_X, MVREG_Y,
    MVREG_COVARIANCE, MVREG_CHOL, MVREG_PRECISION, MVREG_PRECISION_CHOL
using ReactiveKernelsPPLExamples.BoundRegressionExample:
    build_bound_regression_graph, BOUND_RAW_X, BOUND_Y

_host(v::Reactant.AbstractConcreteArray) = Array(v)
_host(v::Reactant.AbstractConcreteNumber) = Reactant.to_number(v)
_host(v::NamedTuple) = NamedTuple{keys(v)}(map(_host, values(v)))
_host(v::Tuple) = map(_host, v)
_host(v) = v

_rapprox(a::Number, b::Number) = isapprox(a, b; rtol = 1e-6, atol = 1e-7)
_rapprox(a::AbstractArray, b::AbstractArray) =
    length(a) == length(b) && all(_rapprox(x, y) for (x, y) in zip(a, b))
_rapprox(a::Tuple, b::Tuple) =
    length(a) == length(b) && all(_rapprox(x, y) for (x, y) in zip(a, b))
_rapprox(a::NamedTuple, b::NamedTuple) = _rapprox(values(a), values(b))
_rapprox(a, b) = false

_trace(v) = v isa AbstractArray ? Reactant.to_rarray(v) :
            Reactant.to_rarray(v; track_numbers = true)

function _compile_run(kernel, inputs)
    traced = map(_trace, inputs)
    compiled = @compile sync = true kernel(traced...)
    _host(compiled(traced...))
end

# Each migrated / new PPL example compiles and executes through the public
# Reactant boundary and reproduces its native output. Object-splice densities
# (normal/cauchy/gamma/poisson/beta/binomial), authored plates, the sequential
# ARMA recursion, the MvNormal parametrizations, and bound partial evaluation are
# all exercised on the exact same authored graph the native path uses.
@testset "PPL examples compile through Reactant" begin
    @testset "linear_regression" begin
        a = evaluate_linear_regression_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "arma11 (vectorized closed form lowers; raw recursion stays native-only)" begin
        # Side-by-side: the model's likelihood/density reduce the vectorized
        # `errors_closed` (a Toeplitz matvec), which LOWERS and reproduces native.
        # The natural sequential `errors` node reads err[t-1]/series[t-1] element
        # by element, so it still does NOT lower (XLA disallows scalar indexing of
        # a traced array) — kept as an explicit tested diagnostic. A sequential-scan
        # (stablehlo.while) lowering would let the natural recursion lower directly;
        # see docs/src/arma11.md.
        a = evaluate_arma11_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
        seq_errors_kernel = prepare(a.model;
            have = (:unconstrained, :series), want = :errors)
        @test_throws Exception _compile_run(seq_errors_kernel, Tuple(a.inputs))
    end
    @testset "poisson_gamma" begin
        a = evaluate_poisson_gamma_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "glm_poisson (posteriordb; bounded-uniform transforms + poisson-log)" begin
        a = evaluate_glm_poisson_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "glm_binomial (posteriordb; binomial-logit GLM)" begin
        a = evaluate_glm_binomial_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "eight_schools_noncentered (posteriordb; non-centered + half-cauchy)" begin
        a = evaluate_eight_schools_noncentered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "glmm_poisson (posteriordb; hierarchical Poisson-log + random effects)" begin
        a = evaluate_glmm_poisson_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "blr (posteriordb; Bayesian linear regression, matvec)" begin
        a = evaluate_blr_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "mesquite (posteriordb; Gaussian linear regression)" begin
        a = evaluate_mesquite_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logmesquite (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logmesquite_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logmesquite_logvolume (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logmesquite_logvolume_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logmesquite_logva (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logmesquite_logva_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logmesquite_logvas (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logmesquite_logvas_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logmesquite_logvash (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logmesquite_logvash_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kilpisjarvi (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kilpisjarvi_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "earn_height (posteriordb; Gaussian linear regression)" begin
        a = evaluate_earn_height_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logearn_height (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logearn_height_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "log10earn_height (posteriordb; Gaussian linear regression)" begin
        a = evaluate_log10earn_height_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logearn_interaction (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logearn_interaction_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logearn_height_male (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logearn_height_male_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logearn_logheight_male (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logearn_logheight_male_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logearn_interaction_z (posteriordb; Gaussian linear regression)" begin
        a = evaluate_logearn_interaction_z_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "arK (posteriordb; AR(K) lag-matrix regression)" begin
        a = evaluate_ark_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "Mh (posteriordb; capture-recapture log_sum_exp, data-mask)" begin
        a = evaluate_mh_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "nes_logit (posteriordb)" begin
        a = evaluate_nes_logit_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_dist (posteriordb)" begin
        a = evaluate_wells_dist_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_dist100_model (posteriordb)" begin
        a = evaluate_wells_dist100_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "dogs_log (posteriordb)" begin
        a = evaluate_dogs_log_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "nes (posteriordb)" begin
        a = evaluate_nes_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_mom_work (posteriordb)" begin
        a = evaluate_kidscore_mom_work_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "Rate_1 (posteriordb; binomial rate)" begin
        a = evaluate_rate_1_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_pooled (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_pooled_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_partially_pooled_centered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_partially_pooled_centered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_partially_pooled_noncentered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_partially_pooled_noncentered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_variable_intercept_centered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_variable_intercept_centered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_county (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_county_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_county_intercept (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_county_intercept_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_variable_intercept_noncentered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_variable_intercept_noncentered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_variable_slope_centered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_variable_slope_centered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_variable_slope_noncentered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_variable_slope_noncentered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_variable_intercept_slope_centered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_variable_intercept_slope_centered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_variable_intercept_slope_noncentered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_variable_intercept_slope_noncentered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_hierarchical_intercept_centered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_hierarchical_intercept_centered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "radon_hierarchical_intercept_noncentered (posteriordb; hierarchical normal)" begin
        a = evaluate_radon_hierarchical_intercept_noncentered_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "rate_2 (posteriordb; binomial rate)" begin
        a = evaluate_rate_2_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "rate_3 (posteriordb; binomial rate)" begin
        a = evaluate_rate_3_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "rate_4 (posteriordb; binomial rate)" begin
        a = evaluate_rate_4_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "rate_5 (posteriordb; binomial rate)" begin
        a = evaluate_rate_5_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "dogs (posteriordb; bernoulli-logit)" begin
        a = evaluate_dogs_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "dogs_hierarchical (posteriordb)" begin
        a = evaluate_dogs_hierarchical_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "seeds (posteriordb)" begin
        a = evaluate_seeds_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "seeds_centered_model (posteriordb)" begin
        a = evaluate_seeds_centered_model_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "seeds_stanified_model (posteriordb)" begin
        a = evaluate_seeds_stanified_model_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "rats_model (posteriordb; hierarchical growth)" begin
        a = evaluate_rats_model_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "sesame_one_pred_a (posteriordb; linear regression)" begin
        a = evaluate_sesame_one_pred_a_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "pilots (posteriordb)" begin
        a = evaluate_pilots_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "lsat (posteriordb)" begin
        a = evaluate_lsat_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "surgical (posteriordb)" begin
        a = evaluate_surgical_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "m0 (posteriordb)" begin
        a = evaluate_m0_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "mb (posteriordb)" begin
        a = evaluate_mb_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "mt (posteriordb)" begin
        a = evaluate_mt_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_dae_model (posteriordb)" begin
        a = evaluate_wells_dae_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_dae_c_model (posteriordb)" begin
        a = evaluate_wells_dae_c_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_interaction_model (posteriordb)" begin
        a = evaluate_wells_interaction_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_daae_c_model (posteriordb)" begin
        a = evaluate_wells_daae_c_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_interaction_c_model (posteriordb)" begin
        a = evaluate_wells_interaction_c_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_dae_inter_model (posteriordb)" begin
        a = evaluate_wells_dae_inter_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "wells_dist100ars_model (posteriordb)" begin
        a = evaluate_wells_dist100ars_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "election88_full (posteriordb; hierarchical logistic)" begin
        a = evaluate_election88_full_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_momhs (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kidscore_momhs_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_momiq (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kidscore_momiq_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_momhsiq (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kidscore_momhsiq_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_interaction (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kidscore_interaction_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_interaction_c (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kidscore_interaction_c_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_interaction_c2 (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kidscore_interaction_c2_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kidscore_interaction_z (posteriordb; Gaussian linear regression)" begin
        a = evaluate_kidscore_interaction_z_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "beta_binomial" begin
        a = evaluate_beta_binomial_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "dugongs" begin
        a = evaluate_dugongs_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "normal_mixture (posteriordb; 2-component mixture)" begin
        a = evaluate_normal_mixture_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "low_dim_gauss_mix_collapse (posteriordb; 2-component mixture)" begin
        a = evaluate_low_dim_gauss_mix_collapse_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "low_dim_gauss_mix (posteriordb; 2-component mixture, ordered)" begin
        a = evaluate_low_dim_gauss_mix_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "gaussian_mixture" begin
        a = evaluate_gaussian_mixture_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "mvnormal_regression (per parametrization)" begin
        g = build_mvnormal_regression_graph()
        q = [0.5, 2.0, -1.0]
        for (port, value) in ((:covariance, MVREG_COVARIANCE),
                              (:chol, MVREG_CHOL),
                              (:precision, MVREG_PRECISION),
                              (:precision_chol, MVREG_PRECISION_CHOL))
            k = prepare(g;
                have = (:unconstrained, :predictors, :responses, port), want = :density)
            native = k(q, MVREG_X, MVREG_Y, value)
            @testset "$port" begin
                @test _rapprox(_compile_run(k, (q, MVREG_X, MVREG_Y, value)), native)
            end
        end
    end
    @testset "bound_regression" begin
        g = build_bound_regression_graph()
        q = [1.0, 2.0, -1.0, log(0.5)]
        unbound = prepare(g;
            have = (:unconstrained, :raw_predictors, :responses), want = :density)
        @test _rapprox(_compile_run(unbound, (q, BOUND_RAW_X, BOUND_Y)),
                       unbound(q, BOUND_RAW_X, BOUND_Y))
        bound = prepare(g;
            have = (:unconstrained, :raw_predictors, :responses), want = :density,
            bound = (; raw_predictors = BOUND_RAW_X))
        @test _rapprox(_compile_run(bound, (q, BOUND_Y)), bound(q, BOUND_Y))
    end
end
