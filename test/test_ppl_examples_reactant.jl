using ReactiveKernels
using Reactant
using Reactant: @compile
using Test
using ReactiveKernelsPPLExamples.LinearRegressionExample:
    evaluate_linear_regression_source
using ReactiveKernelsPPLExamples.ARMA11Example: evaluate_arma11_source
using ReactiveKernelsPPLExamples.GARCH11Example: evaluate_garch11_source
using ReactiveKernelsPPLExamples.HmmExampleExample: evaluate_hmm_example_source
using ReactiveKernelsPPLExamples.HmmGaussianExample: evaluate_hmm_gaussian_source
using ReactiveKernelsPPLExamples.IohmmRegExample: evaluate_iohmm_reg_source
using ReactiveKernelsPPLExamples.PoissonGammaExample: evaluate_poisson_gamma_source
using ReactiveKernelsPPLExamples.GLMPoissonExample: evaluate_glm_poisson_source,
    build_glm_poisson_graph, GLM_POISSON_YEAR, GLM_POISSON_C
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
using ReactiveKernelsPPLExamples.MthModelExample: evaluate_mth_model_source
using ReactiveKernelsPPLExamples.MtbhModelExample: evaluate_mtbh_model_source
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
using ReactiveKernelsPPLExamples.NormalMixtureExample: evaluate_normal_mixture_source
using ReactiveKernelsPPLExamples.LowDimGaussMixCollapseExample: evaluate_low_dim_gauss_mix_collapse_source
using ReactiveKernelsPPLExamples.LowDimGaussMixExample: evaluate_low_dim_gauss_mix_source
using ReactiveKernelsPPLExamples.SurveyModelExample: evaluate_survey_model_source
using ReactiveKernelsPPLExamples.MVNormalRegressionExample:
    build_mvnormal_regression_graph, MVREG_X, MVREG_Y,
    MVREG_COVARIANCE, MVREG_CHOL, MVREG_PRECISION, MVREG_PRECISION_CHOL
using ReactiveKernelsPPLExamples.BoundRegressionExample:
    build_bound_regression_graph, BOUND_RAW_X, BOUND_Y
using ReactiveKernelsPPLExamples.DiamondsExample: evaluate_diamonds_source
using ReactiveKernelsPPLExamples.NormalMixtureKExample: evaluate_normal_mixture_k_source
using ReactiveKernelsPPLExamples.DogsNonhierarchicalExample: evaluate_dogs_nonhierarchical_source
using ReactiveKernelsPPLExamples.LogisticRegressionRHSExample: evaluate_logistic_regression_rhs_source
using ReactiveKernelsPPLExamples.GPRegrExample:
    evaluate_gp_regr_source, GP_REGR_X, GP_REGR_Y
using ReactiveKernelsPPLExamples.AccelGPExample: evaluate_accel_gp_source
using ReactiveKernelsPPLExamples.GPPoisRegrExample: evaluate_gp_pois_regr_source
using ReactiveKernelsPPLExamples.HierarchicalGPExample: evaluate_hierarchical_gp_source,
    HGP_Y, HGP_YEAR_IND, HGP_STATE_IND, HGP_REGION_IND, HGP_STATE_REGION_IND,
    HGP_N_YEARS, HGP_N_REGIONS, HGP_N_STATES, HGP_N_YEARS_OBS
using ReactiveKernelsPPLExamples.HmmDrive0Example: evaluate_hmm_drive_0_source,
    HMM_DRIVE_0_U, HMM_DRIVE_0_V, HMM_DRIVE_0_ALPHA
using ReactiveKernelsPPLExamples.HmmDrive1Example: evaluate_hmm_drive_1_source,
    HMM_DRIVE_1_U, HMM_DRIVE_1_V, HMM_DRIVE_1_ALPHA, HMM_DRIVE_1_TAU, HMM_DRIVE_1_RHO
using ReactiveKernelsPPLExamples.HmmExampleExample: HMM_EXAMPLE_Y, HMM_EXAMPLE_K
using ReactiveKernelsPPLExamples.HmmGaussianExample: HMM_GAUSSIAN_Y, HMM_GAUSSIAN_K
using ReactiveKernelsPPLExamples.IohmmRegExample: IOHMM_REG_Y, IOHMM_REG_U, IOHMM_REG_K
using ReactiveKernelsPPLExamples.MtExample: MT_Y, MT_S, MT_T, MT_M
using ReactiveKernelsPPLExamples.MthModelExample: MTH_Y, MTH_S, MTH_T, MTH_M
using ReactiveKernelsPPLExamples.MtbhModelExample: MTBH_Y, MTBH_YPREV, MTBH_S,
    MTBH_T, MTBH_M
using ReactiveKernelsPPLExamples.MultiOccupancyExample: evaluate_multi_occupancy_source,
    MULTI_OCC_X, MULTI_OCC_N, MULTI_OCC_J, MULTI_OCC_K
using ReactiveKernelsPPLExamples.KroneckerGpExample: evaluate_kronecker_gp_source,
    KRON_X1, KRON_Y
import Enzyme
using DifferentiationInterface: AutoEnzyme

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

# Pinned XLA gaps (signature-checked; any other failure rethrows loudly).
#
# kronecker_gp: the LKJ-Cholesky helpers are closed-form vectorized (snag
# kronecker-gp-rea-dc8cef6f), so the trace reaches the stacked upstream
# Reactant gap: `eigen(Symmetric(::TracedRArray))` dies during tracing in
# `LinearAlgebra.isdiag(::Symmetric)` → Reactant `isbanded`/`_istril` →
# `MethodError: no method matching overloaded_triu(::UpperTriangular{
# TracedRNumber{Float64}, TracedRArray{Float64, 2}}, ::Int64)` (Reactant
# src/stdlibs/LinearAlgebra.jl:366 defines it for `TracedRArray{T, 2}` only;
# the same missing method is upstream EnzymeAD/Reactant.jl#3369 via symmetric
# solve, with no traced eigen/eigvals primal behind it — reactivekernels-use
# §7aa). Measured Reactant 0.2.289, Julia 1.10.11.
_xla_is_kron_eigen_gap(e) =
    e isa MethodError && (msg = sprint(showerror, e);
        occursin("no method matching overloaded_triu", msg) &&
        occursin("UpperTriangular", msg))
# hierarchical_gp compiled reverse: EnzymeMLIR has no adjoint for
# `stablehlo.cholesky` (reactivekernels-use §7f; snag reactant-compile-f877fcfd):
# `Reactant.Compiler.CompilationError: MLIR pass pipeline "all" failed …
# could not compute the adjoint for this operation … "stablehlo.cholesky"`.
_xla_is_cholesky_adjoint_gap(e) =
    e isa Reactant.Compiler.CompilationError && (msg = sprint(showerror, e);
        occursin("could not compute the adjoint for this operation", msg) &&
        occursin("stablehlo.cholesky", msg))

# kronecker_gp LKJ-Cholesky helpers (`_ccl_constraint_lp`,
# `_cholesky_corr_constrain_L`) compiled on a traced position. The sandbox
# methods are defined by the source evaluation, so callers cross the
# world-age barrier with `Base.invokelatest`.
function _kron_lkj_helpers_compiled(sandbox, z, K)
    lp_fn = getfield(sandbox, :_ccl_constraint_lp)
    L_fn = getfield(sandbox, :_cholesky_corr_constrain_L)
    rz = Reactant.to_rarray(z)
    clp = @compile sync = true lp_fn(rz, K)
    cL = @compile sync = true L_fn(rz, K)
    (_host(clp(rz, K)), _host(cL(rz, K)))
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
    @testset "arma11 (natural sequential recursion lowers via scan → stablehlo.while)" begin
        # The likelihood/density reduce the NATURAL sequential `errors`, authored
        # with the `scan` primitive, which lowers the one-step-ahead recurrence to
        # a `stablehlo.while` carry loop (no per-step scalar indexing of a traced
        # array) and reproduces native. `errors_closed` (a vectorized Toeplitz
        # matvec) is kept as an independent numerical cross-check. This flips the
        # former diagnostic (the raw `for`/`err[t-1]` recursion did NOT lower);
        # see docs/src/arma11.md.
        a = evaluate_arma11_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)

        # The natural sequential `errors` now LOWERS through Reactant (was a
        # tested `@test_throws` diagnostic before the scan primitive) and matches
        # native.
        seq_errors_kernel = prepare(a.model;
            have = (:unconstrained, :series), want = :errors)
        native_errors = seq_errors_kernel(a.inputs.q, a.inputs.series)
        @test _rapprox(
            _compile_run(seq_errors_kernel, Tuple(a.inputs)), native_errors)

        # It lowers to a stablehlo.while carry loop rather than unrolling.
        traced = map(_trace, Tuple(a.inputs))
        hlo = repr(Reactant.@code_hlo optimize = false seq_errors_kernel(traced...))
        @test occursin("stablehlo.while", hlo)

        # Independent cross-check: the scan errors equal the vectorized closed form.
        both_kernel = prepare(a.model;
            have = (:unconstrained, :series), want = (:errors, :errors_closed))
        e_scan, e_closed = both_kernel(a.inputs.q, a.inputs.series)
        @test _rapprox(e_scan, e_closed)
    end
    @testset "garch11 (GARCH(1,1) sd recursion; traced raw series → stablehlo.while)" begin
        a = evaluate_garch11_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
        traced = map(_trace, Tuple(a.inputs))
        hlo = repr(Reactant.@code_hlo optimize = false a.kernel(traced...))
        @test occursin("stablehlo.while", hlo)
    end
    @testset "hmm_example (forward algorithm; traced raw series → stablehlo.while)" begin
        a = evaluate_hmm_example_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
        traced = map(_trace, Tuple(a.inputs))
        hlo = repr(Reactant.@code_hlo optimize = false a.kernel(traced...))
        @test occursin("stablehlo.while", hlo)
    end
    @testset "hmm_gaussian (K-state forward; traced raw series → stablehlo.while)" begin
        a = evaluate_hmm_gaussian_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
        traced = map(_trace, Tuple(a.inputs))
        hlo = repr(Reactant.@code_hlo optimize = false a.kernel(traced...))
        @test occursin("stablehlo.while", hlo)
    end
    @testset "iohmm_reg (input-dependent forward; all-bound query compiles, scan unrolls)" begin
        a = evaluate_iohmm_reg_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
        # KNOWN GAP (do not assert while here): this model's per-step scan
        # inputs are K-vectors (rows of the input-dependent transition /
        # emission design), and the scan while-lowering gathers only 1-D
        # traced sequences. Each traced raw-series variant still unrolls
        # (measured HLO without `stablehlo.while`, ~7 MB on the full data),
        # and iterating a traced matrix directly fails Reactant scalar
        # indexing. The all-bound query stays the natural authoring; parity is
        # asserted above on the unrolled compiled program.
    end
    @testset "hmm_drive_0 (posteriordb; exponential-emission HMM, all data bound)" begin
        a = evaluate_hmm_drive_0_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "hmm_drive_1 (posteriordb; Normal-emission HMM, all data bound)" begin
        a = evaluate_hmm_drive_1_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
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
    @testset "mth_model (posteriordb; time + heterogeneity)" begin
        a = evaluate_mth_model_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "mtbh_model (posteriordb; time + behaviour + heterogeneity)" begin
        a = evaluate_mtbh_model_source()
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
    # A scalar parameter (`logit_rate`) with every array port bound, so the only
    # traced argument is a `TracedRNumber`: the plate's data lives entirely in
    # bound host arrays. Compiles to the tensorized body (data-only support mask
    # materialized as `Array{Bool}`, mixed host/traced ops promoted to a concrete
    # traced eltype) rather than the native host-buffer loop.
    @testset "beta_binomial (scalar param, all array data bound)" begin
        a = evaluate_beta_binomial_source()
        kb = prepare(a.model;
            have = (:logit_rate, :trials, :successes), want = a.requested_nodes,
            bound = (; trials = a.inputs.trials, successes = a.inputs.successes))
        native = kb(a.inputs.logit_rate)
        @test _rapprox(_compile_run(kb, (a.inputs.logit_rate,)), native)
    end
    @testset "poisson_gamma (scalar param, all array data bound)" begin
        a = evaluate_poisson_gamma_source()
        kb = prepare(a.model;
            have = (:log_rate, :counts), want = a.requested_nodes,
            bound = (; counts = a.inputs.counts))
        native = kb(a.inputs.log_rate)
        @test _rapprox(_compile_run(kb, (a.inputs.log_rate,)), native)
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
    @testset "Survey_model (posteriordb; discrete-n marginalization)" begin
        a = evaluate_survey_model_source()
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
    @testset "diamonds (posteriordb; brms centered regression, bound design)" begin
        a = evaluate_diamonds_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "normal_mixture_k (posteriordb; natural K-dim simplex mixture, log_sum_exp)" begin
        a = evaluate_normal_mixture_k_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "dogs_nonhierarchical (posteriordb; correlated per-dog, in-graph counts)" begin
        a = evaluate_dogs_nonhierarchical_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "logistic_regression_rhs (posteriordb; regularized horseshoe)" begin
        a = evaluate_logistic_regression_rhs_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "glm_poisson with bound host data (40-lane plate)" begin
        # The benchmark shape: every data port BOUND as a host array, only the
        # unconstrained vector traced.  `counts` is then a host `Vector{Int}`
        # inside the 40-lane likelihood plate (every traced plate keeps the
        # batched/broadcast lowering whatever its lane count), and the Poisson
        # `logpdf` cell guards validity with an authored branch on `observed`.
        # Reactant deduces a broadcast eltype from the RAW host element type
        # before promoting operands, so a guard on a host `Bool` once inferred
        # `Union{Float64,TracedRNumber{Float64}}` — typejoin `Number`, for
        # which Reactant has no traced `similar` — and the plate failed to
        # trace.  The extension promotes host array operands of a vector plate
        # before broadcasting, so the bound graph reproduces native.  The
        # example's design matrix is the cubic trend of `year`.
        model = build_glm_poisson_graph()
        q = [0.2, 0.1, -0.05, 0.03]
        year = GLM_POISSON_YEAR
        X = hcat(ones(length(year)), year, year .^ 2, year .^ 3)
        have = (:unconstrained, :X, :counts)
        all_bound = prepare(model; have, want = :posterior,
            bound = (; X, counts = GLM_POISSON_C))
        @test _rapprox(_compile_run(all_bound, (q,)), all_bound(q))
        counts_bound = prepare(model; have, want = :posterior,
            bound = (; counts = GLM_POISSON_C))
        @test _rapprox(_compile_run(counts_bound, (q, X)), counts_bound(q, X))
    end
    # gp_regr — marginal GP regression with a dense in-graph Cholesky of the
    # exponential-quadratic covariance. The PRIMAL lowers through Reactant with
    # the response `y` traced (the mvnormal-cholesky pattern; `x` is bound so the
    # squared-distance design folds to a constant). The compiled REVERSE gradient
    # of the Cholesky solve does NOT yet lower — the Reactant/EnzymeMLIR pass has
    # no adjoint for `stablehlo.triangular_solve` — so only the primal is checked
    # here (native primal + native Enzyme gradient vs Stan are the authoritative
    # gate in `benchmark/gp_gate.jl`; the Reactant-gradient gap is snag
    # `reactant-compile-f877fcfd` on ReactiveKernels).
    @testset "gp_regr (posteriordb; marginal GP, in-graph cholesky; primal)" begin
        a = evaluate_gp_regr_source()
        q = a.inputs.q
        kb = prepare(a.model; have = (:unconstrained, :x, :y), want = :posterior,
                     bound = (; x = GP_REGR_X))
        native = kb(q, GP_REGR_Y)
        @test _rapprox(_compile_run(kb, (q, GP_REGR_Y)), native)
    end
    # accel_gp — brms Hilbert-space approximate GP (HSGP): a distributional model
    # (latent GP on both the mean and log-sd of a Normal response) built entirely
    # from dense matrix-vector products, no covariance matrix / Cholesky. All data
    # is bound (raw-data bound entry); the whole graph — primal AND compiled
    # reverse gradient — lowers through Reactant (see benchmark/gp_gate.jl axis 4).
    @testset "accel_gp (posteriordb; HSGP distributional GP, all data bound)" begin
        a = evaluate_accel_gp_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    # gp_pois_regr — non-centered latent GP + Poisson-log. The latent field is
    # `f = cholesky(K).L * f_tilde`; the PRIMAL lowers through Reactant (`x`/`k`
    # bound, only q traced). Its compiled REVERSE gradient does NOT lower — the
    # Reactant/EnzymeMLIR pass has no adjoint for the Cholesky FACTOR
    # (`stablehlo.cholesky`), the sibling of gp_regr's `triangular_solve` gap
    # (both snag reactant-compile-f877fcfd). Native primal + native Enzyme
    # gradient vs Stan are the authoritative gate in `benchmark/gp_gate.jl`.
    @testset "gp_pois_regr (posteriordb; latent GP + Poisson-log; primal)" begin
        a = evaluate_gp_pois_regr_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    # hierarchical_gp — hierarchical GP of state presidential votes (dim 933): the
    # ILR simplex transform, dual per-year Cholesky GPs (non-centered factor
    # matmul), index gathers and reshape all lower the Reactant PRIMAL (all data
    # bound, only q traced). Its compiled REVERSE gradient does NOT lower — the
    # dual Cholesky FACTOR reverse hits the `stablehlo.cholesky` adjoint gap
    # (snag reactant-compile-f877fcfd). Native primal + native Enzyme gradient vs
    # Stan are the authoritative gate in `benchmark/gp_gate.jl`.
    @testset "hierarchical_gp (posteriordb; hierarchical GP, ILR simplex; primal)" begin
        a = evaluate_hierarchical_gp_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "multi_occupancy (posteriordb; marginalized occupancy, all data bound)" begin
        a = evaluate_multi_occupancy_source()
        @test _rapprox(_compile_run(a.kernel, Tuple(a.inputs)), a.output)
    end
    @testset "kronecker_gp (posteriordb; Kronecker eigen GP, all data bound)" begin
        gapped = true
        try
            a = evaluate_kronecker_gp_source()
            # Compile outside `@test` so the gap reaches the signature check.
            compiled_output = _compile_run(a.kernel, Tuple(a.inputs))
            @test _rapprox(compiled_output, a.output)
            # Self-firing pin: errors (Unexpected Pass) once the trace lowers,
            # forcing removal of the gate.
            gapped && @test_broken true
        catch e
            gapped && _xla_is_kron_eigen_gap(e) || rethrow()
        end
    end
    # kronecker_gp — the LKJ-Cholesky helpers are closed-form vectorized (no
    # scalar indexing, no data-derived unroll), so they trace under Reactant
    # and reproduce native. The FULL posterior still waits on the upstream
    # `eigen(::Symmetric)` gap (reactivekernels-use §7aa; snag
    # kronecker-gp-rea-dc8cef6f), which owns the whole-example legs.
    @testset "kronecker_gp LKJ helpers trace under Reactant (full posterior waits on upstream eigen §7aa)" begin
        a = evaluate_kronecker_gp_source(; model_only = true)
        K = 30
        z = 0.9 .* sin.((1:((K * (K - 1)) ÷ 2)) .* 0.9 .+ 1.0)
        compiled_lp, compiled_L =
            Base.invokelatest(_kron_lkj_helpers_compiled, a.sandbox, z, K)
        native_lp = Base.invokelatest(
            getfield(a.sandbox, :_ccl_constraint_lp), z, K)
        native_L = Base.invokelatest(
            getfield(a.sandbox, :_cholesky_corr_constrain_L), z, K)
        @test _rapprox(compiled_lp, native_lp)
        @test _rapprox(compiled_L, native_L)
    end
end

# Compiled reverse gradient (XLA gradient leg) for the hand-written examples
# with no StanBlocks counterpart. Each item's posterior-only query binds all
# data, so the unconstrained position is the only traced argument; the gradient
# is `Reactant.@compile ReactiveKernels.ad_value_and_gradient!` on that traced
# position (the acceptance_irt_reactant.jl pattern). The reference is the
# native CPU primal and its central finite-difference gradient.
const _XLA_AD_BACKEND = AutoEnzyme(mode = Enzyme.Reverse)

function _central_fd_gradient(f, q; h = 1e-6)
    g = similar(q)
    for i in eachindex(q)
        qp = copy(q); qp[i] += h
        qm = copy(q); qm[i] -= h
        g[i] = (f(qp) - f(qm)) / (2h)
    end
    g
end

function _xla_value_and_gradient(kernel, q)
    prepared = prepare_ad(kernel, _XLA_AD_BACKEND, q; active = :unconstrained)
    traced_q = Reactant.to_rarray(q)
    traced_gradient = Reactant.to_rarray(similar(q))
    compiled = @compile sync = true ReactiveKernels.ad_value_and_gradient!(
        prepared, traced_gradient, traced_q)
    value, gradient = compiled(prepared, traced_gradient, traced_q)
    (_host(value), Array{Float64}(gradient))
end

const _XLA_GRADIENT_ITEMS = (
    ("hmm_drive_0", evaluate_hmm_drive_0_source, (:unconstrained, :u, :v, :alpha),
        (; u = HMM_DRIVE_0_U, v = HMM_DRIVE_0_V, alpha = HMM_DRIVE_0_ALPHA), nothing),
    ("hmm_drive_1", evaluate_hmm_drive_1_source,
        (:unconstrained, :u, :v, :alpha, :tau, :rho),
        (; u = HMM_DRIVE_1_U, v = HMM_DRIVE_1_V, alpha = HMM_DRIVE_1_ALPHA,
           tau = HMM_DRIVE_1_TAU, rho = HMM_DRIVE_1_RHO), nothing),
    ("hmm_example", evaluate_hmm_example_source, (:unconstrained, :y, :K),
        (; y = HMM_EXAMPLE_Y, K = HMM_EXAMPLE_K), nothing),
    ("hmm_gaussian", evaluate_hmm_gaussian_source, (:unconstrained, :y, :K),
        (; y = HMM_GAUSSIAN_Y, K = HMM_GAUSSIAN_K), nothing),
    ("iohmm_reg", evaluate_iohmm_reg_source, (:unconstrained, :y, :u, :K),
        (; y = IOHMM_REG_Y, u = IOHMM_REG_U, K = IOHMM_REG_K), nothing),
    ("mt", evaluate_mt_source, (:unconstrained, :Y, :s, :T, :M),
        (; Y = MT_Y, s = MT_S, T = MT_T, M = MT_M), nothing),
    ("mth_model", evaluate_mth_model_source, (:unconstrained, :Y, :s, :T, :M),
        (; Y = MTH_Y, s = MTH_S, T = MTH_T, M = MTH_M), nothing),
    ("mtbh_model", evaluate_mtbh_model_source,
        (:unconstrained, :Y, :Yprev, :s, :T, :M),
        (; Y = MTBH_Y, Yprev = MTBH_YPREV, s = MTBH_S, T = MTBH_T, M = MTBH_M),
        nothing),
    ("multi_occupancy", evaluate_multi_occupancy_source,
        (:unconstrained, :X, :n, :J, :K),
        (; X = MULTI_OCC_X, n = MULTI_OCC_N, J = MULTI_OCC_J, K = MULTI_OCC_K),
        nothing),
    ("hierarchical_gp", evaluate_hierarchical_gp_source,
        (:unconstrained, :y, :year_ind, :state_ind, :region_ind, :state_region_ind,
         :N_years, :N_regions, :N_states, :N_years_obs),
        (; y = HGP_Y, year_ind = HGP_YEAR_IND, state_ind = HGP_STATE_IND,
           region_ind = HGP_REGION_IND, state_region_ind = HGP_STATE_REGION_IND,
           N_years = HGP_N_YEARS, N_regions = HGP_N_REGIONS, N_states = HGP_N_STATES,
           N_years_obs = HGP_N_YEARS_OBS),
        _xla_is_cholesky_adjoint_gap),
    ("kronecker_gp", evaluate_kronecker_gp_source, (:unconstrained, :x1, :y),
        (; x1 = KRON_X1, y = KRON_Y), _xla_is_kron_eigen_gap),
)

function _xla_gradient_measure(a, have, bound)
    # A generic (non-symmetric) point near the example's demo position.
    q = a.inputs.q .+ 0.05 .* sin.(1:length(a.inputs.q))
    kernel = prepare(a.model; have, want = :posterior, bound)
    native = kernel(q)
    fd = _central_fd_gradient(kernel, q)
    value, gradient = _xla_value_and_gradient(kernel, q)
    (; native, fd, value, gradient)
end

# `is_gap` is `nothing`, or the signature of that item's pinned gap (above).
@testset "PPL examples compiled reverse gradient ($label)" for (label, evaluate, have, bound, is_gap) in _XLA_GRADIENT_ITEMS
    gapped = is_gap !== nothing
    try
        a = evaluate()
        # The evaluated source defines fresh methods; cross the world-age barrier.
        r = Base.invokelatest(_xla_gradient_measure, a, have, bound)
        @test _rapprox(r.value, r.native)
        @test all(isfinite, r.gradient)
        @test r.gradient ≈ r.fd rtol = 1e-5
        # Self-firing pin: errors (Unexpected Pass) once the gap closes,
        # forcing removal of the gate.
        gapped && @test_broken true
    catch e
        gapped && is_gap(e) || rethrow()
    end
end
