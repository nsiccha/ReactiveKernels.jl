using Test

const PPL_SOURCE_CASES = (
    ("eight-schools.md", "eight_schools.jl", EightSchoolsExample,
     :EIGHT_SCHOOLS_SOURCE, :evaluate_eight_schools_source),
    ("sum-to-zero.md", "sum_to_zero.jl", SumToZeroExample,
     :SUM_TO_ZERO_SOURCE, :evaluate_sum_to_zero_source),
    ("linear-regression.md", "linear_regression.jl", LinearRegressionExample,
     :LINEAR_REGRESSION_SOURCE, :evaluate_linear_regression_source),
    ("beta-binomial.md", "beta_binomial.jl", BetaBinomialExample,
     :BETA_BINOMIAL_SOURCE, :evaluate_beta_binomial_source),
    ("poisson-gamma.md", "poisson_gamma.jl", PoissonGammaExample,
     :POISSON_GAMMA_SOURCE, :evaluate_poisson_gamma_source),
    ("dugongs-growth.md", "dugongs_growth.jl", DugongsGrowthExample,
     :DUGONGS_SOURCE, :evaluate_dugongs_source),
    ("arma11.md", "arma11.jl", ARMA11Example,
     :ARMA11_SOURCE, :evaluate_arma11_source),
    ("covid19imperial.md", "covid19imperial.jl", Covid19ImperialExample,
     :COVID19IMPERIAL_SOURCE, :evaluate_covid19imperial_source),
    ("garch11.md", "garch11.jl", GARCH11Example,
     :GARCH11_SOURCE, :evaluate_garch11_source),
    ("hmm-example.md", "hmm_example.jl", HmmExampleExample,
     :HMM_EXAMPLE_SOURCE, :evaluate_hmm_example_source),
    ("hmm-gaussian.md", "hmm_gaussian.jl", HmmGaussianExample,
     :HMM_GAUSSIAN_SOURCE, :evaluate_hmm_gaussian_source),
    ("iohmm-reg.md", "iohmm_reg.jl", IohmmRegExample,
     :IOHMM_REG_SOURCE, :evaluate_iohmm_reg_source),
    ("mnist-logistic.md", "mnist_logistic.jl", MNISTLogisticExample,
     :MNIST_LOGISTIC_SOURCE, :evaluate_mnist_logistic_source),
    ("mnist-logistic.md", "mnist_logistic.jl", MNISTLogisticExample,
     :MNIST_LOGISTIC_OPTIMIZED_SOURCE, :evaluate_mnist_logistic_optimized_source),
    ("mvnormal-regression.md", "mvnormal_regression.jl", MVNormalRegressionExample,
     :MVNORMAL_REGRESSION_SOURCE, :evaluate_mvnormal_regression_source),
    ("bound-regression.md", "bound_regression.jl", BoundRegressionExample,
     :BOUND_REGRESSION_SOURCE, :evaluate_bound_regression_source),
    ("lda.md", "lda.jl", LDAExample,
     :LDA_SOURCE, :evaluate_lda_source),
    ("nn-rbm.md", "nn_rbm.jl", NNRBMExample,
     :NN_RBM_SOURCE, :evaluate_nn_rbm_source),
    ("gp-regr.md", "gp_regr.jl", GPRegrExample,
     :GP_REGR_SOURCE, :evaluate_gp_regr_source),
    ("gp-pois-regr.md", "gp_pois_regr.jl", GPPoisRegrExample,
     :GP_POIS_REGR_SOURCE, :evaluate_gp_pois_regr_source),
    ("accel-gp.md", "accel_gp.jl", AccelGPExample,
     :ACCEL_GP_SOURCE, :evaluate_accel_gp_source),
    ("hierarchical-gp.md", "hierarchical_gp.jl", HierarchicalGPExample,
     :HIERARCHICAL_GP_SOURCE, :evaluate_hierarchical_gp_source),
    ("losscurve-sislob.md", "losscurve_sislob.jl", LosscurveSislobExample,
     :LOSSCURVE_SISLOB_SOURCE, :evaluate_losscurve_sislob_source),
    ("accel-splines.md", "accel_splines.jl", AccelSplinesExample,
     :ACCEL_SPLINES_SOURCE, :evaluate_accel_splines_source),
    ("state-space-stochastic.md", "state_space_stochastic.jl", StateSpaceStochasticExample,
     :STATE_SPACE_STOCHASTIC_SOURCE, :evaluate_state_space_stochastic_source),
    ("prophet.md", "prophet.jl", ProphetExample,
     :PROPHET_SOURCE, :evaluate_prophet_source),
    ("irt-2pl.md", "irt_2pl.jl", Irt2plExample,
     :IRT_2PL_SOURCE, :evaluate_irt_2pl_source),
    ("2pl-latent-reg-irt.md", "2pl_latent_reg_irt.jl", TwoplLatentRegIrtExample,
     :TWOPL_LR_SOURCE, :evaluate_2pl_latent_reg_irt_source),
    ("hier-2pl.md", "hier_2pl.jl", Hier2plExample,
     :HIER_2PL_SOURCE, :evaluate_hier_2pl_source),
    ("gpcm-latent-reg-irt.md", "gpcm_latent_reg_irt.jl", GpcmLatentRegIrtExample,
     :GPCM_LR_SOURCE, :evaluate_gpcm_latent_reg_irt_source),
    ("grsm-latent-reg-irt.md", "grsm_latent_reg_irt.jl", GrsmLatentRegIrtExample,
     :GRSM_LR_SOURCE, :evaluate_grsm_latent_reg_irt_source),
    ("kronecker-gp.md", "kronecker_gp.jl", KroneckerGpExample,
     :KRON_SOURCE, :evaluate_kronecker_gp_source),
    ("glmm1.md", "glmm1_model.jl", GLMM1ModelExample,
     :GLMM1_SOURCE, :evaluate_glmm1_model_source),
    ("bym2-offset-only.md", "bym2_offset_only.jl", Bym2OffsetOnlyExample,
     :BYM2_SOURCE, :evaluate_bym2_offset_only_source),
    ("bones.md", "bones_model.jl", BonesModelExample,
     :BONES_SOURCE, :evaluate_bones_model_source),
    ("multi-occupancy.md", "multi_occupancy.jl", MultiOccupancyExample,
     :MULTI_OCC_SOURCE, :evaluate_multi_occupancy_source),
    ("lotka-volterra.md", "lotka_volterra.jl", LotkaVolterraExample,
     :LOTKA_VOLTERRA_SOURCE, :evaluate_lotka_volterra_source),
    ("sir.md", "sir.jl", SIRExample,
     :SIR_SOURCE, :evaluate_sir_source),
    ("soil-incubation.md", "soil_incubation.jl", SoilIncubationExample,
     :SOIL_INCUBATION_SOURCE, :evaluate_soil_incubation_source),
    ("hmm-drive-1.md", "hmm_drive_1.jl", HmmDrive1Example,
     :HMM_DRIVE_1_SOURCE, :evaluate_hmm_drive_1_source),
)

# One executed source kernel per registered case: a file carrying
# two cases (the idiomatic and optimized MNIST models) declares exactly two.
const _KERNELS_PER_EXAMPLE_FILE = Dict{String,Int}()
for case in PPL_SOURCE_CASES
    _KERNELS_PER_EXAMPLE_FILE[case[2]] =
        get(_KERNELS_PER_EXAMPLE_FILE, case[2], 0) + 1
end

@testset "PPL examples evaluate their source authority" begin
    for (_, example_name, owner, source_name, evaluator_name) in
        PPL_SOURCE_CASES
        example = read(joinpath(@__DIR__, "..", "src", example_name), String)
        source = getfield(owner, source_name)
        evaluator = getfield(owner, evaluator_name)

        @test length(findall("@kernel model(", example)) ==
              _KERNELS_PER_EXAMPLE_FILE[example_name]
        @test occursin("@kernel model(", source)
        @test occursin(r"\w+_kernel = prepare\(model;", source)
        @test occursin(r"inputs = \(;[\s\S]*?\),\n    model,\n    kernel", source)

        artifact = evaluator()
        @test artifact.source == strip(source, '\n')
        @test parentmodule(artifact.sandbox) === owner
        @test Base.PkgId(artifact.sandbox) == Base.PkgId(owner)
        # `evaluator()` defines the authored recipe closures in a fresh sandbox.
        # Cross that dynamic-evaluation boundary in the latest world, exactly as
        # the source evaluator does, so inlined operations remain reusable on
        # Julia versions with stricter world-age enforcement.
        observed = Base.invokelatest(
            artifact.kernel, Tuple(artifact.inputs)...,
        )
        @test artifact.output == observed
        @test code_expr(artifact.kernel) === code_expr(artifact.kernel)
    end
end

# The model-only evaluation `__init__` runs finds the `model` its source
# defines in a fresh sandbox, a binding Julia 1.12 and later show only in a
# later world than the evaluator's own (so on 1.12 the package failed to load).
@testset "a model-only source evaluation finds the model its source defines" begin
    for (_, _, owner, _, evaluator_name) in PPL_SOURCE_CASES
        evaluator = getfield(owner, evaluator_name)
        hasmethod(evaluator, Tuple{}, (:model_only,)) || continue
        result = evaluator(; model_only = true)
        @test result.model isa ReactiveKernels.KernelSpec
        @test parentmodule(result.sandbox) === owner
    end
end
