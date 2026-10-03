"""
    ReactiveKernelsPPL

Thin PPL compiler for ReactiveKernels: consumes a typed structural plan (slice
1: emitted by the BRM-side RK backend) and generates fittable `@kernel`
programs — packed unconstrained layout, transforms, in-graph preprocessing,
posterior density — addressable through the canonical sampler-query surface.

External-example style nested package: ReactiveKernels core never depends on
it. Slice-1 scope (population GLMs, density+gradient only) is set by the joint
BRM–RK backend plan; the emitter→layer input contract lives in
`src/contract.jl` once agreed.
"""
module ReactiveKernelsPPL

using ReactiveKernels
import DataAPI
import SpecialFunctions
using SpecialFunctions: erfc, loggamma
using LogExpFunctions: logaddexp
using StaticArrays: SMatrix, SVector
using Statistics: mean, std, var

export ColumnRef, ParamName, ColumnData
export LikelihoodFamily, GaussianFam, BernoulliLogitFam, PoissonLogFam,
    BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam,
    BernoulliProbitFam, BernoulliCloglogFam, BinomialProbitFam,
    BinomialCloglogFam, BinomialProbFam, BetaLogitFam, CategoricalLogitFam,
    OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam,
    MvNormalCholeskyFam, CensoredAddpropnormalFam, TgiCategoryFam,
    TgiResponseFam, TgiCensoredFam, NormalIDGLMFam, BernoulliLogitGLMFam,
    PoissonLogGLMFam, MixtureFam, StudentTFam, HurdlePoissonFam,
    ZeroInflatedPoissonFam, InverseGaussianFam, BetaBinomial2Fam, VonMisesFam,
    NegativeBinomialFam, ExponentialLogFam, LogNormalFam, WeibullFam,
    GammaValueFam, WeibullValueFam,
    ZeroInflatedBinomialFam
export LinkFunction, IdentityLink, LogitLink, LogLink, ProbitLink, CloglogLink
export TermKind, InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm, LatentTerm,
    VaryingEffectTerm, SplineSummandTerm, HSGPSummandTerm,
    ScanSummandTerm, MonotonicTerm, MonotonicSummandTerm, MatrixTerm,
    DarSummandTerm, ComposedTerm
export ResponseEvidence, LikelihoodSpec, TermSpec, PredictorSpec
export ScalePredictorRef, MixtureComplementWeights
export PopulationPrior, R2D2Prior, HorseshoePrior, SampledParameter, PlateParameter, VectorParameter, AssignmentSpec, VectorAssignmentSpec, ArrayParameter
export VaryingZRecipe, VaryingMargin, VaryingSdPrior, VaryingDraws, VaryingSlice
export VaryingMultiMembership, VaryingStrata
export SplineBasisBlock, SplineBasis, SplineVector
export HSGPBasis, HyperPrior, HSGPHyperLP, HSGPGrouping
export KernelPlate, LinearPKScheduleSpec, LinearPKEventLPSpec
export EVENT_LP_NAME
export LevelMap
export DesignMatrix
export StructuralPlan, SubmodelScope
export ContractValidationError, validate_plan, validate_structure, validate_data
export topological_order, isbound, bind_data, COLUMN_ROLES
export admitted_families, admitted_terms, admitted_functions, admitted_elementwise
export supports_term
export block_name
export build_kernel, kernel_expr
export DesignShape, DesignBlock, design_shape, coefficient_priors
export LayoutTable, LayoutEntry, assign_layout, coordinate_names
export constrain, unconstrain, logjac, support_of
export ordered_constrain, ordered_unconstrain, ordered_logjac
export simplex_constrain, simplex_unconstrain, simplex_logjac
export lkj_chol_constrain, lkj_chol_unconstrain, lkj_chol_logjac,
    lkj_chol_constrain_hyperspherical, lkj_chol_unconstrain_hyperspherical,
    lkj_chol_logjac_hyperspherical
export lkj_logconst, lkj_corr_cholesky_logpdf
export positive_bijector, unit_bijector, interval_bijector, floored_bijector,
    upper_bijector, BIJECTORS
export coordinate_read, block_read, transform_statements, jacobian_term
export design_name, offset_name, monotonic_name, design_recipe, offset_recipe,
    monotonic_recipe
export preprocessing_recipes
export PPL_NODES, WORKFLOW_WANTS, workflow_wants
export prepare_query, prepare_sampler, SamplerQuery, sampler_value_and_gradient!
export restore_draws
export RKPPLModel, RKPPLBoundModel, RKPPLSubmodel, lower_rkppl, condition, @rkppl, SurfaceLoweringError
export ordered_logistic, monotonic, differenced_ar1, r2d2_coefs,
    horseshoe_coefs, penalized_smooth, t2_smooth, hsgp_effect,
    hsgp_periodic_effect, hsgp_grouped_effect, linear_pk_log_f
export varying_coefs, varying_coefs_correlated, varying_coefs_centered,
    varying_coefs_centered_correlated, varying_stratified,
    varying_stratified_correlated
export tps_basis, cr_basis, t2_basis, hsgp_basis, hsgp_periodic_basis, hsgp_sqrt_spd,
    hsgp_grouped_sqrt_spd, hsgp_periodic_sqrt_spd, hsgp_rho_floors,
    hsgp_periodic_rho_floor, hsgp_matern_sqrt_spd, hsgp_periodic_grouped_sqrt_spd
export ScanSpec, ScanStep, ScanSetup, parse_scan_block
export DarSpec
export LINEAR_EVENT_READ, LINEAR_EVENT_DOSE, LINEAR_EVENT_DOSE_SEGMENT
export build_linear_pk_schedule, linear_pk_read_locs, linear_pk_read_locs_auc
export linear_pk_op_log_dose, linear_pk_event_log_f
export linear_pk_system_3, linear_pk_propagate_3, linear_pk_add_dose_3,
    linear_pk_add_regular_doses_3
export QT_COUPLING_SPINES, QT_OBS_FAMILIES
export admit_qt_spine, admit_qt_obs_family
export qt_loc_assignment, qt_obs_statement, pk_obs_statement
export validate_qt_joint_prep
export TGIOptions, tgi_options
export TGI_OBSERVATIONS, TGI_STRUCTURES, TGI_THRESHOLDS, TGI_MEASURES
export TGI_TIME_SCALE_H, TGI_LOG_PR, TGI_LOG_PD, TGI_RECIST_LOG_PR,
    TGI_RECIST_LOG_PD
export tgi_measure_dim, tgi_threshold_scale, tgi_fixed_cutpoints,
    tgi_estimated_cutpoints, tgi_uses_nadir
export tgi_ratio_loglinear, tgi_ratio_resistant, tgi_log_survival,
    tgi_running_nadir, tgi_nadir_scan_expr, tgi_segmented_nadir,
    tgi_inv_logit
export tgi_normal_lcdf, tgi_log_diff_exp, tgi_interval_logprob,
    tgi_report_logprob
export tgi_category_lpmf, tgi_category_lpmfs,
    tgi_response_lpmf, tgi_response_lpmfs,
    tgi_censored_lpdf, tgi_censored_lpdfs
export tgi_category_stmts, tgi_response_stmts, tgi_censored_stmts
export TGI_CELL_FUNCTIONS
export rk_expm
export rk_symmetric_eigvals, rk_symmetric_eigvecs
export transit_twocmt_unit, transit_twocmt_unit_response, transit_twocmt_rule,
    prepare_transit_twocmt_rule

include("contract.jl")
include("expm_rule.jl")
# Owned symmetric-eigendecomposition rules: defined in
# ReactiveKernelsDistributionKernels (beside `loggamma` / `logbeta`),
# re-exported here with their rule graphs.
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    rk_symmetric_eigvals, rk_symmetric_eigvecs, rk_symmetric_eigvals_rule,
    rk_symmetric_eigvecs_rule
include("pkcells.jl")
include("transit_twocmt.jl")
include("transit_twocmt_rule.jl")
include("pk_rectangular.jl")
include("design.jl")
include("bijectors.jl")
include("mv_slices.jl")
include("layout.jl")
include("preprocessing.jl")
include("generator.jl")
include("arrays.jl")
include("query.jl")
include("distribution_defaults.jl")
include("surface.jl")
include("scan.jl")
include("qt_joint.jl")
include("tgi.jl")
include("smooth_bases.jl")
include("library.jl")

# Late import into the generated-models scope: `PPLGeneratedModels`
# binds its `import`s when `generator.jl` loads, before `tgi.jl`
# defines the per-element likelihood cells — so the joint plates'
# cells register here, after their file (an `import` of a
# not-yet-defined name warns and never binds).
Core.eval(PPLGeneratedModels,
    :(import ..tgi_category_lpmf, ..tgi_response_lpmf, ..tgi_censored_lpdf,
        ..tgi_segmented_nadir))

end # module ReactiveKernelsPPL
