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
using ReactiveKernels: rk_expm_rule, rk_symmetric_eigvals_rule,
    rk_symmetric_eigvecs_rule
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
    BinomialCloglogFam, BinomialProbFam, BetaLogitFam, BetaShapeFam, CategoricalLogitFam,
    OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam,
    MvNormalCholeskyFam, CensoredAddpropnormalFam, NormalIDGLMFam, BernoulliLogitGLMFam,
    PoissonLogGLMFam, MixtureFam, StudentTFam, HurdlePoissonFam,
    ZeroInflatedPoissonFam, InverseGaussianFam, BetaBinomial2Fam, VonMisesFam,
    NegativeBinomialFam, ExponentialLogFam, LogNormalFam, WeibullFam,
    GammaValueFam, WeibullValueFam, CauchyFam,
    ZeroInflatedBinomialFam
export LinkFunction, IdentityLink, LogitLink, LogLink, ProbitLink, CloglogLink
export TermKind, InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm, LatentTerm,
    VaryingEffectTerm,
    ScanSummandTerm, MatrixTerm,
    ComposedTerm
export ResponseEvidence, LikelihoodSpec, TermSpec, PredictorSpec
export ScalePredictorRef, MixtureComplementWeights
export PopulationPrior, SampledParameter, PlateParameter, VectorParameter, AssignmentSpec, VectorAssignmentSpec, ArrayParameter
export VaryingZRecipe, VaryingMargin, VaryingSdPrior, VaryingDraws, VaryingSlice
export VaryingMultiMembership, VaryingStrata
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
export design_name, offset_name, design_recipe, offset_recipe
export preprocessing_recipes
export PPL_NODES, WORKFLOW_WANTS, workflow_wants
export prepare_query, prepare_sampler, SamplerQuery, sampler_value_and_gradient!
export restore_draws
export sampling_logdensity, sampling_geometry, sampling_fragment, LogDensity, ParameterGeometry
export RKPPLModel, RKPPLBoundModel, RKPPLSubmodel, lower_rkppl, condition, @rkppl, SurfaceLoweringError
export ScanSpec, ScanStep, ScanSetup, parse_scan_block
export rk_expm
export rk_symmetric_eigvals, rk_symmetric_eigvecs

include("contract.jl")
include("sampling.jl")
import ReactiveKernelsDistributionKernels: DistributionKernelSources
# The general numerical callables and rule graphs are owned by RK proper;
# existing PPL-qualified names import those same bindings for compatibility.
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

end # module ReactiveKernelsPPL
