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
import SpecialFunctions
using SpecialFunctions: loggamma

export ColumnRef, ParamName, ColumnData
export LikelihoodFamily, GaussianFam, BernoulliLogitFam, PoissonLogFam,
    BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam,
    BernoulliProbitFam, BernoulliCloglogFam, BinomialProbitFam,
    BinomialCloglogFam, BetaLogitFam, CategoricalLogitFam,
    OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam,
    MvNormalCholeskyFam
export LinkFunction, IdentityLink, LogitLink, LogLink, ProbitLink, CloglogLink
export TermKind, InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm, LatentTerm,
    VaryingEffectTerm, SplineSummandTerm, HSGPSummandTerm,
    ScanSummandTerm, MonotonicTerm, MonotonicSummandTerm, MatrixTerm
export ResponseEvidence, LikelihoodSpec, TermSpec, PredictorSpec
export ScalePredictorRef
export PopulationPrior, R2D2Prior, SampledParameter, PlateParameter, VectorParameter, AssignmentSpec, VectorAssignmentSpec
export VaryingZRecipe, VaryingMargin, VaryingSdPrior, VaryingDraws, VaryingSlice
export SplineBasisBlock, SplineBasis, SplineVector
export HSGPBasis
export KernelPlate
export LevelMap
export DesignMatrix
export StructuralPlan
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
export lkj_chol_constrain, lkj_chol_unconstrain, lkj_chol_logjac
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
export RKPPLModel, RKPPLSubmodel, lower_rkppl, @rkppl, SurfaceLoweringError
export ScanSpec, ScanStep, ScanSetup, parse_scan_block

include("contract.jl")
include("design.jl")
include("bijectors.jl")
include("layout.jl")
include("preprocessing.jl")
include("generator.jl")
include("query.jl")
include("surface.jl")
include("scan.jl")

end # module ReactiveKernelsPPL
