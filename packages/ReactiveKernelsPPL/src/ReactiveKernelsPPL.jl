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

export ColumnRef, ParamName
export LikelihoodFamily, GaussianFam, BernoulliLogitFam, PoissonLogFam
export LinkFunction, IdentityLink, LogitLink, LogLink
export TermKind, InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm
export ResponseEvidence, LikelihoodSpec, TermSpec, PredictorSpec
export PopulationPrior, SampledParameter, AssignmentSpec, StructuralPlan
export ContractValidationError, validate_plan
export admitted_families, admitted_terms, admitted_functions, supports_term

include("contract.jl")

end # module ReactiveKernelsPPL
