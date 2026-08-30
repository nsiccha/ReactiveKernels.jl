using Pkg

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const LOCAL_PACKAGE_PATHS = (
    REPOSITORY_ROOT,
    joinpath(@__DIR__, "ReactiveKernelsDistributionKernels"),
    joinpath(@__DIR__, "ReactiveKernelsKernelExamples"),
    joinpath(@__DIR__, "ReactiveKernelsBatchingExamples"),
    joinpath(@__DIR__, "ReactiveKernelsPPLExamples"),
    joinpath(@__DIR__, "ReactiveKernelsCompatibilityExamples"),
)

Pkg.develop([PackageSpec(path = path) for path in LOCAL_PACKAGE_PATHS])
Pkg.instantiate()
Pkg.precompile()
