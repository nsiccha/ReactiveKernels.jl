using Pkg
include(joinpath(@__DIR__, "..", "deps", "source_dependencies.jl"))

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const LOCAL_PACKAGE_PATHS = (
    REPOSITORY_ROOT,
    joinpath(@__DIR__, "ReactiveKernelsDistributionKernels"),
    joinpath(@__DIR__, "ReactiveKernelsKernelExamples"),
    joinpath(@__DIR__, "ReactiveKernelsBatchingExamples"),
    joinpath(@__DIR__, "ReactiveKernelsPPLExamples"),
    joinpath(@__DIR__, "ReactiveKernelsCompatibilityExamples"),
    joinpath(@__DIR__, "ReactiveKernelsPPL"),
)

# The unregistered source pins, MutatingFunctions included (a dependency of
# ReactiveKernelsPPL), come from `rk_source_dependencies`: Julia 1.10 ignores
# [sources], and Pkg.test builds its sandbox from the pruned parent manifest,
# so a package that is neither registered nor in this env fails with "expected
# package ... to be registered".
Pkg.develop([PackageSpec[PackageSpec(path = path) for path in LOCAL_PACKAGE_PATHS];
             rk_source_dependencies()])
Pkg.instantiate()
Pkg.precompile()
