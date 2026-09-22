using Pkg
using UUIDs

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const MUTATING_FUNCTIONS_URL = "https://github.com/nsiccha/MutatingFunctions.jl"
const MUTATING_FUNCTIONS_REV = "4fc41b1c7b774133ceaacc4ff3c34c67b15b87b2"
const MUTATING_FUNCTIONS_UUID = UUID("8a4c2d94-4b3b-4f9e-be63-a3c0cd816e3a")
const LOCAL_PACKAGE_PATHS = (
    REPOSITORY_ROOT,
    joinpath(@__DIR__, "ReactiveKernelsDistributionKernels"),
    joinpath(@__DIR__, "ReactiveKernelsKernelExamples"),
    joinpath(@__DIR__, "ReactiveKernelsBatchingExamples"),
    joinpath(@__DIR__, "ReactiveKernelsPPLExamples"),
    joinpath(@__DIR__, "ReactiveKernelsCompatibilityExamples"),
    joinpath(@__DIR__, "ReactiveKernelsNUTSExamples"),
    joinpath(@__DIR__, "ReactiveKernelsStreamingStats"),
    joinpath(@__DIR__, "ReactiveKernelsHMCDiagnostics"),
    joinpath(@__DIR__, "ReactiveKernelsPPL"),
    joinpath(@__DIR__, "ReactiveKernelsReactantODESolvers"),
)

Pkg.develop([PackageSpec(path = path) for path in LOCAL_PACKAGE_PATHS])
# Unregistered test-only dependencies must be pinned here, not just in the
# example packages' [extras]: Julia 1.10 ignores [sources], and Pkg.test
# builds its sandbox from the pruned parent manifest, so a test extra that
# is neither registered nor in this env fails with "expected package ... to
# be registered". Same pin as test/run_nonallocating_integration.jl.
Pkg.add(PackageSpec(url = MUTATING_FUNCTIONS_URL, rev = MUTATING_FUNCTIONS_REV))
dep = Pkg.dependencies()[MUTATING_FUNCTIONS_UUID]
dep.git_revision == MUTATING_FUNCTIONS_REV || error(
    "expected MutatingFunctions revision $(MUTATING_FUNCTIONS_REV), got $(dep.git_revision)")
Pkg.instantiate()
Pkg.precompile()
