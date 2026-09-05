if VERSION < v"1.11"
    # Julia 1.10 can deadlock while precompiling sibling Reactant extensions.
    get!(ENV, "JULIA_NUM_PRECOMPILE_TASKS", "1")
end
using Pkg
Pkg.activate(@__DIR__)
root = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.develop([
    PackageSpec(path=root),
    PackageSpec(path=joinpath(root, "packages", "ReactiveKernelsDistributionKernels")),
    PackageSpec(path=joinpath(root, "packages", "ReactiveKernelsPPLExamples")),
])
Pkg.instantiate()
