using Pkg
Pkg.activate(@__DIR__)
root = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.develop([
    PackageSpec(path=root),
    PackageSpec(path=joinpath(root, "packages", "ReactiveKernelsDistributionKernels")),
    PackageSpec(path=joinpath(root, "packages", "ReactiveKernelsPPLExamples")),
])
Pkg.instantiate()
