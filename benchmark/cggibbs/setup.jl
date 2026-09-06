# Local environment for the CGGibbs benchmark: develop the repo checkout and add
# the two non-stdlib helpers the driver uses directly.
import Pkg
repo = dirname(dirname(@__DIR__))            # benchmark/cggibbs -> repo root
Pkg.activate(@__DIR__)
Pkg.develop(path = repo)
Pkg.add(["StableRNGs", "LogExpFunctions"])
# cggibbs_vs_nuts.jl additionally needs the NUTS baseline + ESS tooling
# (AdvancedHMC pinned to the repo's 0.8.6):
Pkg.add([Pkg.PackageSpec(name = "AdvancedHMC", version = "0.8.6"),
         Pkg.PackageSpec(name = "MCMCDiagnosticTools"),
         Pkg.PackageSpec(name = "LogDensityProblems")])
Pkg.precompile()
println("cggibbs env ready at ", @__DIR__)
