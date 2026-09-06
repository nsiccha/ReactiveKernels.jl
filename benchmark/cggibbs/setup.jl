# Local environment for the CGGibbs benchmark: develop the repo checkout and add
# the two non-stdlib helpers the driver uses directly.
import Pkg
repo = dirname(dirname(@__DIR__))            # benchmark/cggibbs -> repo root
Pkg.activate(@__DIR__)
Pkg.develop(path = repo)
Pkg.add(["StableRNGs", "LogExpFunctions"])
Pkg.precompile()
println("cggibbs env ready at ", @__DIR__)
