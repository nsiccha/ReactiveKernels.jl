# Materialize a local environment for the Gibbs proof of concept: develop the
# repo checkout and add the two sampling helpers used only by the demo drivers.
import Pkg
repo = dirname(dirname(@__DIR__))            # benchmark/gibbs_poc -> repo root
Pkg.activate(@__DIR__)
Pkg.develop(path = repo)
Pkg.add(["Distributions", "StableRNGs"])
Pkg.precompile()
println("gibbs_poc env ready at ", @__DIR__)
