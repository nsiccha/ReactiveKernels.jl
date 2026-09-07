# Focused verification receipt for the arma11 + dugongs `density`->`posterior`
# node rename (canonical model sources + tests). Runs ONLY those two package
# testsets (native, no Reactant) against the pinned env. NOT a by-construction
# claim — this actually executes the sibling's 4-part parity/structure gates.
import Pkg
const ENV_DIR = joinpath(@__DIR__, "all80-env")
Pkg.activate(ENV_DIR)
using ReactiveKernels
using Test
const TDIR = joinpath(@__DIR__, "..", "packages", "ReactiveKernelsPPLExamples", "test")
@testset "arma11 + dugongs rename verification" begin
    include(joinpath(TDIR, "test_arma11_example.jl"))
    include(joinpath(TDIR, "test_dugongs_example.jl"))
end
println("FOCUSED_RENAME_TESTS_DONE")
