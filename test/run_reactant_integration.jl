using Pkg
using UUIDs

const REACTANT_VERSION = v"0.2.278"
const REACTANT_UUID = UUID("3c362404-f566-11ee-1572-e11a4b42c853")

root = normpath(joinpath(@__DIR__, ".."))
testfile = joinpath(@__DIR__, "test_reactant.jl")

mktempdir() do env
    Pkg.activate(env)
    Pkg.add(PackageSpec(name = "Reactant", version = REACTANT_VERSION))
    Pkg.develop(path = root)
    Pkg.instantiate()

    dep = Pkg.dependencies()[REACTANT_UUID]
    dep.version == REACTANT_VERSION || error(
        "expected Reactant $(REACTANT_VERSION), got $(dep.version)")
    println("REACTANT_DEP\tReactant\t", dep.version)

    julia = Base.julia_cmd()
    run(`$julia --startup-file=no --check-bounds=yes --project=$env $testfile`)
end
