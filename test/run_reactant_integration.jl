using Pkg
using UUIDs

const REACTANT_VERSION = v"0.2.278"
const REACTANT_UUID = UUID("3c362404-f566-11ee-1572-e11a4b42c853")

root = normpath(joinpath(@__DIR__, ".."))
testfile = joinpath(@__DIR__, "test_reactant.jl")

mktempdir() do env
    Pkg.activate(env)
    # The test file runs in `Main`, so every package it imports must be a direct
    # dependency of this isolated environment.  A cold compile can otherwise
    # mask the omission by loading ReactiveKernels' transitive dependencies.
    Pkg.add(PackageSpec(name = "Reactant", version = REACTANT_VERSION))
    Pkg.pin(PackageSpec(name = "Reactant"))
    Pkg.develop(path = root)
    Pkg.add([
        PackageSpec(name = "Enzyme"),
        PackageSpec(name = "LogExpFunctions"),
    ])
    Pkg.instantiate()

    dep = Pkg.dependencies()[REACTANT_UUID]
    dep.version == REACTANT_VERSION || error(
        "expected Reactant $(REACTANT_VERSION), got $(dep.version)")
    println("REACTANT_DEP\tReactant\t", dep.version)

    julia = Base.julia_cmd()
    run(`$julia --startup-file=no --check-bounds=yes --project=$env $testfile`)
end
