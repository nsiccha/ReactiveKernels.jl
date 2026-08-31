using Pkg
using UUIDs

const REACTANT_VERSION = v"0.2.278"
const REACTANT_UUID = UUID("3c362404-f566-11ee-1572-e11a4b42c853")

root = normpath(joinpath(@__DIR__, ".."))
testfile = joinpath(@__DIR__, "test_reactant.jl")
phasepoint_testfile = joinpath(@__DIR__, "test_reactivehmc_phasepoint_reactant.jl")
effect_boundary_testfile = joinpath(
    @__DIR__, "test_effect_boundary_reactant.jl")
mutation_profile_b_testfile = joinpath(
    @__DIR__, "test_mutation_profile_b_reactant.jl")
example_packages = (
    joinpath(root, "packages", "ReactiveKernelsCompatibilityExamples"),
    joinpath(root, "packages", "ReactiveKernelsDistributionKernels"),
    joinpath(root, "packages", "ReactiveKernelsNUTSExamples"),
    joinpath(root, "packages", "ReactiveKernelsPPLExamples"),
)

mktempdir() do env
    Pkg.activate(env)
    # The test file runs in `Main`, so every package it imports must be a direct
    # dependency of this isolated environment.  A cold compile can otherwise
    # mask the omission by loading ReactiveKernels' transitive dependencies.
    Pkg.add(PackageSpec(name = "Reactant", version = REACTANT_VERSION))
    Pkg.pin(PackageSpec(name = "Reactant"))
    Pkg.develop([
        PackageSpec(path = path) for path in (root, example_packages...)
    ])
    Pkg.add([
        PackageSpec(name = "Enzyme"),
        PackageSpec(name = "LambertW"),
        PackageSpec(name = "LogExpFunctions"),
    ])
    Pkg.instantiate()

    dep = Pkg.dependencies()[REACTANT_UUID]
    dep.version == REACTANT_VERSION || error(
        "expected Reactant $(REACTANT_VERSION), got $(dep.version)")
    println("REACTANT_DEP\tReactant\t", dep.version)

    julia = Base.julia_cmd()
    selector = get(ENV, "RK_REACTANT_TESTSET", "all")
    if selector == "all"
        run(`$julia --startup-file=no --check-bounds=yes --project=$env $effect_boundary_testfile`)
        run(`$julia --startup-file=no --check-bounds=yes --project=$env $testfile`)
        run(`$julia --startup-file=no --check-bounds=yes --project=$env $phasepoint_testfile`)
        run(`$julia --startup-file=no --check-bounds=yes --project=$env $mutation_profile_b_testfile`)
    elseif selector == "mutation-profile-b"
        run(`$julia --startup-file=no --check-bounds=yes --project=$env $mutation_profile_b_testfile`)
    else
        error("unknown RK_REACTANT_TESTSET selector: $selector")
    end
end
