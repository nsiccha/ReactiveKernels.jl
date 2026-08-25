using Pkg
using UUIDs

const MUTATING_FUNCTIONS_URL = "https://github.com/nsiccha/MutatingFunctions.jl"
const MUTATING_FUNCTIONS_REV = "b353559ef3e391ae2e2d98256b6967903fdfa410"
const MUTATING_FUNCTIONS_UUID = UUID("8a4c2d94-4b3b-4f9e-be63-a3c0cd816e3a")

root = normpath(joinpath(@__DIR__, ".."))
testfiles = [joinpath(@__DIR__, "test_nonallocating.jl"),
             joinpath(@__DIR__, "test_reactive_nonallocating.jl")]

mktempdir() do env
    Pkg.activate(env)
    Pkg.add(PackageSpec(url = MUTATING_FUNCTIONS_URL,
                        rev = MUTATING_FUNCTIONS_REV))
    dep = Pkg.dependencies()[MUTATING_FUNCTIONS_UUID]
    dep.git_revision == MUTATING_FUNCTIONS_REV || error(
        "expected MutatingFunctions revision $(MUTATING_FUNCTIONS_REV), got $(dep.git_revision)")
    Pkg.develop(path = root)
    Pkg.instantiate()

    println("NONALLOCATING_DEP\tMutatingFunctions\t", dep.git_revision)
    julia = Base.julia_cmd()
    for testfile in testfiles
        run(`$julia --startup-file=no --check-bounds=yes --project=$env $testfile`)
    end
end
