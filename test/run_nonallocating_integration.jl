using Pkg
include(joinpath(@__DIR__, "..", "deps", "source_dependencies.jl"))

# MutatingFunctions comes from ReactiveKernelsPPL's source pin
# (`rk_source_dependencies`), the revision every setup entrypoint uses.
const MUTATING_FUNCTIONS_REV = rk_source_pins()["MutatingFunctions"]["rev"]

root = normpath(joinpath(@__DIR__, ".."))
testfiles = [joinpath(@__DIR__, "test_nonallocating.jl"),
             joinpath(@__DIR__, "test_nonallocating_init.jl"),
             joinpath(@__DIR__, "test_nonallocating_init_ad.jl"),
             joinpath(@__DIR__, "test_nonallocating_nested_products.jl"),
             joinpath(@__DIR__, "test_nonallocating_mnist.jl"),
             joinpath(@__DIR__, "test_nonallocating_ad.jl"),
             joinpath(@__DIR__, "test_nonallocating_repeated.jl"),
             joinpath(@__DIR__, "test_nonallocating_ownership.jl"),
             joinpath(@__DIR__, "test_nonallocating_plate_steps.jl"),
             joinpath(@__DIR__, "test_ppl_glm_nonallocating_ad.jl"),
             joinpath(@__DIR__, "test_ppl_nonallocating_sampler.jl"),
             joinpath(@__DIR__, "test_nonallocating_throwing_exemplar.jl"),
             joinpath(@__DIR__, "test_nonallocating_plate_reverse.jl"),
             joinpath(@__DIR__, "test_scan_plate_nonallocating.jl"),
             joinpath(@__DIR__, "test_reactive_nonallocating.jl"),
             joinpath(root, "packages", "ReactiveKernelsBatchingExamples",
                      "test", "test_batched_nonallocating.jl")]

mktempdir() do env
    Pkg.activate(env)
    # test_batched_nonallocating.jl exercises the reverse-mode gradient path, so
    # the integration environment also needs the AD and benchmarking stack.
    # test_ppl_glm_nonallocating_ad.jl needs the PPL surface plus Distributions
    # oracles for its independent references.
    Pkg.add(["DifferentiationInterface", "Enzyme", "BenchmarkTools",
             "Distributions"])
    Pkg.develop([PackageSpec(path = root); rk_source_dependencies()])
    # test_nonallocating_mnist.jl runs the real MNIST example graph. Julia 1.10
    # ignores [sources], so the nested example packages are developed by path.
    Pkg.develop(path = joinpath(root, "packages",
                                "ReactiveKernelsDistributionKernels"))
    Pkg.develop(path = joinpath(root, "packages", "ReactiveKernelsPPL"))
    Pkg.develop(path = joinpath(root, "packages", "ReactiveKernelsPPLExamples"))
    Pkg.instantiate()

    println("NONALLOCATING_DEP\tMutatingFunctions\t", MUTATING_FUNCTIONS_REV)
    julia = Base.julia_cmd()
    for testfile in testfiles
        run(`$julia --startup-file=no --check-bounds=yes --project=$env $testfile`)
    end
end
