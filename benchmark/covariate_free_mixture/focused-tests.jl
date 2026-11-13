using ReactiveKernels, ReactiveKernelsDistributionKernels, ReactiveKernelsPPL
import Enzyme
println((; julia=VERSION, enzyme=pkgversion(Enzyme),
    library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
    source=readchomp(`git -C $(pkgdir(ReactiveKernels)) rev-parse HEAD`)))
failed = String[]
for (package, files) in (
        (ReactiveKernelsDistributionKernels, (
            "test_product_mixture.jl", "test_bound_binomial_mixture.jl",
            "test_zero_inflated_product_mixture.jl", "test_product_mixture_rows.jl",
            "test_bernoulli_mixture_counts.jl")),
        (ReactiveKernelsPPL, (
            "test_whole_event_invariants.jl",)))
    for file in files
        println("TEST_BEGIN ", nameof(package), "/", file)
        flush(stdout)
        started = time()
        try
            include(joinpath(pkgdir(package), "test", file))
        catch err
            push!(failed, string(nameof(package), "/", file))
            showerror(stdout, err, catch_backtrace())
            println()
        end
        println("TEST_END ", nameof(package), "/", file, " seconds=", time()-started)
        flush(stdout)
    end
end
isempty(failed) || error("Failed files: ", join(failed, ", "))
include(joinpath(@__DIR__, "ragged-logit-tests.jl"))
