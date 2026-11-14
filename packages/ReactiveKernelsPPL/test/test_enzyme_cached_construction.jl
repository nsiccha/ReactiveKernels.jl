using ReactiveKernelsPPL, Test

# Loading Enzyme adds methods (`ndims` and `eltype` for `Enzyme.TupleArray`,
# `convert` to `Array`, LibGit2's `Base.IteratorSize`) that match abstract calls
# on RKPPL's construction path. Before ReactiveKernelsPPLEnzymeExt re-ran the
# construction workload in its own image, a fresh process that loaded Enzyme
# recompiled nearly all of lowering, binding and `build_kernel` on its first
# model (snag `rkppl-cold-port-72991c9f`). A fresh child builds a model the
# workload does not build, through the `@rkppl` surface, and reports how much
# of its compilation was recompilation of invalidated code.
@testset "a process that loads Enzyme keeps RKPPL's construction cached" begin
    fixture = """
        using ReactiveKernels, ReactiveKernelsPPL, Enzyme
        ext = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLEnzymeExt) !== nothing
        model = @rkppl begin
            a ~ Normal(0, 3)
            b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            y .~ Normal.(a .+ b .* x, sigma)
        end
        Base.cumulative_compile_timing(true)
        c0 = Base.cumulative_compile_time_ns()
        build_kernel(model(; x = [0.0, 1.0, 2.0]) | (; y = [0.2, 1.1, 1.9]))
        c1 = Base.cumulative_compile_time_ns()
        print(ext, " ", (c1[1] - c0[1]) / 1e9, " ", (c1[2] - c0[2]) / 1e9)
        """
    out = read(`$(Base.julia_cmd()) --startup-file=no
               --project=$(Base.active_project()) -e $fixture`, String)
    ext, compile, recompile = split(out)
    compile, recompile = parse(Float64, compile), parse(Float64, recompile)
    @test ext == "true"
    # Measured on Julia 1.13.1 with registered Enzyme 0.13.219 (gordito, loaded
    # host): 0.10-0.12 of 1.4-1.8 s with the extension; 0.67-0.68 of 57-62 s
    # without it.
    @test recompile < 0.25 * compile
end
