using ReactiveKernelsPPL, Test

# A package loaded after RK/RKPPL that adds a method to a Base function RK's
# construction code calls on a value inference cannot type invalidates the
# construction code RK's and RKPPL's images hold, so a fresh process
# recompiles lowering, binding and `build_kernel` on its first model (snag
# `downstream-workl-75f8cba4`: JSON 1.x's `lastindex` and `foreach` on its
# lazy values and StructUtils' `hasproperty` invalidated 3,882 RK/RKPPL
# instances, about 30 s per process). A fresh child loads both images, builds
# a model the workload does not build, defines methods of those shapes on its
# own types, and counts the RK/RKPPL instances they invalidate.
@testset "methods like JSON's do not invalidate RK/RKPPL construction" begin
    fixture = raw"""
        using ReactiveKernels, ReactiveKernelsPPL
        model = @rkppl begin
            a ~ Normal(0, 3)
            b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            y .~ Normal.(a .+ b .* x, sigma)
        end
        build_kernel(model(; x = [0.0, 1.0, 2.0]) | (; y = [0.2, 1.1, 1.9]))
        ours(m) = m isa Core.MethodInstance && m.def isa Method &&
            nameof(parentmodule(m.def)) in (:ReactiveKernels, :ReactiveKernelsPPL)
        function invalidated(ex)
            log = ccall(:jl_debug_method_invalidation, Any, (Cint,), 1)
            Core.eval(Main, ex)
            ccall(:jl_debug_method_invalidation, Any, (Cint,), 0)
            count(ours, log)
        end
        # The types and what an AbstractVector needs first, uncounted, so each
        # count is the one method's alone.
        Core.eval(Main, quote
            struct LazyVec <: AbstractVector{Any} end
            Base.size(::LazyVec) = (0,)
            Base.getindex(::LazyVec, ::Int) = nothing
            struct LazyVal end
            struct Selector end
        end)
        counts = (
            lastindex = invalidated(:(Base.lastindex(::LazyVec) = 0)),
            foreach = invalidated(:(Base.foreach(f, ::LazyVal) = nothing)),
            hasproperty = invalidated(:(Base.hasproperty(::Selector, ::Symbol) = false)),
        )
        print(counts.lastindex, " ", counts.foreach, " ", counts.hasproperty)
        """
    out = read(`$(Base.julia_cmd()) --startup-file=no
               --project=$(Base.active_project()) -e $fixture`, String)
    lastindex_n, foreach_n, hasproperty_n = parse.(Int, split(out))
    # Measured in session on RK main 9eb56917 (Julia 1.13.1): JSON's own
    # methods invalidated 1,906 (`lastindex`), 651 (`foreach`) and 1,325
    # (`hasproperty`) RK/RKPPL instances.
    @test foreach_n == 0
    @test hasproperty_n == 0
    @test lastindex_n == 0
end
