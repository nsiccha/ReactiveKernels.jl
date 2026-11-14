using ReactiveKernelsPPL, Test

# A package loaded after RK/RKPPL that adds a method to a Base function RK's
# construction code calls on a value inference cannot type invalidates the
# construction code RK's and RKPPL's images hold, so a fresh process
# recompiles lowering, binding and `build_kernel` on its first model (snag
# `downstream-workl-75f8cba4`: JSON 1.x's `lastindex` and `foreach` on its
# lazy values and StructUtils' `hasproperty` invalidated 3,882 RK/RKPPL
# instances, about 30 s per process; LibGit2's `IteratorSize` for its
# `GitBranchIter`, loaded with Pkg or Enzyme, about 12,000). A fresh child
# loads both images, builds a model the workload does not build, defines
# methods of those shapes on its own types, and counts the RK/RKPPL instances
# they invalidate.
@testset "methods like JSON's and LibGit2's do not invalidate RK/RKPPL construction" begin
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
            struct FakeIter end
            Base.iterate(::FakeIter, s = 1) = s > 2 ? nothing : (s, s + 1)
        end)
        counts = (
            lastindex = invalidated(:(Base.lastindex(::LazyVec) = 0)),
            foreach = invalidated(:(Base.foreach(f, ::LazyVal) = nothing)),
            hasproperty = invalidated(:(Base.hasproperty(::Selector, ::Symbol) = false)),
            iteratorsize = invalidated(:(Base.IteratorSize(::Type{FakeIter}) = Base.SizeUnknown())),
        )
        print(counts.lastindex, " ", counts.foreach, " ", counts.hasproperty, " ",
              counts.iteratorsize)
        """
    out = read(`$(Base.julia_cmd()) --startup-file=no
               --project=$(Base.active_project()) -e $fixture`, String)
    lastindex_n, foreach_n, hasproperty_n, iteratorsize_n = parse.(Int, split(out))
    # Measured in session on RK main 9eb56917 (Julia 1.13.1): JSON's own
    # methods invalidated 1,906 (`lastindex`), 651 (`foreach`) and 1,325
    # (`hasproperty`) RK/RKPPL instances.
    @test foreach_n == 0
    @test hasproperty_n == 0
    @test lastindex_n == 0
    # Measured in session after RK's construction workload and two Wren ports
    # on RK 5a2342956: 11,816 (`IteratorSize(::Type{FakeIter})`).
    @test iteratorsize_n == 0
end

# The same methods defined BEFORE the images load (JSON, Pkg or Enzyme
# imported first) reach construction another way: an image edge to a Base
# function whose matching methods changed fails verification when the image
# loads, and everything compiled on it is compiled again. Loading JSON first
# cost a cold WellsDist build 51-59 s of recompilation on RK 35f8068d9 (snag
# `downstream-workl-75f8cba4`), through `foreach(f, ::Any)` calls that the
# check above does not see; with Pkg first (LibGit2's `IteratorSize` and
# `sort!(::GitRevWalker)`) it cost 7-8 s. A fresh child defines the methods,
# loads both images and counts the RK/RKPPL code that fails verification; a
# control child defines only the types. Julia 1.12+ logs this through
# `Base.Compiler.ReinferUtils`.
@testset "methods like JSON's and LibGit2's defined first fail no RK/RKPPL image edge" begin
    R = isdefined(Base, :Compiler) && isdefined(Base.Compiler, :ReinferUtils) ?
        Base.Compiler.ReinferUtils : nothing
    if R === nothing || !isdefined(R, :debug_method_invalidation)
        @test_skip "the load-time verification log needs Julia 1.12+"
    else
        types = raw"""
            struct LazyVec <: AbstractVector{Any} end
            Base.size(::LazyVec) = (0,)
            Base.getindex(::LazyVec, ::Int) = nothing
            struct LazyVal end
            struct Selector end
            struct FakeIter end
            Base.iterate(::FakeIter, s = 1) = s > 2 ? nothing : (s, s + 1)
            struct Walker end
            """
        methods_like_json = raw"""
            Base.lastindex(::LazyVec) = 0
            Base.foreach(f, ::LazyVal) = nothing
            Base.hasproperty(::Selector, ::Symbol) = false
            Base.IteratorSize(::Type{FakeIter}) = Base.SizeUnknown()
            Base.sort!(w::Walker; kws...) = w
            """
        load = raw"""
            R = Base.Compiler.ReinferUtils
            R.debug_method_invalidation(true)
            using ReactiveKernels, ReactiveKernelsPPL
            log = R._jl_debug_method_invalidation[]
            R.debug_method_invalidation(false)
            owner(x) = x isa Core.CodeInstance ? owner(x.def) :
                (isdefined(Core, :ABIOverride) && x isa Core.ABIOverride) ? owner(x.def) :
                x isa Core.MethodInstance ? (x.def isa Method ? parentmodule(x.def) : x.def) :
                nothing
            ours(x) = (m = owner(x); m isa Module &&
                nameof(m) in (:ReactiveKernels, :ReactiveKernelsPPL))
            # Entries are `(code, "verify_methods", cause)` or
            # `(cause, reason, code, matches)`.
            function failed_codes(log)
                failed, i = 0, 1
                while i + 2 <= length(log)
                    reason = log[i+1]
                    if !(reason isa String)
                        i += 1
                        continue
                    end
                    code = reason == "verify_methods" ? log[i] : log[i+2]
                    failed += ours(code)
                    i += reason == "verify_methods" ? 3 : 4
                end
                return failed
            end
            print(failed_codes(log))
            """
        child(script) = parse(Int, read(`$(Base.julia_cmd()) --startup-file=no
            --project=$(Base.active_project()) -e $script`, String))
        control = child(types * load)
        defined_first = child(types * methods_like_json * load)
        @test defined_first == control
    end
end
