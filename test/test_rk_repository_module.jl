module TestRKRepositoryModule

using ReactiveKernels
using Test

# A package outside ReactiveKernels' repository whose name nevertheless starts
# with "ReactiveKernels", as ReactiveKernelMethods' packages' names do. It is
# loaded from a temporary implicit environment and not precompiled, so no
# cache file is written.
const LOOKALIKE_UUID = Base.UUID("5f3a6a52-6b3e-4d1c-9a51-7f6e0c2b9d41")
const LOOKALIKE = let dir = mktempdir()
    src = mkpath(joinpath(dir, "ReactiveKernelsLookalike", "src"))
    write(joinpath(dir, "ReactiveKernelsLookalike", "Project.toml"),
          "name = \"ReactiveKernelsLookalike\"\nuuid = \"$(LOOKALIKE_UUID)\"\n")
    write(joinpath(src, "ReactiveKernelsLookalike.jl"), """
        module ReactiveKernelsLookalike
        __precompile__(false)
        # A concrete result, computed by a call dispatched at run time.
        double(x::Float64) = (Base.inferencebarrier(x) * 2)::Float64
        end
        """)
    pushfirst!(LOAD_PATH, dir)
    try
        Base.require(Base.PkgId(LOOKALIKE_UUID, "ReactiveKernelsLookalike"))
    finally
        filter!(!=(dir), LOAD_PATH)
    end
end

@kernel uses_lookalike(x::Float64) = begin
    y::Float64 = LOOKALIKE.double(x)
    return y
end

@testset "RK's own packages are its repository's, by package identity" begin
    own = ReactiveKernels._rk_repository_module
    @test own(ReactiveKernels)
    @test !own(Test)
    @test Base.moduleroot(LOOKALIKE) === LOOKALIKE
    @test startswith(string(nameof(LOOKALIKE)), "ReactiveKernels")
    @test !own(LOOKALIKE)
    # The `:rk` audit recurses into RK's methods only.
    recurse = ReactiveKernels._code_recursion(Val(:rk))
    @test recurse(first(methods(ReactiveKernels.prepare)))
    @test !recurse(first(methods(LOOKALIKE.double)))
end

@testset "`:rk` treats a ReactiveKernels*-named package outside the repository as a leaf" begin
    # `double` returns a concrete type, so as a leaf it is silent under `:rk`;
    # `:all` audits its body and finds the run-time dispatch.
    @test_logs min_level = Base.CoreLogging.Warn prepare(uses_lookalike; type_stability_level = :rk)
    @test_logs (:warn, r"ReactiveKernels: .*double @ .*is a call dispatched at run time"s) prepare(
        uses_lookalike; type_stability_level = :all)
end

end
