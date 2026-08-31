@testset "runtests named selectors" begin
    @test _select_test_files(String[]) == _TEST_FILE_ORDER
    @test _select_test_files(["core", "acceptance"]) == _TEST_FILE_ORDER
    @test _select_test_files(["acceptance", "core"]) == _TEST_FILE_ORDER

    @test _select_test_files(["test_stateful", "test_ad.jl"]) ==
        ("test_ad.jl", "test_stateful.jl")
    @test _select_test_files(["test_ad", "test_ad.jl", "ad"]) ==
        ("test_package_boundary.jl", "test_ci_compiled_modules.jl", "test_ad.jl")
    @test _select_test_files(["benchmark"]) ==
        ("test_package_boundary.jl", "test_ci_compiled_modules.jl",
         "test_handwritten_benchmarks.jl")

    error = try
        _select_test_files(["test_stateful", "missing", "../test_ad.jl"])
        nothing
    catch err
        err
    end
    @test error isa ArgumentError
    message = sprint(showerror, error)
    @test occursin("unknown test selector(s): \"../test_ad.jl\", \"missing\"", message)
    @test occursin("Known groups: core, acceptance, ad, benchmark", message)
    @test occursin("Known files: test_package_boundary.jl", message)
end
