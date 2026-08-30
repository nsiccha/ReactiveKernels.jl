using Test

@testset "Pathfinder mathematical kernels are rendered in the docs" begin
    root = joinpath(@__DIR__, "..")
    page = read(joinpath(root, "docs", "src", "pathfinder.md"), String)
    make = read(joinpath(root, "docs", "make.jl"), String)
    examples = read(joinpath(root, "docs", "kernel_examples.jl"), String)
    rendered_gate = read(joinpath(root, "docs", "check_rendered.jl"), String)

    @test occursin("\"Pathfinder approximation\" => \"pathfinder.md\"", make)
    @test occursin("render_pathfinder_kernels(@__MODULE__)", page)
    @test occursin("The mathematics that gets transpiled", page)
    @test occursin("Curvature safeguard and α-recovery", page)
    @test occursin("Inverse-BFGS covariance", page)
    @test occursin("Local Gaussian and ELBO", page)
    @test occursin("Pathfinder.jl 0.10.7", page)
    @test occursin("dba8c9acc25f2905078d428ddd50b5d9276c3847", page)
    @test occursin("not a line-by-line port", page)
    @test occursin("Pathfinder.jl-like compact history", page)
    @test occursin("pathfinder_jl_compact_candidate", page)
    @test occursin("two chronological", page)
    @test occursin("not a sampler API", page)

    @test occursin("pathfinder_kernel_authoring_fixture.jl", examples)
    @test occursin("fixture.pathfinder_candidate", examples)
    @test occursin("fixture.PATHFINDER_CANDIDATE", examples)
    @test occursin("jl_fixture.PATHFINDER_JL_CANDIDATE", examples)
    @test occursin(":pathfinder_inverse_bfgs_geometry", examples)
    @test occursin(":pathfinder_local_gaussian_and_elbo", examples)
    @test occursin(":pathfinder_jl_compact_history", examples)
    @test occursin("\"pathfinder.md\" => 3", rendered_gate)
    for marker in ("Raw input", "Generated kernel", "Compute DAG")
        @test occursin(marker, rendered_gate)
    end
end
