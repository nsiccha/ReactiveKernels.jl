using Test

@testset "ReactiveHMC corpus docs use loaded source authorities exactly once" begin
    root = joinpath(@__DIR__, "..")
    page = read(joinpath(root, "docs", "src", "reactivehmc-corpus.md"), String)
    helpers = read(joinpath(root, "docs", "kernel_examples.jl"), String)
    make = read(joinpath(root, "docs", "make.jl"), String)
    rendered_check = read(joinpath(root, "docs", "check_rendered.jl"), String)

    @test !occursin(r"(?m)^\s*@kernel\s", page)
    for call in (
        "render_reactivehmc_inventory()",
        "render_reactivehmc_phasepoints()",
        "render_reactivehmc_captured_sources(:rke)",
        "render_reactivehmc_captured_sources(:integrators)",
        "render_reactivehmc_captured_sources(:statistics)",
        "render_reactivehmc_captured_sources(:hmc)",
    )
        @test length(split(page, call)) - 1 == 1
    end

    fixtures = (
        "reactivehmc_rke_kernel_fixture.jl" =>
            ("@kernel rke(",),
        "reactivehmc_integrator_kernel_fixture.jl" =>
            ("@kernel generalized_leapfrog!(", "@kernel implicit_midpoint!("),
        "reactivehmc_statistics_kernel_fixture.jl" =>
            ("@kernel statistics_state(",),
        "reactivehmc_hmc_kernel_fixture.jl" =>
            ("@kernel hmc_state(",),
    )
    for (file, markers) in fixtures
        source = read(joinpath(root, "benchmark", file), String)
        @test occursin(file, make)
        @test occursin(file, helpers)
        for marker in markers
            @test length(split(source, marker)) - 1 == 1
            @test occursin(marker, helpers)
        end
    end

    phasepoint_path = joinpath(
        root, "packages", "ReactiveKernelsCompatibilityExamples", "src",
        "preexisting_reactivehmc.jl",
    )
    phasepoints = read(phasepoint_path, String)
    @test length(split(phasepoints, "    @kernel spec(")) - 1 == 3
    @test occursin("preexisting_reactivehmc.jl", helpers)
    @test occursin("reactivehmc_artifacts()", helpers)

    @test occursin(
        "\"ReactiveHMC kernel corpus\" => \"reactivehmc-corpus.md\"",
        make,
    )
    @test occursin("assert_reactivehmc_docs_executed!()", make)
    @test occursin("\"reactivehmc-corpus.md\" => 6", rendered_check)
    @test occursin("\"reactivehmc-corpus.md\" => 5", rendered_check)
    @test occursin("inventory_count == 17", rendered_check)
end
