@testset "reactant suite named selectors" begin
    @test _select_reactant_suites(String[]) == _REACTANT_SUITE_ORDER
    @test _select_reactant_suites(collect(_REACTANT_SUITE_ORDER)) ==
        _REACTANT_SUITE_ORDER
    @test _select_reactant_suites(["core"]) == ("core",)
    @test _select_reactant_suites(["pathfinder", "core"]) ==
        ("core", "pathfinder")

    @test _select_reactant_suites(
        ["test_pathfinder_reactant", "test_pathfinder_reactant.jl"]) ==
        ("pathfinder",)
    @test _select_reactant_suites(
        ["test_nutpie_reactant.jl", "kernel-nuts"]) ==
        ("nutpie", "kernel-nuts")

    error = try
        _select_reactant_suites(["core", "missing", "../test_ad.jl"])
        nothing
    catch err
        err
    end
    @test error isa ArgumentError
    message = sprint(showerror, error)
    @test occursin(
        "unknown Reactant suite selector(s): \"../test_ad.jl\", \"missing\"",
        message)
    @test occursin(
        "Known groups: plate-chains, ref-array-plate, core, nutpie, " *
        "reactivehmc-statistics, reactivehmc-hmc, finite-structural-container, " *
        "kernel-nuts, pathfinder",
        message,
    )
    @test occursin(
        "Known files: test_authored_plate_chains_reactant.jl", message)
end
