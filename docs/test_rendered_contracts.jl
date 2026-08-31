using Test

include(joinpath(@__DIR__, "check_rendered.jl"))

@testset "rendered docs fatal/advisory split" begin
    mktempdir() do root
        source_dir = joinpath(root, "src")
        build_dir = joinpath(root, "build")
        intermediate_dir = joinpath(build_dir, ".documenter")
        rendered_dir = joinpath(build_dir, "dev")
        foreach(mkpath, (source_dir, intermediate_dir, rendered_dir))

        write(joinpath(build_dir, "index.html"), "redirect")
        write(joinpath(source_dir, "fixture.md"), """
# Fixture

| A | B |
| - | - |
""")

        artifact_markup = """
<div class="rk-example" data-rk-artifact-id="example:first" data-rk-artifact-kind="example-panel"></div>
<div class="rk-example" data-rk-artifact-id="example:second" data-rk-artifact-kind="example-panel"></div>
<section class="rk-result-table-section" data-rk-artifact-id="table:first" data-rk-artifact-kind="sortable-table"><htmxo-sortable-table></htmxo-sortable-table></section>
<section class="rk-result-table-section" data-rk-artifact-id="table:second" data-rk-artifact-kind="sortable-table"><htmxo-sortable-table></htmxo-sortable-table></section>
<div class="rk-aov-panel" data-rk-artifact-id="plot:first" data-rk-artifact-kind="aov-panel" v-exec-scripts="'cGF5bG9hZA=='"></div>
"""
        intermediate_path = joinpath(intermediate_dir, "fixture.md")
        rendered_path = joinpath(rendered_dir, "fixture.html")
        write(intermediate_path, artifact_markup)
        write(rendered_path, "<!DOCTYPE html><html><body>$artifact_markup</body></html>")

        contracts = (
            panels = Dict("fixture.md" => 1),
            source_examples = Dict{String,Int}(),
            source_interactions = Dict{String,Int}(),
            sortable_tables = Dict("fixture.md" => 1),
            aov_panels = Dict("fixture.md" => 0),
            structural_markers = Dict("fixture.md" => ("Expected intermediate heading",)),
            body_markers = Dict("fixture.md" => ("Expected rendered heading",)),
        )
        report_path = joinpath(root, "rendered-docs-advisories.json")
        initialize_rendered_docs_report(report_path)
        @test occursin("\"status\": \"pending\"", read(report_path, String))
        advisories = check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts, github_actions = false,
        )

        @test length(advisories) == 7
        @test Set(advisory.contract_kind for advisory in advisories) == Set((
            "raw-markdown-table-count",
            "executable-panel-count",
            "sortable-table-count",
            "aov-panel-count",
            "intermediate-marker",
            "rendered-heading-or-marker",
            "total-executable-panel-count",
        ))
        @test all(advisory.responsible_agent == "ReactiveKernels:docs" for advisory in advisories)
        @test any("example:first" in advisory.artifact_ids for advisory in advisories)

        report = read(report_path, String)
        @test occursin("reactive-kernels-rendered-docs-advisories-v1", report)
        @test occursin("\"status\": \"advisory\"", report)
        @test occursin("\"fatal_error\": null", report)
        @test occursin("\"contract_kind\": \"sortable-table-count\"", report)
        @test occursin("\"artifact_ids\": [\"example:first\"", report)
        @test occursin("\"responsible_agent\": \"ReactiveKernels:docs\"", report)

        annotation = _github_annotation(first(advisories))
        @test occursin("::warning file=docs/src/fixture.md", annotation)

        duplicate = artifact_markup *
            "<div data-rk-artifact-id=\"example:first\" data-rk-artifact-kind=\"example-panel\"></div>\n"
        write(intermediate_path, duplicate)
        write(rendered_path, "<!DOCTYPE html><html><body>$duplicate</body></html>")
        @test_throws ErrorException check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts, github_actions = false,
        )
        fatal_report = read(report_path, String)
        @test occursin("\"status\": \"fatal\"", fatal_report)
        @test occursin("duplicates stable artifact ids", fatal_report)

        missing_payload = replace(
            artifact_markup,
            " v-exec-scripts=\"'cGF5bG9hZA=='\"" => "",
        )
        write(intermediate_path, artifact_markup)
        write(rendered_path, "<!DOCTYPE html><html><body>$missing_payload</body></html>")
        @test_throws ErrorException check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts, github_actions = false,
        )
        @test occursin(
            "VitePress output interactive artifact plot:first",
            read(report_path, String),
        )

        write(intermediate_path, missing_payload)
        write(rendered_path, "<!DOCTYPE html><html><body>$missing_payload</body></html>")
        @test_throws ErrorException check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts, github_actions = false,
        )

        write(intermediate_path, artifact_markup)
        missing_id = replace(
            artifact_markup,
            " data-rk-artifact-id=\"plot:first\"" => "",
        )
        write(rendered_path, "<!DOCTYPE html><html><body>$missing_id</body></html>")
        @test_throws ErrorException check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts, github_actions = false,
        )

        write(intermediate_path, missing_id)
        write(rendered_path, "<!DOCTYPE html><html><body>$missing_id</body></html>")
        @test_throws ErrorException check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts, github_actions = false,
        )
        @test occursin("aov-panel roots but 0 stable declarations", read(report_path, String))

        write(joinpath(source_dir, "fixture.md"), "# Fixture\n")
        write(intermediate_path, artifact_markup)
        write(rendered_path, "<!DOCTYPE html><html><body>$artifact_markup</body></html>")
        matching_contracts = (
            panels = Dict("fixture.md" => 2),
            source_examples = Dict{String,Int}(),
            source_interactions = Dict{String,Int}(),
            sortable_tables = Dict("fixture.md" => 4),
            aov_panels = Dict("fixture.md" => 1),
            structural_markers = Dict{String,Tuple}(),
            body_markers = Dict{String,Tuple}(),
        )
        @test isempty(check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts = matching_contracts,
            github_actions = false,
        ))
        clean_report = read(report_path, String)
        @test occursin("\"status\": \"clean\"", clean_report)
        @test occursin("\"advisories\": [\n  ]", clean_report)

        write(rendered_path, "<html><body>$artifact_markup</body></html>")
        @test_throws ErrorException check_rendered_docs(
            build_dir, ["Fixture" => "fixture.md"];
            source_dir, report_path, contracts = matching_contracts,
            github_actions = false,
        )
        @test occursin("\"status\": \"fatal\"", read(report_path, String))
    end
end
