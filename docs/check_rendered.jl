function _documented_sources(entries)
    sources = String[]
    for entry in entries
        value = last(entry)
        if value isa AbstractString
            push!(sources, value)
        elseif value isa AbstractVector
            append!(sources, _documented_sources(value))
        else
            error("unsupported docs page entry: $(repr(entry))")
        end
    end
    return sources
end

function _rendered_html(build_dir)
    pages = String[]
    root_redirect = normpath(joinpath(build_dir, "index.html"))
    for (root, _, files) in walkdir(build_dir), file in files
        endswith(file, ".html") || continue
        path = normpath(joinpath(root, file))
        path == root_redirect && continue
        file == "404.html" && continue
        push!(pages, path)
    end
    return pages
end

_occurrences(text, needle) = length(split(text, needle)) - 1

"""Fail the docs build if a configured page or executable example vanished."""
function check_rendered_docs(build_dir, page_tree)
    isdir(build_dir) || error("docs build directory is missing: $build_dir")

    root_redirect = joinpath(build_dir, "index.html")
    isfile(root_redirect) && filesize(root_redirect) > 0 ||
        error("docs root redirect is missing or empty: $root_redirect")

    rendered = _rendered_html(build_dir)
    sources = _documented_sources(page_tree)
    source_dir = joinpath(@__DIR__, "src")
    intermediate_dir = joinpath(build_dir, ".documenter")
    length(rendered) == length(sources) ||
        error("rendered site has $(length(rendered)) content pages; navigation config has $(length(sources))")
    expected_panels = Dict(
        "distributions.md" => 11,
        "batched.md" => 1,
        "nuts.md" => 2,
        "eight-schools.md" => 1,
        "linear-regression.md" => 1,
        "beta-binomial.md" => 1,
        "poisson-gamma.md" => 1,
        "dugongs-growth.md" => 1,
        "arma11.md" => 1,
        "gaussian-mixture.md" => 1,
        "online-stats.md" => 2,
    )
    expected_sortable_tables = Dict(
        "batched.md" => 1,
        "compiler.md" => 1,
        "distributions.md" => 5,
        "nuts.md" => 1,
    )
    expected_aov_panels = Dict(
        "batched.md" => 1,
        "distributions.md" => 9,
        "nuts.md" => 1,
    )
    structural_markers = Dict(
        "compiler.md" => ("class=\"rk-pipeline\"",),
        "nuts.md" => ("class=\"rk-status-grid\"", "Full compiled NUTS kernel"),
        "visualization.md" =>
            ("class=\"rk-dag-legend\"", "class=\"rk-comparison-grid\""),
    )

    observed_panels = 0
    for source in sources
        source_body = read(joinpath(source_dir, source), String)
        for (line_number, line) in enumerate(eachline(IOBuffer(source_body)))
            occursin(r"^\s*\|.*\|\s*$", line) && error(
                "raw Markdown table syntax remains in $source:$line_number",
            )
        end

        html_name = replace(basename(source), r"\.md$" => ".html")
        matches = filter(path -> basename(path) == html_name, rendered)
        length(matches) == 1 ||
            error("expected exactly one rendered $source page, found $(length(matches)): $(join(matches, ", "))")

        body = read(only(matches), String)
        filesize(only(matches)) > 0 || error("rendered $source page is empty")
        occursin("<!DOCTYPE html>", body) || error("rendered $source page is not complete HTML")

        expected = get(expected_panels, source, 0)
        observed = _occurrences(body, "class=\"rk-example\"")
        observed == expected ||
            error("rendered $source has $observed executable panels; expected $expected")
        observed_panels += observed

        intermediate_path = joinpath(intermediate_dir, source)
        isfile(intermediate_path) ||
            error("DocumenterVitepress intermediate page is missing: $intermediate_path")
        intermediate = read(intermediate_path, String)
        sortable_tables = _occurrences(intermediate, "htmxo-sortable-table")
        expected_tables = get(expected_sortable_tables, source, 0)
        sortable_tables == expected_tables || error(
            "rendered $source has $sortable_tables sortable result tables; expected $expected_tables",
        )
        aov_panels = _occurrences(intermediate, "class=\"rk-aov-panel\"")
        expected_plots = get(expected_aov_panels, source, 0)
        aov_panels == expected_plots || error(
            "rendered $source has $aov_panels AoV panels; expected $expected_plots",
        )
        for marker in get(structural_markers, source, ())
            occursin(marker, intermediate) ||
                error("rendered $source is missing structural result marker: $marker")
        end

        if source == "eight-schools.md"
            for marker in ("Eight Schools Extraction", "Raw input", "Generated kernel", "Compute DAG")
                occursin(marker, body) || error("Eight Schools page is missing marker: $marker")
            end
        end
        if source == "nuts.md"
            for marker in ("NUTS Phasepoint Hamiltonian", "Raw input", "Generated kernel", "Compute DAG")
                occursin(marker, body) || error("NUTS page is missing three-pane marker: $marker")
            end
        end
    end

    expected_total = sum(values(expected_panels))
    observed_panels == expected_total ||
        error("rendered site has $observed_panels executable panels; expected $expected_total")
    @info "Rendered docs structure verified" pages=length(sources) executable_panels=observed_panels
    return nothing
end
