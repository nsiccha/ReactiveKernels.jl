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
    length(rendered) == length(sources) ||
        error("rendered site has $(length(rendered)) content pages; navigation config has $(length(sources))")
    expected_panels = Dict(
        "distributions.md" => 11,
        "batched.md" => 1,
        "eight-schools.md" => 1,
        "linear-regression.md" => 1,
        "beta-binomial.md" => 1,
        "poisson-gamma.md" => 1,
        "dugongs-growth.md" => 1,
        "arma11.md" => 1,
        "gaussian-mixture.md" => 1,
        "online-stats.md" => 2,
    )

    observed_panels = 0
    for source in sources
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

        if source == "eight-schools.md"
            for marker in ("Eight Schools Density", "Raw input", "Generated kernel", "Compute DAG")
                occursin(marker, body) || error("Eight Schools page is missing marker: $marker")
            end
        end
    end

    expected_total = sum(values(expected_panels))
    observed_panels == expected_total ||
        error("rendered site has $observed_panels executable panels; expected $expected_total")
    @info "Rendered docs structure verified" pages=length(sources) executable_panels=observed_panels
    return nothing
end
