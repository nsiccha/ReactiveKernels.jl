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

function _check_warning_banner_absent(build_dir)
    rendered_assets = String[]
    for (root, _, files) in walkdir(build_dir), file in files
        any(endswith(file, suffix) for suffix in (".html", ".css", ".js")) &&
            push!(rendered_assets, joinpath(root, file))
    end
    isempty(rendered_assets) && error("rendered site contains no HTML/CSS/JS assets")
    rendered = join((read(path, String) for path in rendered_assets), "\n")
    occursin("warning-banner", rendered) &&
        error("rendered site must not contain the removed warning banner")
    occursin("You are viewing the dev branch", rendered) &&
        error("rendered site must not contain the removed dev-branch warning")
    nothing
end

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
        "bijectors.md" => 1,
        "pathfinder.md" => 3,
        "reactivehmc-corpus.md" => 6,
        "nuts.md" => 2,
        "nutpie-diagonal.md" => 2,
        "eight-schools.md" => 1,
        "linear-regression.md" => 1,
        "beta-binomial.md" => 1,
        "poisson-gamma.md" => 1,
        "dugongs-growth.md" => 1,
        "arma11.md" => 1,
        "gaussian-mixture.md" => 1,
        "online-stats.md" => 1,
    )
    expected_source_examples = Dict(
        "reactivehmc-corpus.md" => 5,
    )
    expected_source_interactions = Dict(
        "reactivehmc-corpus.md" => 5,
        "nuts.md" => 1,
        "walnuts.md" => 1,
    )
    expected_sortable_tables = Dict(
        "batched.md" => 1,
        "compiler.md" => 1,
        "distributions.md" => 6,
        "eval-throughput.md" => 1,
        "nuts.md" => 3,
    )
    expected_aov_panels = Dict(
        "batched.md" => 1,
        "distributions.md" => 9,
        "nuts.md" => 2,
    )
    structural_markers = Dict(
        "compiler.md" => ("class=\"rk-pipeline\"",),
        "nuts.md" => (
            "class=\"rk-status-grid\"",
            "Full compiled NUTS kernel",
            "Authored nuts!! native entry",
            "result = fixture.nuts!!(sampler; rng = Random.Xoshiro(1))",
            "Receipt medians for the matched-control corpus",
            "Compilation and first-call costs",
        ),
        "nutpie-diagonal.md" => (
            "@kernel nutpie_diagonal_initialize",
            "@kernel nutpie_diagonal_adaptation",
            "97be9ab88cfaadfafd9e5f4409a3b1d5af62805a",
        ),
        "pathfinder.md" => (
            "Pathfinder Inverse Bfgs Geometry",
            "Pathfinder Local Gaussian And Elbo",
            "Pathfinder Jl Compact History",
            "# Exact build-executed constructor / prepare / call",
            "fixture.pathfinder_candidate",
            "fixture.pathfinder_jl_compact_candidate",
            "output = kernel(Tuple(inputs)...)",
        ),
        "reactivehmc-corpus.md" => (
            "ReactiveHMC kernel corpus",
            "Relativistic kinetic energy",
            "Generalized leapfrog",
            "Implicit midpoint",
            "Trajectory and sampling statistics",
            "Fixed-step HMC",
            "# Exact build-executed constructor / prepare / call",
            "compiled = RK.compile_stateful(",
            "ReactiveKernels.compile_state_transition(",
            "Fixture construction / MethodIR / independent-receipt inspection only",
            "compiler_execution_claimed = false",
            "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
        ),
        "walnuts.md" => (
            "WALNUTS-D as mathematical `@kernel` source",
            "Fixed macro time, dyadic micro grids",
            "macro_step!(ep) = begin",
            "reverse_num_steps = div(reverse_num_steps, 2)",
            "step!(directions, exponentials) = begin",
            "accepted = macro_step!(__self__, ep)",
            "@kernel walnuts!!(state; momentum, directions, exponentials)",
            "WALNUTS depth-10 compiler frontier",
            "captured_method_count",
            "configured_max_depth = 10",
            "proposal_capacity = 12",
            "tree_capacity = 11",
            "structural_container_roundtrip = true",
            "compiler_frontier_executed = true",
            "recursive functional state-machine SCC lowering is not implemented",
            "compiler_execution_claimed = false",
            "Complete authored WALNUTS-D fixture",
        ),
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

        source_examples = _occurrences(body, "class=\"rk-source-example\"")
        expected_sources = get(expected_source_examples, source, 0)
        source_examples == expected_sources || error(
            "rendered $source has $source_examples captured-source examples; expected $expected_sources",
        )

        source_interactions = _occurrences(body, "class=\"rk-source-interaction\"") +
            _occurrences(body, "class=\"rk-source-example\"")
        expected_interactions = get(expected_source_interactions, source, 0)
        source_interactions == expected_interactions || error(
            "rendered $source has $source_interactions source-locked interactions; expected $expected_interactions",
        )

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

        if source == "reactivehmc-corpus.md"
            inventory_count = _occurrences(intermediate, "data-rk-corpus-id=")
            inventory_count == 17 || error(
                "ReactiveHMC corpus page rendered $inventory_count inventory entries; expected 17",
            )
            for marker in (
                    "data-rk-source-authority=\"relativistic_kinetic_energy\"",
                    "data-rk-source-authority=\"generalized_leapfrog\"",
                    "data-rk-source-authority=\"implicit_midpoint\"",
                    "data-rk-source-authority=\"statistics_state\"",
                    "data-rk-source-authority=\"fixed_step_hmc\"",
                )
                count = _occurrences(intermediate, marker)
                count == 1 || error(
                    "ReactiveHMC corpus marker $marker occurred $count times; expected exactly once",
                )
            end
            for marker in (
                    "data-rk-interaction=\"relativistic_kinetic_energy\"",
                    "data-rk-interaction=\"generalized_leapfrog\"",
                    "data-rk-interaction=\"implicit_midpoint\"",
                    "data-rk-interaction=\"statistics_state\"",
                    "data-rk-interaction=\"fixed_step_hmc\"",
                )
                count = _occurrences(intermediate, marker)
                count == 1 || error(
                    "ReactiveHMC interaction marker $marker occurred $count times; expected exactly once",
                )
            end
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
        if source == "nutpie-diagonal.md"
            for marker in ("Nutpie Diagonal Initialize", "Nutpie Diagonal Adaptation",
                           "Raw input", "Generated kernel", "Compute DAG")
                occursin(marker, body) ||
                    error("nutpie diagonal page is missing marker: $marker")
            end
        end
        if source == "pathfinder.md"
            for marker in (
                    "Pathfinder Inverse Bfgs Geometry",
                    "Pathfinder Local Gaussian And Elbo",
                    "Pathfinder Jl Compact History",
                    "Raw input",
                    "Generated kernel",
                    "Compute DAG",
                )
                occursin(marker, body) || error("Pathfinder page is missing marker: $marker")
            end
        end
    end

    expected_total = sum(values(expected_panels))
    observed_panels == expected_total ||
        error("rendered site has $observed_panels executable panels; expected $expected_total")
    _check_warning_banner_absent(build_dir)
    @info "Rendered docs structure verified" pages=length(sources) executable_panels=observed_panels
    return nothing
end
