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

const _RENDERED_DOCS_PAGE_OWNERS = Dict(
    "automatic-differentiation.md" => "ReactiveKernels:enzyme",
    "batched.md" => "ReactiveKernels:batching",
    "distributions.md" => "ReactiveKernels:distributions",
    "nuts.md" => "ReactiveKernels:hmc",
    "walnuts.md" => "ReactiveKernels:hmc",
)

_page_owner(page) = get(_RENDERED_DOCS_PAGE_OWNERS, page, "ReactiveKernels:docs")

function _record_advisory!(advisories; page, contract_kind, expected, observed,
                           artifact_ids = String[], detail = "")
    push!(advisories, (;
        page = String(page),
        responsible_agent = _page_owner(page),
        contract_kind = String(contract_kind),
        expected,
        observed,
        artifact_ids = sort!(unique!(String.(artifact_ids))),
        detail = String(detail),
    ))
    advisories
end

_json_escape(value) = replace(
    String(value), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r",
    '\t' => "\\t",
)
_json(value::AbstractString) = "\"$(_json_escape(value))\""
_json(value::Integer) = string(value)
_json(value::Bool) = value ? "true" : "false"
_json(::Nothing) = "null"
_json(values::AbstractVector) = "[" * join(_json.(values), ", ") * "]"
_json(value) = _json(string(value))

function _write_advisory_report(path, advisories; status, fatal_error = nothing)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"reactive-kernels-rendered-docs-advisories-v1\",")
        println(io, "  \"status\": $(_json(status)),")
        println(io, "  \"source_commit\": $(_json(get(ENV, "GITHUB_SHA", ""))),")
        println(io, "  \"workflow_run_id\": $(_json(get(ENV, "GITHUB_RUN_ID", ""))),")
        println(io, "  \"fatal_error\": $(_json(fatal_error)),")
        println(io, "  \"advisories\": [")
        for (index, advisory) in enumerate(advisories)
            suffix = index == length(advisories) ? "" : ","
            println(io, "    {")
            println(io, "      \"page\": $(_json(advisory.page)),")
            println(io, "      \"responsible_agent\": $(_json(advisory.responsible_agent)),")
            println(io, "      \"contract_kind\": $(_json(advisory.contract_kind)),")
            println(io, "      \"expected\": $(_json(advisory.expected)),")
            println(io, "      \"observed\": $(_json(advisory.observed)),")
            println(io, "      \"artifact_ids\": $(_json(advisory.artifact_ids)),")
            println(io, "      \"detail\": $(_json(advisory.detail))")
            println(io, "    }$suffix")
        end
        println(io, "  ]")
        println(io, "}")
    end
    path
end

function initialize_rendered_docs_report(path = get(
        ENV, "RK_RENDERED_DOCS_REPORT",
        joinpath(@__DIR__, "build", ".reports", "rendered-docs-advisories.json"),
    ))
    _write_advisory_report(path, NamedTuple[]; status = "pending")
end

function _github_escape(value)
    replace(string(value), '%' => "%25", '\r' => "%0D", '\n' => "%0A")
end

function _advisory_message(advisory)
    ids = isempty(advisory.artifact_ids) ? "none declared" :
        join(advisory.artifact_ids, ", ")
    string(
        advisory.contract_kind, " on ", advisory.page, ": expected ",
        advisory.expected, ", observed ", advisory.observed,
        "; stable artifacts: ", ids,
        isempty(advisory.detail) ? "" : "; $(advisory.detail)",
    )
end

function _github_annotation(advisory)
    file = advisory.page == "<site>" ? "docs/check_rendered.jl" :
        "docs/src/$(advisory.page)"
    string(
        "::warning file=", _github_escape(file),
        ",title=Rendered docs presentation drift::",
        _github_escape(_advisory_message(advisory)),
    )
end

function _emit_advisories(advisories; github_actions = get(ENV, "GITHUB_ACTIONS", "") == "true")
    for advisory in advisories
        @warn "Rendered docs presentation drift" page=advisory.page responsible_agent=advisory.responsible_agent contract_kind=advisory.contract_kind expected=advisory.expected observed=advisory.observed artifact_ids=advisory.artifact_ids
        github_actions && println(_github_annotation(advisory))
    end
    nothing
end

const _ARTIFACT_ROOT_CLASSES = (
    (kind = "aov-panel", class = "rk-aov-panel"),
    (kind = "result-assets", class = "rk-result-assets"),
    (kind = "sortable-table", class = "rk-result-table-section"),
    (kind = "example-panel", class = "rk-example"),
    (kind = "source-interaction", class = "rk-source-interaction"),
    (kind = "source-example", class = "rk-source-example"),
)

function _attribute_values(tag, name)
    [match.captures[1] for match in eachmatch(Regex("\\s$name=\"([^\"]*)\""), tag)]
end

function _artifact_declarations(body, stage, source)
    declarations = NamedTuple[]
    for tag_match in eachmatch(r"<[^>]+>", body)
        tag = tag_match.match
        class_attributes = _attribute_values(tag, "class")
        length(class_attributes) <= 1 ||
            error("$stage $source page has a tag with duplicate class attributes")
        classes = isempty(class_attributes) ? String[] : split(only(class_attributes))
        root_kinds = [root.kind for root in _ARTIFACT_ROOT_CLASSES if root.class in classes]
        ids = _attribute_values(tag, "data-rk-artifact-id")
        kinds = _attribute_values(tag, "data-rk-artifact-kind")
        payloads = _attribute_values(tag, "data-rk-exec-payload")

        if isempty(root_kinds)
            isempty(ids) && isempty(kinds) && isempty(payloads) || error(
                "$stage $source page declares stable artifact attributes on a non-root tag",
            )
            continue
        end
        length(root_kinds) == 1 || error(
            "$stage $source page combines multiple semantic artifact root classes on one tag",
        )
        length(ids) == 1 || error(
            "$stage $source $(only(root_kinds)) root must declare exactly one stable artifact id",
        )
        length(kinds) == 1 || error(
            "$stage $source $(only(root_kinds)) root must declare exactly one semantic kind",
        )
        isempty(only(ids)) && error("$stage $source page declares an empty stable artifact id")
        only(kinds) == only(root_kinds) || error(
            "$stage $source root kind $(only(kinds)) does not match class kind $(only(root_kinds))",
        )
        if only(kinds) in ("aov-panel", "result-assets")
            length(payloads) == 1 || error(
                "$stage interactive artifact $(only(ids)) on $source must declare exactly one payload witness",
            )
            isempty(only(payloads)) && error(
                "$stage interactive artifact $(only(ids)) on $source has an empty payload witness",
            )
        else
            isempty(payloads) || error(
                "$stage non-interactive artifact $(only(ids)) on $source declares an executable payload",
            )
        end
        push!(declarations, (;
            id = only(ids),
            kind = only(kinds),
            payload = isempty(payloads) ? nothing : only(payloads),
            tag,
        ))
    end
    declarations
end

function _check_artifact_id_contract(source, intermediate, rendered)
    intermediate_declarations =
        _artifact_declarations(intermediate, "Documenter intermediate", source)
    rendered_declarations = _artifact_declarations(rendered, "VitePress output", source)
    for (stage, declarations, stage_body) in (
            ("Documenter intermediate", intermediate_declarations, intermediate),
            ("VitePress output", rendered_declarations, rendered),
        )
        ids = getproperty.(declarations, :id)
        any(isempty, ids) && error("$stage $source page declares an empty stable artifact id")
        any(declaration -> isempty(declaration.kind), declarations) &&
            error("$stage $source page declares a stable artifact without a semantic kind")
        duplicates = sort!([id for id in unique(ids) if count(==(id), ids) > 1])
        isempty(duplicates) || error(
            "$stage $source page duplicates stable artifact ids: $(join(duplicates, ", "))",
        )
        for declaration in declarations
            declaration.kind in ("aov-panel", "result-assets") || continue
            if stage == "Documenter intermediate"
                binding = "v-exec-scripts=\"'$(declaration.payload)'\""
                occursin(binding, stage_body) || error(
                    "$stage interactive artifact $(declaration.id) on $source has no matching runtime binding",
                )
            end
        end
    end
    intermediate_ids = getproperty.(intermediate_declarations, :id)
    rendered_ids = getproperty.(rendered_declarations, :id)
    Set(intermediate_ids) == Set(rendered_ids) || error(
        "rendered $source did not preserve declared stable artifact ids; " *
        "intermediate=$(sort(intermediate_ids)), output=$(sort(rendered_ids))",
    )
    sort!(intermediate_ids)
end

_contract_override(overrides, name, default) =
    isnothing(overrides) || !hasproperty(overrides, name) ? default : getproperty(overrides, name)

function _check_warning_banner_absent(build_dir, advisories)
    rendered_assets = String[]
    for (root, _, files) in walkdir(build_dir), file in files
        any(endswith(file, suffix) for suffix in (".html", ".css", ".js")) &&
            push!(rendered_assets, joinpath(root, file))
    end
    isempty(rendered_assets) && error("rendered site contains no HTML/CSS/JS assets")
    rendered = join((read(path, String) for path in rendered_assets), "\n")
    for marker in ("warning-banner", "You are viewing the dev branch")
        occurrences = _occurrences(rendered, marker)
        occurrences == 0 || _record_advisory!(
            advisories; page = "<site>", contract_kind = "removed-banner-marker",
            expected = 0, observed = occurrences, artifact_ids = ["site-assets"],
            detail = "marker: $marker",
        )
    end
    nothing
end

function _check_rendered_docs!(advisories, build_dir, page_tree;
                               source_dir = joinpath(@__DIR__, "src"),
                               contracts = nothing)
    isdir(build_dir) || error("docs build directory is missing: $build_dir")

    root_redirect = joinpath(build_dir, "index.html")
    isfile(root_redirect) && filesize(root_redirect) > 0 ||
        error("docs root redirect is missing or empty: $root_redirect")

    rendered = _rendered_html(build_dir)
    sources = _documented_sources(page_tree)
    intermediate_dir = joinpath(build_dir, ".documenter")
    length(rendered) == length(sources) ||
        error("rendered site has $(length(rendered)) content pages; navigation config has $(length(sources))")
    expected_panels = Dict(
        "automatic-differentiation.md" => 1,
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
        "automatic-differentiation.md" => 4,
        "batched.md" => 2,
        "compiler.md" => 1,
        "distributions.md" => 6,
        "eight-schools.md" => 1,
        "eight-schools-reactant.md" => 2,
        "eval-throughput.md" => 1,
        "nuts.md" => 3,
    )
    expected_aov_panels = Dict(
        "automatic-differentiation.md" => 8,
        "batched.md" => 2,
        "distributions.md" => 12,
        "nuts.md" => 2,
    )
    structural_markers = Dict(
        "automatic-differentiation.md" => (
            "Automatic differentiation",
            "Prepare once, then request gradients or value-and-gradient",
            "returned_gradient === gradient_buffer",
            "Normal Loglik",
            "Distribution gradient latency and allocation",
            "distribution-gradient-v1.toml",
            "20 have a zero-byte steady-state gradient path",
            "Eight Schools model gradient matrix",
            "eight-schools-ad-v1.toml",
            "All four differentiable scalar cells",
        ),
        "compiler.md" => ("class=\"rk-pipeline\"",),
        "eight-schools-reactant.md" => (
            "Native RK / Reactant steady-state matrix",
            "Setup, compilation, and first-call costs",
            "benchmark/receipts/eight-schools-reactant-v1.toml",
            "packages/ReactiveKernelsPPLExamples/src/eight_schools.jl",
        ),
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
    body_markers = Dict(
        "eight-schools.md" =>
            ("Eight Schools Extraction", "Raw input", "Generated kernel", "Compute DAG",
             "Primal boundary × outcome matrix"),
        "nuts.md" =>
            ("NUTS Phasepoint Hamiltonian", "Raw input", "Generated kernel", "Compute DAG"),
        "nutpie-diagonal.md" => (
            "Nutpie Diagonal Initialize", "Nutpie Diagonal Adaptation",
            "Raw input", "Generated kernel", "Compute DAG",
        ),
        "pathfinder.md" => (
            "Pathfinder Inverse Bfgs Geometry",
            "Pathfinder Local Gaussian And Elbo",
            "Pathfinder Jl Compact History",
            "Raw input",
            "Generated kernel",
            "Compute DAG",
        ),
    )

    expected_panels = _contract_override(contracts, :panels, expected_panels)
    expected_source_examples =
        _contract_override(contracts, :source_examples, expected_source_examples)
    expected_source_interactions =
        _contract_override(contracts, :source_interactions, expected_source_interactions)
    expected_sortable_tables =
        _contract_override(contracts, :sortable_tables, expected_sortable_tables)
    expected_aov_panels = _contract_override(contracts, :aov_panels, expected_aov_panels)
    structural_markers = _contract_override(contracts, :structural_markers, structural_markers)
    body_markers = _contract_override(contracts, :body_markers, body_markers)

    observed_panels = 0
    all_artifact_ids = String[]
    for source in sources
        source_body = read(joinpath(source_dir, source), String)
        raw_table_lines = [
            line_number for (line_number, line) in enumerate(eachline(IOBuffer(source_body)))
            if occursin(r"^\s*\|.*\|\s*$", line)
        ]
        isempty(raw_table_lines) || _record_advisory!(
            advisories; page = source, contract_kind = "raw-markdown-table-count",
            expected = 0, observed = length(raw_table_lines),
            artifact_ids = ["source-line-$line" for line in raw_table_lines],
            detail = "prefer source-authoritative AoV/HTMXO renderers",
        )

        intermediate_path = joinpath(intermediate_dir, source)
        isfile(intermediate_path) ||
            error("DocumenterVitepress intermediate page is missing: $intermediate_path")
        intermediate = read(intermediate_path, String)

        html_name = replace(basename(source), r"\.md$" => ".html")
        matches = filter(path -> basename(path) == html_name, rendered)
        length(matches) == 1 || error(
            "expected exactly one rendered $source page, found $(length(matches)): " *
            join(matches, ", "),
        )

        rendered_path = only(matches)
        body = read(rendered_path, String)
        filesize(rendered_path) > 0 || error("rendered $source page is empty")
        occursin("<!DOCTYPE html>", body) || error("rendered $source page is not complete HTML")

        artifact_ids = _check_artifact_id_contract(source, intermediate, body)
        append!(all_artifact_ids, ("$source::$id" for id in artifact_ids))

        expected = get(expected_panels, source, 0)
        observed = _occurrences(body, "class=\"rk-example\"")
        observed == expected || _record_advisory!(
            advisories; page = source, contract_kind = "executable-panel-count",
            expected, observed, artifact_ids,
        )
        observed_panels += observed

        source_examples = _occurrences(body, "class=\"rk-source-example\"")
        expected_sources = get(expected_source_examples, source, 0)
        source_examples == expected_sources || _record_advisory!(
            advisories; page = source, contract_kind = "captured-source-example-count",
            expected = expected_sources, observed = source_examples, artifact_ids,
        )

        source_interactions = _occurrences(body, "class=\"rk-source-interaction\"") +
            _occurrences(body, "class=\"rk-source-example\"")
        expected_interactions = get(expected_source_interactions, source, 0)
        source_interactions == expected_interactions || _record_advisory!(
            advisories; page = source, contract_kind = "source-interaction-count",
            expected = expected_interactions, observed = source_interactions, artifact_ids,
        )

        sortable_tables = _occurrences(intermediate, "htmxo-sortable-table")
        expected_tables = get(expected_sortable_tables, source, 0)
        sortable_tables == expected_tables || _record_advisory!(
            advisories; page = source, contract_kind = "sortable-table-count",
            expected = expected_tables, observed = sortable_tables, artifact_ids,
        )

        aov_panels = _occurrences(intermediate, "class=\"rk-aov-panel\"")
        expected_plots = get(expected_aov_panels, source, 0)
        aov_panels == expected_plots || _record_advisory!(
            advisories; page = source, contract_kind = "aov-panel-count",
            expected = expected_plots, observed = aov_panels, artifact_ids,
        )

        for marker in get(structural_markers, source, ())
            occursin(marker, intermediate) || _record_advisory!(
                advisories; page = source, contract_kind = "intermediate-marker",
                expected = marker, observed = "missing", artifact_ids,
                detail = "human-facing structural marker drift",
            )
        end

        if source == "reactivehmc-corpus.md"
            inventory_count = _occurrences(intermediate, "data-rk-corpus-id=")
            inventory_count == 17 || _record_advisory!(
                advisories; page = source, contract_kind = "corpus-inventory-count",
                expected = 17, observed = inventory_count, artifact_ids,
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

        for marker in get(body_markers, source, ())
            occursin(marker, body) || _record_advisory!(
                advisories; page = source, contract_kind = "rendered-heading-or-marker",
                expected = marker, observed = "missing", artifact_ids,
                detail = "human-facing heading or marker drift",
            )
        end
    end

    expected_total = sum(values(expected_panels))
    observed_panels == expected_total || _record_advisory!(
        advisories; page = "<site>", contract_kind = "total-executable-panel-count",
        expected = expected_total, observed = observed_panels, artifact_ids = all_artifact_ids,
    )
    _check_warning_banner_absent(build_dir, advisories)
    return (; advisories, pages = length(sources), executable_panels = observed_panels)
end

"""Enforce fatal render integrity and report presentation drift without blocking deploy."""
function check_rendered_docs(build_dir, page_tree;
                             source_dir = joinpath(@__DIR__, "src"),
                             report_path = get(
                                 ENV, "RK_RENDERED_DOCS_REPORT",
                                 joinpath(build_dir, ".reports", "rendered-docs-advisories.json"),
                             ),
                             contracts = nothing,
                             github_actions = get(ENV, "GITHUB_ACTIONS", "") == "true")
    advisories = NamedTuple[]
    result = try
        _check_rendered_docs!(advisories, build_dir, page_tree; source_dir, contracts)
    catch exception
        _write_advisory_report(
            report_path, advisories;
            status = "fatal", fatal_error = sprint(showerror, exception),
        )
        rethrow()
    end

    status = isempty(advisories) ? "clean" : "advisory"
    _write_advisory_report(report_path, advisories; status)
    _emit_advisories(advisories; github_actions)
    @info "Rendered docs integrity verified" pages=result.pages executable_panels=result.executable_panels advisories=length(advisories) report_path
    return advisories
end
