module ComparisonSourceAttestation

import SHA

export comparator_source_matches_current_delta, COMPARATOR_SOURCE_CURRENT_DELTA
export eight_schools_model_source_matches_recorded_current
export eight_schools_model_source_preserves_published_authority
export EIGHT_SCHOOLS_MODEL_SOURCE_CURRENT_DELTA
export historical_source_pin_errors, recorded_current_source_pin_errors
export mnist_model_source_preserves_published_authority
export mnist_model_source_matches_recorded_current
export MNIST_COMPARATOR_SOURCE_CURRENT_DELTA, MNIST_MODEL_SOURCE_CURRENT_DELTA
export sum_to_zero_model_source_preserves_published_authority
export SUM_TO_ZERO_MODEL_SOURCE_CURRENT_DELTA

const COMPARATOR_SOURCE_CURRENT_DELTA =
    "long-form native/bound/nonallocating matrix around byte-preserved Turing/manual baselines plus terminal definition-only guard"
const EIGHT_SCHOOLS_MODEL_SOURCE_CURRENT_DELTA =
    "published centered hierarchy and every public boundary are preserved; additive scalar-index-free packed extraction and a single-output Jacobian recipe enable Reactant and nonallocating configurations"
const MNIST_COMPARATOR_SOURCE_CURRENT_DELTA =
    "additive two-model native/bound/nonallocating matrix; published Turing/manual AD baselines unchanged; documentation markers plus terminal definition-only include guard"
const MNIST_MODEL_SOURCE_CURRENT_DELTA =
    "published idiomatic model source is byte-preserved; the additive optimized model uses the same natural each-column plate with the reference-coded categorical object"
const SUM_TO_ZERO_MODEL_SOURCE_CURRENT_DELTA =
    "published model body is preserved; scalar-indexed packed extraction and the model_only evaluator/init path are the only current-source deltas"
const EIGHT_SCHOOLS_RECORDED_TO_CURRENT_MODEL_SOURCE_DELTA =
    "recorded model body is preserved; artifact-backed real data, scalar-indexed packed extraction, and the model_only evaluator/init path are the only recorded-to-current deltas"
const _DOCS_BASELINE_MARKERS = (
    "# DOCS-BASELINE-BEGIN: turing",
    "# DOCS-BASELINE-END: turing",
    "# DOCS-BASELINE-BEGIN: manual",
    "# DOCS-BASELINE-END: manual",
)
const _MNIST_DOCS_BASELINE_MARKERS = (
    "# DOCS-BASELINE-BEGIN: turing",
    "# DOCS-BASELINE-END: turing",
    "# DOCS-BASELINE-BEGIN: turing-optimized",
    "# DOCS-BASELINE-END: turing-optimized",
    "# DOCS-BASELINE-BEGIN: manual",
    "# DOCS-BASELINE-END: manual",
)

_normalized_text(text) =
    replace(String(text), "\r\n" => "\n", "\r" => "\n")

_normalized_sha256(text) = bytes2hex(SHA.sha256(_normalized_text(text)))

function _git_revision_blob(root, commit, path)
    object = string(commit, ":", path)
    try
        readchomp(`git -C $root rev-parse $object`)
    catch
        ""
    end
end

function _git_blob_text(root, blob)
    try
        read(`git -C $root cat-file blob $blob`, String)
    catch
        ""
    end
end

function historical_source_pin_errors(root, pin; label = "source",
        commit = get(pin, "commit", ""))
    errors = String[]
    require(condition, message) = condition || push!(errors, message)
    path = String(get(pin, "path", ""))
    commit = String(commit)
    published_blob = String(get(pin, "git_blob", ""))
    published_digest = String(get(pin, "text_sha256", ""))
    require(!isempty(path), "$label published path missing")
    require(occursin(r"^[0-9a-f]{40}$", commit),
            "$label published commit missing")
    require(occursin(r"^[0-9a-f]{40}$", published_blob),
            "$label published blob missing")
    require(occursin(r"^[0-9a-f]{64}$", published_digest),
            "$label published text digest missing")
    (isempty(path) || !occursin(r"^[0-9a-f]{40}$", commit)) && return errors

    git_blob = _git_revision_blob(root, commit, path)
    require(!isempty(git_blob), "$label published source commit is unavailable")
    isempty(git_blob) && return errors
    require(git_blob == published_blob, "$label published Git blob mismatch")
    git_blob == published_blob || return errors
    text = _git_blob_text(root, git_blob)
    require(_normalized_sha256(text) == published_digest,
            "$label published text digest mismatch")
    errors
end

function recorded_current_source_pin_errors(root, current; label = "source")
    errors = String[]
    require(condition, message) = condition || push!(errors, message)
    blob = String(get(current, "git_blob", ""))
    digest = String(get(current, "text_sha256", ""))
    require(occursin(r"^[0-9a-f]{40}$", blob),
            "$label recorded current blob missing")
    require(occursin(r"^[0-9a-f]{64}$", digest),
            "$label recorded current text digest missing")
    occursin(r"^[0-9a-f]{40}$", blob) || return errors
    text = _git_blob_text(root, blob)
    require(!isempty(text), "$label recorded current source blob is unavailable")
    isempty(text) && return errors
    require(_normalized_sha256(text) == digest,
            "$label recorded current text digest mismatch")
    errors
end

function _replace_once(text, replacement, original)
    length(findall(replacement, text)) == 1 || return nothing
    replace(text, replacement => original; count = 1)
end

function _replace_exactly(text, replacement, original, expected_count)
    length(findall(replacement, text)) == expected_count || return nothing
    replace(text, replacement => original; count = expected_count)
end

const _EIGHT_SCHOOLS_ARTIFACT_DATA =
    "using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data\n\nexport EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA\nexport build_eight_schools_graph, demo\nexport EIGHT_SCHOOLS_SOURCE, evaluate_eight_schools_source\n\n# Real data from posteriordb `eight_schools-eight_schools_centered`, loaded from the\n# bundled artifact via PosteriorDB.jl (no hand-inlined arrays).\nlet d = _posteriordb_data(\"eight_schools-eight_schools_centered\")\n    global const NSCHOOLS = Int(d[\"J\"])\n    global const EIGHT_SCHOOLS_Y = Float64.(d[\"y\"])\n    global const EIGHT_SCHOOLS_SIGMA = Float64.(d[\"sigma\"])\nend\n"
const _EIGHT_SCHOOLS_PUBLISHED_DATA =
    "using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source\n\nexport EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA\nexport build_eight_schools_graph, demo\nexport EIGHT_SCHOOLS_SOURCE, evaluate_eight_schools_source\n\nconst NSCHOOLS = 8\n\nconst EIGHT_SCHOOLS_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]\nconst EIGHT_SCHOOLS_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]\n"

function _single_marked_body(text, name)
    begin_marker = "# DOCS-BASELINE-BEGIN: $name\n"
    end_marker = "# DOCS-BASELINE-END: $name\n"
    length(findall(begin_marker, text)) == 1 || return nothing
    length(findall(end_marker, text)) == 1 || return nothing
    after_begin = split(text, begin_marker; limit = 2)[2]
    split(after_begin, end_marker; limit = 2)[1]
end

function _single_delimited_section(text, begin_marker, end_marker)
    length(findall(begin_marker, text)) == 1 || return nothing
    after_begin = split(text, begin_marker; limit = 2)[2]
    isempty(findall(end_marker, after_begin)) && return nothing
    begin_marker * split(after_begin, end_marker; limit = 2)[1] * end_marker
end

function _single_raw_body(text, begin_marker)
    length(findall(begin_marker, text)) == 1 || return nothing
    after_begin = split(text, begin_marker; limit = 2)[2]
    end_marker = "\n\"\"\""
    isempty(findall(end_marker, after_begin)) && return nothing
    split(after_begin, end_marker; limit = 2)[1]
end

function eight_schools_model_source_preserves_published_authority(current, published)
    current = _normalized_text(current)
    published = _normalized_text(published)
    replacements = (
        (
            _EIGHT_SCHOOLS_ARTIFACT_DATA,
            _EIGHT_SCHOOLS_PUBLISHED_DATA,
        ),
        (
            "    # The Jacobian is also available alone. That single-output path lets scalar\n" *
            "    # packed-density queries use the public nonallocating preparation pass,\n" *
            "    # whose recipes are deliberately single-output. Asking for BOTH parameters\n" *
            "    # and the Jacobian can still select the joint producer below.\n" *
            "    log_jacobian::Float64 = log_τ\n\n",
            "",
        ),
        (
            "    # first for a constrain-only query and the joint producer when both outputs\n" *
            "    # are requested together.\n",
            "    # first for a constrain-only query and the second whenever the Jacobian —\n" *
            "    # hence the unconstrained posterior — is wanted.\n",
        ),
        (
            "function evaluate_eight_schools_source(; model_only::Bool = false)",
            "function evaluate_eight_schools_source()",
        ),
        (
            "    ), model_only)\n",
            "    ))\n",
        ),
        (
            "    _EIGHT_SCHOOLS_GRAPH_TEMPLATE[] = evaluate_eight_schools_source(; model_only = true).model\n",
            "    _EIGHT_SCHOOLS_GRAPH_TEMPLATE[] = evaluate_eight_schools_source().model\n",
        ),
    )
    transformed = current
    for (replacement, original) in replacements
        transformed = _replace_once(transformed, replacement, original)
        transformed === nothing && return false
    end
    transformed == published
end

function eight_schools_model_source_matches_recorded_current(current, recorded)
    current = _normalized_text(current)
    recorded = _normalized_text(recorded)
    replacements = (
        (
            _EIGHT_SCHOOLS_ARTIFACT_DATA,
            _EIGHT_SCHOOLS_PUBLISHED_DATA,
        ),
        (
            "    μ::Float64 = unconstrained[1]\n" *
            "    log_τ::Float64 = unconstrained[2]\n",
            "    # One-element reductions retain the ordinary packed-vector boundary while\n" *
            "    # avoiding scalar indexing when the same prepared kernel is traced as a\n" *
            "    # Reactant tensor program. Native Julia specializes these constant slices.\n" *
            "    μ::Float64 = sum(view(unconstrained, 1:1))\n" *
            "    log_τ::Float64 = sum(view(unconstrained, 2:2))\n",
        ),
        (
            "function evaluate_eight_schools_source(; model_only::Bool = false)",
            "function evaluate_eight_schools_source()",
        ),
        (
            "    ), model_only)\n",
            "    ))\n",
        ),
        (
            "    _EIGHT_SCHOOLS_GRAPH_TEMPLATE[] = evaluate_eight_schools_source(; model_only = true).model\n",
            "    _EIGHT_SCHOOLS_GRAPH_TEMPLATE[] = evaluate_eight_schools_source().model\n",
        ),
    )
    transformed = current
    for (replacement, original) in replacements
        transformed = _replace_once(transformed, replacement, original)
        transformed === nothing && return false
    end
    transformed == recorded
end

function mnist_model_source_preserves_published_authority(current, published)
    current = _normalized_text(current)
    published = _normalized_text(published)
    current_source = _single_raw_body(
        current, "const MNIST_LOGISTIC_SOURCE = raw\"\"\"\n")
    published_source = _single_raw_body(
        published, "const MNIST_LOGISTIC_SOURCE = raw\"\"\"\n")
    current_source !== nothing && published_source !== nothing || return false
    current_source == published_source || return false
    current_optimized = _single_raw_body(
        current, "const MNIST_LOGISTIC_OPTIMIZED_SOURCE = raw\"\"\"\n")
    published_optimized = _single_raw_body(
        published, "const MNIST_LOGISTIC_OPTIMIZED_SOURCE = raw\"\"\"\n")
    if published_optimized !== nothing
        current_optimized !== nothing || return false
        current_optimized == published_optimized || return false
    end

    for optimized_anchor in (
            "const MNIST_LOGISTIC_OPTIMIZED_SOURCE = raw\"\"\"",
            "plate(eachcol(nonreference_logits), y)",
            "categorical_logit_ref(observation_logits).logpdf(observed_class)",
            "build_mnist_logistic_optimized_graph() =",
        )
        length(findall(optimized_anchor, current)) == 1 || return false
    end
    for removed_anchor in (
            "_kernel_tensorized_pair",
            "_categorical_logit_columns_kernel",
            "_categorical_logit_ref_columns_kernel",
        )
        isempty(findall(removed_anchor, current)) || return false
    end

    for authority in (
            "_evaluate_ppl_source(MNIST_LOGISTIC_SOURCE, @__MODULE__; bindings = (\n        :MNIST_LOGISTIC_X, :MNIST_LOGISTIC_Y, :NUM_CLASSES,",
            "build_mnist_logistic_graph() = compose(_MNIST_LOGISTIC_GRAPH_TEMPLATE[])",
        )
        length(findall(authority, current)) == 1 || return false
        length(findall(authority, published)) == 1 || return false
    end
    for (model_only_authority, expected_count) in (
            ("function evaluate_mnist_logistic_source(; model_only::Bool = false)", 1),
            ("    ), model_only)\n", 2),
            ("    _MNIST_LOGISTIC_GRAPH_TEMPLATE[] = evaluate_mnist_logistic_source(; model_only = true).model\n", 1),
            ("function evaluate_mnist_logistic_optimized_source(; model_only::Bool = false)", 1),
            ("        evaluate_mnist_logistic_optimized_source(; model_only = true).model\n", 1),
        )
        length(findall(model_only_authority, current)) == expected_count ||
            return false
    end
    true
end

function sum_to_zero_model_source_preserves_published_authority(current, published)
    current = _normalized_text(current)
    published = _normalized_text(published)
    replacements = (
        (
            "    α_s2z::Float64 = unconstrained[1]\n" *
            "    log_τ::Float64 = unconstrained[2]\n",
            "    α_s2z::Float64 = sum(view(unconstrained, 1:1))\n" *
            "    log_τ::Float64 = sum(view(unconstrained, 2:2))\n",
        ),
        (
            "function evaluate_sum_to_zero_source(; model_only::Bool = false)",
            "function evaluate_sum_to_zero_source()",
        ),
        (
            "    ), model_only)\n",
            "    ))\n",
        ),
        (
            "    _SUM_TO_ZERO_GRAPH_TEMPLATE[] = evaluate_sum_to_zero_source(; model_only = true).model\n",
            "    _SUM_TO_ZERO_GRAPH_TEMPLATE[] = evaluate_sum_to_zero_source().model\n",
        ),
    )
    transformed = current
    for (replacement, original) in replacements
        transformed = _replace_once(transformed, replacement, original)
        transformed === nothing && return false
    end
    transformed == published
end

function mnist_model_source_matches_recorded_current(current, recorded)
    current = _normalized_text(current)
    recorded = _normalized_text(recorded)
    transformations = (
        ("function evaluate_mnist_logistic_source(; model_only::Bool = false)",
         "function evaluate_mnist_logistic_source()", 1),
        ("    ), model_only)\n", "    ))\n", 2),
        ("    _MNIST_LOGISTIC_GRAPH_TEMPLATE[] = evaluate_mnist_logistic_source(; model_only = true).model\n",
         "    _MNIST_LOGISTIC_GRAPH_TEMPLATE[] = evaluate_mnist_logistic_source().model\n", 1),
        ("function evaluate_mnist_logistic_optimized_source(; model_only::Bool = false)",
         "function evaluate_mnist_logistic_optimized_source()", 1),
        ("        evaluate_mnist_logistic_optimized_source(; model_only = true).model\n",
         "        evaluate_mnist_logistic_optimized_source().model\n", 1),
    )
    transformed = current
    for (replacement, original, expected_count) in transformations
        transformed = _replace_exactly(
            transformed, replacement, original, expected_count)
        transformed === nothing && return false
    end
    transformed == recorded
end

function comparator_source_matches_current_delta(
        current, published, definition_only_guard)
    current = _normalized_text(current)
    marker_lines = filter(
        line -> startswith(line, "# DOCS-BASELINE-"), split(current, '\n'))
    marker_lines in (collect(_DOCS_BASELINE_MARKERS),
                     collect(_MNIST_DOCS_BASELINE_MARKERS)) || return false
    endswith(current, definition_only_guard) || return false
    length(findall(definition_only_guard, current)) == 1 || return false
    published = _normalized_text(published)
    for name in ("turing", "manual")
        body = _single_marked_body(current, name)
        body !== nothing && !isempty(body) || return false
        length(findall(body, published)) == 1 || return false
    end
    if marker_lines == collect(_MNIST_DOCS_BASELINE_MARKERS)
        optimized = _single_marked_body(current, "turing-optimized")
        optimized !== nothing && !isempty(optimized) || return false
    end
    true
end

end # module ComparisonSourceAttestation
