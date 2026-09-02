module ComparisonSourceAttestation

export comparator_source_matches_current_delta, COMPARATOR_SOURCE_CURRENT_DELTA
export eight_schools_model_source_preserves_published_authority
export EIGHT_SCHOOLS_MODEL_SOURCE_CURRENT_DELTA
export mnist_model_source_preserves_published_authority
export MNIST_COMPARATOR_SOURCE_CURRENT_DELTA, MNIST_MODEL_SOURCE_CURRENT_DELTA

const COMPARATOR_SOURCE_CURRENT_DELTA =
    "long-form native/bound/nonallocating matrix around byte-preserved Turing/manual baselines plus terminal definition-only guard"
const EIGHT_SCHOOLS_MODEL_SOURCE_CURRENT_DELTA =
    "published centered hierarchy and every public boundary are preserved; additive scalar-index-free packed extraction and a single-output Jacobian recipe enable Reactant and nonallocating configurations"
const MNIST_COMPARATOR_SOURCE_CURRENT_DELTA =
    "additive two-model native/bound/nonallocating matrix; published Turing/manual AD baselines unchanged; documentation markers plus terminal definition-only include guard"
const MNIST_MODEL_SOURCE_CURRENT_DELTA =
    "published idiomatic model source is byte-preserved; the additive optimized model uses the same natural each-column plate with the reference-coded categorical object"
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

function eight_schools_model_source_preserves_published_authority(
        current, published)
    current = _normalized_text(current)
    published = _normalized_text(published)
    replacements = (
        (
            "    # One-element reductions retain the ordinary packed-vector boundary while\n" *
            "    # avoiding scalar indexing when the same prepared kernel is traced as a\n" *
            "    # Reactant tensor program. Native Julia specializes these constant slices.\n" *
            "    μ::Float64 = sum(view(unconstrained, 1:1))\n" *
            "    log_τ::Float64 = sum(view(unconstrained, 2:2))\n",
            "    μ::Float64 = unconstrained[1]\n" *
            "    log_τ::Float64 = unconstrained[2]\n",
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
    )
    transformed = current
    for (new, old) in replacements
        length(findall(new, transformed)) == 1 || return false
        transformed = replace(transformed, new => old; count = 1)
    end
    transformed == published
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
            "_evaluate_ppl_source(MNIST_LOGISTIC_SOURCE, @__MODULE__; bindings = (\n        :MNIST_LOGISTIC_X, :MNIST_LOGISTIC_Y, :NUM_CLASSES,\n    ))",
            "build_mnist_logistic_graph() = compose(_MNIST_LOGISTIC_GRAPH_TEMPLATE[])",
        )
        length(findall(authority, current)) == 1 || return false
        length(findall(authority, published)) == 1 || return false
    end
    true
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
