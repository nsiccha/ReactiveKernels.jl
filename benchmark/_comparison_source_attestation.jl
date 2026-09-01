module ComparisonSourceAttestation

export comparator_source_matches_current_delta, COMPARATOR_SOURCE_CURRENT_DELTA

const COMPARATOR_SOURCE_CURRENT_DELTA =
    "documentation-only baseline markers plus terminal definition-only include guard"
const _DOCS_BASELINE_MARKERS = (
    "# DOCS-BASELINE-BEGIN: turing",
    "# DOCS-BASELINE-END: turing",
    "# DOCS-BASELINE-BEGIN: manual",
    "# DOCS-BASELINE-END: manual",
)

_normalized_text(text) =
    replace(String(text), "\r\n" => "\n", "\r" => "\n")

function comparator_source_matches_current_delta(
        current, published, definition_only_guard)
    current = _normalized_text(current)
    marker_lines = filter(
        line -> startswith(line, "# DOCS-BASELINE-"), split(current, '\n'))
    marker_lines == collect(_DOCS_BASELINE_MARKERS) || return false

    without_markers = current
    for marker in _DOCS_BASELINE_MARKERS
        without_markers = replace(
            without_markers, marker * "\n" => ""; count = 1)
    end
    expected = replace(
        _normalized_text(published),
        r"run_comparison\(\)\n$" => definition_only_guard)
    without_markers == expected
end

end # module ComparisonSourceAttestation
