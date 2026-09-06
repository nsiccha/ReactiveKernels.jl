# Docs renderer for the Gibbs / CGGibbs investigation page (source-only, not
# executed): it reads the authored kernels from benchmark/ at build time and
# renders the marked slices, so the code shown on the page cannot drift from the
# code that actually runs.
module GibbsDocs

using Markdown

const ROOT = normpath(joinpath(@__DIR__, ".."))   # docs/ -> repo root

function _read_marked(relpath, marker)
    path = joinpath(ROOT, relpath)
    isfile(path) || error("Gibbs docs source missing: $path")
    source = replace(read(path, String), "\r\n" => "\n", "\r" => "\n")
    start, stop = "# BEGIN $marker\n", "# END $marker"
    count(start, source) == 1 && count(stop, source) == 1 ||
        error("Gibbs docs source marker drift for $marker in $relpath")
    body = split(split(source, start; limit = 2)[2], stop; limit = 2)[1]
    Markdown.MD(Markdown.Code("julia", rstrip(body)))
end

render_ssvs_graph() =
    _read_marked("benchmark/gibbs_poc/ssvs_gibbs.jl", "GIBBS_SSVS_GRAPH")
render_cggibbs_conditional() =
    _read_marked("benchmark/cggibbs/cggibbs.jl", "CGGIBBS_CONDITIONAL")

end # module GibbsDocs
