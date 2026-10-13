# Model display for an existing build. The code pane lowers the stored spec to
# obtain its actual planned expression; bound and prepared panes read the
# expressions the caller supplies. Display does not bind, build, prepare or
# evaluate the model again.

"""
    RKPPLModelView

Separate views of one built `@rkppl` program, returned by [`model_view`](@ref):

- `inputs` / `outputs`: the kernel's HAVE inputs and WANT outputs;
- `coordinates`: the packed sampler coordinates, `coordinate_names(layout)`;
- `code`: the planned ReactiveKernels expression, `code_expr(spec)`;
- `graph`: the structural graph, `kernel_graph(spec)`; its text display lists
  values and recipes with plate and scan bodies nested, and its HTML display is
  the interactive DAG;
- `source`: the generated pre-build kernel definition
  `kernel_expr(bound, layout)`, or `nothing` when no bound program was given;
- `prepared`: the exact expression of a supplied prepared query, or `nothing`.

Code expressions are printed without display rewrites. Only line-number nodes
are removed from a copy for presentation.

`show` renders the views as labeled sections in plain text or HTML.
"""
struct RKPPLModelView
    spec::KernelSpec
    layout::LayoutTable
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    coordinates::Vector{Symbol}
    code::Expr
    graph::ReactiveKernels.Graph
    source::Union{Nothing,Expr}
    prepared::Union{Nothing,Expr}
end

"""
    model_view(built; bound = nothing, query = nothing) -> RKPPLModelView
    model_view(spec, layout; bound = nothing, query = nothing) -> RKPPLModelView

Display an existing build. `built` is the `(; spec, layout)` returned by
[`build_kernel`](@ref), including the built model a BRM `RKBRMI` backend
retains as `backend.model`. The built program, structural graph and coordinates
need nothing else.

Pass `bound`, the bound `StructuralPlan` that was built, to add the generated
pre-build kernel definition, and `query`, a kernel returned by
[`prepare_query`](@ref) or a [`SamplerQuery`](@ref), to add its prepared
program (with its bound data folded in as literals). These values are only
read; the model is not bound, built, prepared or evaluated again. The code
pane lowers the stored spec to obtain its planned expression.
"""
model_view(built::NamedTuple; kwargs...) = model_view(built.spec, built.layout; kwargs...)

function model_view(spec::KernelSpec, layout::LayoutTable;
                    bound = nothing, query = nothing)
    RKPPLModelView(spec, layout, copy(spec.have_names), copy(spec.want_names),
        coordinate_names(layout), _model_visible_expr(code_expr(spec)), kernel_graph(spec),
        _model_source(bound, layout), _model_prepared(query))
end

_model_visible_expr(expr::Expr) = Base.remove_linenums!(deepcopy(expr))

_model_source(::Nothing, layout) = nothing
function _model_source(bound::StructuralPlan, layout::LayoutTable)
    _model_visible_expr(kernel_expr(bound, layout))
end
_model_source(bound, layout) = throw(ArgumentError(
    "model_view: `bound` must be the bound StructuralPlan that was built; got $(typeof(bound))"))

_model_prepared(::Nothing) = nothing
_model_prepared(query::SamplerQuery) = _model_prepared(query.kernel)
_model_prepared(query) = _model_visible_expr(code_expr(query))

const _MODEL_VIEW_SECTIONS = (
    (:code, "Planned ReactiveKernels program", "code_expr(spec)"),
    (:graph, "Structural graph", "kernel_graph(spec)"),
    (:source, "Generated pre-build kernel definition", "kernel_expr(bound, layout)"),
    (:prepared, "Prepared query program", "code_expr(query)"),
)

_joined(names) = isempty(names) ? "(none)" : join(string.(names), ", ")

function _model_view_header(io::IO, view::RKPPLModelView)
    println(io, "Inputs (HAVE): ", _joined(view.inputs))
    println(io, "Outputs (WANT): ", _joined(view.outputs))
    print(io, "Coordinates (", length(view.coordinates), "): ", _joined(view.coordinates))
end

Base.show(io::IO, view::RKPPLModelView) =
    print(io, "RKPPLModelView(", length(view.coordinates), " coordinates, ",
          _joined(view.inputs), " -> ", _joined(view.outputs), ")")

function Base.show(io::IO, mime::MIME"text/plain", view::RKPPLModelView)
    println(io, "RKPPL model view")
    _model_view_header(io, view)
    for (field, title, call) in _MODEL_VIEW_SECTIONS
        print(io, "\n\n── ", title, ": ", call, " ──\n")
        content = getfield(view, field)
        if content === nothing
            print(io, field === :source ?
                "(not shown: pass `bound`, the bound program that was built)" :
                "(not shown: pass `query`, a prepared query)")
        else
            field === :graph ? show(io, mime, content) :
                print(IOContext(io, :limit => false), content)
        end
    end
end

function Base.show(io::IO, mime::MIME"text/html", view::RKPPLModelView)
    esc = ReactiveKernels._html_escape
    print(io, "<div class=\"rkppl-model-view\"><pre class=\"rkppl-model-view-header\">",
          esc(sprint(_model_view_header, view)), "</pre>")
    for (field, title, call) in _MODEL_VIEW_SECTIONS
        content = getfield(view, field)
        content === nothing && continue
        print(io, "<details", field === :code ? " open" : "", "><summary>", esc(title),
              " <code>", esc(call), "</code></summary>")
        if field === :graph
            show(io, mime, visualize(content))
            print(io, "<pre class=\"rk-graph-listing\">",
                  esc(sprint(show, MIME"text/plain"(), content)), "</pre>")
        else
            print(io, "<pre><code class=\"language-julia\">",
                  esc(sprint(print, content; context = :limit => false)),
                  "</code></pre>")
        end
        print(io, "</details>")
    end
    print(io, "</div>")
end
