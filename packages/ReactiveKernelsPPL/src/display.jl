# Model display for an existing build. Every view reads values the caller
# already holds — the `build_kernel` result and, optionally, the bound program
# and a prepared query — so displaying a model never lowers, binds, builds,
# prepares or evaluates it again.

"""
    RKPPLModelView

Separate views of one built `@rkppl` program, returned by [`model_view`](@ref):

- `inputs` / `outputs`: the kernel's HAVE inputs and WANT outputs;
- `coordinates`: the packed sampler coordinates, `coordinate_names(layout)`;
- `code`: the built ReactiveKernels program, `readable_code(spec)` — every
  operation, including composed child kernels, shown by its authored source;
- `graph`: the structural graph, `kernel_graph(spec)`; its text display lists
  values and recipes with plate and scan bodies nested, and its HTML display is
  the interactive DAG;
- `source`: the generated pre-build `@kernel` program
  `kernel_expr(bound, layout)`, or `nothing` when no bound program was given;
- `prepared`: the readable prepared program of a supplied query, or `nothing`.

`show` renders the views as labeled sections in plain text or HTML.
"""
struct RKPPLModelView
    spec::KernelSpec
    layout::LayoutTable
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    coordinates::Vector{Symbol}
    code::ReadableCode
    graph::ReactiveKernels.Graph
    source::Union{Nothing,ReadableCode}
    prepared::Union{Nothing,ReadableCode}
end

"""
    model_view(built; bound = nothing, query = nothing) -> RKPPLModelView
    model_view(spec, layout; bound = nothing, query = nothing) -> RKPPLModelView

Display an existing build. `built` is the `(; spec, layout)` returned by
[`build_kernel`](@ref), including the built model a BRM `RKBRMI` backend
retains as `backend.model`. The built program, structural graph and coordinates
need nothing else.

Pass `bound`, the bound `StructuralPlan` that was built, to add the generated
pre-build `@kernel` program, and `query`, a kernel returned by
[`prepare_query`](@ref) or a [`SamplerQuery`](@ref), to add its prepared
program (with its bound data folded in as literals). These values are only
read; the model is not lowered, bound, built, prepared or evaluated again.
"""
model_view(built::NamedTuple; kwargs...) = model_view(built.spec, built.layout; kwargs...)

function model_view(spec::KernelSpec, layout::LayoutTable;
                    bound = nothing, query = nothing)
    RKPPLModelView(spec, layout, copy(spec.have_names), copy(spec.want_names),
        coordinate_names(layout), readable_code(spec), kernel_graph(spec),
        _model_source(bound, layout), _model_prepared(query))
end

_model_source(::Nothing, layout) = nothing
function _model_source(bound::StructuralPlan, layout::LayoutTable)
    def = kernel_expr(bound, layout)
    readable_code(Expr(:macrocall, Symbol("@kernel"), nothing, def);
                  modules = (PPLGeneratedModels,))
end
_model_source(bound, layout) = throw(ArgumentError(
    "model_view: `bound` must be the bound StructuralPlan that was built; got $(typeof(bound))"))

_model_prepared(::Nothing) = nothing
_model_prepared(query::SamplerQuery) = _model_prepared(query.kernel)
_model_prepared(query) = readable_code(query)

const _MODEL_VIEW_SECTIONS = (
    (:code, "Built ReactiveKernels program", "readable_code(spec)"),
    (:graph, "Structural graph", "kernel_graph(spec)"),
    (:source, "Generated pre-build @kernel program", "kernel_expr(bound, layout)"),
    (:prepared, "Prepared query program", "readable_code(query)"),
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
            show(io, mime, content)
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
            show(io, mime, content)
        end
        print(io, "</details>")
    end
    print(io, "</div>")
end
