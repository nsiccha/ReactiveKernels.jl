# DAG visualization with a library-backed interactive surface.
#
# DOT is the portable interchange format, and a compact SVG renderer remains a
# zero-setup static fallback. Interactive HTML is rendered by Cytoscape.js with
# the ELK layered layout: established libraries own graph geometry, routing,
# zooming, and panning while ReactiveKernels supplies only its graph model,
# semantic styling, and inspector. Recipes remain explicit nodes so multi-input
# and multi-output computations are represented without ambiguous hyper-edges.

"""
    DAGVisualization

An inspectable visualization of a [`Graph`](@ref) or [`Plan`](@ref). Construct
one with [`visualize`](@ref), display it in a rich Julia frontend for SVG, pass
it to [`dot_source`](@ref), or save it with [`save_visualization`](@ref).
"""
struct DAGVisualization{T<:Union{Graph,Plan}}
    subject::T
    alternatives::Bool
    orientation::Symbol

    function DAGVisualization(subject::T, alternatives::Bool,
                              orientation::Symbol) where {T<:Union{Graph,Plan}}
        _check_orientation(orientation)
        subject isa Graph && alternatives &&
            throw(ArgumentError("alternatives applies only to Plan visualizations"))
        new{T}(subject, alternatives, orientation)
    end
end

function _check_orientation(orientation::Symbol)
    orientation in (:horizontal, :vertical) ||
        throw(ArgumentError("orientation must be :horizontal or :vertical"))
    orientation
end

"""
    visualize(graph; orientation=:horizontal) -> DAGVisualization
    visualize(plan; alternatives=false, orientation=:horizontal) -> DAGVisualization

Create a DAG visualization. Values and recipes are separate
nodes, with edges `value → recipe → value`, so multi-input and multi-output
recipes remain unambiguous.

For a `Plan`, the default shows only selected recipes. Set `alternatives=true`
to add backward-reachable but unselected candidates as muted dashed nodes.
`orientation` may be `:horizontal` (left to right) or `:vertical` (top to
bottom).

Rich Julia displays render the result as Cytoscape.js + ELK interactive HTML
when supported, with self-contained SVG as the static and offline fallback.
Documentation bundles the libraries locally; standalone rich displays load
the same pinned libraries on demand. The underlying `Graph` and `Plan` types
expose both representations directly, so a notebook can display them without
calling `visualize` explicitly.
"""
function visualize(x::Graph; alternatives::Bool = false,
                   orientation::Symbol = :horizontal)
    DAGVisualization(x, alternatives, orientation)
end

function visualize(x::Plan; alternatives::Bool = false,
                   orientation::Symbol = :horizontal)
    DAGVisualization(x, alternatives, orientation)
end

visualize(spec::KernelSpec; kwargs...) = visualize(spec.graph; kwargs...)

struct _VizNode
    id::String
    label::String
    detail::String
    kind::Symbol
    state::Symbol
end

struct _VizEdge
    src::String
    dst::String
    state::Symbol
end

struct _VizModel
    title::String
    nodes::Vector{_VizNode}
    edges::Vector{_VizEdge}
end

_value_node_id(id::Int) = "v_$id"
_recipe_node_id(id::Int) = "r_$id"

function _value_groups(g::Graph)
    groups = Dict{Int,Vector{Value}}()
    for value in values(g.values)
        push!(get!(groups, canon_id(g, value.id), Value[]), value)
    end
    for aliases in values(groups)
        sort!(aliases; by = v -> v.id)
    end
    groups
end

function _value_text(aliases::Vector{Value}, id::Int)
    labels = unique(string(v.name) for v in aliases)
    types = unique(string(valtype(v)) for v in aliases)
    join(labels, " ≡ "), "::" * join(types, " ≡ ") * " · value $id"
end

function _value_state(id::Int, have::Set{Int}, want::Set{Int})
    id in have && id in want && return :havewant
    id in have && return :have
    id in want && return :want
    :value
end

function _viz_model(v::DAGVisualization)
    subject = v.subject
    g = subject isa Plan ? subject.graph : subject
    recipes = subject isa Plan ?
        (v.alternatives ? subject.candidates : subject.recipes) : g.recipes
    selected = subject isa Plan ? Set(r.id for r in subject.recipes) : Set{Int}()
    have = subject isa Plan ? Set(canon_id(g, x.id) for x in subject.have) : Set{Int}()
    want = subject isa Plan ? Set(canon_id(g, x.id) for x in subject.want) : Set{Int}()

    value_ids = Set{Int}()
    if subject isa Graph
        for x in values(g.values)
            push!(value_ids, canon_id(g, x.id))
        end
    else
        union!(value_ids, have, want)
    end
    for r in recipes
        for x in r.inputs
            push!(value_ids, canon_id(g, x.id))
        end
        for x in r.outputs
            push!(value_ids, canon_id(g, x.id))
        end
    end

    nodes = _VizNode[]
    value_groups = _value_groups(g)
    for id in sort!(collect(value_ids))
        label, detail = _value_text(value_groups[id], id)
        state = _value_state(id, have, want)
        boundary = state === :have ? "HAVE · " : state === :want ? "WANT · " :
                   state === :havewant ? "HAVE + WANT · " : ""
        push!(nodes, _VizNode(_value_node_id(id), label, boundary * detail,
                             :value, state))
    end
    for r in sort!(collect(recipes); by = x -> x.id)
        state = r.effectful ? :effectful :
                subject isa Graph ? :recipe :
                !(r.id in selected) ? :alternative : :selected
        status = state === :alternative ? "not selected · " :
                 state === :effectful ? "effectful · " :
                 subject isa Plan ? "selected · " : ""
        detail = status * "cost $(r.cost) · recipe $(r.id)"
        push!(nodes, _VizNode(_recipe_node_id(r.id), _opname(r.op), detail,
                             :recipe, state))
    end

    edges = _VizEdge[]
    seen_edges = Set{Tuple{String,String,Symbol}}()
    for r in sort!(collect(recipes); by = x -> x.id)
        state = subject isa Plan && !(r.id in selected) ? :alternative : :selected
        rid = _recipe_node_id(r.id)
        for x in r.inputs
            edge = (_value_node_id(canon_id(g, x.id)), rid, state)
            edge in seen_edges || (push!(seen_edges, edge); push!(edges, _VizEdge(edge...)))
        end
        for x in r.outputs
            edge = (rid, _value_node_id(canon_id(g, x.id)), state)
            edge in seen_edges || (push!(seen_edges, edge); push!(edges, _VizEdge(edge...)))
        end
    end

    title = subject isa Plan ? "ReactiveKernels plan · total cost $(subject.cost)" :
                               "ReactiveKernels graph"
    _VizModel(title, nodes, edges)
end

_dot_escape(s) = replace(string(s), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n")

function _dot_attrs(node::_VizNode)
    shape = node.kind === :value ? "ellipse" : "box"
    fill, stroke, style = if node.state === :have
        ("#dcfce7", "#15803d", "filled,bold")
    elseif node.state === :want
        ("#ffedd5", "#c2410c", "filled,bold")
    elseif node.state === :havewant
        ("#fef3c7", "#a16207", "filled,bold")
    elseif node.state === :alternative
        ("#f8fafc", "#94a3b8", "filled,dashed")
    elseif node.state === :effectful
        ("#fee2e2", "#b91c1c", "filled,dashed")
    elseif node.state === :selected
        ("#dbeafe", "#2563eb", "filled,rounded")
    elseif node.kind === :recipe
        ("#f1f5f9", "#475569", "filled,rounded")
    else
        ("#ffffff", "#475569", "filled")
    end
    label = _dot_escape(node.label * "\n" * node.detail)
    "label=\"$label\", shape=$shape, style=\"$style\", fillcolor=\"$fill\", color=\"$stroke\""
end

function _write_dot(io::IO, v::DAGVisualization)
    model = _viz_model(v)
    rankdir = v.orientation === :horizontal ? "LR" : "TB"
    println(io, "digraph ReactiveKernels {")
    println(io, "  graph [rankdir=$rankdir, bgcolor=\"transparent\", label=\"$(_dot_escape(model.title))\", labelloc=t, fontname=\"sans-serif\"];")
    println(io, "  node [fontname=\"sans-serif\", fontsize=10, margin=\"0.12,0.08\"];")
    println(io, "  edge [color=\"#64748b\", arrowsize=0.7];")
    for node in model.nodes
        println(io, "  $(node.id) [$(_dot_attrs(node))];")
    end
    for edge in model.edges
        attrs = edge.state === :alternative ? " [color=\"#94a3b8\", style=dashed]" : ""
        println(io, "  $(edge.src) -> $(edge.dst)$attrs;")
    end
    print(io, "}")
end

"""
    dot_source(graph_or_plan; alternatives=false, orientation=:horizontal) -> String
    dot_source(visualization) -> String

Return a Graphviz DOT description. DOT is generated without invoking Graphviz
and can be stored, diffed, or rendered by any compatible external tool.
"""
dot_source(v::DAGVisualization) = sprint(_write_dot, v)
dot_source(x::Union{Graph,Plan}; kwargs...) = dot_source(visualize(x; kwargs...))
dot_source(spec::KernelSpec; kwargs...) = dot_source(visualize(spec; kwargs...))

Base.show(io::IO, ::MIME"text/vnd.graphviz", v::DAGVisualization) = _write_dot(io, v)
Base.show(io::IO, mime::MIME"text/vnd.graphviz", x::Union{Graph,Plan}) =
    show(io, mime, visualize(x))

function Base.show(io::IO, v::DAGVisualization)
    model = _viz_model(v)
    print(io, "DAGVisualization(", typeof(v.subject).name.name, ", ",
          count(n -> n.kind === :value, model.nodes), " values, ",
          count(n -> n.kind === :recipe, model.nodes), " recipes)")
end

function Base.show(io::IO, ::MIME"text/plain", v::DAGVisualization)
    show(io, v)
    print(io, "\n  rich display: Cytoscape.js + ELK interactive HTML",
          "\n  fallback: self-contained SVG",
          "\n  interchange: dot_source(view)",
          "\n  files: save_visualization(\"plan.html\", view), \"plan.svg\", or \"plan.dot\"")
end

_xml_escape(s) = replace(string(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
                         '"' => "&quot;", '\'' => "&apos;")

function _write_json_string(io::IO, value)
    print(io, '"')
    for char in string(value)
        if char == '"'
            print(io, "\\\"")
        elseif char == '\\'
            print(io, "\\\\")
        elseif char == '\b'
            print(io, "\\b")
        elseif char == '\f'
            print(io, "\\f")
        elseif char == '\n'
            print(io, "\\n")
        elseif char == '\r'
            print(io, "\\r")
        elseif char == '\t'
            print(io, "\\t")
        elseif char == '<'
            # An application/json script must never contain a literal
            # `</script`, even when a user-controlled node name does.
            print(io, "\\u003c")
        elseif Int(char) < 0x20
            print(io, "\\u", lpad(string(Int(char); base = 16), 4, '0'))
        else
            print(io, char)
        end
    end
    print(io, '"')
end

function _write_html_model(io::IO, model::_VizModel, orientation::Symbol)
    print(io, "{\"title\":")
    _write_json_string(io, model.title)
    print(io, ",\"orientation\":")
    _write_json_string(io, orientation)
    print(io, ",\"nodes\":[")
    for (index, node) in enumerate(model.nodes)
        index == 1 || print(io, ',')
        print(io, "{\"data\":{\"id\":")
        _write_json_string(io, node.id)
        print(io, ",\"label\":")
        _write_json_string(io, node.label)
        print(io, ",\"detail\":")
        _write_json_string(io, node.detail)
        print(io, ",\"kind\":")
        _write_json_string(io, node.kind)
        print(io, ",\"state\":")
        _write_json_string(io, node.state)
        print(io, "}}")
    end
    print(io, "],\"edges\":[")
    for (index, edge) in enumerate(model.edges)
        index == 1 || print(io, ',')
        print(io, "{\"data\":{\"id\":")
        _write_json_string(io, string(edge.src, "__", edge.dst))
        print(io, ",\"source\":")
        _write_json_string(io, edge.src)
        print(io, ",\"target\":")
        _write_json_string(io, edge.dst)
        print(io, ",\"state\":")
        _write_json_string(io, edge.state)
        print(io, "}}")
    end
    print(io, "]}")
end

# Losslessly wrap long labels into SVG tspans. This deliberately never clips or
# ellipsizes user-readable names or types.
function _svg_lines(s::AbstractString; width::Int = 34)
    chars = collect(s)
    isempty(chars) && return [""]
    [String(chars[i:min(i + width - 1, length(chars))]) for i in 1:width:length(chars)]
end

function _node_lines(node::_VizNode)
    vcat(_svg_lines(node.label), _svg_lines(node.detail))
end

function _node_size(node::_VizNode)
    lines = _node_lines(node)
    width = max(96.0, 7.2 * maximum(length, lines) + 28.0)
    height = max(52.0, 18.0 * length(lines) + 18.0)
    (width, height)
end

function _ranked_nodes(model::_VizModel)
    ids = [n.id for n in model.nodes]
    indegree = Dict(id => 0 for id in ids)
    successors = Dict(id => String[] for id in ids)
    predecessors = Dict(id => String[] for id in ids)
    for edge in model.edges
        push!(successors[edge.src], edge.dst)
        push!(predecessors[edge.dst], edge.src)
        indegree[edge.dst] += 1
    end

    rank = Dict(id => 0 for id in ids)
    ready = sort!([id for id in ids if indegree[id] == 0])
    visited = Set{String}()
    while !isempty(ready)
        id = popfirst!(ready)
        push!(visited, id)
        for dst in successors[id]
            rank[dst] = max(rank[dst], rank[id] + 1)
            indegree[dst] -= 1
            if indegree[dst] == 0
                insert!(ready, searchsortedfirst(ready, dst), dst)
            end
        end
    end

    # A raw Graph may contain cycles even though every successful Plan is a DAG.
    # Keep such nodes visible at a stable final rank; backward edges make the
    # cycle apparent without pretending the full graph was plannable.
    if length(visited) != length(ids)
        cycle_rank = isempty(visited) ? 0 : maximum(rank[id] for id in visited) + 1
        for id in sort!([id for id in ids if !(id in visited)])
            rank[id] = cycle_rank
        end
    end

    raw_ranks = sort!(unique(collect(values(rank))))
    compact = Dict(r => i - 1 for (i, r) in enumerate(raw_ranks))
    groups = [String[] for _ in raw_ranks]
    for id in sort(ids)
        push!(groups[compact[rank[id]] + 1], id)
    end

    # A pair of barycentric sweeps removes most crossings in ordinary small
    # DAGs while keeping the renderer compact and dependency-free.
    for _ in 1:2
        position = Dict{String,Float64}()
        for i in eachindex(groups)
            for (j, id) in enumerate(groups[i])
                position[id] = j
            end
            sort!(groups[i]; by = id -> begin
                ps = get(predecessors, id, String[])
                known = [position[p] for p in ps if haskey(position, p)]
                (isempty(known) ? position[id] : sum(known) / length(known), id)
            end)
            for (j, id) in enumerate(groups[i])
                position[id] = j
            end
        end
        empty!(position)
        for i in reverse(eachindex(groups))
            for (j, id) in enumerate(groups[i])
                position[id] = j
            end
            sort!(groups[i]; by = id -> begin
                ss = get(successors, id, String[])
                known = [position[s] for s in ss if haskey(position, s)]
                (isempty(known) ? position[id] : sum(known) / length(known), id)
            end)
            for (j, id) in enumerate(groups[i])
                position[id] = j
            end
        end
    end
    groups
end

function _layout(model::_VizModel, orientation::Symbol)
    isempty(model.nodes) &&
        return (Dict{String,Tuple{Float64,Float64}}(),
                Dict{String,Tuple{Float64,Float64}}(), 360.0, 100.0)
    groups = _ranked_nodes(model)
    by_id = Dict(n.id => n for n in model.nodes)
    sizes = Dict(id => _node_size(by_id[id]) for id in keys(by_id))
    margin = 32.0
    title_height = 38.0
    rank_gap = 92.0
    item_gap = 28.0
    positions = Dict{String,Tuple{Float64,Float64}}()

    if orientation === :horizontal
        rank_widths = [maximum(sizes[id][1] for id in group) for group in groups]
        group_heights = [sum(sizes[id][2] for id in group) + item_gap * max(0, length(group) - 1)
                         for group in groups]
        canvas_height = max(180.0, maximum(group_heights) + 2margin + title_height)
        x = margin
        for (i, group) in enumerate(groups)
            cx = x + rank_widths[i] / 2
            y = title_height + margin + (canvas_height - title_height - 2margin - group_heights[i]) / 2
            for id in group
                positions[id] = (cx, y + sizes[id][2] / 2)
                y += sizes[id][2] + item_gap
            end
            x += rank_widths[i] + rank_gap
        end
        canvas_width = max(240.0, x - rank_gap + margin)
    else
        rank_heights = [maximum(sizes[id][2] for id in group) for group in groups]
        group_widths = [sum(sizes[id][1] for id in group) + item_gap * max(0, length(group) - 1)
                        for group in groups]
        canvas_width = max(240.0, maximum(group_widths) + 2margin)
        y = title_height + margin
        for (i, group) in enumerate(groups)
            cy = y + rank_heights[i] / 2
            x = margin + (canvas_width - 2margin - group_widths[i]) / 2
            for id in group
                positions[id] = (x + sizes[id][1] / 2, cy)
                x += sizes[id][1] + item_gap
            end
            y += rank_heights[i] + rank_gap
        end
        canvas_height = max(180.0, y - rank_gap + margin)
    end
    positions, sizes, canvas_width, canvas_height
end

function _svg_edge_path(edge::_VizEdge, positions, sizes, orientation)
    sx, sy = positions[edge.src]
    dx, dy = positions[edge.dst]
    sw, sh = sizes[edge.src]
    dw, dh = sizes[edge.dst]
    if orientation === :horizontal
        x1, y1 = sx + sw / 2, sy
        x2, y2 = dx - dw / 2, dy
        bend = (x1 + x2) / 2
        "M $x1 $y1 C $bend $y1, $bend $y2, $x2 $y2"
    else
        x1, y1 = sx, sy + sh / 2
        x2, y2 = dx, dy - dh / 2
        bend = (y1 + y2) / 2
        "M $x1 $y1 C $x1 $bend, $x2 $bend, $x2 $y2"
    end
end

function _svg_node_class(node::_VizNode)
    string(node.kind, " ", node.state)
end

function _write_svg(io::IO, v::DAGVisualization)
    model = _viz_model(v)
    positions, sizes, width, height = _layout(model, v.orientation)
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100%\" viewBox=\"0 0 $width $height\" role=\"img\" aria-label=\"$(_xml_escape(model.title))\" data-rk-svg>")
    println(io, "<style>")
    println(io, ".rk-bg{fill:#fff}.rk-title{font:600 15px system-ui,sans-serif;fill:#0f172a}.rk-node text{font:12px system-ui,sans-serif;fill:#0f172a;pointer-events:none}.rk-node .detail{font-size:10px;fill:#475569}.rk-node rect,.rk-node ellipse{stroke-width:1.5}.rk-node.value ellipse{fill:#fff;stroke:#475569}.rk-node.recipe rect{fill:#f1f5f9;stroke:#475569}.rk-node.selected rect{fill:#dbeafe;stroke:#2563eb}.rk-node.have ellipse{fill:#dcfce7;stroke:#15803d;stroke-width:2.5}.rk-node.want ellipse{fill:#ffedd5;stroke:#c2410c;stroke-width:2.5}.rk-node.havewant ellipse{fill:#fef3c7;stroke:#a16207;stroke-width:2.5}.rk-node.alternative rect{fill:#f8fafc;stroke:#94a3b8;stroke-dasharray:6 4}.rk-node.effectful rect{fill:#fee2e2;stroke:#b91c1c;stroke-dasharray:6 4}.rk-node.is-inspected rect,.rk-node.is-inspected ellipse{stroke:#7c3aed;stroke-width:4}.rk-node:focus{outline:none}.rk-node:focus rect,.rk-node:focus ellipse{stroke:#7c3aed;stroke-width:4}.rk-edge{fill:none;stroke:#64748b;stroke-width:1.5}.rk-edge.alternative{stroke:#94a3b8;stroke-dasharray:6 4}")
    println(io, "</style>")
    println(io, "<defs><marker id=\"rk-arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#64748b\"/></marker></defs>")
    println(io, "<rect class=\"rk-bg\" x=\"0\" y=\"0\" width=\"$width\" height=\"$height\"/>")
    println(io, "<text class=\"rk-title\" x=\"20\" y=\"25\">$(_xml_escape(model.title))</text>")
    for edge in model.edges
        path = _svg_edge_path(edge, positions, sizes, v.orientation)
        println(io, "<path class=\"rk-edge $(edge.state)\" d=\"$path\" marker-end=\"url(#rk-arrow)\" data-rk-src=\"$(_xml_escape(edge.src))\" data-rk-dst=\"$(_xml_escape(edge.dst))\"/>")
    end
    for node in model.nodes
        cx, cy = positions[node.id]
        w, h = sizes[node.id]
        tooltip = _xml_escape(node.label * " — " * node.detail)
        println(io, "<g class=\"rk-node $(_svg_node_class(node))\" role=\"button\" tabindex=\"0\" aria-label=\"$tooltip\" data-rk-id=\"$(_xml_escape(node.id))\" data-rk-kind=\"$(_xml_escape(node.kind))\" data-rk-state=\"$(_xml_escape(node.state))\" data-rk-label=\"$(_xml_escape(node.label))\" data-rk-detail=\"$(_xml_escape(node.detail))\">")
        println(io, "<title>$tooltip</title>")
        if node.kind === :value
            println(io, "<ellipse cx=\"$cx\" cy=\"$cy\" rx=\"$(w / 2)\" ry=\"$(h / 2)\"/>")
        else
            println(io, "<rect x=\"$(cx - w / 2)\" y=\"$(cy - h / 2)\" width=\"$w\" height=\"$h\" rx=\"8\"/>")
        end
        lines = _node_lines(node)
        first_detail = length(_svg_lines(node.label)) + 1
        line_height = 15.0
        y = cy - line_height * (length(lines) - 1) / 2
        println(io, "<text text-anchor=\"middle\">")
        for (i, line) in enumerate(lines)
            class = i >= first_detail ? " class=\"detail\"" : ""
            println(io, "<tspan$class x=\"$cx\" y=\"$y\">$(_xml_escape(line))</tspan>")
            y += line_height
        end
        println(io, "</text></g>")
    end
    print(io, "</svg>")
end

Base.show(io::IO, ::MIME"image/svg+xml", v::DAGVisualization) = _write_svg(io, v)
Base.show(io::IO, mime::MIME"image/svg+xml", x::Union{Graph,Plan}) =
    show(io, mime, visualize(x))

const _DAG_HTML_STYLE = raw"""
<style>
.rk-dag{--rk-bg:#fff;--rk-surface:#f8fafc;--rk-border:#dbe4f0;--rk-text:#132238;--rk-muted:#64748b;--rk-accent:#4f46e5;--rk-accent-soft:#eef2ff;container-type:inline-size;color:var(--rk-text);background:var(--rk-bg);border:1px solid var(--rk-border);border-radius:18px;overflow:hidden;font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;box-shadow:0 18px 48px rgba(30,41,59,.10),0 2px 8px rgba(30,41,59,.06)}
.rk-dag *{box-sizing:border-box}.rk-dag-toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 14px;border-bottom:1px solid var(--rk-border);background:linear-gradient(135deg,#fff 0%,#f8faff 72%,#f1f5ff 100%)}.rk-dag-heading{display:flex;min-width:0;flex:1;align-items:center;gap:10px}.rk-dag-mark{width:10px;height:10px;flex:0 0 auto;border-radius:999px;background:linear-gradient(135deg,#4f46e5,#8b5cf6);box-shadow:0 0 0 5px rgba(99,102,241,.11)}.rk-dag-title{overflow:hidden;font-size:13px;font-weight:750;letter-spacing:-.01em;text-overflow:ellipsis;white-space:nowrap}.rk-dag-engine{border:1px solid #c7d2fe;border-radius:999px;background:#eef2ff;color:#4338ca;padding:3px 8px;font-size:10px;font-weight:750;letter-spacing:.04em;text-transform:uppercase}.rk-dag-controls{display:flex;align-items:center;gap:5px}.rk-dag-toolbar button{display:inline-grid;min-width:32px;height:30px;place-items:center;appearance:none;border:1px solid var(--rk-border);border-radius:9px;background:rgba(255,255,255,.86);color:var(--rk-text);padding:0 9px;font:700 12px/1 inherit;cursor:pointer;box-shadow:0 1px 2px rgba(15,23,42,.04)}.rk-dag-toolbar button:hover{border-color:#a5b4fc;background:#eef2ff;color:#4338ca}.rk-dag-toolbar button:focus-visible,.rk-dag-picker select:focus-visible{outline:3px solid rgba(79,70,229,.22);outline-offset:2px}.rk-dag-status{width:100%;color:var(--rk-muted);font-size:11px}.rk-dag[data-rk-renderer="cytoscape-elk"] .rk-dag-status{display:none}
.rk-dag-workspace{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,26%);min-height:430px;max-height:min(760px,76vh)}.rk-dag-canvas{position:relative;min-width:0;min-height:430px;overflow:hidden;background-color:#fbfdff;background-image:radial-gradient(circle at 1px 1px,rgba(100,116,139,.17) 1px,transparent 0);background-size:22px 22px}.rk-dag-canvas:focus-visible{outline:3px solid rgba(79,70,229,.24);outline-offset:-3px}.rk-dag-stage,.rk-dag-fallback{position:absolute;inset:0}.rk-dag-stage[hidden],.rk-dag-fallback[hidden]{display:none}.rk-dag-fallback{overflow:hidden}.rk-dag-fallback svg{display:block;width:100%;height:100%;max-height:none}.rk-dag-empty{display:grid;height:100%;min-height:300px;place-items:center;color:var(--rk-muted);font-weight:600}
.rk-dag[data-rk-orientation="vertical"] .rk-dag-workspace,.rk-dag[data-rk-orientation="vertical"] .rk-dag-canvas{min-height:680px}
.rk-dag-inspector{min-width:0;overflow:auto;border-left:1px solid var(--rk-border);background:linear-gradient(180deg,#f8fafc 0%,#fff 100%);padding:18px}.rk-dag-picker{display:grid;gap:6px;margin:0 0 18px;color:var(--rk-muted);font-size:11px;font-weight:750;letter-spacing:.045em;text-transform:uppercase}.rk-dag-picker select{width:100%;min-width:0;border:1px solid var(--rk-border);border-radius:10px;background:var(--rk-bg);color:var(--rk-text);padding:8px 30px 8px 10px;font:600 12px/1.35 inherit;text-transform:none}.rk-dag-inspector h3{margin:0 0 14px;font-size:15px;line-height:1.35;letter-spacing:-.01em;overflow-wrap:anywhere}.rk-dag-inspector dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:8px 11px;margin:0}.rk-dag-inspector dt{font-size:11px;font-weight:750;color:var(--rk-muted);letter-spacing:.035em;text-transform:uppercase}.rk-dag-inspector dd{margin:0;overflow-wrap:anywhere;white-space:normal}.rk-dag-inspector .rk-muted{color:var(--rk-muted)}
@media(max-width:700px){.rk-dag-engine{display:none}.rk-dag-workspace{grid-template-columns:1fr;max-height:none}.rk-dag-inspector{border-top:1px solid var(--rk-border);border-left:0}.rk-dag-canvas{min-height:350px}}
@container(max-width:700px){.rk-dag-engine{display:none}.rk-dag-workspace{grid-template-columns:1fr;max-height:none}.rk-dag-inspector{border-top:1px solid var(--rk-border);border-left:0}.rk-dag-canvas{min-height:350px}}
.dark .rk-dag{--rk-bg:#111827;--rk-surface:#172033;--rk-border:#334155;--rk-text:#e5e7eb;--rk-muted:#a8b3c5;background:#111827;box-shadow:0 18px 52px rgba(0,0,0,.34)}.dark .rk-dag-toolbar{background:linear-gradient(135deg,#172033 0%,#161c32 100%)}.dark .rk-dag-engine{border-color:#4338ca;background:#252354;color:#c7d2fe}.dark .rk-dag-toolbar button{background:#1f2937}.dark .rk-dag-toolbar button:hover{background:#272b55}.dark .rk-dag-canvas{background-color:#101725;background-image:radial-gradient(circle at 1px 1px,rgba(148,163,184,.14) 1px,transparent 0)}.dark .rk-dag-inspector{background:linear-gradient(180deg,#172033 0%,#111827 100%)}
</style>
"""

const _DAG_HTML_SCRIPT = raw"""
<script>
(() => {
  const script = document.currentScript;
  const root = script && script.closest('.rk-dag');
  if (!root || root.dataset.rkReady === 'true' || root.dataset.rkReady === 'loading') return;
  root.dataset.rkReady = 'loading';
  const canvas = root.querySelector('[data-rk-canvas]');
  const stage = root.querySelector('[data-rk-stage]');
  const fallback = root.querySelector('[data-rk-fallback]');
  const modelNode = root.querySelector('[data-rk-model]');
  const status = root.querySelector('[data-rk-status]');
  if (!canvas || !stage || !fallback || !modelNode) return;

  let model;
  try {
    model = JSON.parse(modelNode.textContent || '{}');
  } catch (error) {
    root.dataset.rkReady = 'fallback';
    if (status) status.textContent = 'Static fallback · graph data could not be read';
    return;
  }
  if (!model.nodes || model.nodes.length === 0) {
    root.dataset.rkReady = 'true';
    root.dataset.rkRenderer = 'static-empty';
    if (status) status.textContent = 'Empty graph';
    return;
  }

  const hasElk = () => {
    try { return Boolean(window.cytoscape && window.cytoscape('layout', 'elk')); }
    catch (_) { return false; }
  };
  const loadScript = (src, integrity, ready) => new Promise((resolve, reject) => {
    if (ready()) { resolve(); return; }
    const existing = document.querySelector(`script[data-rk-vendor="${src}"]`);
    if (existing) {
      existing.addEventListener('load', () => ready() ? resolve() : reject(new Error(src)), {once:true});
      existing.addEventListener('error', reject, {once:true});
      return;
    }
    const tag = document.createElement('script');
    tag.src = src;
    tag.integrity = integrity;
    tag.crossOrigin = 'anonymous';
    tag.dataset.rkVendor = src;
    tag.addEventListener('load', () => ready() ? resolve() : reject(new Error(src)), {once:true});
    tag.addEventListener('error', reject, {once:true});
    document.head.appendChild(tag);
  });

  const libraries = window.__rkDagLibraries ||
    (window.__rkDagLoadBundledLibraries && window.__rkDagLoadBundledLibraries()) ||
    (window.__rkDagLibraries = (async () => {
    await loadScript(
      'https://unpkg.com/cytoscape@3.33.4/dist/cytoscape.min.js',
      'sha256-vNg/DjHrF1AmqBHbbcHyS0MmAA7f+kAqENB0jFvlV7Q=',
      () => Boolean(window.cytoscape),
    );
    await loadScript(
      'https://unpkg.com/elkjs@0.9.3/lib/elk.bundled.js',
      'sha256-sHRavX8jzZFpChWH43ftvhn9cjPHgzACkJNnIFRiFtQ=',
      () => Boolean(window.ELK),
    );
    await loadScript(
      'https://unpkg.com/cytoscape-elk@2.3.0/dist/cytoscape-elk.js',
      'sha256-Jay7neqbDcb+8OebaXVH7lblefDhwirkNTEx5yx+FAo=',
      hasElk,
    );
  })());

  libraries.then(() => {
    if (!hasElk()) throw new Error('ELK layout was not registered');
    const horizontal = model.orientation === 'horizontal';
    const isDark = () => document.documentElement.classList.contains('dark') ||
      (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    const styleFor = dark => {
      const palette = dark ? {
        text:'#e5e7eb', muted:'#9ca3af', edge:'#718096', value:'#182235', recipe:'#1e293b',
        selected:'#252b56', have:'#113b2d', want:'#492919', havewant:'#463a16', alternative:'#18202e'
      } : {
        text:'#172033', muted:'#64748b', edge:'#94a3b8', value:'#ffffff', recipe:'#f8fafc',
        selected:'#eef2ff', have:'#ecfdf5', want:'#fff7ed', havewant:'#fffbeb', alternative:'#f8fafc'
      };
      return [
        {selector:'node',style:{
          'shape':'ellipse','width':156,'height':62,'padding':'4px',
          'background-color':palette.value,'border-color':'#94a3b8','border-width':1.5,
          'label':'data(label)','color':palette.text,'font-size':'12px','font-weight':600,
          'font-family':'Inter, ui-sans-serif, system-ui, sans-serif','text-wrap':'wrap',
          'text-max-width':'160px','text-valign':'center','text-halign':'center',
          'min-zoomed-font-size':8,'transition-property':'opacity, border-width, border-color, background-color',
          'transition-duration':'140ms'
        }},
        {selector:'node[kind = "recipe"]',style:{'shape':'round-rectangle','background-color':palette.recipe,'border-color':'#64748b'}},
        {selector:'node[state = "selected"]',style:{'background-color':palette.selected,'border-color':'#4f46e5','border-width':2.2}},
        {selector:'node[state = "have"]',style:{'background-color':palette.have,'border-color':'#10b981','border-width':2.2}},
        {selector:'node[state = "want"]',style:{'background-color':palette.want,'border-color':'#f97316','border-width':2.2}},
        {selector:'node[state = "havewant"]',style:{'background-color':palette.havewant,'border-color':'#eab308','border-width':2.2}},
        {selector:'node[state = "alternative"]',style:{'background-color':palette.alternative,'border-color':'#94a3b8','border-style':'dashed'}},
        {selector:'node[state = "effectful"]',style:{'background-color':dark ? '#471d25' : '#fff1f2','border-color':'#e11d48','border-style':'dashed'}},
        {selector:'edge',style:{
          'width':1.6,'line-color':palette.edge,'target-arrow-color':palette.edge,
          'target-arrow-shape':'triangle','arrow-scale':.85,'curve-style':'taxi',
          'taxi-direction':horizontal ? 'rightward' : 'downward','taxi-turn':24,
          'taxi-turn-min-distance':10,'transition-property':'opacity, width, line-color, target-arrow-color',
          'transition-duration':'140ms'
        }},
        {selector:'edge[state = "alternative"]',style:{'line-style':'dashed','line-color':'#94a3b8','target-arrow-color':'#94a3b8'}},
        {selector:'.rk-focus',style:{'border-color':'#7c3aed','border-width':4,'z-index':20}},
        {selector:'edge.rk-focus',style:{'line-color':'#7c3aed','target-arrow-color':'#7c3aed','width':3,'z-index':20}},
        {selector:'.rk-neighbor',style:{'opacity':1}},
        {selector:'.rk-dimmed',style:{'opacity':.16}}
      ];
    };
    stage.hidden = false;
    const cy = window.cytoscape({
      container: stage,
      elements: [...model.nodes, ...model.edges],
      minZoom: .18,
      maxZoom: 3.5,
      boxSelectionEnabled: false,
      style: styleFor(isDark())
    });
    const layout = cy.layout({
      name:'elk', fit:true, padding:44, animate:false, nodeDimensionsIncludeLabels:true,
      elk:{
        'elk.algorithm':'layered','elk.direction':horizontal ? 'RIGHT' : 'DOWN',
        'elk.edgeRouting':'ORTHOGONAL','elk.spacing.nodeNode':'38',
        'elk.layered.spacing.nodeNodeBetweenLayers':'72',
        'elk.layered.nodePlacement.strategy':'NETWORK_SIMPLEX',
        'elk.layered.crossingMinimization.strategy':'LAYER_SWEEP'
      }
    });
    const field = name => root.querySelector(`[data-rk-inspect-${name}]`);
    const picker = root.querySelector('[data-rk-picker]');
    const labelFor = collection => collection.map(item => item.data('label')).join(' · ');
    const inspect = node => {
      if (!node || node.empty()) return;
      cy.elements().removeClass('rk-focus rk-neighbor rk-dimmed');
      const neighborhood = node.closedNeighborhood();
      cy.elements().difference(neighborhood).addClass('rk-dimmed');
      neighborhood.addClass('rk-neighbor');
      node.addClass('rk-focus');
      node.connectedEdges().addClass('rk-focus');
      field('title').textContent = node.data('label');
      field('kind').textContent = node.data('kind');
      field('state').textContent = node.data('state');
      field('detail').textContent = node.data('detail');
      const incoming = node.incomers('node');
      const outgoing = node.outgoers('node');
      field('incoming').textContent = incoming.length ? labelFor(incoming) : 'None';
      field('outgoing').textContent = outgoing.length ? labelFor(outgoing) : 'None';
      if (picker) picker.value = node.id();
    };
    const fit = () => { cy.resize(); cy.fit(undefined, 44); };
    const home = () => {
      fit();
      if (cy.zoom() < .68) {
        cy.zoom(.68);
        cy.center();
      }
    };
    const zoom = factor => {
      const level = Math.max(cy.minZoom(), Math.min(cy.maxZoom(), cy.zoom() * factor));
      cy.zoom({level, renderedPosition:{x:cy.width()/2,y:cy.height()/2}});
    };
    root.querySelector('[data-rk-fit]').addEventListener('click', fit);
    root.querySelector('[data-rk-zoom-in]').addEventListener('click', () => zoom(1.22));
    root.querySelector('[data-rk-zoom-out]').addEventListener('click', () => zoom(1/1.22));
    if (picker) picker.addEventListener('change', () => inspect(cy.$id(picker.value)));
    canvas.addEventListener('keydown', event => {
      if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(1.22); }
      else if (event.key === '-') { event.preventDefault(); zoom(1/1.22); }
      else if (event.key === '0') { event.preventDefault(); fit(); }
    });
    cy.on('tap', 'node', event => inspect(event.target));
    cy.on('tap', event => {
      if (event.target !== cy) return;
      cy.elements().removeClass('rk-focus rk-neighbor rk-dimmed');
      if (picker) picker.value = '';
    });
    cy.one('layoutstop', () => {
      stage.hidden = false;
      fallback.hidden = true;
      root.dataset.rkReady = 'true';
      root.dataset.rkRenderer = 'cytoscape-elk';
      requestAnimationFrame(home);
      root.dispatchEvent(new CustomEvent('rk-dag:ready', {bubbles:true}));
    });
    layout.run();
    root.rkDag = Object.freeze({
      fit, zoomIn:() => zoom(1.22), zoomOut:() => zoom(1/1.22),
      resize:() => requestAnimationFrame(home),
      inspect:id => inspect(cy.$id(id)), renderer:'cytoscape-elk', cy
    });
  }).catch(error => {
    root.dataset.rkReady = 'fallback';
    root.dataset.rkRenderer = 'svg-fallback';
    if (status) status.textContent = 'Static SVG fallback · interactive libraries unavailable';
    console.warn('ReactiveKernels DAG enhancement failed', error);
  });
})();
</script>
"""

function _write_html(io::IO, v::DAGVisualization; document::Bool = false)
    model = _viz_model(v)
    title = _xml_escape(model.title)
    if document
        println(io, "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>$title</title></head><body style=\"margin:16px;background:#f8fafc\">")
    end
    print(io, "<div class=\"rk-dag\" data-rk-orientation=\"$(_xml_escape(v.orientation))\" aria-label=\"Interactive $title\">", _DAG_HTML_STYLE)
    print(io, "<div class=\"rk-dag-toolbar\" role=\"toolbar\" aria-label=\"DAG navigation\">",
          "<div class=\"rk-dag-heading\"><span class=\"rk-dag-mark\" aria-hidden=\"true\"></span>",
          "<strong class=\"rk-dag-title\">$title</strong>",
          "<span class=\"rk-dag-engine\">Cytoscape · ELK</span></div>",
          "<div class=\"rk-dag-controls\"><button type=\"button\" data-rk-zoom-out aria-label=\"Zoom out\">−</button>",
          "<button type=\"button\" data-rk-fit>Fit</button>",
          "<button type=\"button\" data-rk-zoom-in aria-label=\"Zoom in\">+</button></div>",
          "<span class=\"rk-dag-status\" data-rk-status>Loading interactive graph…</span></div>")
    print(io, "<div class=\"rk-dag-workspace\"><div class=\"rk-dag-canvas\" data-rk-canvas tabindex=\"0\" role=\"application\" aria-label=\"Interactive DAG; drag to pan, scroll to zoom, and select a node to inspect\">",
          "<div class=\"rk-dag-stage\" data-rk-stage hidden></div><div class=\"rk-dag-fallback\" data-rk-fallback>")
    _write_svg(io, v)
    print(io, "</div></div><aside class=\"rk-dag-inspector\" aria-live=\"polite\" aria-label=\"Node inspector\">",
          "<label class=\"rk-dag-picker\">Inspect node<select data-rk-picker><option value=\"\">Choose a value or recipe</option>")
    for node in model.nodes
        print(io, "<option value=\"$(_xml_escape(node.id))\">$(_xml_escape(node.label))</option>")
    end
    print(io, "</select></label>",
          "<h3 data-rk-inspect-title>Select a value or recipe</h3>",
          "<dl><dt>Kind</dt><dd data-rk-inspect-kind class=\"rk-muted\">—</dd>",
          "<dt>State</dt><dd data-rk-inspect-state class=\"rk-muted\">—</dd>",
          "<dt>Details</dt><dd data-rk-inspect-detail class=\"rk-muted\">—</dd>",
          "<dt>Incoming</dt><dd data-rk-inspect-incoming class=\"rk-muted\">—</dd>",
          "<dt>Outgoing</dt><dd data-rk-inspect-outgoing class=\"rk-muted\">—</dd></dl></aside></div>",
          "<script type=\"application/json\" data-rk-model>")
    _write_html_model(io, model, v.orientation)
    print(io, "</script>", _DAG_HTML_SCRIPT, "</div>")
    document && print(io, "</body></html>")
end

Base.show(io::IO, ::MIME"text/html", v::DAGVisualization) = _write_html(io, v)
Base.show(io::IO, mime::MIME"text/html", x::Union{Graph,Plan}) =
    show(io, mime, visualize(x))

"""
    save_visualization(path, graph_or_plan; alternatives=false, orientation=:horizontal)
    save_visualization(path, visualization)

Write a visualization to `.html`, `.svg`, `.dot`, or `.gv`. HTML loads pinned
Cytoscape.js + ELK assets when its host has not bundled them and always carries
a self-contained SVG fallback. SVG is the dependency-free static surface, and
DOT preserves the graph for Graphviz and other compatible tools. Returns `path`.
"""
function save_visualization(path::AbstractString, v::DAGVisualization)
    ext = lowercase(splitext(path)[2])
    data = if ext == ".html"
        sprint(io -> _write_html(io, v; document = true))
    elseif ext == ".svg"
        sprint(show, MIME"image/svg+xml"(), v)
    elseif ext in (".dot", ".gv")
        dot_source(v)
    else
        throw(ArgumentError("save_visualization supports .html, .svg, .dot, and .gv paths"))
    end
    open(path, "w") do io
        write(io, data)
    end
    path
end

save_visualization(path::AbstractString, x::Union{Graph,Plan}; kwargs...) =
    save_visualization(path, visualize(x; kwargs...))
save_visualization(path::AbstractString, spec::KernelSpec; kwargs...) =
    save_visualization(path, visualize(spec; kwargs...))
