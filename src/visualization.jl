# Dependency-free DAG visualization.
#
# DOT is the portable interchange format: callers can feed it to Graphviz,
# editors, documentation systems, or their own renderer.  A compact layered
# SVG renderer supplies zero-setup rich display in Julia notebooks and IDEs.
# Recipes remain explicit nodes so multi-input and multi-output computations
# are represented without inventing ambiguous hyper-edge notation.

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

"""
    visualize(graph; orientation=:horizontal) -> DAGVisualization
    visualize(plan; alternatives=false, orientation=:horizontal) -> DAGVisualization

Create a dependency-free DAG visualization. Values and recipes are separate
nodes, with edges `value → recipe → value`, so multi-input and multi-output
recipes remain unambiguous.

For a `Plan`, the default shows only selected recipes. Set `alternatives=true`
to add backward-reachable but unselected candidates as muted dashed nodes.
`orientation` may be `:horizontal` (left to right) or `:vertical` (top to
bottom).

Rich Julia displays render the result as self-contained SVG. The underlying
`Graph` and `Plan` types also expose this SVG representation directly, so a
notebook can display them without calling `visualize` explicitly.
"""
function _check_orientation(orientation::Symbol)
    orientation in (:horizontal, :vertical) ||
        throw(ArgumentError("orientation must be :horizontal or :vertical"))
    orientation
end

function visualize(x::Graph; alternatives::Bool = false,
                   orientation::Symbol = :horizontal)
    DAGVisualization(x, alternatives, orientation)
end

function visualize(x::Plan; alternatives::Bool = false,
                   orientation::Symbol = :horizontal)
    DAGVisualization(x, alternatives, orientation)
end

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
    print(io, "\n  rich display: self-contained SVG",
          "\n  interchange: dot_source(view)",
          "\n  files: save_visualization(\"plan.svg\", view) or save_visualization(\"plan.dot\", view)")
end

_xml_escape(s) = replace(string(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
                         '"' => "&quot;", '\'' => "&apos;")

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
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100%\" viewBox=\"0 0 $width $height\" role=\"img\" aria-label=\"$(_xml_escape(model.title))\">")
    println(io, "<style>")
    println(io, ".rk-bg{fill:#fff}.rk-title{font:600 15px system-ui,sans-serif;fill:#0f172a}.rk-node text{font:12px system-ui,sans-serif;fill:#0f172a}.rk-node .detail{font-size:10px;fill:#475569}.rk-node rect,.rk-node ellipse{stroke-width:1.5}.rk-node.value ellipse{fill:#fff;stroke:#475569}.rk-node.recipe rect{fill:#f1f5f9;stroke:#475569}.rk-node.selected rect{fill:#dbeafe;stroke:#2563eb}.rk-node.have ellipse{fill:#dcfce7;stroke:#15803d;stroke-width:2.5}.rk-node.want ellipse{fill:#ffedd5;stroke:#c2410c;stroke-width:2.5}.rk-node.havewant ellipse{fill:#fef3c7;stroke:#a16207;stroke-width:2.5}.rk-node.alternative rect{fill:#f8fafc;stroke:#94a3b8;stroke-dasharray:6 4}.rk-node.effectful rect{fill:#fee2e2;stroke:#b91c1c;stroke-dasharray:6 4}.rk-edge{fill:none;stroke:#64748b;stroke-width:1.5}.rk-edge.alternative{stroke:#94a3b8;stroke-dasharray:6 4}")
    println(io, "</style>")
    println(io, "<defs><marker id=\"rk-arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#64748b\"/></marker></defs>")
    println(io, "<rect class=\"rk-bg\" x=\"0\" y=\"0\" width=\"$width\" height=\"$height\"/>")
    println(io, "<text class=\"rk-title\" x=\"20\" y=\"25\">$(_xml_escape(model.title))</text>")
    for edge in model.edges
        path = _svg_edge_path(edge, positions, sizes, v.orientation)
        println(io, "<path class=\"rk-edge $(edge.state)\" d=\"$path\" marker-end=\"url(#rk-arrow)\"/>")
    end
    for node in model.nodes
        cx, cy = positions[node.id]
        w, h = sizes[node.id]
        tooltip = _xml_escape(node.label * " — " * node.detail)
        println(io, "<g class=\"rk-node $(_svg_node_class(node))\">")
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

"""
    save_visualization(path, graph_or_plan; alternatives=false, orientation=:horizontal)
    save_visualization(path, visualization)

Write a visualization to `.svg`, `.dot`, or `.gv`. SVG is self-contained and
requires no renderer at read time; DOT preserves the graph for Graphviz and
other compatible tools. Returns `path`.
"""
function save_visualization(path::AbstractString, v::DAGVisualization)
    ext = lowercase(splitext(path)[2])
    data = if ext == ".svg"
        sprint(show, MIME"image/svg+xml"(), v)
    elseif ext in (".dot", ".gv")
        dot_source(v)
    else
        throw(ArgumentError("save_visualization supports .svg, .dot, and .gv paths"))
    end
    open(path, "w") do io
        write(io, data)
    end
    path
end

save_visualization(path::AbstractString, x::Union{Graph,Plan}; kwargs...) =
    save_visualization(path, visualize(x; kwargs...))
