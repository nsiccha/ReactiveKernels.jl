using ReactiveKernels
using Test

viz_cheap(x) = x
viz_combined(x) = (x + 1, x + 10)
viz_b(a) = a + 100
viz_c(a) = a + 1000
viz_finish(b, c) = b + c

@testset "DAG visualization" begin
    g = Graph()
    u = value!(g, Symbol("u<&\""), Float64)
    a = value!(g, :a, Float64)
    b = value!(g, :b, Float64)
    c = value!(g, :c, Float64)
    out = value!(g, :out, Float64)

    add!(g, u => a, viz_cheap; cost = 1.0)
    add!(g, u => (a, b), viz_combined; cost = 1.2)
    add!(g, a => b, viz_b; cost = 1.0)
    add!(g, a => c, viz_c; cost = 1.0)
    add!(g, (b, c) => out, viz_finish; cost = 1.0)

    p = plan(g; have = (u,), want = (out,))
    view = visualize(p)

    @test view isa DAGVisualization
    @test occursin("DAGVisualization(Plan", sprint(show, MIME"text/plain"(), view))

    source = dot_source(view)
    @test startswith(source, "digraph ReactiveKernels")
    @test occursin("rankdir=LR", source)
    @test occursin("viz_combined", source)
    @test occursin("viz_finish", source)
    @test !occursin("viz_b", source)
    @test occursin("u<&\\\"", source)
    @test source == sprint(show, MIME"text/vnd.graphviz"(), p)

    with_alternatives = dot_source(p; alternatives = true, orientation = :vertical)
    @test occursin("rankdir=TB", with_alternatives)
    @test occursin("viz_b", with_alternatives)
    @test occursin("style=dashed", with_alternatives)

    svg = sprint(show, MIME"image/svg+xml"(), view)
    @test startswith(svg, "<svg")
    @test occursin("viz_combined", svg)
    @test occursin("u&lt;&amp;&quot;", svg)
    @test occursin("<title>", svg)
    @test !occursin('…', svg)
    @test svg == sprint(show, MIME"image/svg+xml"(), p)

    html = sprint(show, MIME"text/html"(), view)
    @test showable(MIME"text/html"(), view)
    @test startswith(html, "<div class=\"rk-dag\"")
    @test occursin("data-rk-fit", html)
    @test occursin("data-rk-zoom-in", html)
    @test occursin("data-rk-canvas", html)
    @test occursin("data-rk-inspect-title", html)
    @test occursin("data-rk-id=\"r_", html)
    @test occursin("data-rk-src=", html)
    @test occursin("root.rkDag", html)
    @test occursin("u&lt;&amp;&quot;", html)
    @test !occursin("<script src=", html)
    @test html == sprint(show, MIME"text/html"(), p)

    graph_source = dot_source(g)
    @test occursin("viz_cheap", graph_source)
    @test occursin("viz_b", graph_source)
    @test occursin("viz_combined", graph_source)

    mktempdir() do dir
        svg_path = save_visualization(joinpath(dir, "plan.svg"), p)
        html_path = save_visualization(joinpath(dir, "plan.html"), p)
        dot_path = save_visualization(joinpath(dir, "plan.dot"), view)
        @test startswith(read(svg_path, String), "<svg")
        @test startswith(read(html_path, String), "<!doctype html>")
        @test occursin("root.rkDag", read(html_path, String))
        @test startswith(read(dot_path, String), "digraph")
        @test_throws ArgumentError save_visualization(joinpath(dir, "plan.png"), p)
    end

    @test_throws ArgumentError visualize(p; orientation = :diagonal)
    @test_throws ArgumentError visualize(g; alternatives = true)

    empty_graph = Graph()
    @test occursin("digraph ReactiveKernels", dot_source(empty_graph))
    @test occursin("ReactiveKernels graph",
                   sprint(show, MIME"image/svg+xml"(), empty_graph))
    @test occursin("Select a value or recipe",
                   sprint(show, MIME"text/html"(), empty_graph))

    alias_graph = Graph()
    alias_input = value!(alias_graph, :alias_input, Int)
    alias_1 = value!(alias_graph, :alias_1, Int)
    alias_2 = value!(alias_graph, :alias_2, Int)
    add!(alias_graph, alias_input => alias_1, identity; cse_key = :same)
    add!(alias_graph, alias_input => alias_2, identity; cse_key = :same)
    @test occursin("alias_1 ≡ alias_2", dot_source(alias_graph))

    cyclic_graph = Graph()
    cycle_a = value!(cyclic_graph, :cycle_a, Int)
    cycle_b = value!(cyclic_graph, :cycle_b, Int)
    add!(cyclic_graph, cycle_a => cycle_b, identity)
    add!(cyclic_graph, cycle_b => cycle_a, identity; effectful = true)
    cyclic_svg = sprint(show, MIME"image/svg+xml"(), cyclic_graph)
    @test occursin("cycle_a", cyclic_svg)
    @test occursin("cycle_b", cyclic_svg)
    @test occursin("rk-node recipe effectful", cyclic_svg)
end

@testset "visualization exercises corrected planner boundaries" begin
    # HAVE is authoritative even when a selected recipe has overlapping
    # multi-output. The visualization must show the actual valid plan, while
    # prepared execution must preserve the supplied `a` boundary value.
    g = Graph()
    x = value!(g, :x, Int)
    a = value!(g, :a, Int)
    b = value!(g, :b, Int)
    out = value!(g, :out, Int)
    add!(g, x => (a, b), x -> (100, x))
    add!(g, (a, b) => out, +)
    p = plan(g; have = (x, a), want = (out,))
    @test occursin("a", dot_source(p))
    @test prepare(p)(2, 10) == 12

    # Overlapping producers still admit a valid acyclic plan. This was a
    # planner regression discovered during the visualization lane.
    g2 = Graph()
    x2 = value!(g2, :x, Int)
    a2 = value!(g2, :a, Int)
    b2 = value!(g2, :b, Int)
    c2 = value!(g2, :c, Int)
    d2 = value!(g2, :d, Int)
    add!(g2, d2 => (a2, b2), d -> (d, d + 1))
    add!(g2, x2 => (a2, c2), x -> (x, x + 2))
    add!(g2, a2 => d2, a -> a + 3)
    p2 = plan(g2; have = (x2,), want = (b2, c2))
    @test length(p2.recipes) == 3
    @test occursin("total cost 3.0", sprint(show, MIME"image/svg+xml"(), p2))

    # Reserved-looking value names remain ordinary data in all renderers and
    # cannot collide with the generated kernel's internal argument names.
    g3 = Graph()
    reserved = value!(g3, :__ops__, Int)
    reserved_out = value!(g3, :__rk_values__, Int)
    add!(g3, reserved => reserved_out, identity)
    p3 = plan(g3; have = (reserved,), want = (reserved_out,))
    @test prepare(p3)(7) == 7
    @test occursin("__ops__", dot_source(p3))
    @test occursin("__rk_values__", sprint(show, MIME"text/html"(), p3))
end
