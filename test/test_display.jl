using ReactiveKernels
using Test

# Human-readable display of existing graphs, plans and kernels: the `Graph`
# text listing, `readable_code`, and source-aware recipe labels in `explain`
# and DAG views. Every view must read the authored sources, never print an
# operation's parameterized type, and never mutate what it displays.

@kernel display_standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5 * z^2
end

@kernel display_location_scale(standard, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)
    standardized(x::Float64)::Float64 = (x - location) / scale
    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.logpdf(z) - log_scale
    end
end

@kernel display_normal = display_location_scale(display_standard_normal)

@kernel display_model(x::Vector{Float64}, location, scale) = begin
    shifted = location + 1.0
    pointwise = plate(x, shifted, scale) do xi, li, si
        display_normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

_display_model_reference(x, location, scale) =
    sum(-0.5 * log(2π) - 0.5 * ((xi - (location + 1.0)) / scale)^2 - log(scale)
        for xi in x)

_has_linenumber(x) = x isa LineNumberNode ||
    x isa Expr && any(_has_linenumber, x.args)

@testset "graph, plan and code display" begin
    graph = kernel_graph(display_model)
    shown = sprint(show, graph)
    listing = sprint(show, MIME"text/plain"(), graph)

    @testset "Graph text listing is structural, not a struct dump" begin
        @test shown == "Graph($(length(ReactiveKernels._value_groups(graph))) values, " *
                       "$(length(graph.recipes)) recipes)"
        @test startswith(listing, "Graph with ")
        @test occursin("\nValues:\n", listing)
        @test occursin("\nRecipes:\n", listing)
        @test occursin("x::Vector{Float64}", listing)
        # A captured source shows as a function of the recipe's own inputs.
        @test occursin("shifted = (location) -> location + 1.0", listing)
        # The authored plate is listed with its scalar body nested beneath it,
        # and the composed endpoint's source is recovered from its definition.
        @test occursin("pointwise = plate(x, shifted, scale)", listing)
        @test occursin("plate body: have (", listing)
        @test occursin("standardized = (x, location, scale) -> (x - location) / scale",
                       listing)
        @test !occursin("Dict{", listing)
        @test !occursin("_KernelSourceOp", listing)
        @test !occursin("RuntimeGeneratedFunction", listing)
        # The listing names every registered recipe exactly once at top level.
        for recipe in graph.recipes
            @test occursin("[$(recipe.id)] ", listing)
        end
    end

    @testset "explain and DAG labels read recipe sources" begin
        p = plan(display_model)
        explained = explain(p)
        @test occursin("shifted = (location) -> location + 1.0", explained)
        @test occursin("pointwise = plate(x, shifted, scale)", explained)
        @test !occursin("_KernelSourceOp", explained)
        dot = dot_source(display_model)
        @test occursin("(location) -> location + 1.0", dot)
        @test !occursin("_KernelSourceOp", dot)
        html = sprint(show, MIME"text/html"(), visualize(display_model))
        @test !occursin("_KernelSourceOp", html)
        # Non-source operations keep their established `op(inputs)` lines.
        @test occursin("= sum(pointwise)", explained)
    end

    @testset "readable_code over a spec, plan and prepared kernel" begin
        from_spec = readable_code(display_model)
        from_plan = readable_code(plan(display_model))
        @test from_spec isa ReadableCode
        @test string(from_spec) == string(from_plan)
        text = string(from_spec)
        @test !occursin("#=", text)
        @test !occursin("__ops__", text)
        @test !occursin("_KernelSourceOp", text)
        @test !_has_linenumber(from_spec.expr)
        @test startswith(text, "# authored sources evaluated in: ")
        @test (@__MODULE__) in from_spec.modules
        @test occursin("\nfunction (x::Vector{Float64}, location::Any, scale::Any)\n" *
                       "    shifted = location + 1.0\n", text)
        @test occursin("(display_normal(li, si)).logpdf(xi)", text)
        @test occursin("shifted = location + 1.0", text)
        @test endswith(text, "\nend")
        # The displayed program is ordinary Julia.
        parsed = Meta.parseall(text)
        @test !any(arg -> arg isa Expr && arg.head === :error, parsed.args)
        @test sprint(show, MIME"text/plain"(), from_spec) == text
        @test sprint(show, from_spec) == text

        kernel = prepare(display_model)
        xs = [-1.0, 0.5, 2.0]
        @test kernel(xs, 0.2, 1.3) ≈ _display_model_reference(xs, 0.2, 1.3)
        prepared = string(readable_code(kernel))
        @test !occursin("#=", prepared)
        @test !occursin("__ops__", prepared)
        # `code_expr` remains the exact executable AST.
        @test code_expr(kernel) === kernel.ast
    end

    @testset "readable_code of an expression is a non-mutating copy" begin
        source = quote
            y = 2x
            @inline f(z) = z + y
            f(y)
        end
        original = deepcopy(source)
        code = readable_code(source; modules = (Base,))
        @test source == original
        @test _has_linenumber(source)
        @test !_has_linenumber(code.expr)
        @test code.modules == [Base]
        @test !occursin("#=", string(code))
        @test occursin("@inline f(z) = z + y", string(code))
    end

    @testset "rich display of readable code escapes HTML" begin
        code = readable_code(:(x < y && y > z))
        html = sprint(show, MIME"text/html"(), code)
        @test startswith(html, "<pre class=\"rk-readable-code\"><code class=\"language-julia\">")
        @test occursin("x &lt; y &amp;&amp; y &gt; z", html)
    end

    @testset "empty graph" begin
        @test sprint(show, Graph()) == "Graph(0 values, 0 recipes)"
        @test sprint(show, MIME"text/plain"(), Graph()) ==
              "Graph with 0 values and 0 recipes\nValues:\nRecipes: (none)"
    end
end

# The public structural inspection contract: `recipe_kind` classifies a
# recipe and `recipe_inventory` walks plate and scan bodies recursively. One
# source authority is defined here and replayed from its printed form.
const INVENTORY_SOURCE = raw"""
@kernel inventory_groups(groups, scale::Float64) = begin
    group_total = plate(groups, Ref(scale)) do observations, sigma
        pointwise = plate(observations, Ref(sigma)) do observation, s
            -log(s) - 0.5 * (observation / s)^2
        end
        sum(pointwise)
    end
    total = sum(group_total)
    return total
end
@kernel inventory_path(xs, gain) = begin
    updates = scan(xs, Ref(gain); init = 0.0) do carry, x, g
        next = carry + x * g
        (next, next)
    end
    return updates
end
@kernel inventory_panel(x, gain) = begin
    totals = plate(eachcol(x), Ref(gain)) do xs, g
        history = inventory_path(xs, g)
        sum(history)
    end
    total = sum(totals)
    return total
end
"""
Core.eval(@__MODULE__, Meta.parseall(INVENTORY_SOURCE))

_structure(program) = [(entry.kind, entry.depth) for entry in recipe_inventory(program)
                       if entry.kind !== :ordinary]
_body(entry) = entry.kind === :plate ? plate_body(entry.recipe) : scan_body(entry.recipe)

@testset "recipe kinds and the recursive structural inventory" begin
    groups = [[0.2, 0.7], Float64[], [-1.2]]
    x = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    # Each column contributes the sum of its running totals of `x * gain`.
    panel_reference(x, gain) = sum(sum(cumsum(column .* gain)) for column in eachcol(x))

    @testset "recipe_kind classifies every recipe" begin
        recipes = kernel_graph(inventory_panel).recipes
        @test all(in((:plate, :scan, :ordinary)), recipe_kind.(recipes))
        panel_plate = only(filter(recipe -> recipe_kind(recipe) === :plate, recipes))
        body = plate_body(panel_plate).recipes
        @test count(recipe -> recipe_kind(recipe) === :scan, body) == 1
        @test recipe_kind(only(filter(recipe -> recipe.outputs[1].name === :total, recipes))) ===
              :ordinary
    end

    @testset "entries are depth-first, with depth and parent" begin
        for program in (inventory_groups, kernel_graph(inventory_groups),
                        plan(inventory_groups), prepare(inventory_groups))
            @test _structure(program) == [(:plate, 0), (:plate, 1)]
        end
        for program in (inventory_panel, kernel_graph(inventory_panel),
                        plan(inventory_panel), prepare(inventory_panel))
            @test _structure(program) == [(:plate, 0), (:scan, 1)]
            entries = recipe_inventory(program)
            @test entries isa Vector
            for (index, entry) in enumerate(entries)
                @test entry.kind === recipe_kind(entry.recipe)
                if entry.parent == 0
                    @test entry.depth == 0
                else
                    @test entry.parent < index
                    @test entries[entry.parent].kind !== :ordinary
                    @test entries[entry.parent].depth == entry.depth - 1
                end
                # A plate or scan is followed by exactly its body's recipes.
                entry.kind === :ordinary && continue
                @test [child.recipe for child in entries if child.parent == index] ==
                      _body(entry).recipes
            end
        end
    end

    @testset "bound data and printed-source replay keep the structure" begin
        bound_panel = prepare(inventory_panel; bound = (; x))
        @test _structure(bound_panel) == [(:plate, 0), (:scan, 1)]
        @test bound_panel(0.5) ≈ panel_reference(x, 0.5)
        bound_groups = prepare(inventory_groups; bound = (; groups))
        @test _structure(bound_groups) == [(:plate, 0), (:plate, 1)]

        replay = Module(gensym(:InventoryReplay))
        Core.eval(replay, :(using ReactiveKernels))
        printed = join((sprint(Base.show_unquoted, definition)
                        for definition in Meta.parseall(INVENTORY_SOURCE).args
                        if !(definition isa LineNumberNode)), "\n")
        Core.eval(replay, Meta.parseall(printed))
        replayed_panel = getfield(replay, :inventory_panel)
        replayed_groups = getfield(replay, :inventory_groups)
        @test _structure(replayed_panel) == [(:plate, 0), (:scan, 1)]
        @test _structure(replayed_groups) == [(:plate, 0), (:plate, 1)]
        replayed_bound = prepare(replayed_panel; bound = (; x))
        @test _structure(replayed_bound) == [(:plate, 0), (:scan, 1)]
        @test replayed_bound(0.5) ≈ panel_reference(x, 0.5)
    end

    @testset "the inventory only reads its argument" begin
        graph = kernel_graph(inventory_panel)
        recipes = copy(graph.recipes)
        version = graph.version
        listing = sprint(show, MIME"text/plain"(), graph)
        recipe_inventory(graph)
        @test graph.recipes == recipes
        @test graph.version == version
        @test sprint(show, MIME"text/plain"(), graph) == listing
        # The listing renders the same walk: one line per inventory entry plus
        # one body header per plate or scan.
        @test occursin("    scan body: have (", listing)
        recipe_lines = filter(line -> occursin(r"^\s*\[-?\d+\] ", line),
                              split(listing, '\n'))
        @test length(recipe_lines) == length(recipe_inventory(graph))
    end
end
