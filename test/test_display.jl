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
