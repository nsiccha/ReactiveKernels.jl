using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# `model_view` displays an existing build: the planned ReactiveKernels program,
# its structural graph and coordinates from `build_kernel`'s `(; spec, layout)`
# alone, plus the generated pre-build program and a prepared query's program
# when the caller supplies them. Its code pane lowers the stored spec; display
# never binds, builds, prepares or evaluates the model again. Self-contained fixture.

module ModelViewFixture
const CALLS = Ref(0)
halved(x) = (CALLS[] += 1; x ./ 2)
end

@testset "prepared display prints the executable expression" begin
    model = @rkppl begin
        sigma ~ Exponential(1.0)
        y .~ Normal.(0.0, sigma)
    end
    bound = model() | (; y = [0.2, 0.5, -0.1])
    built = build_kernel(bound)
    query = prepare_query(built, bound, :sampler)
    displayed = model_view(built; query).prepared
    exact = Base.remove_linenums!(deepcopy(code_expr(query)))
    @test displayed == exact
    @test sprint(print, displayed) == sprint(print, exact)
    @test !occursin("# elided graph aliases", sprint(print, displayed))
    @test Base.invokelatest(query, [0.1]) isa Float64
end

_has_linenumber(x) = x isa LineNumberNode ||
    x isa Expr && any(_has_linenumber, x.args)

@testset "model_view of an existing build" begin
    model = @rkppl begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1.0)
        xs = ModelViewFixture.halved(x)
        mu = a .+ b .* xs
        @plate for i in eachindex(y2)
            y2[i] ~ Normal(a + b * x2[i], sigma)
        end
        y .~ Normal.(mu, sigma)
    end
    xs = [0.1, 0.4, -0.3, 1.2]
    bound = model(; x = xs, x2 = [0.3, 0.7]) |
        (; y = [0.2, 0.5, -0.1, 1.0], y2 = [0.1, 0.9])
    built = build_kernel(bound)
    query = prepare_query(built, bound, :sampler)
    u = [0.1, -0.2, 0.3]
    expected = Base.invokelatest(query, u)
    calls = ModelViewFixture.CALLS[]
    source_before = kernel_expr(bound, built.layout)
    @test calls >= 1

    @testset "built program, graph and coordinates from the build alone" begin
        view = model_view(built)
        @test view isa RKPPLModelView
        @test view.coordinates == coordinate_names(built.layout)
        # The data-only `xs` was computed once by `bind_data` and is a bound input.
        @test view.inputs == [:unconstrained, :x, :x2, :xs, :y, :y2]
        @test view.outputs == [:posterior]
        @test view.graph === kernel_graph(built.spec)
        @test view.source === nothing
        @test view.prepared === nothing
        code = string(view.code)
        @test view.code == Base.remove_linenums!(deepcopy(code_expr(built.spec)))
        @test !occursin("#=", code)
        @test !_has_linenumber(view.code)
        @test occursin("__ops__", code)
        @test occursin("posterior", code)

        text = sprint(show, MIME"text/plain"(), view)
        @test startswith(text, "RKPPL model view\nInputs (HAVE): ")
        @test occursin("Coordinates (3): a, b, sigma", text)
        @test occursin("── Planned ReactiveKernels program: code_expr(spec) ──\n", text)
        @test occursin("── Structural graph: kernel_graph(spec) ──\nGraph with ", text)
        @test occursin("plate body: have (", text)
        @test occursin("(not shown: pass `bound`", text)
        @test occursin("(not shown: pass `query`", text)
        @test !occursin("#=", text)
        @test !occursin("Dict{", text)
        @test occursin(code, text)
        @test sprint(show, view) ==
              "RKPPLModelView(3 coordinates, $(join(view.inputs, ", ")) -> " *
              "$(join(view.outputs, ", ")))"

        html = sprint(show, MIME"text/html"(), view)
        @test startswith(html, "<div class=\"rkppl-model-view\">")
        @test occursin("<details open><summary>Planned ReactiveKernels program", html)
        @test occursin("<summary>Structural graph", html)
        @test occursin("<pre class=\"rk-graph-listing\">", html)
        @test !occursin("Generated pre-build", html)
    end

    @testset "pre-build source and prepared query when supplied" begin
        view = model_view(built; bound, query)
        source = string(view.source)
        @test view.source == Base.remove_linenums!(deepcopy(source_before))
        @test occursin(" = ReactiveKernelsPPL._ppl_range_values(", source)
        @test !occursin("#=", source)
        @test !_has_linenumber(view.source)
        prepared = string(view.prepared)
        @test !occursin("#=", prepared)
        @test occursin("posterior", prepared)
        @test view.prepared == Base.remove_linenums!(deepcopy(code_expr(query)))

        text = sprint(show, MIME"text/plain"(), view)
        @test occursin("── Generated pre-build kernel definition: kernel_expr(bound, layout) ──\n" *
                       source, text)
        @test occursin("── Prepared query program: code_expr(query) ──\n", text)
        html = sprint(show, MIME"text/html"(), view)
        @test occursin("<summary>Generated pre-build kernel definition", html)
        @test occursin("<summary>Prepared query program", html)

        # A SamplerQuery shows its prepared value kernel.
        sampler = SamplerQuery(query, nothing, built.layout)
        @test model_view(built; query = sampler).prepared == view.prepared
        @test model_view(built.spec, built.layout).code == view.code
    end

    @testset "displaying reads the existing values only" begin
        # No data-only call reran, the bound program and its generated source
        # are unchanged, and the prepared query still evaluates the same value.
        @test ModelViewFixture.CALLS[] == calls
        @test kernel_expr(bound, built.layout) == source_before
        @test Base.invokelatest(query, u) == expected
        @test_throws ArgumentError model_view(built; bound = :not_a_plan)
    end
end
