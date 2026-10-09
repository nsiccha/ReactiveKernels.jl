using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# `model_view` displays an existing build: the built ReactiveKernels program,
# its structural graph and coordinates from `build_kernel`'s `(; spec, layout)`
# alone, plus the generated pre-build program and a prepared query's program
# when the caller supplies them. Displaying never lowers, binds, builds,
# prepares or evaluates the model again. Self-contained fixture.

module ModelViewFixture
const CALLS = Ref(0)
halved(x) = (CALLS[] += 1; x ./ 2)
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
        @test string(readable_code(built.spec)) == code
        @test !occursin("#=", code)
        @test !occursin("__ops__", code)
        @test !occursin("_KernelSourceOp", code)
        @test !_has_linenumber(view.code.expr)
        @test occursin("ReactiveKernelsPPL.PPLGeneratedModels", code)
        @test occursin("posterior", code)

        text = sprint(show, MIME"text/plain"(), view)
        @test startswith(text, "RKPPL model view\nInputs (HAVE): ")
        @test occursin("Coordinates (3): a, b, sigma", text)
        @test occursin("── Built ReactiveKernels program: readable_code(spec) ──\n", text)
        @test occursin("── Structural graph: kernel_graph(spec) ──\nGraph with ", text)
        @test occursin("plate body: have (", text)
        @test occursin("(not shown: pass `bound`", text)
        @test occursin("(not shown: pass `query`", text)
        @test !occursin("#=", text)
        @test !occursin("_KernelSourceOp", text)
        @test !occursin("Dict{", text)
        @test occursin(code, text)
        @test sprint(show, view) ==
              "RKPPLModelView(3 coordinates, $(join(view.inputs, ", ")) -> " *
              "$(join(view.outputs, ", ")))"

        html = sprint(show, MIME"text/html"(), view)
        @test startswith(html, "<div class=\"rkppl-model-view\">")
        @test occursin("<details open><summary>Built ReactiveKernels program", html)
        @test occursin("<summary>Structural graph", html)
        @test occursin("<pre class=\"rk-graph-listing\">", html)
        @test !occursin("Generated pre-build", html)
        @test !occursin("_KernelSourceOp", html)
    end

    @testset "pre-build source and prepared query when supplied" begin
        view = model_view(built; bound, query)
        source = string(view.source)
        # A global prints by its bare name; the comment names its module.
        @test startswith(source, "# authored sources evaluated in: " *
                         "ReactiveKernelsPPL.PPLGeneratedModels, ReactiveKernelsPPL\n" *
                         "@kernel ppl_model(" )
        @test occursin(" = _ppl_range_values(", source)
        @test !occursin("ReactiveKernelsPPL._ppl_range_values", source)
        @test occursin(") = begin\n    a::Float64 = ", source)
        @test endswith(source, "\nend")
        @test !occursin("#=", source)
        @test !_has_linenumber(view.source.expr)
        prepared = string(view.prepared)
        @test !occursin("#=", prepared)
        @test !occursin("__ops__", prepared)
        @test occursin("posterior", prepared)
        @test view.prepared.expr == readable_code(query).expr

        text = sprint(show, MIME"text/plain"(), view)
        @test occursin("── Generated pre-build @kernel program: kernel_expr(bound, layout) ──\n" *
                       source, text)
        @test occursin("── Prepared query program: readable_code(query) ──\n", text)
        html = sprint(show, MIME"text/html"(), view)
        @test occursin("<summary>Generated pre-build @kernel program", html)
        @test occursin("<summary>Prepared query program", html)

        # A SamplerQuery shows its prepared value kernel.
        sampler = SamplerQuery(query, nothing, built.layout)
        @test model_view(built; query = sampler).prepared.expr == view.prepared.expr
        @test model_view(built.spec, built.layout).code.expr == view.code.expr
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
