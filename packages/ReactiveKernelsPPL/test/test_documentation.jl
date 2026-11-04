module DocumentationTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Serialization, Test

const PPL = ReactiveKernelsPPL
const RAW = "## Location\n\nRaw **Markdown**, `code`, \$\\alpha\$ and <em>HTML</em>.\n"

"Reusable **leaf** definition."
@rkppl documented_leaf() = begin
    "A local coefficient."
    b ~ Normal(0, 1)
    return b
end

"Reusable outer definition."
@rkppl documented_outer() = begin
    "The nested use."
    child ~ documented_leaf()
    return child
end

@rkppl documented_scan() = begin
    @scan begin
        "Initial scoped state."
        h[1] ~ Normal(0, 1)
        for t in 2:T
            "Scoped state transition."
            h[t] ~ Normal(h[t-1], 1)
        end
    end
    return h
end

"A definition retained by its original value."
@rkppl redefined() = begin
    a ~ Normal(0, 1)
    return a
end
const original_definition = redefined
@rkppl redefined() = begin
    a ~ Normal(0, 2)
    return a
end

records(x) = [(d.path, d.owner, d.markdown, d.definition, d.origin) for d in
    rkppl_documentation(x)]
textat(x, path) = [d.markdown for d in rkppl_documentation(x; path)]
program(x) = sprint(print, Base.remove_linenums!(deepcopy(kernel_expr(x, assign_layout(x)))))

@testset "authored documentation capture and source replay" begin
    ast = quote
        "Intercept."
        a ~ Normal(0, 1)
        "Slope."
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        "Observed outcome."
        y .~ Normal.(mu, 1)
    end
    # The emitter form of an ordinary Julia docstring wrapper.
    i = findfirst(x -> Meta.isexpr(x, :(=), 2) && first(x.args) === :mu, ast.args)
    ast.args[i] = Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
        LineNumberNode(23, :authored_model), RAW, ast.args[i])
    m = RKPPLModel(ast, @__MODULE__)
    docs = rkppl_documentation(m)
    @test getfield.(docs, :path) == [(:a,), (:b,), (:mu,), (:y,)]
    @test textat(m, (:mu,)) == [RAW]
    @test only(rkppl_documentation(m; path=(:mu,))).file === :authored_model
    @test only(rkppl_documentation(m; path=(:mu,))).line == 23
    empty!(docs)
    @test length(rkppl_documentation(m)) == 4
    @test_throws ArgumentError rkppl_documentation(m; path=:mu)

    # Print the actual captured source, removing only line nodes, then parse.
    source = sprint(print, Base.remove_linenums!(deepcopy(m.ast)))
    replay = RKPPLModel(Meta.parse(source), @__MODULE__)
    @test records(replay) == records(m)
    @test occursin("Observed outcome.", source)
    @test records(deepcopy(m)) == records(m)
    @test records(lower_rkppl(ast, (:x, :y); conditioned=(:y,), mod=@__MODULE__)) == records(m)

    data = (; x=[0.2, -0.5, 0.7], y=[0.1, 0.4, -0.2])
    binding = m(; x=data.x)
    @test records(binding) == records(m)
    bound = binding | (; data.y)
    @test records(bound) == records(m)
    @test records(bind_data(bound, (; x=[0.3, 0.1], y=[0.7, 0.2]))) == records(m)
    @test records(condition(bound; y=data.y)) == records(m)
    @test records(deepcopy(bound)) == records(m)
    io = IOBuffer()
    serialize(io, bound)
    seekstart(io)
    @test records(deserialize(io)) == records(m)

    plain = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1)
    end, data; conditioned=(:y,), mod=@__MODULE__)
    plain = bind_data(plain, data)
    @test program(bound) == program(plain)
    @test coordinate_names(assign_layout(bound)) == coordinate_names(assign_layout(plain))
    built, baseline = build_kernel(bound), build_kernel(plain)
    @test PPL._same_program(built.program, baseline.program)
    @test !occursin("Observed outcome.", string(built.program))
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    u = [0.2, -0.3]
    query = prepare_sampler(built, bound, u; backend)
    reference = prepare_sampler(baseline, plain, u; backend)
    gradient, expected = similar(u), similar(u)
    value, _ = sampler_value_and_gradient!(query, gradient, u)
    other, _ = sampler_value_and_gradient!(reference, expected, u)
    @test isequal(value, other)
    @test isequal(gradient, expected)

    pinned = merge(m, (; a=0.25))
    @test records(pinned) == records(m)
    @test records(pinned(; x=data.x) | (; data.y)) == records(m)
    replacement = merge(m, :(a ~ Normal(1, 2)))
    @test records(replacement) == records(m)
    annotated = merge(m, quote
        "Replacement intercept."
        a ~ Normal(1, 2)
    end)
    @test textat(annotated, (:a,)) == ["Replacement intercept."]
    @test textat(m, (:a,)) == ["Intercept."]
end

@testset "definition, instance and nested declaration provenance" begin
    @test textat(documented_leaf, ()) == ["Reusable **leaf** definition."]
    @test textat(original_definition, ()) == ["A definition retained by its original value."]
    @test isempty(rkppl_documentation(redefined))
    m = @rkppl begin
        "First component."
        first ~ documented_outer()
        "Second component."
        second ~ documented_outer()
        "A literal name containing dots."
        var"first.child.b" ~ Normal(0, 1)
    end
    plan = bind_data(m())
    docs = rkppl_documentation(plan)
    @test first(docs).owner === :instance
    @test textat(plan, (:first,)) == ["First component.", "Reusable outer definition."]
    @test textat(plan, (:first, :child)) == ["The nested use.", "Reusable **leaf** definition."]
    @test textat(plan, (:first, :child, :b)) == ["A local coefficient."]
    @test textat(plan, (:second, :child, :b)) == ["A local coefficient."]
    @test textat(plan, (Symbol("first.child.b"),)) == ["A literal name containing dots."]
    d = only(rkppl_documentation(plan; path=(:first, :child, :b)))
    @test d.definition === :documented_leaf
    @test d.origin == (:b,)
    @test d.file !== nothing && d.line > 0
    @test records(bind_data(plan, (;))) == records(plan)

    scoped = merge(m, quote
        "Caller replacement."
        first.child.b ~ Normal(1, 2)
    end)
    @test textat(bind_data(scoped()), (:first, :child, :b)) == ["Caller replacement."]
    fixed = merge(m, (; var"first.child.b"=0.4))
    @test textat(bind_data(fixed()), (:first, :child, :b)) == ["A local coefficient."]
    variant = merge(documented_leaf, (; b=0.2))
    @test records(variant) == records(documented_leaf)
    annotated = merge(documented_leaf, quote
        "A revised local coefficient."
        b ~ Normal(1, 2)
    end)
    @test textat(annotated, (:b,)) == ["A revised local coefficient."]
    @test textat(annotated, ()) == ["Reusable **leaf** definition."]

    source = Meta.parse("\"Definition replay.\"\n@rkppl replayed() = begin\n\"Local replay.\"\na ~ Normal(0, 1)\nreturn a\nend")
    printed = sprint(print, Base.remove_linenums!(deepcopy(source)))
    mod = Module(:DocumentationReplay)
    Core.eval(mod, :(using ReactiveKernelsPPL))
    replayed = Core.eval(mod, Meta.parse(printed))
    @test textat(replayed, ()) == ["Definition replay."]
    @test textat(replayed, (:a,)) == ["Local replay."]
end

@testset "plate documentation is structural, independent of row count" begin
    m = @rkppl begin
        @plate for i in eachindex(x)
            "Per-cell component."
            z[i] ~ documented_leaf()
            "Cell observation."
            y[i] ~ Normal.(z[i], 1)
        end
    end
    plans = [m(; x=zeros(n)) | (; y=zeros(n)) for n in (2, 5)]
    @test records(plans[1]) == records(plans[2])
    @test textat(plans[1], (:z,)) == ["Per-cell component.", "Reusable **leaf** definition."]
    @test textat(plans[1], (:z, :b)) == ["A local coefficient."]
    @test textat(plans[1], (:y,)) == ["Cell observation."]
end

@testset "scan setup and step documentation" begin
    ast = quote
        @scan begin
            "Initial state."
            h[1] ~ Normal(0, 1)
            for t in 2:T
                "State transition."
                h[t] ~ Normal(h[t-1], 1)
            end
        end
    end
    captured = RKPPLModel(ast, @__MODULE__)
    plan = lower_rkppl(ast, (:T,); mod=@__MODULE__)
    @test records(plan) == records(captured)
    @test textat(plan, (:h,)) == ["Initial state.", "State transition."]
    @test length(plan.scans) == 1
    nested = lower_rkppl(quote
        trajectory ~ documented_scan()
    end, (:T,); mod=@__MODULE__)
    @test textat(nested, (:trajectory, :h)) ==
        ["Initial scoped state.", "Scoped state transition."]
end

@testset "documentation does not keep unused frozen inputs live" begin
    m = @rkppl begin
        "An unused data definition."
        unused = sum(x_fit)
        center = sum(x_fit) / length(x_fit)
        a ~ Normal(0, 1)
        mu = a .+ x .- center
        y .~ Normal.(mu, 1)
    end
    fit = m(; x_fit=[1.0, 3.0], x=[0.0, 1.0]) | (; y=[0.1, 0.2])
    frozen = frozen_values(fit, :center)
    data = (; x=[2.0, 4.0], y=[0.3, 0.4])
    design = bind_data(fit, data; frozen)
    reference = bind_data(fit, (; data..., x_fit=[1.0, 3.0]))
    @test records(design) == records(fit)
    @test program(design) == program(reference)
    built = build_kernel(fit)
    actual = prepare_query(built, design, :sampler)
    expected = prepare_query(built, reference, :sampler)
    @test isequal(Base.invokelatest(actual, [0.2]), Base.invokelatest(expected, [0.2]))
end

# The accessor must not even attempt component resolution or data callbacks.
struct UnresolvedFragment end
ReactiveKernelsPPL.sampling_fragment(::UnresolvedFragment) = error("accessor resolved a component")
const unresolved = UnresolvedFragment()
@testset "documentation queries only inspect retained records" begin
    m = @rkppl begin
        "Unresolved component."
        value ~ unresolved()
    end
    @test textat(m, (:value,)) == ["Unresolved component."]
end
end
