module FreshConstructorTests

using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test
const RK = ReactiveKernels
const PPL = ReactiveKernelsPPL

function fresh_plan(shift)
    mod = Module(gensym(:FreshHelpers))
    Core.eval(mod, :(using ReactiveKernels, ReactiveKernelsPPL))
    model = Core.eval(mod, quote
        @kernel local_mean(a::Float64) = begin
            shifted::Float64 = a + $shift
            return shifted
        end
        @rkppl begin
            a ~ Normal(0, 1)
            mu = local_mean(a)
            y ~ Normal(mu, 1)
        end
    end)
    (; mod, plan = model() | (; y = 0.0))
end

uses_module(x, mod) = x isa GlobalRef && x.mod === mod
uses_module(x::QuoteNode, mod) = uses_module(x.value, mod)
uses_module(x::Expr, mod) = any(a -> uses_module(a, mod), x.args)

@testset "fresh builds share instructions and use their own helper graphs" begin
    shifts = (1.0, 2.0, -0.5, 1.0)
    cases = fresh_plan.(shifts)
    first_build = build_kernel(first(cases).plan)
    cache = PPL._kernel_constructors
    probe = RK._KernelConstructors(cache.mod)
    def = kernel_expr(first(cases).plan, first_build.layout)
    RK._kernel_eval_definition(probe, def)
    key = only(keys(probe.thunks))
    thunk = cache.thunks[key]
    jobs = [Threads.@spawn build_kernel(c.plan) for c in Base.tail(cases)]
    builds = [first_build; fetch.(jobs)]
    @test cache.thunks[key] === thunk
    for c in Base.tail(cases)
        RK._kernel_eval_definition(probe, kernel_expr(c.plan, assign_layout(c.plan)))
    end
    @test length(probe.thunks) == 1
    @test all(i -> builds[i].spec.graph !== first_build.spec.graph, 2:length(builds))

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    samplers = [prepare_sampler(built, c.plan, [0.0]; backend) for (built, c) in zip(builds, cases)]
    for i in (4, 2, 3, 1)
        query, shift = samplers[i], shifts[i]
        source = code_expr(query.kernel)
        @test uses_module(source, cases[i].mod)
        @test !any(j -> j != i && uses_module(source, cases[j].mod), eachindex(cases))
        for u in (0.0, 0.7)
            v, g = sampler_value_and_gradient!(query, zeros(1), [u])
            @test v ≈ -log(2pi) - (u^2 + (u + shift)^2) / 2
            @test only(g) ≈ -2u - shift
        end
    end

    # Compare the ordinary evaluator on the very same emission. Its helper
    # namespace is then identical too, so actual generated code can be compared.
    c = last(cases)
    built = last(builds)
    def = kernel_expr(c.plan, built.layout)
    spec = RK._kernel_eval_definition(PPL.PPLGeneratedModels, def)
    control = merge(built, (; spec = PPL._register_packed_check!(spec, built.layout)))
    ordinary = prepare_query(control, c.plan, :sampler)
    cached = prepare_query(built, c.plan, :sampler)
    @test isequal(code_expr(ordinary), code_expr(cached))
    @test isequal(RK._program_expression(ordinary.f), RK._program_expression(cached.f))
end

struct TypeCell
    value::Union{DataType,Nothing}
end

function Base.getindex(cell::TypeCell)
    yield()
    cell.value === nothing && error("missing element type")
    cell.value
end

function typed_definition(value)
    annotation = Expr(:ref, QuoteNode(value))
    :(identity_kernel(x::$annotation) = begin
        y = identity(x)
        return y
    end)
end

@noinline function transient_build(cache)
    cell = Ref{Any}(Float64)
    RK._kernel_eval_definition(cache, typed_definition(cell))
    WeakRef(cell)
end

@testset "the constructor releases its build inputs, including on an error" begin
    cache = RK._KernelConstructors(PPL.PPLGeneratedModels)
    weak = transient_build(cache)
    GC.gc()
    @test weak.value === nothing
    thunk = only(values(cache.thunks))
    outer = Any[:outer]
    task_local_storage(RK._CONSTRUCTOR_INPUTS, outer) do
        @test_throws ErrorException RK._kernel_eval_definition(cache, typed_definition(TypeCell(nothing)))
        @test RK._constructor_inputs() === outer
        @test only(values(cache.thunks)) === thunk
    end
    types = (Float32, Int64, Float64)
    jobs = [Threads.@spawn RK._kernel_eval_definition(cache, typed_definition(TypeCell(T))) for T in types]
    for (T, job) in zip(types, jobs)
        @test fetch(job).ports[:x] isa Value{T}
    end
end

@testset "Julia lowers authored type expressions on every fresh build" begin
    cache = RK._KernelConstructors(PPL.PPLGeneratedModels)
    for (annotation, value) in ((:(Vector{T} where T <: AbstractFloat), [2.5]),
                                (:(typeof(2^-1)), 0.5),
                                (:(let; local_type(x) = x; local_type(Float64); end), 2.5))
        def = :(identity_kernel(x::$annotation) = begin
            y = identity(x)
            return y
        end)
        for _ in 1:2
            k = prepare(RK._kernel_eval_definition(cache, def))
            @test Base.invokelatest(k, value) == value
        end
    end
    @test isempty(cache.thunks)
end

vocabulary(x) = sort(unique(x))
const LEVEL_MODEL = quote
    classes = vocabulary(group_fit)
    coef[levels(classes)] .~ Normal.(0, 1)
    mu = coef[group]
    y .~ Normal.(mu, 1)
end

@noinline function build_levels(cache, groups; source=LEVEL_MODEL, check=true)
    train = (; group=groups, group_fit=groups, y=Float64.(groups))
    design = (; group=[2, 1], y=[2., 4.])
    lower(d; kwargs...) = lower_rkppl(source, d; mod=(@__MODULE__), conditioned=(:y,), kwargs...)
    fit = frozen_values(bind_data(lower(train), train), :classes)
    plan = bind_data(lower(design; frozen=fit), design)
    # bind_data's structural-analysis table sweeps dead lowering owners
    # periodically. Give construction a separate copy to measure what the
    # build handle itself can keep alive after the caller drops its plan.
    check || (plan = deepcopy(plan))
    cached = build_kernel(plan; cache)
    if check
        fresh = build_kernel(plan)
        u = zeros(cached.layout.total)
        @test cached.layout.total == length(unique(groups)) + (source !== LEVEL_MODEL)
        @test cached.program == fresh.program
        target = prepare_query(cached, plan, :sampler)
        @test Base.invokelatest(target, u) ≈ Base.invokelatest(prepare_query(fresh, plan, :sampler), u)
        @test_throws PPL.ContractValidationError Base.invokelatest(target, zeros(length(u)+1))
    end
    (; levels=WeakRef(plan.frozen.values.classes), design=WeakRef(plan.columns[:y]), graph=WeakRef(cached.spec.graph))
end

@testset "an explicit constructor uses each frozen layout and releases the fit" begin
    cache = KernelBuildCache()
    build_levels(cache, [1, 2, 1])
    first_entry = cache.entry
    @test first_entry !== nothing
    build_levels(cache, [1, 2, 3])
    # This particular program reads the fitted width from a graph input.
    @test cache.entry === first_entry
    extended = copy(LEVEL_MODEL)
    push!(extended.args, :(extra ~ Normal(0, 1)))
    build_levels(cache, [1, 2, 3]; source=extended)
    @test cache.entry !== first_entry
    @test any(a.run === b.run for a in first_entry.constructor.steps,
              b in cache.entry.constructor.steps)

    refs = build_levels(cache, [4, 2, 1]; check=false)
    GC.gc(true)
    @test refs.levels.value === nothing
    @test refs.design.value === nothing
    @test refs.graph.value === nothing
end

@testset "one constructor splices independent generated helper namespaces" begin
    cache = KernelBuildCache()
    cases = fresh_plan.((1., -2., 3.))
    first_build = build_kernel(cases[1].plan; cache)
    entry = cache.entry
    @test entry !== nothing
    others = fetch.([Threads.@spawn build_kernel(c.plan; cache) for c in cases[2:end]])
    @test cache.entry === entry
    builds = [first_build; others]
    for (built, c, shift) in zip(builds, cases, (1., -2., 3.))
        target = prepare_query(built, c.plan, :sampler)
        @test Base.invokelatest(target, [0.5]) ≈ -log(2pi) - (0.5^2 + (0.5+shift)^2)/2
    end
    old = kernel_revision(first_build.spec)
    value!(first_build.spec.graph, :additional, Float64)
    @test old != kernel_revision(first_build.spec)
    @test all(v -> v.name !== :additional, values(others[1].spec.graph.values))
end

@testset "generated native calls get the code of the current namespace" begin
    cache = KernelBuildCache()
    source = quote
        a ~ Normal(0, 1)
        mu = adjustment(a)
        y ~ Normal(mu, 1)
    end
    entry = nothing
    for shift in (1., 4.)
        mod = Module(gensym(:NativeFit))
        generated = RK.compile(:(x -> x + $shift))
        Core.eval(mod, :(const adjustment = $generated))
        data = (; y=0.)
        plan = bind_data(lower_rkppl(source, data; mod, conditioned=(:y,)), data)
        built = build_kernel(plan; cache)
        @test cache.entry !== nothing
        entry === nothing || @test cache.entry === entry
        entry = cache.entry
        target = prepare_query(built, plan, :sampler)
        @test Base.invokelatest(target, [0.5]) ≈ -log(2pi) - (0.5^2 + (0.5+shift)^2)/2
    end
end

@testset "a unit-interval parameter reuses its graph constructor" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        y ~ Bernoulli(p)
    end
    plan = model() | (; y=true)
    cache = KernelBuildCache()
    first = build_kernel(plan; cache)
    entry = cache.entry
    @test entry !== nothing
    second = build_kernel(plan; cache)
    @test cache.entry === entry
    @test second.spec.graph !== first.spec.graph
    staged = prepare_query(second, plan, :sampler)
    fresh = prepare_query(build_kernel(plan), plan, :sampler)
    for u in ([-1.0], [0.5])
        p = 1 / (1 + exp(-only(u)))
        density = Base.invokelatest(staged, u)
        @test density ≈ log(12) + 3log(p) + 3log1p(-p)
        @test density ≈ Base.invokelatest(fresh, u)
    end
end

function check_constructor(source, designs)
    cache = KernelBuildCache()
    for data in designs
        plan = bind_data(lower_rkppl(source, data; mod=@__MODULE__, conditioned=(:y,)), data)
        first = build_kernel(plan; cache)
        entry = cache.entry
        @test entry !== nothing
        second = build_kernel(plan; cache)
        @test cache.entry === entry
        @test first.spec.graph !== second.spec.graph
        reference = build_kernel(plan)
        queries = map(m -> prepare_query(m, plan, :sampler), (first, second, reference))
        for v in (-0.3, 0.4)
            u = fill(v, reference.layout.total)
            a, b, r = map(q -> Base.invokelatest(q, u), queries)
            @test a ≈ r
            @test b ≈ r
        end
    end
end

@testset "finite interval constructors use the current bound data" begin
    designs = ((; y=[0.2, -0.1, 0.3], lower=0.4, upper=2.0),
               (; y=[0.5, 1.2], lower=-1.5, upper=0.75))
    for prior in (:(Uniform(0, 1)), :(Uniform(0.3, 2)), :(Uniform(lower, upper)),
                  :(truncated(Normal(0, 1), lower, upper)))
        check_constructor(quote
            p ~ $prior
            y .~ Normal.(p, 1)
        end, designs)
    end
end

@testset "scaled correlated coefficients retain graph assembly" begin
    check_constructor(quote
        b[1:2, 1:2] .~ Normal.(0, 1)
        L ~ LKJCholesky(2, 2.0)
        s[1:2] .~ LogNormal.(0, 1)
        v = b * transpose(s .* L)
        mu = v[g, 1] .+ x .* v[g, 2]
        y .~ Normal.(mu, 1)
    end, ((; g=[1, 2, 1], x=[1., 0., 2.], y=[0., 1., 0.5]),))
end

@testset "monotonic effects retain nested numerical calls" begin
    check_constructor(quote
        w ~ Dirichlet(alpha)
        v = cumsum(vcat(0.0, w))
        mu = v[r]
        y .~ Normal.(mu, 1)
    end, ((; alpha=[2., 3.], r=[1, 3, 2], y=[0.2, 0.75, 0.6]),))
end

@testset "nested numerical calls use the current model's binding" begin
    cache = KernelBuildCache()
    source = quote
        a ~ Normal(0, 1)
        mu = 2 * adjust(a)
        y ~ Normal(mu, 1)
    end
    entries = []
    for f in (sin, sin, cos)
        mod = Module(gensym(:Numerics))
        Core.eval(mod, :(const adjust = $f))
        data = (; y=1.5)
        plan = bind_data(lower_rkppl(source, data; mod, conditioned=(:y,)), data)
        built = build_kernel(plan; cache)
        @test cache.entry !== nothing
        push!(entries, cache.entry)
        query = prepare_query(built, plan, :sampler)
        a = 0.3
        @test Base.invokelatest(query, [a]) ≈ -log(2pi) - (a^2 + (data.y - 2f(a))^2)/2
    end
    @test entries[1] === entries[2]
end

@testset "a fitted closure keeps its ordinary build behavior" begin
    cache = KernelBuildCache()
    mod = Module(gensym(:FittedCapture))
    bias = [1.]
    f = let bias=bias
        x -> x + only(bias)
    end
    @test !PPL._build_type(Vector{T} where T<:typeof(f))
    @test PPL._build_type(Union{Nothing,Vector{T}} where T)
    Core.eval(mod, :(const adjustment = $f))
    source = quote
        a ~ Normal(0, 1)
        mu = adjustment(a)
        y ~ Normal(mu, 1)
    end
    data = (; y=0.)
    for shift in (1., 3.)
        bias[1] = shift
        plan = bind_data(lower_rkppl(source, data; mod, conditioned=(:y,)), data)
        built = build_kernel(plan; cache)
        @test cache.entry === nothing
        @test Base.invokelatest(prepare_query(built, plan, :sampler), [0.]) ≈
            -log(2pi) - shift^2/2
    end
end

end
