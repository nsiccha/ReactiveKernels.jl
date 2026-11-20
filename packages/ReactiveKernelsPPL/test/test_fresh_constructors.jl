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

end
