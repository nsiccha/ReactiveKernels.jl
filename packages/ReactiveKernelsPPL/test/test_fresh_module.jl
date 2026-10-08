module FreshModuleLoweringTests
using Distributions, ReactiveKernels, ReactiveKernelsPPL, Test

function fresh_namespace()
    mod = Module(gensym(:FreshLowering))
    Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
    Core.eval(mod, :(import ReactiveKernels))
    Core.eval(mod, quote
        ReactiveKernels.@kernel basis(x) = begin
            doubled = x .* 2
            return doubled
        end
        basis_alias = basis
        plain_basis(x) = x .* 2
    end)
    helpers = Module(:Helpers)
    Core.eval(helpers, :(import ReactiveKernels))
    Core.eval(helpers, :(ReactiveKernels.@kernel imported_basis(x) = begin
        doubled = x .* 2
        return doubled
    end))
    Core.eval(mod, :(const Helpers = $helpers))
    Core.eval(mod, :(using .Helpers: imported_basis))
    Core.eval(mod, quote
        @rkppl transformed(x) = begin
            doubled = basis(x)
            doubled
        end
    end)
    mod
end

const DATA = (; x = [-0.5, 0.0, 0.5, 1.0], y = [-1.0, 0.1, 1.0, 0.4])

function exercise(lower; head = :basis, submodel = false, input = DATA,
        parametric = false)
    mod = fresh_namespace()
    # This function stays in its entry world after eval. Lowering itself must
    # discover the bindings; a caller-side invokelatest would hide the bug.
    call = head === :globalref ? GlobalRef(mod, :basis) : head
    argument = parametric ? :(intercept .+ x) : :x
    definition = submodel ? :(mean ~ transformed(x)) :
        Expr(:(=), :mean, Expr(:call, call, argument))
    ast = quote
        $definition
        intercept ~ Normal(0.0, 1.0)
        y .~ Normal.(intercept .+ mean, 1.0)
    end
    source = deepcopy(ast)
    data = deepcopy(input)
    bound = bind_data(lower(ast, data, mod), data)
    @test ast == source
    @test data == input
    @test isbound(bound)
    built = build_kernel(bound)
    @test built.layout.total == 1
    @test kernel_expr(bound, built.layout) ==
        Base.invokelatest(kernel_expr, bound, built.layout)
    density = prepare_query(built, bound, :sampler)
    for intercept in (-0.3, 0.0, 0.6)
        location = (parametric ? 3 : 1) .* intercept .+ 2 .* data.x
        expected = logpdf(Normal(), intercept) +
            sum(logpdf.(Normal.(location, 1.0), data.y))
        # Query execution has its own documented generated-code boundary.
        @test Base.invokelatest(density, [intercept]) ≈ expected
    end
    @test data == input
end

const LOWER_NAMES = (ast, data, mod) ->
    lower_rkppl(ast, keys(data); mod, conditioned = (:y,))

@testset "fresh kernels in parameter-dependent generated calls" begin
    exercise(LOWER_NAMES; parametric = true)
end

@testset "fresh-module values retain their data domain" begin
    for n in (0, 1, 7)
        x = Float64.(1:n) ./ max(n, 1) .- 0.5
        exercise(LOWER_NAMES; input = (; x, y = 2 .* x .+ 0.1))
    end
    exercise((ast, data, mod) -> lower_rkppl(ast, data; mod, conditioned = (:y,));
        input = (; x = 0.5, y = 1.1))
end

@testset "fresh-module bindings through public lowering entry points" begin
    for lower in (
        LOWER_NAMES,
        (ast, data, mod) -> lower_rkppl(ast, data; mod, conditioned = (:y,)),
        (ast, data, mod) -> lower_rkppl(ast, Dict(pairs(data)); mod, conditioned = (:y,)),
        (ast, data, mod) -> lower_rkppl(RKPPLModel(ast, mod), data; conditioned = (:y,)),
        (ast, data, mod) -> lower_rkppl(RKPPLModel(ast, mod), Dict(pairs(data)); conditioned = (:y,)),
        (ast, data, mod) -> RKPPLModel(ast, mod)(; x = data.x) | (; y = data.y),
    )
        exercise(lower)
    end
end

@testset "fresh aliases, imports, qualified calls and submodel bodies" begin
    for head in (:basis_alias, :plain_basis, :imported_basis,
            :(Helpers.imported_basis), :globalref)
        exercise(LOWER_NAMES; head)
    end
    exercise(LOWER_NAMES; submodel = true)
    exercise((ast, data, mod) -> RKPPLModel(ast, mod)(; x = data.x) | (; y = data.y);
        submodel = true)
end

@testset "same source prepares across fresh lowering namespaces" begin
    ast = quote
        intercept ~ Normal(0.0, 1.0)
        mean = basis(intercept .+ x)
        y .~ Normal.(mean, 1.0)
    end
    lower(mod, data) = bind_data(lower_rkppl(ast, keys(data);
        mod, conditioned = (:y,)), data)
    other_data = (; x = [-0.25, 0.25], y = [-0.4, 0.7])
    built_plan = lower(fresh_namespace(), DATA)
    other_plan = lower(fresh_namespace(), other_data)
    built = build_kernel(built_plan)
    own = build_kernel(other_plan)
    reused = prepare_query(built, other_plan, :sampler)
    direct = prepare_query(own, other_plan, :sampler)
    for u in ([-0.3], [0.0], [0.6])
        @test Base.invokelatest(reused, u) == Base.invokelatest(direct, u)
    end
end

@testset "undefined names still fail at lowering" begin
    mod = fresh_namespace()
    ast = :(begin
        mean = missing_basis(x)
        y .~ Normal.(mean, 1.0)
    end)
    # Required by ordinary name resolution: no binding supplies this callee.
    @test_throws SurfaceLoweringError lower_rkppl(ast, DATA; mod, conditioned = (:y,))
end
end
