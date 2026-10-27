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

# A nested cell's per-index argument (`reference[i] .* sigma`) lowers to a
# private gensym-named producer. Every lowering names it anew, so its place
# among the other definitions must not follow name hashes.
module SelectedArgumentHelpers
level_codes(labels, source) = Int[findfirst(isequal(l), sort(unique(source))) for l in labels]
column(draws, codes, margin) = draws[codes, margin]
end

function selected_argument_namespace()
    mod = Module(gensym(:FreshSelected))
    Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
    Core.eval(mod, :(import ReactiveKernels))
    Core.eval(mod, :(const Helpers = $SelectedArgumentHelpers))
    Core.eval(mod, :(using .Helpers: level_codes, column))
    Core.eval(mod, quote
        @rkppl group_effects(g) = begin
            tau[1:1] .~ Exponential.(0.9)
            z[levels(g), 1:1] .~ Normal.(0, 1)
            return z .* tau[1]
        end
        @rkppl population_effects(X, ncoef, loc, scale) = begin
            beta[1:ncoef] .~ Normal.(loc, scale)
            return X * beta
        end
        ReactiveKernels.@kernel cell_reader(xs, alpha) = begin
            cells = ReactiveKernels.plate(xs, alpha) do x, a
                x * a
            end
            return cells
        end
        ReactiveKernels.@kernel shifted_normal(value, location, reference, scale) = begin
            r = ((value - location) + reference) / scale
            logdensity = (-0.5 * log(2pi) - log(scale)) - 0.5 * r * r
            return logdensity
        end
    end)
    mod
end

@testset "per-index cell arguments keep their place across fresh lowerings" begin
    ast = quote
        b ~ group_effects(subject)
        codes = level_codes(subject, subject)
        effect = column(b, codes, 1)
        X = hcat(ones(length(subject)))
        pop ~ population_effects(X, 1, 0.0, 0.7)
        alpha = pop .+ effect
        sigma ~ Exponential(0.9)
        cells = cell_reader(x, alpha)
        @plate for i = eachindex(y)
            y[i] .~ LogDensity.(shifted_normal, cells[i], reference[i] .* sigma, sigma)
        end
    end
    data = Dict{Symbol,Any}(:subject => ["b", "empty", "a"],
        :x => [[0.0, 0.5, 1.0], Float64[], [0.0, 0.3, 0.8, 1.4]],
        :reference => [[0.01, 0.02, 0.03], Float64[], [0.01, 0.02, 0.03, 0.04]],
        :y => [[0.1, 0.2, 0.4], Float64[], [0.0, 0.1, 0.3, 0.4]])
    lower() = bind_data(lower_rkppl(ast, data;
        mod = selected_argument_namespace(), conditioned = (:y,)), data)
    first_plan = lower()
    built = build_kernel(first_plan)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    expected = Base.invokelatest(prepare_query(built, first_plan, :sampler), u)
    @test isfinite(expected)
    for _ in 1:4
        density = prepare_query(built, lower(), :sampler)
        @test Base.invokelatest(density, u) == expected
    end
end

# Caller-owned laws whose geometry reads an argument: the program holds the
# authored argument expressions as data, so a function of the model module
# they call is a module reference in that data. `Block`'s shape at each index
# is its first argument; `Shifted`'s support starts at its argument.
module CallerLawHelpers
using ReactiveKernelsPPL
import Distributions
struct Block{L,S}
    n::Int
    loc::L
    scale::S
end
ReactiveKernelsPPL.sampling_logdensity(d::Block, x) =
    sum(Distributions.logpdf.(Distributions.Normal.(d.loc, d.scale), x); init = 0.0)
ReactiveKernelsPPL.sampling_geometry(::Type{Block}, args, shape) =
    ParameterGeometry((args[1],), args[1]; support = :real,
        constrain = (u, shape, args...) -> collect(u),
        unconstrain = (x, shape, args...) -> collect(x),
        logjac = (u, shape, args...) -> 0.0)
struct Shifted{L}
    lower::L
end
# Unnormalized, so the value shows the bound the program evaluated.
ReactiveKernelsPPL.sampling_logdensity(d::Shifted, x) = x > d.lower ? -x : -Inf
ReactiveKernelsPPL.sampling_geometry(::Type{Shifted}, args, shape) =
    ParameterGeometry(shape, 1; support = :lower_bounded,
        constrain = (u, shape, lower) -> lower + exp(u[1]),
        unconstrain = (x, shape, lower) -> [log(x - lower)],
        logjac = (u, shape, lower) -> u[1])
end

function caller_law_namespace()
    mod = Module(gensym(:FreshCallerLaw))
    Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
    Core.eval(mod, :(const Helpers = $CallerLawHelpers))
    Core.eval(mod, :(using .Helpers: Block, Shifted))
    Core.eval(mod, :(block_size(k) = k + 1))
    Core.eval(mod, :(lower_of(k) = k / 4))
    mod
end

@testset "a caller-owned law's arguments call model-module functions across fresh lowerings" begin
    # One joint latent per index, of `block_size(k[i])` values: 3 + 4 + 2
    # coordinates beside `s`'s.
    cell = quote
        s ~ LogNormal(0.0, 1.0)
        @plate for i in eachindex(k)
            z[i] ~ Block(block_size(k[i]), 0.0, s)
        end
    end
    cell_prior(u) = logpdf(LogNormal(0.0, 1.0), exp(u[1])) +
        sum(logpdf.(Normal(0.0, exp(u[1])), u[2:end]))
    # One model-level latent above `lower_of(k)` = 0.5.
    model = quote
        z ~ Shifted(lower_of(k))
    end
    model_prior(u) = -(0.5 + exp(u[1]))
    for (ast, data, total, prior) in ((cell, (; k = [2, 3, 1]), 10, cell_prior),
            (model, (; k = 2), 1, model_prior))
        lower() = bind_data(lower_rkppl(ast, data; mod = caller_law_namespace()), data)
        first_plan = lower()
        built = build_kernel(first_plan)
        @test built.layout.total == total
        u = [0.1i - 0.4 for i in 1:total]
        @test Base.invokelatest(prepare_query(built, first_plan, :prior), u) ≈ prior(u)
        # Only the first coordinate is transformed: `s`, or `z` above its bound.
        @test Base.invokelatest(prepare_query(built, first_plan, :log_jacobian), u) ≈ u[1]
        for _ in 1:2
            other = lower()
            own = Base.invokelatest(prepare_query(build_kernel(other), other, :prior), u)
            @test Base.invokelatest(prepare_query(built, other, :prior), u) == own
        end
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
