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
    # Each lowering namespace is a new module, and `basis` a new kernel in it;
    # the two builds still prepare to one executable type, so the second
    # reuses the code compiled for the first (snag `compiled-reader-be3cadbb`).
    @test typeof(reused) === typeof(direct)
    for u in ([-0.3], [0.0], [0.6])
        @test Base.invokelatest(reused, u) == Base.invokelatest(direct, u)
    end
end

@testset "fresh namespaces share a kernel's branch closures" begin
    # A kernel with a lazy branch, evaluated anew in each lowering namespace (as
    # BRM evaluates its kernels). Julia lowers the branch's tensorized arms
    # itself, so each namespace defined new closure types; the two builds now
    # take one type, as they read the same bindings (snag `compiled-reader-be3cadbb`).
    function branch_namespace()
        mod = Module(gensym(:FreshBranch))
        Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
        Core.eval(mod, :(import ReactiveKernels))
        Core.eval(mod, :(ReactiveKernels.@kernel signed(v) = begin
            w = sum(v) > 0 ? v : -v
            return w
        end))
        mod
    end
    ast = quote
        intercept ~ Normal(0.0, 1.0)
        mean = signed(intercept .+ x)
        y .~ Normal.(mean, 1.0)
    end
    queries = map(1:2) do _
        plan = bind_data(lower_rkppl(ast, keys(DATA); mod = branch_namespace(),
            conditioned = (:y,)), DATA)
        prepare_query(build_kernel(plan), plan, :sampler)
    end
    @test typeof(queries[1]) === typeof(queries[2])
    for intercept in (-0.3, 0.6)
        v = intercept .+ DATA.x
        location = sum(v) > 0 ? v : -v
        expected = logpdf(Normal(), intercept) + sum(logpdf.(Normal.(location, 1.0), DATA.y))
        @test Base.invokelatest(queries[2], [intercept]) ≈ expected
        @test Base.invokelatest(queries[1], [intercept]) == Base.invokelatest(queries[2], [intercept])
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

@testset "one program built twice compiles once, and another stays another" begin
    # A build evaluates its definition anew. What the expansion names (a lifted
    # cell invariant, in a recipe whose split nodes carry it in their type) must
    # come out the same, or every reader of the type compiles again on every
    # build (snag `gensym-in-kernel-54164b85`). Two programs that differ only in
    # a literal share those names, so the guard against evaluating one with the
    # other's build (snag `rkppl-opaque-loc-37de7b80`) has to hold regardless.
    source(scale) = quote
        log_k ~ Normal(0.0, 1.0)
        z[levels(g)] .~ Normal.(0.0, 1.0)
        s ~ Exponential(1.0)
        @plate for k in eachindex(y)
            y[k] ~ Normal(sin(exp(log_k)) * cos(x[k] * w[k]) * $scale + z[g[k]],
                s + exp(log_k))
        end
    end
    data = (; x = [-0.5, 0.0, 0.5, 1.0], w = [0.1, 0.2, 0.3, 0.4], g = [1, 1, 2, 2],
        y = [-1.0, 0.1, 1.0, 0.4])
    function build(ast)
        mod = Module(gensym(:FreshBuild))
        Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
        plan = bind_data(lower_rkppl(ast, keys(data); mod, conditioned = (:y,)), data)
        built = build_kernel(plan)
        (; plan, built, query = prepare_query(built, plan, :sampler))
    end
    operations(spec) = Set(string(typeof(r.op)) for r in spec.graph.recipes)
    one, two, other = build(source(1.5)), build(source(1.5)), build(source(2.5))
    @test typeof(one.query) === typeof(two.query)
    @test operations(one.built.spec) == operations(two.built.spec)
    # The comparison means something only if a generated name reaches a type.
    @test any(occursin("##cell_invariant#", t) for t in operations(one.built.spec))
    @test typeof(prepare_query(one.built, two.plan, :sampler)) === typeof(one.query)
    @test_throws "different program" prepare_query(one.built, other.plan, :sampler)
    density(u, scale) = begin
        log_k, s, z = u[1], exp(u[2]), u[3:4]
        lp = logpdf(Normal(0.0, 1.0), log_k) + logpdf(Exponential(1.0), s) + u[2] +
             sum(logpdf.(Normal(0.0, 1.0), z))
        for k in eachindex(data.y)
            m = sin(exp(log_k)) * cos(data.x[k] * data.w[k]) * scale + z[data.g[k]]
            lp += logpdf(Normal(m, s + exp(log_k)), data.y[k])
        end
        lp
    end
    for u in ([0.2, -0.3, 0.4, 0.1], [-0.6, 0.5, -0.2, 0.7])
        @test Base.invokelatest(one.query, u) == Base.invokelatest(two.query, u)
        @test Base.invokelatest(one.query, u) ≈ density(u, 1.5)
        @test Base.invokelatest(other.query, u) ≈ density(u, 2.5)
    end

    # What lowering itself names reaches the split nodes' types too: a selected
    # argument with no definition of its own.
    selected = quote
        z[levels(g)] .~ Normal.(0.0, 1.0)
        s ~ Exponential(1.0)
        @plate for k in eachindex(y)
            d = log1p(exp(z[g[k]]) * w[k])
            y[k] ~ Normal(d + cos(x[k]), exp(d) + s)
        end
    end
    one, two = build(selected), build(selected)
    @test any(occursin("##_rkppl_selected#", t) for t in operations(one.built.spec))
    @test operations(one.built.spec) == operations(two.built.spec)
    @test typeof(one.query) === typeof(two.query)
    u = [-0.2, 0.3, -0.4]
    z = u[2:3]
    lp = logpdf(Exponential(1.0), exp(u[1])) + u[1] + sum(logpdf.(Normal(0.0, 1.0), z))
    for k in eachindex(data.y)
        d = log1p(exp(z[data.g[k]]) * data.w[k])
        lp += logpdf(Normal(d + cos(data.x[k]), exp(d) + exp(u[1])), data.y[k])
    end
    @test Base.invokelatest(one.query, u) ≈ lp
    @test Base.invokelatest(two.query, u) == Base.invokelatest(one.query, u)
end

@testset "a submodel called twice keeps two sets of generated names" begin
    # A submodel lowers inside its caller and into one flat program, so what its
    # lowering names must differ between the calls (snag `gensym-in-kernel-54164b85`).
    mod = Module(gensym(:FreshBuild))
    Core.eval(mod, :(using ReactiveKernelsPPL, Distributions))
    Core.eval(mod, quote
        @rkppl term(v, rows) = begin
            b ~ Normal(0.0, 1.0)
            @plate for k in eachindex(rows)
                d = log1p(exp(b) * v[k]^2)
                rows[k] ~ Normal(d + cos(v[k]), exp(d) + 1.0)
            end
            return b
        end
    end)
    ast = quote
        first_term ~ term(x, y)
        second_term ~ term(w, r)
    end
    data = (; x = [-0.5, 0.5], w = [0.1, 0.4], y = [-1.0, 0.1], r = [1.0, 0.4])
    plan = bind_data(lower_rkppl(ast, keys(data); mod, conditioned = (:y, :r)), data)
    built = build_kernel(plan)
    selected = Set(m.match for r in built.spec.graph.recipes
                   for m in eachmatch(r"##_rkppl_selected#\d+", string(typeof(r.op))))
    @test length(selected) >= 2
    query = prepare_query(built, plan, :sampler)
    u = [0.2, -0.3]
    expected = sum(logpdf(Normal(0.0, 1.0), b) for b in u)
    for (b, v, o) in ((u[1], data.x, data.y), (u[2], data.w, data.r)), k in eachindex(o)
        d = log1p(exp(b) * v[k]^2)
        expected += logpdf(Normal(d + cos(v[k]), exp(d) + 1.0), o[k])
    end
    @test Base.invokelatest(query, u) ≈ expected
end

# A model call (`model(; data) | (; y)`) lowers through `_lower_model`, not
# `lower_rkppl`. What it names (a cell scan's seeds, a selected cell argument)
# follows the source as well, so one declaration bound at two sizes generates
# one `kernel_expr`, and two builds of one binding share their types
# (snag `kernel-expr-of-a-26ed1e5c`).
@testset "a model call lowers one source to one set of generated names" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        @plate for g in eachindex(y)
            @scan begin
                h[1] = x[1, g]
                for t in 2:H
                    h[t] = a * h[t - 1] + x[t, g]
                end
            end
            y[g] .~ Normal.(h[idx[g]], 1)
        end
    end
    bound(G, H) = (x = reshape(collect(range(-1.0, 1.0; length = H * G)), H, G);
        idx = [collect(1:2:H) for _ in 1:G];
        model(; x, H, idx) | (; y = [fill(0.1, length(i)) for i in idx]))
    small, large = bound(2, 5), bound(3, 7)
    small_built, large_built = build_kernel(small), build_kernel(large)
    source = kernel_expr(small, small_built.layout)
    generated = Set(m.match for m in eachmatch(r"##[A-Za-z_0-9]+#\d+", string(source)))
    @test any(n -> startswith(n, "##h_seed1#"), generated)
    @test any(n -> startswith(n, "##_rkppl_selected#"), generated)
    @test source == kernel_expr(large, large_built.layout)

    # A selected argument's name reaches the split nodes' types, so two builds
    # of one binding share them only if the model call names it by its source.
    selected = @rkppl begin
        z[levels(g)] .~ Normal.(0.0, 1.0)
        s ~ Exponential(1.0)
        @plate for k in eachindex(y)
            d = log1p(exp(z[g[k]]) * w[k])
            y[k] ~ Normal(d + cos(x[k]), exp(d) + s)
        end
    end
    data = (; x = [-0.5, 0.0, 0.5, 1.0], w = [0.1, 0.2, 0.3, 0.4], g = [1, 1, 2, 2])
    observed = (; y = [-1.0, 0.1, 1.0, 0.4])
    operations(spec) = Set(string(typeof(r.op)) for r in spec.graph.recipes)
    one = build_kernel(selected(; data...) | observed)
    two = build_kernel(selected(; data...) | observed)
    @test any(occursin("##_rkppl_selected#", t) for t in operations(one.spec))
    @test operations(one.spec) == operations(two.spec)
end

# A scan with two or more outputs (two carried arrays, or a carried array and a
# sampled innovation) reads each trajectory out of one shared recurrence through
# a column plate, `ReactiveKernels._scan_column_expr`. Its row is named by the
# source too, so binding one declaration again, at the same or at other sizes,
# spells one `kernel_expr` (snag `kernel-expr-scan-1a8411f4`).
@testset "a scan's trajectory columns are named by the source" begin
    model = @rkppl begin
        s ~ Exponential(1.0)
        step ~ Exponential(1.0)
        @plate for g in eachindex(idx)
            @scan begin
                a[1] = x[1, g]
                b[1] = 0.0
                for t in 2:H
                    a[t] = 0.5 * a[t - 1] + x[t, g]
                    b[t] = 0.9 * b[t - 1] + a[t - 1]
                end
            end
            @scan begin
                l[1] = 0.0
                for k in 2:K
                    e ~ Normal(0, 1)
                    l[k] = l[k - 1] + step * e
                end
            end
            y[g] .~ Normal.(a[idx[g]] .+ b[idx[g]] .+ l[day[g]], s)
        end
    end
    function bound(G, H)
        x = reshape(collect(range(-1.0, 1.0; length = H * G)), H, G)
        idx = [collect(g:H) for g in 1:G]
        day = [cld.(i, 3) for i in idx]
        model(; x, H, idx, day, K = cld(H, 3)) | (; y = [fill(0.1, length(i)) for i in idx])
    end
    source(b) = kernel_expr(b, build_kernel(b).layout)
    small = source(bound(2, 5))
    generated = Set(m.match for m in eachmatch(r"##[A-Za-z_0-9]+#\d+", string(small)))
    @test count(n -> startswith(n, "##scan_row#"), generated) >= 2
    @test small == source(bound(2, 5))
    @test small == source(bound(3, 7))
end
end
