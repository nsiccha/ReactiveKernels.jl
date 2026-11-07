using DifferentiationInterface
using Distributions
using Enzyme
using LogExpFunctions: logistic
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_bound_extent_nodes(ex) = ex isa Expr ?
    1 + sum(_bound_extent_nodes, ex.args; init = 0) : 0

@testset "declared arrays sized by a bound integer" begin
    ast = quote
        z[1:J] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[k]
        y .~ Normal.(mu, 1)
    end
    plan = lower_rkppl(ast, (:J, :k, :y); conditioned = (:y,))
    domain_plan = lower_rkppl(quote
        z[domain] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[k]
        y .~ Normal.(mu, 1)
    end, (:domain, :k, :y); conditioned = (:y,))
    sizes = Tuple{Int,Int}[]
    for J in (4, 40)
        k, y = [1, 3], [0.2, -0.3]
        bound = bind_data(plan, Dict{Symbol,Any}(:J => J, :k => k, :y => y))
        built = build_kernel(bound)
        @test bound.n_obs == length(y)
        @test built.layout.total == J + 1
        u = [0.3sin(i) for i in 1:built.layout.total]
        values = constrain(built.layout, u)
        want = sum(logpdf.(Normal(), values.z)) + logpdf(Normal(), values.a) +
            sum(logpdf.(Normal.(values.a .+ values.z[k], 1), y))
        @test _query(built.spec, bound, :posterior, u) ≈ want
        # A supplied 1:J domain is equivalent but adds an input to the model.
        domain = bind_data(domain_plan,
            Dict{Symbol,Any}(:domain => 1:J, :k => k, :y => y))
        domain_built = build_kernel(domain)
        @test domain_built.layout.total == built.layout.total
        @test _query(domain_built.spec, domain, :posterior, u) ≈ want
        push!(sizes, (_bound_extent_nodes(kernel_expr(bound, built.layout)),
            _bound_extent_nodes(code_expr(built.spec))))
        J == 4 && _check_gradient(built.spec, bound, u)
    end
    # The latent dimension changes the layout; it does not replicate work in
    # the RK kernel definition or planned program.
    @test length(unique(sizes)) == 1

    # Observed labels are not the declared support: 2 and 4 are unobserved.
    levels_plan = lower_rkppl(quote
        z[levels(k)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[k]
        y .~ Normal.(mu, 1)
    end, (:k, :y); conditioned = (:y,))
    levels_bound = bind_data(levels_plan,
        Dict{Symbol,Any}(:k => [1, 3], :y => [0.2, -0.3]))
    @test build_kernel(levels_bound).layout.total == 3
end

@testset "bound array extent expressions and validation" begin
    plan = lower_rkppl(quote
        K = J + 1
        z[1:K] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:J, :y); conditioned = (:y,))
    @test build_kernel(bind_data(plan,
        Dict{Symbol,Any}(:J => 3, :y => [0.2]))).layout.total == 5

    two_axes = lower_rkppl(quote
        B[1:(2 * J), 1:J] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:J, :y); conditioned = (:y,))
    @test build_kernel(bind_data(two_axes,
        Dict{Symbol,Any}(:J => 2, :y => [0.2]))).layout.total == 9

    level_expr = lower_rkppl(quote
        z[1:(length(levels(g)) + 1)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:g, :y); conditioned = (:y,))
    @test build_kernel(bind_data(level_expr,
        Dict{Symbol,Any}(:g => [1, 2], :y => [0.2]))).layout.total == 4

    direct = lower_rkppl(quote
        z[1:J] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:J, :y); conditioned = (:y,))
    @test build_kernel(bind_data(direct,
        Dict{Symbol,Any}(:J => 0, :y => [0.2]))).layout.total == 1
    for bad in (-1, 1.5, true)
        # A declared extent is an ordinary nonnegative integer size.
        @test_throws ContractValidationError bind_data(direct,
            Dict{Symbol,Any}(:J => bad, :y => [0.2]))
    end
    @test_throws ContractValidationError bind_data(direct,
        Dict{Symbol,Any}(:y => [0.2]))

    live = lower_rkppl(quote
        a ~ Normal(0, 1)
        z[1:a] .~ Normal.(0, 1)
        y .~ Normal.(a, 1)
    end, (:y,); conditioned = (:y,))
    # A sampled value cannot set the structural coordinate pack at bind.
    @test_throws ContractValidationError bind_data(live,
        Dict{Symbol,Any}(:y => [0.2]))
end

# Sizes and data-derived numbers are runtime model inputs (user principle
# `ODLPlaus:precompile/decisions/2026-11-06T01-09-46-212-0ikeidi`; snag
# `rkppl-sized-vect-69604d32`): every extent of one declaration, and every value
# of a data-derived transform bound, generates the same program and the same
# prepared query type, and one build serves every binding.
_extent_central(f, u; h = 1e-6) =
    [(f(u .+ h .* (eachindex(u) .== i)) - f(u .- h .* (eachindex(u) .== i))) / 2h for i in eachindex(u)]

# Each binding's density from the host's constrained values and log-Jacobian,
# which do not read the generated program: `(constrained, data) -> density`
# without the Jacobian.
function _extent_check(source, datasets, want; conditioned = (:y,), same_type = true)
    plan = lower_rkppl(source, first(datasets); conditioned)
    bound = [bind_data(plan, d) for d in datasets]
    builds = build_kernel.(bound)
    programs = [Base.remove_linenums!(kernel_expr(b, built.layout))
        for (b, built) in zip(bound, builds)]
    @test all(==(first(programs)), programs)
    queries = [prepare_query(built, b, :sampler) for (b, built) in zip(bound, builds)]
    same_type && @test allequal(typeof.(queries))
    for (b, built, q, d) in zip(bound, builds, queries, datasets)
        u = [0.3sin(i) for i in 1:built.layout.total]
        expected = want(constrain(built.layout, u), d) + logjac(built.layout, u)
        @test Base.invokelatest(q, u) ≈ expected
        # The first binding's build serves this one, derivative included.
        reused = prepare_query(first(builds), b, :sampler)
        @test Base.invokelatest(reused, u) ≈ expected
        isempty(u) && continue
        ad = prepare_ad(reused, AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
        value, gradient = Base.invokelatest(ad_value_and_gradient, ad, u)
        @test value ≈ expected
        @test gradient ≈ _extent_central(x -> Base.invokelatest(reused, x), u) atol = 1e-5
        # So does its sampler, which packs this binding's coordinates.
        sampler = prepare_sampler(first(builds), b, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        sampled, sampled_gradient = sampler_value_and_gradient!(sampler, zeros(length(u)), u)
        @test sampled ≈ expected
        @test sampled_gradient ≈ gradient
    end
    return programs
end

_normals(x) = sum(logpdf.(Normal(), x); init = 0.0)
# A value the model computes from data, as BRM's field support is.
_extent_support(n) = (geometry = (n = n,),)

@testset "one program for every extent of every packed block" begin
    @testset "one-axis arrays, the empty extent, and blocks packed after them" begin
        programs = _extent_check(quote
                z[1:n] .~ LogNormal.(mu, sd)
                M[1:2, 1:3] .~ Normal.(0, 1)
                w[1:3] .~ Normal.(0, 1)
                y .~ Normal.(z, obs_sd)
            end,
            [(n = n, mu = -0.2, sd = 0.3, obs_sd = 0.7, y = [0.9 + 0.1i for i in 1:n]) for n in (2, 4)],
            (v, d) -> sum(logpdf.(LogNormal(d.mu, d.sd), v.z)) +
                _normals(v.M) + _normals(v.w) + sum(logpdf.(Normal.(v.z, d.obs_sd), d.y)))
        # The program reads the extent itself. `M` and `w` keep their literal
        # shapes and start where the data-sized block ends.
        text = string(first(programs))
        @test occursin("view(unconstrained, 1:n)", text)
        @test occursin("_ppl_first_M::Int = n + 1", text)
        @test occursin("reshape(_ppl_arrflat_M, 2, 3)", text)
        # An empty array keeps the program. Its prepared query does not: RK
        # splits a data-bound branch of the likelihood by its lanes, and zero
        # lanes keep the guarded branch.
        _extent_check(quote
                z[1:n] .~ LogNormal.(mu, sd)
                w[1:2] .~ Normal.(0, 1)
                y .~ Normal.(w[1], 1.0)
            end,
            [(n = n, mu = 0.1, sd = 0.3, y = [0.9, 1.0]) for n in (2, 0)],
            (v, d) -> sum(logpdf.(LogNormal(d.mu, d.sd), v.z); init = 0.0) + _normals(v.w) +
                sum(logpdf.(Normal.(v.w[1], 1.0), d.y)); same_type = false)
    end
    @testset "two-axis arrays with both axes data" begin
        _extent_check(quote
                Z[1:r, 1:k] .~ Normal.(0, 1)
                t[1:2] .~ Normal.(0, 1)
                y .~ Normal.(Z[ids, 1] .+ t[1], 1.0)
            end,
            [(r = 2, k = 8, ids = [1, 2, 1], y = [0.9, 1.0, 1.1]),
                (r = 3, k = 24, ids = [1, 3, 2], y = [0.9, 1.0, 1.1])],
            (v, d) -> _normals(v.Z) + _normals(v.t) +
                sum(logpdf.(Normal.(v.Z[d.ids, 1] .+ v.t[1], 1.0), d.y)))
        _extent_check(quote
                Z[1:r, 1:k] .~ Exponential.(1.0)
                y .~ Normal.(Z[ids, 1], 1.0)
            end,
            [(r = 2, k = 3, ids = [1, 2, 1], y = [0.9, 1.0, 1.1]),
                (r = 4, k = 2, ids = [1, 3, 2], y = [0.9, 1.0, 1.1])],
            (v, d) -> sum(logpdf.(Exponential(1.0), v.Z)) +
                sum(logpdf.(Normal.(v.Z[d.ids, 1], 1.0), d.y)))
        _extent_check(quote
                eachrow(P[1:g, 1:3]) .~ Dirichlet(ones(3))
                w[1:2] .~ Normal.(0, 1)
                y .~ Normal.(P[ids, 1] .+ w[1], 1.0)
            end,
            [(g = 2, ids = [1, 2], y = [0.9, 1.0]), (g = 4, ids = [1, 4], y = [0.9, 1.0])],
            (v, d) -> sum(logpdf(Dirichlet(ones(3)), Vector(r)) for r in eachrow(v.P)) +
                _normals(v.w) + sum(logpdf.(Normal.(v.P[d.ids, 1] .+ v.w[1], 1.0), d.y)))
        # A selection of an array's axes: the levels are the binding's, and other
        # labels keep the program and its build (`rkppl-use` section 14).
        plan = lower_rkppl(quote
                b[levels(g)] .~ Normal.(0, 1)
                y .~ Normal.(b[g], 1.0)
            end, (g = ["a", "b", "a"], y = [0.9, 1.0, 1.1]); conditioned = (:y,))
        level_bound = [bind_data(plan, (g = g, y = [0.9, 1.0, 1.1]))
            for g in (["a", "b", "a"], ["a", "b", "c"])]
        level_builds = build_kernel.(level_bound)
        @test ==(map((b, l) -> Base.remove_linenums!(kernel_expr(b, l.layout)), level_bound, level_builds)...)
        @test ==(map((b, l) -> typeof(prepare_query(l, b, :sampler)), level_bound, level_builds)...)
    end
    @testset "an extent read through a property of bound data" begin
        # `support.geometry.n` is a property of a value, as the model reads it,
        # not a module path (snag `gmrf-field-packe-ab48cb09`).
        programs = _extent_check(quote
                support = _extent_support(N)
                z[1:support.geometry.n] .~ Normal.(0, 1)
                y .~ Normal.(z[k], 1.0)
            end,
            [(N = n, k = [1, n], y = [0.9, 1.0]) for n in (2, 5)],
            (v, d) -> _normals(v.z) + sum(logpdf.(Normal.(v.z[d.k], 1.0), d.y)))
        @test occursin("Base.getproperty(Base.getproperty(support, :geometry), :n)",
            string(first(programs)))
    end
    @testset "plate latents, the empty loop included" begin
        loop = quote
            @plate for i in eachindex(x)
                a[i] ~ Normal(x[i], 1)
            end
            w[1:2] .~ Normal.(0, 1)
            y .~ Normal.(w[1], 1.0)
        end
        want = (v, d) -> sum(logpdf.(Normal.(d.x, 1), v.a); init = 0.0) + _normals(v.w) +
            sum(logpdf.(Normal.(v.w[1], 1.0), d.y))
        _extent_check(loop, [(x = [0.9, 1.0], y = [0.5]), (x = [0.9, 1.0, 1.1, 0.7], y = [0.5])], want)
        _extent_check(loop, [(x = [0.9, 1.0], y = [0.5]), (x = Float64[], y = [0.5])], want;
            same_type = false)
        # A latent over an observation's rows stays literal: the observation can
        # be absent, and a design binding of a fit's build generates this program.
        over_response = lower_rkppl(quote
                @plate for i in eachindex(y)
                    a[i] ~ Normal(0, 1)
                    y[i] ~ Normal(a[i], 1.0)
                end
            end, (y = [0.9, 1.0],); conditioned = (:y,))
        fit = bind_data(over_response, (y = [0.9, 1.0],))
        @test occursin("view(unconstrained, 1:2)",
            string(kernel_expr(fit, build_kernel(fit).layout)))
    end
    @testset "scans, centered and non-centered" begin
        _extent_check(quote
                @scan begin
                    s[1] ~ Normal(0, 1)
                    for t in 2:T
                        s[t] ~ Normal(s[t - 1], 1.0)
                    end
                end
                w[1:2] .~ Normal.(0, 1)
                y .~ Normal.(s, 1.0)
            end,
            [(T = 2, y = [0.9, 1.0]), (T = 4, y = [0.9, 1.0, 1.1, 0.7])],
            (v, d) -> logpdf(Normal(), v.s[1]) +
                sum(logpdf.(Normal.(v.s[1:(end - 1)], 1.0), v.s[2:end])) +
                _normals(v.w) + sum(logpdf.(Normal.(v.s, 1.0), d.y)))
        _extent_check(quote
                @scan begin
                    s[1] ~ Normal(0, 1)
                    for t in 2:T
                        e ~ Normal(0, 1)
                        s[t] = s[t - 1] + 0.5 * e
                    end
                end
                w[1:2] .~ Normal.(0, 1)
                y .~ Normal.(s, 1.0)
            end,
            [(T = 2, y = [0.9, 1.0]), (T = 4, y = [0.9, 1.0, 1.1, 0.7])],
            (v, d) -> begin
                # The host returns the sampled seed and innovations.
                z = v._ppl_scan_z_s
                _normals(z) + _normals(v.w) +
                    sum(logpdf.(Normal.(z[1] .+ 0.5 .* [0.0; cumsum(z[2:end])], 1.0), d.y))
            end)
    end
    @testset "ordered thresholds and their level count" begin
        logit_mass(c, k) = k == 1 ? log(logistic(c[1])) :
            k == length(c) + 1 ? log1p(-logistic(c[end])) :
            log(logistic(c[k]) - logistic(c[k - 1]))
        ordered = quote
            c ~ Ordered(Normal(0, 1), K - 1)
            k .~ OrderedLogistic.(0.0, Ref(c))
        end
        want = (v, d) -> _normals(v.c) + sum(logit_mass.(Ref(v.c), d.k))
        _extent_check(ordered, [(K = 3, k = [1, 2, 3]), (K = 4, k = [1, 2, 3, 4])], want;
            conditioned = (:k,))
        # Two levels have no interior observation. RK splits the data-bound
        # branch by the arms its lanes take, so that query's type is its own.
        _extent_check(ordered, [(K = 2, k = [1, 2, 1]), (K = 3, k = [1, 2, 3])], want;
            conditioned = (:k,), same_type = false)
        # A level count inferred from the response stays as declared: observed
        # levels need not be all of them, and the response can be absent.
        inferred = lower_rkppl(quote
                c ~ Ordered(Normal(0, 1), length(levels(k)) - 1)
                k .~ OrderedLogistic.(0.0, Ref(c))
            end, (k = [1, 2, 3],); conditioned = (:k,))
        fit = bind_data(inferred, (k = [1, 2, 3],))
        @test occursin("view(unconstrained, 1:2)",
            string(kernel_expr(fit, build_kernel(fit).layout)))
    end
    @testset "a bound name sizes ordered and stopping thresholds" begin
        # BRM binds a threshold count beside the data, as it binds an array
        # axis or an LKJ dimension (snag `rkppl-threshold-932a2e27`). The bare
        # name is that count; the program reads it like `K - 1` above.
        cumulative_lp(c, eta, k) = k == 1 ? log(logistic(c[1] - eta)) :
            k == length(c) + 1 ? log1p(-logistic(c[end] - eta)) :
            log(logistic(c[k] - eta) - logistic(c[k - 1] - eta))
        stopping_lp(t, eta, k) =
            sum((log1p(-logistic(t[j] - eta)) for j in 1:k-1); init = 0.0) +
            (k == length(t) + 1 ? 0.0 : log(logistic(t[k] - eta)))
        data = [(n_cuts = 2, x = [0.4, -0.3, 0.9], k = [1, 2, 3]),
            (n_cuts = 3, x = [0.4, -0.3, 0.9, 0.1], k = [1, 2, 3, 4])]
        want(lp, name) = (v, d) -> _normals(v.b) + _normals(v[name]) +
            sum(lp.(Ref(v[name]), v.b .* d.x, d.k))
        for (source, lp, name) in (
                (quote
                    b ~ Normal(0, 1)
                    c ~ Ordered(Normal(0, 1), n_cuts)
                    k .~ OrderedLogistic.(b .* x, Ref(c))
                end, cumulative_lp, :c),
                (quote
                    b ~ Normal(0, 1)
                    c ~ Ordered(Normal(0, 1), n_cuts)
                    k .~ Ordinal.(Cumulative(), LogitLink(), b .* x, Ref(c))
                end, cumulative_lp, :c),
                (quote
                    b ~ Normal(0, 1)
                    t[1:n_cuts] .~ Normal.(0, 1)
                    k .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(t))
                end, stopping_lp, :t))
            programs = _extent_check(source, data, want(lp, name); conditioned = (:k,))
            @test occursin("n_cuts", string(first(programs)))
            plan = lower_rkppl(source, first(data); conditioned = (:k,))
            for bad in (1.5, -1, true)
                @test_throws "length must be a nonnegative integer" bind_data(plan,
                    merge(first(data), (n_cuts = bad,)))
            end
        end
        # A sampled value cannot set the structural coordinate pack at bind.
        live = lower_rkppl(quote
                a ~ Normal(0, 1)
                c ~ Ordered(Normal(0, 1), a)
                k .~ OrderedLogistic.(0.0, Ref(c))
            end, (k = [1, 2, 3],); conditioned = (:k,))
        @test_throws "needs data-only value a" bind_data(live, (k = [1, 2, 3],))
    end
    @testset "data-derived transform bounds are inputs, not numbers" begin
        programs = _extent_check(quote
                floor_ = minimum(x) / 2
                s ~ truncated(Normal(0, 1), floor_, Inf)
                y .~ Normal.(s, 1.0)
            end,
            [(x = [1.0, 2.0], y = [0.9, 1.0]), (x = [3.0, 2.5], y = [0.9, 1.0])],
            (v, d) -> logpdf(truncated(Normal(0, 1), minimum(d.x) / 2, Inf), v.s) +
                sum(logpdf.(Normal.(v.s, 1.0), d.y)))
        @test !occursin("1.25", string(first(programs)))
        # `restricted` keeps the prior's unnormalized density over the same
        # data-derived bound.
        programs = _extent_check(quote
                floor_ = minimum(x) / 2
                s ~ restricted(Normal(0, 1), floor_, Inf)
                y .~ Normal.(s, 1.0)
            end,
            [(x = [1.0, 2.0], y = [0.9, 1.0]), (x = [3.0, 2.5], y = [0.9, 1.0])],
            (v, d) -> logpdf(Normal(0, 1), v.s) + sum(logpdf.(Normal.(v.s, 1.0), d.y)))
        @test !occursin("1.25", string(first(programs)))
        _extent_check(quote
                s ~ Uniform(a, b)
                y .~ Normal.(s, 1.0)
            end,
            [(a = -1.0, b = 2.0, y = [0.9, 1.0]), (a = -3.0, b = 5.0, y = [0.9, 1.0])],
            (v, d) -> logpdf(Uniform(d.a, d.b), v.s) + sum(logpdf.(Normal.(v.s, 1.0), d.y)))
        _extent_check(quote
                z[1:n] .~ Normal.(0, 1)
                s[1:m] .~ Uniform.(lo, hi)
                y .~ Normal.(s[ids], 1.0)
            end,
            [(n = 2, m = 2, lo = 0.0, hi = 2.0, ids = [1, 2], y = [0.9, 1.0]),
                (n = 5, m = 3, lo = 0.5, hi = 2.5, ids = [1, 3], y = [0.9, 1.0])],
            (v, d) -> _normals(v.z) + sum(logpdf.(Uniform(d.lo, d.hi), v.s)) +
                sum(logpdf.(Normal.(v.s[d.ids], 1.0), d.y)))
        # A literal bound stays literal: it is part of the authored source.
        literal = bind_data(lower_rkppl(quote
                s ~ truncated(Normal(0, 1), 0.25, Inf)
                y .~ Normal.(s, 1.0)
            end, (y = [0.9, 1.0],); conditioned = (:y,)), (y = [0.9, 1.0],))
        @test occursin("0.25", string(kernel_expr(literal, build_kernel(literal).layout)))
    end
    @testset "literal extents stay literal" begin
        # A model whose positions depend on no data is the program it always was.
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                w[1:2] .~ Normal.(0, 1)
                M[1:2, 1:3] .~ Normal.(0, 1)
                y .~ Normal.(a .+ w[1], 1.0)
            end, (y = [0.9, 1.0],); conditioned = (:y,))
        bound = bind_data(plan, (y = [0.9, 1.0],))
        text = string(kernel_expr(bound, build_kernel(bound).layout))
        @test occursin("view(unconstrained, 2:3)", text)
        @test occursin("view(unconstrained, 4:9)", text)
        @test occursin("reshape(_ppl_arrflat_M, 2, 3)", text)
        @test !occursin("_ppl_first_", text) && !occursin("_ppl_last_", text)
    end
    @testset "a cut needs only the model's own data" begin
        # Positions read the data the model already has, so they add no input
        # for `have` or `bound` to name (`rkppl-use` section 14).
        data = (n = 3, m = 2, y = [0.9, 1.0])
        plan = lower_rkppl(quote
                z[1:n] .~ Normal.(0, 1)
                w[1:m] .~ Normal.(0, 1)
                mu = w[1] .+ z[1]
                y .~ Normal.(mu, 1.0)
            end, data; conditioned = (:y,))
        bound = bind_data(plan, data)
        built = build_kernel(bound)
        u = [0.3sin(i) for i in 1:built.layout.total]
        v = constrain(built.layout, u)
        cut = Base.invokelatest(ReactiveKernels.prepare, built.spec;
            have = (:unconstrained, keys(data)...), want = :w, bound = data)
        @test Base.invokelatest(cut, u) == v.w
        # Without `bound=`, the same cut reads the extents at run time.
        live = Base.invokelatest(ReactiveKernels.prepare, built.spec;
            have = (:unconstrained, :n, :m), want = :w)
        @test Base.invokelatest(live, u, 3, 2) == v.w
    end
    @testset "a cell reads its own coordinates through data-derived positions" begin
        ids = [2, 1, 2, 3]
        data(n, k) = (n = n, k = k, ids = ids, x = [0.2, 0.3, -0.4, 0.0],
            y = [0.1, 0.7, -0.2, 0.5])
        plan = lower_rkppl(quote
                a[1:n] .~ Normal.(0, 1)
                Z[1:n, 1:k] .~ Normal.(0, 1)
                mu = Z[ids, 1] .* x .+ a[ids]
                y .~ Normal.(mu, 1.0)
            end, data(3, 2); conditioned = (:y,))
        for (n, k) in ((3, 2), (4, 3))
            bound = bind_data(plan, data(n, k))
            built = build_kernel(bound)
            q = prepare_cell_query(built, bound, :y; locality = :error)
            u = [0.3sin(i) for i in 1:built.layout.total]
            v = constrain(built.layout, u)
            mu = v.Z[ids, 1] .* data(n, k).x .+ v.a[ids]
            for i in eachindex(ids)
                @test Base.invokelatest(q, u, i) ≈ logpdf(Normal(mu[i], 1.0), data(n, k).y[i])
            end
            # Cell `i` reads `a[ids[i]]` and row `ids[i]` of `Z`, packed after `a`:
            # a position for each column, whatever the extents, and no other row.
            reads = Base.invokelatest(partition_footprint, q.kernel, u, 2).unconstrained
            @test sort!(collect(reads)) == [ids[2]; [n + ids[2] + n * (j - 1) for j in 1:k]]
        end
    end
    @testset "a partition footprint reads this binding's packed positions" begin
        # Snag `partition-plan-n-2b8a0d4a`: `W` and `b` start where the level-sized
        # `Z` ends, a position read from the binding. The footprint kernel spliced
        # those positions into its body, so every binding with another level count
        # compiled the footprint again (10-17 s per ODL binding).
        plan = lower_rkppl(quote
                sigma ~ Exponential(1)
                Z[levels(group), 1:2] .~ Normal.(0, 1)
                W[levels(group)] .~ Normal.(0, 1)
                b ~ Normal(0, 1)
                mu = b .+ Z[group, 1] .+ x .* Z[group, 2] .+ W[group]
                y .~ Normal.(mu, sigma)
            end, (:group, :subject, :x, :y); conditioned = (:y,))
        footprint_functions = map(((2, 3, 4, 5), (3, 1, 2, 2, 7, 1))) do sizes
            G = length(sizes)
            group = [g for (g, n) in enumerate(sizes) for _ in 1:n]
            bound = bind_data(plan, (; group, subject = cld.(group, 2),
                x = sin.(eachindex(group)), y = cos.(eachindex(group))))
            built = build_kernel(bound)
            q = prepare_partition(built, bound, :y; by = :subject,
                partitioned = (:Z => [cld(g, 2) for g in 1:G],))
            u = [0.3sin(i) for i in 1:built.layout.total]
            fp = Base.invokelatest(ReactiveKernels.partition_footprint_kernel, q.kernel)
            # The same lowering with every coordinate selected: positions this
            # binding's lowering computes, as before the fix.
            everything = :unconstrained =>
                ReactiveKernels._FootprintSelection(trues(built.layout.total))
            literal = Base.invokelatest(ReactiveKernels._partition_footprint_kernel,
                q.kernel, everything)
            first_of(name) = only(e.offset for e in built.layout.entries if e.name === name)
            Zpos(g) = [first_of(:Z) - 1 + g + G * (j - 1) for j in 1:2]
            for label in partition_labels(q)
                reads = Base.invokelatest(fp, u, label, nothing).unconstrained
                @test reads == Base.invokelatest(literal, u, label, nothing).unconstrained
                @test first_of(:sigma) in reads && first_of(:b) in reads
                @test first_of(:W) + G - 1 in reads
                @test all(g -> Zpos(g) ⊆ reads, filter(g -> cld(g, 2) == label, 1:G))
                @test all(g -> isdisjoint(Zpos(g), reads), filter(g -> cld(g, 2) != label, 1:G))
            end
            fp.f
        end
        # One footprint body for both level counts: its type is its generated code.
        @test ==(typeof.(footprint_functions)...)
    end
    @testset "one retained build serves larger, smaller and empty bindings in turn" begin
        # The reporter's sequence (`ODLPlaus:precompile`, RK `bcb8134e`).
        data(n) = (n = n, prior_mu = -0.2, prior_sd = 0.3, observation_sd = 0.7,
            x = [0.5 + 0.1i for i in 1:n], y = [0.9 + 0.1i for i in 1:n])
        plan = lower_rkppl(quote
                z[1:n] .~ LogNormal.(prior_mu, prior_sd)
                y .~ Normal.(x .* z, observation_sd)
            end, data(2); conditioned = (:y,))
        built = build_kernel(bind_data(plan, data(2)))
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        for n in (2, 4, 2, 0, 2)
            d = data(n)
            bound = bind_data(plan, d)
            u = [0.3sin(i) for i in 1:n]
            @test assign_layout(bound).total == n
            q = prepare_query(built, bound, :sampler)
            z = exp.(u)
            want = sum(logpdf.(LogNormal(d.prior_mu, d.prior_sd), z); init = 0.0) + sum(u; init = 0.0) +
                sum(logpdf.(Normal.(d.x .* z, d.observation_sd), d.y); init = 0.0)
            @test Base.invokelatest(q, u) ≈ want
            sampler = prepare_sampler(built, bound, u; backend)
            @test sampler.layout.total == n
            value, gradient = sampler_value_and_gradient!(sampler, zeros(n), u)
            @test value ≈ want
            @test coordinate_names(sampler.layout) == [Symbol("z.", i) for i in 1:n]
            @test length(gradient) == n
            n == 0 && continue
            @test gradient ≈ _extent_central(x -> Base.invokelatest(q, x), u) atol = 1e-5
            # A cell query reads this binding's rows and coordinates too.
            cells = prepare_cell_query(built, bound, :y; locality = :error)
            for i in 1:n
                @test Base.invokelatest(cells, u, i) ≈ logpdf(Normal(d.x[i] * z[i], d.observation_sd), d.y[i])
            end
        end
        # A wrong-length point still names the binding's own size.
        @test_throws "exemplar length 3 ≠ layout total 4" prepare_sampler(built,
            bind_data(plan, data(4)), zeros(3); backend)
    end
    @testset "a rebinding reads its own extents" begin
        plan = lower_rkppl(quote
                z[1:n] .~ Normal.(0, 1)
                y .~ Normal.(z[1], 1.0)
            end, (n = 2, y = [0.9]); conditioned = (:y,))
        bound = bind_data(plan, (n = 4, y = [0.9]))
        @test build_kernel(bind_data(bound, (n = 2, y = [0.9]))).layout.total == 2
    end
end

# A level axis's labels are data: another number of them, or other labels, keeps
# the program, its type and its build (user question on station counts, and
# decision `ReactiveKernels:snag.rkppl-sized-vect-69604d32/decisions/2026-11-06T18-01-39-643-18hmyfb`).
_level_code(g, labels) = [findfirst(==(x), labels) for x in g]

@testset "one build serves other level labels" begin
    @testset "a gather over levels, by count, label and row order" begin
        source = quote
            a ~ Normal(0, 1)
            c[levels(g)] .~ Normal.(0, 1)
            y .~ Normal.(a .+ c[g], 1.0)
        end
        want = (v, d) -> logpdf(Normal(), v.a) + _normals(v.c) +
            sum(logpdf.(Normal.(v.a .+ v.c[_level_code(d.g, sort(unique(d.g)))], 1.0), d.y))
        y(n) = [0.2sin(i) for i in 1:n]
        _extent_check(source, [
            (g = ["s1", "s2", "s3", "s1"], y = y(4)),
            (g = ["s1", "s2", "s3", "s4", "s5", "s1"], y = y(6)),
            (g = ["a", "b", "c", "a"], y = y(4)),
            (g = ["c", "b", "a"], y = y(3)),
            (g = ["s1", "s2"], y = y(2))], want)
        # Numbers as labels: 2, 3, 4 is not 1, 2, 3.
        _extent_check(source, [(g = [1, 2, 3, 1], y = y(4)), (g = [2, 3, 4, 2], y = y(4))], want)
    end
    @testset "first-occurrence labels and a selection of the levels" begin
        first_seen = quote
            c[unique(g)] .~ Normal.(0, 1)
            y .~ Normal.(c[g], 1.0)
        end
        _extent_check(first_seen, [
            (g = ["b", "a", "b"], y = [0.1, 0.2, 0.3]),
            (g = ["z", "x", "y", "z"], y = [0.1, 0.2, 0.3, 0.4])],
            (v, d) -> _normals(v.c) +
                sum(logpdf.(Normal.(v.c[_level_code(d.g, unique(d.g))], 1.0), d.y)))
        selected = quote
            c[levels(g)[2:end]] .~ Normal.(0, 1)
            y .~ Normal.(1.0, 1.0)
        end
        _extent_check(selected, [
            (g = ["a", "b", "c"], y = [0.1, 0.2]),
            (g = ["p", "q", "r", "s", "t"], y = [0.1, 0.2])],
            (v, d) -> _normals(v.c) + sum(logpdf.(Normal(1.0, 1.0), d.y)))
    end
    @testset "a data vector as an axis, and a plate over levels" begin
        over_values = quote
            c[ids] .~ Normal.(0, 1)
            y .~ Normal.(sum(c), 1.0)
        end
        _extent_check(over_values, [
            (ids = [10, 20], y = [0.1, 0.2]),
            (ids = [30, 40, 50, 60], y = [0.1, 0.2])],
            (v, d) -> _normals(v.c) + sum(logpdf.(Normal(sum(v.c), 1.0), d.y)))
        plate = quote
            @plate for k in levels(g)
                c[k] ~ Normal(0, 1)
            end
            y .~ Normal.(c[g], 1.0)
        end
        _extent_check(plate, [
            (g = ["a", "b", "a"], y = [0.1, 0.2, 0.3]),
            (g = ["p", "q", "r", "s"], y = [0.1, 0.2, 0.3, 0.4])],
            (v, d) -> _normals(v.c) +
                sum(logpdf.(Normal.(v.c[_level_code(d.g, sort(unique(d.g)))], 1.0), d.y)))
    end
    @testset "a reused build names this binding's coordinates" begin
        # A declared array's coordinates are positions (`c.1`), whatever the labels.
        source = quote
            c[levels(g)] .~ Normal.(0, 1)
            y .~ Normal.(c[g], 1.0)
        end
        plan = lower_rkppl(source, (g = ["a", "b"], y = [0.1, 0.2]); conditioned = (:y,))
        fit, other = bind_data(plan, (g = ["a", "b"], y = [0.1, 0.2])),
            bind_data(plan, (g = ["p", "q"], y = [0.1, 0.2]))
        built = build_kernel(fit)
        @test coordinate_names(ReactiveKernelsPPL._binding_build(built, other).layout) ==
            coordinate_names(build_kernel(other).layout) == [Symbol("c.1"), Symbol("c.2")]
        @test ReactiveKernelsPPL._binding_build(built, fit) === built
        # A factor term's coefficients are named by their levels (`mu.g_b`), so
        # a reused build takes this binding's names, and its values are this
        # binding's own build's.
        y4 = [0.1, 0.2, 0.3, 0.4]
        factor = StructuralPlan(
            LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma,
                nothing, ResponseEvidence(:none, nothing, nothing), :y_resp)],
            PredictorSpec[PredictorSpec(:mu, IdentityLink,
                TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                        :intercept),
                    TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)], :mu)],
            PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
                PopulationPrior(:mu, :g, 0.0, 0.5)],
            SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
                nothing, :sigma)],
            AssignmentSpec[], Dict{Symbol,AbstractVector}(:y => y4,
                :g => ["a", "b", "c", "a"]), 4;
            levelmaps = LevelMap[LevelMap(:mu, :g, ["b", "c"], :levels, (2, :end))])
        fbuilt = build_kernel(factor)
        relabeled = bind_data(factor, Dict{Symbol,AbstractVector}(:y => y4,
            :g => ["p", "q", "r", "p"]))
        fown = build_kernel(relabeled)
        reused = ReactiveKernelsPPL._binding_build(fbuilt, relabeled)
        @test coordinate_names(fbuilt.layout) ==
            [Symbol("mu.Intercept"), Symbol("mu.g_b"), Symbol("mu.g_c"), :sigma]
        @test coordinate_names(reused.layout) == coordinate_names(fown.layout) ==
            [Symbol("mu.Intercept"), Symbol("mu.g_q"), Symbol("mu.g_r"), :sigma]
        u = [0.3, 0.1, -0.2, 0.0]
        @test Base.invokelatest(prepare_query(fbuilt, relabeled, :sampler), u) ≈
            Base.invokelatest(prepare_query(fown, relabeled, :sampler), u)
    end
    @testset "a cell and a partition read this binding's labels" begin
        source = quote
            c[levels(g)] .~ Normal.(0, 1)
            @plate for i in eachindex(y)
                y[i] ~ Normal(c[g[i]], 1.0)
            end
        end
        fit_data = (g = ["a", "b", "a"], y = [0.1, 0.2, 0.3])
        other_data = (g = ["p", "q", "r", "p"], y = [0.1, 0.2, 0.3, 0.4])
        plan = lower_rkppl(source, fit_data; conditioned = (:y,))
        fit, other = bind_data(plan, fit_data), bind_data(plan, other_data)
        built, own = build_kernel(fit), build_kernel(other)
        u = [0.3, -0.2, 0.5]
        reused_cell = prepare_cell_query(built, other, :y; locality = :error, u0 = u)
        own_cell = prepare_cell_query(own, other, :y; locality = :error, u0 = u)
        @test [reused_cell(u, i) for i in 1:4] ≈ [own_cell(u, i) for i in 1:4]
        by = (; y = other_data.g)
        @test partition_labels(prepare_partition(built, other, :y; by)) ==
            partition_labels(prepare_partition(own, other, :y; by))
    end
end
