using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test, LinearAlgebra
using DifferentiationInterface: AutoEnzyme, SecondOrder

const _CF_INLINE = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    sigma ~ Exponential(1)
    eta = b .* x .+ z[g]
    @plate for k in eachindex(y)
        y[k] ~ Normal(log1p(sum(exp.(eta[r1[k]]))) + sum(eta[r2[k]] .^ 2), sigma)
    end
end

const _CF_SINGLE = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    sigma ~ Exponential(1)
    eta = b .+ z[g]
    @plate for k in eachindex(y)
        y[k] ~ Normal(sum(exp.(eta[rows[k]])), sigma)
    end
end

const _CF_DIRECT = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    scale ~ Exponential(1)
    sigma ~ Exponential(1)
    eta = b .+ scale .* z[g]
    @plate for k in eachindex(y)
        y[k] ~ Normal(sum(exp.(eta[rows[k]])) + eta[anchor[k]], sigma)
    end
end

const _CF_DIRECT_MIXED = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    scale ~ Exponential(1)
    sigma ~ Exponential(1)
    eta = b .+ scale .* z[g]
    @plate for k in eachindex(y)
        s1 = sum(exp.(eta[rows[k]]))
        s2 = sum(eta[rows2[k]] .^ 2)
        mean[k] = log1p(s1) + s2 + eta[anchor[k]]^2 + getindex(eta, other[k]) + eta[1]
    end
    y .~ Normal.(mean, sigma)
end

# Both the packed point and the reduction grow. A warmed factor call must
# neither traverse all terms nor allocate a shadow of the whole packed point.
function _cf_cost(call)
    call(); call(); call()
    bytes = minimum(@allocated(call()) for _ in 1:3)
    ns = minimum(1:5) do _
        start = time_ns()
        for _ in 1:50
            call()
        end
        (time_ns() - start) / 50
    end
    (; bytes, ns)
end

_cf_nodes(x) = x isa Expr ? 1 + sum(_cf_nodes, x.args; init = 0) : 1

const _CF_LOCAL = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    sigma ~ Exponential(1)
    eta = b .* x .+ z[g]
    @plate for k in eachindex(y)
        terms = exp.(eta[r1[k]])
        s1 = sum(terms)
        s2 = sum(eta[r2[k]] .^ 2)
        y[k] ~ Normal(log1p(s1) + s2, sigma)
    end
end

const _CF_EARLIER = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    sigma ~ Exponential(1)
    eta = b .* x .+ z[g]
    weights = exp.(eta)
    @plate for k in eachindex(y)
        s1 = sum(weights[r1[k]])
        s2 = sum(eta[r2[k]] .^ 2)
        mean[k] = log1p(s1) + s2
    end
    y .~ Normal.(mean, sigma)
end

const _CF_ROW_SCALE = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    sigma ~ Exponential(1)
    eta = b .* x .+ z[g]
    @plate for k in eachindex(y)
        mean[k] = log1p(sum(exp.(eta[r1[k]]))) + sum(eta[r2[k]] .^ 2)
    end
    row_scales = sigma .* (1 .+ abs.(y))
    y .~ Normal.(mean, row_scales)
end

_cf_data(n = 6) = (; g = collect(1:n), x = sin.(collect(1:n)),
    r1 = [[1, 2, 2], [3], Int[]], r2 = [[2, 3], [1, 4, 6], Int[]],
    y = [1.2, -0.4, 0.3])
_cf_unit(i, n) = Float64[j == i for j in 1:n]
_cf_flat(x::Tuple) = vcat(x...)

# A consumer-owned point that records actual scalar reads. Passing it through
# the public query must not copy/traverse the entire packed vector.
struct _CFReadPoint <: AbstractVector{Float64}
    data::Vector{Float64}
    reads::Vector{Int}
end
Base.size(u::_CFReadPoint) = size(u.data)
Base.IndexStyle(::Type{_CFReadPoint}) = IndexLinear()
Base.getindex(u::_CFReadPoint, i::Int) = (push!(u.reads, i); u.data[i])

function _cf_support(fp, u, args...)
    reads = Base.invokelatest(fp, u, args...).unconstrained
    reads === :whole ? collect(eachindex(u)) : reads
end

const _CF_SHARED = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    sigma ~ Exponential(1)
    eta = b .* x .+ z[g]
    @plate for k in eachindex(y1)
        s1 = sum(exp.(eta[r1[k]]))
        s2 = sum(eta[r2[k]] .^ 2)
        m1[k] = s1
        m2[k] = s2
        m3[k] = s1 + s2 + sum(sin.(eta[r1[k]]))
    end
    y1 .~ Normal.(m1, sigma)
    y2 .~ Normal.(m2, sigma)
    y3 .~ Normal.(m3, sigma)
end

const _CF_NO_SUM = @rkppl begin
    b ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for k in eachindex(y)
        y[k] ~ Normal(b * x[k], sigma)
    end
end

const _CF_DATA_SUM = @rkppl begin
    b ~ Normal(0, 1)
    sigma ~ Exponential(1)
    offset = sum(x .^ 2)
    @plate for k in eachindex(y)
        y[k] ~ Normal(b * x[k] + offset, sigma)
    end
end

# Match the registered-Enzyme gap recorded by the compact-cell tests. Dogfood
# validation sets RK_TEST_REQUIRE_STATIC_ENZYME=1 so any such failure is fatal.
function _cf_static(f)
    try
        f()
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        get(ENV, "RK_TEST_REQUIRE_STATIC_ENZYME", "0") == "1" && rethrow()
        @test occursin("Mismatched activity", sprint(showerror, err))
        @test_broken false
        nothing
    end
end

# This file deliberately uses ordinary static Reverse. A runtime-activity
# backend would not validate the intended sparse Laplace path.
const _CF_REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)
const _CF_SECOND = SecondOrder(_CF_REVERSE, AutoEnzyme(; mode = Enzyme.Forward))
const _CF_THIRD = SecondOrder(_CF_SECOND, AutoEnzyme(; mode = Enzyme.Forward))

function _cf_build(model, data)
    bound = model(; data.g, data.x, data.r1, data.r2) | (; data.y)
    built = build_kernel(bound)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    (; bound, built, f)
end

@testset "factored cell values and authored spellings" begin
    data = _cf_data()
    @testset "$spelling" for (spelling, model) in
            (("inline", _CF_INLINE), ("local", _CF_LOCAL),
             ("earlier plate", _CF_EARLIER), ("row scale", _CF_ROW_SCALE))
        (; bound, built, f) = _cf_build(model, data)
        @test length(cell_reductions(f)) == 2
        @test Set(Tuple(reduction_rows(f, j, 1)) for j in 1:2) ==
              Set((Tuple(data.r1[1]), Tuple(data.r2[1])))
        @test all(isempty(reduction_rows(f, j, 3)) for j in 1:2)
        u = collect(range(-0.2, 0.3; length = built.layout.total))
        # The full pointwise query is independent of cell row pushdown; use
        # it and an independent formula as the primal oracles for every spelling.
        whole = prepare_query(built, bound, :pointwise)(u).y
        names = string.(coordinate_names(built.layout))
        b = u[findfirst(==("b"), names)]
        sigma = exp(u[findfirst(==("sigma"), names)])
        bi, si = findfirst(==("b"), names), findfirst(==("sigma"), names)
        eta = [b * data.x[r] + u[findfirst(==("z.$(data.g[r])"), names)]
               for r in eachindex(data.x)]
        # Primal queries use the same factor graphs as derivative preparation.
        sums = Vector{Vector{Float64}}()
        for j in 1:2
            k = prepare_cell_factor_query(f, j)
            @test k isa CellFactorQuery
            fp = partition_footprint_kernel(k.kernel)
            for c in 1:3, (t, row) in enumerate(reduction_rows(f, j, c))
                point = _CFReadPoint(u, Int[])
                @test k(point, c, t) ≈ k(u, c, t)
                expected = sort([bi, findfirst(==("z.$(data.g[row])"), names)])
                @test sort(unique(point.reads)) == expected
                @test _cf_support(fp, u, c, t) == expected
            end
            push!(sums, [sum((k(u, c, t) for t in eachindex(reduction_rows(f, j, c)));
                            init = 0.0) for c in 1:3])
            @test_throws BoundsError k(u, 3, 1)
        end
        outer = prepare_cell_factor_query(f, :outer)
        ofp = partition_footprint_kernel(outer.kernel)
        for c in 1:3
            scale = model === _CF_ROW_SCALE ? sigma * (1 + abs(data.y[c])) : sigma
            expected = logpdf(Normal(log1p(sum(exp.(eta[data.r1[c]]))) +
                                     sum(abs2, eta[data.r2[c]]), scale), data.y[c])
            @test outer(u, [s[c] for s in sums], c) ≈ expected
            point = _CFReadPoint(u, Int[])
            @test outer(point, [s[c] for s in sums], c) ≈ expected
            @test sort(unique(point.reads)) == [si]
            @test _cf_support(ofp, u, [s[c] for s in sums], c) == [si]
            @test whole[c] ≈ expected
        end
        @test_throws ContractValidationError outer(u, Float64[], 1)
        @test_throws ContractValidationError prepare_cell_factor_query(f, 0)
    end
end

@testset "ordinary cell windows include every named reduction's rows" begin
    costs = NamedTuple[]
    for (model, n) in ((_CF_LOCAL, 6), (_CF_INLINE, 6), (_CF_EARLIER, 6),
                       (_CF_LOCAL, 6_000))
        data = _cf_data(n)
        bound = model(; data.g, data.x, data.r1, data.r2) | (; data.y)
        built = build_kernel(bound)
        q = prepare_cell_query(built, bound, :y; locality = :error)
        fp = partition_footprint_kernel(q.kernel)
        u = collect(range(-0.2, 0.3; length = built.layout.total))
        names = string.(coordinate_names(built.layout))
        bi, si = findfirst(==("b"), names), findfirst(==("sigma"), names)
        zi = [findfirst(==("z.$g"), names) for g in data.g]
        eta = u[bi] .* data.x .+ u[zi]
        sigma = exp(u[si])
        whole = prepare_query(built, bound, :pointwise)(u).y
        sampler = n == 6 ? prepare_cell_sampler(
            built, bound, :y, u; backend = _CF_REVERSE, locality = :error) : nothing
        for c in eachindex(data.y)
            terms = exp.(eta[data.r1[c]])
            mu = log1p(sum(terms)) + sum(abs2, eta[data.r2[c]])
            expected = logpdf(Normal(mu, sigma), data.y[c])
            @test q(u, c) ≈ expected ≈ whole[c]
            rows = union(data.r1[c], data.r2[c])
            support = sort!(unique([bi; si; zi[rows]]))
            point = _CFReadPoint(u, Int[])
            @test q(point, c) ≈ expected
            @test sort!(unique(point.reads)) == support
            @test _cf_support(fp, u, c) == support
            if sampler !== nothing
                # Independent derivative, including repeated and overlapping
                # rows. This is ordinary Reverse, without runtime activity.
                dmu = zeros(n)
                for (j, r) in enumerate(data.r1[c])
                    dmu[r] += terms[j] / (1 + sum(terms))
                end
                for r in data.r2[c]
                    dmu[r] += 2eta[r]
                end
                residual = data.y[c] - mu
                expected_gradient = zeros(length(u))
                expected_gradient[bi] = residual / sigma^2 * dot(dmu, data.x)
                expected_gradient[si] = residual^2 / sigma^2 - 1
                expected_gradient[zi] .= residual / sigma^2 .* dmu
                value, gradient = cell_value_and_gradient!(sampler, similar(u), u, c)
                @test value ≈ expected
                @test gradient ≈ expected_gradient atol = 1e-12
            end
        end
        if model === _CF_LOCAL
            q(u, 2)
            push!(costs, (; bytes = minimum(@allocated(q(u, 2)) for _ in 1:3),
                           nodes = _cf_nodes(q.kernel.ast)))
            # A bad row in the second reduction is checked only when its
            # cell is requested; another cell keeps its ordinary result.
            bad_r2 = copy(data.r2)
            bad_r2[2] = [n + 1]
            bad_bound = model(; data.g, data.x, data.r1, r2 = bad_r2) | (; data.y)
            bad_q = prepare_cell_query(build_kernel(bad_bound), bad_bound, :y)
            @test bad_q(u, 1) ≈ whole[1]
            @test_throws BoundsError bad_q(u, 2)
        end
    end
    @test costs[1].bytes == costs[2].bytes
    @test costs[1].nodes == costs[2].nodes
end

@testset "row metadata on a singleton observation domain" begin
    data = _cf_data()
    singleton = merge(data, (; y = data.y[1:1], r1 = data.r1[1:1], r2 = data.r2[1:1]))
    (; f) = _cf_build(_CF_ROW_SCALE, singleton)
    @test Set(Tuple(reduction_rows(f, j, 1)) for j in 1:2) ==
          Set((Tuple(data.r1[1]), Tuple(data.r2[1])))
end

@testset "a global data sum stays in the outer computation" begin
    x, y = [0.3, -0.8], [1.0, -0.2]
    bound = _CF_DATA_SUM(; x) | (; y)
    built = build_kernel(bound)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    @test isempty(cell_reductions(f))
    outer = prepare_cell_factor_query(f, :outer)
    u = [0.2, -0.3]
    values = [outer(u, Float64[], c) for c in eachindex(y)]
    @test values ≈ prepare_query(built, bound, :pointwise)(u).y
    @test values ≈ [logpdf(Normal(u[1] * x[c] + sum(abs2, x), exp(u[2])), y[c])
                    for c in eachindex(y)]
end

@testset "shared plate outputs discover only their own sums" begin
    data = _cf_data()
    bound = _CF_SHARED(; data.g, data.x, data.r1, data.r2) |
            (; y1 = data.y, y2 = data.y, y3 = data.y)
    built = build_kernel(bound)
    full = prepare_query(built, bound, :pointwise)(zeros(built.layout.total))
    for (observation, count) in ((:y1, 1), (:y2, 1), (:y3, 3))
        f = prepare_cell_factors(built, bound, observation; locality = :error)
        @test length(cell_reductions(f)) == count
        u = zeros(built.layout.total)
        sums = map(1:count) do j
            k = prepare_cell_factor_query(f, j)
            sum((k(u, 1, t) for t in eachindex(reduction_rows(f, j, 1))); init = 0.0)
        end
        outer = prepare_cell_factor_query(f, :outer)
        @test outer(u, sums, 1) ≈ getproperty(full, observation)[1]
        mean = observation === :y2 ? 0.0 : Float64(length(data.r1[1]))
        @test outer(u, sums, 1) ≈ logpdf(Normal(mean, 1), data.y[1])
    end
end

@testset "factored cell gradient, Hessian and third chain rule" begin
    _cf_static() do
        data = _cf_data()
        (; bound, built, f) = _cf_build(_CF_INLINE, data)
        n = built.layout.total
        u = collect(range(-0.2, 0.3; length = n))
        v = cos.(collect(1:n)) ./ n
        names = string.(coordinate_names(built.layout))
        bi = findfirst(==("b"), names)
        si = findfirst(==("sigma"), names)
        # Discover each selection through the public value query, exactly as
        # a consumer must before any compact derivative operator exists.
        term_values = [prepare_cell_factor_query(f, j) for j in 1:2]
        term_footprints = [partition_footprint_kernel(q.kernel) for q in term_values]
        selection(j, c, t) = _cf_support(term_footprints[j], u, c, t)
        outer_value = prepare_cell_factor_query(f, :outer)
        outer_footprint = partition_footprint_kernel(outer_value.kernel)
        oi = _cf_support(outer_footprint, u, zeros(2), 1)
        @test oi == [si]
        units = ntuple(i -> _cf_unit(i, 2), 2)
        term_ops = map(1:2) do j
            indices = selection(j, 1, 1)
            (prepare_cell_factor_ad(prepare_ad, f, j, u, indices, _CF_REVERSE),
             prepare_cell_factor_ad(prepare_ad_hvp, f, j, u, indices, _CF_SECOND, units),
             prepare_cell_factor_ad(prepare_ad_third, f, j, u, indices, _CF_THIRD, (v[indices],)))
        end
        outer_units = ntuple(3) do i
            e = _cf_unit(i, 3)
            (e[1:2], e[3:3])
        end
        outer_g = prepare_cell_factor_ad(prepare_ad, f, :outer, u, oi, _CF_REVERSE)
        outer_h = prepare_cell_factor_ad(prepare_ad_hvp, f, :outer, u, oi,
                                        _CF_SECOND, outer_units)
        outer_t = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, oi,
                                        _CF_THIRD, ((zeros(2), zeros(1)),))
        allindices = collect(1:n)
        full_units = ntuple(i -> _cf_unit(i, n), n)
        whole_g = prepare_compact_cell_ad(prepare_ad, built, bound, :y, u, allindices, _CF_REVERSE)
        whole_h = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y,
                                        u, allindices, _CF_SECOND, full_units)
        whole_t = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y,
                                        u, allindices, _CF_THIRD, (v,))
        for c in 1:3
            sums, J = zeros(2), zeros(3, n)
            Hs, Ts = [zeros(n,n) for _ in 1:3], [zeros(n) for _ in 1:3]
            J[3, si] = 1
            for j in 1:2
                tg, th, tt = term_ops[j]
                for (t, row) in enumerate(reduction_rows(f, j, c))
                    indices = selection(j, c, t)
                    point = (u, u[indices], indices, c, t)
                    value, gradient = ad_value_and_gradient(tg, point...)
                    _, columns = ad_gradient_and_hvp(th, units, point...)
                    _, _, (third,) = ad_gradient_hvp_and_third(tt, (v[indices],), point...)
                    sums[j] += value
                    J[j, indices] .+= gradient
                    Hs[j][indices, indices] .+= hcat(columns...)
                    Ts[j][indices] .+= third
                end
            end
            point = (u, sums, u[oi], oi, c)
            value, gout = ad_value_and_gradient(outer_g, point...)
            _, columns = ad_gradient_and_hvp(outer_h, outer_units, point...)
            gF, HF = _cf_flat(gout), hcat(_cf_flat.(columns)...)
            w = J * v
            _, _, (tout,) = ad_gradient_hvp_and_third(outer_t, ((w[1:2], w[3:3]),), point...)
            gradient = J' * gF
            hessian = J' * HF * J + sum(gF[j] .* Hs[j] for j in 1:3)
            acceleration = [dot(v, H * v) for H in Hs]
            B = reduce(vcat, (permutedims(H * v) for H in Hs))
            third = J' * _cf_flat(tout) + J' * HF * acceleration + 2B' * HF * w +
                    sum(gF[j] .* Ts[j] for j in 1:3)
            expected_value, expected_g = ad_value_and_gradient(whole_g, u, u, allindices, c)
            _, expected_H = ad_gradient_and_hvp(whole_h, full_units, u, u, allindices, c)
            _, _, (expected_T,) = ad_gradient_hvp_and_third(whole_t, (v,), u, u, allindices, c)
            @test value ≈ expected_value
            @test gradient ≈ expected_g atol = 2e-10
            @test hessian ≈ hcat(expected_H...) atol = 2e-9
            @test third ≈ expected_T atol = 2e-8
        end
    end
end

@testset "one reduction keeps vector-shaped outer AD exemplars" begin
    _cf_static() do
        rows, y = [[1, 2, 2], [3], Int[]], [3.2, -0.4, 0.3]
        bound = _CF_SINGLE(; g = [1, 2, 3], rows) | (; y)
        built = build_kernel(bound)
        f = prepare_cell_factors(built, bound, :y; locality = :error)
        @test length(cell_reductions(f)) == 1
        u = collect(range(-0.2, 0.3; length = built.layout.total))
        si = findfirst(==(:sigma), coordinate_names(built.layout))
        term = prepare_cell_factor_query(f, 1)
        outer = prepare_cell_factor_query(f, :outer)
        indices = _cf_support(partition_footprint_kernel(outer.kernel), u, [0.0], 1)
        @test indices == [si]
        units = (([1.0], [0.0]), ([0.0], [1.0]))
        v = ([0.4], [-0.3])
        # Exercise both an ordinary exemplar and an empty reduction. The
        # calls retain the same vector shapes for every cell.
        for cell0 in (1, 3)
            g = prepare_cell_factor_ad(prepare_ad, f, :outer, u, indices, _CF_REVERSE; cell0)
            h = prepare_cell_factor_ad(prepare_ad_hvp, f, :outer, u, indices,
                                       _CF_SECOND, units; cell0)
            t = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, indices,
                                       _CF_THIRD, (v,); cell0)
            for c in eachindex(y)
                s = sum((term(u, c, k) for k in eachindex(reduction_rows(f, 1, c))); init = 0.0)
                point = (u, [s], u[indices], indices, c)
                value, gradient = ad_value_and_gradient(g, point...)
                gh, columns = ad_gradient_and_hvp(h, units, point...)
                gt, (hv,), (tv,) = ad_gradient_hvp_and_third(t, (v,), point...)
                d, e = y[c] - s, exp(-2u[si])
                expected_g = [d * e, -1 + d^2 * e]
                H = [-e -2d*e; -2d*e -2d^2*e]
                a, b = only(v[1]), only(v[2])
                third = [4 * e * a * b + 4 * d * e * b^2,
                         2 * e * a^2 + 8 * d * e * a * b + 4 * d^2 * e * b^2]
                @test value ≈ outer(u, [s], c) ≈ logpdf(Normal(s, exp(u[si])), y[c])
                @test _cf_flat(gradient) ≈ expected_g
                @test _cf_flat(gh) ≈ expected_g
                @test _cf_flat(gt) ≈ expected_g
                @test hcat(_cf_flat.(columns)...) ≈ H
                @test _cf_flat(hv) ≈ H * _cf_flat(v)
                @test _cf_flat(tv) ≈ third
            end
        end
        # A sum-only active point still has two vectors, the second empty.
        sum_only = (([1.0], Float64[]),)
        h = prepare_cell_factor_ad(prepare_ad_hvp, f, :outer, u, Int[],
                                   _CF_SECOND, sum_only)
        gradient, (hv,) = ad_gradient_and_hvp(h, sum_only, u, [3.0], Float64[], Int[], 1)
        @test isempty(gradient[2]) && isempty(hv[2])
        @test only(hv[1]) ≈ -exp(-2u[si])
    end
end

@testset "no reductions and an empty first cell" begin
    _cf_static() do
        bound = _CF_NO_SUM(; x = [0.3, -0.8]) | (; y = [1.0, -0.2])
        built = build_kernel(bound)
        f = prepare_cell_factors(built, bound, :y; locality = :error)
        @test isempty(cell_reductions(f))
        u, indices = [0.2, -0.3], [1, 2]
        value_query = prepare_cell_factor_query(f, :outer)
        @test _cf_support(partition_footprint_kernel(value_query.kernel), u, Float64[], 2) == indices
        q = prepare_cell_factor_ad(prepare_ad, f, :outer, u, indices, _CF_REVERSE)
        whole = prepare_compact_cell_ad(prepare_ad, built, bound, :y, u, indices, _CF_REVERSE)
        value, (sums_gradient, gradient) = ad_value_and_gradient(q, u, Float64[], u, indices, 2)
        expected_value, expected_gradient = ad_value_and_gradient(whole, u, u, indices, 2)
        @test isempty(sums_gradient)
        @test value ≈ expected_value
        @test value_query(u, Float64[], 2) ≈ expected_value
        @test gradient ≈ expected_gradient

        data = merge(_cf_data(), (; r1 = [Int[], [1, 2], Int[]], r2 = [Int[], [1], Int[]]))
        (; built, f) = _cf_build(_CF_INLINE, data)
        u = zeros(built.layout.total)
        indices = collect(1:built.layout.total)
        term = prepare_cell_factor_ad(prepare_ad, f, 1, u, indices, _CF_REVERSE; cell0 = 2)
        value, gradient = ad_value_and_gradient(term, u, u, indices, 2, 1)
        @test isfinite(value)
        @test all(isfinite, gradient)
        @test_throws BoundsError ad_value_and_gradient(term, u, u, indices, 1, 1)
    end
end

@testset "factor call cost is independent of term count and model length" begin
    _cf_static() do
        costs = map((32, 4096)) do n
            data = merge(_cf_data(n), (; r1 = [collect(1:n), Int[], Int[]],
                                       r2 = [collect(1:n), Int[], Int[]]))
            (; built, f) = _cf_build(_CF_INLINE, data)
            u = zeros(built.layout.total)
            names = string.(coordinate_names(built.layout))
            ti = sort([findfirst(==("b"), names), findfirst(==("z.1"), names)])
            oi = [findfirst(==("sigma"), names)]
            term = prepare_cell_factor_ad(prepare_ad_third, f, 1, u, ti,
                                          _CF_THIRD, ([0.2, -0.3],))
            outer = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, oi,
                                           _CF_THIRD, (([0.2, -0.3], [0.1]),))
            ta, oa, sums = u[ti], u[oi], [Float64(n), 0.0]
            termcall() = ad_gradient_hvp_and_third(term, ([0.2, -0.3],), u, ta, ti, 1, 1)
            outercall() = ad_gradient_hvp_and_third(outer, (([0.2, -0.3], [0.1]),),
                                                  u, sums, oa, oi, 1)
            tq, oq = prepare_cell_factor_query(f, 1), prepare_cell_factor_query(f, :outer)
            tfp, ofp = partition_footprint_kernel(tq.kernel), partition_footprint_kernel(oq.kernel)
            point = _CFReadPoint(u, Int[])
            @test tq(point, 1, 1) ≈ tq(u, 1, 1)
            @test sort(unique(point.reads)) == ti
            @test length(point.reads) <= 8
            empty!(point.reads)
            @test oq(point, sums, 1) ≈ oq(u, sums, 1)
            @test sort(unique(point.reads)) == oi
            @test length(point.reads) <= 4
            @test _cf_support(tfp, u, 1, 1) == ti
            @test _cf_support(ofp, u, sums, 1) == oi
            (; term_value = merge(_cf_cost(() -> tq(u, 1, 1)),
                                 (; nodes = _cf_nodes(code_expr(tq.kernel)))),
               outer_value = merge(_cf_cost(() -> oq(u, sums, 1)),
                                   (; nodes = _cf_nodes(code_expr(oq.kernel)))),
               term_footprint = merge(_cf_cost(() -> _cf_support(tfp, u, 1, 1)),
                                      (; nodes = _cf_nodes(code_expr(tfp)))),
               outer_footprint = merge(_cf_cost(() -> _cf_support(ofp, u, sums, 1)),
                                       (; nodes = _cf_nodes(code_expr(ofp)))),
               term = merge(_cf_cost(termcall), (; nodes = _cf_nodes(code_expr(term.prepared.kernel)))),
               outer = merge(_cf_cost(outercall), (; nodes = _cf_nodes(code_expr(outer.prepared.kernel)))))
        end
        @info "factored cell cost (32 vs 4096 rows and latent coordinates)" costs
        for name in (:term, :outer, :term_value, :outer_value, :term_footprint, :outer_footprint)
            small, large = getproperty(costs[1], name), getproperty(costs[2], name)
            @test large.bytes <= small.bytes + 4096
            @test large.ns < 8small.ns
            @test large.nodes == small.nodes
        end
    end
end

# Independent small normal-density oracle. Coordinates are the supplied sum
# followed by the four selected packed coordinates in their public order.
function _cf_direct_derivatives(q, names, y)
    b, a, s, z = (1 + findfirst(n -> startswith(n, key), names)
                  for key in ("b", "scale", "sigma", "z."))
    scale = exp(q[a])
    d, e = y - (q[1] + q[b] + scale * q[z]), exp(-2q[s])
    dm = zeros(5)
    dm[1] = dm[b] = 1
    dm[a], dm[z] = scale * q[z], scale
    Hm = zeros(5, 5)
    Hm[a, a], Hm[a, z], Hm[z, a] = scale * q[z], scale, scale
    gradient = d * e * dm
    gradient[s] += -1 + d^2 * e
    H = -e * (dm * dm') + d * e * Hm
    H[:, s] .-= 2d * e * dm
    H[s, :] .-= 2d * e * dm
    H[s, s] -= 2d^2 * e
    gradient, H
end

@testset "outer factors keep direct scalar reads local" begin
    _cf_static() do
        costs = map((16, 64, 4096)) do n
            rows, anchor, y = [collect(1:n), [n, 2, 2], Int[]], [1, n, 2], [1.2, -0.4, 0.3]
            bound = _CF_DIRECT(; g = collect(1:n), rows, anchor) | (; y)
            built = build_kernel(bound)
            f = prepare_cell_factors(built, bound, :y; locality = :error)
            @test length(cell_reductions(f)) == 1
            u = collect(range(-0.2, 0.3; length = built.layout.total))
            names = string.(coordinate_names(built.layout))
            at(name) = findfirst(==(name), names)
            globals = [at("b"), at("scale"), at("sigma")]
            eta = [u[at("b")] + exp(u[at("scale")]) * u[at("z.$r")] for r in 1:n]
            outer, term = prepare_cell_factor_query(f, :outer), prepare_cell_factor_query(f, 1)
            ofp, tfp = partition_footprint_kernel(outer.kernel), partition_footprint_kernel(term.kernel)
            selection(c) = sort([globals; at("z.$(anchor[c])")])
            oi = selection(1)
            v = ([0.2], [0.1, -0.3, 0.2, -0.1])
            derivative = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, oi, _CF_THIRD, (v,))
            for c in eachindex(y)
                oi = selection(c)
                sums = [sum(exp, eta[rows[c]]; init = 0.0)]
                expected = logpdf(Normal(sums[1] + eta[anchor[c]], exp(u[at("sigma")])), y[c])
                point = _CFReadPoint(u, Int[])
                @test outer(point, sums, c) ≈ expected
                @test sort(unique(point.reads)) == oi
                @test length(point.reads) <= 8
                @test _cf_support(ofp, u, sums, c) == oi
                @test reduction_rows(f, 1, c) == rows[c]
                for t in (isempty(rows[c]) ? () : (1, length(rows[c])))
                    ti = sort([at("b"), at("scale"), at("z.$(rows[c][t])")])
                    @test term(u, c, t) ≈ exp(eta[rows[c][t]])
                    @test _cf_support(tfp, u, c, t) == ti
                end
                q, direction = [sums; u[oi]], _cf_flat(v)
                g, H = _cf_direct_derivatives(q, names[oi], y[c])
                h = 1e-5
                _, Hp = _cf_direct_derivatives(q + h * direction, names[oi], y[c])
                _, Hm = _cf_direct_derivatives(q - h * direction, names[oi], y[c])
                gradient, (hv,), (third,) = ad_gradient_hvp_and_third(
                    derivative, (v,), u, sums, u[oi], oi, c)
                @test _cf_flat(gradient) ≈ g
                @test _cf_flat(hv) ≈ H * direction
                @test _cf_flat(third) ≈ (Hp - Hm) * direction / (2h) rtol = 2e-7
            end
            oi = selection(1)
            sums, active = [0.0], u[oi]
            (; value = merge(_cf_cost(() -> outer(u, sums, 1)),
                             (; nodes = _cf_nodes(code_expr(outer.kernel)))),
               footprint = merge(_cf_cost(() -> _cf_support(ofp, u, sums, 1)),
                                 (; nodes = _cf_nodes(code_expr(ofp)))),
               derivative = merge(_cf_cost(() -> ad_gradient_hvp_and_third(
                    derivative, (v,), u, sums, active, oi, 1)),
                    (; nodes = _cf_nodes(code_expr(derivative.prepared.kernel)))))
        end
        @info "outer factor with direct reads (16, 64, 4096 latent levels)" costs
        for name in (:value, :footprint, :derivative), large in costs[2:end]
            a, b = getproperty(first(costs), name), getproperty(large, name)
            @test b.bytes <= a.bytes + 4096
            @test b.ns < 8a.ns
            @test b.nodes == a.nodes
        end
    end
end

@testset "multiple sums and direct reads composed from an earlier plate" begin
    g, rows, rows2 = [1, 2, 1, 3], [[1, 2, 2], Int[]], [[3, 4], Int[]]
    anchor, other, y = [4, 2], [1, 2], [0.4, -0.2]
    bound = _CF_DIRECT_MIXED(; g, rows, rows2, anchor, other) | (; y)
    built = build_kernel(bound)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    @test length(cell_reductions(f)) == 2
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    names = string.(coordinate_names(built.layout))
    at(name) = findfirst(==(name), names)
    eta = [u[at("b")] + exp(u[at("scale")]) * u[at("z.$r")] for r in g]
    outer = prepare_cell_factor_query(f, :outer)
    fp = partition_footprint_kernel(outer.kernel)
    whole = prepare_query(built, bound, :pointwise)(u).y
    terms = [prepare_cell_factor_query(f, j) for j in 1:2]
    for c in eachindex(y)
        sums = [sum((terms[j](u, c, t)
                     for t in eachindex(reduction_rows(f, j, c))); init = 0.0) for j in 1:2]
        expected = logpdf(Normal(log1p(sum(exp, eta[rows[c]]; init = 0.0)) +
            sum(abs2, eta[rows2[c]]; init = 0.0) + eta[anchor[c]]^2 + eta[other[c]] + eta[1],
            exp(u[at("sigma")])), y[c])
        oi = sort(unique([at("b"), at("scale"), at("sigma"),
                          at("z.$(g[anchor[c]])"), at("z.$(g[other[c]])"), at("z.$(g[1])")]))
        point = _CFReadPoint(u, Int[])
        @test outer(point, sums, c) ≈ whole[c] ≈ expected
        @test sort(unique(point.reads)) == oi
        @test _cf_support(fp, u, sums, c) == oi
    end
end

@testset "direct factor reads retain singleton bounds" begin
    for anchor in ([1], [2])
        bound = _CF_DIRECT(; g = [1], rows = [[1]], anchor) | (; y = [0.4])
        built = build_kernel(bound)
        f = prepare_cell_factors(built, bound, :y; locality = :error)
        outer = prepare_cell_factor_query(f, :outer)
        u = zeros(built.layout.total)
        if anchor == [1]
            @test outer(u, [1.0], 1) ≈ logpdf(Normal(1, 1), 0.4)
            @test _cf_support(partition_footprint_kernel(outer.kernel), u, [1.0], 1) == collect(1:4)
        else
            @test_throws BoundsError outer(u, [1.0], 1)
        end
    end
end

# A destructuring outside a plate is one recipe with several outputs. Factoring
# splits it at its index and sum boundaries like any other recipe, so an index
# read or a sum in its right-hand side prepares and factors as it does in a
# statement of its own (snag `prepare-cell-fac-469bcafb`).
_cf_basis(t, k, p) = (hcat(sin.(t ./ p), cos.(t ./ p)) .* k, Float64[1.0, 2.0], 0.1)
_cf_decay(w, rho) = exp.(-w .* rho)
@rkppl _cf_effect(P, w, floor) = begin
    rho ~ restricted(LogNormal(0.0, 1.0), floor, Inf)
    z[axes(P, 2)] .~ Normal.(0, 1)
    return P * (_cf_decay(w, rho) .* z)
end
const _CF_BASIS_INLINE = @rkppl begin
    (P, w, floor) = _cf_basis(t, modes[1], settings[1])
    f ~ _cf_effect(P, w, floor)
    @plate for i in eachindex(y)
        y[i] ~ Normal(f[i], 1)
    end
end
const _CF_BASIS_HOISTED = @rkppl begin
    k = modes[1]
    p = settings[1]
    (P, w, floor) = _cf_basis(t, k, p)
    f ~ _cf_effect(P, w, floor)
    @plate for i in eachindex(y)
        y[i] ~ Normal(f[i], 1)
    end
end
_cf_location_scale(s, a) = (s + a, exp(a))
const _CF_TUPLE_SUM_INLINE = @rkppl begin
    a ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    (m, sd) = _cf_location_scale(sum(exp.(z[g[rows]])), a)
    @plate for i in eachindex(y)
        y[i] ~ Normal(m, sd)
    end
end
const _CF_TUPLE_SUM_HOISTED = @rkppl begin
    a ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    total = sum(exp.(z[g[rows]]))
    (m, sd) = _cf_location_scale(total, a)
    @plate for i in eachindex(y)
        y[i] ~ Normal(m, sd)
    end
end

@testset "a destructuring's right-hand side factors as its own statements do" begin
    data = (; t = collect(1.0:6.0), modes = [2], settings = [24.0])
    y = [0.1, 0.2, -0.1, 0.4, 0.0, 0.3]
    function basis(model)
        bound = model(; data...) | (; y)
        built = build_kernel(bound)
        # Strict factors prepare under strict type stability, data prefix included.
        ReactiveKernels._with_type_stability(:error, :all) do
            @test prepare_cell_factors(built, bound, :y; locality = :error) isa CellFactors
        end
        # The location reads `f` whole, so its outer query needs the lenient policy.
        f = prepare_cell_factors(built, bound, :y; locality = :ignore)
        u = collect(range(-0.2, 0.3; length = built.layout.total))
        outer = prepare_cell_factor_query(f, :outer)
        (; count = length(cell_reductions(f)),
           whole = prepare_query(built, bound, :pointwise)(u).y,
           values = [outer(u, Float64[], c) for c in eachindex(y)],
           support = _cf_support(partition_footprint_kernel(outer.kernel), u, Float64[], 1))
    end
    inline, hoisted = basis(_CF_BASIS_INLINE), basis(_CF_BASIS_HOISTED)
    @test inline.count == hoisted.count == 0
    @test inline.values ≈ inline.whole
    @test inline.values ≈ hoisted.values
    @test inline.support == hoisted.support

    sum_data = (; g = [1, 2, 3, 4, 5, 6], rows = [2, 5])
    sum_y = [0.1, 0.2, -0.1]
    function summed(model)
        bound = model(; sum_data...) | (; y = sum_y)
        built = build_kernel(bound)
        f = prepare_cell_factors(built, bound, :y; locality = :error)
        u = collect(range(-0.2, 0.3; length = built.layout.total))
        names = string.(coordinate_names(built.layout))
        term, outer = prepare_cell_factor_query(f, 1), prepare_cell_factor_query(f, :outer)
        rows = reduction_rows(f, 1, 1)
        total = sum(term(u, 1, t) for t in eachindex(rows))
        term_indices = _cf_support(partition_footprint_kernel(term.kernel), u, 1, 1)
        outer_indices = _cf_support(partition_footprint_kernel(outer.kernel), u, [total], 1)
        (; f, u, rows, total, term_indices, outer_indices,
           count = length(cell_reductions(f)),
           term_names = names[term_indices], outer_names = names[outer_indices],
           values = [outer(u, [total], c) for c in eachindex(sum_y)],
           whole = prepare_query(built, bound, :pointwise)(u).y)
    end
    a, b = summed(_CF_TUPLE_SUM_INLINE), summed(_CF_TUPLE_SUM_HOISTED)
    @test a.count == b.count == 1
    @test collect(a.rows) == collect(b.rows) == [2, 5]
    @test a.term_names == b.term_names == ["z.2"]
    # The outer reads the sum, not the two `z` it adds.
    @test a.outer_names == b.outer_names == ["a"]
    @test a.total ≈ b.total
    @test a.values ≈ a.whole
    @test a.values ≈ b.values

    _cf_static() do
        derivative(c) = (
            last(ad_value_and_gradient(prepare_cell_factor_ad(prepare_ad, c.f, :outer, c.u,
                    c.outer_indices, _CF_REVERSE),
                c.u, [c.total], c.u[c.outer_indices], c.outer_indices, 1)),
            last(ad_value_and_gradient(prepare_cell_factor_ad(prepare_ad, c.f, 1, c.u,
                    c.term_indices, _CF_REVERSE),
                c.u, c.u[c.term_indices], c.term_indices, 1, 1)))
        (outer_a, term_a), (outer_b, term_b) = derivative(a), derivative(b)
        @test all(outer_a .≈ outer_b)
        @test term_a ≈ term_b
    end
end
