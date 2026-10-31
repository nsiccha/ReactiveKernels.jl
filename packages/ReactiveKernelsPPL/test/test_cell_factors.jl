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
        # The full pointwise query is independent of cell row pushdown. The
        # ordinary cell path has an existing shared-predictor window bug for
        # the local spelling (owner finding 856af074); use the full density
        # and an independent formula as the primal oracles for every spelling.
        whole = prepare_query(built, bound, :pointwise)(u).y
        names = string.(coordinate_names(built.layout))
        b = u[findfirst(==("b"), names)]
        sigma = exp(u[findfirst(==("sigma"), names)])
        eta = [b * data.x[r] + u[findfirst(==("z.$(data.g[r])"), names)]
               for r in eachindex(data.x)]
        # Primal queries use the same factor graphs as derivative preparation.
        sums = Vector{Vector{Float64}}()
        for j in 1:2
            spec, want = ReactiveKernelsPPL._cell_factor_spec(f, j)
            k = prepare(spec; have = (ReactiveKernelsPPL._query_have(bound)...,
                :_ppl_cell, :_ppl_cell_term), want,
                bound = ReactiveKernelsPPL._query_bound(bound))
            push!(sums, [sum((k(u, c, t) for t in eachindex(reduction_rows(f, j, c)));
                            init = 0.0) for c in 1:3])
            @test_throws BoundsError k(u, 3, 1)
        end
        spec, want = ReactiveKernelsPPL._cell_factor_spec(f, :outer)
        outer = prepare(spec; have = (ReactiveKernelsPPL._query_have(bound)...,
            :_ppl_cell, :_ppl_cell_sums), want,
            bound = ReactiveKernelsPPL._query_bound(bound))
        for c in 1:3
            scale = model === _CF_ROW_SCALE ? sigma * (1 + abs(data.y[c])) : sigma
            expected = logpdf(Normal(log1p(sum(exp.(eta[data.r1[c]]))) +
                                     sum(abs2, eta[data.r2[c]]), scale), data.y[c])
            @test outer(u, c, [s[c] for s in sums]) ≈ expected
            @test whole[c] ≈ expected
        end
    end
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
    spec, want = ReactiveKernelsPPL._cell_factor_spec(f, :outer)
    outer = prepare(spec; have = (ReactiveKernelsPPL._query_have(bound)...,
        :_ppl_cell, :_ppl_cell_sums), want,
        bound = ReactiveKernelsPPL._query_bound(bound))
    u = [0.2, -0.3]
    values = [outer(u, c, Float64[]) for c in eachindex(y)]
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
            spec, want = ReactiveKernelsPPL._cell_factor_spec(f, j)
            k = prepare(spec; have = (ReactiveKernelsPPL._query_have(bound)...,
                :_ppl_cell, :_ppl_cell_term), want,
                bound = ReactiveKernelsPPL._query_bound(bound))
            sum((k(u, 1, t) for t in eachindex(reduction_rows(f, j, 1))); init = 0.0)
        end
        spec, want = ReactiveKernelsPPL._cell_factor_spec(f, :outer)
        outer = prepare(spec; have = (ReactiveKernelsPPL._query_have(bound)...,
            :_ppl_cell, :_ppl_cell_sums), want,
            bound = ReactiveKernelsPPL._query_bound(bound))
        @test outer(u, 1, sums) ≈ getproperty(full, observation)[1]
        mean = observation === :y2 ? 0.0 : Float64(length(data.r1[1]))
        @test outer(u, 1, sums) ≈ logpdf(Normal(mean, 1), data.y[1])
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
        selection(row) = sort([bi, findfirst(==("z.$(data.g[row])"), names)])
        units = ntuple(i -> _cf_unit(i, 2), 2)
        term_ops = map(1:2) do j
            indices = selection(first(reduction_rows(f, j, 1)))
            (prepare_cell_factor_ad(prepare_ad, f, j, u, indices, _CF_REVERSE),
             prepare_cell_factor_ad(prepare_ad_hvp, f, j, u, indices, _CF_SECOND, units),
             prepare_cell_factor_ad(prepare_ad_third, f, j, u, indices, _CF_THIRD, (v[indices],)))
        end
        outer_units = ntuple(3) do i
            e = _cf_unit(i, 3)
            (e[1:2], e[3:3])
        end
        outer_g = prepare_cell_factor_ad(prepare_ad, f, :outer, u, [si], _CF_REVERSE)
        outer_h = prepare_cell_factor_ad(prepare_ad_hvp, f, :outer, u, [si],
                                        _CF_SECOND, outer_units)
        outer_t = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, [si],
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
                    indices = selection(row)
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
            point = (u, sums, u[[si]], [si], c)
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

@testset "no reductions and an empty first cell" begin
    _cf_static() do
        bound = _CF_NO_SUM(; x = [0.3, -0.8]) | (; y = [1.0, -0.2])
        built = build_kernel(bound)
        f = prepare_cell_factors(built, bound, :y; locality = :error)
        @test isempty(cell_reductions(f))
        u, indices = [0.2, -0.3], [1, 2]
        q = prepare_cell_factor_ad(prepare_ad, f, :outer, u, indices, _CF_REVERSE)
        whole = prepare_compact_cell_ad(prepare_ad, built, bound, :y, u, indices, _CF_REVERSE)
        value, (sums_gradient, gradient) = ad_value_and_gradient(q, u, Float64[], u, indices, 2)
        expected_value, expected_gradient = ad_value_and_gradient(whole, u, u, indices, 2)
        @test isempty(sums_gradient)
        @test value ≈ expected_value
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
            (; term = merge(_cf_cost(termcall), (; nodes = _cf_nodes(code_expr(term.prepared.kernel)))),
               outer = merge(_cf_cost(outercall), (; nodes = _cf_nodes(code_expr(outer.prepared.kernel)))))
        end
        @info "factored cell cost (32 vs 4096 rows and latent coordinates)" costs
        for name in (:term, :outer)
            small, large = getproperty(costs[1], name), getproperty(costs[2], name)
            @test large.bytes <= small.bytes + 4096
            @test large.ns < 8small.ns
            @test large.nodes == small.nodes
        end
    end
end
