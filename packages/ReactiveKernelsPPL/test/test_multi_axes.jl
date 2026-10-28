using Distributions
using DifferentiationInterface
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Each original model is checked against an independent density. Adding an
# unrelated observation axis must preserve its parameters, prior, Jacobian
# and gradient, while adding exactly the new observations' density.
function _ma_check(ast, data, mean; scale = (nt -> 1.0), slots = (), gradient = true,
        transform = identity, host_values = constrain)
    cols = Dict{Symbol,Any}(pairs(data))
    base = bind_data(transform(lower_rkppl(ast, cols;
        conditioned = keys(cols))), cols)
    bbase = build_kernel(base)
    both_ast = Expr(:block, ast.args..., :(other .~ Normal.(other_x, 0.7)))
    both_cols = merge(cols, Dict(:other => [0.2, -0.3, 0.5],
        :other_x => [0.1, 0.4, -0.2]))
    both = bind_data(transform(lower_rkppl(both_ast, both_cols;
        conditioned = keys(both_cols))), both_cols)
    built = build_kernel(both)
    @test both.n_obs == base.n_obs + 3
    @test coordinate_names(built.layout) == coordinate_names(bbase.layout)
    for slot in slots
        @test !isempty(getfield(both, slot))
    end
    extra = sum(logpdf.(Normal.(both_cols[:other_x], 0.7), both_cols[:other]))
    q = prepare_query(built, both, :sampler)
    qbase = prepare_query(bbase, base, :sampler)
    ll = prepare_query(built, both, :likelihood)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    nt = host_values(built.layout, u)
    expected = sum(logpdf.(Normal.(mean(both, built.layout, nt, u), scale(nt)), cols[:y]))
    @test Base.invokelatest(ll, u) ≈ expected + extra rtol = 1e-11
    @test Base.invokelatest(q, u) ≈ Base.invokelatest(qbase, u) + extra rtol = 1e-11
    if gradient
        prep = prepare_ad(q, AutoEnzyme(; mode = Enzyme.Reverse), u;
            active = :unconstrained)
        _, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, prep, similar(u), u)
        h = cbrt(eps(Float64))
        ref = map(eachindex(u)) do i
            hi, lo = copy(u), copy(u)
            hi[i] += h
            lo[i] -= h
            (Base.invokelatest(qbase, hi) - Base.invokelatest(qbase, lo)) / (2h)
        end
        @test g ≈ ref rtol = 2e-5 atol = 2e-7
    end
    return (; bound = both, built, q, u)
end

@rkppl _ma_declared_columns(X) = begin
    c[axes(X, 2)] .~ Normal.(0, 1)
    return c
end

@testset "multiple observation axes: declared prior dependencies" begin
    cols = (; y = [0.2, -0.1, 0.4, 0.8], x = [-1.0, 0.5, 2.0, 1.0])
    _ma_check(quote
        R2 ~ Beta(1, 1)
        phi ~ Dirichlet([1.0])
        tau ~ HalfNormal(1)
        a ~ Normal(0, 1)
        b ~ Normal(0, sqrt(phi[1] * R2 * tau^2))
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, cols, (p, l, nt, u) -> nt.a .+ nt.b .* cols.x;
        slots = (:parameters, :vector_parameters))
    _ma_check(quote
        a ~ Normal(0, 1)
        raw ~ Normal(0, 1)
        lambda ~ HalfCauchy(1)
        tau ~ HalfCauchy(1)
        b = raw * lambda * tau
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, cols, (p, l, nt, u) -> nt.a .+
        nt.raw * nt.lambda * nt.tau .* cols.x;
        slots = (:parameters,))
end

_ma_segment(lay, name) = let e = only(e for e in lay.entries if e.name === name)
    e.offset:(e.offset + e.size - 1)
end

@testset "multiple observation axes: latent plates and scans" begin
    cols = (; y = [0.1, 0.2, 0.3, -0.1])
    for range in (:(eachindex(y)), :(1:4))
        ast = quote
            @plate for i in $range
                theta[i] ~ Normal(0, 1)
                y[i] ~ Normal(theta[i], 0.8)
            end
        end
        _ma_check(ast, cols, (p, l, nt, u) -> nt.theta;
            scale = nt -> 0.8, slots = (:plate_parameters,))
    end
    # A defined axis retains its identity through the surface and layout.
    _ma_check(quote
        axis = y .+ 0.0
        @plate for i in eachindex(axis)
            theta[i] ~ Normal(0, 1)
        end
        y .~ Normal.(theta, 1.0)
    end, cols, (p, l, nt, u) -> nt.theta; slots = (:plate_parameters, :derived))
    _ma_check(quote
        phi ~ Normal(0, 0.5)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], 1)
            end
        end
        y .~ Normal.(h, 1.0)
    end, cols, (p, l, nt, u) -> nt.h; slots = (:scans,))
    _ma_check(quote
        phi ~ Normal(0, 0.5)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                e ~ Normal(0, 1)
                h[t] = phi * h[t - 1] + e
            end
        end
        y .~ Normal.(h, 1.0)
    end, cols, (p, l, nt, u) -> begin
        z = u[_ma_segment(l, :_ppl_scan_z_h)]
        h = [z[1]]
        for t in 2:length(z)
            push!(h, nt.phi * h[end] + z[t])
        end
        h
    end; slots = (:scans,))

end

@testset "multiple observation axes: matrices and declared arrays" begin
    cols = (; y = [0.2, -0.1, 0.4, 0.8], x = [-1.0, 0.5, 2.0, 1.0])
    _ma_check(quote
        X = hcat(x)
        alpha ~ Normal(0, 2)
        b[axes(X, 2)] .~ Normal.(0, 1)
        y ~ NormalIDGLM(X, alpha, b, 1.0)
    end, cols, (p, l, nt, u) -> nt.alpha .+ cols.x .* only(nt.b);
        slots = (:matrices, :array_parameters))
    result = _ma_check(quote
        X = hcat(x)
        center = x .+ 0.2
        z[axes(X, 1)] .~ Normal.(center, 1)
        mu = z[rows]
        y .~ Normal.(mu, 1.0)
    end, merge(cols, (; rows = collect(1:4))), (p, l, nt, u) -> nt.z;
        slots = (:array_parameters, :derived))
    @test result.bound.columns[:X] == hcat(cols.x)
    _ma_check(quote
        a ~ Normal(0, 1)
        X = hcat(x)
        b ~ _ma_declared_columns(X)
        mu = a .+ X * b
        y .~ Normal.(mu, 1.0)
    end, cols, (p, l, nt, u) -> nt.a .+ cols.x .* only(nt.b.c);
        slots = (:matrices, :array_parameters, :submodel_scopes))
end

@testset "multiple observation axes: packed missing observations" begin
    cols = (; y = [0.1, -0.3], Jobs_y = [1, 4], x = [-1.0, 0.5, 2.0, 1.0])
    fx = _ma_check(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, cols, (p, l, nt, u) -> nt.a .+ nt.b .* cols.x[cols.Jobs_y];
        transform = p -> ReactiveKernelsPPL._with(p; responses =
            [ReactiveKernelsPPL._with(r; mi_jobs = r.response === :y ? :Jobs_y : nothing)
                for r in p.responses]))
    bad = ReactiveKernelsPPL._with(fx.bound;
        columns = merge(fx.bound.columns, Dict(:Jobs_y => [1, 5])))
    @test_throws "1:n_obs (4)" validate_data(bad)
end

@testset "multiple observation axes: independent indexed observations" begin
    cols = Dict{Symbol,Any}(:x1 => [0.5, 1.0, 1.5, 2.0], :y1 => [0.4, 1.1, 1.4, 2.2],
        :x2 => [0.5, 1.0, 1.5], :y2 => [1, 2, 3])
    ast = quote
        b ~ Normal(0, 1)
        @plate for i in eachindex(y1)
            mu1[i] = b * x1[i]
            y1[i] ~ Normal(mu1[i], 1)
        end
        @plate for j in eachindex(y2)
            mu2[j] = exp(b * x2[j])
            y2[j] ~ Poisson(mu2[j])
        end
    end
    p = bind_data(lower_rkppl(ast, cols; conditioned=keys(cols)), cols)
    @test p.n_obs == 7
    built = build_kernel(p)
    q = prepare_query(built, p, :sampler)
    expected = sum(logpdf.(Normal.(0.2 .* cols[:x1], 1), cols[:y1])) +
        sum(logpdf.(Poisson.(exp.(0.2 .* cols[:x2])), cols[:y2])) + logpdf(Normal(), 0.2)
    @test Base.invokelatest(q, [0.2]) ≈ expected
    cols[:y] = [0.1, 0.3, 0.2, -0.2, 0.4]
    cols[:other] = [-0.1, 0.5]
    push!(ast.args, :(y .~ Normal.(b, 1)), :(other .~ Normal.(b, 1)))
    mixed = bind_data(lower_rkppl(ast, cols; conditioned=keys(cols)), cols)
    @test mixed.n_obs == 14
    mbuilt = build_kernel(mixed)
    mq = prepare_query(mbuilt, mixed, :sampler)
    @test Base.invokelatest(mq, [0.2]) ≈ expected +
        sum(logpdf.(Normal(0.2, 1), cols[:y])) + sum(logpdf.(Normal(0.2, 1), cols[:other]))
    ad = prepare_ad(mq, AutoEnzyme(; mode=Enzyme.Reverse), [0.2]; active=:unconstrained)
    _, g = Base.invokelatest(ad_value_and_gradient!, ad, zeros(1), [0.2])
    dg = sum(cols[:x1] .* (cols[:y1] .- 0.2 .* cols[:x1])) +
        sum(cols[:x2] .* (cols[:y2] .- exp.(0.2 .* cols[:x2]))) +
        sum(cols[:y] .- 0.2) + sum(cols[:other] .- 0.2) - 0.2
    @test only(g) ≈ dg
    # A shared input must still cover the authored observation axis.
    mismatch = deepcopy(ast)
    mismatch.args[end-1] = :(y .~ Normal.(b .* x1, 1))
    @test_throws "column length 4 ≠ the 5 rows of y" bind_data(
        lower_rkppl(mismatch, cols; conditioned=keys(cols)), cols)
end

@testset "multiple observation axes: shared coefficients and ambiguous trajectories" begin
    cols = Dict{Symbol,Any}(:y => [0.2, -0.1, 0.4, 0.8], :x => [-1.0, 0.5, 2.0, 1.0],
        :other => [0.2, -0.3, 0.5], :other_x => [0.1, 0.4, -0.2])
    ast = quote
        X = hcat(x)
        W = hcat(other_x)
        b[axes(X, 2)] .~ Normal.(0, 1)
        mu = X * b
        other_mu = W * b
        y .~ Normal.(mu, 1)
        other .~ Normal.(other_mu, 1)
    end
    p = bind_data(lower_rkppl(ast, cols; conditioned=keys(cols)), cols)
    built = build_kernel(p)
    @test built.layout.total == 1
    @test p.n_obs == 7
    ll = prepare_query(built, p, :likelihood)
    @test Base.invokelatest(ll, [0.2]) ≈
        sum(logpdf.(Normal.(0.2 .* cols[:x], 1), cols[:y])) +
        sum(logpdf.(Normal.(0.2 .* cols[:other_x], 1), cols[:other]))
    delete!(cols, :x)
    delete!(cols, :other_x)
    ambiguous = quote
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(h[t-1], 1)
            end
        end
        y .~ Normal.(h, 1)
        other .~ Normal.(h, 1)
    end
    @test_throws "reads or feeds different row counts" bind_data(
        lower_rkppl(ambiguous, cols; conditioned=keys(cols)), cols)
end
