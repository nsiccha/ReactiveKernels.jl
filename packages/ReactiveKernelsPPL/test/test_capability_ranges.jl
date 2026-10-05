using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_range_fixture(kind, n)
    x = collect(range(0.2, 0.8; length=n))
    y = collect(range(-0.2, 0.4; length=n))
    kind === :inactive && n > 1 && (x[2:end] .= -1)
    kind === :longer_inactive && (x = [x; fill(-1.0, n)])
    # Literal loops select their authored rows: an offset tail `2:n` (empty
    # at n = 0), a single `3:3` row, over a vector or a matrix column.
    literal = kind in (:literal_tail, :literal_whole, :literal_single, :literal_matrix)
    rows = kind === :literal_single ? (3:3) : (2:n)
    data = Dict{Symbol,Any}(:y => kind in (:matrix, :free_matrix, :axis1_matrix_inactive, :literal_matrix) ? hcat(y, y .+ 10) : y,
        :x => kind === :axis1_matrix_inactive ? hcat(x, fill(-1.0, n)) : x)
    # Rejected partial-observation fixtures retain the old missing entries
    # to verify that missing data cannot authorize an authored subset.
    observed = kind in (:top_singleton, :singleton, :inactive, :singleton_latent) ?
        (1:min(n, 1)) : literal ? rows : (1:n)
    # A matrix response observed by column (`y[i, 1]`, or `y[i]` over its
    # first axis) leaves its second column `missing`.
    if kind in (:matrix, :literal_matrix, :axis1_matrix_inactive)
        data[:y] = Union{Missing,Float64}[j == 1 && i in observed ? y[i] : missing
            for i in 1:n, j in 1:2]
    elseif observed != 1:n
        data[:y] = Union{Missing,Float64}[i in observed ? y[i] : missing for i in 1:n]
    end
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
    end
    if kind === :literal_whole
        # A whole value read at the loop index is gathered at those rows.
        push!(ast.args, :(mu = a .+ b .* x))
        push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, Expr(:call, :(:), first(rows), last(rows))),
                Expr(:block, :(y[i] ~ Normal(mu[i], 0.7))))))
    elseif literal
        lhs = kind === :literal_matrix ? :(y[i, 1]) : :(y[i])
        push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, Expr(:call, :(:), first(rows), last(rows))),
                Expr(:block, :(mu = a + b * x[i]), Expr(:call, :~, lhs, :(Normal(mu, 0.7)))))))
    elseif kind in (:cross_index, :cross_axis, :colon, :top_singleton)
        push!(ast.args, :(mu = a .+ b .* x))
        lhs = kind === :cross_index ? :(y[eachindex(x)]) :
            kind === :cross_axis ? :(y[axes(x, 1)]) :
            kind === :colon ? :(y[:]) : :(y[axes(y, 2)])
        push!(ast.args, Expr(:call, :.~, lhs, :(Normal.(mu, 0.7))))
    else
        iterator = kind === :cross_cell ? :(eachindex(x)) :
            kind in (:free_matrix, :longer_inactive) ? :(eachindex(y)) :
            kind in (:singleton, :inactive, :singleton_latent) ? :(axes(y, 2)) : :(axes(y, 1))
        lhs = kind === :matrix ? :(y[i, 1]) : :(y[i])
        value = kind in (:inactive, :longer_inactive, :axis1_matrix_inactive) ?
            :(a + b * sqrt(x[i])) : :(a + b * x[i])
        body = kind === :free_matrix ? quote theta[i] ~ Normal(a, 0.8) end :
            kind === :singleton_latent ? quote
                theta[i] ~ Normal(a, 0.8)
                y[i] ~ Normal(theta[i] + b * x[i], 0.7)
            end : Expr(:block, :(mu = $value), Expr(:call, :~, lhs, :(Normal(mu, 0.7))))
        push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, iterator), body)))
    end
    plan = lower_rkppl(ast, data; conditioned=keys(data))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    latent = kind in (:singleton_latent, :free_matrix)
    cells = kind === :singleton_latent ? 1 : 2n
    values = latent ? (; a=0.25, b=-0.3, theta=fill(0.1, cells)) : (; a=0.25, b=-0.3)
    u = unconstrain(built.layout, values)
    function pointwise(v)
        p = constrain(built.layout, v)
        kind === :free_matrix && return Float64[]
        literal && return [logpdf(Normal(p.a + p.b*x[i], 0.7), y[i]) for i in rows]
        kind === :singleton_latent && return [logpdf(Normal(p.theta[i] + p.b*x[i], 0.7), y[i]) for i in axes(y, 2)]
        if kind in (:longer_inactive, :axis1_matrix_inactive)
            return [logpdf(Normal(p.a + p.b*sqrt(x[i]), 0.7), y[i]) for i in eachindex(y)]
        end
        if kind in (:singleton, :inactive)
            return [logpdf(Normal(p.a + p.b * (kind === :inactive ? sqrt(x[i]) : x[i]), 0.7), y[i])
                for i in axes(y, 2)]
        end
        observed = kind === :top_singleton ? y[axes(y, 2)] : y
        return logpdf.(Normal.(p.a .+ p.b .* x, 0.7), observed)
    end
    function oracle(v)
        p = constrain(built.layout, v)
        prior = logpdf(Normal(), p.a) + logpdf(Normal(), p.b)
        latent && (prior += sum(logpdf.(Normal(p.a, 0.8), p.theta); init=0.0))
        prior + sum(pointwise(v); init=0.0)
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    return (; kind, plan, bound, built, u, sampler, pointwise, oracle, data)
end

function _cap_range_fd(f, u)
    h = cbrt(eps(Float64))
    [(f(u+h*e)-f(u-h*e))/(2h) for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

@testset "response ranges preserve Julia indexing and selected cells" begin
    for kind in (:cross_index, :cross_axis, :colon, :cross_cell, :free_matrix, :longer_inactive)
        fx = _cap_range_fixture(kind, 4)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_range_fd(fx.oracle, fx.u) rtol=5e-6
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u)
        if kind === :free_matrix
            @test pw == (;)
            @test length(constrain(fx.built.layout, fx.u).theta) == 8
        else
            @test pw.y ≈ fx.pointwise(fx.u)
            @test size(pw.y) == size(fx.pointwise(fx.u))
        end
    end
end

@testset "authored response subsets are refused even with missing outside them" begin
    for kind in (:top_singleton, :singleton, :inactive, :matrix, :singleton_latent,
            :axis1_matrix_inactive, :literal_tail, :literal_whole, :literal_single, :literal_matrix)
        @test_throws ContractValidationError _cap_range_fixture(kind, 4)
    end
end

@testset "empty indexed observation loops" begin
    for kind in (:cross_index, :cross_axis, :colon, :cross_cell, :matrix, :free_matrix,
            :literal_tail, :literal_whole, :literal_matrix)
        fx = _cap_range_fixture(kind, 0)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_range_fd(fx.oracle, fx.u) rtol=5e-6
        @test Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), fx.u) == 0
    end
end

@testset "literal plate observations cover the whole response" begin
    x, y = collect(range(0.2, 0.8; length=6)), collect(range(-0.2, 0.4; length=6))
    loop(R) = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in $R
            mu = a + b * x[i]
            y[i] ~ Normal(mu, 0.7)
        end
    end
    # A literal loop covers the whole response and skips missing entries
    # automatically, retaining the full pointwise axis.
    for (R, rows) in ((:(1:6), 1:6),)
        data = (; x, y = Union{Missing,Float64}[
            i == 3 ? missing : y[i] for i in eachindex(y)])
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(loop(R), data; conditioned = keys(data)), data)
        built = build_kernel(bound)
        u = unconstrain(built.layout, (; a = 0.25, b = -0.3))
        pointwise(v) = [i == 3 ? 0.0 : logpdf(Normal(v[1] + v[2] * x[i], 0.7), y[i]) for i in rows]
        oracle(v) = logpdf(Normal(), v[1]) + logpdf(Normal(), v[2]) +
            sum(pointwise(v); init = 0.0)
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u)
        @test gradient ≈ _cap_range_fd(oracle, u) rtol = 5e-6
        output = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
        @test output.y ≈ pointwise(u)
        @test length(output.y) == length(rows)
        @test isequal(data, saved)
    end
    # Refused under provisional user decision `1uhcm3b`: observation statements
    # cover the whole response, with missing entries skipped by binding.
    for R in (:(2:6), :(1:4))
        plan = lower_rkppl(loop(R), (; y, x); conditioned=(:y, :x))
        @test_throws ContractValidationError bind_data(plan, (; y, x))
    end
    # Missing entries do not authorize a partial authored range.
    ym = Union{Missing,Float64}[i == 3 ? missing : y[i] for i in 1:6]
    @test_throws ContractValidationError bind_data(lower_rkppl(loop(:(3:3)),
        (; y = ym, x); conditioned=(:y, :x)), (; y = ym, x))
    for R in (:(2:7), :(0:3))
        plan = lower_rkppl(loop(R), (; y, x); conditioned=(:y, :x))
        # refused: the loop reads `y[7]` / `y[0]` outside the six bound rows
        # (standing @rkppl language principle 3: Julia indexing safety).
        @test_throws ContractValidationError bind_data(plan, (; y, x))
    end
end
