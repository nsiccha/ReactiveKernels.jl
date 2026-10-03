using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_range_fixture(kind, n)
    x = collect(range(0.2, 0.8; length=n))
    y = collect(range(-0.2, 0.4; length=n))
    kind === :inactive && n > 1 && (x[2:end] .= -1)
    kind === :longer_inactive && (x = [x; fill(-1.0, n)])
    data = Dict(:y => kind in (:matrix, :free_matrix, :axis1_matrix_inactive) ? hcat(y, y .+ 10) : y,
        :x => kind === :axis1_matrix_inactive ? hcat(x, fill(-1.0, n)) : x)
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
    end
    if kind in (:cross_index, :cross_axis, :colon, :top_singleton)
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
    for kind in (:cross_index, :cross_axis, :colon, :top_singleton, :cross_cell, :singleton, :inactive, :matrix, :singleton_latent, :free_matrix, :longer_inactive, :axis1_matrix_inactive)
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

@testset "empty indexed observation loops" begin
    for kind in (:cross_index, :cross_axis, :colon, :cross_cell, :matrix, :free_matrix)
        fx = _cap_range_fixture(kind, 0)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_range_fd(fx.oracle, fx.u) rtol=5e-6
        @test Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), fx.u) == 0
    end
end
