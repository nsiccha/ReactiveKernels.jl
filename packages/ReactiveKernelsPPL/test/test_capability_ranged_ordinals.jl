using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_ranged_ordinal(kind, n, singleton)
    data = Dict(:x => collect(range(-0.4, 0.7; length=n)),
        :y => [mod1(i, 3)+1 for i in 1:n])
    # Observations cover the whole response and binding skips missing
    # entries (provisional USER `1uhcm3b`): `singleton` keeps only y[1].
    singleton && (data[:y] = Union{Missing,Int}[i == 1 ? data[:y][i] : missing
        for i in 1:n])
    ast = quote b ~ Normal(0, 2) end
    push!(ast.args, kind === :stopping ? :(c[1:3] .~ Normal.(0, 1)) : :(c ~ Ordered(Normal(0, 1), 3)))
    obj = kind === :ordered ? :(OrderedLogistic(b*x[i], Ref(c))) :
        kind === :cumulative ? :(Ordinal(Cumulative(), LogitLink(), b*x[i], Ref(c))) :
        :(Ordinal(StoppingRatio(), LogitLink(), b*x[i], Ref(c)))
    push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
        Expr(:for, Expr(:(=), :i, :(eachindex(y))), Expr(:block, Expr(:call, :~, :(y[i]), obj)))))
    bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; b=0.35, c=[-0.7, 0.1, 0.8]))
    be = only(e for e in built.layout.entries if e.name === :b)
    ce = only(e for e in built.layout.entries if e.name === :c)
    function values(v)
        z = v[ce.offset:ce.offset+ce.size-1]
        c = kind === :stopping ? z : cumsum([z[1]; exp.(z[2:end])])
        v[be.offset], c
    end
    function pointwise(v)
        b, c = values(v)
        map(eachindex(data[:y])) do i
            y, eta = data[:y][i], b*data[:x][i]
            # A skipped entry leaves zero in pointwise output.
            ismissing(y) && return 0.0
            if kind === :stopping
                lp = sum((log(ccdf(Logistic(), c[j]-eta)) for j in 1:y-1); init=0.0)
                y == 4 ? lp : lp + log(cdf(Logistic(), c[y]-eta))
            else
                hi = y == 4 ? 1.0 : cdf(Logistic(), c[y]-eta)
                lo = y == 1 ? 0.0 : cdf(Logistic(), c[y-1]-eta)
                log(hi-lo)
            end
        end
    end
    function oracle(v)
        b, c = values(v)
        jac = kind === :stopping ? 0.0 : sum(v[ce.offset+1:ce.offset+ce.size-1])
        logpdf(Normal(0, 2), b) + sum(logpdf.(Normal(), c)) + jac + sum(pointwise(v))
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    (; kind, singleton, bound, built, u, sampler, oracle, pointwise)
end

@testset "indexed ordinal cells retain selected observations and stages" begin
    for kind in (:ordered, :cumulative, :stopping), singleton in (false, true)
        fx = _cap_ranged_ordinal(kind, 6, singleton)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_range_fd(fx.oracle, fx.u) rtol=5e-6
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y
        @test pw ≈ fx.pointwise(fx.u)
        @test size(pw) == (6,)
    end
end

@testset "indexed scale support is checked only in selected cells" begin
    # Binding skips the missing cells, so only y[1]'s scale is checked.
    data = Dict(:y => Union{Missing,Float64}[0.2, missing, missing],
        :s => [0.7, -1.0, -2.0])
    ast = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a, s[i])
        end
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a=0.3))
    @test Base.invokelatest(prepare_query(built, bound, :likelihood), u) ≈ logpdf(Normal(0.3, 0.7), 0.2)
    @test Base.invokelatest(prepare_query(built, bound, :pointwise), u).y ≈ [logpdf(Normal(0.3, 0.7), 0.2), 0, 0]
    bad = merge(data, Dict(:s => [-0.7, 1.0, 2.0]))
    @test_throws ContractValidationError bind_data(lower_rkppl(ast, bad; conditioned=keys(bad)), bad)
end
