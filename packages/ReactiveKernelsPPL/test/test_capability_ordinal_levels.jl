using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_ordinal_fixture(kind, n, observed)
    data = Dict(:x => collect(range(-0.5, 0.6; length=n)),
        :y => [observed[mod1(i, length(observed))] for i in 1:n])
    cumulative = kind !== :stopping
    ast = quote
        b ~ Normal(0, 2)
    end
    push!(ast.args, cumulative ? :(c ~ Ordered(Normal(0, 1), 3)) :
        :(c[1:3] .~ Normal.(0, 1)))
    push!(ast.args, kind === :ordered ? :(y .~ OrderedLogistic.(b .* x, Ref(c))) :
        kind === :cumulative ? :(y .~ Ordinal.(Cumulative(), LogitLink(), b .* x, Ref(c))) :
        :(y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(c))))
    plan = lower_rkppl(ast, data; conditioned=keys(data))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; b=0.35, c=[-0.7, 0.1, 0.8]))
    be = only(e for e in built.layout.entries if e.name === :b)
    ce = only(e for e in built.layout.entries if e.name === :c)
    function values(v)
        z = v[ce.offset:ce.offset+ce.size-1]
        c = cumulative ? cumsum([z[1]; exp.(z[2:end])]) : z
        return v[be.offset], c
    end
    logistic(z) = cdf(Logistic(), z)
    function pointwise(v)
        b, c = values(v)
        map(data[:x], data[:y]) do x, y
            eta = b*x
            if cumulative
                hi = y == 4 ? 1.0 : logistic(c[y]-eta)
                lo = y == 1 ? 0.0 : logistic(c[y-1]-eta)
                log(hi-lo)
            else
                lp = sum((log(1-logistic(c[j]-eta)) for j in 1:y-1); init=0.0)
                y == 4 ? lp : lp+log(logistic(c[y]-eta))
            end
        end
    end
    function oracle(v)
        b, c = values(v)
        jac = cumulative ? sum(v[ce.offset+1:ce.offset+ce.size-1]) : 0.0
        sum(pointwise(v))+logpdf(Normal(0, 2), b)+sum(logpdf.(Normal(), c))+jac
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    return (; kind, ast, data, bound, built, u, values, pointwise, oracle, sampler)
end

function _cap_ordinal_fd(f, u)
    h = cbrt(eps(Float64))
    [(f(u+h*e)-f(u-h*e))/(2h) for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

@testset "declared ordinal categories survive incomplete observations" begin
    for kind in (:ordered, :cumulative, :stopping), observed in ((1, 3), (1, 4), (1,))
        fx = _cap_ordinal_fixture(kind, 6, observed)
        @test only(fx.bound.responses).n_levels == 4
        @test only(fx.bound.vector_parameters).size == 3
        @test constrain(fx.built.layout, fx.u).c ≈ last(fx.values(fx.u))
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_ordinal_fd(fx.oracle, fx.u) rtol=5e-6
        @test Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y ≈ fx.pointwise(fx.u)
        bad = merge(fx.data, Dict(:y => [0, 1, 1, 1, 1, 1]))
        @test_throws ContractValidationError bind_data(lower_rkppl(fx.ast, bad; conditioned=keys(bad)), bad)
        bad = merge(fx.data, Dict(:y => [5, 1, 1, 1, 1, 1]))
        @test_throws ContractValidationError bind_data(lower_rkppl(fx.ast, bad; conditioned=keys(bad)), bad)
    end
end
