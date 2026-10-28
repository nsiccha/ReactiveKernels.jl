using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(@__DIR__, "../../../examples/rkppl_repeated_pd.jl"))

function _cap_repeated_pd(groups)
    data = RepeatedPDExample.synthetic_data(groups)
    bound = RepeatedPDExample.model(; subject=data.subject, assay=data.assay, time=data.time) | (; y=data.y)
    built = build_kernel(bound)
    baseline = collect(range(-0.2, 0.3; length=3groups))
    u = unconstrain(built.layout, (; baseline, effect=[0.2, -0.3, 0.4], rate=0.15))
    function pointwise(v)
        p = constrain(built.layout, v)
        [logpdf(Normal(p.baseline[data.subject[i]] + p.effect[data.assay[i]]*(1-exp(-p.rate*data.time[i])), 0.7), data.y[i])
            for i in eachindex(data.y)]
    end
    function oracle(v)
        p = constrain(built.layout, v)
        sum(logpdf.(Normal(), p.baseline)) + sum(logpdf.(Normal(), p.effect)) +
            logpdf(Exponential(), p.rate) + log(p.rate) + sum(pointwise(v))
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    (; data, bound, built, u, sampler, oracle, pointwise)
end

@testset "explicit PD function preserves replicate measurement rows" begin
    for groups in (1, 3)
        fx = _cap_repeated_pd(groups)
        p = constrain(fx.built.layout, fx.u)
        mean = RepeatedPDExample.pd_mean.(fx.data.time,
            p.baseline[fx.data.subject], p.effect[fx.data.assay], p.rate)
        @test fx.data.subject[2:3] == [1, 1]
        @test fx.data.assay[2:3] == [2, 2]
        @test fx.data.time[2:3] == [0.0, 0.0]
        @test mean[2] == mean[3]
        @test fx.bound.n_obs == 10groups
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_range_fd(fx.oracle, fx.u) rtol=5e-6
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y
        @test length(pw) == 10groups
        @test pw ≈ fx.pointwise(fx.u)
    end
end
