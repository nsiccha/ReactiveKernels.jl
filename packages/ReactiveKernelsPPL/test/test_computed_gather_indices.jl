using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

include("fixtures/computed_gather_indices.jl")

@testset "computed gather indices bind once and preserve Julia array values" begin
    for kind in (:named, :split, :alias, :second, :inline, :inline_response, :data_expr, :opaque, :levels),
            (G, n) in ((2, 6), (5, 18), (0, 0))
        fx = _cgi_build(kind, G, n)
        saved = deepcopy(fx.data)
        expected = _cgi_oracle(fx, fx.u)
        @test fx.built.layout.total == G + 1
        @test fx.bound.n_obs == n
        @test fx.calls == (kind in (:data_expr, :levels) ? 0 : 1)
        draws = _cgi_query(fx, :draws)
        @test Base.invokelatest(draws, fx.u) ≈ expected.draws
        @test size(Base.invokelatest(draws, fx.u)) == size(expected.draws)
        pointwise = prepare_query(fx.built, fx.bound, :pointwise)
        @test Base.invokelatest(pointwise, fx.u).y ≈ expected.pointwise
        for (port, value) in ((:prior, expected.prior), (:log_jacobian, expected.jacobian))
            query = prepare_query(fx.built, fx.bound, port)
            @test Base.invokelatest(query, fx.u) ≈ value
        end
        before = ComputedGatherModels.calls[]
        q = prepare_sampler(fx.built, fx.bound, fx.u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
        oracle(w) = _cgi_oracle(fx, w).value
        @test value ≈ expected.value
        @test gradient ≈ _cgi_findiff(oracle, fx.u) rtol=1e-5 atol=1e-7
        nextu = fx.u .+ 0.03
        value, gradient = sampler_value_and_gradient!(q, similar(nextu), nextu)
        @test value ≈ oracle(nextu)
        @test gradient ≈ _cgi_findiff(oracle, nextu) rtol=1e-5 atol=1e-7
        @test ComputedGatherModels.calls[] == before
        @test fx.data == saved
    end
end

@testset "computed gather index validation" begin
    source = _cgi_source(:named)
    plan = lower_rkppl(source, (:g, :x, :y); mod=ComputedGatherModels, conditioned=(:y,))
    # Refused: positive integer axis positions; Bool and noninteger raw
    # indices are covered with a data-only identity below, without conversion.
    for bad in ([0, 1], [-1, 1], [1, 3])
        @test_throws ContractValidationError bind_data(plan, Dict(:g=>bad, :x=>ones(2), :y=>zeros(2)))
    end
    raw = quote
        scale ~ Exponential(1)
        z[1:2, 1:1] .~ Normal.(0, 1)
        draws = z .* scale
        index = passthrough(g)
        result = draws[index, 1] .* x
        y .~ Normal.(result, 1)
    end
    p = lower_rkppl(raw, (:g, :x, :y); mod=ComputedGatherModels, conditioned=(:y,))
    for bad in ([true, false], [1.0, 2.0])
        # Refused: gather positions must be Integer and not Bool (§3).
        @test_throws ContractValidationError bind_data(p, Dict(:g=>bad, :x=>ones(2), :y=>zeros(2)))
    end
    live = quote
        scale ~ Exponential(1)
        z[1:2, 1:1] .~ Normal.(0, 1)
        draws = z .* scale
        index = passthrough(g .* scale)
        result = draws[index, 1] .* x
        y .~ Normal.(result, 1)
    end
    # Refused: sampled-value-dependent gather indices violate the data-only contract (§2).
    @test_throws ContractValidationError lower_rkppl(live, (:g, :x, :y);
        mod=ComputedGatherModels, conditioned=(:y,))
end
