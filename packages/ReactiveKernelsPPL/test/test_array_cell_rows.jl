using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

include("fixtures/array_cell_rows.jl")

@testset "ordinary callable array cells collect Julia matrices" begin
    for kind in (:helper, :direct, :literal, :columns, :alias, :matrix, :submodel), n in (0, 3, 11)
        fx = _acr_build(kind, n)
        saved = deepcopy(fx.data)
        expected = _acr_expected(kind, fx.data, fx.u)
        @test coordinate_names(fx.built.layout) == [:a, :b, :sigma]
        @test fx.bound.n_obs == length(fx.data.y)
        # Whole and indexed consumers share the collected Julia matrix.
        location = _acr_query(fx, :location)
        @test Base.invokelatest(location, fx.u) ≈ expected.location
        @test size(Base.invokelatest(location, fx.u)) == (n, 2)
        # A column expression becomes a predictor; the other consumers
        # remain model-level values with authored query names.
        if kind !== :columns
            reads = _acr_query(fx, :reads)
            @test Base.invokelatest(reads, fx.u) ≈ expected.reads
        end
        pointwise = prepare_query(fx.built, fx.bound, :pointwise)
        terms = Base.invokelatest(pointwise, fx.u)
        @test keys(terms) == (:y,)
        @test size(terms.y) == size(fx.data.y)
        @test terms.y ≈ expected.pointwise
        q = prepare_sampler(fx.built, fx.bound, fx.u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
        oracle(w) = _acr_expected(kind, fx.data, w).value
        @test value ≈ expected.value
        @test gradient ≈ _acr_findiff(oracle, fx.u) rtol=1e-5 atol=1e-7
        @test fx.data == saved
    end
end
