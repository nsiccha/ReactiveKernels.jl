using DifferentiationInterface
using Enzyme
using ReactiveKernels
using Test

isdefined(@__MODULE__, :GuardedNormalizers) || include("fixtures/guarded_normalizers.jl")

@testset "data-only work inside parameter-support endpoint arms" begin
    G = GuardedNormalizers
    for family in (:NB2, :NB1, :Gamma, :Beta, :StudentT, :ZIP, :Hurdle, :BetaBinomial)
        @testset "$family, $n observations" for n in (3, 17)
            f = G.fixture(family, n)
            original = deepcopy(f.data)
            k = G.build(f)
            caches = [r.op.value for r in k.plan.recipes if
                r.op isa ReactiveKernels._BoundConstant &&
                startswith(String(only(r.outputs).name), "bound_plate_")]
            @test !isempty(caches)
            y = f.data.y
            lg = G.loggamma
            terms = family === :Gamma ? (log.(y),) :
                family === :Beta ? (log.(y), log1p.(-y)) :
                family === :StudentT ?
                    (lg.((f.data.df .+ 1) ./ 2) .- lg.(f.data.df ./ 2) .-
                     0.5 .* log.(f.data.df .* pi),) :
                family === :BetaBinomial ?
                    (lg.(f.data.trials .+ 1.0) .- lg.(y .+ 1.0) .-
                     lg.(f.data.trials .- y .+ 1.0),) : (lg.(y .+ 1.0),)
            for term in terms
                @test any(v -> v isa AbstractVector && length(v) == n &&
                          (isapprox(v, term) || isapprox(v, -term)), caches)
            end
            uninferred = Base.invokelatest(ReactiveKernels._uninferred_values,
                                           k, (Vector{Float64},))
            @test length(uninferred) == 0
            for u in ([0.3, -0.2, 0.7], [-0.4, 0.5, -0.3])
                @test Base.invokelatest(k, u) ≈ f.oracle(u) rtol=2e-12
                ad = Base.invokelatest(prepare_ad, k,
                    AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
                value, gradient = Base.invokelatest(ad_value_and_gradient!, ad, similar(u), u)
                @test value ≈ f.oracle(u) rtol=2e-12
                @test gradient ≈ G.findiff(f.oracle, u) rtol=2e-5 atol=2e-7
            end
            @test f.data == original
        end
        @testset "$family support guard" begin
            f = G.fixture(family, 3; raw_nuisance = true)
            k = G.build(f)
            valid = [0.3, -0.2, 0.7]
            invalid = [0.3, -0.2, -0.2]
            @test Base.invokelatest(k, valid) ≈ f.oracle(valid) rtol=2e-12
            @test Base.invokelatest(k, invalid) == -Inf
            ad = Base.invokelatest(prepare_ad, k,
                AutoEnzyme(; mode = Enzyme.Reverse), valid; active = :unconstrained)
            value, gradient = Base.invokelatest(ad_value_and_gradient!, ad,
                                               similar(invalid), invalid)
            @test value == -Inf
            @test gradient ≈ -invalid
        end
    end
end
