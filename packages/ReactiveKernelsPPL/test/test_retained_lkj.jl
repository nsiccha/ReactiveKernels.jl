using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions, LinearAlgebra, Test

function _rlkj_case(K, eta; literal = false, observed = nothing)
    dimension = literal ? K : :(size(M, 2))
    ast = quote
        L ~ LKJCholesky($dimension, $eta)
        mu = L[$K, 1] .* x
        y .~ Normal.(mu, 0.7)
    end
    data = Dict{Symbol,Any}(:M => zeros(literal ? 3 : 7, K),
        :x => [0.3, -0.5, 0.7], :y => [0.1, -0.2, 0.3])
    conditioned = observed === nothing ? (:y,) : (:y, :L)
    observed === nothing || (data[:L] = observed)
    plan = lower_rkppl(ast, keys(data); conditioned)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    (; data, bound, built, K, eta)
end

function _rlkj_oracle(fx, u)
    L = lkj_chol_constrain(u, fx.K)
    prior = fx.K == 1 ? 0.0 :
        logpdf(LKJCholesky(fx.K, fx.eta), Cholesky(LowerTriangular(L)))
    likelihood = sum(logpdf.(Normal.(L[fx.K, 1] .* fx.data[:x], 0.7), fx.data[:y]))
    prior + likelihood + lkj_chol_logjac(u, fx.K)
end

# Independent closed form of prior + Jacobian in the partial correlations:
# every z[i,j] carries (K-i+2eta-1)/2 * log(1-z[i,j]^2). The likelihood
# reads z[1,K], whose coordinate is the first in the last column block.
function _rlkj_gradient(fx, u)
    out = similar(u)
    p = 0
    for j in 2:fx.K, i in 1:j-1
        p += 1
        out[p] = -(fx.K - i + 2fx.eta - 1) * tanh(u[p])
    end
    if fx.K > 1
        p = (fx.K - 1) * (fx.K - 2) ÷ 2 + 1
        z = tanh(u[p])
        out[p] += sum((fx.data[:y] .- z .* fx.data[:x]) .* fx.data[:x]) /
            0.7^2 * (1 - z^2)
    end
    out
end

@testset "declared LKJ factors retain triangular iteration" begin
    counts = Int[]
    for literal in (false, true), eta in (1.0, 2.3), K in (1, 2, 4, 8)
        fx = _rlkj_case(K, eta; literal)
        original = deepcopy(fx.data)
        u = [0.3 * sin(i) for i in 1:fx.built.layout.total]
        entry = only(fx.built.layout.entries)
        @test entry.dims == [K, K]
        @test length(u) == K * (K - 1) ÷ 2
        @test coordinate_names(fx.built.layout) == [Symbol("L.", i) for i in eachindex(u)]
        k = prepare_query(fx.built, fx.bound, :sampler)
        @test Base.invokelatest(k, u) ≈ _rlkj_oracle(fx, u) rtol = 1e-12
        factor = Base.invokelatest(prepare, fx.built.spec;
            have = :unconstrained, want = :L)
        # Exact arithmetic/packing parity, including the empty K=1 slice.
        L = Base.invokelatest(factor, u)
        @test L == lkj_chol_constrain(u, K)
        @test istril(L)
        @test all(i -> sum(abs2, L[i, :]) ≈ 1, 1:K)
        @test unconstrain(fx.built.layout, (; L)) ≈ u
        jac = prepare_query(fx.built, fx.bound, :log_jacobian)
        @test Base.invokelatest(jac, u) == lkj_chol_logjac(u, K)
        q = prepare_sampler(fx.built, fx.bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, grad = sampler_value_and_gradient!(q, similar(u), u)
        @test value ≈ _rlkj_oracle(fx, u) rtol = 1e-12
        @test grad ≈ _rlkj_gradient(fx, u) rtol = 1e-10 atol = 1e-11
        @test fx.data == original
        @test u == [0.3 * sin(i) for i in eachindex(u)]
        K > 1 && push!(counts, length(fx.built.spec.graph.recipes))
    end
    @test allequal(counts)
end

@testset "conditioned LKJ factors retain the same diagonal prior" begin
    for K in (1, 2, 5), eta in (1.0, 2.3)
        L = lkj_chol_constrain(fill(0.2, K * (K - 1) ÷ 2), K)
        fx = _rlkj_case(K, eta; observed = L)
        original = deepcopy(fx.data)
        @test fx.built.layout.total == 0
        k = prepare_query(fx.built, fx.bound, :sampler)
        prior = K == 1 ? 0.0 : logpdf(LKJCholesky(K, eta), Cholesky(LowerTriangular(L)))
        likelihood = sum(logpdf.(Normal.(L[K, 1] .* fx.data[:x], 0.7), fx.data[:y]))
        @test Base.invokelatest(k, Float64[]) ≈ prior + likelihood
        @test fx.data == original
    end
end
