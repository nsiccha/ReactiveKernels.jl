using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions, LinearAlgebra, Test

function _rlkj_case(K, eta; literal = false, observed = nothing, axis = 2)
    dimension = literal ? K : :(size(M, $axis))
    ast = quote
        L ~ LKJCholesky($dimension, $eta)
        mu = L[$K, 1] .* x
        y .~ Normal.(mu, 0.7)
    end
    shape = literal ? (3, K) : axis == 1 ? (K, 7) : (7, K)
    data = Dict{Symbol,Any}(:M => zeros(shape...),
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

function _rlkj_stack_case(K, S)
    ast = quote
        e ~ Exponential(1)
        @plate for k in levels(s)
            L[k] ~ LKJCholesky($K, e)
        end
        z[levels(s), 1:$K] .~ Normal.(0, 1)
        @plate for i in eachindex(s)
            r[i, 1:$K] = L[s[i]] * z[s[i], :]
        end
        y .~ Normal.(r[:, $K], 0.7)
    end
    data = Dict(:y => [0.2 * cos(i) for i in 1:11], :s => [mod1(i, S) for i in 1:11])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    (; bound, built = build_kernel(bound))
end

# From K = 4 on, an entry multiplies at least two square roots, so a
# different product order would change its last bits.
@testset "LKJ vine twins agree exactly with the host transform" begin
    for K in (3, 4, 6), S in (1, 3)
        fx = _rlkj_stack_case(K, S)
        u = [0.4 * sin(1.7i) for i in 1:fx.built.layout.total]
        factor = Base.invokelatest(prepare, fx.built.spec;
            have = :unconstrained, want = :L)
        G = Base.invokelatest(factor, u)
        entry = only(e for e in fx.built.layout.entries if e.name === :L)
        P = K * (K - 1) ÷ 2
        U = reshape(u[entry.offset:(entry.offset + P * S - 1)], P, S)
        @test size(G) == (K, K, S)
        @test all(k -> G[:, :, k] == lkj_chol_constrain(U[:, k], K), 1:S)
        @test G == constrain(fx.built.layout, u).L
    end
    # The scalar twin of a hand-built `:cholesky_corr_lkj` vector parameter.
    for K in (3, 4, 6)
        u = [0.4 * sin(1.7i) for i in 1:(K * (K - 1) ÷ 2)]
        stmts = ReactiveKernelsPPL._lkj_vine_statements(:F, K, p -> :(u[$p]))
        entries = (j >= i ? ReactiveKernelsPPL._rl_name(:F, j, i) : 0.0
            for j in 1:K, i in 1:K)
        G = Core.eval(@__MODULE__, :(let u = $u
            $(stmts...)
            [$(entries...)]
        end))
        @test reshape(G, K, K) == lkj_chol_constrain(u, K)
    end
end
