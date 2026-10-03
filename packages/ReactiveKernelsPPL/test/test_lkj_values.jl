using Distributions
using LinearAlgebra
using ReactiveKernelsPPL
using Test

# Independent density and finite-difference checks use the helpers in
# test_generator.jl. Eta is an ordinary prior argument, including at 1.
function _lkv_build(K, eta; sampled = true, n = 5, uplo = 'L', data_width = false)
    location = uplo === 'U' ? :(L[1, 2]) : :(L[2, 1])
    width = data_width ? :(size(M, 1)) : K
    ast = quote
        $(sampled ? :(e ~ Exponential(1)) : nothing)
        L ~ LKJCholesky($width, $eta, $uplo)
        y .~ Normal.($location, 0.7)
    end
    filter!(!isnothing, ast.args)
    data = Dict{Symbol,Any}(:y => [0.2 * cos(i) for i in 1:n])
    data_width && (data[:M] = zeros(K, 1))
    sampled || (data[:e] = 1.3)
    plan = lower_rkppl(ast, data; conditioned = (:y,))
    bound = bind_data(plan, data)
    return bound, build_kernel(bound)
end

@testset "Live LKJ values on retained data-sized factors" begin
    for K in (2, 4), uplo in ('L', 'U')
        bound, built = _lkv_build(K, :e; uplo, data_width = true)
        u = [0.25 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        factor = Cholesky(uplo === 'U' ? UpperTriangular(nt.L) : LowerTriangular(nt.L))
        @test size(nt.L) == (K, K)
        @test _query(built.spec, bound, :prior, u) ≈
            logpdf(Exponential(1), nt.e) + logpdf(LKJCholesky(K, nt.e, uplo), factor)
        @test unconstrain(built.layout, nt) ≈ u
        _check_gradient(built.spec, bound, u)
    end
end

function _lkv_stack_build(S, n; uplo = 'L')
    ast = quote
        e ~ Exponential(1)
        @plate for k in levels(s)
            L[k] ~ LKJCholesky(2, e, $uplo)
        end
        z[levels(s), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(s)
            r[i, 1:2] = L[s[i]] * z[s[i], :]
        end
        y .~ Normal.(r[:, 2], 0.7)
    end
    data = Dict(:y => [0.2 * cos(i) for i in 1:n], :s => [mod1(i, S) for i in 1:n])
    plan = lower_rkppl(ast, data; conditioned = (:y,))
    bound = bind_data(plan, data)
    return bound, build_kernel(bound)
end

@testset "LKJ scalar prior values" begin
    for K in (2, 3), eta in (:e, :(exp(e))), uplo in ('L', 'U')
        bound, built = _lkv_build(K, eta; uplo)
        for e in (0.4, 1.0, 2.3)
            u = [0.25 * sin(i) for i in 1:built.layout.total]
            u[findfirst(==(:e), coordinate_names(built.layout))] = log(e)
            nt = constrain(built.layout, u)
            shape = eta === :e ? nt.e : exp(nt.e)
            factor = Cholesky(uplo === 'U' ? UpperTriangular(nt.L) : LowerTriangular(nt.L))
            prior = logpdf(Exponential(1), nt.e) + logpdf(LKJCholesky(K, shape, uplo), factor)
            location = uplo === 'U' ? nt.L[1, 2] : nt.L[2, 1]
            ll = sum(logpdf.(Normal(location, 0.7), bound.columns[:y]))
            @test uplo === 'U' ? istriu(nt.L) : istril(nt.L)
            @test all(i -> norm(uplo === 'U' ? nt.L[:, i] : nt.L[i, :]) ≈ 1, 1:K)
            @test unconstrain(built.layout, nt) ≈ u
            @test _query(built.spec, bound, :prior, u) ≈ prior atol = 1e-11
            @test _query(built.spec, bound, :likelihood, u) ≈ ll atol = 1e-11
            @test _query(built.spec, bound, :posterior, u) ≈
                prior + ll + logjac(built.layout, u) atol = 1e-11
            _check_gradient(built.spec, bound, u)
        end
    end
    for eta in (:e, :(2 * e))
        bound, built = _lkv_build(3, eta; sampled = false)
        u = [0.15 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        shape = eta === :e ? 1.3 : 2.6
        @test _query(built.spec, bound, :prior, u) ≈
            logpdf(LKJCholesky(3, shape), Cholesky(LowerTriangular(nt.L))) atol = 1e-11
        _check_gradient(built.spec, bound, u)
    end
    # Reject an invalid live value before evaluating its normalizer or
    # diagonal terms; loggamma is undefined at negative integers.
    bound, built = _lkv_build(3, :(e - 2.3); sampled = false)
    @test _query(built.spec, bound, :prior, zeros(built.layout.total)) === -Inf
end

@testset "Shared live LKJ eta across strata" begin
    for (S, n) in ((2, 5), (4, 13)), uplo in ('L', 'U')
        bound, built = _lkv_stack_build(S, n; uplo)
        u = [0.25 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        prior = logpdf(Exponential(1), nt.e) + sum(logpdf.(Normal(), nt.z)) + sum(
            logpdf(LKJCholesky(2, nt.e, uplo), Cholesky(uplo === 'U' ?
                UpperTriangular(nt.L[:, :, k]) : LowerTriangular(nt.L[:, :, k])))
            for k in 1:S)
        @test all(k -> uplo === 'U' ? istriu(nt.L[:, :, k]) : istril(nt.L[:, :, k]), 1:S)
        @test unconstrain(built.layout, nt) ≈ u
        @test _query(built.spec, bound, :prior, u) ≈ prior atol = 1e-11
        ll = sum(logpdf(Normal(dot(nt.L[2, :, bound.columns[:s][i]],
            nt.z[bound.columns[:s][i], :]), 0.7),
            bound.columns[:y][i]) for i in 1:n)
        @test _query(built.spec, bound, :likelihood, u) ≈ ll atol = 1e-11
        _check_gradient(built.spec, bound, u)
    end
end
