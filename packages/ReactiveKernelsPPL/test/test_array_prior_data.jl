using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_apd_data(n = 8, K = 3) = Dict(:y => [0.2 * sin(i) for i in 1:n],
    :k => [mod1(i, K) for i in 1:n])

function _apd_build(ast, data; values = false)
    plan = lower_rkppl(ast, values ? data : keys(data); conditioned = (:y,))
    bound = bind_data(plan, data)
    return (; plan, bound, built = build_kernel(bound), data)
end

function _apd_difference(f, u; h = cbrt(eps(Float64)))
    map(eachindex(u)) do i
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        (f(up) - f(dn)) / (2h)
    end
end

function _apd_check(ast, data, prior, locations; values = false)
    original = deepcopy(data)
    fx = _apd_build(ast, data; values)
    @test fx.bound.n_obs == length(data[:y])
    u = [0.25 * cos(i) for i in 1:fx.built.layout.total]
    function oracle(u)
        th = constrain(fx.built.layout, u)
        likelihood = sum(logpdf.(Normal.(locations(th)[data[:k]], 0.7), data[:y]))
        return prior(th) + likelihood + logjac(fx.built.layout, u)
    end
    th = constrain(fx.built.layout, u)
    query = prepare_query(fx.built, fx.bound, :prior)
    @test Base.invokelatest(query, u) ≈ prior(th)
    q = prepare_sampler(fx.built, fx.bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    @test value ≈ oracle(u)
    @test grad ≈ _apd_difference(oracle, u) rtol = 1e-5 atol = 1e-7
    @test data == original
    return fx
end

@testset "whole data arguments of declared-array priors" begin
    mu0 = [0.2, -0.3, 0.5]
    L = [1.0 0.0 0.0; 0.2 0.8 0.0; -0.1 0.3 1.2]
    S0 = [1.0 0.2; 0.2 0.8]
    alpha = [1.2, 2.1, 0.8]
    for values in (false, true)
        _apd_check(:(begin
            z[1:3] .~ Normal.(mu0, 1)
            y .~ Normal.(z[k], 0.7)
        end), merge(_apd_data(), Dict(:mu0 => mu0)),
            th -> sum(logpdf.(Normal.(mu0, 1), th.z)), th -> th.z; values)
        _apd_check(:(begin
            eachrow(B[levels(k), 1:3]) .~ MvNormalCholesky(mu0, L)
            y .~ Normal.(B[k, 1], 0.7)
        end), merge(_apd_data(), Dict(:mu0 => mu0, :L => L)),
            th -> sum(logpdf(MvNormal(mu0, L * L'), row) for row in eachrow(th.B)),
            th -> th.B[:, 1]; values)
        _apd_check(:(begin
            eachrow(B[levels(k), 1:2]) .~ MvNormal(zeros(2), S0)
            y .~ Normal.(B[k, 1], 0.7)
        end), merge(_apd_data(), Dict(:S0 => S0)),
            th -> sum(logpdf(MvNormal(zeros(2), S0), row) for row in eachrow(th.B)),
            th -> th.B[:, 1]; values)
        _apd_check(:(begin
            eachrow(P[levels(k), 1:3]) .~ Dirichlet(alpha)
            y .~ Normal.(P[k, 1], 0.7)
        end), merge(_apd_data(), Dict(:alpha => alpha)),
            th -> sum(logpdf(Dirichlet(alpha), row) for row in eachrow(th.P)),
            th -> th.P[:, 1]; values)
        # A names-only plan can bind either a shared scalar or an element vector.
        _apd_check(:(begin
            z[1:3] .~ Normal.(mu0, 1)
            y .~ Normal.(z[k], 0.7)
        end), merge(_apd_data(), Dict(:mu0 => 0.2)),
            th -> sum(logpdf.(Normal(0.2, 1), th.z)), th -> th.z; values)
    end
end

@testset "array prior data retains its value through definitions" begin
    mu0 = [0.2, -0.3, 0.5]
    for argument in (:m0, :(mu0 .+ a))
        ast = quote
            a ~ Normal(0, 1)
            m0 = mu0 .+ a
            z[1:3] .~ Normal.($argument, 1)
            y .~ Normal.(z[k], 0.7)
        end
        _apd_check(ast, merge(_apd_data(), Dict(:mu0 => mu0)),
            th -> logpdf(Normal(0, 1), th.a) +
                sum(logpdf.(Normal.(mu0 .+ th.a, 1), th.z)), th -> th.z)
    end
end

@testset "array prior data keeps shape and observation checks" begin
    ast = :(begin
        z[1:3] .~ Normal.(mu0, 1)
        y .~ Normal.(z[k], 0.7)
    end)
    plan = lower_rkppl(ast, (:y, :k, :mu0); conditioned = (:y,))
    # Refused: per-element prior arguments must match the declared array length (§3).
    @test_throws ContractValidationError bind_data(plan,
        merge(_apd_data(), Dict(:mu0 => [0.2, 0.5])))
    # Rebinding the same plan preserves both shared and per-element semantics.
    for mean in (0.2, [0.2, -0.3, 0.5])
        @test bind_data(plan, merge(_apd_data(), Dict(:mu0 => mean))).n_obs == 8
    end
    dual = :(begin
        z[1:3] .~ Normal.(mu0, 1)
        eta = z[k] .+ mu0
        y .~ Normal.(eta, 0.7)
    end)
    aligned = lower_rkppl(dual, (:y, :k, :mu0); conditioned = (:y,))
    # Refused: a value also read per observation must match that observation axis.
    @test_throws ContractValidationError bind_data(aligned,
        merge(_apd_data(), Dict(:mu0 => [0.2, -0.3, 0.5])))
end
