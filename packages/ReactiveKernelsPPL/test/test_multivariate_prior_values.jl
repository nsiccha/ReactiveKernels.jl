using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels,
    ReactiveKernelsPPL, Test

# A multivariate prior reads each argument as a value, so a computed mean,
# covariance or factor means its named definition (rkppl-use §3). The cases
# are a squared-exponential Gaussian-process regression: a covariance built
# from a data matrix of squared distances plus a `UniformScaling` noise
# diagonal, with the mean, covariance and factor written inline or named.

const _MP_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_mp_x(n) = collect(range(-1.5, 2.0; length = n))
_mp_D(n) = (_mp_x(n) .- _mp_x(n)') .^ 2
_mp_y(n) = sin.(_mp_x(n)) .+ 0.1 .* cos.(3 .* _mp_x(n))
_mp_u(built) = [0.2cos(i + 0.4) for i in 1:built.layout.total]
_mp_K(D, nt) = nt.alpha^2 .* exp.(-0.5 .* D ./ nt.rho^2) + nt.sigma * I

function _mp_fd(f, u; h = 1e-6)
    return [(f(u .+ h .* (eachindex(u) .== i)) - f(u .- h .* (eachindex(u) .== i))) / 2h
        for i in eachindex(u)]
end

_mp_model(body) = Core.eval(@__MODULE__, Expr(:macrocall, Symbol("@rkppl"),
    LineNumberNode(@__LINE__, Symbol(@__FILE__)), body))

const _MP_PRIORS = quote
    rho ~ LogNormal(0, 0.5)
    alpha ~ HalfNormal(2)
    sigma ~ HalfNormal(1)
end

_mp_prior(nt) = logpdf(LogNormal(0, 0.5), nt.rho) +
    logpdf(truncated(Normal(0, 2), 0, Inf), nt.alpha) +
    logpdf(truncated(Normal(0, 1), 0, Inf), nt.sigma)

# The sampler value against the independent oracle `density(constrained
# values)` plus the transform's log Jacobian, the native Enzyme gradient
# against central differences, and the caller's data left unchanged.
function _mp_check(plan, data, density; backend = _MP_BACKEND)
    saved = deepcopy(data)
    built = build_kernel(plan)
    u = _mp_u(built)
    oracle(v) = density(constrain(built.layout, v)) + logjac(built.layout, v)
    q = prepare_sampler(built, plan, u; backend)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ oracle(u) rtol = 1e-10
    @test g ≈ _mp_fd(oracle, u) rtol = 1e-5 atol = 1e-6
    @test data == saved
    return built
end

@testset "multivariate priors: computed covariance, mean and factor arguments" begin
    n = 8
    D, y = _mp_D(n), _mp_y(n)
    observed(nt) = _mp_prior(nt) + logpdf(MvNormal(zeros(n), Symmetric(_mp_K(D, nt))), y)
    spellings = [
        # The covariance's data matrix is read only for the declared axis and
        # inside the covariance: a model-level matrix, not an observation column.
        quote
            K = alpha^2 .* exp.(-0.5 .* D ./ rho^2) + sigma * I
            y[axes(D, 1)] ~ MvNormal(zeros(N), K)
        end,
        quote
            mu = zeros(N)
            C = alpha^2 .* exp.(-0.5 .* D ./ rho^2)
            K = C + sigma * I
            y[axes(D, 1)] ~ MvNormal(mu, K)
        end,
        # Inline arguments, a qualified constant and a shape read of the
        # declaration itself, which reads its declared axis.
        quote
            y[axes(D, 1)] ~ MvNormal(zeros(length(y)),
                alpha^2 .* exp.(-0.5 .* D ./ rho^2) + sigma * LinearAlgebra.I)
        end,
        quote
            K = alpha^2 .* exp.(-0.5 .* D ./ rho^2) .+ sigma .* Id
            y[axes(D, 1)] ~ MvNormal(zeros(N), K)
        end,
        quote
            K = alpha^2 .* exp.(-0.5 .* D ./ rho^2) + sigma * I
            y[axes(D, 1)] ~ MvNormalCholesky(zeros(N), rk_cholesky_lower(K))
        end,
        # A property of a value is `getproperty`, inline or named.
        quote
            K = alpha^2 .* exp.(-0.5 .* D ./ rho^2) + sigma * I
            y[axes(D, 1)] ~ MvNormalCholesky(zeros(N), cholesky(Symmetric(K)).L)
        end,
        quote
            K = alpha^2 .* exp.(-0.5 .* D ./ rho^2) + sigma * I
            L = cholesky(K).L
            y[axes(D, 1)] ~ MvNormalCholesky(zeros(size(D, 1)), L)
        end,
    ]
    for body in spellings
        m = _mp_model(Expr(:block, _MP_PRIORS.args..., body.args...))
        data = (; D, N = n, Id = Matrix(1.0I, n, n))
        built = _mp_check(m(; data...) | (; y), merge(data, (; y)), observed)
        @test coordinate_names(built.layout) == [:rho, :alpha, :sigma]
    end
end

@testset "multivariate priors: a latent draw beside observations of another length" begin
    n, m = 6, 9
    D = _mp_D(n)
    idx = [mod1(3i, n) for i in 1:m]
    z = [0.3sin(i) for i in 1:m]
    model = @rkppl begin
        rho ~ LogNormal(0, 0.5)
        alpha ~ HalfNormal(2)
        sigma ~ HalfNormal(1)
        s ~ Exponential(1)
        K = alpha^2 .* exp.(-0.5 .* D ./ rho^2) + sigma * I
        f[axes(D, 1)] ~ MvNormal(zeros(N), K)
        z .~ Normal.(f[idx], s)
    end
    density(nt) = _mp_prior(nt) + logpdf(Exponential(1), nt.s) +
        logpdf(MvNormal(zeros(n), Symmetric(_mp_K(D, nt))), nt.f) +
        sum(logpdf.(Normal.(nt.f[idx], nt.s), z))
    data = (; D, N = n, idx)
    built = _mp_check(model(; data...) | (; z), merge(data, (; z)), density)
    @test length(constrain(built.layout, _mp_u(built)).f) == n
end

# A non-centered latent GP: the latent values are a lower Cholesky factor
# times standard-normal draws, observed through Poisson counts. The constant
# jitter `1e-9 * I` is bound-only, so partial evaluation folds the
# `UniformScaling{Float64}` into the prepared callable's operation table, and
# plain `AutoEnzyme` must prove that callable readonly. Enzyme 0.13.216 and
# the dogfood line reject that proof at the store that initializes Enzyme's
# own by-value copy of the callable (`innerparm`), not at any write in the
# model: `benchmark/repro_enzyme_innerparm_split_store_readonly.jl` reproduces
# it with Enzyme only. `Const(f)` differentiates the same call exactly, so the
# annotated backend checks the model on every Enzyme; the plain backend
# accepts only the exact gradient or that specific readonly diagnostic.
@testset "multivariate priors: a non-centered latent GP with a constant jitter" begin
    n = 6
    D = _mp_D(n)
    k = [3, 1, 0, 2, 5, 4]
    spellings = [
        quote
            L = rk_cholesky_lower(K)
        end,
        quote
            L = cholesky(Symmetric(K)).L
        end,
    ]
    function density(nt)
        K = nt.alpha^2 .* exp.(-0.5 .* D ./ nt.rho^2) + 1e-9 * I
        f = cholesky(Symmetric(K)).L * nt.z
        logpdf(LogNormal(0, 0.5), nt.rho) +
            logpdf(truncated(Normal(0, 2), 0, Inf), nt.alpha) +
            sum(logpdf.(Normal(0, 1), nt.z)) + sum(logpdf.(Poisson.(exp.(f)), k))
    end
    data = (; D, N = n)
    for factor in spellings
        m = _mp_model(quote
            rho ~ LogNormal(0, 0.5)
            alpha ~ HalfNormal(2)
            z[1:N] .~ Normal.(0, 1)
            K = alpha^2 .* exp.(-0.5 .* D ./ rho^2) + 1e-9 * I
            $(factor.args...)
            f = L * z
            k .~ PoissonLog.(f)
        end)
        plan = m(; data...) | (; k)
        built = _mp_check(plan, merge(data, (; k)), density;
            backend = AutoEnzyme(; mode = Enzyme.Reverse,
                                 function_annotation = Enzyme.Const))
        @test coordinate_names(built.layout) ==
            [:rho; :alpha; [Symbol("z.", i) for i in 1:n]]
        outcome = try
            _mp_check(plan, merge(data, (; k)), density)
        catch err
            err
        end
        if outcome isa Enzyme.Compiler.EnzymeMutabilityException
            @test occursin("Function argument passed to autodiff cannot be proven readonly",
                           sprint(showerror, outcome))
            # Upstream Enzyme readonly-proof gap (snag readonly-proof-f-34071c89);
            # drop with docs/src/constraints.md once a registered Enzyme repairs it.
            @test_broken false
        elseif outcome isa Exception
            throw(outcome)
        end
    end
end

@testset "multivariate priors: shared computed slice arguments" begin
    M = [0.2 -0.1; 0.0 0.3; -0.4 0.1]
    S0 = [1.0 0.3; 0.3 0.8]
    k = [1, 2, 3, 2, 1, 3]
    y = [0.1sin(i) for i in eachindex(k)]
    density(nt) = logpdf(Exponential(1), nt.s2) +
        sum(logpdf(MvNormal(M[j, :], nt.s2 .* S0), nt.B[j, :]) for j in 1:3) +
        sum(logpdf.(Normal.(nt.B[k, 1], 0.7), y))
    inline = @rkppl begin
        s2 ~ Exponential(1)
        eachrow(B[1:3, 1:2]) .~ MvNormal.(eachrow(M), Ref(s2 .* S0))
        y .~ Normal.(B[k, 1], 0.7)
    end
    named = @rkppl begin
        s2 ~ Exponential(1)
        S = s2 .* S0
        eachrow(B[1:3, 1:2]) .~ MvNormal.(eachrow(M), Ref(S))
        y .~ Normal.(B[k, 1], 0.7)
    end
    data = (; M, S0, k)
    for model in (inline, named)
        _mp_check(model(; data...) | (; y), merge(data, (; y)), density)
    end
end

@testset "multivariate priors: native solve and factor equal the traced forms" begin
    P = ReactiveKernelsPPL
    S = [2.0 0.4 -0.2; 0.4 1.5 0.3; -0.2 0.3 1.1]
    generic_factor(A, k) = invoke(P._covariance_cholesky, Tuple{Any,Any}, A, k)
    @test P._covariance_cholesky(S, 3) ≈ generic_factor(S, 3) rtol = 1e-14
    @test P._covariance_cholesky(S, 3) ≈ cholesky(S).L rtol = 1e-14
    @test P._covariance_cholesky(fill(4.0, 1, 1), 1) == [2.0;;]
    F = P._covariance_cholesky(S, 3)
    generic_solve(X, F, d) = invoke(P._lower_solve_rows_logpdf, Tuple{Any,Any,Any}, X, F, d)
    for X in ([0.3 -0.2 0.5], [0.3 -0.2 0.5; 0.1 0.4 -0.6], zeros(0, 3))
        @test P._lower_solve_rows_logpdf(X, F, 0.7) ≈ generic_solve(X, F, 0.7) rtol = 1e-14
    end
    # refused: a covariance must be symmetric positive definite (distribution domain, P3)
    @test_throws ArgumentError P._covariance_cholesky([1.0 0.5; 0.0 1.0], 2)
    # refused: a covariance must be symmetric positive definite (distribution domain, P3)
    @test_throws ArgumentError P._covariance_cholesky([1.0 2.0; 2.0 1.0], 2)
    # refused: the factor's size must match the slice length (Julia's DimensionMismatch)
    @test_throws DimensionMismatch P._lower_solve_rows_logpdf([0.3 -0.2], F, 0.0)
end
