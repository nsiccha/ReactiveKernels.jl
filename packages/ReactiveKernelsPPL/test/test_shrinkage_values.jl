# Explicit shrinkage bodies composed with indexed and monotonic values.
# Every prior is authored; there is no built-in Horseshoe/R2D2 expansion.
using Distributions
using Statistics: var
using ReactiveKernels
using ReactiveKernelsPPL
using Test

@rkppl _sh_horseshoe_levels(g) = begin
    tau ~ HalfCauchy(0.7)
    lambda[levels(g)] .~ HalfCauchy.(1.2)
    z[levels(g)] .~ Normal.(0, 1)
    return tau .* lambda .* z
end

@rkppl _sh_horseshoe_scalar() = begin
    tau ~ HalfCauchy(0.7)
    lambda ~ HalfCauchy(1.2)
    z ~ Normal(0, 1)
    return tau * lambda * z
end

const _SH_INDEXED = quote
    a ~ Normal(0, 1)
    b ~ _sh_horseshoe_levels(g)
    mu = a .+ b[g]
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end

const _SH_MONOTONIC = quote
    a ~ Normal(0, 1)
    b1 ~ Normal(0, 1)
    zeta ~ Dirichlet(alpha_m)
    m ~ monotonic(c, zeta)
    b3 ~ _sh_horseshoe_scalar()
    mu = a .+ b1 .* x1 .+ b3 .* m
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end

# The contrast changes with zeta, so its variance and the conditional
# coefficient scale must be evaluated in the parameter-dependent graph.
# The shipped r2d2_coefs data-matrix helper is a separate contract.
const _SH_R2_MONOTONIC = quote
    a ~ Normal(0, 1)
    R2 ~ Beta(1.2, 2.1)
    phi ~ Dirichlet([1.0, 2.0])
    tau ~ HalfNormal(0.8)
    zeta ~ Dirichlet(alpha_m)
    m ~ monotonic(c, zeta)
    vx = var(x1)
    vm = var(m)
    b1 ~ Normal(0, sqrt(phi[1] * R2 * tau^2 / vx))
    b3 ~ Normal(0, sqrt(phi[2] * R2 * tau^2 / vm))
    mu = a .+ b1 .* x1 .+ b3 .* m
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end

function _sh_case(kind, n, groups)
    y = [0.3 * cos(i) for i in 1:n]
    g = [mod1(i, groups) for i in 1:n]
    if kind == :indexed
        return (; kind, ast = _SH_INDEXED, data = (; y, g))
    end
    x1 = [0.2 * i - 0.5 + 0.1 * cos(i) for i in 1:n]
    alpha_m = [1.0 + i / groups for i in 1:(groups - 1)]
    ast = kind == :monotonic ? _SH_MONOTONIC : _SH_R2_MONOTONIC
    return (; kind, ast, data = (; y, x1, c = g, alpha_m))
end

function _sh_model(case)
    bound = bind_data(lower_rkppl(case.ast, case.data; conditioned = (:y,)),
        Dict{Symbol,ColumnData}(pairs(case.data)))
    return bound, build_kernel(bound)
end

function _sh_expected_names(case)
    names = Symbol[:a, :sigma]
    if case.kind == :indexed
        groups = length(unique(case.data.g))
        push!(names, Symbol("b.tau"))
        append!(names, [Symbol("b.$p.$i") for p in (:lambda, :z) for i in 1:groups])
    else
        push!(names, :b1)
        append!(names, [Symbol("zeta.$i") for i in 1:(length(case.data.alpha_m) - 1)])
        if case.kind == :monotonic
            append!(names, [Symbol("b3.$p") for p in (:tau, :lambda, :z)])
        else
            append!(names, [:b3, :R2, :tau, Symbol("phi.1")])
        end
    end
    return Set(names)
end

# Independent scalar stick-breaking calculation, including its determinant.
# The public layout uses the +log(K-j) offset; do not call its transform.
function _sh_simplex(readu, name, K)
    simplex = zeros(K)
    remaining, jac = 1.0, 0.0
    for j in 1:(K - 1)
        z = 1 / (1 + exp(-(readu(Symbol("$name.$j")) + log(K - j))))
        simplex[j] = remaining * z
        jac += log(remaining) + log(z) + log1p(-z)
        remaining *= 1 - z
    end
    simplex[K] = remaining
    return simplex, jac
end

function _sh_unpack(case, layout, u)
    indices = Dict(name => i for (i, name) in enumerate(coordinate_names(layout)))
    readu(name) = u[indices[name]]
    a, sigma = readu(:a), exp(readu(:sigma))
    jac = readu(:sigma)
    if case.kind == :indexed
        G = length(unique(case.data.g))
        tau = exp(readu(Symbol("b.tau")))
        lambda = [exp(readu(Symbol("b.lambda.$i"))) for i in 1:G]
        z = [readu(Symbol("b.z.$i")) for i in 1:G]
        jac += readu(Symbol("b.tau")) + sum(readu(Symbol("b.lambda.$i")) for i in 1:G)
        return (; a, sigma, b = (; tau, lambda, z)), jac
    end
    b1 = readu(:b1)
    zeta, zj = _sh_simplex(readu, :zeta, length(case.data.alpha_m))
    jac += zj
    if case.kind == :monotonic
        tau = exp(readu(Symbol("b3.tau")))
        lambda = exp(readu(Symbol("b3.lambda")))
        z = readu(Symbol("b3.z"))
        jac += readu(Symbol("b3.tau")) + readu(Symbol("b3.lambda"))
        return (; a, sigma, b1, zeta, b3 = (; tau, lambda, z)), jac
    end
    R2 = 1 / (1 + exp(-readu(:R2)))
    phi, pj = _sh_simplex(readu, :phi, 2)
    tau = exp(readu(:tau))
    jac += log(R2) + log1p(-R2) + pj + readu(:tau)
    return (; a, sigma, b1, zeta, b3 = readu(:b3), R2, phi, tau), jac
end

_sh_halfcauchy(scale, x) = logpdf(truncated(Cauchy(0, scale), 0, Inf), x)
_sh_halfnormal(scale, x) = logpdf(truncated(Normal(0, scale), 0, Inf), x)

function _sh_oracle_parts(case, layout, u)
    q, jac = _sh_unpack(case, layout, u)
    prior = logpdf(Normal(), q.a) + logpdf(Exponential(1), q.sigma)
    if case.kind == :indexed
        prior += _sh_halfcauchy(0.7, q.b.tau) +
            sum(_sh_halfcauchy(1.2, x) for x in q.b.lambda) +
            sum(logpdf(Normal(), x) for x in q.b.z)
        coefficients = q.b.tau .* q.b.lambda .* q.b.z
        mu = [q.a + coefficients[g] for g in case.data.g]
    else
        prior += logpdf(Dirichlet(case.data.alpha_m), q.zeta)
        contrast = [0.0; cumsum(q.zeta)]
        m = [contrast[c] for c in case.data.c]
        if case.kind == :monotonic
            prior += logpdf(Normal(), q.b1) + _sh_halfcauchy(0.7, q.b3.tau) +
                _sh_halfcauchy(1.2, q.b3.lambda) + logpdf(Normal(), q.b3.z)
            b3 = q.b3.tau * q.b3.lambda * q.b3.z
        else
            # Sample variances are recomputed from the current contrast.
            vx = sum((x - sum(case.data.x1) / length(case.data.x1))^2
                for x in case.data.x1) / (length(case.data.x1) - 1)
            vm = sum((x - sum(m) / length(m))^2 for x in m) / (length(m) - 1)
            sd1 = sqrt(q.phi[1] * q.R2 * q.tau^2 / vx)
            sd3 = sqrt(q.phi[2] * q.R2 * q.tau^2 / vm)
            prior += logpdf(Beta(1.2, 2.1), q.R2) +
                logpdf(Dirichlet([1.0, 2.0]), q.phi) + _sh_halfnormal(0.8, q.tau) +
                logpdf(Normal(0, sd1), q.b1) + logpdf(Normal(0, sd3), q.b3)
            b3 = q.b3
        end
        mu = [q.a + q.b1 * x + b3 * mm for (x, mm) in zip(case.data.x1, m)]
    end
    likelihood = sum(logpdf(Normal(mu[i], q.sigma), case.data.y[i])
        for i in eachindex(mu))
    return (; prior, likelihood, jac, posterior = prior + likelihood + jac)
end

@testset "explicit shrinkage on indexed and monotonic values: math and coordinates" begin
    for (n, groups) in ((7, 3), (13, 3), (13, 6)), kind in (:indexed, :monotonic, :r2)
        @testset "$kind / n=$n / groups=$groups" begin
            case = _sh_case(kind, n, groups)
            original = deepcopy(case.data)
            bound, built = _sh_model(case)
            @test case.data == original
            @test Set(coordinate_names(built.layout)) == _sh_expected_names(case)
            bound_before = deepcopy(bound.columns)
            for shift in (0.0, 0.27)
                u = [0.2 * sin(i) + shift for i in 1:built.layout.total]
                q, jac = _sh_unpack(case, built.layout, u)
                constrained = constrain(built.layout, u)
                for name in keys(q)
                    value = getproperty(q, name)
                    if value isa NamedTuple
                        for part in keys(value)
                            @test getproperty(getproperty(constrained, name), part) ≈ getproperty(value, part)
                        end
                    else
                        @test getproperty(constrained, name) ≈ value
                    end
                end
                @test unconstrain(built.layout, constrained) ≈ u
                @test logjac(built.layout, u) ≈ jac
                parts = _sh_oracle_parts(case, built.layout, u)
                @test _query(built.spec, bound, :prior, u) ≈ parts.prior
                @test _query(built.spec, bound, :likelihood, u) ≈ parts.likelihood
                _check_model_math(built, bound, u, w -> _sh_oracle_parts(case, built.layout, w).posterior)
            end
            @test bound.columns == bound_before
            @test case.data == original
            if kind == :r2
                @test !haskey(bound.columns, :m)
                @test !haskey(bound.columns, :vm)
                u = zeros(built.layout.total)
                v = copy(u)
                v[only(findall(==(Symbol("zeta.1")), coordinate_names(built.layout)))] = 0.8
                @test _sh_oracle_parts(case, built.layout, u).prior !=
                    _sh_oracle_parts(case, built.layout, v).prior
                @test _query(built.spec, bound, :prior, v) ≈
                    _sh_oracle_parts(case, built.layout, v).prior
            end
        end
    end
end
