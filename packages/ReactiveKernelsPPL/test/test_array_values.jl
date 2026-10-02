using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Declared array parameters (`ArrayParameter`): `L ~ LKJCholesky(K, eta)`,
# sized `z[1:K] .~ ...` / `z[levels(g)] .~ ...` / two-axis
# `z[levels(g), 1:K] .~ ...`, and bound data matrices, all read as plain
# values (`L[2, 1]`, `z[1]`, `z[g]`, `B * w`); a free simplex (a
# `VectorParameter` read as a model-level value) is read by position too.
# Densities are checked against Distributions.jl, gradients against
# central differences. Helpers `_query` / `_check_gradient` come from
# test_generator.jl (included earlier).

_av_y() = [0.3, -1.2, 2.1, 0.7, -0.4, 1.5, 0.2, -0.8]
_av_x() = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 2.0, -1.5]
_av_x2() = [1.0, 0.25, -0.75, 2.0, -1.0, 0.5, 0.0, 1.25]
_av_g() = ["b", "a", "c", "a", "b", "c", "a", "b"]
_av_B() = hcat(_av_x(), _av_x2(), ones(8))

# Fixed off-origin unconstrained points (no randomness in the oracles).
_av_point(n) = [0.37 * sin(1.3 * i) - 0.2 for i in 1:n]

_av_node(built, bound, node, u) = _query(built.spec, bound, node, u)

@testset "array values: LKJCholesky factor read as a matrix" begin
    m = @rkppl begin
        L ~ LKJCholesky(3, 2.0)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        w = L[2, 1] .* x .+ L[3, 2] .* x2
        mu = a .+ w
        y .~ Normal.(mu, sigma)
    end
    y, x, x2 = _av_y(), _av_x(), _av_x2()
    bound = m(; y, x, x2)
    @test only(bound.array_parameters).family === :lkj_cholesky
    built = build_kernel(bound)
    @test built.layout.total == 3 + 2
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    L = nt.L
    @test L isa Matrix{Float64} && size(L) == (3, 3)
    @test istril(L)
    @test all(i -> norm(L[i, :]) ≈ 1, 1:3)
    @test unconstrain(built.layout, nt) ≈ u
    a = nt.a  # the intercept coefficient of `mu`
    prior = logpdf(LKJCholesky(3, 2.0), Cholesky(LowerTriangular(L))) +
        logpdf(Normal(0, 1), a) + logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    lik = sum(logpdf.(Normal.(a .+ L[2, 1] .* x .+ L[3, 2] .* x2,
        nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
end

@testset "array values: LKJCholesky density matches Distributions" begin
    # The prior node alone (no other parameters) over several K and eta,
    # including eta == 1 (the uniform-correlation branch) and K == 1.
    for (K, eta) in ((1, 2.0), (2, 1.0), (2, 0.7), (3, 1.0), (4, 3.5))
        ast = :(begin
            L ~ LKJCholesky($K, $eta)
            a ~ Normal(0, 1)
            sigma ~ Exponential(1)
            w = L[$K, 1] .* x
            mu = a .+ w
            y .~ Normal.(mu, sigma)
        end)
        bound = bind_data(lower_rkppl(ast, (:y, :x)),
            Dict{Symbol,ColumnData}(:y => _av_y(), :x => _av_x()))
        built = build_kernel(bound)
        u = _av_point(built.layout.total)
        nt = constrain(built.layout, u)
        want = logpdf(LKJCholesky(K, eta), Cholesky(LowerTriangular(nt.L))) +
            logpdf(Normal(0, 1), nt.a) +
            logpdf(Exponential(1), nt.sigma)
        @test _av_node(built, bound, :prior, u) ≈ want
    end
end

@testset "array values: sized vector with per-element scales and B * w" begin
    m = @rkppl begin
        tau ~ HalfNormal(1)
        lambda[1:3] .~ HalfCauchy.(1)
        w[axes(B, 2)] .~ Normal.(0, lambda .* tau)
        s[1:3] .~ Normal.([0.0, 1.0, -1.0], [1.0, 2.0, 0.5])
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        v = s[2] .* x
        mu = a .+ B * w .+ v
        y .~ Normal.(mu, sigma)
    end
    y, x, B = _av_y(), _av_x(), _av_B()
    bound = m(; y, x, B)
    @test Set(p.name for p in bound.array_parameters) == Set([:lambda, :w, :s])
    built = build_kernel(bound)
    @test built.layout.total == 1 + 3 + 3 + 3 + 2
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test nt.lambda isa Vector{Float64} && length(nt.lambda) == 3
    @test all(>(0), nt.lambda)
    @test unconstrain(built.layout, nt) ≈ u
    prior = logpdf(truncated(Normal(0, 1), 0, Inf), nt.tau) +
        sum(logpdf.(truncated.(Cauchy.(0, 1), 0, Inf), nt.lambda)) +
        sum(logpdf.(Normal.(0, nt.lambda .* nt.tau), nt.w)) +
        sum(logpdf.(Normal.([0.0, 1.0, -1.0], [1.0, 2.0, 0.5]), nt.s)) +
        logpdf(Normal(0, 1), nt.a) + logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    lik = sum(logpdf.(Normal.(nt.a .+ B * nt.w .+ nt.s[2] .* x,
        nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
end

@testset "array values: free simplex read by position" begin
    # The simplex stays a VectorParameter (a model-level value, as for
    # `cumsum(phi)`); reading it by position needs no array parameter.
    m = @rkppl begin
        phi ~ Dirichlet([2.0, 1.0, 3.0])
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1)
        w = phi[1] .* x .+ phi[2] .* x2 .+ phi[3]
        mu = a .+ b .* w
        y .~ Normal.(mu, sigma)
    end
    y, x, x2 = _av_y(), _av_x(), _av_x2()
    bound = m(; y, x, x2)
    @test isempty(bound.array_parameters)
    @test only(bound.vector_parameters).name === :phi
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test sum(nt.phi) ≈ 1 && all(>(0), nt.phi)
    prior = logpdf(Dirichlet([2.0, 1.0, 3.0]), nt.phi) +
        logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 2), nt.b) +
        logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    w = nt.phi[1] .* x .+ nt.phi[2] .* x2 .+ nt.phi[3]
    lik = sum(logpdf.(Normal.(nt.a .+ nt.b .* w, nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
end

@testset "array values: level gather and a two-axis array" begin
    # Non-centered varying intercept and a correlated two-margin effect:
    # `z[g]` / `Z[g, j]` look each observation's level up on the
    # `levels(g)` axis (sorted: "a", "b", "c").
    m = @rkppl begin
        tau ~ HalfNormal(1)
        z[levels(g)] .~ Normal.(0, 1)
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ HalfNormal.(1)
        Z[levels(g), 1:2] .~ Normal.(0, 1)
        M = (sd .* L)'
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        r = tau .* z[g] .+ Z[g, :] * M[:, 1] .+ (Z[g, :] * M[:, 2]) .* x
        mu = a .+ r
        y .~ Normal.(mu, sigma)
    end
    y, x, g = _av_y(), _av_x(), _av_g()
    bound = m(; y, x, g)
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test size(nt.Z) == (3, 2)
    @test unconstrain(built.layout, nt) ≈ u
    codes = [findfirst(==(v), ["a", "b", "c"]) for v in g]
    M = (nt.sd .* nt.L)'
    @test M ≈ (Diagonal(nt.sd) * nt.L)'
    r = nt.tau .* nt.z[codes] .+ nt.Z[codes, :] * M[:, 1] .+
        (nt.Z[codes, :] * M[:, 2]) .* x
    lik = sum(logpdf.(Normal.(nt.a .+ r, nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    prior = logpdf(truncated(Normal(0, 1), 0, Inf), nt.tau) +
        sum(logpdf.(Normal(0, 1), nt.z)) +
        logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(nt.L))) +
        sum(logpdf.(truncated(Normal(0, 1), 0, Inf), nt.sd)) +
        sum(logpdf.(Normal(0, 1), nt.Z)) +
        logpdf(Normal(0, 1), nt.a) + logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
    # Coordinates follow Stan's matrix naming (`Z.i.j`, column-major).
    names = coordinate_names(built.layout)
    @test Symbol("Z.2.1") in names && Symbol("Z.1.2") in names
    @test findfirst(==(Symbol("Z.2.1")), names) <
        findfirst(==(Symbol("Z.1.2")), names)
end

@testset "array values: an integer axis gathers by position" begin
    m = @rkppl begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ z[k]
        y .~ Normal.(mu, sigma)
    end
    k = [1, 2, 3, 1, 2, 3, 3, 2]
    bound = m(; y = _av_y(), k)
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    lik = sum(logpdf.(Normal.(nt.a .+ nt.z[k], nt.sigma), _av_y()))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    _check_gradient(built.spec, bound, u)
end

@testset "array values: fail closed" begin
    bindm(ast, data) = bind_data(lower_rkppl(ast, Tuple(keys(data))),
        Dict{Symbol,ColumnData}(pairs(data)))
    y, x = _av_y(), _av_x()
    # `~` on a sized declaration.
    @test_throws SurfaceLoweringError lower_rkppl(:(begin
        z[1:3] ~ Normal(0, 1)
        y .~ Normal.(z[1], 1.0)
    end), (:y,))
    # A bare array combined with per-observation data.
    @test_throws SurfaceLoweringError lower_rkppl(:(begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        w = z .+ x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x))
    # Undotted elementwise math over an array is a Julia error.
    @test_throws SurfaceLoweringError lower_rkppl(:(begin
        z[1:3] .~ Normal.(0, 1)
        s ~ Exponential(1)
        v = s + z
        a ~ Normal(0, 1)
        w = v[1] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x))
    # LKJCholesky: non-literal eta, upper factor.
    @test_throws SurfaceLoweringError lower_rkppl(:(begin
        e ~ Exponential(1)
        L ~ LKJCholesky(2, e)
        a ~ Normal(0, 1)
        w = L[2, 1] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(:(begin
        L ~ LKJCholesky(2, 2.0, 'U')
        a ~ Normal(0, 1)
        w = L[2, 1] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x))
    # A data matrix whose columns do not match the array.
    @test_throws ContractValidationError bindm(:(begin
        w[1:2] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ B * w
        y .~ Normal.(mu, 1.0)
    end), (; y, B = _av_B()))
    # A vector where `B * w` needs a matrix.
    @test_throws ContractValidationError bindm(:(begin
        w[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ B * w
        y .~ Normal.(mu, 1.0)
    end), (; y, B = x))
    # Out-of-bounds literal position.
    @test_throws ContractValidationError bindm(:(begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        w = z[4] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (; y, x))
    # A gather by values not on the levels axis.
    @test_throws ContractValidationError bindm(:(begin
        z[levels(g)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        w = z[h] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (; y, x, g = _av_g(), h = ["a", "b", "z", "a", "b", "c", "a", "b"]))
    # An integer axis gathered by non-integers.
    @test_throws ContractValidationError bindm(:(begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[g]
        y .~ Normal.(mu, 1.0)
    end), (; y, g = _av_g()))
    # Per-element literal arguments of the wrong length.
    @test_throws ContractValidationError bindm(:(begin
        z[1:3] .~ Normal.(0, [1.0, 2.0])
        a ~ Normal(0, 1)
        w = z[1] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (; y, x))
end
