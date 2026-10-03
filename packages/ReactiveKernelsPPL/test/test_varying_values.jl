using Distributions
using LinearAlgebra
using ReactiveKernelsPPL
using Test

# Explicit bodies replace the legacy margin/grouping templates. Priors and
# parameterization are the same for membership and stratified uses.
@rkppl _cv_coefs(g, eta) = begin
    sd[1:2] .~ Gamma.(2, 1)
    L ~ LKJCholesky(2, eta)
    z[levels(g), 1:2] .~ Normal.(0, 1)
    return z * (sd .* L)'
end

@rkppl _cv_strata(g, s, eta) = begin
    sd[levels(s), 1:2] .~ Gamma.(2, 1)
    @plate for k in levels(s)
        L[k] ~ LKJCholesky(2, eta)
    end
    z[levels(g), 1:2] .~ Normal.(0, 1)
    @plate for i in eachindex(g)
        b[i, 1:2] = (sd[s[i], :] .* L[s[i]]) * z[g[i], :]
    end
    return b
end

@rkppl _cv_distinct(g, eta) = begin
    sd1 ~ Gamma(2, 1)
    sd2 ~ HalfCauchy(2)
    sd = vcat(sd1, sd2)
    L ~ LKJCholesky(2, eta)
    z[levels(g), 1:2] .~ Normal.(0, 1)
    return z * (sd .* L)'
end

@rkppl _cv_mmstrata(g, h, s, w1, w2, eta) = begin
    gg = vcat(g, h)
    sd[levels(s), 1:2] .~ Gamma.(2, 1)
    @plate for k in levels(s)
        L[k] ~ LKJCholesky(2, eta)
    end
    z[levels(gg), 1:2] .~ Normal.(0, 1)
    @plate for i in eachindex(g)
        F = sd[s[i], :] .* L[s[i]]
        b[i, 1:2] = (w1[i] * (F * z[g[i], :]) +
            w2[i] * (F * z[h[i], :])) / (w1[i] + w2[i])
    end
    return b
end

function _cv_program(kind, n, S)
    head = quote
        eta ~ Exponential(1)
        a ~ Normal(0, 1)
    end
    body = kind in (:membership, :distinct) ? quote
            gg = vcat(g, h)
            d ~ $(kind === :distinct ? :_cv_distinct : :_cv_coefs)(gg, eta)
            r1 = (w1 .* d[g, 1] .+ w2 .* d[h, 1]) ./ (w1 .+ w2)
            r2 = (w1 .* d[g, 2] .+ w2 .* d[h, 2]) ./ (w1 .+ w2)
        end : kind === :single ? quote
            d ~ _cv_coefs(g, eta)
            r1 = d[g, 1]
            r2 = d[g, 2]
        end : kind === :both ? quote
            d ~ _cv_mmstrata(g, h, s, w1, w2, eta)
            r1 = d[:, 1]
            r2 = d[:, 2]
        end : quote
            d ~ _cv_strata(g, s, eta)
            r1 = d[:, 1]
            r2 = d[:, 2]
        end
    tail = quote
        w = a .* x
        mu = r1 .+ w .* r2
        nu = .-r1 .+ (x .* w) .* r2
        y .~ Normal.(mu, 0.7)
        y2 .~ Normal.(nu, 1.1)
    end
    ast = Expr(:block, head.args..., body.args..., tail.args...)
    data = Dict{Symbol,Any}(:y => [0.2 * cos(i) for i in 1:n],
        :y2 => [0.3 * sin(i) for i in 1:n], :x => [0.1i for i in 1:n],
        :g => [mod1(i, S + 1) for i in 1:n])
    if kind in (:membership, :distinct, :both)
        data[:h] = [mod1(i + 1, S + 1) for i in 1:n]
        data[:w1] = [0.2 + 0.1 * isodd(i) for i in 1:n]
        data[:w2] = 1 .- data[:w1]
    end
    kind in (:stratified, :both) && (data[:s] = [mod1(i, S) for i in 1:n])
    return ast, data
end

function _cv_bound(kind, n, S)
    ast, data = _cv_program(kind, n, S)
    plan = lower_rkppl(ast, data; mod = @__MODULE__, conditioned = (:y, :y2))
    return bind_data(plan, data)
end

function _cv_build(kind, n, S)
    bound = _cv_bound(kind, n, S)
    return bound, build_kernel(bound)
end

# The same membership math with a factor dimension derived from bound data.
# That dimension must keep its prior reduction; the literal-pair scalar
# equation must not specialize it merely because the resolved size is two.
function _cv_data_width_build()
    _, data = _cv_program(:membership, 7, 2)
    ast = quote
        eta ~ Exponential(1)
        a ~ Normal(0, 1)
        gg = vcat(g, h)
        sd[1:2] .~ Gamma.(2, 1)
        L ~ LKJCholesky(size(M, 1), eta)
        z[levels(gg), 1:2] .~ Normal.(0, 1)
        d = z * (sd .* L)'
        r1 = (w1 .* d[g, 1] .+ w2 .* d[h, 1]) ./ (w1 .+ w2)
        r2 = (w1 .* d[g, 2] .+ w2 .* d[h, 2]) ./ (w1 .+ w2)
        w = a .* x
        mu = r1 .+ w .* r2
        nu = .-r1 .+ (x .* w) .* r2
        y .~ Normal.(mu, 0.7)
        y2 .~ Normal.(nu, 1.1)
    end
    data[:M] = zeros(2, 1)
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y, :y2)), data)
    return bound, build_kernel(bound)
end

function _cv_oracle(bound, nt, kind)
    c = bound.columns
    sd = kind === :distinct ? [nt.d.sd1, nt.d.sd2] : nt.d.sd
    sdprior = kind === :distinct ? logpdf(Gamma(2, 1), nt.d.sd1) +
        logpdf(truncated(Cauchy(0, 2), 0, Inf), nt.d.sd2) :
        sum(logpdf.(Gamma(2, 1), sd))
    prior = logpdf(Exponential(1), nt.eta) + logpdf(Normal(), nt.a) +
        sdprior + sum(logpdf.(Normal(), nt.d.z))
    if kind in (:membership, :distinct, :single)
        prior += logpdf(LKJCholesky(2, nt.eta), Cholesky(LowerTriangular(nt.d.L)))
        B = nt.d.z * (Diagonal(sd) * nt.d.L)'
        R = kind === :single ? B[c[:g], :] :
            (c[:w1] .* B[c[:g], :] .+ c[:w2] .* B[c[:h], :]) ./ (c[:w1] .+ c[:w2])
    else
        prior += sum(logpdf(LKJCholesky(2, nt.eta),
            Cholesky(LowerTriangular(nt.d.L[:, :, k]))) for k in axes(nt.d.L, 3))
        R = reduce(vcat, [permutedims((Diagonal(nt.d.sd[c[:s][i], :]) *
            nt.d.L[:, :, c[:s][i]]) * (kind === :both ?
                (c[:w1][i] .* nt.d.z[c[:g][i], :] .+
                 c[:w2][i] .* nt.d.z[c[:h][i], :]) ./ (c[:w1][i] + c[:w2][i]) :
                nt.d.z[c[:g][i], :])) for i in eachindex(c[:y])])
    end
    w = nt.a .* c[:x]
    mu = R[:, 1] .+ w .* R[:, 2]
    nu = .-R[:, 1] .+ (c[:x] .* w) .* R[:, 2]
    likelihood = sum(logpdf.(Normal.(mu, 0.7), c[:y])) +
        sum(logpdf.(Normal.(nu, 1.1), c[:y2]))
    return prior, likelihood
end

@testset "Varying values: explicit membership and stratified bodies" begin
    for kind in (:membership, :distinct, :stratified, :single, :both), (n, S) in ((7, 2), (19, 4))
        bound, built = _cv_build(kind, n, S)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        prior, likelihood = _cv_oracle(bound, nt, kind)
        @test _query(built.spec, bound, :prior, u) ≈ prior atol = 1e-10
        @test _query(built.spec, bound, :likelihood, u) ≈ likelihood atol = 1e-10
        @test unconstrain(built.layout, nt) ≈ u
        @test hasproperty(nt, :d) && hasproperty(nt.d, :L) &&
            (kind === :distinct ? hasproperty(nt.d, :sd1) &&
                hasproperty(nt.d, :sd2) : hasproperty(nt.d, :sd))
        _check_gradient(built.spec, bound, u)
    end
end

@testset "Varying values: data-sized shared factor" begin
    bound, built = _cv_data_width_build()
    u = [0.2 * sin(i) for i in 1:built.layout.total]
    nt = constrain(built.layout, u)
    reference = (; nt.eta, nt.a, d = (; nt.sd, nt.L, nt.z))
    prior, likelihood = _cv_oracle(bound, reference, :membership)
    @test _query(built.spec, bound, :prior, u) ≈ prior atol = 1e-10
    @test _query(built.spec, bound, :likelihood, u) ≈ likelihood atol = 1e-10
    @test unconstrain(built.layout, nt) ≈ u
    _check_gradient(built.spec, bound, u)
end
