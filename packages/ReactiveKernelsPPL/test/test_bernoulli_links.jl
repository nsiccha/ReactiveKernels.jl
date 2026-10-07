# Bernoulli links end to end (SB-parity pair `fam-bernoulli-links`, RK side):
# sampler-query values vs Distributions oracles and native Enzyme gradients
# vs central differences; test_bernoulli_links_reactant.jl adds Reactant/XLA
# primal + compiled-gradient parity and the traced program size. Binomial
# probit/cloglog twins ride the same generator path and are locked at n=8.
# (`_GEN_BACKEND` comes from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: Bernoulli, Binomial, Normal, cdf, logpdf
using Enzyme
using Random: Xoshiro, rand, randn
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Lower + bind + build + query a links program; return
# `(bound, built, kern)`.
function _links_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern
end

function _links_findiff(f, u; h = cbrt(eps(Float64)))
    g = similar(u, Float64)
    for i in eachindex(u)
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

const _LINKS_X8 = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.2, -0.3]
const _LINKS_Y8 = [false, true, false, true, true, false, true, false]

# Agreed parity dataset with the BRM peer (todo 0qy4gep on
# BayesianRegressionModels:rk:parity-fam-bernoulli-links): regenerated
# verbatim here, with checksums gated before any pin (an RNG-stream drift
# fails loudly instead of blessing shifted pins).
function _bernoulli_links_data()
    rng = Xoshiro(2609)
    x = randn(rng, 100)
    eta = 0.5 .- 0.25 .* x
    s = [rand(rng) < 1 / (1 + exp(-e)) for e in eta]
    return x, Int.(s)
end

# n=100 programs per Bernoulli link plus the independent oracle row.
function _links_parity_progs()
    return [
        ("logit", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.(logistic.(eta))
        end, (eta, yy) -> sum(logpdf.(Bernoulli.(1 ./ (1 .+ exp.(-eta))), yy))),
        ("probit", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.(normcdf.(eta))
        end, (eta, yy) -> sum(logpdf.(Bernoulli.(cdf.(Ref(Normal()), eta)), yy))),
        ("cloglog", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.(cexpexp.(eta))
        end, (eta, yy) -> sum(logpdf.(Bernoulli.(1 .- exp.(-exp.(eta))), yy))),
    ]
end

# Binomial-twin stretch dataset (same Xoshiro(2609) stream as the
# Bernoulli data — identical x, then per-link Binomial draws in cl/cp/cc
# stream order), with checksums gated before any pin.
function _bernoulli_links_binom_data()
    rng = Xoshiro(2609)
    x = randn(rng, 100)
    eta = 0.5 .- 0.25 .* x
    n = fill(5, 100)
    pl = 1 ./ (1 .+ exp.(-eta))
    pp = cdf.(Ref(Normal()), eta)
    pc = 1 .- exp.(-exp.(eta))
    cl = [rand(rng, Binomial(nn, p)) for (nn, p) in zip(n, pl)]
    cp = [rand(rng, Binomial(nn, p)) for (nn, p) in zip(n, pp)]
    cc = [rand(rng, Binomial(nn, p)) for (nn, p) in zip(n, pc)]
    return x, n, cl, cp, cc
end

_links_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T14-21-56-212-19fcsp1 on
# BayesianRegressionModels:rk:parity-fam-bernoulli-links, BRM 61f2a22,
# StanBlocks 24578c3, BridgeStan 2.9.0): full posterior at u_unc,
# propto=false, no Jacobian (unconstrained-reals-only models), BridgeStan
# AD grads. Pins compare by coordinate name (SB pins below are in SB
# declaration order [b0, b1]).
@testset "bernoulli links SB parity" begin
    x, y = _bernoulli_links_data()
    @test length(y) == 100 && sum(y) == 56
    cols = Dict{Symbol,AbstractVector}(:y => y, :x => x)
    sb = Dict(
        "logit" => (-71.1083613684301,
            [-6.731688596946846, 2.056004273514957]),
        "probit" => (-74.70474010682705,
            [-22.163518695057945, 9.180926009900183]),
        "cloglog" => (-87.74575244667403,
            [-52.8528057278787, 17.566693811623846]),
    )
    for (name, prog, _) in _links_parity_progs()
        @testset "$name" begin
            # SB: a, b ~ std_normal; eta ~ 1 + x; y ~ Bernoulli(link).
            # u (SB order [b0, b1]) = [0.5, -0.25].
            bound, built, kern = _links_query(prog, cols)
            names = coordinate_names(built.layout)
            u = _links_sb_vec(names, [:a => 0.5,
                :b => -0.25])
            want, wantg = sb[name]
            @test abs(Base.invokelatest(kern, u) - want) < 1e-12
            prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
            g = similar(u)
            sampler_value_and_gradient!(prep, g, u)
            @test all(isfinite, g)
            @test maximum(abs.(g .- _links_sb_vec(names,
                [names[1] => wantg[1], names[2] => wantg[2]]))) < 1e-10
        end
    end
    @testset "binomial twins" begin
        xb, n, cl, cp, cc = _bernoulli_links_binom_data()
        @test xb == x
        @test sum(cl) == 298 && sum(cp) == 352 && sum(cc) == 393
        @test cl[1:3] == [3, 3, 3] && cp[1:3] == [2, 1, 3] &&
            cc[1:3] == [3, 4, 5]
        twins = [
            ("logit", cl, quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Binomial.(n, logistic.(mu))
            end, -150.21083717068223,
            [-13.658442984734227, 0.6277072431875812]),
            ("probit", cp, quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Binomial.(n, normcdf.(mu))
            end, -141.05985441626325,
            [12.6379028940814, -9.979531779005764]),
            ("cloglog", cc, quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Binomial.(n, cexpexp.(mu))
            end, -115.99032528998896,
            [-15.62227803566569, -17.65335024052007]),
        ]
        for (name, yy, prog, want, wantg) in twins
            @testset "$name" begin
                # SB: a, b ~ std_normal; mu ~ 1 + x; c ~ Binomial(n, link).
                # u (SB order [b0, b1]) = [0.5, -0.25].
                tcols = Dict{Symbol,AbstractVector}(:y => yy, :x => xb,
                    :n => n)
                bound, built, kern = _links_query(prog, tcols)
                names = coordinate_names(built.layout)
                u = _links_sb_vec(names, [:a => 0.5,
                    :b => -0.25])
                @test abs(Base.invokelatest(kern, u) - want) < 1e-12
                prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
                g = similar(u)
                sampler_value_and_gradient!(prep, g, u)
                @test all(isfinite, g)
                @test maximum(abs.(g .- _links_sb_vec(names,
                    [names[1] => wantg[1], names[2] => wantg[2]]))) < 1e-10
            end
        end
    end
end

@testset "bernoulli links n=100 values + Enzyme" begin
    x, y = _bernoulli_links_data()
    @test length(y) == 100 && sum(y) == 56
    @test x[1:3] ==
        [-0.26271505251141175, 0.7650795618053199, -1.91306237132541]
    @test y[1:3] == [0, 1, 0]
    @test x[98:100] ==
        [0.21799186807655632, -1.1763253133647555, 0.5715959926473191]
    @test y[98:100] == [1, 0, 0]
    cols = Dict{Symbol,AbstractVector}(:y => y, :x => x)
    u = [0.5, -0.25]
    for (name, prog, llfun) in _links_parity_progs()
        @testset "$name" begin
            bound, built, kern = _links_query(prog, cols)
            nt = constrain(built.layout, u)
            lp = [nt.a, nt.b]
            eta = lp[1] .+ lp[2] .* x
            pr = logpdf(Normal(0, 1), lp[1]) + logpdf(Normal(0, 1), lp[2])
            native = Base.invokelatest(kern, u)
            @test native ≈ llfun(eta, y) + pr
            q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
            g = similar(u)
            val, _ = sampler_value_and_gradient!(q, g, u)
            @test val ≈ native rtol = 1e-12
            @test g ≈
                _links_findiff(w -> Base.invokelatest(kern, w), u) rtol = 1e-5 atol = 1e-7
        end
    end
end
