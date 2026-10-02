using Distributions
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

include("fixtures/level_plate_values.jl")

# `@plate` cells beyond scalars (user decision 01gqbsq, stratified prong):
# a cell means one iteration of the Julia loop, whatever the shape of its
# values. Per-level cells (`@plate for k in levels(g)`) lower to the
# whole-array declarations they mean. Helper: `_canon` (test_corpus.jl).

_pc_canon(ex, data) = sprint(_canon, lower_rkppl(ex, data; mod = @__MODULE__, conditioned = data))

@testset "per-level plate cells are their whole-array declarations" begin
    data = (:y, :x, :g, :s)
    # One scalar draw per level: `u[k] ~ Normal(0, su)` is `u[levels(s)] .~ Normal.(0, su)`.
    # One row per level: `b[j, :] ~ D` is the row statement.
    plated = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1); su ~ HalfNormal(1)
        sd[1:2] .~ HalfNormal.(1); L ~ LKJCholesky(2, 1.0); F = sd .* L
        @plate for k in levels(s)
            u[k] ~ Normal(0, su)
        end
        @plate for j in levels(g)
            b[j, :] ~ MvNormalCholesky(zeros(2), F)
        end
        mu = a .+ b[g, 1] .+ x .* b[g, 2] .+ u[s]
        y .~ Normal.(mu, sigma)
    end
    written = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1); su ~ HalfNormal(1)
        sd[1:2] .~ HalfNormal.(1); L ~ LKJCholesky(2, 1.0); F = sd .* L
        @plate for k in levels(s)
            u[k] ~ Normal(0, su)
        end
        @plate for j in levels(g)
            b[j, 1:2] ~ MvNormalCholesky(zeros(2), F)
        end
        mu = a .+ b[g, 1] .+ x .* b[g, 2] .+ u[s]
        y .~ Normal.(mu, sigma)
    end
    whole = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1); su ~ HalfNormal(1)
        sd[1:2] .~ HalfNormal.(1); L ~ LKJCholesky(2, 1.0); F = sd .* L
        u[levels(s)] .~ Normal.(0, su)
        eachrow(b[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), F)
        mu = a .+ b[g, 1] .+ x .* b[g, 2] .+ u[s]
        y .~ Normal.(mu, sigma)
    end
    @test _pc_canon(plated, data) == _pc_canon(whole, data)
    @test _pc_canon(written, data) == _pc_canon(whole, data)
end

@testset "per-level LKJ factors and per-index cells that hold arrays" begin
    data = (:y, :x, :g, :s)
    cols = Dict{Symbol,AbstractVector}(:s => repeat([1, 2], 12),
        :g => repeat([1, 2, 3, 4], 6), :x => collect(range(-1.3, 1.7; length = 24)),
        :y => [0.8 * sin(0.7 * i) for i in 1:24])
    # Columns per component and a row per index lower to one model.
    cols_prog = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1)
        sd[levels(s), 1:2] .~ HalfNormal.(1)
        @plate for k in levels(s)
            L[k] ~ LKJCholesky(2, 1.0)
        end
        z[levels(g), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(g)
            F = sd[s[i], :] .* L[s[i]]
            row = F * z[g[i], :]
            r1[i] = row[1]
            r2[i] = row[2]
        end
        mu = a .+ r1 .+ x .* r2
        y .~ Normal.(mu, sigma)
    end
    rows_prog = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1)
        sd[levels(s), 1:2] .~ HalfNormal.(1)
        @plate for k in levels(s)
            L[k] ~ LKJCholesky(2, 1.0)
        end
        z[levels(g), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(g)
            b[i, 1:2] = (sd[s[i], :] .* L[s[i]]) * z[g[i], :]
        end
        mu = a .+ b[:, 1] .+ x .* b[:, 2]
        y .~ Normal.(mu, sigma)
    end
    for prog in (cols_prog, rows_prog)
        plan = bind_data(lower_rkppl(prog, data; mod = @__MODULE__, conditioned = data), cols)
        built = build_kernel(plan)
        u = [0.3 * sin(1.1 * i + 0.3) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        @test size(nt.L) == (2, 2, 2)
        @test unconstrain(built.layout, nt) ≈ u
        mu = map(1:24) do i
            si, gi = cols[:s][i], cols[:g][i]
            row = (Diagonal(nt.sd[si, :]) * nt.L[:, :, si]) * nt.z[gi, :]
            nt.a + row[1] + cols[:x][i] * row[2]
        end
        hn(v) = logpdf(truncated(Normal(0, 1), 0, Inf), v)
        oracle = logpdf(Normal(0, 5), nt.a) +
            logpdf(Exponential(1), nt.sigma) + sum(hn, nt.sd) +
            sum(logpdf(LKJCholesky(2, 1.0),
                Cholesky(LowerTriangular(nt.L[:, :, k]))) for k in 1:2) +
            sum(logpdf.(Normal(0, 1), nt.z)) +
            sum(logpdf.(Normal.(mu, nt.sigma), cols[:y])) +
            logjac(built.layout, u)
        @test prepare_query(built, plan, :sampler)(u) ≈ oracle
        _check_gradient(built.spec, plan, u)
    end
end

@testset "per-level arguments and deterministic values" begin
    for kind in (:varying, :latent, :deterministic, :constant), S in (1, 3)
        built, bound, cols, u = _plv_build(kind, 12, S)
        saved = deepcopy(cols)
        value, c = _plv_oracle(built, cols, kind, u)
        @test prepare_query(built, bound, :sampler)(u) ≈ value
        @test _query(built.spec, bound, :c, u) ≈ c
        @test unconstrain(built.layout, constrain(built.layout, u)) ≈ u
        grad = _check_gradient(built.spec, bound, u)
        @test grad ≈ _findiff_grad(w -> first(_plv_oracle(built, cols, kind, w)), u)
        @test cols == saved
    end
    # Symbolic levels index declared arrays by level, without exposing
    # ordinal positions as the loop's authored values.
    built, bound, cols, u = _plv_build(:latent, 12, 3; labels = ["c", "a", "b"])
    @test prepare_query(built, bound, :sampler)(u) ≈
        first(_plv_oracle(built, cols, :latent, u))
    _check_gradient(built.spec, bound, u)

    data = (:y, :g, :m)
    lower(cell) = lower_rkppl(Expr(:block, :(a ~ Normal(0, 5)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
            Expr(:for, :(k = levels(g)), Expr(:block, cell))),
        :(mu = a .+ c[g]), :(y .~ Normal.(mu, 1.0))), data; conditioned = data)
    @test lower(:(c[k] ~ Normal(m[k], 1))) isa StructuralPlan
    @test lower(:(c[k] = 2.0)) isa StructuralPlan
    selected = quote
        z[levels(g)[2:end]] .~ Normal.(0, 1)
        @plate for k in levels(g)
            c[k] = z[k]
        end
        y .~ Normal.(c[g], 1)
    end
    @test_throws SurfaceLoweringError lower_rkppl(selected, (:y, :g); conditioned = (:y, :g))
    otheraxis = quote
        z[levels(h)] .~ Normal.(0, 1)
        @plate for k in levels(g)
            c[k] = z[k]
        end
        y .~ Normal.(c[g], 1)
    end
    @test_throws SurfaceLoweringError lower_rkppl(otheraxis, (:y, :g, :h); conditioned = (:y, :g, :h))
end

@testset "per-level assignments keep inactive branches inactive" begin
    built, bound, u, oracle = _plv_lazy_build()
    @test prepare_query(built, bound, :sampler)(u) ≈ oracle(u)
    @test _check_gradient(built.spec, bound, u) ≈ _findiff_grad(oracle, u)
end
