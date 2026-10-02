using Distributions
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# `@plate` cells beyond scalars (user decision 01gqbsq, stratified prong):
# a cell means one iteration of the Julia loop, whatever the shape of its
# values. Per-level cells (`@plate for k in levels(g)`) lower to the
# whole-array declarations they mean. Helper: `_canon` (test_corpus.jl).

_pc_canon(ex, data) = sprint(_canon, lower_rkppl(ex, data; mod = @__MODULE__))

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

@testset "per-level plate cells not built yet" begin
    data = (:y, :g)
    lower(cell) = lower_rkppl(Expr(:block, :(a ~ Normal(0, 5)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
            Expr(:for, :(k = levels(g)), Expr(:block, cell))),
        :(mu = a .+ c[g]), :(y .~ Normal.(mu, 1.0))), data)
    # Not built yet (todo 069fxln): per-level arguments, one LKJ factor
    # per level, and per-level definitions are valid Julia loops.
    @test_broken (lower(:(c[k] ~ Normal(m[k], 1))); true)
    @test_broken (lower(:(c[k] ~ LKJCholesky(2, 1.0))); true)
    @test_broken (lower(:(c[k] = 2.0)); true)
end
