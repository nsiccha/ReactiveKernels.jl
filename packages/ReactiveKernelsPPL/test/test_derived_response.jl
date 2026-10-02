using Distributions: Normal, Exponential, logpdf
using ReactiveKernelsPPL
using Test

# Derived responses (G3): `.~` over a deterministic definition
# (`ly = log.(earn)` then `ly .~ Normal.(mu, s)`) — the logearn/logmesquite
# shape. Cross-file helpers come from test_generator.jl (`_GEN_BACKEND`)
# and test_prior_vocab.jl (`_pv_query`, `_pv_posterior`, `_pv_enzyme_check`,
# `_pv_reactant`), both included before this file in runtests.jl.

const _DR_M1 = quote
    b1 ~ Flat()
    b2 ~ Flat()
    s ~ Exponential(1)
    ly = log.(earn)
    mu = b1 .+ b2 .* x
    ly .~ Normal.(mu, s)
end
const _DR_EARN = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
const _DR_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
_dr_cols() = Dict{Symbol,AbstractVector}(
    :earn => copy(_DR_EARN), :x => copy(_DR_X))
_dr_bigcols() = Dict{Symbol,AbstractVector}(
    :earn => [2.0^i for i in 0:11], :x => repeat(_DR_X, 2))

function _dr_m1_oracle(b1::Real, b2::Real, s::Real)
    ly = log.(_DR_EARN)
    mu = b1 .+ b2 .* _DR_X
    ll = sum(logpdf(Normal(m, s), y) for (m, y) in zip(mu, ly))
    # Flat coefficient priors contribute exactly 0.0.
    return ll + logpdf(Exponential(1), s) + log(s)
end

@testset "derived response admission" begin
    plan = lower_rkppl(_DR_M1, (:earn, :x))
    r = only(plan.responses)
    @test r.response === :ly
    @test r.label === :ly_resp
    @test [d.name for d in plan.derived] == [:ly]
    @test [p.name for p in plan.parameters] == [:b1, :b2, :s]
    @test isempty(plan.population_priors)
    fams = Dict(p.name => p.family for p in plan.parameters)
    @test fams == Dict(:b1 => :flat, :b2 => :flat, :s => :exponential)
    # Forward order lowers identically.
    fwd = lower_rkppl(quote
            b1 ~ Flat()
            b2 ~ Flat()
            s ~ Exponential(1)
            ly .~ Normal.(mu, s)
            mu = b1 .+ b2 .* x
            ly = log.(earn)
        end, (:earn, :x))
    @test [rr.response for rr in fwd.responses] == [:ly]
    @test [d.name for d in fwd.derived] == [:ly]
    # Bind materializes the response from bound data.
    bound = bind_data(plan, _dr_cols())
    @test bound.columns[:ly] ≈ log.(_DR_EARN)
    @test bound.roles[:ly] === :response
    @test bound.n_obs == 6
end

@testset "derived response fail-closed battery" begin
    cases = (
        ("scalar definition",
            quote
                t = 2.0
                mu = b1 .+ b2 .* x
                t .~ Normal.(mu, s)
            end,
            "t is a scalar definition"),
        ("second observation",
            quote
                ly = log.(earn)
                mu = b1 .+ b2 .* x
                ly .~ Normal.(mu, s)
                ly .~ Normal.(mu, s)
            end,
            "already observed by a `.~` response"),
        ("predictor definition",
            quote
                mu = b1 .+ b2 .* x
                mu .~ Normal.(mu, s)
            end,
            "response mu is a predictor definition"),
        ("range over derived",
            quote
                ly = log.(earn)
                mu = b1 .+ b2 .* x
                ly[1:6] .~ Normal.(mu, s)
            end,
            "broadcast bare"),
        ("levels over definition",
            quote
                c = x .* 2
                mu = c[g] .+ b .* x
                c[levels(g)] .~ Normal.(0, 1)
            end,
            "sizes a coefficient prior but `c` is a deterministic definition"),
        ("direct sampled read",
            quote
                s ~ Exponential(1)
                ly = earn .* s
                mu = b1 .+ b2 .* x
                ly .~ Normal.(mu, s)
            end,
            "response ly is a predictor definition"),
        ("scalar tilde over definition",
            quote
                ly = log.(earn)
                mu = b1 .+ b2 .* x
                ly ~ Normal(0, 1)
            end,
            "defined twice"),
        ("design matrix LHS",
            quote
                X = hcat(1, x)
                mu = X * b
                X .~ Normal.(mu, s)
            end,
            "is a design matrix"),
    )
    for (label, prog, msg) in cases
        err = try
            lower_rkppl(prog, (:earn, :x, :g))
            nothing
        catch e
            e
        end
        @test err isa SurfaceLoweringError
        @test occursin(msg, sprint(showerror, err))
    end
end

@testset "derived response bind" begin
    plan = lower_rkppl(_DR_M1, (:earn, :x))
    # A caller column the model derives is a shadowing bug, not data.
    err = try
        bind_data(plan, Dict{Symbol,AbstractVector}(:earn => copy(_DR_EARN),
                :x => copy(_DR_X), :ly => copy(_DR_EARN)))
        nothing
    catch e
        e
    end
    @test err isa ContractValidationError
    @test occursin("drop it from bind_data", sprint(showerror, err))
    # A parameter-dependent response definition is rejected during lowering.
    perr = try
        pplan = lower_rkppl(quote
                b1 ~ Flat()
                b2 ~ Flat()
                s ~ Exponential(1)
                t = s * 2
                ly = earn .* t
                mu = b1 .+ b2 .* x
                ly .~ Normal.(mu, s)
            end, (:earn, :x))
        bind_data(pplan, _dr_cols())
        nothing
    catch e
        e
    end
    @test perr isa SurfaceLoweringError
    @test occursin("response ly is a predictor definition", sprint(showerror, perr))
    # Transitive data-only chains materialize (centering included).
    cplan = lower_rkppl(quote
            b1 ~ Flat()
            b2 ~ Flat()
            s ~ Exponential(1)
            l2 = log.(earn)
            zc = x .- mean(x)
            ly = l2 .+ zc
            mu = b1 .+ b2 .* x
            ly .~ Normal.(mu, s)
        end, (:earn, :x))
    cbound = bind_data(cplan, _dr_cols())
    using_mean = sum(_DR_X) / length(_DR_X)
    @test cbound.columns[:ly] ≈ log.(_DR_EARN) .+ (_DR_X .- using_mean)
    # Materialized values flow into response validation: a Float-derived
    # Bernoulli response fails exactly like a raw Float column.
    berr = try
        bplan = lower_rkppl(quote
                b1 ~ Flat()
                b2 ~ Flat()
                yb = earn .- 1.0
                mu = b1 .+ b2 .* x
                yb .~ BernoulliLogit.(mu)
            end, (:earn, :x))
        bind_data(bplan, _dr_cols())
        nothing
    catch e
        e
    end
    @test berr isa ContractValidationError
    @test occursin("Bernoulli response must be Bool or 0/1 integers",
        sprint(showerror, berr))
end

@testset "derived response values vs oracles" begin
    _, _, kern, lay = _pv_query(_DR_M1, _dr_cols())
    @test _pv_posterior(kern, lay, (b1 = 0.5, b2 = -0.25, s = 1.3)) ≈
        _dr_m1_oracle(0.5, -0.25, 1.3) rtol = 1e-12
end

@testset "derived response Enzyme gradients" begin
    _pv_enzyme_check(_DR_M1, _dr_cols(), (b1 = 0.5, b2 = -0.25, s = 1.3))
end

@testset "derived response under Reactant" begin
    # Vector-mu Normal-id: default pipeline (narrowed §7n scope —
    # scalar-mu only).
    fx = _pv_reactant(_DR_M1, _dr_cols())
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
    # The derived local is a retained broadcast: doubling n_obs adds no
    # HLO lines (core constraint 1 — no data-derived unrolling).
    @test _pv_reactant(_DR_M1, _dr_bigcols()).lines == fx.lines
end
