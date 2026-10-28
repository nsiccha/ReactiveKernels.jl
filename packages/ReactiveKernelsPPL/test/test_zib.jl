# Zero-inflated Binomial response (the m0 capture-recapture marginal
# shape): `ZeroInflatedBinomial.(n, p, zi)` over a Beta-sampled
# probability p (the BinomialProb precedent) plus the ZIP-convention zi
# slot (structural-zero probability, parameter or literal). Surface
# admission, value parity vs a Distributions.jl hand oracle,
# Enzyme-vs-findiff gradients, O(1) emission, Reactant/XLA value+grad
# (in test_zib_reactant.jl).
# (`_findiff_grad` / `_GEN_BACKEND` come from test_generator.jl, included
# first.)
#
# SB parity: NO COUNTERPART (documented outcome, not a failure). The
# partner's probe records (test/sb_sweep_probe_records.jsonl, 8 probes:
# R1/R3/ARK/R2/R4/DUG0/DUG1/DUGh; briefs 2026-09-28T12-50-19-611-xlkh75
# and 2026-09-28T15-12-36-933-u490ai on
# BayesianRegressionModels:rk:kernel:everything) bank no Z probe, and the
# 115 case records hold no ZeroInflatedBinomial case — the nearest
# neighbor, `m0`, is an (omega,p) capture-recapture model on different
# data. Stan has no zero_inflated_binomial builtin, so no hand-built Z
# probe exists to pin under `_ZIB_SB`. Requested points for the record:
#   Z1: cols s=[1,0,2,0,3,1,0,2]; probe p=0.6, zi=0.3; model
#       p ~ Beta(1,1), zi ~ Beta(1,1), s ~ ZeroInflatedBinomial(3,p,zi).
#   Z2: same cols; probe p=0.6, zi=0.25 literal; model
#       p ~ Beta(1,1), s ~ ZeroInflatedBinomial(3,p,0.25).
# The Distributions-oracle value tests below stand in place of SB parity
# (same shape as the verified-wall notes in test_bare_location.jl).
using DifferentiationInterface
using Distributions: Beta, Binomial, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Lower + bind + build + query a ZIB program; return
# `(bound, built, kern, layout)`.
function _zib_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_zib_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

_zib_logaddexp(a, b) = max(a, b) + log1p(exp(-abs(a - b)))

# Hand oracle for one ZIB cell (independent of the layout): the stable
# two-arm form over Distributions' Binomial.
function _zib_oracle_cell(y::Int, n::Int, p::Float64, zi::Float64)
    bb = logpdf(Binomial(n, p), y)
    return y == 0 ? _zib_logaddexp(log(zi), log1p(-zi) + bb) :
        log1p(-zi) + bb
end

const _ZIB_S = [1, 0, 2, 0, 3, 1, 0, 2]
const _ZIB_N = 3
_zib_cols() = Dict{Symbol,AbstractVector}(:s => copy(_ZIB_S))
const _ZIB_PROG = quote
    p ~ Beta(1.0, 1.0)
    zi ~ Beta(1.0, 1.0)
    s .~ ZeroInflatedBinomial.(3, p, zi)
end
const _ZIB_LITERAL_PROG = quote
    p ~ Beta(1.0, 1.0)
    s .~ ZeroInflatedBinomial.(3, p, 0.25)
end
const _ZIB_TRIALS_COL_PROG = quote
    p ~ Beta(1.0, 1.0)
    zi ~ Beta(1.0, 1.0)
    s .~ ZeroInflatedBinomial.(n, p, zi)
end
_zib_trials_cols() = Dict{Symbol,AbstractVector}(:s => copy(_ZIB_S),
    :n => fill(_ZIB_N, length(_ZIB_S)))

@testset "zib admission" begin
    @testset "sampled zi" begin
        plan = lower_rkppl(_ZIB_PROG, (:s,); conditioned = (:s,))
        r = only(plan.responses)
        @test r.family === ZeroInflatedBinomialFam
        @test r.link === IdentityLink
        @test r.predictor === :p
        @test r.trials === 3
        @test r.zi === :zi
        @test Set(p.name for p in plan.parameters) == Set([:p, :zi])
        @test all(p -> p.family === :beta, plan.parameters)
    end
    @testset "literal zi" begin
        plan = lower_rkppl(_ZIB_LITERAL_PROG, (:s,); conditioned = (:s,))
        r = only(plan.responses)
        @test r.family === ZeroInflatedBinomialFam
        @test r.zi === 0.25
    end
    @testset "trials column" begin
        plan = lower_rkppl(_ZIB_TRIALS_COL_PROG, (:s, :n); conditioned = (:s, :n))
        r = only(plan.responses)
        @test r.family === ZeroInflatedBinomialFam
        @test r.trials === :n
    end
end

@testset "zib contract failures" begin
    # Non-Beta p fails at contract (surface admits sampled params).
    # admitted: a link or value that can leave a slot's support; an out-of-support value has -Inf density (10gzbm9 support-links)
    @test (lower_rkppl(quote
        p ~ Normal(0.0, 1.0)
        zi ~ Beta(1.0, 1.0)
        s .~ ZeroInflatedBinomial.(3, p, zi)
    end, (:s,); conditioned = (:s,)); true)
    # zi literal outside [0, 1].
    # refused: zi literal 1.5 outside [0,1]
    @test_throws ContractValidationError lower_rkppl(quote
        p ~ Beta(1.0, 1.0)
        s .~ ZeroInflatedBinomial.(3, p, 1.5)
    end, (:s,); conditioned = (:s,))
    # zi names nothing in the plan.
    # refused: undeclared name (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        p ~ Beta(1.0, 1.0)
        s .~ ZeroInflatedBinomial.(3, p, nosuch)
    end, (:s,); conditioned = (:s,))
    # Response exceeding trials fails at bind.
    plan = lower_rkppl(_ZIB_PROG, (:s,); conditioned = (:s,))
    bad = Dict{Symbol,AbstractVector}(:s => [1, 0, 4, 0])
    # refused: response exceeds trials (wrong data)
    @test_throws ContractValidationError bind_data(plan, bad)
end

@testset "zib value parity" begin
    @testset "sampled zi" begin
        _, _, kern, lay = _zib_query(_ZIB_PROG, _zib_cols())
        q = (p = 0.6, zi = 0.3)
        got = _zib_posterior(kern, lay, q)
        # Hand oracle: Beta priors + ZIB likelihood + logit Jacobians
        # (log(p*(1-p))), fully independent of the layout.
        want = logpdf(Beta(1.0, 1.0), 0.6) + logpdf(Beta(1.0, 1.0), 0.3) +
               sum(_zib_oracle_cell(y, _ZIB_N, 0.6, 0.3) for y in _ZIB_S) +
               log(0.6 * 0.4) + log(0.3 * 0.7)
        @test got ≈ want rtol = 1e-12
    end
    @testset "literal zi" begin
        _, _, kern, lay = _zib_query(_ZIB_LITERAL_PROG, _zib_cols())
        q = (p = 0.6,)
        got = _zib_posterior(kern, lay, q)
        want = logpdf(Beta(1.0, 1.0), 0.6) +
               sum(_zib_oracle_cell(y, _ZIB_N, 0.6, 0.25) for y in _ZIB_S) +
               log(0.6 * 0.4)
        @test got ≈ want rtol = 1e-12
    end
    @testset "trials column matches literal" begin
        _, _, kern, lay = _zib_query(_ZIB_TRIALS_COL_PROG, _zib_trials_cols())
        q = (p = 0.6, zi = 0.3)
        got = _zib_posterior(kern, lay, q)
        _, _, kern2, lay2 = _zib_query(_ZIB_PROG, _zib_cols())
        want = _zib_posterior(kern2, lay2, q)
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _zib_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _zib_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "zib Enzyme gradients" begin
    @testset "sampled zi" begin
        _zib_enzyme_check(_ZIB_PROG, _zib_cols(), (p = 0.6, zi = 0.3))
    end
    @testset "literal zi" begin
        _zib_enzyme_check(_ZIB_LITERAL_PROG, _zib_cols(), (p = 0.6,))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the ZIB plate must not unroll over observations.
function _zib_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        heads[string(st.head)] = get(heads, string(st.head), 0) + 1
    end
    return heads
end

@testset "zib emission is O(1) in n_obs" begin
    h8 = _zib_statement_heads(_ZIB_PROG, _zib_cols())
    h16 = _zib_statement_heads(_ZIB_PROG,
        Dict{Symbol,AbstractVector}(:s => vcat(_ZIB_S, _ZIB_S)))
    @test h8 == h16
end
