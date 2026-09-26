# Inverse-Gaussian / Wald response (SB `brm_inverse_gaussian_lpdf`
# mirror): surface admission, value parity vs a Distributions.jl oracle
# (literal / LogNormal-sampled / per-observation lambda),
# Enzyme-vs-findiff gradients, Reactant/XLA value+grad, and O(1)
# emission. W1/W2 SB-parity constants land with the peer lane's
# BridgeStan numbers. (`_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: InverseGaussian, Normal, LogNormal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query an IG program; return
# `(bound, built, kern, layout)`.
function _ig_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_ig_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar Wald log-density (Distributions.jl oracle; SB matches it
# operation-for-operation).
_ig_ref(y::Real, mu::Real, lam::Real) = logpdf(InverseGaussian(mu, lam), y)

const _IG_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _IG_Y = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0]
_ig_cols() = Dict{Symbol,AbstractVector}(:y => copy(_IG_Y), :x => copy(_IG_X))

@testset "ig surface admission" begin
    @testset "literal lambda" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), 1.5)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === InverseGaussianFam
        @test r.link === LogLink
        @test r.predictor === :eta
        @test r.scale === 1.5
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "LogNormal-sampled lambda" begin
        plan = lower_rkppl(quote
                lam ~ LogNormal(-0.3, 1.0)
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lam)
            end, (:y, :x))
        r = only(plan.responses)
        @test (r.family, r.scale) === (InverseGaussianFam, :lam)
        @test only(plan.parameters).family === :lognormal
    end
    @testset "per-observation lambda column" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lamc)
            end, (:y, :x, :lamc))
        @test only(plan.responses).scale === :lamc
    end
    @testset "modeled lambda deferred" begin
        @test_throws ContractValidationError lower_rkppl(quote
                eta = a .+ b .* x
                ls = c .+ d .* x
                y .~ InverseGaussian.(exp.(eta), exp.(ls))
            end, (:y, :x))
    end
end

@testset "ig value parity" begin
    @testset "literal lambda" begin
        _, _, kern, lay = _ig_query(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), 1.5)
            end, _ig_cols())
        q = (eta = [0.5, -0.25],)
        got = _ig_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _IG_X)
        want = sum(_ig_ref(y, m, 1.5) for (y, m) in zip(_IG_Y, mu)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "LogNormal-sampled lambda" begin
        _, _, kern, lay = _ig_query(quote
                lam ~ LogNormal(-0.3, 1.0)
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lam)
            end, _ig_cols())
        q = (eta = [0.5, -0.25], lam = 1.2)
        got = _ig_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _IG_X)
        want = sum(_ig_ref(y, m, q.lam) for (y, m) in zip(_IG_Y, mu)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2]) +
            logpdf(LogNormal(-0.3, 1.0), q.lam) +
            logjac(lay, unconstrain(lay, q))
        @test got ≈ want rtol = 1e-12
    end
    @testset "per-observation lambda column" begin
        cols = _ig_cols()
        cols[:lamc] = [0.5, 1.5, 2.5, 1.0, 2.0, 0.8]
        _, _, kern, lay = _ig_query(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lamc)
            end, cols)
        q = (eta = [0.5, -0.25],)
        got = _ig_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _IG_X)
        want = sum(_ig_ref(y, m, l)
            for (y, m, l) in zip(_IG_Y, mu, cols[:lamc])) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _ig_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _ig_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "ig Enzyme gradients" begin
    @testset "literal lambda" begin
        _ig_enzyme_check(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), 1.5)
            end, _ig_cols(), (eta = [0.5, -0.25],))
    end
    @testset "LogNormal-sampled lambda" begin
        _ig_enzyme_check(quote
                lam ~ LogNormal(-0.3, 1.0)
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lam)
            end, _ig_cols(), (eta = [0.5, -0.25], lam = 1.2))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the IG plate must not unroll over observations.
function _ig_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        heads[string(st.head)] = get(heads, string(st.head), 0) + 1
    end
    return heads
end

@testset "ig emission is O(1) in n_obs" begin
    prog = quote
        lam ~ LogNormal(-0.3, 1.0)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), lam)
    end
    h6 = _ig_statement_heads(prog, _ig_cols())
    h12 = _ig_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_IG_Y, _IG_Y), :x => vcat(_IG_X, _IG_X)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _ig_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_ig_reactant_measure, built, bound, post_q, u)
end

function _ig_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "ig under Reactant" begin
    progs = [
        ("literal lambda", quote
            eta = a .+ b .* x
            y .~ InverseGaussian.(exp.(eta), 1.5)
        end, _ig_cols()),
        ("LogNormal-sampled lambda", quote
            lam ~ LogNormal(-0.3, 1.0)
            eta = a .+ b .* x
            y .~ InverseGaussian.(exp.(eta), lam)
        end, _ig_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _ig_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _ig_reactant(prog, _ig_cols())
        large = _ig_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_IG_Y, _IG_Y), :x => vcat(_IG_X, _IG_X)))
        @test small.lines == large.lines
    end
end
