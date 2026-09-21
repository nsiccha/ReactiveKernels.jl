# Whole-data GLM objects (`y ~ NormalIDGLM/BernoulliLogitGLM/PoissonLogGLM`)
# under dataflow and non-allocating reverse AD. The desugared models carry a
# beta-prior plate plus fused link/likelihood steps; NA gradients agree with
# dataflow, Distributions.jl oracles, and central differences. Runs only
# through `test/run_nonallocating_integration.jl`, not the default suite,
# because it needs the unregistered MutatingFunctions weak dependency.
using ReactiveKernels
using ReactiveKernelsPPL
using MutatingFunctions
using DifferentiationInterface
using DifferentiationInterface: AutoEnzyme
using Distributions: Bernoulli, Poisson, Normal, logpdf
import Enzyme
using Test

const GLM_NA_BACKEND = AutoEnzyme(mode = Enzyme.Reverse,
    function_annotation = Enzyme.Const)

const _GLM_X1 = [0.5, -0.3, 1.2, 0.1, -0.8, 0.4]
const _GLM_X2 = [1.0, 0.5, -0.5, 0.3, 0.9, -0.2]
const _GLM_XB = hcat(_GLM_X1, _GLM_X2)

# Lowering/prepare run at top level: defining the fused-source closures and
# calling them from inside one function frame trips Julia world-age gating.
const _GLM_CASES = Any[]

begin
    nast = :(begin
        X = hcat(x1, x2)
        alpha ~ Normal(0, 5)
        beta[axes(X, 2)] .~ Normal.(0, 2)
        y ~ NormalIDGLM(X, alpha, beta, 2.0)
    end)
    yn = [1.5, -0.5, 2.0, 0.3, -1.2, 0.8]
    nplan = lower_rkppl(nast, [:y, :x1, :x2])
    nbound = bind_data(nplan, Dict(:y => yn, :x1 => _GLM_X1, :x2 => _GLM_X2))
    nbuilt = build_kernel(nbound)
    ncols = sort!(collect(keys(nbound.columns)))
    nbnt = NamedTuple{Tuple(ncols)}(Tuple(nbound.columns[k] for k in ncols))
    nhave = (:unconstrained, ncols...)
    ndf = prepare(nbuilt.spec; have = nhave, want = :posterior, bound = nbnt)
    nna = prepare_nonallocating(nbuilt.spec; have = nhave, want = :posterior,
        bound = nbnt)
    un = [0.1, -0.2, 0.3]
    nnt = ReactiveKernelsPPL.constrain(nbuilt.layout, un)
    neta = nnt.alpha .+ _GLM_XB * nnt.beta
    noracle = sum(logpdf.(Normal.(neta, 2.0), yn)) +
        logpdf(Normal(0, 5), nnt.alpha) +
        sum(logpdf.(Normal(0, 2), nnt.beta))
    push!(_GLM_CASES, ("normal-id", ndf, nna, un, noracle))
end

begin
    bast = :(begin
        X = hcat(x1, x2)
        alpha ~ Normal(0, 5)
        beta[axes(X, 2)] .~ Normal.(0, 2)
        y ~ BernoulliLogitGLM(X, alpha, beta)
    end)
    yb = [1, 0, 1, 1, 0, 1]
    bplan = lower_rkppl(bast, [:y, :x1, :x2])
    bbound = bind_data(bplan, Dict(:y => yb, :x1 => _GLM_X1, :x2 => _GLM_X2))
    bbuilt = build_kernel(bbound)
    bcols = sort!(collect(keys(bbound.columns)))
    bbnt = NamedTuple{Tuple(bcols)}(Tuple(bbound.columns[k] for k in bcols))
    bhave = (:unconstrained, bcols...)
    bdf = prepare(bbuilt.spec; have = bhave, want = :posterior, bound = bbnt)
    bna = prepare_nonallocating(bbuilt.spec; have = bhave, want = :posterior,
        bound = bbnt)
    ub = [0.1, -0.2, 0.3]
    bnt = ReactiveKernelsPPL.constrain(bbuilt.layout, ub)
    beta_ = bnt.alpha .+ _GLM_XB * bnt.beta
    boracle = sum(logpdf.(Bernoulli.(1 ./ (1 .+ exp.(-beta_))), yb)) +
        logpdf(Normal(0, 5), bnt.alpha) +
        sum(logpdf.(Normal(0, 2), bnt.beta))
    push!(_GLM_CASES, ("bernoulli-logit", bdf, bna, ub, boracle))
end

begin
    past = :(begin
        X = hcat(x1, x2)
        alpha ~ Normal(0, 5)
        beta[axes(X, 2)] .~ Normal.([0.0, 0.0], [2.0, 3.0])
        y ~ PoissonLogGLM(X, alpha, beta)
    end)
    yp = [3, 1, 5, 2, 0, 4]
    pplan = lower_rkppl(past, [:y, :x1, :x2])
    pbound = bind_data(pplan, Dict(:y => yp, :x1 => _GLM_X1, :x2 => _GLM_X2))
    pbuilt = build_kernel(pbound)
    pcols = sort!(collect(keys(pbound.columns)))
    pbnt = NamedTuple{Tuple(pcols)}(Tuple(pbound.columns[k] for k in pcols))
    phave = (:unconstrained, pcols...)
    pdf = prepare(pbuilt.spec; have = phave, want = :posterior, bound = pbnt)
    pna = prepare_nonallocating(pbuilt.spec; have = phave, want = :posterior,
        bound = pbnt)
    up = [0.2, 0.1, -0.1]
    pnt = ReactiveKernelsPPL.constrain(pbuilt.layout, up)
    peta = pnt.alpha .+ _GLM_XB * pnt.beta
    poracle = sum(logpdf.(Poisson.(exp.(peta)), yp)) +
        logpdf(Normal(0, 5), pnt.alpha) + logpdf(Normal(0, 2), pnt.beta[1]) +
        logpdf(Normal(0, 3), pnt.beta[2])
    push!(_GLM_CASES, ("poisson-log", pdf, pna, up, poracle))
end

function _glm_findiff(kern, u0; h = 1e-6)
    g = similar(u0, Float64)
    for i in eachindex(u0)
        up = copy(u0)
        up[i] += h
        dn = copy(u0)
        dn[i] -= h
        g[i] = (kern(up) - kern(dn)) / (2h)
    end
    return g
end

@testset "GLM objects dataflow/nonallocating AD agreement" begin
    @test Base.get_extension(ReactiveKernels,
        :ReactiveKernelsMutatingFunctionsExt) !== nothing
    for (tag, df, na, u0, oracle) in _GLM_CASES
        ad_df = prepare_ad(df, GLM_NA_BACKEND, u0; active = :unconstrained)
        ad_na = prepare_ad(na, GLM_NA_BACKEND, u0; active = :unconstrained)
        g_df, g_na = similar(u0), similar(u0)
        v_df, _ = ad_value_and_gradient!(ad_df, g_df, u0)
        v_na, _ = ad_value_and_gradient!(ad_na, g_na, u0)
        @test v_df ≈ oracle
        @test v_na ≈ oracle
        @test v_na ≈ v_df
        @test g_na ≈ g_df
        @test g_na ≈ _glm_findiff(df, u0) rtol = 1e-5
    end
end
