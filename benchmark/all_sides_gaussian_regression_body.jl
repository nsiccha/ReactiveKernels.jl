# All-sides-optimized Gaussian linear regression comparison.
# RK @kernel vs optimized Turing @model vs Stan (naive / glm / conjugate) via BridgeStan.
# Fair boundary: setup/compile/first-call excluded; sufficient statistics precomputed once
# for every conjugate side (as Stan transformed_data / Turing make_model / RK data prefix).
# Timing: min of 200 samples per round, median over ROUNDS rounds (the lane's methodology).
# BridgeStan resolves its install from BRIDGESTAN or its documented default path
# (~/.bridgestan/bridgestan-<version>); we leave it to that default.
using Random, LinearAlgebra, Statistics, Dates
using BenchmarkTools
import TOML
using ReactiveKernels
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme

const LOG2PI = log(2π)
const ROUNDS = 12

# Exact linking helper from DynamicPPL benchmarks/posteriordb.jl (stable_ldf):
# LogDensityFunction over the LINKED (unconstrained) space.
function stable_ldf(model; adtype=nothing, logdensity=DynamicPPL.getlogjoint_internal)
    vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _, vi = DynamicPPL.init!!(model, vi, DynamicPPL.InitFromUniform(-2.1, -1.9), DynamicPPL.LinkAll())
    fixed_vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.FixedTransformAccumulator())
    _, fixed_vi = DynamicPPL.init!!(model, fixed_vi, DynamicPPL.InitFromUniform(-2.1, -1.9), DynamicPPL.LinkAll())
    transforms = DynamicPPL.get_fixed_transforms(fixed_vi)
    vecvals = DynamicPPL.getacc(vi, Val(:VectorValue)).values
    vecvals = DynamicPPL.update_transforms!!(vecvals, transforms)
    return DynamicPPL.LogDensityFunction(model, logdensity, vecvals; adtype, fix_transforms=false)
end

# ---------- data / sufficient statistics ----------
function gen_data(N, K; seed=20260906)
    rng = MersenneTwister(seed)
    X = randn(rng, N, K); βtrue = randn(rng, K)
    y = 1.5 .+ X*βtrue .+ 0.7 .* randn(rng, N)
    X, y
end
suffstats(X, y) = (; XtX = X'X, Xty = X'y, xsum = vec(sum(X; dims=1)),
                     ysum = sum(y), yty = dot(y,y), N = size(X,1), K = size(X,2))

# ---------- oracle (plain Julia, full-constant unconstrained target) ----------
nlp(x, m, s) = -0.5*LOG2PI - log(s) - 0.5*((x-m)/s)^2
function oracle_density(q, X, y)
    K = size(X,2); α = q[1]; β = @view q[2:K+1]; logσ = q[K+2]; σ = exp(logσ)
    prior = nlp(α,0,10) + sum(b->nlp(b,0,10), β) + nlp(σ,0,5)
    μ = α .+ X*β
    like = sum(n->nlp(y[n], μ[n], σ), 1:length(y))
    prior + like + logσ
end
function oracle_grad(q, X, y)
    g = similar(q); h = 1e-6
    for i in eachindex(q)
        qp = copy(q); qp[i]+=h; qm = copy(q); qm[i]-=h
        g[i] = (oracle_density(qp,X,y) - oracle_density(qm,X,y))/(2h)
    end
    g
end

# ---------- tiny JSON ----------
jnum(x::Integer) = string(x)
jnum(x::Real) = (v=Float64(x); isinteger(v) ? string(Int(v))*".0" : repr(v))
jvec(v) = "[" * join(jnum.(v), ",") * "]"
jmat(M) = "[" * join([jvec(view(M,i,:)) for i in 1:size(M,1)], ",") * "]"
json_obj(pairs) = "{" * join(["\"$k\":$v" for (k,v) in pairs], ",") * "}"

# ---------- Stan sources ----------
const STAN_NAIVE = raw"""
data { int<lower=0> N; int<lower=0> K; matrix[N,K] X; vector[N] y; }
parameters { real alpha; vector[K] beta; real<lower=0> sigma; }
model {
  vector[N] mu = alpha + X*beta;
  target += normal_lpdf(alpha|0,10); target += normal_lpdf(beta|0,10); target += normal_lpdf(sigma|0,5);
  for (n in 1:N) target += normal_lpdf(y[n]|mu[n],sigma);
}
"""
const STAN_GLM = raw"""
data { int<lower=0> N; int<lower=0> K; matrix[N,K] X; vector[N] y; }
parameters { real alpha; vector[K] beta; real<lower=0> sigma; }
model {
  target += normal_lpdf(alpha|0,10); target += normal_lpdf(beta|0,10); target += normal_lpdf(sigma|0,5);
  target += normal_id_glm_lpdf(y | X, alpha, beta, sigma);
}
"""
const STAN_CONJ = raw"""
data { int<lower=0> N; int<lower=0> K; matrix[K,K] XtX; vector[K] Xty; vector[K] xsum; real ysum; real yty; }
parameters { real alpha; vector[K] beta; real<lower=0> sigma; }
model {
  target += normal_lpdf(alpha|0,10); target += normal_lpdf(beta|0,10); target += normal_lpdf(sigma|0,5);
  real sse = yty - 2*alpha*ysum - 2*dot_product(beta,Xty) + N*square(alpha)
             + 2*alpha*dot_product(xsum,beta) + quad_form(XtX, beta);
  target += -0.5*N*log(2*pi()) - N*log(sigma) - 0.5*sse/square(sigma);
}
"""
function stan_model(src, data_json, tag)
    dir = mktempdir(); f = joinpath(dir, "$tag.stan"); write(f, src)
    BridgeStan.StanModel(f, data_json)
end

# ---------- Turing (conjugate, sufficient-statistic) ----------
Turing.@model function turing_conj(XtX, Xty, xsum, ysum, yty, N, K)
    alpha ~ Distributions.Normal(0,10)
    beta ~ Distributions.MvNormal(zeros(K), 10.0^2*I)
    sigma ~ Distributions.truncated(Distributions.Normal(0,5); lower=0)
    sse = yty - 2*alpha*ysum - 2*dot(beta,Xty) + N*alpha^2 + 2*alpha*dot(xsum,beta) + dot(beta, XtX*beta)
    DynamicPPL.@addlogprob! -0.5*N*LOG2PI - N*log(sigma) - 0.5*sse/sigma^2
end

# ---------- RK (conjugate) ----------
rk_alpha(q) = q[1]
rk_beta(q) = @view q[2:end-1]
rk_logsigma(q) = q[end]
rk_sigma(ls) = exp(ls)
rk_prior(alpha, beta, sigma) = nlp(alpha,0,10) + sum(b->nlp(b,0,10), beta) + nlp(sigma,0,5)
rk_sse(alpha, beta, XtX, Xty, xsum, ysum, yty, N) =
    yty - 2*alpha*ysum - 2*dot(beta,Xty) + N*alpha^2 + 2*alpha*dot(xsum,beta) + dot(beta, XtX*beta)
rk_like(sse, logsigma, sigma, N) = -0.5*N*LOG2PI - N*logsigma - 0.5*sse/sigma^2
rk_post(prior, like, logjac) = prior + like + logjac
rk_model = @kernel model(unconstrained, XtX, Xty, xsum, ysum, yty, N) = begin
    alpha = rk_alpha(unconstrained); beta = rk_beta(unconstrained)
    log_s = rk_logsigma(unconstrained); s = rk_sigma(log_s)
    prior = rk_prior(alpha, beta, s); log_jacobian = log_s
    sse = rk_sse(alpha, beta, XtX, Xty, xsum, ysum, yty, N)
    likelihood = rk_like(sse, log_s, s, N)
    posterior = rk_post(prior, likelihood, log_jacobian)
    return posterior
end

# ---------- timing ----------
function bench(thunk; rounds=ROUNDS)
    b = @benchmarkable $thunk()
    ts = Float64[]; bytes = Int[]
    for _ in 1:rounds
        e = minimum(run(b; samples=200, seconds=0.2))
        push!(ts, e.time); push!(bytes, e.memory)
    end
    (; median_ns=median(ts), min_ns=minimum(ts), median_bytes=Int(median(bytes)))
end

fmt(ns) = ns < 1e3 ? "$(round(ns;digits=1)) ns" : ns < 1e6 ? "$(round(ns/1e3;digits=2)) µs" : "$(round(ns/1e6;digits=3)) ms"

function run_size(N, K)
    X, y = gen_data(N, K); ss = suffstats(X, y)
    q = [0.3; 0.1 .* collect(1.0:K); log(0.6)]
    ref = oracle_density(q, X, y); gref = oracle_grad(q, X, y)
    gscale = maximum(abs, gref)

    dj_raw  = json_obj(["N"=>N, "K"=>K, "X"=>jmat(X), "y"=>jvec(y)])
    dj_conj = json_obj(["N"=>N, "K"=>K, "XtX"=>jmat(ss.XtX), "Xty"=>jvec(ss.Xty),
                        "xsum"=>jvec(ss.xsum), "ysum"=>jnum(ss.ysum), "yty"=>jnum(ss.yty)])
    sm_naive = stan_model(STAN_NAIVE, dj_raw, "stan_naive")
    sm_glm   = stan_model(STAN_GLM,   dj_raw, "stan_glm")
    sm_conj  = stan_model(STAN_CONJ,  dj_conj, "stan_conj")

    tm  = turing_conj(ss.XtX, ss.Xty, ss.xsum, ss.ysum, ss.yty, N, K)
    ldf = stable_ldf(tm; adtype=ADTypes.AutoMooncake(config=nothing))

    kern   = prepare(rk_model; have=(:unconstrained,:XtX,:Xty,:xsum,:ysum,:yty,:N), want=:posterior)
    kern_b = prepare(rk_model; have=(:unconstrained,:XtX,:Xty,:xsum,:ysum,:yty,:N), want=:posterior,
                     bound=(; ss.XtX, ss.Xty, ss.xsum, ss.ysum, ss.yty, ss.N))
    gbuf = similar(q)
    prep_b = prepare_ad(kern_b, AutoEnzyme(mode=Enzyme.Reverse, function_annotation=Enzyme.Const), q; active=:unconstrained)

    # parity gates (gradient vs FD oracle; value up to additive constant)
    gerr(g) = maximum(abs, g .- gref) / gscale
    checks = Dict{String,Float64}()
    checks["stan_naive"] = gerr(BridgeStan.log_density_gradient(sm_naive, q; propto=false, jacobian=true)[2])
    checks["stan_glm"]   = gerr(BridgeStan.log_density_gradient(sm_glm,   q; propto=false, jacobian=true)[2])
    checks["stan_conj"]  = gerr(BridgeStan.log_density_gradient(sm_conj,  q; propto=false, jacobian=true)[2])
    checks["turing"]     = gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, q)[2])
    checks["rk"]         = gerr(ReactiveKernels.ad_value_and_gradient!(prep_b, gbuf, q)[2])
    for (k,v) in checks
        v < 1e-3 || error("PARITY FAIL $k gradient rel err $v")
    end

    rows = Dict{String,Any}[]
    add!(kind, impl, r) = push!(rows, Dict("kind"=>kind, "impl"=>impl,
        "median_ns"=>r.median_ns, "min_ns"=>r.min_ns, "median_bytes"=>r.median_bytes))

    # PRIMAL
    add!("primal","stan_naive", bench(()->BridgeStan.log_density(sm_naive, q; propto=false, jacobian=true)))
    add!("primal","stan_glm",   bench(()->BridgeStan.log_density(sm_glm,   q; propto=false, jacobian=true)))
    add!("primal","stan_conj",  bench(()->BridgeStan.log_density(sm_conj,  q; propto=false, jacobian=true)))
    add!("primal","turing_conj",bench(()->DynamicPPL.LogDensityProblems.logdensity(ldf, q)))
    add!("primal","rk_conj_native", bench(()->kern(q, ss.XtX, ss.Xty, ss.xsum, ss.ysum, ss.yty, ss.N)))
    add!("primal","rk_conj_bound",  bench(()->kern_b(q)))
    # GRADIENT
    add!("gradient","stan_conj", bench(()->BridgeStan.log_density_gradient(sm_conj, q; propto=false, jacobian=true)))
    add!("gradient","turing_conj", bench(()->DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, q)))
    add!("gradient","rk_conj_bound", bench(()->ReactiveKernels.ad_value_and_gradient!(prep_b, gbuf, q)))

    println("\n===== N=$N K=$K  (dim=$(K+2)) =====   [parity max grad rel err: $(round(maximum(values(checks));sigdigits=3))]")
    for kind in ("primal","gradient")
        println("  -- $kind --")
        base = kind=="primal" ? first(r["median_ns"] for r in rows if r["kind"]=="primal" && r["impl"]=="stan_conj") :
                                 first(r["median_ns"] for r in rows if r["kind"]=="gradient" && r["impl"]=="stan_conj")
        for r in rows
            r["kind"]==kind || continue
            ratio = r["median_ns"]/base
            println("     $(rpad(r["impl"],16)) $(lpad(fmt(r["median_ns"]),10))   $(lpad(string(round(ratio;digits=2))*"x vs stan_conj",22))   $(r["median_bytes"]) B")
        end
    end
    (; N, K, rows)
end

results = [run_size(5000, 20), run_size(5000, 100)]

# receipt
receipt = Dict(
    "schema" => "all-sides-gaussian-regression-v1",
    "generated_at" => string(now()),
    "environment" => Dict("julia"=>string(VERSION), "arch"=>string(Sys.ARCH),
        "cpu"=>Sys.cpu_info()[1].model,
        "stan"=>"2.39.0 (BridgeStan 2.9.0)"),
    "methodology" => "median over $ROUNDS rounds of BenchmarkTools minimum(200 samples); setup/compile/first-call excluded; sufficient statistics precomputed once for all conjugate sides",
    "sizes" => [Dict("N"=>r.N, "K"=>r.K, "dim"=>r.K+2, "measurements"=>r.rows) for r in results],
)
outp = get(ENV, "AS_OUTPUT", "")
if outp != ""
    open(outp, "w") do io; TOML.print(io, receipt); end
    println("\nwrote receipt: $outp")
end
println("\nALLSIDES_DONE")
