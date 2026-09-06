# Fair all-sides benchmark on posteriordb GLM models — GLM_Poisson & GLM_Binomial.
#
# Motivation: the DynamicPPL posteriordb benchmark reports Turing/Stan ratios
# against the NAIVE posteriordb reference Stan. Here every side is optimized and
# ReactiveKernels is added: RK @kernel (one authored graph, lowers to native +
# Reactant) vs Stan (naive loop / vectorized / fused-GLM where it exists) vs
# optimized Turing (the posteriordb-style translation). Weak-normal unbounded
# priors replace posteriordb's bounded uniforms so the likelihood dominates the
# timing; parity is gated against an independent finite-difference oracle.
#
# Fair boundary: setup / compile / first-call excluded; the data-only additive
# constant (lgamma(C+1) / lchoose(N,C)) is precomputed once for every side.
# Timing: median over 12 rounds of BenchmarkTools minimum(200 samples).
using Random, LinearAlgebra, Statistics, SpecialFunctions, LogExpFunctions, Dates
using BenchmarkTools
import TOML
using ReactiveKernels
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme

const ROUNDS = 12
nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2

# ---------- shared infra ----------
jnum(x::Integer) = string(x)
jnum(x::Real) = (v = Float64(x); isinteger(v) ? string(Int(v)) * ".0" : repr(v))
jvec(v) = "[" * join(jnum.(v), ",") * "]"
jmat(M) = "[" * join([jvec(view(M, i, :)) for i in 1:size(M, 1)], ",") * "]"
jobj(p) = "{" * join(["\"$k\":$v" for (k, v) in p], ",") * "}"
stan_model(src, dj, tag) = (d = mktempdir(); f = joinpath(d, "$tag.stan"); write(f, src); BridgeStan.StanModel(f, dj))

function stable_ldf(model; adtype = nothing, logdensity = DynamicPPL.getlogjoint_internal)
    vi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _, vi = DynamicPPL.init!!(model, vi, DynamicPPL.InitFromUniform(-2.1, -1.9), DynamicPPL.LinkAll())
    fvi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.FixedTransformAccumulator())
    _, fvi = DynamicPPL.init!!(model, fvi, DynamicPPL.InitFromUniform(-2.1, -1.9), DynamicPPL.LinkAll())
    tr = DynamicPPL.get_fixed_transforms(fvi)
    vv = DynamicPPL.getacc(vi, Val(:VectorValue)).values
    vv = DynamicPPL.update_transforms!!(vv, tr)
    DynamicPPL.LogDensityFunction(model, logdensity, vv; adtype, fix_transforms = false)
end

function bench(thunk; rounds = ROUNDS)
    b = @benchmarkable $thunk()
    ts = Float64[]; by = Int[]
    for _ in 1:rounds
        e = minimum(run(b; samples = 200, seconds = 0.2)); push!(ts, e.time); push!(by, e.memory)
    end
    (; median_ns = median(ts), median_bytes = Int(median(by)))
end
fmt(ns) = ns < 1e3 ? "$(round(ns; digits=1)) ns" : ns < 1e6 ? "$(round(ns/1e3; digits=2)) µs" : "$(round(ns/1e6; digits=3)) ms"

# ==================== GLM_Poisson (Poisson-log cubic-trend GLM) ====================
function gen_pois(n; seed = 20260906)
    rng = MersenneTwister(seed); yr = collect(range(-1.5, 1.5; length = n)); X = hcat(yr, yr .^ 2, yr .^ 3)
    ab = [0.8, 0.4, -0.2, 0.1]; η = ab[1] .+ X * ab[2:4]
    C = [rand(rng, Distributions.Poisson(exp(min(e, 6.0)))) for e in η]
    X, C
end
oracle_pois(q, X, C) = (α = q[1]; β = @view q[2:4]; η = α .+ X * β;
    sum(x -> nlp(x, 0, 5), q) + dot(C, η) - sum(exp, η) - sum(c -> loggamma(c + 1.0), C))
const STAN_POIS_NAIVE = raw"""
data { int<lower=0> n; int<lower=0> K; matrix[n,K] X; array[n] int<lower=0> C; }
parameters { real alpha; vector[K] beta; }
model { target += normal_lpdf(alpha|0,5); target += normal_lpdf(beta|0,5);
  vector[n] eta = alpha + X*beta; for (i in 1:n) target += poisson_log_lpmf(C[i]|eta[i]); }
"""
const STAN_POIS_VEC = raw"""
data { int<lower=0> n; int<lower=0> K; matrix[n,K] X; array[n] int<lower=0> C; }
parameters { real alpha; vector[K] beta; }
model { target += normal_lpdf(alpha|0,5); target += normal_lpdf(beta|0,5);
  target += poisson_log_lpmf(C | alpha + X*beta); }
"""
const STAN_POIS_GLM = raw"""
data { int<lower=0> n; int<lower=0> K; matrix[n,K] X; array[n] int<lower=0> C; }
parameters { real alpha; vector[K] beta; }
model { target += normal_lpdf(alpha|0,5); target += normal_lpdf(beta|0,5);
  target += poisson_log_glm_lpmf(C | X, alpha, beta); }
"""
Turing.@model function turing_pois(X, C, K)
    alpha ~ Distributions.Normal(0, 5); beta ~ Distributions.MvNormal(zeros(K), 25.0 * I)
    C ~ Distributions.product_distribution([Distributions.Poisson(exp(e)) for e in (alpha .+ X * beta)])
end
pois_alpha(q) = q[1]; pois_beta(q) = @view q[2:end]
pois_eta(a, b, X) = a .+ X * b; pois_prior(q) = sum(x -> nlp(x, 0, 5), q)
pois_ll(C, eta, cterm) = dot(C, eta) - sum(exp, eta) - cterm
pois_post(p, l) = p + l
rk_pois = @kernel model(unconstrained, X, C, cterm) = begin
    a = pois_alpha(unconstrained); b = pois_beta(unconstrained); eta = pois_eta(a, b, X)
    prior = pois_prior(unconstrained); likelihood = pois_ll(C, eta, cterm); posterior = pois_post(prior, likelihood)
    return posterior
end

# ==================== GLM_Binomial (binomial-logit GLM) ====================
function gen_bin(n; seed = 20260906)
    rng = MersenneTwister(seed); yr = collect(range(-1.5, 1.5; length = n)); X = hcat(yr, yr .^ 2)
    ab = [0.2, 0.6, -0.3]; lp = ab[1] .+ X * ab[2:3]; N = fill(50, n)
    C = [rand(rng, Distributions.Binomial(N[i], logistic(lp[i]))) for i in 1:n]
    X, C, N
end
lchoose(N, c) = loggamma(N + 1.0) - loggamma(c + 1.0) - loggamma(N - c + 1.0)
oracle_bin(q, X, C, N) = (α = q[1]; β = @view q[2:3]; η = α .+ X * β;
    sum(x -> nlp(x, 0, 100), q) + sum(lchoose(N[i], C[i]) for i in eachindex(C)) + dot(C, η) - dot(N, log1pexp.(η)))
const STAN_BIN_NAIVE = raw"""
data{int<lower=0> n;int<lower=0> K;matrix[n,K] X;array[n] int C;array[n] int N;}
parameters{real alpha;vector[K] beta;}
model{target+=normal_lpdf(alpha|0,100);target+=normal_lpdf(beta|0,100);
 vector[n] lp=alpha+X*beta; for(i in 1:n) target+=binomial_logit_lpmf(C[i]|N[i],lp[i]);}
"""
const STAN_BIN_VEC = raw"""
data{int<lower=0> n;int<lower=0> K;matrix[n,K] X;array[n] int C;array[n] int N;}
parameters{real alpha;vector[K] beta;}
model{target+=normal_lpdf(alpha|0,100);target+=normal_lpdf(beta|0,100);
 target+=binomial_logit_lpmf(C|N,alpha+X*beta);}
"""
Turing.@model function turing_bin(X, C, N, K)
    alpha ~ Distributions.Normal(0, 100); beta ~ Distributions.MvNormal(zeros(K), 100.0^2 * I)
    lp = alpha .+ X * beta
    C ~ Distributions.product_distribution([Turing.BinomialLogit(N[i], lp[i]) for i in eachindex(N)])
end
bin_alpha(q) = q[1]; bin_beta(q) = @view q[2:end]
bin_eta(a, b, X) = a .+ X * b; bin_prior(q) = sum(x -> nlp(x, 0, 100), q)
bin_ll(C, Nf, eta, cterm) = cterm + dot(C, eta) - dot(Nf, log1pexp.(eta))
bin_post(p, l) = p + l
rk_bin = @kernel model(unconstrained, X, C, Nf, cterm) = begin
    a = bin_alpha(unconstrained); b = bin_beta(unconstrained); eta = bin_eta(a, b, X)
    prior = bin_prior(unconstrained); likelihood = bin_ll(C, Nf, eta, cterm); posterior = bin_post(prior, likelihood)
    return posterior
end

# ---------- generic runner ----------
function fd_grad(f, q); g = similar(q); h = 1e-6
    for i in eachindex(q); qp = copy(q); qp[i] += h; qm = copy(q); qm[i] -= h; g[i] = (f(qp) - f(qm)) / (2h); end; g
end

function run_model(name, q, oracle, stan_specs, ldf, kern_b, prep, gbuf, glm_baseline)
    ref = oracle(q); gref = fd_grad(oracle, q); gscale = maximum(abs, gref)
    gerr(g) = maximum(abs, g .- gref) / gscale
    rows = Dict{String,Any}[]
    add!(kind, impl, r) = push!(rows, Dict("kind" => kind, "impl" => impl, "median_ns" => r.median_ns, "median_bytes" => r.median_bytes))
    # parity + primal for each stan variant
    for (impl, sm) in stan_specs
        gerr(BridgeStan.log_density_gradient(sm, q; propto = false, jacobian = true)[2]) < 1e-3 || error("parity $name/$impl")
        add!("primal", impl, bench(() -> BridgeStan.log_density(sm, q; propto = false, jacobian = true)))
    end
    gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, q)[2]) < 1e-3 || error("parity $name/turing")
    gerr(ReactiveKernels.ad_value_and_gradient!(prep, gbuf, q)[2]) < 1e-3 || error("parity $name/rk")
    add!("primal", "turing", bench(() -> DynamicPPL.LogDensityProblems.logdensity(ldf, q)))
    add!("primal", "rk", bench(() -> kern_b(q)))
    # gradient: optimized-stan baseline, turing, rk
    base_sm = Dict(stan_specs)[glm_baseline]
    add!("gradient", "stan", bench(() -> BridgeStan.log_density_gradient(base_sm, q; propto = false, jacobian = true)))
    add!("gradient", "turing", bench(() -> DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf, q)))
    add!("gradient", "rk", bench(() -> ReactiveKernels.ad_value_and_gradient!(prep, gbuf, q)))
    println("\n===== $name  dim=$(length(q)) =====")
    for kind in ("primal", "gradient")
        bkey = kind == "primal" ? glm_baseline : "stan"
        base = first(r["median_ns"] for r in rows if r["kind"] == kind && r["impl"] == bkey)
        println("  -- $kind (baseline optimized Stan) --")
        for r in rows; r["kind"] == kind || continue
            println("     $(rpad(r["impl"],12)) $(lpad(fmt(r["median_ns"]),10))  $(lpad(string(round(r["median_ns"]/base; digits=2))*"x",8))  $(r["median_bytes"])B")
        end
    end
    Dict("model" => name, "dim" => length(q), "measurements" => rows)
end

results = Any[]

# GLM_Poisson
let
    X, C = gen_pois(100); q = [0.7, 0.3, -0.15, 0.08]; Cf = Float64.(C); cterm = sum(c -> loggamma(c + 1.0), C)
    dj = jobj(["n" => 100, "K" => 3, "X" => jmat(X), "C" => jvec(C)])
    stans = ["stan_naive" => stan_model(STAN_POIS_NAIVE, dj, "gp_naive"),
             "stan_vec" => stan_model(STAN_POIS_VEC, dj, "gp_vec"),
             "stan_glm" => stan_model(STAN_POIS_GLM, dj, "gp_glm")]
    ldf = stable_ldf(turing_pois(X, C, 3); adtype = ADTypes.AutoMooncake(config = nothing))
    kb = prepare(rk_pois; have = (:unconstrained, :X, :C, :cterm), want = :posterior, bound = (; X = X, C = Cf, cterm = cterm))
    prep = prepare_ad(kb, AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const), q; active = :unconstrained)
    push!(results, run_model("GLM_Poisson", q, qq -> oracle_pois(qq, X, C), stans, ldf, kb, prep, similar(q), "stan_glm"))
end
# GLM_Binomial
let
    X, C, N = gen_bin(100); q = [0.1, 0.5, -0.2]; Cf = Float64.(C); Nf = Float64.(N); cterm = sum(lchoose(N[i], C[i]) for i in eachindex(C))
    dj = jobj(["n" => 100, "K" => 2, "X" => jmat(X), "C" => jvec(C), "N" => jvec(N)])
    stans = ["stan_naive" => stan_model(STAN_BIN_NAIVE, dj, "gb_naive"),
             "stan_vec" => stan_model(STAN_BIN_VEC, dj, "gb_vec")]
    ldf = stable_ldf(turing_bin(X, C, N, 2); adtype = ADTypes.AutoMooncake(config = nothing))
    kb = prepare(rk_bin; have = (:unconstrained, :X, :C, :Nf, :cterm), want = :posterior, bound = (; X = X, C = Cf, Nf = Nf, cterm = cterm))
    prep = prepare_ad(kb, AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const), q; active = :unconstrained)
    push!(results, run_model("GLM_Binomial", q, qq -> oracle_bin(qq, X, C, N), stans, ldf, kb, prep, similar(q), "stan_vec"))
end

outp = get(ENV, "AS_OUTPUT", "")
if outp != ""
    receipt = Dict("schema" => "fair-posteriordb-glm-v1", "generated_at" => string(now()),
        "methodology" => "median over $ROUNDS rounds of BenchmarkTools minimum(200 samples); setup/compile/first-call excluded; RK vs optimized Stan (naive/vectorized/glm) vs optimized Turing; weak-normal priors; parity vs FD oracle",
        "environment" => Dict("julia" => string(VERSION), "arch" => string(Sys.ARCH), "cpu" => Sys.cpu_info()[1].model, "stan" => "2.39.0 (BridgeStan 2.9.0)"),
        "models" => results)
    open(outp, "w") do io; TOML.print(io, receipt); end
    println("\nwrote receipt: $outp")
end
println("\nFAIR_GLM_DONE")
