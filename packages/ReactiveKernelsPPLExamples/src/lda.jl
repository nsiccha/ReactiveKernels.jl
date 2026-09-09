module LDAExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export LDA_DOC, LDA_W, LDA_ALPHA, LDA_BETA, LDA_M
export build_lda_graph, lda_fixture, demo
export LDA_SOURCE, evaluate_lda_source

# posteriordb `ldaK2` / `ldaK5` — latent Dirichlet allocation with the discrete
# topic assignment marginalized analytically per word (`log_sum_exp` over topics),
# exactly as the .stan model block writes it. The two posteriordb models
# (`three_men1-ldaK2`, `prideprejudice_chapter-ldaK5`) share ONE data-generic
# graph. Its HAVE ports are exactly `unconstrained`, `doc`, `w`, `alpha`, `beta`,
# and `M`: the concentration vectors carry the dimensions (K = length(alpha),
# V = length(beta)); for ldaK2 alpha/beta are the .stan transformed-data
# ones-vectors, while for ldaK5 they are data. This module
# executes the graph on `three_men1` (K=2); the benchmark also binds
# `prideprejudice_chapter` (K=5).
#
# Parameters (Stan declaration order): array[M] simplex[K] theta, then
# array[K] simplex[V] phi. Each simplex uses the Stan-2.39 inverse-ILR transform,
# x = softmax(sum_to_zero_constrain(y)), whose log-abs-Jacobian is
# sum(log x) + 0.5*log(dim). The sum-to-zero basis Wstz (a function of the simplex
# dimension only) is built IN-GRAPH from the bound dimension, so ONLY raw data is
# bound. Priors theta[m] ~ Dirichlet(alpha), phi[k] ~ Dirichlet(beta); the
# per-word likelihood is sum_n log_sum_exp_k (log theta[doc_n,k] + log phi[k,w_n]).

# Real `three_men1` data (K=2 posterior; V=249, M=6, N=4999), loaded via
# PosteriorDB.jl. For ldaK2 the .stan computes alpha=ones(K), beta=ones(V) in
# transformed data, so they are bound here as the corresponding ones-vectors.
let d = _posteriordb_data("three_men1-ldaK2")
    global const LDA_DOC = Int.(d["doc"])
    global const LDA_W = Int.(d["w"])
    global const LDA_M = Int(d["M"])
    global const LDA_ALPHA = ones(2)               # ldaK2 transformed-data topic prior
    global const LDA_BETA = ones(Int(d["V"]))      # ldaK2 transformed-data word prior
end

const LDA_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: dirichlet
using LogExpFunctions: logsumexp

@kernel model(unconstrained::Vector{Float64},
              doc::Vector{Int}, w::Vector{Int},
              alpha::Vector{Float64}, beta::Vector{Float64}, M::Int) = begin
    K::Int = length(alpha)
    V::Int = length(beta)
    N::Int = length(w)

    # Stan-2.39 simplex transform is the inverse ILR, x = softmax(z) with
    # z = sum_to_zero_constrain(y). sum_to_zero_constrain is the fixed linear map
    # z = Wstz*y; Wstz(D) is D×(D-1) with W[p,j] = c_j (j>=p) - (p-1)c_j (j==p-1),
    # c_j = 1/sqrt(j(j+1)). Built here in-graph from the bound simplex dimension
    # (data-only, folds under `bound`), so only raw data is bound.
    jK = collect(1:(K - 1))
    cK::Vector{Float64} = 1.0 ./ sqrt.(jK .* (jK .+ 1.0))
    pK = collect(1:K)
    Wstz_K::Matrix{Float64} =
        (transpose(jK) .>= pK) .* transpose(cK) .+
        (transpose(jK) .== (pK .- 1)) .* ((-(pK .- 1)) .* transpose(cK))
    jV = collect(1:(V - 1))
    cV::Vector{Float64} = 1.0 ./ sqrt.(jV .* (jV .+ 1.0))
    pV = collect(1:V)
    Wstz_V::Matrix{Float64} =
        (transpose(jV) .>= pV) .* transpose(cV) .+
        (transpose(jV) .== (pV .- 1)) .* ((-(pV .- 1)) .* transpose(cV))

    # unpack: theta (M blocks of K-1) then phi (K blocks of V-1), Stan colmajor
    nth::Int = M * (K - 1)
    ThetaU::Matrix{Float64} = reshape(view(unconstrained, 1:nth), K - 1, M)
    PhiU::Matrix{Float64} =
        reshape(view(unconstrained, (nth + 1):length(unconstrained)), V - 1, K)

    # constrain: z = Wstz*U per column, softmax; carry the log-probabilities
    Zt::Matrix{Float64} = Wstz_K * ThetaU               # K × M
    Zp::Matrix{Float64} = Wstz_V * PhiU                 # V × K
    lse_theta = plate(eachcol(Zt)) do zc
        logsumexp(zc)
    end
    lse_phi = plate(eachcol(Zp)) do zc
        logsumexp(zc)
    end
    LOG_THETA::Matrix{Float64} = Zt .- transpose(lse_theta)   # K × M
    LOG_PHI::Matrix{Float64} = Zp .- transpose(lse_phi)       # V × K
    THETA::Matrix{Float64} = exp.(LOG_THETA)
    PHI::Matrix{Float64} = exp.(LOG_PHI)

    # log-abs-Jacobian: per simplex sum(log x) + 0.5*log(dim)
    log_jacobian::Float64 =
        sum(LOG_THETA) + M * (0.5 * log(K)) +
        sum(LOG_PHI) + K * (0.5 * log(V))

    # Dirichlet priors (shared `dirichlet` object per simplex column; the shared
    # concentration vector is passed via Ref so it is a declared caller port)
    theta_prior_terms = plate(eachcol(THETA), Ref(alpha)) do th, a
        dirichlet(a).logpdf(th)
    end
    phi_prior_terms = plate(eachcol(PHI), Ref(beta)) do ph, b
        dirichlet(b).logpdf(ph)
    end
    prior::Float64 = sum(theta_prior_terms) + sum(phi_prior_terms)

    # likelihood: sum_n log_sum_exp_k (log theta[doc_n,k] + log phi[k,w_n]).
    # flat gathers with per-word column indices derived in-graph from raw doc,w:
    #   LOG_THETA[k,doc_n] = lt[(doc_n-1)K + k];  LOG_PHI[w_n,k] = lp[(k-1)V + w_n].
    lt::Vector{Float64} = vec(LOG_THETA)
    lp::Vector{Float64} = vec(LOG_PHI)
    kcol = collect(1:K)
    idx_theta::Matrix{Int} = kcol .+ transpose((doc .- 1) .* K)     # K × N
    kVcol = collect(0:(K - 1)) .* V
    idx_phi::Matrix{Int} = kVcol .+ transpose(w)                    # K × N
    GAMMA::Matrix{Float64} = lt[idx_theta] .+ lp[idx_phi]           # K × N
    pointwise = plate(eachcol(GAMMA)) do g
        logsumexp(g)
    end
    likelihood::Float64 = sum(pointwise)

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

doc = LDA_DOC
w = LDA_W
alpha = LDA_ALPHA
beta = LDA_BETA
M = LDA_M
K = length(alpha)
V = length(beta)
dim = M * (K - 1) + K * (V - 1)
q = 0.1 .* sin.(collect(1.0:dim))

requested_nodes = (:prior, :likelihood, :log_jacobian, :posterior)
evaluation_kernel = prepare(model;
    have = (:unconstrained, :doc, :w, :alpha, :beta, :M),
    want = requested_nodes,
    bound = (; doc, w, alpha, beta, M))
output = evaluation_kernel(q)
prior, likelihood, log_jacobian, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ prior + likelihood + log_jacobian

pointwise_extraction = prepare(model;
    have = (:unconstrained, :doc, :w, :alpha, :beta, :M),
    want = :pointwise,
    bound = (; doc, w, alpha, beta, M))
pointwise = pointwise_extraction(q)
@assert likelihood ≈ sum(pointwise)

docs_example = (;
    name = :lda_density,
    origin = "posteriordb ldaK2/ldaK5 — latent Dirichlet allocation, topic marginalized (build executed on three_men1)",
    inputs = (; q),
    model,
    kernel = evaluation_kernel,
    output,
    requested_nodes,
    pointwise_extraction,
    pointwise,
    dirichlet_object = dirichlet,
)
"""

function evaluate_lda_source(; model_only::Bool = false)
    _evaluate_ppl_source(LDA_SOURCE, @__MODULE__; bindings = (
        :LDA_DOC, :LDA_W, :LDA_ALPHA, :LDA_BETA, :LDA_M,
    ), model_only)
end

"""
    lda_fixture()

The `three_men1` (K=2) example inputs as `(; doc, w, alpha, beta, M)` — the
ldaK2 transformed-data ones-vector priors `alpha = ones(2)`, `beta = ones(V)`.
The benchmark also binds `prideprejudice_chapter` (K=5, data-supplied alpha/beta)
to the same authored source.
"""
lda_fixture() = (; doc = LDA_DOC, w = LDA_W, alpha = LDA_ALPHA, beta = LDA_BETA, M = LDA_M)

const _LDA_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LDA_GRAPH_TEMPLATE[] = evaluate_lda_source(; model_only = true).model
    nothing
end

"""
    build_lda_graph()

Build the posteriordb `ldaK2`/`ldaK5` latent Dirichlet allocation model as a
declarative `ReactiveKernels.KernelSpec`. The graph is data-generic: bind `doc`,
`w`, the Dirichlet concentrations `alpha` (length K) and `beta` (length V), and
the document count `M` at `prepare` time
(`have = (:unconstrained, :doc, :w, :alpha, :beta, :M)`), so the SAME source
certifies both `three_men1-ldaK2` and `prideprejudice_chapter-ldaK5`. Each
simplex uses the Stan-2.39 inverse-ILR transform (sum-to-zero basis built
in-graph from the bound dimension) with its `sum(log x) + 0.5 log(dim)` Jacobian;
the discrete topic label is marginalized per word via a stable `log_sum_exp`.
Named nodes for the Dirichlet prior, per-word and summed likelihood, transform
Jacobian, and the unconstrained joint density.
"""
build_lda_graph() = compose(_LDA_GRAPH_TEMPLATE[])

function demo()
    model = build_lda_graph()
    fx = lda_fixture()
    doc, w, alpha, beta, M = fx.doc, fx.w, fx.alpha, fx.beta, fx.M
    K = length(alpha); V = length(beta)
    dim = M * (K - 1) + K * (V - 1)
    q = 0.1 .* sin.(collect(1.0:dim))
    kb = prepare(model; have = (:unconstrained, :doc, :w, :alpha, :beta, :M),
        want = :posterior, bound = (; doc, w, alpha, beta, M))
    println("ldaK$K unconstrained log posterior (three_men1) = ", kb(q))
    nothing
end

end # module LDAExample

if abspath(PROGRAM_FILE) == @__FILE__
    LDAExample.demo()
end
