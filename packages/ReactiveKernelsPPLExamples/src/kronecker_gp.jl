module KroneckerGpExample

using ReactiveKernels
using LinearAlgebra: Symmetric
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export KRON_X1, KRON_Y
export build_kronecker_gp_graph, demo
export KRON_SOURCE, evaluate_kronecker_gp_source

# posteriordb `synthetic_grid_RBF_kernels-kronecker_gp` — a Kronecker-structured
# GP over a 2-D grid: one RBF-kernel margin (var1, bw1) and one correlation
# margin (Cholesky factor L with an LKJ(2) prior) combine as
# K = K1 ⊗ K2 + sigma1², diagonalized exactly through the symmetric
# eigendecompositions of both margins (Stan's `eigenvectors_sym` /
# `eigenvalues_sym`), so the marginal likelihood is
#   -0.5 · sum(y ⊙ (Q1ᵀ ⊗ Q2ᵀ) y ⊘ (R2R1ᵀ + sigma1)) − 0.5 · sum(log(R2R1ᵀ + sigma1)).
#
#   var1  ~ LogNormal(0, 1)          (lower=0 ⇒ exp transform, Jacobian u)
#   bw1   ~ Cauchy(0, 2.5)           (lower=0 ⇒ exp transform, Jacobian u)
#   sigma1~ LogNormal(0, 1)          (lower=1e-5 ⇒ exp transform, Jacobian u)
#   L     ~ LKJ-Corr-Cholesky(2)     (Stan's exact tanh partial-sum transform)
#
# The data-derived squared-distance matrix xd is computed IN-GRAPH from the
# bound locations x1, exactly Stan's transformed-data loop. Data: n1 = n2 = 30
# (the reference model's data contract requires length(x1) = n1), y is 30×30.
let d = _posteriordb_data("synthetic_grid_RBF_kernels-kronecker_gp")
    global const KRON_X1 = Float64.(d["x1"])
    global const KRON_Y = Float64.(d["y"] isa AbstractMatrix ? d["y"] :
        reduce(vcat, d["y"]))
end

const KRON_SOURCE = raw"""
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    lognormal, lkj_corr_cholesky
using LinearAlgebra: I, Symmetric, eigen
# Owned symmetric-eigendecomposition rules (native Enzyme reverse through the
# margins; raw `eigen(Symmetric(...))` reaches LAPACK `syevr!`, which Enzyme
# cannot differentiate).
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    rk_symmetric_eigvals, rk_symmetric_eigvecs

# --- Stan `kron_mvprod`: (A ⊗ B) v without materializing the Kronecker product,
# where V = reshape(v, n2, n1). ---
_kron_mvprod(A, B, V) = transpose(A * transpose(B * V))

# --- Stan `cholesky_corr_constrain` (the constraining transform ONLY; the
# log-Jacobian terms are accumulated by `_ccl_constraint_lp` below so each
# helper stays single-valued). z holds the row-major lower-triangle of the
# K×K factor; row i (2..K) consumes z entries (1..i-1). ---
#
# Both helpers are spelled WITHOUT scalar indexing so the all-bound posterior
# query traces under Reactant (`z[k]` on a traced z is refused by the
# scalar-indexing guard). The spelling is a closed form, not an unrolled
# loop: with s_j the running within-row sum of squares, each step writes
# w_j = z_j·√(1−s_{j−1}) and s_j = s_{j−1}+w_j², hence
# 1−s_j = (1−s_{j−1})·(1−z_j²) and log(1−s_j) is a within-row prefix sum of
# t = log(1−z²). Row maps come from triangular numbers: position p sits in
# row i(p) = ⌈(1+√(1+8p))/2⌉ at within-row index m(p) = p−(i−1)(i−2)/2.
# docs/src/constraints.md forbids trace-time unrolling a data-derived trip
# count (K is bound data), so no `for i in 2:K` appears here.
_cholesky_corr_constrain_L(z, K) = begin
    n_free = length(z)
    @assert n_free == (K * (K - 1)) ÷ 2
    t = log.(1 .- z .^ 2)
    ps = 1:n_free
    irow = ceil.(Int, (1 .+ sqrt.(1 .+ 8 .* ps)) ./ 2)
    # Within-row prefix sums of t: M[p,q] = 1 iff q is an earlier position
    # in p's row, so M*t holds the √(1−s) argument of every position.
    M = Float64.((irow .== reshape(irow, 1, n_free)) .&
                 (reshape(ps, 1, n_free) .< reshape(ps, n_free, 1)))
    ze = z .* exp.(0.5 .* (M * t))
    Irow = (1:K) .+ zeros(Int, 1, K)
    Jcol = zeros(Int, K, 1) .+ (1:K)'
    posmat = ifelse.(Jcol .< Irow,
                     (Irow .- 1) .* (Irow .- 2) .÷ 2 .+ Jcol, 1)
    strict = Float64.(Jcol .< Irow) .* ze[posmat]
    # Row totals of t (row 1 contributes nothing, so d[1] = exp(0) = 1).
    R = Float64.(((1:K) .== reshape(irow, 1, n_free)) .& ((1:K) .> 1))
    d = exp.(0.5 .* (R * t))
    strict + Float64.(Irow .== Jcol) .* reshape(d, K, 1)
end

# The `0.5·log(1 − sum_sqs)` partial-sum Jacobian terms of Stan's
# `cholesky_corr_constrain` (the tanh `corr_constrain` terms are added in-graph).
# Closed form of the same recurrence `_cholesky_corr_constrain_L` evaluates:
# each t_m = log(1−z_m²) is counted (i−1−m) times in its row, so the
# position weight is max(i−1−m, 0) — broadcast, gather-free, no scalar reads.
_ccl_constraint_lp(z, K) = begin
    n_free = length(z)
    @assert n_free == (K * (K - 1)) ÷ 2
    ps = 1:n_free
    irow = ceil.(Int, (1 .+ sqrt.(1 .+ 8 .* ps)) ./ 2)
    m = ps .- (irow .- 1) .* (irow .- 2) .÷ 2
    w = max.(irow .- 1 .- m, 0)
    0.5 * sum(w .* log.(1 .- z .^ 2))
end

@kernel model(unconstrained::Vector{Float64},
              x1::Vector{Float64}, y::Matrix{Float64}) = begin
    n2::Int = size(y, 1)
    n1::Int = size(y, 2)

    # Stan unconstrained order: var1, bw1, L_free[(n2(n2−1))/2], sigma1.
    u_var1::Float64 = unconstrained[1]
    u_bw1::Float64 = unconstrained[2]
    n_free::Int = (n2 * (n2 - 1)) ÷ 2
    L_free::AbstractVector{Float64} = view(unconstrained, 3:(2 + n_free))
    u_sigma1::Float64 = unconstrained[3 + n_free]

    # Lower-bound transforms (exp) with their log-Jacobians; sigma1's bound is
    # 1e-5, so Stan constrains it as 1e-5 + exp(u) (Jacobian still u).
    var1::Float64 = exp(u_var1)
    bw1::Float64 = exp(u_bw1)
    sigma1::Float64 = 0.00001 + exp(u_sigma1)

    # LKJ-Cholesky transform: corr_constrain (tanh) terms + the partial-sum
    # terms, exactly Stan's `cholesky_corr_constrain(y, K, lp)` with Jacobian.
    z::Vector{Float64} = tanh.(L_free)
    corr_lp::Float64 = sum(log.(1.0 .- z .^ 2))
    partial_lp::Float64 = _ccl_constraint_lp(z, n2)
    L::Matrix{Float64} = _cholesky_corr_constrain_L(z, n2)
    log_jacobian::Float64 = u_var1 + u_bw1 + u_sigma1 + corr_lp + partial_lp

    # Priors.
    var1_prior::Float64 = lognormal(0.0, 1.0).logpdf(var1)
    sigma1_prior::Float64 = lognormal(0.0, 1.0).logpdf(sigma1)
    bw1_prior::Float64 =
        -(log(pi) + log(2.5) + log1p((bw1 / 2.5) * (bw1 / 2.5)))
    L_prior::Float64 = lkj_corr_cholesky(2.0).logpdf(L)
    prior::Float64 = var1_prior + bw1_prior + sigma1_prior + L_prior

    # Stan transformed data: xd[i,j] = −(x1[i] − x1[j])² (the RBF exponent arg).
    xd::Matrix{Float64} = -((x1 .- transpose(x1)) .^ 2)
    xd1::Matrix{Float64} = xd[1:n1, 1:n1]

    # Transformed parameters: both margin eigendecompositions (exact, dense).
    Sigma1::Matrix{Float64} = var1 .* exp.(xd1 .* bw1)
    Sigma1_reg::Matrix{Float64} = Sigma1 + 0.00001 .* Matrix{Float64}(I, n1, n1)
    Q1::Matrix{Float64} = rk_symmetric_eigvecs(Sigma1_reg)
    R1::Vector{Float64} = rk_symmetric_eigvals(Sigma1_reg)
    Lambda::Matrix{Float64} = L * transpose(L)
    Q2::Matrix{Float64} = rk_symmetric_eigvecs(Lambda)
    R2::Vector{Float64} = rk_symmetric_eigvals(Lambda)

    # eigenvalues[i,j] = R2[i]·R1[j] + sigma1  (Stan `calculate_eigenvalues`).
    eigenvalues::Matrix{Float64} = R2 .* transpose(R1) .+ sigma1

    # Marginal likelihood through the Kronecker eigenspace.
    # Stan whitens y with (Q1ᵀ⊗Q2ᵀ), divides the whitened coefficients
    # elementwise by the product eigenvalues, rotates back with (Q1⊗Q2), and
    # contracts against y — the `./ eigenvalues` binds INSIDE the outer
    # kron_mvprod's V argument (verified against the generated C++).
    whitened::Matrix{Float64} = _kron_mvprod(transpose(Q1), transpose(Q2), y)
    scaled::Matrix{Float64} = whitened ./ eigenvalues
    rotated::Matrix{Float64} = _kron_mvprod(Q1, Q2, scaled)
    quad::Matrix{Float64} = y .* rotated
    likelihood::Float64 = -0.5 * sum(quad) - 0.5 * sum(log.(eigenvalues))

    posterior::Float64 = prior + likelihood + log_jacobian

    return posterior
end

q = let dim = 2 + (size(KRON_Y, 1) * (size(KRON_Y, 1) - 1)) ÷ 2 + 1
    # Deterministic small coordinates in (−0.1, 0.1).
    0.1 .* sin.((1:dim) .+ 0.5)
end
x1 = KRON_X1
y = KRON_Y

requested_nodes = (:prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :x1, :y),
    want = requested_nodes,
    bound = (; x1, y))

output = density_kernel(q)
prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isfinite(posterior)

docs_example = (;
    name = :kronecker_gp_posterior,
    origin = "posteriordb kronecker_gp — Kronecker-structured GP over a 2-D grid",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    lognormal_object = lognormal,
    lkj_corr_cholesky_object = lkj_corr_cholesky,
)
"""

function evaluate_kronecker_gp_source(; model_only::Bool = false)
    _evaluate_ppl_source(KRON_SOURCE, @__MODULE__; bindings = (
        :KRON_X1, :KRON_Y,
    ), model_only)
end

const _KRON_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KRON_GRAPH_TEMPLATE[] = evaluate_kronecker_gp_source(; model_only = true).model
    nothing
end

"""
    build_kronecker_gp_graph()

Build the posteriordb `kronecker_gp` model (a Kronecker-structured GP over a
2-D grid) as a declarative `ReactiveKernels.KernelSpec`. The RBF margin
(`var1`, `bw1`) and the correlation margin (`L`, LKJ(2)) are each exactly
diagonalized in-graph through the owned `rk_symmetric_eigvecs` /
`rk_symmetric_eigvals` rules (`eigen(Symmetric(·))` with generated AD rules);
the marginal likelihood is the Stan `kron_mvprod` contraction against the
product eigenspectrum. The
LKJ-Cholesky constraining transform (tanh partial-sum form) and its exact
log-Jacobian are authored in-graph; the squared-distance data matrix `xd` is
derived in-graph from the bound locations `x1`. Data-generic: dims read from
data.
"""
function build_kronecker_gp_graph()
    compose(_KRON_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kronecker_gp_graph()
    posterior_plan = plan(model;
                          have = (:unconstrained, :x1, :y),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    nothing
end

end # module KroneckerGpExample

if abspath(PROGRAM_FILE) == @__FILE__
    KroneckerGpExample.demo()
end
