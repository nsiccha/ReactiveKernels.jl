module NNRBMExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export NN_RBM_X, NN_RBM_Y, NN_RBM_K, NN_RBM_J
export build_nn_rbm_graph, nn_rbm_fixture, demo
export NN_RBM_SOURCE, evaluate_nn_rbm_source

# posteriordb `nn_rbm1bJ{10,100}` — a single-hidden-layer neural-network softmax
# classifier in the RBM parametrization of Lampinen & Vehtari (2001). The two
# posteriordb models (`mnist_100-nn_rbm1bJ10`, `mnist-nn_rbm1bJ100`) share ONE
# data-generic graph: J (number of hidden units) is a HAVE port, K the class
# count, and the design matrix `x`, labels `y` are data. This module executes
# the graph on the small `mnist_100` (N=100) example; the benchmark binds full
# MNIST (N=60000) with J=100 to the same authored source.
#
# Parameters (Stan declaration order, matrices column-major in the unconstrained
# vector): sigma2_alpha>0, sigma2_beta>0, alpha[M,J], beta[J,K-1], alpha1[J],
# beta1[K-1]. The two positive scales use the exp transform (each Jacobian += its
# log-argument). The forward pass is v = [1  tanh(x*alpha .+ alpha1)*beta .+ beta1]
# per observation (class 1 is the reference with a fixed logit of 1, exactly as
# the .stan `append_col(ones, ...)` writes it), and each label is a
# `categorical_logit` over that observation's K logits. This is NOT the existing
# `mnist_logistic` model (that is a plain multinomial-logistic classifier with a
# zero-logit reference and no hidden layer).

# Real `mnist_100` data (N=100 train images, 784 pixels, 10 classes), loaded via
# PosteriorDB.jl. The benchmark rebinds full MNIST via the same HAVE ports.
let d = _posteriordb_data("mnist_100-nn_rbm1bJ10")
    global const NN_RBM_X = Matrix{Float64}(d["x"])   # 100 × 784
    global const NN_RBM_Y = Int.(d["y"])              # 1-based class indices 1..10
    global const NN_RBM_K = Int(d["K"])               # 10
    global const NN_RBM_J = 10                         # hidden units (transformed data)
end

const NN_RBM_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, inverse_gamma, categorical_logit

@kernel model(unconstrained::Vector{Float64},
              x::Matrix{Float64}, y::Vector{Int}, K::Int, J::Int) = begin
    N::Int = size(x, 1)
    M::Int = size(x, 2)

    # transformed-data prior scales (Lampinen & Vehtari 2001). nu = 0.5 so
    # 1/nu = 2 and s2_0 = (0.05 / dim^2)^2. Data-only (folds under `bound`).
    nu_alpha::Float64 = 0.5
    s2_0_alpha::Float64 = (0.05 / M^(1.0 / nu_alpha))^2
    nu_beta::Float64 = 0.5
    s2_0_beta::Float64 = (0.05 / J^(1.0 / nu_beta))^2

    # unpack unconstrained, Stan declaration order (matrices column-major):
    #   sigma2_alpha, sigma2_beta, alpha[M,J], beta[J,K-1], alpha1[J], beta1[K-1]
    u_s2a::Float64 = unconstrained[1]
    u_s2b::Float64 = unconstrained[2]
    sigma2_alpha::Float64 = exp(u_s2a)
    sigma2_beta::Float64 = exp(u_s2b)
    off_a::Int = 2
    alpha::Matrix{Float64} =
        reshape(view(unconstrained, (off_a + 1):(off_a + M * J)), M, J)
    off_b::Int = off_a + M * J
    beta::Matrix{Float64} =
        reshape(view(unconstrained, (off_b + 1):(off_b + J * (K - 1))), J, K - 1)
    off_a1::Int = off_b + J * (K - 1)
    alpha1::Vector{Float64} = unconstrained[(off_a1 + 1):(off_a1 + J)]
    off_b1::Int = off_a1 + J
    beta1::Vector{Float64} = unconstrained[(off_b1 + 1):(off_b1 + K - 1)]

    # positive-scalar (exp) transform Jacobians: each contributes its log-argument
    log_jacobian::Float64 = u_s2a + u_s2b

    # forward pass. x1 * append_row(alpha1, alpha) = ones*alpha1 + x*alpha, so the
    # hidden activations are tanh(x*alpha .+ alpha1); likewise the output layer is
    # tanh_out*beta .+ beta1. The K logits per observation are [1  H*beta .+ beta1]:
    # class 1 is the reference with the fixed logit 1 (the .stan ones column).
    pre::Matrix{Float64} = x * alpha .+ transpose(alpha1)     # N × J
    H::Matrix{Float64} = tanh.(pre)
    HB::Matrix{Float64} = H * beta .+ transpose(beta1)        # N × (K-1)
    nonref::Matrix{Float64} = permutedims(HB)                 # (K-1) × N
    logits::Matrix{Float64} = vcat(ones(1, N), nonref)        # K × N (row 1 = 1s)

    # priors. to_vector(alpha) ~ Normal(0, sqrt(sigma2_alpha)) is a shared-scale
    # whole-vector reduction (one normalization, not a per-cell plate); alpha1,
    # beta1 reuse the standard `normal` object; the two scales reuse `inverse_gamma`.
    sd_alpha::Float64 = sqrt(sigma2_alpha)
    sd_beta::Float64 = sqrt(sigma2_beta)
    n_a::Int = M * J
    n_b::Int = J * (K - 1)
    alpha_prior::Float64 =
        n_a * (-0.5 * log(2π) - log(sd_alpha)) - 0.5 * sum(alpha .^ 2) / sigma2_alpha
    beta_prior::Float64 =
        n_b * (-0.5 * log(2π) - log(sd_beta)) - 0.5 * sum(beta .^ 2) / sigma2_beta
    alpha1_terms = plate(alpha1) do a
        normal(0.0, 1.0).logpdf(a)
    end
    beta1_terms = plate(beta1) do b
        normal(0.0, 1.0).logpdf(b)
    end
    s2a_prior::Float64 =
        inverse_gamma(nu_alpha / 2, nu_alpha * s2_0_alpha / 2).logpdf(sigma2_alpha)
    s2b_prior::Float64 =
        inverse_gamma(nu_beta / 2, nu_beta * s2_0_beta / 2).logpdf(sigma2_beta)
    prior::Float64 =
        alpha_prior + beta_prior + sum(alpha1_terms) + sum(beta1_terms) +
        s2a_prior + s2b_prior

    # likelihood: y[n] ~ categorical_logit(v[n]) over the K logits column
    pointwise = plate(eachcol(logits), y) do lc, yn
        categorical_logit(lc).logpdf(yn)
    end
    likelihood::Float64 = sum(pointwise)

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

x = NN_RBM_X
y = NN_RBM_Y
K = NN_RBM_K
J = NN_RBM_J
M = size(x, 2)
dim = 2 + M * J + J * (K - 1) + J + (K - 1)
q = 0.05 .* collect(1.0:dim) .- 0.1

# One plan extracts prior, summed likelihood, Jacobian, and the joint density from
# the packed sampler vector; a pointwise cut is an alternate query on the same plate.
requested_nodes = (:prior, :likelihood, :log_jacobian, :posterior)
evaluation_kernel = prepare(model;
    have = (:unconstrained, :x, :y, :K, :J),
    want = requested_nodes,
    bound = (; x, y, K, J))
output = evaluation_kernel(q)
prior, likelihood, log_jacobian, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ prior + likelihood + log_jacobian

pointwise_extraction = prepare(model;
    have = (:unconstrained, :x, :y, :K, :J),
    want = :pointwise,
    bound = (; x, y, K, J))
pointwise = pointwise_extraction(q)
@assert likelihood ≈ sum(pointwise)

docs_example = (;
    name = :nn_rbm_density,
    origin = "posteriordb nn_rbm1b — single-hidden-layer neural-network softmax classifier (RBM parametrization; build executed on mnist_100)",
    inputs = (; q),
    model,
    kernel = evaluation_kernel,
    output,
    requested_nodes,
    pointwise_extraction,
    pointwise,
    normal_object = normal,
    inverse_gamma_object = inverse_gamma,
    categorical_logit_object = categorical_logit,
)
"""

function evaluate_nn_rbm_source(; model_only::Bool = false)
    _evaluate_ppl_source(NN_RBM_SOURCE, @__MODULE__; bindings = (
        :NN_RBM_X, :NN_RBM_Y, :NN_RBM_K, :NN_RBM_J,
    ), model_only)
end

"""
    nn_rbm_fixture()

The `mnist_100` example inputs as `(; x, y, K, J)` — `x` is `100 × 784`, `y` are
one-based class indices, `K = 10`, `J = 10`. The benchmark binds full MNIST
(`N = 60000`, `J = 100`) to the same authored source instead.
"""
nn_rbm_fixture() = (; x = NN_RBM_X, y = NN_RBM_Y, K = NN_RBM_K, J = NN_RBM_J)

const _NN_RBM_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _NN_RBM_GRAPH_TEMPLATE[] = evaluate_nn_rbm_source(; model_only = true).model
    nothing
end

"""
    build_nn_rbm_graph()

Build the posteriordb `nn_rbm1b` single-hidden-layer neural-network softmax
classifier as a declarative `ReactiveKernels.KernelSpec`. The graph is
data-generic: bind `x`, `y`, `K`, and the hidden-unit count `J` at `prepare`
time (`have = (:unconstrained, :x, :y, :K, :J)`), so the SAME source certifies
both `mnist_100-nn_rbm1bJ10` and `mnist-nn_rbm1bJ100`. Two positive scales use
the exp transform (+ Jacobian); the forward pass is a `tanh` hidden layer feeding
a reference-coded softmax whose class-1 logit is fixed at 1. Named nodes for the
prior, per-observation and summed likelihood, transform Jacobian, and the
unconstrained joint density.
"""
build_nn_rbm_graph() = compose(_NN_RBM_GRAPH_TEMPLATE[])

function demo()
    model = build_nn_rbm_graph()
    fx = nn_rbm_fixture()
    x, y, K, J = fx.x, fx.y, fx.K, fx.J
    M = size(x, 2)
    dim = 2 + M * J + J * (K - 1) + J + (K - 1)
    q = 0.05 .* collect(1.0:dim) .- 0.1
    kb = prepare(model; have = (:unconstrained, :x, :y, :K, :J),
        want = :posterior, bound = (; x, y, K, J))
    println("nn_rbm1bJ$J unconstrained log posterior (mnist_100) = ", kb(q))
    nothing
end

end # module NNRBMExample

if abspath(PROGRAM_FILE) == @__FILE__
    NNRBMExample.demo()
end
