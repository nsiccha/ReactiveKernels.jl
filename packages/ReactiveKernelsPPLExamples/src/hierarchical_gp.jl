module HierarchicalGPExample

using ReactiveKernels
using LinearAlgebra
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export HGP_Y, HGP_YEAR_IND, HGP_STATE_IND, HGP_REGION_IND, HGP_STATE_REGION_IND
export HGP_N_YEARS, HGP_N_REGIONS, HGP_N_STATES, HGP_N_YEARS_OBS
export build_hierarchical_gp_graph, demo
export HIERARCHICAL_GP_SOURCE, evaluate_hierarchical_gp_source

# posteriordb `state_wide_presidential_votes-hierarchical_gp` — a hierarchical
# Gaussian-process model of state presidential vote shares (Trangucci StanCon
# 2017): per-year region and state GPs (each a sum of a long- and short-range
# exponential-quadratic kernel over the year axis, non-centered through the
# Cholesky factor), plus year/state/region random effects, with a Dirichlet
# variance decomposition. Real full data (N = 550) via PosteriorDB.jl.
let d = _posteriordb_data("state_wide_presidential_votes-hierarchical_gp")
    global const HGP_Y = Float64.(d["y"])                     # N observed vote shares
    global const HGP_YEAR_IND = Int.(d["year_ind"])           # N -> 1..N_years
    global const HGP_STATE_IND = Int.(d["state_ind"])         # N -> 1..N_states
    global const HGP_REGION_IND = Int.(d["region_ind"])       # N -> 1..N_regions
    global const HGP_STATE_REGION_IND = Int.(d["state_region_ind"])  # N_states -> 1..N_regions
    global const HGP_N_YEARS = Int(d["N_years"])              # 14
    global const HGP_N_REGIONS = Int(d["N_regions"])          # 10
    global const HGP_N_STATES = Int(d["N_states"])            # 50
    global const HGP_N_YEARS_OBS = Int(d["N_years_obs"])      # 11
end

const HIERARCHICAL_GP_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, dirichlet
using LinearAlgebra: I, Symmetric, cholesky

# Stan's simplex transform is softmax(sum_to_zero_constrain(y)); sum_to_zero is
# the LINEAR ILR "pivot coordinates" map, so it is a fixed K×(K-1) basis matrix.
_hgp_sum_to_zero(y) = begin
    N = length(y); z = zeros(N + 1); sum_w = 0.0
    for i in N:-1:1
        n = Float64(i); w = y[i] / sqrt(n * (n + 1)); sum_w += w
        z[i] += sum_w; z[i + 1] -= w * n
    end
    z
end
_hgp_stz_basis(K) =
    reduce(hcat, [_hgp_sum_to_zero(Float64.(1:(K-1) .== j)) for j in 1:(K-1)])
# transformed-data year axis and its squared-distance design.
_hgp_years(n) = collect(1.0:n)
_hgp_sq_dist(v) = [(v[i] - v[j])^2 for i in eachindex(v), j in eachindex(v)]
# column-major linear index of M[row_ind[n], col_ind[n]] for an nrows×* matrix M.
_hgp_lin(row_ind, col_ind, nrows) = (col_ind .- 1) .* nrows .+ row_ind
# weibull(alpha, sigma) log density (Stan propto=false), x>0.
_hgp_weibull(x, alpha, sigma) =
    log(alpha) - log(sigma) + (alpha - 1) * (log(x) - log(sigma)) - (x / sigma)^alpha

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64},
              year_ind::Vector{Int}, state_ind::Vector{Int},
              region_ind::Vector{Int}, state_region_ind::Vector{Int},
              N_years::Int, N_regions::Int, N_states::Int, N_years_obs::Int) = begin
    # Parameter block, Stan declaration order (matrices column-major). The GP
    # coefficient matrices are non-centered standard-normal draws.
    n_region_block::Int = N_years * N_regions
    n_state_block::Int  = N_years * N_states
    GP_region_std::Matrix{Float64} =
        reshape(view(unconstrained, 1:n_region_block), N_years, N_regions)
    GP_state_std::Matrix{Float64} =
        reshape(view(unconstrained, (n_region_block + 1):(n_region_block + n_state_block)),
                N_years, N_states)
    o2::Int = n_region_block + n_state_block
    year_std::AbstractVector{Float64}   = view(unconstrained, (o2 + 1):(o2 + N_years_obs))
    state_std::AbstractVector{Float64}  = view(unconstrained, (o2 + N_years_obs + 1):(o2 + N_years_obs + N_states))
    region_std::AbstractVector{Float64} = view(unconstrained, (o2 + N_years_obs + N_states + 1):(o2 + N_years_obs + N_states + N_regions))
    base::Int = o2 + N_years_obs + N_states + N_regions
    u_tot_var::Float64 = unconstrained[base + 1]
    prop_var_unc::AbstractVector{Float64} = view(unconstrained, (base + 2):(base + 17))
    mu::Float64 = unconstrained[base + 18]
    u_len_region_long::Float64  = unconstrained[base + 19]
    u_len_state_long::Float64   = unconstrained[base + 20]
    u_len_region_short::Float64 = unconstrained[base + 21]
    u_len_state_short::Float64  = unconstrained[base + 22]

    # Positive-constraint (exp) transforms with the exact lower=0 Jacobian.
    tot_var::Float64 = exp(u_tot_var)
    length_GP_region_long::Float64  = exp(u_len_region_long)
    length_GP_state_long::Float64   = exp(u_len_state_long)
    length_GP_region_short::Float64 = exp(u_len_region_short)
    length_GP_state_short::Float64  = exp(u_len_state_short)

    # Simplex transform prop_var = softmax(sum_to_zero_constrain(prop_var_unc));
    # the sum-to-zero map is the bound 17×16 ILR basis (folded by `bound=`).
    stz_basis::Matrix{Float64} = _hgp_stz_basis(17)
    z_raw::Vector{Float64} = stz_basis * prop_var_unc
    z_max::Float64 = maximum(z_raw)
    z_exp::Vector{Float64} = exp.(z_raw .- z_max)
    prop_var::Vector{Float64} = z_exp ./ sum(z_exp)
    # log|J| of the ILR simplex = Σ log(prop_var) + 0.5·log(K).
    jac_simplex::Float64 = sum(log.(prop_var)) + 0.5 * log(17.0)
    log_jacobian::Float64 = u_tot_var + u_len_region_long + u_len_state_long +
                            u_len_region_short + u_len_state_short + jac_simplex

    # Variance decomposition vars = 17·prop_var·tot_var and the derived SDs.
    vars::Vector{Float64} = (17.0 * tot_var) .* prop_var
    sigma_year::Float64   = sqrt(vars[1])
    sigma_region::Float64 = sqrt(vars[2])
    sigma_state::Vector{Float64} = sqrt.(view(vars, 3:12))     # one per region (10)
    sigma_GP_region_long::Float64  = sqrt(vars[13])
    sigma_GP_state_long::Float64   = sqrt(vars[14])
    sigma_GP_region_short::Float64 = sqrt(vars[15])
    sigma_GP_state_short::Float64  = sqrt(vars[16])
    sigma_error::Float64 = sqrt(vars[17])

    # Random effects. state_re scales each state's std by its region's SD.
    region_re::Vector{Float64} = sigma_region .* region_std
    year_re::Vector{Float64}   = sigma_year .* year_std
    state_re::Vector{Float64}  = sigma_state[state_region_ind] .* state_std

    # Per-year GP covariances (long + short exp-quad kernels over the year axis)
    # with the 1e-6 diagonal jitter; the non-centered GPs are L · std.
    sq_dist_years::Matrix{Float64} = _hgp_sq_dist(_hgp_years(N_years))
    jitter::Matrix{Float64} = 1.0e-6 .* Matrix(I, N_years, N_years)
    cov_region::Matrix{Float64} =
        sigma_GP_region_long^2 .* exp.(-0.5 .* sq_dist_years ./ length_GP_region_long^2) .+
        sigma_GP_region_short^2 .* exp.(-0.5 .* sq_dist_years ./ length_GP_region_short^2) .+ jitter
    cov_state::Matrix{Float64} =
        sigma_GP_state_long^2 .* exp.(-0.5 .* sq_dist_years ./ length_GP_state_long^2) .+
        sigma_GP_state_short^2 .* exp.(-0.5 .* sq_dist_years ./ length_GP_state_short^2) .+ jitter
    GP_region::Matrix{Float64} = cholesky(Symmetric(cov_region)).L * GP_region_std
    GP_state::Matrix{Float64}  = cholesky(Symmetric(cov_state)).L * GP_state_std

    # obs_mu[n] = mu + year_re[year_ind] + state_re[state_ind] + region_re[region_ind]
    #             + GP_region[year_ind, region_ind] + GP_state[year_ind, state_ind].
    # The two GP contributions are column-major gathers of the flattened matrices.
    lin_region::Vector{Int} = _hgp_lin(year_ind, region_ind, N_years)
    lin_state::Vector{Int}  = _hgp_lin(year_ind, state_ind, N_years)
    gp_region_flat::Vector{Float64} = vec(GP_region)
    gp_state_flat::Vector{Float64}  = vec(GP_state)
    obs_mu::Vector{Float64} =
        mu .+ year_re[year_ind] .+ state_re[state_ind] .+ region_re[region_ind] .+
        gp_region_flat[lin_region] .+ gp_state_flat[lin_state]

    # Likelihood yₙ ~ Normal(obs_muₙ, sigma_error).
    pointwise = plate(y, obs_mu, sigma_error) do yi, mi, s
        normal(mi, s).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    # Priors. All the *_std blocks are iid standard normal — that is exactly the
    # raw unconstrained q[1:base] slice, so one plate covers them.
    std_block::AbstractVector{Float64} = view(unconstrained, 1:base)
    std_pointwise = plate(std_block) do e
        normal(0.0, 1.0).logpdf(e)
    end
    prior_std::Float64 = sum(std_pointwise)
    prior_mu::Float64       = normal(0.5, 0.5).logpdf(mu)
    prior_tot_var::Float64  = gamma(3.0, 3.0).logpdf(tot_var)
    prior_prop_var::Float64 = dirichlet(fill(2.0, 17)).logpdf(prop_var)
    prior_lengths::Float64 =
        _hgp_weibull(length_GP_region_long, 30.0, 8.0) +
        _hgp_weibull(length_GP_state_long, 30.0, 8.0) +
        _hgp_weibull(length_GP_region_short, 30.0, 3.0) +
        _hgp_weibull(length_GP_state_short, 30.0, 3.0)
    log_prior::Float64 = prior_std + prior_mu + prior_tot_var + prior_prop_var + prior_lengths

    constrained_logdensity::Float64 = log_prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

nyr = HGP_N_YEARS; nrg = HGP_N_REGIONS; nst = HGP_N_STATES; nyo = HGP_N_YEARS_OBS
q = zeros(nyr * nrg + nyr * nst + nyo + nst + nrg + 1 + 16 + 1 + 4)
q[nyr * nrg + nyr * nst + nyo + nst + nrg + 18] = 0.5   # mu near the data mean
y = HGP_Y
year_ind = HGP_YEAR_IND
state_ind = HGP_STATE_IND
region_ind = HGP_REGION_IND
state_region_ind = HGP_STATE_REGION_IND
N_years = HGP_N_YEARS; N_regions = HGP_N_REGIONS; N_states = HGP_N_STATES; N_years_obs = HGP_N_YEARS_OBS

requested_nodes = (:log_prior, :likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y, :year_ind, :state_ind, :region_ind, :state_region_ind,
            :N_years, :N_regions, :N_states, :N_years_obs),
    want = requested_nodes,
    bound = (; y, year_ind, state_ind, region_ind, state_region_ind,
               N_years, N_regions, N_states, N_years_obs))

output = density_kernel(q)
log_prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :hierarchical_gp_posterior,
    origin = "posteriordb hierarchical_gp — hierarchical GP of state presidential votes",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    gamma_object = gamma,
    dirichlet_object = dirichlet,
)
"""

function evaluate_hierarchical_gp_source(; model_only::Bool = false)
    _evaluate_ppl_source(HIERARCHICAL_GP_SOURCE, @__MODULE__; bindings = (
        :HGP_Y, :HGP_YEAR_IND, :HGP_STATE_IND, :HGP_REGION_IND, :HGP_STATE_REGION_IND,
        :HGP_N_YEARS, :HGP_N_REGIONS, :HGP_N_STATES, :HGP_N_YEARS_OBS,
    ), model_only)
end

const _HIERARCHICAL_GP_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _HIERARCHICAL_GP_GRAPH_TEMPLATE[] = evaluate_hierarchical_gp_source(; model_only = true).model
    nothing
end

"""
    build_hierarchical_gp_graph()

Build the posteriordb `hierarchical_gp` model (a hierarchical Gaussian-process
model of state presidential vote shares) as a declarative
`ReactiveKernels.KernelSpec`. The Dirichlet variance decomposition uses Stan's
exact ILR simplex transform (`softmax(sum_to_zero_constrain(·))` with its
`Σ log(x) + 0.5·log K` Jacobian), each per-year GP is a sum of a long- and
short-range exponential-quadratic kernel applied non-centered through the
Cholesky factor, and the observation mean gathers the year/state/region random
effects and the two GP matrices. Positive scale parameters use the log
transform with the exact Jacobian; the shape-derived designs (year axis,
squared distances, column-major gather indices, ILR basis) are in-graph nodes
folded by `bound=`.
"""
function build_hierarchical_gp_graph()
    compose(_HIERARCHICAL_GP_GRAPH_TEMPLATE[])
end

function demo()
    model = build_hierarchical_gp_graph()
    nyr = HGP_N_YEARS; nrg = HGP_N_REGIONS; nst = HGP_N_STATES; nyo = HGP_N_YEARS_OBS
    q = zeros(nyr * nrg + nyr * nst + nyo + nst + nrg + 1 + 16 + 1 + 4)
    q[nyr * nrg + nyr * nst + nyo + nst + nrg + 18] = 0.5

    log_prior, likelihood, log_jacobian, posterior =
        prepare(model;
            have = (:unconstrained, :y, :year_ind, :state_ind, :region_ind, :state_region_ind,
                    :N_years, :N_regions, :N_states, :N_years_obs),
            want = (:log_prior, :likelihood, :log_jacobian, :posterior))(
            q, HGP_Y, HGP_YEAR_IND, HGP_STATE_IND, HGP_REGION_IND, HGP_STATE_REGION_IND,
            HGP_N_YEARS, HGP_N_REGIONS, HGP_N_STATES, HGP_N_YEARS_OBS)
    println("  log prior      = ", log_prior)
    println("  log likelihood = ", likelihood)
    println("  log |Jacobian| = ", log_jacobian)
    println("  log posterior  = ", posterior)
    nothing
end

end # module HierarchicalGPExample

if abspath(PROGRAM_FILE) == @__FILE__
    HierarchicalGPExample.demo()
end
