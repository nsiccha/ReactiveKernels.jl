using ReactiveKernelsPPLExamples.HierarchicalGPExample
using ReactiveKernelsPPLExamples.HierarchicalGPExample:
    HGP_Y, HGP_YEAR_IND, HGP_STATE_IND, HGP_REGION_IND, HGP_STATE_REGION_IND,
    HGP_N_YEARS, HGP_N_REGIONS, HGP_N_STATES, HGP_N_YEARS_OBS
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, dirichlet
using LinearAlgebra
using SpecialFunctions: loggamma

# Graph-independent reference oracle for posteriordb hierarchical_gp: the ILR
# simplex variance decomposition, dual per-year Cholesky GPs, year/state/region
# random effects, and the weibull/gamma/dirichlet/normal priors.
function _hgp_reference(q, y, year_ind, state_ind, region_ind, state_region_ind,
                       nyr, nrg, nst, nyo)
    _n(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    _g(v, a, b) = a * log(b) - loggamma(a) + (a - 1) * log(v) - b * v
    _w(v, k, l) = log(k) - log(l) + (k - 1) * (log(v) - log(l)) - (v / l)^k
    _stz(yv) = begin
        N = length(yv); z = zeros(N + 1); sw = 0.0
        for i in N:-1:1
            n = Float64(i); w = yv[i] / sqrt(n * (n + 1)); sw += w; z[i] += sw; z[i + 1] -= w * n
        end
        z
    end
    o1 = nyr * nrg; o2 = o1 + nyr * nst
    GP_region_std = reshape(q[1:o1], nyr, nrg)
    GP_state_std = reshape(q[(o1 + 1):o2], nyr, nst)
    year_std = q[(o2 + 1):(o2 + nyo)]
    state_std = q[(o2 + nyo + 1):(o2 + nyo + nst)]
    region_std = q[(o2 + nyo + nst + 1):(o2 + nyo + nst + nrg)]
    base = o2 + nyo + nst + nrg
    tot_var = exp(q[base + 1]); prop_var_unc = q[(base + 2):(base + 17)]; mu = q[base + 18]
    llr = exp(q[base + 19]); lsl = exp(q[base + 20]); lrs = exp(q[base + 21]); lss = exp(q[base + 22])
    z_raw = _stz(prop_var_unc); e = exp.(z_raw .- maximum(z_raw)); prop_var = e ./ sum(e)
    jac_simplex = sum(log.(prop_var)) + 0.5 * log(17.0)
    log_jacobian = q[base + 1] + q[base + 19] + q[base + 20] + q[base + 21] + q[base + 22] + jac_simplex
    vars = (17.0 * tot_var) .* prop_var
    sigma_year = sqrt(vars[1]); sigma_region = sqrt(vars[2]); sigma_state = sqrt.(vars[3:12])
    sglr = sqrt(vars[13]); sgsl = sqrt(vars[14]); sgrs = sqrt(vars[15]); sgss = sqrt(vars[16])
    sigma_error = sqrt(vars[17])
    region_re = sigma_region .* region_std
    year_re = sigma_year .* year_std
    state_re = sigma_state[state_region_ind] .* state_std
    years = collect(1.0:nyr); sqd = [(years[i] - years[j])^2 for i in 1:nyr, j in 1:nyr]
    jit = 1e-6 * Matrix(I, nyr, nyr)
    cov_region = sglr^2 .* exp.(-0.5 .* sqd ./ llr^2) .+ sgrs^2 .* exp.(-0.5 .* sqd ./ lrs^2) .+ jit
    cov_state = sgsl^2 .* exp.(-0.5 .* sqd ./ lsl^2) .+ sgss^2 .* exp.(-0.5 .* sqd ./ lss^2) .+ jit
    GP_region = cholesky(Symmetric(cov_region)).L * GP_region_std
    GP_state = cholesky(Symmetric(cov_state)).L * GP_state_std
    obs_mu = [mu + year_re[year_ind[k]] + state_re[state_ind[k]] + region_re[region_ind[k]] +
              GP_region[year_ind[k], region_ind[k]] + GP_state[year_ind[k], state_ind[k]]
              for k in eachindex(y)]
    likelihood = sum(_n.(y, obs_mu, sigma_error))
    conc = fill(2.0, 17)
    dirichlet_lp = loggamma(sum(conc)) - sum(loggamma.(conc)) + sum((conc .- 1) .* log.(prop_var))
    log_prior = sum(_n.(q[1:base], 0.0, 1.0)) + _n(mu, 0.5, 0.5) + _g(tot_var, 3.0, 3.0) +
                dirichlet_lp + _w(llr, 30.0, 8.0) + _w(lsl, 30.0, 8.0) +
                _w(lrs, 30.0, 3.0) + _w(lss, 30.0, 3.0)
    (; log_prior, likelihood, log_jacobian, posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — hierarchical_gp (posteriordb hierarchical GP)" begin
    artifact = evaluate_hierarchical_gp_source()
    @test artifact.source == strip(HIERARCHICAL_GP_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    @test artifact.dirichlet_object === dirichlet

    @testset "authored on the reusable-endpoint surface" begin
        @test occursin("_hgp_stz_basis(17)", HIERARCHICAL_GP_SOURCE)   # ILR simplex basis
        @test occursin("dirichlet(fill(2.0, 17)).logpdf(prop_var)", HIERARCHICAL_GP_SOURCE)
        @test occursin("cholesky(Symmetric(cov_region)).L * GP_region_std", HIERARCHICAL_GP_SOURCE)
        @test occursin("_hgp_weibull(length_GP_region_long, 30.0, 8.0)", HIERARCHICAL_GP_SOURCE)
    end

    model = artifact.model
    nyr = HGP_N_YEARS; nrg = HGP_N_REGIONS; nst = HGP_N_STATES; nyo = HGP_N_YEARS_OBS
    dim = nyr * nrg + nyr * nst + nyo + nst + nrg + 1 + 16 + 1 + 4
    for seed in (1, 2)
        q = 0.25 .* [sin(0.5 * seed * i) for i in 1:dim]
        q[nyr * nrg + nyr * nst + nyo + nst + nrg + 18] = 0.5   # mu
        ref = _hgp_reference(q, HGP_Y, HGP_YEAR_IND, HGP_STATE_IND, HGP_REGION_IND,
                             HGP_STATE_REGION_IND, nyr, nrg, nst, nyo)
        log_prior, likelihood, log_jacobian, posterior =
            prepare(model;
                have = (:unconstrained, :y, :year_ind, :state_ind, :region_ind, :state_region_ind,
                        :N_years, :N_regions, :N_states, :N_years_obs),
                want = (:log_prior, :likelihood, :log_jacobian, :posterior),
                bound = (; y = HGP_Y, year_ind = HGP_YEAR_IND, state_ind = HGP_STATE_IND,
                    region_ind = HGP_REGION_IND, state_region_ind = HGP_STATE_REGION_IND,
                    N_years = nyr, N_regions = nrg, N_states = nst, N_years_obs = nyo))(q)
        @test log_prior ≈ ref.log_prior
        @test likelihood ≈ ref.likelihood
        @test log_jacobian ≈ ref.log_jacobian
        @test posterior ≈ ref.posterior
        @test posterior ≈ log_prior + likelihood + log_jacobian
    end
end
