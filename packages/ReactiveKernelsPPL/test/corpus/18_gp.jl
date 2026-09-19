# data: y x
begin
    a ~ Normal(0, 1)
    rho_gp ~ LogNormal(0, 1)
    sigma_gp ~ LogNormal(0, 1)
    @plate for i in eachindex(y)
        z_gp[i] ~ Normal(0, 1)
    end
    f_gp = gp_chol_latent(gp_exp_quad_cov(x, sigma_gp, rho_gp, 1e-9), z_gp)
    mu = a .+ f_gp
    y .~ Normal.(mu, 0.5)
end
