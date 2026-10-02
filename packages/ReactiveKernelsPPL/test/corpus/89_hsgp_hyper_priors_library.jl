# data: y x x2
# 89_hsgp_hyper_priors as plain statements: the `hsgp_effect` body with the
# stated length-scale and sd priors in place of its defaults (a stated
# length-scale prior carries no validity floor).
begin
    b0 ~ StudentT(3, -13, 36)
    (PHI, lambda) = hsgp_basis(x; k = 8)
    h_rho ~ InverseGamma(1.124909, 0.0177)
    h_sigma ~ truncated(StudentT(3, 0, 36), 0, Inf)
    h_z[axes(PHI, 2)] .~ Normal.(0, 1)
    mu = b0 .+ PHI * (hsgp_sqrt_spd(lambda, h_sigma, h_rho) .* h_z)
    s0 ~ StudentT(3, 0, 10)
    (PHI2, lambda2) = hsgp_basis(x2; k = 8)
    h2_rho ~ InverseGamma(1.124909, 0.0177)
    h2_sigma ~ truncated(StudentT(3, 0, 36), 0, Inf)
    h2_z[axes(PHI2, 2)] .~ Normal.(0, 1)
    lsig = s0 .+ PHI2 * (hsgp_sqrt_spd(lambda2, h2_sigma, h2_rho) .* h2_z)
    y .~ Normal.(mu, exp.(lsig))
end
