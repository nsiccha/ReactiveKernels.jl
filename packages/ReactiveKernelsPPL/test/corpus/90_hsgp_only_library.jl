# data: y x x2
# 90_hsgp_only as plain statements: coefficient-free HSGP location and
# log-scale predictors with stated LogNormal(0, 4) hyper priors.
begin
    (PHI, lambda) = hsgp_basis(x; k = 20)
    h_rho ~ LogNormal(0, 4)
    h_sigma ~ LogNormal(0, 4)
    h_z[axes(PHI, 2)] .~ Normal.(0, 1)
    mu = PHI * (hsgp_sqrt_spd(lambda, h_sigma, h_rho) .* h_z)
    (PHI2, lambda2) = hsgp_basis(x2; k = 20)
    h2_rho ~ LogNormal(0, 4)
    h2_sigma ~ LogNormal(0, 4)
    h2_z[axes(PHI2, 2)] .~ Normal.(0, 1)
    lsig = PHI2 * (hsgp_sqrt_spd(lambda2, h2_sigma, h2_rho) .* h2_z)
    y .~ Normal.(mu, exp.(lsig))
end
