# data: y x z
# 31_hsgp_aniso as plain statements: one length scale per axis, each with
# its own validity floor (`hsgp_effect` shares one length scale).
begin
    a ~ Normal(0, 1)
    (PHI, lambda) = hsgp_basis(x, z; k = (4, 3), c = (1.5, 2.0))
    (rho_floor_1, rho_floor_2) = hsgp_rho_floors(lambda)
    rho_1 ~ truncated(LogNormal(0, 1), rho_floor_1, Inf)
    rho_2 ~ truncated(LogNormal(0, 1), rho_floor_2, Inf)
    sigma_f ~ LogNormal(0, 1)
    z_f[axes(PHI, 2)] .~ Normal.(0, 1)
    f = PHI * (hsgp_sqrt_spd(lambda, sigma_f, [rho_1, rho_2]) .* z_f)
    mu = a .+ f
    y .~ Normal.(mu, 1.5)
end
