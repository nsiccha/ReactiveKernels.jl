# data: y x x2
# 91_spline_sd_prior as plain statements: the `penalized_smooth` body with
# the stated smoothing-sd priors in place of its HalfNormal(1) default.
begin
    (Xf, Zp) = tps_basis(x)
    (Xf2, Zp2) = tps_basis(x2)
    b0 ~ StudentT(3, -13, 36)
    s0 ~ StudentT(3, 0, 10)
    s_b[axes(Xf, 2)] .~ Flat.()
    s_sd ~ truncated(StudentT(3, 0, 36), 0, Inf)
    s_z[axes(Zp, 2)] .~ Normal.(0, 1)
    s2_b[axes(Xf2, 2)] .~ Flat.()
    s2_sd ~ truncated(StudentT(3, 0, 10), 0, Inf)
    s2_z[axes(Zp2, 2)] .~ Normal.(0, 1)
    mu = b0 .+ Xf * s_b .+ Zp * (s_sd .* s_z)
    lsig = s0 .+ Xf2 * s2_b .+ Zp2 * (s2_sd .* s2_z)
    y .~ Normal.(mu, exp.(lsig))
end
