# data: y x x2
# accel_splines: distributional spline pair with stated smoothing-sd
# priors (SB `sd(mu, s(x)) ~ LocationScale(0, 36, TDist(3))`).
begin
    spline_basis(:s_x, x; sd = StudentT(3, 0, 36))
    spline_basis(:s_x2, x2; sd = StudentT(3, 0, 10))
    b0 ~ StudentT(3, -13, 36)
    s0 ~ StudentT(3, 0, 10)
    mu = b0 .+ spline(:s_x)
    lsig = s0 .+ spline(:s_x2)
    y .~ Normal.(mu, exp.(lsig))
end
