# data: y x x2
# accel_gp: distributional HSGP pair with stated hyper priors (SB
# `length_scale(:, hsgp(x)) ~ InverseGamma(...)`, `sd(:, hsgp(x)) ~ ...`);
# a stated length scale drops the validity floor.
begin
    b0 ~ StudentT(3, -13, 36)
    mu = b0 .+ hsgp(:h_x)
    hsgp_basis(:h_x, x; k = 8, length_scale = InverseGamma(1.124909, 0.0177),
        sd = StudentT(3, 0, 36))
    s0 ~ StudentT(3, 0, 10)
    lsig = s0 .+ hsgp(:h_x2)
    hsgp_basis(:h_x2, x2; k = 8, length_scale = InverseGamma(1.124909, 0.0177),
        sd = StudentT(3, 0, 36))
    y .~ Normal.(mu, exp.(lsig))
end
