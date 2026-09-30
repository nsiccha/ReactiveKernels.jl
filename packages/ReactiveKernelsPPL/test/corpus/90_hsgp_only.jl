# data: y x x2
# brm_hsgp: coefficient-free HSGP-only location and log-scale predictors
# (SB `mu ~ 0 + hsgp(x)`, `log(sigma) ~ 0 + hsgp(x2)`).
begin
    mu = hsgp(:h_x)
    hsgp_basis(:h_x, x; k = 20, length_scale = LogNormal(0, 4),
        sd = LogNormal(0, 4))
    lsig = hsgp(:h_x2)
    hsgp_basis(:h_x2, x2; k = 20, length_scale = LogNormal(0, 4),
        sd = LogNormal(0, 4))
    y .~ Normal.(mu, exp.(lsig))
end
