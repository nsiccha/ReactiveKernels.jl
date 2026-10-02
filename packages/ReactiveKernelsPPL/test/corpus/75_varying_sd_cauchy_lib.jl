# data: y sigma g
# The body of `r ~ varying_coefs(g)` written out, with a half-Cauchy sd
# instead of the shipped half-Normal.
begin
    mu ~ Normal(0, 5)
    r_sd ~ HalfCauchy(5)
    r_z[levels(g)] .~ Normal.(0, 1)
    r = r_sd .* r_z
    eta = mu .+ r[g]
    y .~ Normal.(eta, sigma)
end
