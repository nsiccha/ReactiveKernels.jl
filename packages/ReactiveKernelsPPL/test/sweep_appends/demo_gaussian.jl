@rkppl begin
    mu_b1 ~ Normal(0.0, 1.0)
    mu_b2 ~ Normal(0.0, 1.0)
    mu = mu_b1 .+ mu_b2 .* x
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
