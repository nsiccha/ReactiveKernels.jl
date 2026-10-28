# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1)
    nu ~ Gamma(2, 0.1)
    mu = a .+ b .* x
    y .~ StudentT.(nu, mu, sigma)
end
