# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    sigma ~ Exponential(1)
    mu = a .+ b .* x
    y .~ LogNormal.(mu, sigma)
end
