# data: y c
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    s ~ Dirichlet([1.0, 2.0])
    mu = a .+ b .* mo(c, s)
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
