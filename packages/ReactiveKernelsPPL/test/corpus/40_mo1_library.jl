# data: y c
begin
    a ~ Normal(0, 1)
    s ~ Dirichlet([1.0, 2.0])
    m ~ monotonic(c, s)
    mu = a .+ m
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
