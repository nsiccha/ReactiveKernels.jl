# data: y c
begin
    zeta ~ Dirichlet([1.0, 2.0])
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    cum = cumsum(vcat(0.0, zeta))
    m = cum[c]
    mu = a .+ b .* m
    y .~ Normal.(mu, sigma)
end
