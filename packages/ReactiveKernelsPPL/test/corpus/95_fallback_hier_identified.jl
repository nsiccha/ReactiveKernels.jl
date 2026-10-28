# data: y g
begin
    a ~ Normal(0, 5)
    sg ~ HalfNormal(1)
    c[levels(g)] .~ Normal.(0, sg)
    sigma ~ Exponential(1)
    mu = a .+ c[g]
    y .~ Normal.(mu, sigma)
end
