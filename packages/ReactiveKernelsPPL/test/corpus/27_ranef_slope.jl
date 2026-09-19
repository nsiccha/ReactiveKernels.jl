# data: y x g
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    mu = a .+ ranef(g)
    y .~ Normal.(mu, sigma)
    ranef_bucket(g) do
        mu => [x]
    end
end
