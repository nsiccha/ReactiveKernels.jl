# data: y x z g
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    w = x .* z
    mu = a .+ ranef(g)
    y .~ Normal.(mu, sigma)
    ranef_bucket(g) do
        mu => [1, w]
    end
end
