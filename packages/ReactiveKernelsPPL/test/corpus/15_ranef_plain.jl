# data: y g
begin
    a ~ Normal(0, 1)
    mu = a .+ ranef(g)
    y .~ Normal.(mu, 1.5)
    ranef_bucket(g) do
        mu => [1]
    end
end
