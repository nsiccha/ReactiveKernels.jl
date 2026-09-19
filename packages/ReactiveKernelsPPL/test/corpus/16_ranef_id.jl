# data: y x g
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1)
    mu = a .+ b .* x .+ ranef(:ID, g)
    y .~ Normal.(mu, sigma)
    ranef_bucket(:ID, g; eta = 1.0) do
        mu => [1, x]
    end
end
