# data: y x
begin
    a ~ Normal(0, 1)
    mu = a .+ hsgp(:h_p)
    y .~ Normal.(mu, 1.5)
    hsgp_basis(:h_p, x; k = 4, cov = :periodic, period = 2.0)
end
