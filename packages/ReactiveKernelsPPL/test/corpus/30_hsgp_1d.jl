# data: y x
begin
    a ~ Normal(0, 1)
    mu = a .+ hsgp(:h_x)
    y .~ Normal.(mu, 1.5)
    hsgp_basis(:h_x, x; k = 4)
end
