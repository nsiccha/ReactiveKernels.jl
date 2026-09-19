# data: y x z
begin
    a ~ Normal(0, 1)
    mu = a .+ hsgp(:h_xz)
    y .~ Normal.(mu, 1.5)
    hsgp_basis(:h_xz, x, z; k = (4, 3), c = (1.5, 2.0), iso = false)
end
