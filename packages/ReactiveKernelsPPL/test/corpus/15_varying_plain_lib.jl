# data: y g
begin
    a ~ Normal(0, 1)
    r ~ varying_coefs(g)
    mu = a .+ r[g]
    y .~ Normal.(mu, 1.5)
end
