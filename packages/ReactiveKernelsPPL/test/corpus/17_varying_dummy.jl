# data: y c g
begin
    a ~ Normal(0, 1)
    r ~ varying_effect(g, [1, dummy(c, 2)])
    mu = a .+ r
    y .~ Normal.(mu, 1.5)
end
