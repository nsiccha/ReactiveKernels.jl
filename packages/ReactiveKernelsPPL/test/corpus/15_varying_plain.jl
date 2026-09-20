# data: y g
begin
    a ~ Normal(0, 1)
    r ~ varying_effect(g, [1])
    mu = a .+ r
    y .~ Normal.(mu, 1.5)
end
