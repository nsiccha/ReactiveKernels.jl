# data: y sigma g
begin
    mu ~ Normal(0, 5)
    r ~ varying_effect(g, [1]; sd = Cauchy(0, 5))
    eta = mu .+ r
    y .~ Normal.(eta, sigma)
end
