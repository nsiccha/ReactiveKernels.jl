# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    k ~ LogNormal(0.0, 0.3)
    eta = a .+ b .* x
    y .~ Weibull.(k, exp.(eta))
end
