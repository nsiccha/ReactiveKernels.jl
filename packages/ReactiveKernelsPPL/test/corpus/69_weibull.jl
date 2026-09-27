# data: y x
begin
    k ~ LogNormal(0.0, 0.3)
    eta = a .+ b .* x
    y .~ Weibull.(k, exp.(eta))
end
