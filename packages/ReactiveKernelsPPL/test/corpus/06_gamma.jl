# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    alpha ~ Exponential(1.0)
    eta = a .+ b .* x
    y .~ Gamma.(alpha, exp.(eta) ./ alpha)
end
