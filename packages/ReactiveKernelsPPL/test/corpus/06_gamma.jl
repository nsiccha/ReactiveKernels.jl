# data: y x
begin
    alpha ~ Exponential(1.0)
    eta = a .+ b .* x
    y .~ Gamma.(alpha, exp.(eta) ./ alpha)
end
