# data: y x
begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 2.5)
    kappa ~ Gamma(2.0, 0.1)
    mu = a .+ b .* x
    y .~ CircularVonMises.(mu, kappa, -pi, pi)
end
