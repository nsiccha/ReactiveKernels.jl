# data: y x x2
begin
    phi ~ Dirichlet([2.0, 1.0, 3.0])
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1)
    w = phi[1] .* x .+ phi[2] .* x2 .+ phi[3]
    mu = a .+ b .* w
    y .~ Normal.(mu, sigma)
end
