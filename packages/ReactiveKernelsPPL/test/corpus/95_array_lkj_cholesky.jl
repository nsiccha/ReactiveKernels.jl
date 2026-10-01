# data: y x x2
begin
    L ~ LKJCholesky(3, 2.0)
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    w = L[2, 1] .* x .+ L[3, 2] .* x2
    mu = a .+ w
    y .~ Normal.(mu, sigma)
end
