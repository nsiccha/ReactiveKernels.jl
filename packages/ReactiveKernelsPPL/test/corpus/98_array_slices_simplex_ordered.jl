# data: y x g
begin
    eachcol(P[1:3, levels(g)]) .~ Dirichlet(3, 1.0)
    eachrow(C[levels(g), 1:2]) .~ Ordered(Normal(0, 1), 2)
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    r = C[g, 1] .+ C[g, 2] .* x .+ P[1, 1] .* x
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
