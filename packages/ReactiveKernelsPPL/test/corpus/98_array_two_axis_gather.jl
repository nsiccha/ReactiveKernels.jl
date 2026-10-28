# data: y x g
begin
    tau ~ HalfNormal(1)
    z[levels(g)] .~ Normal.(0, 1)
    L ~ LKJCholesky(2, 2.0)
    sd[1:2] .~ HalfNormal.(1)
    Z[levels(g), 1:2] .~ Normal.(0, 1)
    M = (sd .* L)'
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    r = tau .* z[g] .+ Z[g, :] * M[:, 1] .+ (Z[g, :] * M[:, 2]) .* x
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
