# data: y g1 g2
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    gg = vcat(g1, g2)
    r ~ varying_coefs(gg)
    mu = a .+ r[g1] ./ 2 .+ r[g2] ./ 2
    y .~ Normal.(mu, sigma)
end
