# data: y x g1 g2 w1 w2
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    gg = vcat(g1, g2)
    r ~ varying_coefs_correlated(gg, 2)
    wt = w1 .+ w2
    r1 = r[g1, 1] .+ x .* r[g1, 2]
    r2 = r[g2, 1] .+ x .* r[g2, 2]
    mu = a .+ (w1 ./ wt) .* r1 .+ (w2 ./ wt) .* r2
    y .~ Normal.(mu, sigma)
end
