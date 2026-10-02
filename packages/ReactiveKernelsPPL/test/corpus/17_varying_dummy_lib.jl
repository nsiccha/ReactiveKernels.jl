# data: y c g
begin
    a ~ Normal(0, 1)
    r ~ varying_coefs_correlated(g, 2)
    mu = a .+ r[g, 1] .+ (c .== 2) .* r[g, 2]
    y .~ Normal.(mu, 1.5)
end
