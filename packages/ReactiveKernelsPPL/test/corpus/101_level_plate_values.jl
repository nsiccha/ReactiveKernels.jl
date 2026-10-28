# data: y g
begin
    a ~ Normal(0, 2)
    z[levels(g)] .~ Normal.(0, 1)
    @plate for k in levels(g)
        c[k] ~ Normal(z[k], 0.7)
        d[k] = a + 2c[k]
    end
    mu = d[g]
    y .~ Normal.(mu, 0.8)
end
