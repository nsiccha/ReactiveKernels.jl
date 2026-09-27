@rkppl begin
    c[levels(g)] .~ Normal.(0.0, 2.0)
    sigma ~ Exponential(1.0)
    mu = c[g]
    y .~ Normal.(mu, sigma)
end
