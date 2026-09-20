# data: y1 y2 x g
begin
    a1 ~ Normal(0, 1)
    a2 ~ Normal(0, 1)
    s ~ Exponential(1)
    d ~ varying_draws(g, [1, x]; eta = 2.0)
    r1 ~ varying_slice(d, 1)
    r2 ~ varying_slice(d, 2)
    mu1 = a1 .+ r1
    mu2 = a2 .+ r2
    y1 .~ Normal.(mu1, s)
    y2 .~ Normal.(mu2, s)
end
