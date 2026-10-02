# data: y x
begin
    a ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    b ~ Normal(0, 1)
    s ~ Exponential(1)
    u ~ Uniform(0,100)
end
