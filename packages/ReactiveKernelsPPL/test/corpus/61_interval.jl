# data: y x hi
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ interval_censored.(Normal.(mu, s), hi)
    s ~ Exponential(1)
end
