# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    s ~ Exponential(1)
    mu = a .+ b .* x
    y[eachindex(y)] .~ Normal.(mu, s)
end
