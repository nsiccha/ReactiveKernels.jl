# data: y x w
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ weighted.(Normal.(mu, s), w)
end
