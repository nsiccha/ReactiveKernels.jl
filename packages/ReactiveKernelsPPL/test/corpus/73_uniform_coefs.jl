# data: y x z
begin
    a ~ Flat()
    b1 ~ Uniform(-100, 0)
    b2 ~ Uniform(0, 100)
    mu = a .+ b1 .* x .+ b2 .* z
    y .~ BernoulliLogit.(mu)
end
