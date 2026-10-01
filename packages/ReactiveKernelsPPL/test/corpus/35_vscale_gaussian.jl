# data: y x z
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    d ~ Normal(0, 1)
    mu = a .+ b .* x
    sigma = c .+ d .* z
    y .~ Normal.(mu, exp.(sigma))
end
