# data: y x g
begin
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    z[1:length(levels(g)) - 1] .~ Normal.(0, 1)
    mu = a .+ z[1] .* x
    y .~ Normal.(mu, sigma)
end
