# data: y x X
begin
    vx = var.(eachcol(X))
    sx = sqrt(sum(vx))
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    xs = x ./ sx
    mu = a .+ b .* xs
    y .~ Normal.(mu, sigma)
end
