# data: y x z
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    d ~ Normal(0, 1)
    mu = a .+ b .* x
    lognu = c .+ d .* z
    y .~ StudentT.(exp.(lognu), mu, 2.0)
end
