# data: y x B
begin
    tau ~ HalfNormal(1)
    lambda[1:3] .~ HalfCauchy.(1)
    w[axes(B, 2)] .~ Normal.(0, lambda .* tau)
    s[1:3] .~ Normal.([0.0, 1.0, -1.0], [1.0, 2.0, 0.5])
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    v = s[2] .* x
    mu = a .+ B * w .+ v
    y .~ Normal.(mu, sigma)
end
