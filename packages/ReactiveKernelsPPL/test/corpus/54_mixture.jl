# data: y x
begin
    a1 ~ Normal(0, 1)
    b1 ~ Normal(0, 1)
    mu1 = a1 .+ b1 .* x
    mu2 ~ Normal(0.0, 5.0)
    sigma ~ Exponential(1.0)
    y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)], [0.3, 0.7])
end
