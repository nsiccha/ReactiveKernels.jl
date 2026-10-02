# data: y
begin
    mu1 ~ Normal(0.0, 2.0)
    mu2 ~ Normal(0.0, 2.0)
    sigma1 ~ Exponential(1.0)
    sigma2 ~ Exponential(1.0)
    theta ~ Beta(5.0, 5.0)
    y .~ MixtureModel.(vcat.(Normal.(mu1, sigma1), Normal.(mu2, sigma2)), Ref([theta, 1.0 - theta]))
end
