# data: y sigma
begin
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5), 0, Inf)
    theta[axes(y, 1)] .~ Normal.(mu, tau)
    y .~ Normal.(theta, sigma)
end
