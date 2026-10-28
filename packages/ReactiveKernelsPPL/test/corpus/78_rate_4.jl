# data: k n
begin
    theta ~ Beta(1.0, 1.0)
    thetaprior ~ Beta(1.0, 1.0)
    k .~ Binomial.(n, theta)
end
