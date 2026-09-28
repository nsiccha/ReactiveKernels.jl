# data: k1 k2 n1 n2
begin
    theta ~ Beta(1.0, 1.0)
    k1 .~ Binomial.(n1, theta)
    k2 .~ Binomial.(n2, theta)
end
