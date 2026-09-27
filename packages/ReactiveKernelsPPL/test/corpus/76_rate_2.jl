# data: k1 k2 n1 n2
begin
    theta1 ~ Beta(1.0, 1.0)
    theta2 ~ Beta(1.0, 1.0)
    k1 .~ Binomial.(n1, theta1)
    k2 .~ Binomial.(n2, theta2)
end
